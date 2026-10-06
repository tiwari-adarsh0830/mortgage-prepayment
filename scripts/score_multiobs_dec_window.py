"""
score_multiobs_dec_window.py -- scores a frozen multiobs hazard checkpoint on
ONE right-aligned window per test loan, ending EXACTLY at Dec of
--cutoff_year, against realized CY{cutoff_year+1} outcomes. Used for both the
cutoff_2002 primary run and the cutoff_2020 matched control -- same function,
different --cutoff_year/--seq_dir/--ckpt_path/--map_era/--include_pre2013.

WHY THE PANEL IS TRUNCATED AT DEC OF cutoff_year+1, NOT cutoff_year
--------------------------------------------------------------------
build_sequences_multiobs()'s gather is inherently backward-only from each
observation's row_idx (it never reads positions after end_pos), so passing
it a df that extends past ref_month cannot leak future information into the
gathered FEATURES for that observation -- confirmed by reading the gather
code (rel = (W-1)-arange(W) ranges 0..W-1, so src_idx = end_pos-rel never
exceeds end_pos).

But select_observations()'s eligibility rules (_eligible_candidates) compute
term_t as "L-1 for a censored loan," where L is "number of rows in THIS
cutoff-filtered panel." If the panel were truncated at exactly Dec-cutoff_year,
every loan still active past that date (i.e. every loan in our target
population, which is defined as active in the forecast year) is, within that
truncated panel, censored with term_t==L-1==row_idx(ref_month) -- so the
rule `row_idx < term_t` excludes the ref_month row for the entire population
of interest. This is not a bug: that rule exists during TRAINING to keep the
label-defining lookahead window from landing on/past a real event or the
panel's edge. It is a label-availability rule, and our label is unused here
(realized comes from a separate raw pass, see below) -- but rather than drop
the rule, extending the panel through Dec of the FOLLOWING year makes it
trivially and correctly satisfied for every genuinely-active loan (term_t
becomes either a real future payoff row, or L-1 of the EXTENDED panel, both
strictly after ref_month's row_idx), with no special-casing.

One combined per-vintage pass therefore produces BOTH the Dec-cutoff_year
feature windows AND the CY{cutoff_year+1} realized outcomes.

CP/U CATEGORY MAPS ARE A PER-CHECKPOINT SETTING
-------------------------------------------------
--map_era fixed  : current load_vintage_filtered() defaults (CP=4, U=3) --
                   use for checkpoints trained on the post-CP/U-fix rebuild
                   (cutoff_2002's frozen hist build, job 18054077).
--map_era prefix : {'R':0,'C':1,'P':2} / {'SF':0,'PU':1,'CO':2,'MH':3} (no
                   CP/U key) -- use for checkpoints trained BEFORE the fix
                   (cutoff_2020 f0.2/L33/IPW seed 42/7/123, all dated
                   2026-09-11/13, days before the fix commit 2a5b283 on
                   2026-09-19).

Usage:
    python scripts/score_multiobs_dec_window.py \\
        --cutoff_year 2002 \\
        --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist \\
        --ckpt_path outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist/hazard_best.pt \\
        --map_era fixed --include_pre2013 --seed_label seed42 \\
        --out_dir outputs/rolling/dec_window_cutoff_2002_seed42
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_sequences_multiobs_zbc import (
    load_vintage_filtered, load_pmms, load_zhvi, _prepare_panel,
    _eligible_candidates, build_sequences_multiobs, PRE2013_VINTAGES,
    _MODERN_VINTAGES, _vintage_quarter_start_yyyymm, dec_yyyymm, BASE,
    FEATURE_COLS, MAX_SEQ_LEN, PRE2013_CELL_SAMPLE_PATH,
)
from train_hazard_multiobs import PrepaymentTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = 1               # label horizon the frozen checkpoints were trained with (h1)
MIN_HIST = 1        # default, matches the training sbatch files (not overridden)
# Bump whenever build_combined_pass's output schema or logic changes
# incompatibly (e.g. adding current_actual_upb) -- baked into both cache
# path layers so an old cache is never silently reused after such a change.
CACHE_VERSION = 'v2_upb'

# Pre-fix category maps -- see module docstring. Only ADDED codes differ from
# the current fixed maps (CP=4, U=3); everything else is byte-identical, so
# "prefix" is exactly "fixed with 4->0 (CP) and 3->0 (U)".
_PREFIX_LOAN_PURPOSE_MAP  = {'R': 0, 'C': 1, 'P': 2}
_PREFIX_PROPERTY_TYPE_MAP = {'SF': 0, 'PU': 1, 'CO': 2, 'MH': 3}


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ckpt.get('config', {})
    # Fallback (for a checkpoint saved without a 'config'/'input_dim' key)
    # derives input_dim from the checkpoint's OWN input_proj weight matrix,
    # not the builder's N_FEATURES literal -- that tensor's shape is the
    # actual dimension this model was trained with, by construction.
    fallback_input_dim = ckpt['model_state']['input_proj.weight'].shape[1]
    m = PrepaymentTransformer(
        input_dim=cfg.get('input_dim', fallback_input_dim), d_model=cfg.get('d_model', 64),
        n_heads=cfg.get('n_heads', 4), n_layers=cfg.get('n_layers', 2),
        dim_ff=cfg.get('dim_ff', 256), dropout=cfg.get('dropout', 0.1),
        max_seq=cfg.get('max_seq', MAX_SEQ_LEN),
    ).to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    print(f'Loaded checkpoint {ckpt_path} (AUC={ckpt.get("auc", "?")})', flush=True)
    return m


def relevant_vintages(cutoff_year: int, include_pre2013: bool) -> list[str]:
    cutoff_ym = dec_yyyymm(cutoff_year)
    all_vintages = (PRE2013_VINTAGES + _MODERN_VINTAGES) if include_pre2013 else _MODERN_VINTAGES
    return [v for v in all_vintages if _vintage_quarter_start_yyyymm(v) <= cutoff_ym]


def population_hash(cutoff_year: int, test_ids: np.ndarray, map_era: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(cutoff_year).encode())
    h.update(map_era.encode())
    h.update(np.sort(test_ids).tobytes())
    return h.hexdigest()[:16]


def _atomic_pickle_write(obj, path: str):
    """Write via a per-process temp file + os.replace, so a killed job never
    leaves a partially-written cache file that looks complete on resume."""
    tmp_path = f'{path}.tmp{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def build_combined_pass(cutoff_year: int, include_pre2013: bool, map_era: str,
                         test_ids_set: set, cache_dir: str,
                         extra_keep_cols: list[str] | None = None,
                         cell_sample_path: str = PRE2013_CELL_SAMPLE_PATH):
    """ONE pass over the relevant vintage files, truncated at Dec of
    cutoff_year+1 (not cutoff_year -- see module docstring). Returns the
    concatenated, feature-complete df restricted to test_ids_set.

    Two cache layers, both atomic (temp file + os.replace), BOTH keyed by
    POPULATION HASH as well as truncation yyyymm/CACHE_VERSION/cutoff_year/
    map_era. The per-vintage layer is population-keyed too, not just
    cutoff_year/map_era -- load_vintage_filtered(keep_ids=test_ids_set)
    restricts each cached per-vintage frame to whatever population wrote it,
    so a DIFFERENT population sharing the same cutoff_year/map_era (e.g. a
    365,146-loan k5_h1 matched population vs. a 230,670-loan f0.2/L33 test
    split, both cutoff_2020) would silently read back an incomplete/wrong
    per-vintage frame if the directory were shared. (Caught 2026-09-25 before
    it could cause exactly that -- see docs/mistakes_and_lessons.md.)
      1. Per-vintage: each vintage's filtered frame is written to its own
         file under cache_dir/_per_vintage_{cutoff_year}_{map_era}_trunc{trunc_ym}_{pop_hash}_{CACHE_VERSION}/{v}.pkl
         as soon as it's built. A resumed run of the SAME population skips
         any vintage whose file already exists instead of re-reading that
         multi-GB CSV.
      2. Combined: the full concatenated frame, keyed the same way, so a
         fully-resumed rerun (e.g. a different seed/checkpoint scoring the
         same population) skips straight past both the raw scan AND the
         per-vintage concat.

    cell_sample_path: threaded straight to load_vintage_filtered's
    cell_sample_path (see its docstring) -- gates which pre-2013 loans are
    even candidates before intersecting with test_ids_set. Default is the
    original pre-30y-filter sample, for backward compatibility; pass e.g.
    outputs/pre2013_cell_sample_30y_loans.csv when test_ids_set itself was
    drawn from that sample, so the keep_ids intersection in
    load_vintage_filtered doesn't silently drop loans that are in
    test_ids_set but not in the (wrong) default cell sample.
    """
    _extra_keep_cols = extra_keep_cols or ['current_actual_upb']
    # Fold the extra-cols schema into the cache path -- a different schema is
    # a different cache-compatible-object entirely, and this must never
    # silently collide with the 'current_actual_upb'-only cache another
    # caller (e.g. score_rolling_one_step.py) already wrote/reads under the
    # SAME cache_dir/pop_hash (see the population-hash isolation bug this
    # module's docstring already documents -- same failure class).
    #
    # Also fold a FEATURE_COLS fingerprint in -- a manually-bumped
    # CACHE_VERSION has now twice failed to catch a load_vintage_filtered
    # schema change (bb8dece's term/post-mod fix, see
    # score_matched_intersection_2003.py's docstring; e96857f's harp_eligible
    # column, which crashed all 10 cutoff_2002_seq decwin jobs 2026-10-06
    # against a combined-pass pickle written 2026-10-02, before that commit).
    # Hashing FEATURE_COLS itself makes the cache self-invalidating on any
    # future schema change, instead of depending on a human remembering to
    # bump a string.
    _feat_fp = hashlib.blake2b(','.join(FEATURE_COLS).encode(), digest_size=4).hexdigest()
    _schema_tag = f'_feat{_feat_fp}' + (
        '' if _extra_keep_cols == ['current_actual_upb'] else
        '_cols' + ''.join(f'-{c}' for c in _extra_keep_cols))
    trunc_ym = dec_yyyymm(cutoff_year + 1)
    pop_hash = population_hash(cutoff_year, np.array(sorted(test_ids_set)), map_era)
    combined_path = os.path.join(
        cache_dir, f'_raw_combined_pass_{pop_hash}_trunc{trunc_ym}_{CACHE_VERSION}{_schema_tag}.pkl')
    if os.path.exists(combined_path):
        print(f'Cache hit (combined): {combined_path}', flush=True)
        with open(combined_path, 'rb') as f:
            return pickle.load(f)

    lp_map, pt_map = (
        (None, None) if map_era == 'fixed' else
        (_PREFIX_LOAN_PURPOSE_MAP, _PREFIX_PROPERTY_TYPE_MAP)
    )
    pmms_rates = load_pmms()
    zhvi_df = load_zhvi()

    vintage_cache_dir = os.path.join(
        cache_dir, f'_per_vintage_{cutoff_year}_{map_era}_trunc{trunc_ym}_{pop_hash}_{CACHE_VERSION}{_schema_tag}')
    os.makedirs(vintage_cache_dir, exist_ok=True)

    frames = []
    for v in relevant_vintages(cutoff_year, include_pre2013):
        v_path = os.path.join(vintage_cache_dir, f'{v}.pkl')
        if os.path.exists(v_path):
            print(f'  {v}: cache hit ({v_path})', flush=True)
            with open(v_path, 'rb') as f:
                df = pickle.load(f)
        else:
            df = load_vintage_filtered(v, pmms_rates, zhvi_df, trunc_ym, keep_ids=test_ids_set,
                                        loan_purpose_map=lp_map, property_type_map=pt_map,
                                        extra_keep_cols=_extra_keep_cols,
                                        cell_sample_path=cell_sample_path)
            _atomic_pickle_write(df, v_path)
            print(f'  {v}: {"empty" if df is None or df.empty else len(df)} rows '
                  f'(written {v_path})', flush=True)
        if df is not None and not df.empty:
            frames.append(df)
    full_df = pd.concat(frames, ignore_index=True)
    del frames

    os.makedirs(cache_dir, exist_ok=True)
    _atomic_pickle_write(full_df, combined_path)
    print(f'Cached combined pass: {combined_path} ({len(full_df):,} rows)', flush=True)
    return full_df


def build_dec_window_obs(full_df: pd.DataFrame, cutoff_year: int):
    """Restrict to loans with a Dec-{cutoff_year} row AND >=1 row in
    CY{cutoff_year+1} ("active in {cutoff_year+1}"), apply
    _eligible_candidates()'s feature-validity rules (min_hist, calendar-gap
    filter) to that Dec-{cutoff_year} row specifically, and emit an obs
    table in select_observations()'s _OBS_COLS schema for
    build_sequences_multiobs(obs=...).

    Returns (obs_df, diagnostics: dict of exclusion counts).
    """
    ref_ym = dec_yyyymm(cutoff_year)
    fy_lo, fy_hi = (cutoff_year + 1) * 100 + 1, (cutoff_year + 1) * 100 + 12

    diag = {}
    diag['test_loans_total'] = full_df['loan_id'].nunique()

    active_fy_ids = set(full_df.loc[(full_df['yyyymm'] >= fy_lo) & (full_df['yyyymm'] <= fy_hi),
                                     'loan_id'].unique().tolist())
    diag['active_in_forecast_year'] = len(active_fy_ids)

    dec_ids = set(full_df.loc[full_df['yyyymm'] == ref_ym, 'loan_id'].unique().tolist())
    diag['has_dec_row_raw'] = len(dec_ids)

    target_ids = active_fy_ids & dec_ids
    diag['active_and_has_dec_row'] = len(target_ids)
    diag['active_no_dec_row_excluded'] = len(active_fy_ids - dec_ids)

    panel = _prepare_panel(full_df)
    dec_rows = panel[(panel['loan_id'].isin(target_ids)) & (panel['yyyymm'] == ref_ym)].copy()
    diag['post_dropna_dec_row_loss'] = len(target_ids) - dec_rows['loan_id'].nunique()

    # Safety net: this population should never contain a scored row that IS
    # itself a payoff event (that is exactly the leak select_observations()'s
    # term_t rule guards against for prepaid loans; here it should be
    # structurally impossible since we require >=1 row AFTER ref_month).
    n_dec_is_event = int((dec_rows['zero_balance_code_actual'] == 1.0).sum())
    assert n_dec_is_event == 0, (
        f'{n_dec_is_event} scored Dec-{cutoff_year} rows have zero_balance_code_actual==1 -- '
        f'these are event rows, not censoring-boundary rows; do not score them as an '
        f'ordinary window, investigate the population filter.')

    elig = _eligible_candidates(full_df, H, MIN_HIST)
    elig_key = set(zip(elig['loan_id'], elig['row_idx']))
    dec_rows['eligible'] = list(zip(dec_rows['loan_id'], dec_rows['row_idx']))
    dec_rows['eligible'] = dec_rows['eligible'].isin(elig_key)
    diag['excluded_by_eligibility_rules'] = int((~dec_rows['eligible']).sum())
    dec_rows = dec_rows[dec_rows['eligible']].copy()
    diag['final_population'] = dec_rows['loan_id'].nunique()

    obs = pd.DataFrame({
        'loan_id':          dec_rows['loan_id'].to_numpy(),
        't':                dec_rows['row_idx'].astype(int).to_numpy(),
        'ref_month':        dec_rows['yyyymm'].astype(int).to_numpy(),
        'label':            np.zeros(len(dec_rows), dtype=np.float32),   # UNUSED -- realized comes from the raw pass
        'is_terminal':      np.zeros(len(dec_rows), dtype=bool),
        'incl_prob':        np.ones(len(dec_rows), dtype=np.float32),
        'n_eligible':       np.ones(len(dec_rows), dtype=np.int32),
        'k_actual':         np.ones(len(dec_rows), dtype=np.int32),
        'age_at_ref':       dec_rows['loan_age_months'].astype(float).to_numpy(),
        'incentive_at_ref': dec_rows['refi_incentive'].astype(float).to_numpy(),
    })
    return obs, dec_rows, diag


def verify_checkpoint(model, seq_dir: str, results_json_path: str, ckpt_dict: dict,
                       batch_size: int = 8192, tol: float = 1e-4):
    """Before trusting any score from this checkpoint: reproduce its own
    reported best_auc on the frozen test_seq.npy/test_mask.npy/test_labels.npy
    it trained/validated against. Also reports whether the checkpoint's
    saved dict contains a 'config' entry."""
    print(f"Checkpoint dict contains 'config': {'config' in ckpt_dict}"
          + (f' -> {ckpt_dict["config"]}' if 'config' in ckpt_dict else ''), flush=True)

    seqs   = np.load(os.path.join(seq_dir, 'test_seq.npy'),    mmap_mode='r')
    masks  = np.load(os.path.join(seq_dir, 'test_mask.npy'),   mmap_mode='r')
    labels = np.load(os.path.join(seq_dir, 'test_labels.npy'))
    h_vals = score(model, np.asarray(seqs), np.asarray(masks), batch_size)
    computed_auc = roc_auc_score(labels, h_vals)

    with open(results_json_path) as f:
        expected_auc = json.load(f)['best_auc']
    diff = abs(computed_auc - expected_auc)
    print(f'Checkpoint AUC verification: computed={computed_auc:.6f} '
          f'expected(best_auc)={expected_auc:.6f} diff={diff:.2e}', flush=True)
    assert diff < tol, (
        f'Checkpoint AUC mismatch: computed {computed_auc:.6f} vs. results.json '
        f'best_auc {expected_auc:.6f} (diff {diff:.2e} >= tol {tol:.0e}) -- this checkpoint/seq_dir '
        f'pairing does not reproduce its own reported score. STOP, do not trust any downstream number.')
    print('Checkpoint AUC verification PASSED.', flush=True)


def verify_windows(sequences: np.ndarray, masks: np.ndarray, obs: pd.DataFrame,
                    dec_rows: pd.DataFrame, scaler, n_check: int = 200, seed: int = 0,
                    tol: float = 1e-4):
    """Window-construction sanity check, run after build_sequences_multiobs:
    for n_check random scored loans, compare sequences[:, -1, :] against
    scaler.transform() of that loan's Dec-row FEATURE_COLS -- IN SCALED
    SPACE, not inverse-transformed back to raw units. original_upb is
    dollar-scale (mean ~2e5), so inverse-transforming loses ~0.01 of
    precision through float32 multiply-by-scale_ alone -- comparing there
    would false-fail at atol 1e-4 on a value the forward direction (the
    actual scaler.transform() call build_sequences_multiobs uses) recovers
    exactly. Comparing in scaled space checks the same thing the gather
    itself does, at the precision it actually operates at. Also asserts
    masks[:, -1] is True for EVERY row (not just the sample), since
    ref_month should always be unmasked by construction for this population."""
    assert bool(np.asarray(masks)[:, -1].all()), (
        f'{int((~np.asarray(masks)[:, -1]).sum())}/{len(masks)} scored windows have the LAST '
        f'timestep masked out -- ref_month is not present at position -1 for every row, '
        f'contradicting the right-aligned construction. STOP, do not trust any score.')

    rng = np.random.default_rng(seed)
    n = min(n_check, len(sequences))
    idx = rng.choice(len(sequences), size=n, replace=False)
    loan_ids = obs['loan_id'].to_numpy()
    dec_by_id = dec_rows.set_index('loan_id')

    actual_scaled = sequences[idx, -1, :]
    raw_expected = dec_by_id.loc[loan_ids[idx], FEATURE_COLS].to_numpy()
    expected_scaled = scaler.transform(raw_expected)
    max_abs_diff = float(np.abs(actual_scaled - expected_scaled).max())
    assert max_abs_diff < tol, (
        f'Window check FAILED: max abs diff {max_abs_diff:.2e} >= tol {tol:.0e} between '
        f'sequences[:,-1,:] and scaler.transform() of the loan\'s Dec row -- the gather does not '
        f'match the Dec-window population. STOP, do not trust any score.')
    print(f'Window check PASSED: {n} random loans, max abs diff {max_abs_diff:.2e} < {tol:.0e} '
          f'(scaled space); masks[:,-1] all True for all {len(sequences):,} rows.', flush=True)


def score(model, sequences: np.ndarray, masks: np.ndarray, batch_size: int = 8192):
    n = len(sequences)
    h_vals = np.zeros(n, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sb = torch.from_numpy(np.ascontiguousarray(sequences[i:i + batch_size])).to(DEVICE)
            mb = torch.from_numpy(np.ascontiguousarray(masks[i:i + batch_size])).to(DEVICE)
            logits = model(sb, mask=mb)                     # NO return_per_timestep -- frozen-hazard v1
            h_vals[i:i + sb.shape[0]] = torch.sigmoid(logits).cpu().numpy()
    return h_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff_year', type=int, required=True)
    ap.add_argument('--seq_dir', type=str, required=True,
                     help='Frozen build dir to reuse the scaler.pkl and test_loan_ids_split.npy from.')
    ap.add_argument('--ckpt_path', type=str, default=None,
                     help='Required unless --build_cache_only.')
    ap.add_argument('--map_era', choices=['fixed', 'prefix'], required=True)
    ap.add_argument('--include_pre2013', action='store_true')
    ap.add_argument('--seed_label', type=str, default=None,
                     help='Required unless --build_cache_only.')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default=None,
                     help='Shared raw-pass cache dir, independent of --out_dir so seed42/seed7/seed123 '
                          'runs for the SAME cutoff_year+map_era hit the same cache instead of each '
                          'redoing the ~raw-file scan. Default: outputs/rolling/_dec_window_raw_cache.')
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--cell_sample', type=str, default=PRE2013_CELL_SAMPLE_PATH,
                     help='loan_id CSV gating the historical-era (PRE2013_VINTAGES) population, '
                          'threaded to build_combined_pass the same way '
                          'prepare_sequences_multiobs_zbc.py --cell_sample is. Default is the '
                          'original pre-30y-filter sample, for backward compatibility; pass e.g. '
                          'outputs/pre2013_cell_sample_30y_loans.csv when --seq_dir is a _30y build.')
    ap.add_argument('--build_cache_only', action='store_true',
                     help='CPU-only mode: run build_combined_pass (+ population diagnostics) and exit. '
                          'No checkpoint load, no scoring, no GPU needed. Intended for the CPU '
                          'build job that scoring jobs then depend on (sbatch --dependency=afterok).')
    args = ap.parse_args()
    if not args.build_cache_only:
        assert args.ckpt_path and args.seed_label, '--ckpt_path and --seed_label are required unless --build_cache_only'

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')
    os.makedirs(cache_dir, exist_ok=True)

    with open(os.path.join(args.seq_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    # Item 12: confirm test_loan_ids_split.npy exists in --seq_dir (explicit,
    # readable failure instead of a generic np.load FileNotFoundError).
    split_path = os.path.join(args.seq_dir, 'test_loan_ids_split.npy')
    assert os.path.exists(split_path), f'test_loan_ids_split.npy not found in --seq_dir: {args.seq_dir}'
    test_ids = np.load(split_path, allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    print(f'Test split loans (frozen build): {len(test_ids_set):,}', flush=True)

    # PMMS series, Dec-{cutoff_year} through Dec-{cutoff_year+1}: rates moved
    # during the forecast year, and that affects how the forecast/realized
    # comparison reads -- saved alongside the scores, not folded into any metric.
    # load_pmms() keys its dict by 'reporting_period', an MMYYYY-format int
    # (month*10000+year, e.g. April 1971 -> 41971) -- see load_pmms() docstring
    # trace in data/pmms_monthly.csv's own 'reporting_period' column.
    pmms_rates_full = load_pmms()
    pmms_rows = []
    for y, m in [(args.cutoff_year, 12)] + [(args.cutoff_year + 1, mm) for mm in range(1, 13)]:
        mmyyyy = m * 10000 + y
        pmms_rows.append({'year': y, 'month': m, 'rate_30yr': pmms_rates_full.get(mmyyyy)})
    pmms_df = pd.DataFrame(pmms_rows)
    assert pmms_df['rate_30yr'].notna().all(), (
        f'PMMS series has None/NaN rate(s): {pmms_df[pmms_df["rate_30yr"].isna()].to_dict("records")} '
        f'-- STOP, do not save a series with missing months.')
    pmms_df.to_csv(
        os.path.join(args.out_dir, f'pmms_series_{args.cutoff_year}_{args.cutoff_year + 1}.csv'), index=False)

    full_df = build_combined_pass(args.cutoff_year, args.include_pre2013, args.map_era,
                                   test_ids_set, cache_dir=cache_dir, cell_sample_path=args.cell_sample)

    obs, dec_rows, diag = build_dec_window_obs(full_df, args.cutoff_year)
    print('Population diagnostics:', flush=True)
    for k, v in diag.items():
        print(f'  {k}: {v:,}', flush=True)

    if args.build_cache_only:
        print('--build_cache_only: cache populated, exiting before checkpoint load/scoring.', flush=True)
        return

    sequences, masks, _labels, _prepay_t, loan_ids_out, extras = build_sequences_multiobs(
        full_df, scaler, k_draws=1, H=H, min_hist=MIN_HIST, obs=obs)
    print(f'Built {len(sequences):,} Dec-{args.cutoff_year} windows, shape {sequences.shape}', flush=True)

    # Item 10: window-construction sanity check.
    verify_windows(sequences, masks, obs, dec_rows, scaler)

    model = load_checkpoint(args.ckpt_path)
    ckpt_dict = torch.load(args.ckpt_path, map_location=DEVICE)

    # Item 9: reproduce this checkpoint's own reported AUC on its frozen test
    # set before trusting anything scored from it.
    results_json_path = os.path.join(os.path.dirname(args.ckpt_path), 'results.json')
    verify_checkpoint(model, args.seq_dir, results_json_path, ckpt_dict, args.batch_size)

    h_vals = score(model, sequences, masks, args.batch_size)

    fy = args.cutoff_year + 1
    fy_lo, fy_hi = fy * 100 + 1, fy * 100 + 12
    fy_window = full_df[(full_df['yyyymm'] >= fy_lo) & (full_df['yyyymm'] <= fy_hi)]
    # Realized definition matches forecast_rolling_cpr.read_coupon_and_realized():
    # active_set = any test loan appearing in ANY forecast-year row; prepaid_set =
    # subset with zero_balance_code_actual==1 in any forecast-year row. aggregate()
    # then sets realized = loan_id.isin(prepaid_set) -- a non-prepay termination
    # (zbc present, not 1) reads as realized=0, IDENTICAL to a loan with no event
    # at all. We match that here: prepaid_set below is the only realized=1 source.
    prepaid_set = set(fy_window.loc[fy_window['zero_balance_code_actual'] == 1.0, 'loan_id'].unique().tolist())
    non_prepay_term_set = set(fy_window.loc[fy_window['zero_balance_code_actual'].notna() &
                                             (fy_window['zero_balance_code_actual'] != 1.0), 'loan_id'].unique().tolist())
    n_non_prepay_term = len(non_prepay_term_set & set(loan_ids_out.tolist()))
    print(f'Non-prepay terminations in the forecast year among scored loans (counted as NOT '
          f'prepaid, matching aggregate()): {n_non_prepay_term:,}/{len(loan_ids_out):,}', flush=True)

    dec_rows_by_id = dec_rows.set_index('loan_id')
    out = pd.DataFrame({
        'loan_id':            loan_ids_out,
        'h_t':                h_vals,
        'note_rate':          dec_rows_by_id.loc[loan_ids_out, 'original_interest_rate'].to_numpy(),
        'incentive_at_ref':   dec_rows_by_id.loc[loan_ids_out, 'refi_incentive'].to_numpy(),
        'current_actual_upb': dec_rows_by_id.loc[loan_ids_out, 'current_actual_upb'].to_numpy(),
        'realized_prepay':    [int(lid in prepaid_set) for lid in loan_ids_out],
    })
    out['annual_pp'] = 1.0 - (1.0 - np.clip(out['h_t'], 1e-7, 1 - 1e-7)) ** 12

    out_path = os.path.join(args.out_dir, f'dec_window_scores_{args.seed_label}.csv')
    out.to_csv(out_path, index=False)
    print(f'Saved: {out_path}', flush=True)
    print(out[['h_t', 'annual_pp']].describe(), flush=True)


if __name__ == '__main__':
    main()
