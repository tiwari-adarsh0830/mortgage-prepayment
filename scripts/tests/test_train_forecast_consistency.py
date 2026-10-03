"""test_train_forecast_consistency.py -- standing test: the TRAINING path
and the FORECASTING path must produce IDENTICAL sequences and predictions
for the same (loan_id, ref_month) observations, and a multi-month forecast
must use the window ending at each forecast month, not a frozen earlier one.

Written 2026-09-26 after two separate incidents where the model was fed
data one way in training and another way in forecasting (see
docs/mistakes_and_lessons.md, STANDING TESTS section, for the history).
MUST be rerun -- and pass -- after any change to data prep, the sequence
builder, feature definitions, category maps, scalers, or any scoring/
forecast script, and before any forecast number is reported, written into
README, or sent to the advisor. If it fails, nothing downstream is trusted
until the cause is found.

CHECK 1 (inputs and predictions): samples ~2,000 TRAINING observations from
a build's stored train split, rebuilds them through the FORECASTING path
(raw vintage scan -> build_sequences_multiobs(obs=...)) using that build's
own scaler and the checkpoint's CP/U category maps, and asserts the two
paths agree: sequences (max abs diff < 1e-6), masks (exact), and model
predictions from the same checkpoint (max abs diff < 1e-6).

COST NOTE -- bounded, not global-uniform, sampling: a truly global-uniform
sample over the whole train split would require scanning every relevant
vintage file for the cutoff (up to ~500GB across 32 modern-era files for
cutoff_2020) every single run. To keep this rerunnable after every pipeline
change, Check 1 is bounded to ONE vintage file per cutoff per run -- but
picked AT RANDOM (os.urandom-seeded, logged as `vintage_pick_seed`) from
that cutoff's full relevant-vintage list, not fixed to the same small file
every time, so the check can't be gamed by a lucky one-time choice and
different runs exercise different vintages' data over time. Runtime
therefore varies run to run with which vintage was drawn (as little as ~30s
for a 2GB file, potentially several minutes for a 20GB+ one) -- this
narrows which vintage-file diversity gets exercised on any SINGLE run, NOT
the gather/scaler/CP-U-map machinery under test, which is identical for
every vintage.

RUN VIA srun/sbatch, NOT directly on the login node: reading even a small
vintage CSV plus the ~1.5M-loan train-split keep_ids filter briefly needs
more headroom than the shared login node reliably has (observed:
OOM-killed running directly on the login node 2026-09-26; passed cleanly
under `srun --mem=40G`). Example:
    srun --account=torch_pr_932_general --partition=cpu_short --mem=40G \\
         --cpus-per-task=4 --time=00:20:00 \\
         python scripts/tests/test_train_forecast_consistency.py

CHECK 2 (time alignment): for a sample of one-step-ahead forecast
loan-months (TEST population, reusing score_rolling_one_step.py's already-
cached combined raw pass -- cheap, no new raw scan), asserts the window
predicting month t+1 ends exactly at month t: last timestep unmasked,
features equal that row's scaled features.

NEGATIVE CONTROLS (must FAIL -- proving the checks catch real bugs, not
just that the pipes are connected):
  (a) TRAIL_SEQ_DIR's old LEFT-aligned trailing-builder sequences
      (prepare_sequences_trailing_zbc.py's convention), matched by loan_id
      against the correct multiobs train sample -- must fail Check 1.
  (b) build_dec_window_obs's frozen-at-Dec-cutoff window, checked against a
      LATER month's actual features -- must fail Check 2.

Run:
    cd /scratch/at7095/mortgage_prepayment
    python scripts/tests/test_train_forecast_consistency.py
"""
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from prepare_sequences_multiobs_zbc import (
    load_pmms, load_zhvi, load_vintage_filtered, build_sequences_multiobs,
    _prepare_panel, dec_yyyymm, FEATURE_COLS, PRE2013_CELL_SAMPLE_PATH,
)
from score_multiobs_dec_window import (
    build_combined_pass, build_dec_window_obs, load_checkpoint, score,
    verify_windows, relevant_vintages, BASE,
    _PREFIX_LOAN_PURPOSE_MAP, _PREFIX_PROPERTY_TYPE_MAP,
)
from score_rolling_one_step import build_rolling_eligible, reference_months

CHECK1_TOL = 1e-6
N_CHECK1_SAMPLE = 2000
SEED = 0
CACHE_DIR = os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')
TRAIL_SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_trail')

CASES = [
    dict(
        name='cutoff_2002_seed42_hist',
        cutoff_year=2002,
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist'),
        ckpt_path=os.path.join(BASE, 'outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist/hazard_best.pt'),
        map_era='fixed',
        include_pre2013=True,
        has_trail_control=False,  # TRAIL_SEQ_DIR is a cutoff_2020-only build
    ),
    dict(
        name='cutoff_2020_f0.2_seed42_GOLDEN_BACKUP',
        cutoff_year=2020,
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP'),
        ckpt_path=os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33/hazard_best.pt'),
        map_era='prefix',
        include_pre2013=False,
        has_trail_control=True,
    ),
    dict(
        name='cutoff_2002_seed42_30y',
        cutoff_year=2002,
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y'),
        ckpt_path=os.path.join(BASE, 'outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s42/hazard_best.pt'),
        map_era='fixed',
        include_pre2013=True,
        has_trail_control=False,  # TRAIL_SEQ_DIR is a cutoff_2020-only build
        cell_sample_path=os.path.join(BASE, 'outputs/pre2013_cell_sample_30y_loans.csv'),
    ),
    dict(
        name='cutoff_2020_seed42_30y',
        cutoff_year=2020,
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_30y'),
        ckpt_path=os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_cutoff_2020_30y_s42/hazard_best.pt'),
        map_era='fixed',
        include_pre2013=False,
        has_trail_control=True,  # TRAIL_SEQ_DIR is cutoff_2020 -- applies here (unlike the cutoff_2002 cases)
    ),
]

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail and not cond else ''), flush=True)
    if not cond:
        FAILURES.append(name)


def check_expect_fail(name, cond_should_fail, detail=''):
    """For negative controls: cond_should_fail is True means the buggy path
    WAS caught (test PASSES, i.e. the negative control worked). False means
    the buggy path silently agreed -- a hole in the check itself."""
    status = 'PASS (bug caught)' if cond_should_fail else 'FAIL (bug NOT caught)'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail else ''), flush=True)
    if not cond_should_fail:
        FAILURES.append(name)


# ── Check 1: inputs and predictions ────────────────────────────────────────

def check1_positive(case):
    t0 = time.time()
    seq_dir = case['seq_dir']
    print(f"\n=== Check 1 (positive): {case['name']} ===", flush=True)

    train_loan_ids  = np.load(os.path.join(seq_dir, 'train_loan_ids.npy'), allow_pickle=True)
    train_ref_month = np.load(os.path.join(seq_dir, 'train_ref_month.npy'))
    train_age       = np.load(os.path.join(seq_dir, 'train_age_at_ref.npy'))
    train_incentive = np.load(os.path.join(seq_dir, 'train_incentive_at_ref.npy'))
    train_incl_prob = np.load(os.path.join(seq_dir, 'train_incl_prob.npy'))
    train_is_term   = np.load(os.path.join(seq_dir, 'train_is_terminal.npy'))
    train_n_elig    = np.load(os.path.join(seq_dir, 'train_n_eligible.npy'))
    train_k_actual  = np.load(os.path.join(seq_dir, 'train_k_actual.npy'))
    train_labels    = np.load(os.path.join(seq_dir, 'train_labels.npy'))
    train_split_ids = np.load(os.path.join(seq_dir, 'train_loan_ids_split.npy'), allow_pickle=True)

    with open(os.path.join(seq_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    pmms_rates = load_pmms()
    zhvi_df    = load_zhvi()
    cutoff_ym  = dec_yyyymm(case['cutoff_year'])
    lp_map, pt_map = (
        (None, None) if case['map_era'] == 'fixed' else
        (_PREFIX_LOAN_PURPOSE_MAP, _PREFIX_PROPERTY_TYPE_MAP)
    )

    # Picked at random each run (not a fixed small file) so the check can't be
    # gamed by a lucky choice of vintage -- logged so a specific run's pick is
    # reproducible (re-seed with the same vintage_pick_seed).
    candidate_vintages = relevant_vintages(case['cutoff_year'], case['include_pre2013'])
    vintage_pick_seed = int.from_bytes(os.urandom(4), 'big')
    vintage = str(np.random.default_rng(vintage_pick_seed).choice(candidate_vintages))
    print(f'  Check 1 vintage picked at random: {vintage} (1 of {len(candidate_vintages)} '
          f'relevant vintages, vintage_pick_seed={vintage_pick_seed})', flush=True)
    print(f'  Loading vintage {vintage} restricted to full train split '
          f'({len(train_split_ids):,} loan_ids)...', flush=True)
    df = load_vintage_filtered(vintage, pmms_rates, zhvi_df, cutoff_ym,
                                keep_ids=set(train_split_ids.tolist()),
                                loan_purpose_map=lp_map, property_type_map=pt_map,
                                cell_sample_path=case.get('cell_sample_path', PRE2013_CELL_SAMPLE_PATH))
    assert df is not None and not df.empty, f'{vintage} produced no rows for this train split -- STOP.'

    vintage_loan_ids = set(df['loan_id'].unique().tolist())
    in_vintage = np.array([lid in vintage_loan_ids for lid in train_loan_ids])
    candidate_idx = np.where(in_vintage)[0]
    check(f'{case["name"]}: candidate train observations from {vintage} > 0',
          len(candidate_idx) > 0, f'{len(candidate_idx)} candidates')
    if len(candidate_idx) == 0:
        return None

    rng = np.random.default_rng(SEED)
    n_sample = min(N_CHECK1_SAMPLE, len(candidate_idx))
    sample_idx = np.sort(rng.choice(candidate_idx, size=n_sample, replace=False))
    print(f'  Sampled {n_sample:,}/{len(candidate_idx):,} candidate observations '
          f'(vintage {vintage}, train split total {len(train_loan_ids):,})', flush=True)

    train_seq  = np.load(os.path.join(seq_dir, 'train_seq.npy'),  mmap_mode='r')
    train_mask = np.load(os.path.join(seq_dir, 'train_mask.npy'), mmap_mode='r')
    stored_seq  = np.asarray(train_seq[sample_idx])
    stored_mask = np.asarray(train_mask[sample_idx])
    sample_loan_ids  = train_loan_ids[sample_idx]

    panel = df.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
    panel['row_idx'] = panel.groupby('loan_id').cumcount()
    panel_key = panel.set_index(['loan_id', 'yyyymm'])['row_idx']

    obs_rows = []
    keep = np.zeros(n_sample, dtype=bool)
    for j, i in enumerate(sample_idx):
        lid, ref_m = train_loan_ids[i], int(train_ref_month[i])
        try:
            t = int(panel_key.loc[(lid, ref_m)])
        except KeyError:
            continue
        keep[j] = True
        obs_rows.append({
            'loan_id': lid, 't': t, 'ref_month': ref_m,
            'label': float(train_labels[i]), 'is_terminal': bool(train_is_term[i]),
            'incl_prob': float(train_incl_prob[i]), 'n_eligible': int(train_n_elig[i]),
            'k_actual': int(train_k_actual[i]), 'age_at_ref': float(train_age[i]),
            'incentive_at_ref': float(train_incentive[i]),
        })
    n_missing_row = n_sample - int(keep.sum())
    check(f'{case["name"]}: every sampled train observation has exactly one '
          f'matching raw panel row at its stored ref_month',
          n_missing_row == 0, f'{n_missing_row}/{n_sample} missing')
    obs = pd.DataFrame(obs_rows)
    stored_seq, stored_mask = stored_seq[keep], stored_mask[keep]
    sample_loan_ids = sample_loan_ids[keep]

    sequences, masks, _l, _p, loan_ids_out, _extras = build_sequences_multiobs(
        df, scaler, k_draws=1, H=1, min_hist=1, obs=obs)
    assert list(loan_ids_out) == list(sample_loan_ids), \
        'obs/loan_ids_out order mismatch -- build_sequences_multiobs must preserve obs row order.'

    max_seq_diff = float(np.abs(sequences - stored_seq).max())
    mask_match = bool(np.array_equal(masks, stored_mask))
    n_mask_mismatch = int((masks != stored_mask).sum())
    print(f'  n={len(sample_loan_ids):,}  max_seq_diff={max_seq_diff:.3e}  '
          f'mask_match={mask_match}  n_mask_mismatch={n_mask_mismatch}', flush=True)
    check(f'{case["name"]}: rebuilt sequences match stored train_seq (< {CHECK1_TOL:.0e})',
          max_seq_diff < CHECK1_TOL, f'max abs diff={max_seq_diff:.3e}')
    check(f'{case["name"]}: rebuilt masks match stored train_mask exactly',
          mask_match, f'{n_mask_mismatch} mismatched entries')

    model = load_checkpoint(case['ckpt_path'])
    h_stored  = score(model, stored_seq, stored_mask)
    h_rebuilt = score(model, sequences, masks)
    max_pred_diff = float(np.abs(h_stored - h_rebuilt).max())
    print(f'  n={len(sample_loan_ids):,}  max_pred_diff={max_pred_diff:.3e}', flush=True)
    check(f'{case["name"]}: predictions from the same checkpoint match (< {CHECK1_TOL:.0e})',
          max_pred_diff < CHECK1_TOL, f'max abs diff={max_pred_diff:.3e}')

    elapsed = time.time() - t0
    print(f'  Check 1 positive elapsed: {elapsed:.1f}s', flush=True)
    return {
        'sample_loan_ids': sample_loan_ids, 'sample_ref_month': obs['ref_month'].to_numpy(),
        'stored_seq': stored_seq, 'stored_mask': stored_mask, 'model': model,
    }


def check1_negative_trail(case, positive_result):
    """Negative control (a): TRAIL_SEQ_DIR's old LEFT-aligned trailing
    sequences, matched by loan_id against the correct multiobs train
    sample, must NOT match -- proving Check 1 would catch the exact bug
    found 2026-09-24 (forecast_matched_population_cpr.py scoring a
    right-aligned-trained model on left-aligned trailing sequences)."""
    if not case.get('has_trail_control') or positive_result is None:
        return
    print(f"\n=== Check 1 (negative control a): {case['name']} vs TRAIL_SEQ_DIR ===", flush=True)

    trail_ids  = np.load(os.path.join(TRAIL_SEQ_DIR, 'train_loan_ids.npy'), allow_pickle=True)
    trail_seq  = np.load(os.path.join(TRAIL_SEQ_DIR, 'train_seq.npy'),  mmap_mode='r')
    trail_mask = np.load(os.path.join(TRAIL_SEQ_DIR, 'train_mask.npy'), mmap_mode='r')
    trail_pos  = {lid: idx for idx, lid in enumerate(trail_ids)}  # first occurrence

    sample_loan_ids = positive_result['sample_loan_ids']
    stored_seq  = positive_result['stored_seq']
    stored_mask = positive_result['stored_mask']

    overlap_idx, trail_idx = [], []
    for i, lid in enumerate(sample_loan_ids):
        if lid in trail_pos:
            overlap_idx.append(i)
            trail_idx.append(trail_pos[lid])
    check(f'{case["name"]}: negative control (a) has shared loans with TRAIL_SEQ_DIR to test',
          len(overlap_idx) > 0, f'{len(overlap_idx)} shared loans')
    if not overlap_idx:
        return

    correct_seq  = stored_seq[overlap_idx]
    correct_mask = stored_mask[overlap_idx]
    trail_seq_sub  = np.asarray(trail_seq[trail_idx])
    trail_mask_sub = np.asarray(trail_mask[trail_idx])

    per_obs_max_diff = np.abs(correct_seq - trail_seq_sub).reshape(len(overlap_idx), -1).max(axis=1)
    n_disagree = int((per_obs_max_diff >= CHECK1_TOL).sum())
    check_expect_fail(
        f'{case["name"]}: negative control (a) -- TRAIL_SEQ_DIR (left-aligned) sequences '
        f'disagree with the correct (right-aligned) train sequences',
        n_disagree > 0,
        f'{n_disagree}/{len(overlap_idx)} shared observations disagree '
        f'(max abs diff over all: {float(per_obs_max_diff.max()):.3e})')


# ── Check 2: time alignment ─────────────────────────────────────────────────

def _load_test_population(case):
    seq_dir = case['seq_dir']
    with open(os.path.join(seq_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    test_ids = np.load(os.path.join(seq_dir, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    full_df = build_combined_pass(case['cutoff_year'], case['include_pre2013'], case['map_era'],
                                   test_ids_set, CACHE_DIR,
                                   cell_sample_path=case.get('cell_sample_path', PRE2013_CELL_SAMPLE_PATH))
    return scaler, full_df


def check2_positive(case, full_df, scaler):
    t0 = time.time()
    print(f"\n=== Check 2 (positive): {case['name']} ===", flush=True)
    elig = build_rolling_eligible(full_df)
    ref_months = reference_months(case['cutoff_year'])
    sample_months = [ref_months[0], ref_months[len(ref_months) // 2], ref_months[-1]]

    for ref_ym in sample_months:
        month_rows_all = elig[elig['yyyymm'] == ref_ym]
        month_rows = month_rows_all[month_rows_all['forward_adjacent']].copy()
        if month_rows.empty:
            check(f'{case["name"]}: check 2 month {ref_ym} has eligible forward-adjacent rows',
                  False, 'zero rows -- cannot test this month')
            continue
        obs = pd.DataFrame({
            'loan_id':          month_rows['loan_id'].to_numpy(),
            't':                month_rows['row_idx'].astype(int).to_numpy(),
            'ref_month':        month_rows['yyyymm'].astype(int).to_numpy(),
            'label':            np.zeros(len(month_rows), dtype=np.float32),
            'is_terminal':      np.zeros(len(month_rows), dtype=bool),
            'incl_prob':        np.ones(len(month_rows), dtype=np.float32),
            'n_eligible':       np.ones(len(month_rows), dtype=np.int32),
            'k_actual':         np.ones(len(month_rows), dtype=np.int32),
            'age_at_ref':       month_rows['loan_age_months'].astype(float).to_numpy(),
            'incentive_at_ref': month_rows['refi_incentive'].astype(float).to_numpy(),
        })
        sequences, masks, _l, _p, _lids, _extras = build_sequences_multiobs(
            full_df, scaler, k_draws=1, H=1, min_hist=1, obs=obs)
        try:
            verify_windows(sequences, masks, obs, month_rows, scaler)
            check(f'{case["name"]}: check 2 month {ref_ym} -- window ends exactly at ref_month '
                  f'(last timestep unmasked, features match)', True)
        except AssertionError as e:
            check(f'{case["name"]}: check 2 month {ref_ym} -- window ends exactly at ref_month',
                  False, str(e))
    print(f'  Check 2 positive elapsed: {time.time() - t0:.1f}s', flush=True)


def check2_negative_frozen(case, full_df, scaler):
    """Negative control (b): a frozen-at-Dec-cutoff window, checked against
    a LATER month's actual features, must NOT match -- proving Check 2
    would catch the exact bug found 2026-09-24 (first 2003 test forecast
    later months from frozen December features)."""
    print(f"\n=== Check 2 (negative control b): {case['name']} ===", flush=True)
    obs, dec_rows, diag = build_dec_window_obs(full_df, case['cutoff_year'])
    sequences, masks, _l, _p, loan_ids_out, _extras = build_sequences_multiobs(
        full_df, scaler, k_draws=1, H=1, min_hist=1, obs=obs)

    later_ym = (case['cutoff_year'] + 1) * 100 + 6   # June of the forecast year
    panel = _prepare_panel(full_df)
    later_rows = panel[(panel['loan_id'].isin(set(loan_ids_out.tolist()))) &
                        (panel['yyyymm'] == later_ym)].set_index('loan_id')

    common_ids = [lid for lid in loan_ids_out if lid in later_rows.index]
    check(f'{case["name"]}: negative control (b) has loans active at both Dec-cutoff and June '
          f'of the forecast year to test', len(common_ids) > 0, f'{len(common_ids)} loans')
    if not common_ids:
        return

    id_to_pos = {lid: i for i, lid in enumerate(loan_ids_out)}
    idx = [id_to_pos[lid] for lid in common_ids]
    frozen_last_ts = sequences[idx, -1, :]
    later_raw = later_rows.loc[common_ids, FEATURE_COLS].to_numpy()
    later_scaled = scaler.transform(later_raw)

    per_obs_max_diff = np.abs(frozen_last_ts - later_scaled).max(axis=1)
    n_disagree = int((per_obs_max_diff >= CHECK1_TOL).sum())
    check_expect_fail(
        f'{case["name"]}: negative control (b) -- frozen Dec-cutoff window\'s last timestep '
        f'disagrees with June-of-forecast-year\'s actual features (i.e. is NOT that month\'s window)',
        n_disagree > 0,
        f'{n_disagree}/{len(common_ids)} loan-months disagree '
        f'(max abs diff over all: {float(per_obs_max_diff.max()):.3e})')


def run(cases):
    overall_t0 = time.time()
    for case in cases:
        pos_result = check1_positive(case)
        check1_negative_trail(case, pos_result)
        scaler, full_df = _load_test_population(case)
        check2_positive(case, full_df, scaler)
        check2_negative_frozen(case, full_df, scaler)

    print(f'\nTotal elapsed: {time.time() - overall_t0:.1f}s', flush=True)
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S): {FAILURES}')
        sys.exit(1)
    print(f'All checks passed ({len(cases)} cases).')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=str, default=None,
                     help="run only the named CASES entry (see CASES[*]['name']); "
                          'default runs the full standing-test suite (all cases).')
    args = ap.parse_args()
    if args.case is None:
        run(CASES)
    else:
        matches = [c for c in CASES if c['name'] == args.case]
        if not matches:
            print(f'no case named {args.case!r}; known: {[c["name"] for c in CASES]}')
            sys.exit(1)
        run(matches)
