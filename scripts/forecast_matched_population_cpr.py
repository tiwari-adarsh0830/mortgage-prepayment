"""
forecast_matched_population_cpr.py — matched-population CPR comparison across
the origination-anchored, trailing-window, and multiobs hazard models
(cutoff_2020 -> FY2021), reusing forecast_rolling_cpr.py's inference/aggregate
machinery UNCHANGED.

Why the TRAILING test sequences for the multiobs model, not the multiobs ones
--------------------------------------------------------------------------
aggregate() needs exactly ONE forward-looking window per loan, ending at the
Dec 2020 cutoff, to produce one hazard-derived annual prepay probability per
loan. The multiobs test set
(data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1/test_seq.npy) has
MULTIPLE sampled ref_months per loan by design (observing the same loan at
several ages to separate incentive from burnout), so it is not a
one-row-per-loan-ending-at-cutoff population and is not usable here. This
script scores the multiobs-TRAINED MODEL against the TRAILING builder's test
sequences (data/sequences_rolling/cutoff_2020_zbc_trail/) instead, exactly
the population the origination and trailing runs already use.

Why NOT forecast_rolling_cpr.py's infer_test_set() for the multiobs model
---------------------------------------------------------------------------
infer_test_set() calls model(seq, mask, return_per_timestep=True) and
gathers the hazard at each loan's last real timestep -- correct for the
origination/trailing models, whose training loop (evaluate_hazard) gives
every timestep's logit a per-month-hazard meaning. The multiobs checkpoint's
classifier was NEVER trained that way: train_hazard_multiobs.py's evaluate()
calls model(seq, mask) with NO return_per_timestep -- one pooled logit per
whole window, no per-timestep meaning at all. score_multiobs_model() below
mirrors evaluate() exactly. infer_test_set() IS reused unchanged for the
origination and trailing models, for which it is the correct scoring path.

Population mismatch: matched via intersection
-------------------------------------------------
The multiobs prep run reused the trailing builder's train/test loan-id SPLIT
(`--reuse_from data/sequences_rolling/cutoff_2020_zbc_trail`), but
select_observations() (prepare_sequences_multiobs_zbc.py) can emit ZERO
sampled rows for a split-assigned loan with too little post-dropna history --
that loan stays in the split but is absent from the multiobs test_loan_ids.npy.
Measured: origination and trailing test sets are identical (374,182 loans,
symmetric diff 0); multiobs is a strict subset missing 9,036 of those loans
(365,146 loans). All three pooled/dispersion numbers below are therefore
computed on the 365,146-loan INTERSECTION of all three populations, so the
comparison isolates model differences, not population differences.

Base-rate correction: origination and trailing get one, multiobs does NOT
-----------------------------------------------------------------------------
origination/trailing apply prior_shift_offset() -- a logit shift undoing
HazardSampler's known 50/50 prepaid/non-prepaid draw, which is well-defined
because that sampler draws BOTH classes at controllable, known rates.
Multiobs's sampling design has no analogous single-scalar correction:
  - the MANDATORY/terminal draw is always included when eligible (incl_prob
    = 1.0 by construction) -- not sampled at any rate at all;
  - the remaining pool draws' incl_prob varies PER OBSERVATION by loan
    length / stratum size (k_s/n_s), not by a single global class rate.
A King-Zeng-style flat intercept shift assumes one scalar can characterize
the class-conditional sampling odds; here the "sampling odds" are a
per-observation quantity (incl_prob, emitted but never applied -- this
checkpoint trained with --use_ipw=False), so no such scalar is theoretically
justified. A properly justified correction would require either an
--use_ipw retrain of train_hazard_multiobs.py, or per-observation
reweighting via incl_prob applied at inference -- NEITHER has been done.
This script therefore reports multiobs's pooled_ratio UNCORRECTED
(logit_offset=0.0) and marks it not comparable to origination's 0.8631 or
trailing's 1.1273 everywhere it is printed.

Per-coupon DISPERSION (max ratio .. min ratio across coupons) is reported as
the PRIMARY result instead: it is the metric that actually answers the
burnout-inversion question this whole line of investigation is about, and it
is comparatively robust to a uniform/proportional base-rate bias, since such
a bias scales forecast_cpr similarly across coupons and largely cancels out
of a forecast/realized RATIO comparison (it does not cancel out of the
pooled ratio, which is why that number needs the caveat and dispersion does
not).

Do NOT modify forecast_rolling_cpr.py or any existing output CSV -- its
infer_test_set(), read_coupon_and_realized(), aggregate(), and
prior_shift_offset() are imported and called UNCHANGED. This script writes
only NEW files (matched-population CSVs alongside the existing ones, never
overwriting them).

Usage:
    python scripts/forecast_matched_population_cpr.py
"""
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_rolling_cpr import (
    load_model, infer_test_set, read_coupon_and_realized, aggregate,
    prior_shift_offset, DEVICE, BASE,
)

CUTOFF_YEAR = 2020
BATCH_SIZE  = 8192

ORIG_LABEL_SUFFIX     = '_zbc'
TRAIL_LABEL_SUFFIX    = '_zbc_trail'
MULTIOBS_LABEL_SUFFIX = '_multiobs_k5_h1'

ORIG_SEQ_DIR     = os.path.join(BASE, f'data/sequences_rolling/cutoff_{CUTOFF_YEAR}{ORIG_LABEL_SUFFIX}')
TRAIL_SEQ_DIR    = os.path.join(BASE, f'data/sequences_rolling/cutoff_{CUTOFF_YEAR}{TRAIL_LABEL_SUFFIX}')
# NOTE: prepare_sequences_multiobs_zbc.py names its own output dir
# `cutoff_{YEAR}_zbc_multiobs_k{k}_h{H}` (its own docstring), which is NOT
# `cutoff_{YEAR}{MULTIOBS_LABEL_SUFFIX}` -- that template only matches
# train_hazard_multiobs.py's OUTPUT dir naming (no `_zbc` infix). Hardcoded
# separately so the two naming conventions can't be silently conflated again.
MULTIOBS_SEQ_DIR = os.path.join(BASE, f'data/sequences_rolling/cutoff_{CUTOFF_YEAR}_zbc_multiobs_k5_h1')

ORIG_OUT_DIR     = os.path.join(BASE, f'outputs/rolling/cutoff_{CUTOFF_YEAR}{ORIG_LABEL_SUFFIX}')
TRAIL_OUT_DIR    = os.path.join(BASE, f'outputs/rolling/cutoff_{CUTOFF_YEAR}{TRAIL_LABEL_SUFFIX}')
MULTIOBS_OUT_DIR = os.path.join(BASE, f'outputs/rolling/cutoff_{CUTOFF_YEAR}{MULTIOBS_LABEL_SUFFIX}')

# read_coupon_and_realized() is a ~40-min raw pass over 32 vintage CSVs and
# depends only on (cutoff_year, population) -- neither changes across reruns
# of this script for follow-up diagnostics, so cache it. Keyed by a hash of
# the sorted population so a stale cache from a different population can
# never be silently reused.
RAW_PASS_CACHE = os.path.join(MULTIOBS_OUT_DIR, '_raw_pass_cache.pkl')

MULTIOBS_CAVEAT = (
    "NOT COMPARABLE to origination (0.8631) or trailing (1.1273): multiobs has no "
    "analogous base-rate correction. Its mandatory/terminal draw is always included "
    "when eligible (incl_prob=1.0 by construction, not sampled at any rate), while "
    "pool-draw incl_prob varies per observation by loan length/stratum size -- a "
    "single scalar (King-Zeng-style) shift cannot represent that, so none is applied "
    "here. A justified correction would require an --use_ipw retrain or per-observation "
    "incl_prob reweighting at inference; neither has been done."
)


def score_multiobs_model(model, seq_dir: str, batch_size: int = BATCH_SIZE):
    """Mirrors train_hazard_multiobs.py's evaluate(): model(seq, mask), NO
    return_per_timestep, sigmoid(pooled logit) -- one score per loan, since
    seq_dir here is the trailing builder's one-row-per-loan test set. See
    module docstring for why infer_test_set() is wrong for this checkpoint."""
    seqs  = np.load(os.path.join(seq_dir, 'test_seq.npy'),      mmap_mode='r')
    masks = np.load(os.path.join(seq_dir, 'test_mask.npy'),     mmap_mode='r')
    ids   = np.load(os.path.join(seq_dir, 'test_loan_ids.npy'), allow_pickle=True)
    n = len(seqs)
    print(f'Trailing test set (scoring input for multiobs model): {n:,} loans  ({seqs.shape})', flush=True)

    h_vals = np.zeros(n, dtype=np.float32)
    model.eval()
    n_batches = (n + batch_size - 1) // batch_size
    with torch.no_grad():
        for b, i in enumerate(range(0, n, batch_size)):
            sb = torch.from_numpy(np.ascontiguousarray(seqs[i:i + batch_size])).to(DEVICE)
            mb = torch.from_numpy(np.ascontiguousarray(masks[i:i + batch_size])).to(DEVICE)
            logits = model(sb, mask=mb)                       # NO return_per_timestep
            h_vals[i:i + sb.shape[0]] = torch.sigmoid(logits).cpu().numpy()
            if b % 10 == 0 or b == n_batches - 1:
                print(f'  score batch {b + 1}/{n_batches} ({i + sb.shape[0]:,}/{n:,})', flush=True)
    print(f'  h_t mean={h_vals.mean():.5f}  max={h_vals.max():.4f}', flush=True)
    return ids, h_vals


def filter_to_population(loan_ids, h_vals, keep_set):
    keep_mask = np.fromiter((lid in keep_set for lid in loan_ids), dtype=bool, count=len(loan_ids))
    return loan_ids[keep_mask], h_vals[keep_mask]


def population_hash(population: set) -> str:
    arr = np.array(sorted(population), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def cached_read_coupon_and_realized(cutoff_year: int, population: set, cache_path: str):
    """Wraps read_coupon_and_realized() UNCHANGED with an on-disk cache keyed
    by (cutoff_year, population hash) -- the raw pass itself is not modified,
    only whether it runs at all on a given invocation."""
    pop_hash = population_hash(population)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('cutoff_year') == cutoff_year and cache.get('pop_hash') == pop_hash:
            print(f'Reusing cached raw pass ({cache_path}) -- population hash matches.', flush=True)
            return cache['coupon_map'], cache['active_set'], cache['prepaid_set']
        print('Raw-pass cache exists but does not match (cutoff_year/population changed) -- recomputing.', flush=True)

    coupon_map, active_set, prepaid_set, _zip3, _origdate = read_coupon_and_realized(cutoff_year, population)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'cutoff_year': cutoff_year, 'pop_hash': pop_hash,
                     'coupon_map': coupon_map, 'active_set': active_set,
                     'prepaid_set': prepaid_set}, f)
    return coupon_map, active_set, prepaid_set


def mean_h_adj_by_coupon(loan_ids, h_vals, coupon_map, active_set, logit_offset: float = 0.0) -> pd.DataFrame:
    """Mirrors aggregate()'s preprocessing (loan_id.isin(active_set), the
    SAME coupon derivation, and the SAME offset-adjustment formula) so the
    returned monthly h_adj is exactly the value aggregate() feeds into
    1-(1-h)^12 internally and discards -- NOT a reimplementation of
    aggregate() itself (which remains the sole source of forecast_cpr /
    realized_cpr everywhere else in this file), just an exposed view of the
    one intermediate aggregate() does not return.
    """
    df = pd.DataFrame({'loan_id': loan_ids, 'h_t': h_vals})
    df = df[df['loan_id'].isin(active_set)].copy()
    df['note_rate'] = df['loan_id'].map(coupon_map)
    df = df.dropna(subset=['note_rate'])
    df['coupon'] = ((df['note_rate'] - 0.5) * 2).round() / 2

    _h = df['h_t'].clip(1e-7, 1 - 1e-7)
    if logit_offset != 0.0:
        _h = 1.0 / (1.0 + np.exp(-(np.log(_h / (1 - _h)) + logit_offset)))
    df['h_adj'] = _h

    out = df.groupby('coupon')['h_adj'].agg(['mean', 'count']).reset_index()
    out.columns = ['coupon', 'h_t_mean_monthly', 'n_loans_check']
    return out


def pooled_comparison(df: pd.DataFrame, min_n: int = 5000, coupon_range=(2.0, 5.0)) -> dict:
    """The pooling/dispersion math traced out of the Sep 1/Aug 31 README
    sections (f35b0bc, 690cbb8) and verified numerically against the
    surviving rolling_cpr_forecast.csv files for the origination (pooled
    ratio 0.8631) and trailing (1.1273) runs — see main()'s self-check.
    First committed, reusable version of this math; do not reimplement it
    inline a third time.

      dispersion   = max(ratio) .. min(ratio) across the filtered coupon
                     rows, where ratio(coupon) = forecast_cpr / realized_cpr.
      pooled_ratio = pooled_forecast / pooled_realized, where pooled_forecast
                     and pooled_realized are n_loans-WEIGHTED means across
                     the filtered rows (re-pooling the underlying loan
                     population) — NOT a mean of the per-coupon ratios; the
                     two are not the same number.

    This function makes no judgment about whether pooled_ratio is a fair
    comparison across two dataframes -- that depends on whether both were
    produced with an equivalent base-rate correction, which is a caller-side
    concern (see MULTIOBS_CAVEAT in this file).
    """
    lo, hi = coupon_range
    f = df[(df['coupon'] >= lo) & (df['coupon'] <= hi) & (df['n_loans'] >= min_n)].copy()
    if f.empty:
        raise ValueError(f'pooled_comparison: no coupon rows survive the filter '
                          f'(range={coupon_range}, min_n={min_n})')

    f['ratio'] = f['forecast_cpr'] / f['realized_cpr']
    n_total = int(f['n_loans'].sum())
    pooled_forecast = float((f['forecast_cpr'] * f['n_loans']).sum() / n_total)
    pooled_realized = float((f['realized_cpr'] * f['n_loans']).sum() / n_total)

    return {
        'table':           f[['coupon', 'forecast_cpr', 'realized_cpr', 'n_loans', 'ratio']].reset_index(drop=True),
        'dispersion_max':  float(f['ratio'].max()),
        'dispersion_min':  float(f['ratio'].min()),
        'n_total':         n_total,
        'pooled_forecast': pooled_forecast,
        'pooled_realized': pooled_realized,
        'pooled_ratio':    pooled_forecast / pooled_realized,
    }


def main():
    print(f'Device: {DEVICE}', flush=True)

    # ── Step 0: population check across all three runs ──────────────────────
    orig_ids_raw  = np.load(os.path.join(ORIG_SEQ_DIR,     'test_loan_ids.npy'), allow_pickle=True)
    trail_ids_raw = np.load(os.path.join(TRAIL_SEQ_DIR,    'test_loan_ids.npy'), allow_pickle=True)
    multi_ids_raw = np.load(os.path.join(MULTIOBS_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    orig_set  = set(orig_ids_raw.tolist())
    trail_set = set(trail_ids_raw.tolist())
    multi_set = set(multi_ids_raw.tolist())

    print(f'Origination test loans (unique): {len(orig_set):,}', flush=True)
    print(f'Trailing test loans (unique):    {len(trail_set):,}', flush=True)
    print(f'Multiobs test loans (unique):    {len(multi_set):,}', flush=True)
    print(f'len(set(trail_ids) ^ set(multiobs_ids)) = {len(trail_set ^ multi_set):,}', flush=True)

    population = orig_set & trail_set & multi_set
    print(f'\nIntersection of all three (MATCHED population for the comparison below): '
          f'{len(population):,}', flush=True)
    print(f'  dropped from origination: {len(orig_set - population):,}', flush=True)
    print(f'  dropped from trailing:    {len(trail_set - population):,}', flush=True)
    print(f'  dropped from multiobs:    {len(multi_set - population):,}', flush=True)

    # ── Step 1: score all three models on their own native test sequences ───
    print('\n[1/4] Scoring origination model (infer_test_set, unchanged)...', flush=True)
    orig_model = load_model(CUTOFF_YEAR, ORIG_LABEL_SUFFIX)
    ids_orig, h_orig = infer_test_set(CUTOFF_YEAR, orig_model, BATCH_SIZE, ORIG_LABEL_SUFFIX)

    print('\n[1/4] Scoring trailing model (infer_test_set, unchanged)...', flush=True)
    trail_model = load_model(CUTOFF_YEAR, TRAIL_LABEL_SUFFIX)
    ids_trail, h_trail = infer_test_set(CUTOFF_YEAR, trail_model, BATCH_SIZE, TRAIL_LABEL_SUFFIX)

    print('\n[1/4] Scoring multiobs model (score_multiobs_model, on trailing sequences)...', flush=True)
    multi_model = load_model(CUTOFF_YEAR, MULTIOBS_LABEL_SUFFIX)
    ids_multi, h_multi = score_multiobs_model(multi_model, TRAIL_SEQ_DIR, BATCH_SIZE)

    # ── Step 2: restrict all three to the matched population ────────────────
    ids_orig,  h_orig  = filter_to_population(ids_orig,  h_orig,  population)
    ids_trail, h_trail = filter_to_population(ids_trail, h_trail, population)
    ids_multi, h_multi = filter_to_population(ids_multi, h_multi, population)
    assert len(ids_orig) == len(ids_trail) == len(ids_multi) == len(population), (
        f'Matched-population filter did not land on the same count for all three: '
        f'orig={len(ids_orig)} trail={len(ids_trail)} multi={len(ids_multi)} '
        f'expected={len(population)}')
    print(f'\nAll three scored populations restricted to {len(population):,} loans.', flush=True)

    # ── Step 3: ONE raw pass for coupon + realized, shared by all three ─────
    print('\n[2/4] Raw pass for coupon + realized (shared across all three, cached)...', flush=True)
    coupon_map, active_set, prepaid_set = cached_read_coupon_and_realized(
        CUTOFF_YEAR, population, RAW_PASS_CACHE)

    # ── Step 4: aggregate -- unchanged; multiobs UNCORRECTED (see caveat) ────
    print('\n[3/4] Aggregating to coupon-level CPR (matched population)...', flush=True)
    off_orig  = prior_shift_offset(ORIG_SEQ_DIR)
    off_trail = prior_shift_offset(TRAIL_SEQ_DIR)
    off_multi = 0.0   # deliberately uncorrected -- see MULTIOBS_CAVEAT

    result_orig  = aggregate(ids_orig,  h_orig,  coupon_map, active_set, prepaid_set,
                              logit_offset=off_orig,  already_annual=False)
    result_trail = aggregate(ids_trail, h_trail, coupon_map, active_set, prepaid_set,
                              logit_offset=off_trail, already_annual=False)
    result_multi = aggregate(ids_multi, h_multi, coupon_map, active_set, prepaid_set,
                              logit_offset=off_multi, already_annual=False)

    for name, res, off, out_dir in [
        ('origination', result_orig,  off_orig,  ORIG_OUT_DIR),
        ('trailing',    result_trail, off_trail, TRAIL_OUT_DIR),
        ('multiobs',    result_multi, off_multi, MULTIOBS_OUT_DIR),
    ]:
        res['logit_offset']  = off
        res['cutoff_year']   = CUTOFF_YEAR
        res['forecast_year'] = CUTOFF_YEAR + 1
        res['time_varying']  = False
        res['matched_pop_n'] = len(population)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'rolling_cpr_forecast_matched.csv')
        res.to_csv(out_path, index=False)
        print(f'\nSaved ({name}, matched population): {out_path}', flush=True)
        print(res.to_string(index=False), flush=True)

    # ── Step 5: self-check -- pooled_comparison() against the KNOWN, UNRESTRICTED values ──
    orig_full_df  = pd.read_csv(os.path.join(BASE, 'outputs/rolling/cutoff_2020_zbc/rolling_cpr_forecast.csv'))
    trail_full_df = pd.read_csv(os.path.join(BASE, 'outputs/rolling/cutoff_2020_zbc_trail/rolling_cpr_forecast.csv'))
    orig_full_pc  = pooled_comparison(orig_full_df)
    trail_full_pc = pooled_comparison(trail_full_df)

    print('\n=== Self-check: pooled_comparison() against known README values (full, unrestricted population) ===')
    print(f"origination: pooled_ratio={orig_full_pc['pooled_ratio']:.4f}  (known: 0.8631)  "
          f"dispersion={orig_full_pc['dispersion_max']:.3f}..{orig_full_pc['dispersion_min']:.3f}  (known: 1.92..0.62)")
    print(f"trailing:    pooled_ratio={trail_full_pc['pooled_ratio']:.4f}  (known: 1.1273)  "
          f"dispersion={trail_full_pc['dispersion_max']:.3f}..{trail_full_pc['dispersion_min']:.3f}  (known: 3.29..0.56)")
    assert abs(orig_full_pc['pooled_ratio'] - 0.8631) < 5e-4, (
        'pooled_comparison() does NOT reproduce the known origination pooled ratio -- '
        'STOP, do not trust the matched-population numbers below.')
    assert abs(trail_full_pc['pooled_ratio'] - 1.1273) < 5e-4, (
        'pooled_comparison() does NOT reproduce the known trailing pooled ratio -- '
        'STOP, do not trust the matched-population numbers below.')
    print('Self-check PASSED.')

    # ── Step 6: matched-population pooled + dispersion, all three ───────────
    orig_pc  = pooled_comparison(result_orig)
    trail_pc = pooled_comparison(result_trail)
    multi_pc = pooled_comparison(result_multi)

    print(f'\n{"=" * 78}')
    print(f'PRIMARY RESULT -- per-coupon DISPERSION (max ratio .. min ratio, '
          f'coupons 2.0-5.0, n>=5000)')
    print(f'Matched population: {len(population):,} loans. This is the metric that answers')
    print(f'the burnout-inversion question, and is comparatively robust to a uniform/')
    print(f'proportional base-rate bias (such a bias scales forecast_cpr similarly across')
    print(f'coupons and largely cancels out of a forecast/realized ratio comparison).')
    print(f'{"=" * 78}')
    print(f"{'run':<14}{'dispersion (max..min)':>26}")
    for name, pc in [('origination', orig_pc), ('trailing', trail_pc), ('multiobs', multi_pc)]:
        disp = f"{pc['dispersion_max']:.3f}..{pc['dispersion_min']:.3f}"
        print(f"{name:<14}{disp:>26}")

    print(f'\n{"=" * 78}')
    print('SECONDARY -- pooled ratio (population-level forecast/realized)')
    print(f'{"=" * 78}')
    print(f"{'run':<14}{'pooled_ratio':>14}")
    for name, pc in [('origination', orig_pc), ('trailing', trail_pc), ('multiobs', multi_pc)]:
        flag = '  <-- NOT COMPARABLE, see caveat below' if name == 'multiobs' else ''
        print(f"{name:<14}{pc['pooled_ratio']:>14.4f}{flag}")
    print(f'\n*** CAVEAT on multiobs pooled_ratio ***')
    print(MULTIOBS_CAVEAT)

    # ── Step 7: full combined per-coupon table, matched population ──────────
    merged = (
        orig_pc['table'][['coupon', 'realized_cpr', 'forecast_cpr', 'ratio', 'n_loans']]
            .rename(columns={'forecast_cpr': 'orig_forecast', 'ratio': 'orig_ratio'})
            .merge(trail_pc['table'][['coupon', 'forecast_cpr', 'ratio']]
                   .rename(columns={'forecast_cpr': 'trail_forecast', 'ratio': 'trail_ratio'}),
                   on='coupon')
            .merge(multi_pc['table'][['coupon', 'forecast_cpr', 'ratio']]
                   .rename(columns={'forecast_cpr': 'multi_forecast', 'ratio': 'multi_ratio'}),
                   on='coupon')
            .sort_values('coupon')
    )
    print(f'\n{"=" * 78}')
    print(f'FULL TABLE -- matched population ({len(population):,} loans), '
          f'cutoff_2020 -> FY2021, coupons 2.0-5.0, n_loans >= 5000')
    print('origination-matched / trailing-matched / multiobs-uncorrected(matched)')
    print(f'{"=" * 78}')
    print(merged.to_string(index=False))

    # ── Step 8: mean monthly h_t (pre-annualization) per coupon, all three ──
    lo, hi, min_n = 2.0, 5.0, 5000
    h_orig_tab  = mean_h_adj_by_coupon(ids_orig,  h_orig,  coupon_map, active_set, off_orig)
    h_trail_tab = mean_h_adj_by_coupon(ids_trail, h_trail, coupon_map, active_set, off_trail)
    h_multi_tab = mean_h_adj_by_coupon(ids_multi, h_multi, coupon_map, active_set, off_multi)

    def _filt(t):
        return t[(t['coupon'] >= lo) & (t['coupon'] <= hi) & (t['n_loans_check'] >= min_n)]
    h_orig_tab, h_trail_tab, h_multi_tab = _filt(h_orig_tab), _filt(h_trail_tab), _filt(h_multi_tab)

    h_merged = (
        h_orig_tab.rename(columns={'h_t_mean_monthly': 'orig_h_t'})[['coupon', 'orig_h_t', 'n_loans_check']]
            .merge(h_trail_tab.rename(columns={'h_t_mean_monthly': 'trail_h_t'})[['coupon', 'trail_h_t']], on='coupon')
            .merge(h_multi_tab.rename(columns={'h_t_mean_monthly': 'multi_h_t'})[['coupon', 'multi_h_t']], on='coupon')
    )
    full = merged.merge(h_merged, on='coupon').sort_values('coupon')
    assert (full['n_loans'] == full['n_loans_check']).all(), (
        'n_loans mismatch between aggregate() and mean_h_adj_by_coupon() -- '
        'the coupon/active_set filtering has diverged, do not trust the h_t table.')

    print(f'\n{"=" * 78}')
    print('MONTHLY h_t (pre-annualization) vs annual forecast_cpr, per coupon')
    print('h_t is the mean monthly hazard fed into 1-(1-h)^12 (post logit_offset, '
          'multiobs offset=0.0). At h_t~0.05, (1-h)^12 barely bends off linear '
          '(12h); at h_t>=0.10, it is compressing hard toward the ceiling.')
    print(f'{"=" * 78}')
    print(full[['coupon', 'realized_cpr',
                 'orig_h_t', 'orig_forecast',
                 'trail_h_t', 'trail_forecast',
                 'multi_h_t', 'multi_forecast']].to_string(index=False))

    print(f'\n{"=" * 78}')
    print(f'For reference -- original UNRESTRICTED numbers (374,182 loans, existing CSVs)')
    print(f'{"=" * 78}')
    print(f"{'run':<14}{'pooled_ratio':>14}{'dispersion (max..min)':>26}")
    for name, pc in [('origination', orig_full_pc), ('trailing', trail_full_pc)]:
        disp = f"{pc['dispersion_max']:.3f}..{pc['dispersion_min']:.3f}"
        print(f"{name:<14}{pc['pooled_ratio']:>14.4f}{disp:>26}")

    print('\nDone.')


if __name__ == '__main__':
    main()
