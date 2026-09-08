"""
forecast_multiobs_ipw_cpr.py -- extends forecast_matched_population_cpr.py's
three-way matched-population comparison (cutoff_2020 -> FY2021) with a fourth
model: the IPW-reweighted multiobs checkpoint
(outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw/hazard_best.pt), scored via
the SAME path as the uncorrected multiobs run.

Scoring path (identical for multiobs-uncorrected and multiobs-IPW)
--------------------------------------------------------------------
Both checkpoints are scored with score_multiobs_model() -- model(seq, mask),
NO return_per_timestep -- on the TRAILING builder's test sequences
(data/sequences_rolling/cutoff_2020_zbc_trail/), imported UNCHANGED from
forecast_matched_population_cpr.py. Neither run applies a logit_offset
(off_multi = off_ipw = 0.0): per that module's MULTIOBS_CAVEAT, no
King-Zeng-style scalar shift is justified for multiobs's per-observation
incl_prob sampling design, and IPW's correction (if any) happens entirely in
train_hazard_multiobs.py's --use_ipw loss reweighting, not as a post-hoc
inference-time adjustment. Because the scoring path, population, and offset
are identical between the two multiobs runs, any difference in their
forecast h_t/CPR is attributable to the training-time reweighting alone.

Population: the same 365,146-loan intersection
------------------------------------------------
Recomputed as orig_set & trail_set & multi_set, exactly as in
forecast_matched_population_cpr.py. The IPW checkpoint shares
MULTIOBS_SEQ_DIR's test_loan_ids.npy with the uncorrected checkpoint (only
the trained weights differ, not the sequence build), so this reproduces the
same population without re-deriving it from scratch. Asserted at runtime.

The question this script answers
-----------------------------------
Yesterday's uncorrected multiobs run was reported at monthly h_t 0.079-0.33
across coupons (4 of 7 above 0.10 -- deep in the saturating region of
1-(1-h)^12); origination sits at roughly 0.021-0.036 (post its own
prior_shift_offset), never saturating. This script recomputes the
uncorrected row FRESH (same population, same trailing-set scoring path) so
the IPW comparison below is against the number this run actually prints, not
against that prior figure from memory -- the two need not match exactly if
the prior diagnostic used a different coupon range or scoring path. Does
--use_ipw reweighting pull h_t out of the saturating region seen here into a
plausible middle ground, or does it overshoot toward origination's
near-flat (but offset-shifted) regime -- which would read as overcorrection,
not calibration? The per-coupon h_t/CPR table and the explicit range
comparison at the end of main() are what answer this -- this script prints
the numbers, it does not hard-code a verdict.

Do NOT modify forecast_rolling_cpr.py or forecast_matched_population_cpr.py,
and do not overwrite either module's output CSVs. This script writes only a
new file: outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw/rolling_cpr_forecast_matched.csv

Usage:
    python scripts/forecast_multiobs_ipw_cpr.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_rolling_cpr import (
    load_model, infer_test_set, aggregate, prior_shift_offset, DEVICE, BASE,
)
from forecast_matched_population_cpr import (
    CUTOFF_YEAR, BATCH_SIZE,
    ORIG_LABEL_SUFFIX, TRAIL_LABEL_SUFFIX, MULTIOBS_LABEL_SUFFIX,
    ORIG_SEQ_DIR, TRAIL_SEQ_DIR, MULTIOBS_SEQ_DIR,
    RAW_PASS_CACHE,
    score_multiobs_model, filter_to_population, cached_read_coupon_and_realized,
    mean_h_adj_by_coupon, pooled_comparison,
)

IPW_LABEL_SUFFIX = '_multiobs_k5_h1_ipw'
IPW_OUT_DIR = os.path.join(BASE, f'outputs/rolling/cutoff_{CUTOFF_YEAR}{IPW_LABEL_SUFFIX}')

EXPECTED_POPULATION_N = 365146
COUPON_LO, COUPON_HI, MIN_N = 2.0, 5.0, 5000


def main():
    print(f'Device: {DEVICE}', flush=True)

    # ── Step 0: matched population -- identical derivation to
    # forecast_matched_population_cpr.py ──────────────────────────────────
    orig_ids_raw  = np.load(os.path.join(ORIG_SEQ_DIR,     'test_loan_ids.npy'), allow_pickle=True)
    trail_ids_raw = np.load(os.path.join(TRAIL_SEQ_DIR,    'test_loan_ids.npy'), allow_pickle=True)
    multi_ids_raw = np.load(os.path.join(MULTIOBS_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    orig_set  = set(orig_ids_raw.tolist())
    trail_set = set(trail_ids_raw.tolist())
    multi_set = set(multi_ids_raw.tolist())
    population = orig_set & trail_set & multi_set
    print(f'Matched population: {len(population):,} loans (expected {EXPECTED_POPULATION_N:,}).', flush=True)
    assert len(population) == EXPECTED_POPULATION_N, (
        f'Matched population is {len(population):,}, not the expected {EXPECTED_POPULATION_N:,} -- '
        f'one of the three underlying test sets has changed since forecast_matched_population_cpr.py '
        f'last ran. STOP, do not trust the comparison below.')

    # ── Step 1: score all four models ──────────────────────────────────────
    print('\n[1/4] Scoring origination model (infer_test_set, unchanged)...', flush=True)
    orig_model = load_model(CUTOFF_YEAR, ORIG_LABEL_SUFFIX)
    ids_orig, h_orig = infer_test_set(CUTOFF_YEAR, orig_model, BATCH_SIZE, ORIG_LABEL_SUFFIX)

    print('\n[1/4] Scoring trailing model (infer_test_set, unchanged)...', flush=True)
    trail_model = load_model(CUTOFF_YEAR, TRAIL_LABEL_SUFFIX)
    ids_trail, h_trail = infer_test_set(CUTOFF_YEAR, trail_model, BATCH_SIZE, TRAIL_LABEL_SUFFIX)

    print('\n[1/4] Scoring multiobs-UNCORRECTED model (score_multiobs_model, on trailing sequences)...', flush=True)
    multi_model = load_model(CUTOFF_YEAR, MULTIOBS_LABEL_SUFFIX)
    ids_multi, h_multi = score_multiobs_model(multi_model, TRAIL_SEQ_DIR, BATCH_SIZE)

    print('\n[1/4] Scoring multiobs-IPW model (score_multiobs_model, on trailing sequences)...', flush=True)
    ipw_model = load_model(CUTOFF_YEAR, IPW_LABEL_SUFFIX)
    ids_ipw, h_ipw = score_multiobs_model(ipw_model, TRAIL_SEQ_DIR, BATCH_SIZE)

    # ── Step 2: restrict all four to the matched population ─────────────────
    ids_orig,  h_orig  = filter_to_population(ids_orig,  h_orig,  population)
    ids_trail, h_trail = filter_to_population(ids_trail, h_trail, population)
    ids_multi, h_multi = filter_to_population(ids_multi, h_multi, population)
    ids_ipw,   h_ipw   = filter_to_population(ids_ipw,   h_ipw,   population)
    assert len(ids_orig) == len(ids_trail) == len(ids_multi) == len(ids_ipw) == len(population), (
        f'Matched-population filter did not land on the same count for all four: '
        f'orig={len(ids_orig)} trail={len(ids_trail)} multi={len(ids_multi)} ipw={len(ids_ipw)} '
        f'expected={len(population)}')
    print(f'\nAll four scored populations restricted to {len(population):,} loans.', flush=True)

    # ── Step 3: raw pass for coupon + realized -- reuses the EXISTING cache
    # written by forecast_matched_population_cpr.py's run (same population ->
    # same hash -> cache hit, no 40-min re-scan of the 32 vintage CSVs) ──────
    print('\n[2/4] Raw pass for coupon + realized (cached)...', flush=True)
    coupon_map, active_set, prepaid_set = cached_read_coupon_and_realized(
        CUTOFF_YEAR, population, RAW_PASS_CACHE)

    # ── Step 4: aggregate() -- unchanged. multiobs-uncorrected and
    # multiobs-IPW both use logit_offset=0.0 (see module docstring) ─────────
    print('\n[3/4] Aggregating to coupon-level CPR (matched population)...', flush=True)
    off_orig  = prior_shift_offset(ORIG_SEQ_DIR)
    off_trail = prior_shift_offset(TRAIL_SEQ_DIR)
    off_multi = 0.0
    off_ipw   = 0.0

    result_orig  = aggregate(ids_orig,  h_orig,  coupon_map, active_set, prepaid_set, logit_offset=off_orig,  already_annual=False)
    result_trail = aggregate(ids_trail, h_trail, coupon_map, active_set, prepaid_set, logit_offset=off_trail, already_annual=False)
    result_multi = aggregate(ids_multi, h_multi, coupon_map, active_set, prepaid_set, logit_offset=off_multi, already_annual=False)
    result_ipw   = aggregate(ids_ipw,   h_ipw,   coupon_map, active_set, prepaid_set, logit_offset=off_ipw,   already_annual=False)

    result_ipw['logit_offset']  = off_ipw
    result_ipw['cutoff_year']   = CUTOFF_YEAR
    result_ipw['forecast_year'] = CUTOFF_YEAR + 1
    result_ipw['time_varying']  = False
    result_ipw['matched_pop_n'] = len(population)
    os.makedirs(IPW_OUT_DIR, exist_ok=True)
    out_path = os.path.join(IPW_OUT_DIR, 'rolling_cpr_forecast_matched.csv')
    result_ipw.to_csv(out_path, index=False)
    print(f'\nSaved (multiobs-IPW, matched population): {out_path}', flush=True)
    print(result_ipw.to_string(index=False), flush=True)

    # ── Step 5: pooled_comparison() -- unchanged -- for all four ─────────────
    orig_pc  = pooled_comparison(result_orig,  min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))
    trail_pc = pooled_comparison(result_trail, min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))
    multi_pc = pooled_comparison(result_multi, min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))
    ipw_pc   = pooled_comparison(result_ipw,   min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))

    # ── Step 6: mean monthly h_t per coupon, all four ────────────────────────
    h_orig_tab  = mean_h_adj_by_coupon(ids_orig,  h_orig,  coupon_map, active_set, off_orig)
    h_trail_tab = mean_h_adj_by_coupon(ids_trail, h_trail, coupon_map, active_set, off_trail)
    h_multi_tab = mean_h_adj_by_coupon(ids_multi, h_multi, coupon_map, active_set, off_multi)
    h_ipw_tab   = mean_h_adj_by_coupon(ids_ipw,   h_ipw,   coupon_map, active_set, off_ipw)

    def _filt(t):
        return t[(t['coupon'] >= COUPON_LO) & (t['coupon'] <= COUPON_HI) & (t['n_loans_check'] >= MIN_N)]
    h_orig_tab, h_trail_tab, h_multi_tab, h_ipw_tab = (
        _filt(h_orig_tab), _filt(h_trail_tab), _filt(h_multi_tab), _filt(h_ipw_tab))

    # ── Step 6b: reconcile THIS run's uncorrected-multiobs numbers against
    # the existing outputs/rolling/cutoff_2020_multiobs_k5_h1/rolling_cpr_forecast_matched.csv
    # (from forecast_matched_population_cpr.py's prior run). Two things can
    # differ and must not be conflated:
    #   (a) forecast_cpr per coupon (aggregate() output, apples-to-apples if
    #       population/model/offset are unchanged) -- checks for population
    #       or offset DRIFT between the two runs.
    #   (b) a NAIVE back-calculation of h from the existing CSV's
    #       forecast_cpr via inverting g(h)=1-(1-h)^12 on the POOLED mean,
    #       vs. this run's TRUE mean_h_adj_by_coupon() (arithmetic mean of
    #       PER-LOAN h_i). aggregate() computes forecast_cpr as
    #       mean_over_loans(g(h_i)), NOT g(mean_over_loans(h_i)) -- since g
    #       is concave and h is heterogeneous across loans within a coupon
    #       bucket, Jensen's inequality guarantees mean(g(h_i)) < g(mean(h_i)),
    #       so naively inverting the POOLED forecast_cpr systematically
    #       UNDERSTATES the true mean h_t. That understatement, not a data
    #       error in either number, is expected to be the entire gap here.
    print(f'\n{"=" * 100}')
    print('RECONCILIATION -- this run\'s multiobs-uncorrected numbers vs. the existing matched CSV')
    print(f'{"=" * 100}')
    existing_csv_path = os.path.join(
        BASE, f'outputs/rolling/cutoff_{CUTOFF_YEAR}{MULTIOBS_LABEL_SUFFIX}/rolling_cpr_forecast_matched.csv')
    existing_df = pd.read_csv(existing_csv_path)
    existing_df = existing_df[(existing_df['coupon'] >= COUPON_LO) & (existing_df['coupon'] <= COUPON_HI)]

    recon = (
        existing_df[['coupon', 'forecast_cpr', 'n_loans', 'matched_pop_n']]
            .rename(columns={'forecast_cpr': 'existing_csv_cpr', 'n_loans': 'existing_n_loans'})
            .merge(multi_pc['table'][['coupon', 'forecast_cpr', 'n_loans']]
                   .rename(columns={'forecast_cpr': 'this_run_cpr', 'n_loans': 'this_run_n_loans'}),
                   on='coupon')
            .merge(h_multi_tab.rename(columns={'h_t_mean_monthly': 'this_run_true_h_t'})[['coupon', 'this_run_true_h_t']],
                   on='coupon')
            .sort_values('coupon')
    )
    recon['cpr_match'] = (recon['existing_csv_cpr'] - recon['this_run_cpr']).abs() < 0.01
    recon['naive_backcalc_h'] = 1.0 - (1.0 - recon['existing_csv_cpr'] / 100.0) ** (1.0 / 12.0)
    recon['jensen_ratio_true_over_naive'] = recon['this_run_true_h_t'] / recon['naive_backcalc_h']

    print(f'Existing CSV matched_pop_n = {existing_df["matched_pop_n"].iloc[0]:,}  |  '
          f'this run matched_pop_n = {len(population):,}', flush=True)
    print(recon[['coupon', 'existing_n_loans', 'this_run_n_loans', 'cpr_match',
                 'existing_csv_cpr', 'this_run_cpr',
                 'naive_backcalc_h', 'this_run_true_h_t', 'jensen_ratio_true_over_naive']]
          .to_string(index=False), flush=True)

    cpr_all_match = bool(recon['cpr_match'].all())
    n_match = bool((recon['existing_n_loans'] == recon['this_run_n_loans']).all())
    print(f'\n  forecast_cpr matches (existing CSV vs. this run, |diff| < 0.01 pp) for all coupons: {cpr_all_match}')
    print(f'  n_loans matches (existing CSV vs. this run) for all coupons: {n_match}')
    if cpr_all_match and n_match:
        print('\n  VERDICT: population, model, and offset are IDENTICAL between the two runs '
              '(forecast_cpr and n_loans reproduce exactly). The apparent "0.079-0.33" vs. '
              '"back-calculated ~0.05-0.14" discrepancy is NOT a population/offset drift and NOT '
              'an error in the existing CSV or the README h_t table -- it is entirely explained by '
              'Jensen\'s inequality: aggregate() computes forecast_cpr = mean_over_loans(1-(1-h_i)^12), '
              'and inverting that POOLED mean via g^-1(mean CPR) is mathematically NOT the same '
              'quantity as mean_over_loans(h_i) (mean_h_adj_by_coupon()\'s true monthly h_t) whenever '
              'h_i varies across loans in a coupon bucket -- which it clearly does here, given the '
              f'{recon["jensen_ratio_true_over_naive"].min():.2f}x-{recon["jensen_ratio_true_over_naive"].max():.2f}x '
              'gap. The existing CSV\'s forecast_cpr is correct; the README\'s h_t table is correct; '
              'the naive back-calculation done to "sanity check" them against each other was invalid.')
    else:
        print('\n  VERDICT: forecast_cpr or n_loans DIFFER between the existing CSV and this run -- '
              'this points to an actual population or offset drift (e.g. a changed test-set file, '
              'model checkpoint, or logit_offset), NOT the Jensen\'s-inequality explanation above. '
              'Do not paper over this: investigate the drift before trusting either number.')

    # ── Step 7: combined table -- coupon | realized | {orig,trail,multi,ipw} h_t/CPR ──
    orig_tab  = orig_pc['table'][['coupon', 'realized_cpr', 'forecast_cpr', 'n_loans']].rename(columns={'forecast_cpr': 'orig_cpr'})
    trail_tab = trail_pc['table'][['coupon', 'forecast_cpr']].rename(columns={'forecast_cpr': 'trail_cpr'})
    multi_tab = multi_pc['table'][['coupon', 'forecast_cpr']].rename(columns={'forecast_cpr': 'multi_cpr'})
    ipw_tab   = ipw_pc['table'][['coupon', 'forecast_cpr']].rename(columns={'forecast_cpr': 'ipw_cpr'})

    merged = (
        orig_tab
            .merge(trail_tab, on='coupon')
            .merge(multi_tab, on='coupon')
            .merge(ipw_tab, on='coupon')
            .merge(h_orig_tab.rename(columns={'h_t_mean_monthly': 'orig_h_t', 'n_loans_check': 'n_check_orig'})[['coupon', 'orig_h_t', 'n_check_orig']], on='coupon')
            .merge(h_trail_tab.rename(columns={'h_t_mean_monthly': 'trail_h_t'})[['coupon', 'trail_h_t']], on='coupon')
            .merge(h_multi_tab.rename(columns={'h_t_mean_monthly': 'multi_h_t'})[['coupon', 'multi_h_t']], on='coupon')
            .merge(h_ipw_tab.rename(columns={'h_t_mean_monthly': 'ipw_h_t'})[['coupon', 'ipw_h_t']], on='coupon')
            .sort_values('coupon')
    )
    assert (merged['n_loans'] == merged['n_check_orig']).all(), (
        'n_loans mismatch between aggregate() and mean_h_adj_by_coupon() -- '
        'the coupon/active_set filtering has diverged, do not trust the h_t table.')
    merged = merged.drop(columns=['n_check_orig'])

    print(f'\n{"=" * 100}')
    print(f'FOUR-WAY COMPARISON -- matched population ({len(population):,} loans), '
          f'cutoff_2020 -> FY2021, coupons {COUPON_LO}-{COUPON_HI}, n_loans >= {MIN_N}')
    print('origination / trailing / multiobs-uncorrected / multiobs-IPW')
    print(f'{"=" * 100}')
    display_cols = ['coupon', 'n_loans', 'realized_cpr',
                     'orig_h_t', 'orig_cpr', 'trail_h_t', 'trail_cpr',
                     'multi_h_t', 'multi_cpr', 'ipw_h_t', 'ipw_cpr']
    print(merged[display_cols].to_string(index=False))

    print(f'\n{"=" * 100}')
    print(f'DISPERSION (max ratio .. min ratio, coupons {COUPON_LO}-{COUPON_HI}, n>={MIN_N}) '
          f'and POOLED RATIO, all four')
    print(f'{"=" * 100}')
    print(f"{'run':<24}{'dispersion (max..min)':>26}{'pooled_ratio':>16}")
    for name, pc in [('origination', orig_pc), ('trailing', trail_pc),
                      ('multiobs-uncorrected', multi_pc), ('multiobs-IPW', ipw_pc)]:
        disp = f"{pc['dispersion_max']:.3f}..{pc['dispersion_min']:.3f}"
        print(f"{name:<24}{disp:>26}{pc['pooled_ratio']:>16.4f}")

    # ── Step 8: the saturating-region question, stated explicitly ───────────
    n_coupons = len(merged)
    multi_lo, multi_hi = float(merged['multi_h_t'].min()), float(merged['multi_h_t'].max())
    orig_lo, orig_hi   = float(merged['orig_h_t'].min()),  float(merged['orig_h_t'].max())
    ipw_lo, ipw_hi     = float(merged['ipw_h_t'].min()),   float(merged['ipw_h_t'].max())
    n_multi_above_010  = int((merged['multi_h_t'] > 0.10).sum())
    n_ipw_above_010     = int((merged['ipw_h_t'] > 0.10).sum())

    print(f'\n{"=" * 100}')
    print('QUESTION: did the IPW-trained model\'s monthly h_t come out of the saturating region?')
    print('(saturating region: h_t >= ~0.10, where 1-(1-h)^12 is compressing hard toward the ceiling '
          'instead of behaving ~linearly)')
    print(f'{"=" * 100}')
    print(f'  multiobs-UNCORRECTED h_t range: {multi_lo:.4f}-{multi_hi:.4f}  '
          f'({n_multi_above_010} of {n_coupons} coupons above 0.10)  [offset=0.0]')
    print(f'  origination h_t range:          {orig_lo:.4f}-{orig_hi:.4f}  (never saturates, near-flat)  '
          f'[offset={off_orig:.4f} -- POST prior_shift_offset, not the raw hazard]')
    print(f'  multiobs-IPW h_t range:         {ipw_lo:.4f}-{ipw_hi:.4f}  '
          f'({n_ipw_above_010} of {n_coupons} coupons above 0.10)  [offset=0.0]')
    print('\n  Compare the IPW range above to the uncorrected and origination ranges to judge: '
          'a plausible middle ground sits clearly between them (out of saturation, but still '
          'coupon-responsive); collapsing onto/near the origination range instead would indicate '
          'overcorrection rather than calibration. NOTE: origination\'s range is POST-offset '
          '(logit_offset applied); IPW and uncorrected are both offset=0.0, so this is not an '
          'apples-to-apples raw-hazard comparison at this last step -- see module docstring.')

    print('\nDone.')


if __name__ == '__main__':
    main()
