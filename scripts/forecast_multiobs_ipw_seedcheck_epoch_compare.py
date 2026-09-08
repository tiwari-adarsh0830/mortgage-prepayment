"""
forecast_multiobs_ipw_seedcheck_epoch_compare.py -- epoch1(best)-vs-epoch50
(final) coupon-level CPR comparison on the SEED-CONTROLLED IPW run
(outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a/), i.e. step 3
of the seed-determinism investigation.

Background
----------
forecast_multiobs_ipw_epoch_compare.py already asked "does the tight,
epoch-1 calibration hold at the final epoch?" but on a run where same-seed
training was NOT yet known to be reproducible -- any best-vs-final
difference it found was confounded with ordinary run-to-run variance.
scripts/diag/compare_seedcheck_checkpoints.py then trained two runs
(_seedcheck_a / _seedcheck_b, jobs 17140541/17140542) with
torch.manual_seed(42) + cudnn.deterministic=True +
use_deterministic_algorithms(True), and found hazard_best.pt (epoch 4, AUC
0.7181) and hazard_final.pt (epoch 50, AUC 0.7090) BIT-IDENTICAL between A
and B. Either run is therefore a canonical stand-in for "the" seeded IPW
run; this script scores _seedcheck_a's two checkpoints -- there is nothing
left to gain from re-scoring B as well, since it would print the same
numbers.

This mirrors forecast_multiobs_ipw_epoch_compare.py's scoring path exactly
(score_multiobs_model on the trailing builder's test sequences, the same
365,146-loan matched population, logit_offset=0.0 for both checkpoints --
see forecast_matched_population_cpr.py's MULTIOBS_CAVEAT for why no
King-Zeng offset applies to multiobs), but points at _seedcheck_a instead
of the original non-seed-controlled epoch1 rerun, and does NOT repeat the
cross-run checkpoint-identity check -- that question (A vs B) was already
answered by compare_seedcheck_checkpoints.py, not this script's job.

Does NOT modify forecast_rolling_cpr.py, forecast_matched_population_cpr.py,
forecast_multiobs_ipw_cpr.py, or forecast_multiobs_ipw_epoch_compare.py, and
does not overwrite any existing output CSV. Writes two new files:
    outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a/rolling_cpr_forecast_matched_best.csv
    outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a/rolling_cpr_forecast_matched_final.csv

Usage:
    python scripts/forecast_multiobs_ipw_seedcheck_epoch_compare.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_rolling_cpr import PrepaymentTransformer, DEVICE, BASE, MAX_SEQ, N_FEATURES, aggregate
from forecast_matched_population_cpr import (
    CUTOFF_YEAR, BATCH_SIZE,
    ORIG_SEQ_DIR, TRAIL_SEQ_DIR, MULTIOBS_SEQ_DIR,
    RAW_PASS_CACHE,
    score_multiobs_model, filter_to_population, cached_read_coupon_and_realized,
    mean_h_adj_by_coupon, pooled_comparison,
)

SEEDCHECK_DIR = os.path.join(BASE, f'outputs/rolling/cutoff_{CUTOFF_YEAR}_multiobs_k5_h1_ipw_seedcheck_a')

EXPECTED_POPULATION_N = 365146
COUPON_LO, COUPON_HI, MIN_N = 2.0, 5.0, 5000


def load_model_from_checkpoint(path: str) -> PrepaymentTransformer:
    """Same construction as forecast_rolling_cpr.load_model(), but takes an
    explicit checkpoint path instead of deriving hazard_best.pt from
    (cutoff_year, label_suffix) -- load_model() has no way to ask for
    hazard_final.pt, and that function is off-limits to modify."""
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt.get('config', {})
    m = PrepaymentTransformer(
        input_dim=cfg.get('input_dim', N_FEATURES), d_model=cfg.get('d_model', 64),
        n_heads=cfg.get('n_heads', 4), n_layers=cfg.get('n_layers', 2),
        dim_ff=cfg.get('dim_ff', 256), dropout=cfg.get('dropout', 0.1),
        max_seq=cfg.get('max_seq', MAX_SEQ),
    ).to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    print(f'Loaded {os.path.basename(path)}: epoch={ckpt.get("epoch","?")} '
          f'AUC={ckpt.get("auc","?"):.4f} device={DEVICE}', flush=True)
    return m


def score_and_report(label: str, ckpt_path: str, population: set,
                      coupon_map, active_set, prepaid_set, csv_suffix: str):
    print(f'\n{"=" * 100}')
    print(f'{label}  ({ckpt_path})')
    print(f'{"=" * 100}')
    model = load_model_from_checkpoint(ckpt_path)
    ids, h = score_multiobs_model(model, TRAIL_SEQ_DIR, BATCH_SIZE)
    ids, h = filter_to_population(ids, h, population)
    assert len(ids) == len(population), (
        f'{label}: filtered population is {len(ids):,}, expected {len(population):,}')

    result = aggregate(ids, h, coupon_map, active_set, prepaid_set,
                        logit_offset=0.0, already_annual=False)
    result['logit_offset']  = 0.0
    result['cutoff_year']   = CUTOFF_YEAR
    result['forecast_year'] = CUTOFF_YEAR + 1
    result['time_varying']  = False
    result['matched_pop_n'] = len(population)
    out_path = os.path.join(SEEDCHECK_DIR, f'rolling_cpr_forecast_matched_{csv_suffix}.csv')
    result.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    pc = pooled_comparison(result, min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))
    h_tab = mean_h_adj_by_coupon(ids, h, coupon_map, active_set, logit_offset=0.0)
    h_tab = h_tab[(h_tab['coupon'] >= COUPON_LO) & (h_tab['coupon'] <= COUPON_HI)
                  & (h_tab['n_loans_check'] >= MIN_N)]

    full = pc['table'].merge(
        h_tab.rename(columns={'h_t_mean_monthly': 'h_t'})[['coupon', 'h_t']], on='coupon')
    full = full[['coupon', 'n_loans', 'realized_cpr', 'h_t', 'forecast_cpr', 'ratio']].sort_values('coupon')

    print(f"\n{'coupon':>7}{'n_loans':>10}{'realized':>10}{'h_t':>9}{'forecast_cpr':>14}{'ratio':>9}")
    for _, row in full.iterrows():
        print(f"{row['coupon']:>7.1f}{int(row['n_loans']):>10,}{row['realized_cpr']:>10.3f}"
              f"{row['h_t']:>9.4f}{row['forecast_cpr']:>14.3f}{row['ratio']:>9.4f}")

    h_lo, h_hi = float(h_tab['h_t_mean_monthly'].min()), float(h_tab['h_t_mean_monthly'].max())
    n_saturating = int((h_tab['h_t_mean_monthly'] > 0.10).sum())
    n_coupons = len(h_tab)

    print(f"\n  dispersion (max..min): {pc['dispersion_max']:.3f}..{pc['dispersion_min']:.3f}   "
          f"pooled_ratio: {pc['pooled_ratio']:.4f}")
    print(f"  h_t range: {h_lo:.4f}-{h_hi:.4f}   ({n_saturating} of {n_coupons} coupons above 0.10)")

    return {
        'dispersion': (pc['dispersion_max'], pc['dispersion_min']),
        'pooled_ratio': pc['pooled_ratio'],
        'h_range': (h_lo, h_hi),
        'n_saturating': n_saturating,
        'n_coupons': n_coupons,
        'table': full,
    }


def main():
    print(f'Device: {DEVICE}', flush=True)
    print(f'Scoring checkpoints from: {SEEDCHECK_DIR}', flush=True)
    print('(cross-run A-vs-B checkpoint identity already verified bit-identical '
          'by scripts/diag/compare_seedcheck_checkpoints.py -- not repeated here.)', flush=True)

    orig_ids_raw  = np.load(os.path.join(ORIG_SEQ_DIR,     'test_loan_ids.npy'), allow_pickle=True)
    trail_ids_raw = np.load(os.path.join(TRAIL_SEQ_DIR,    'test_loan_ids.npy'), allow_pickle=True)
    multi_ids_raw = np.load(os.path.join(MULTIOBS_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    population = set(orig_ids_raw.tolist()) & set(trail_ids_raw.tolist()) & set(multi_ids_raw.tolist())
    print(f'Matched population: {len(population):,} loans (expected {EXPECTED_POPULATION_N:,}).', flush=True)
    assert len(population) == EXPECTED_POPULATION_N, (
        f'Matched population is {len(population):,}, not the expected {EXPECTED_POPULATION_N:,}.')

    coupon_map, active_set, prepaid_set = cached_read_coupon_and_realized(
        CUTOFF_YEAR, population, RAW_PASS_CACHE)

    best_path  = os.path.join(SEEDCHECK_DIR, 'hazard_best.pt')
    final_path = os.path.join(SEEDCHECK_DIR, 'hazard_final.pt')

    best_stats  = score_and_report('BEST checkpoint (epoch 4, AUC 0.7181)',
                                    best_path,  population, coupon_map, active_set, prepaid_set, 'best')
    final_stats = score_and_report('FINAL checkpoint (epoch 50, AUC 0.7090)',
                                    final_path, population, coupon_map, active_set, prepaid_set, 'final')

    print(f'\n{"=" * 100}')
    print('SUMMARY: does the calibration hold at full (epoch-50) training, on the '
          'seed-controlled run where run-to-run variance is no longer a confound?')
    print(f'{"=" * 100}')
    def _fmt(stats):
        disp = f"{stats['dispersion'][0]:.3f}..{stats['dispersion'][1]:.3f}"
        hrng = f"{stats['h_range'][0]:.4f}-{stats['h_range'][1]:.4f}"
        sat  = f"{stats['n_saturating']}/{stats['n_coupons']}"
        return disp, stats['pooled_ratio'], hrng, sat

    print(f"{'':<10}{'dispersion (max..min)':>26}{'pooled_ratio':>16}{'h_t range':>20}{'n_saturating':>16}")
    best_disp, best_pr, best_hrng, best_sat = _fmt(best_stats)
    print(f"{'best':<10}{best_disp:>26}{best_pr:>16.4f}{best_hrng:>20}{best_sat:>16}")
    final_disp, final_pr, final_hrng, final_sat = _fmt(final_stats)
    print(f"{'final':<10}{final_disp:>26}{final_pr:>16.4f}{final_hrng:>20}{final_sat:>16}")

    print('\nDone.')


if __name__ == '__main__':
    main()
