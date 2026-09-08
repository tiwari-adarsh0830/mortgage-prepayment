"""
forecast_multiobs_ipw_epoch_compare.py -- does the multiobs-IPW checkpoint's
tight, monotonic coupon-level calibration hold at the FINAL training epoch, or
was it specific to the epoch-1 state that train_hazard_multiobs.py happened to
save as hazard_best.pt?

Background
----------
The overnight --use_ipw run (job 17099441, see
outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_epoch1_20260907/) never beat
its epoch-1 AUC (0.7127) across 50 epochs, so hazard_best.pt -- the ONLY
checkpoint that existed before this script -- is the epoch-1 weights.
forecast_multiobs_ipw_cpr.py scored that checkpoint and reported (matched
365,146-loan population, coupons 2.0-5.0):
    dispersion (max..min): 1.412..0.734   pooled_ratio: 1.1195
    h_t range: 0.0158-0.0455              0 of 7 coupons above the 0.10
                                           saturating threshold

train_hazard_multiobs.py was patched to also save hazard_final.pt
unconditionally after the training loop (the epoch-50 state, regardless of
whether it beat best_auc), and the IPW run was repeated from scratch (prior
epoch-1 output moved to the _epoch1_20260907 suffix, not overwritten) so both
checkpoints exist side by side from the SAME training run.

This script scores hazard_best.pt and hazard_final.pt from that rerun with
the IDENTICAL pipeline forecast_multiobs_ipw_cpr.py used (score_multiobs_model
on the trailing builder's test sequences, same matched population, same
coupon/realized cache, logit_offset=0.0 for both -- see that module's
docstring for why no offset is applied to multiobs checkpoints), and prints
the same coupon table / dispersion / pooled_ratio / h_t-range diagnostics for
each, so the two checkpoints are compared apples-to-apples and against the
epoch-1 numbers quoted above.

Does NOT modify forecast_rolling_cpr.py, forecast_matched_population_cpr.py,
or forecast_multiobs_ipw_cpr.py, and does not overwrite any existing output
CSV. Writes two new files:
    outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw/rolling_cpr_forecast_matched_best.csv
    outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw/rolling_cpr_forecast_matched_final.csv

Checkpoint-identity check
--------------------------
Both this rerun and last night's run (job 17099441) peaked at epoch 1 with
nearly the same AUC (0.7124 vs. 0.7127). Same-seed training determinism is
NOT assumed -- data loader shuffling, CUDA nondeterminism, etc. can all break
bit-for-bit reproducibility even with a fixed seed. Before comparing this
run's epoch-1 CPR table to numbers attributed to last night's epoch-1
checkpoint, this script tensor-compares this run's hazard_best.pt against
outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_epoch1_20260907/hazard_best.pt
(where last night's checkpoint was moved, not deleted) and reports whether
every parameter tensor is bit-identical, and if not, the max absolute
difference.

Usage:
    python scripts/forecast_multiobs_ipw_epoch_compare.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_rolling_cpr import PrepaymentTransformer, DEVICE, BASE, MAX_SEQ, N_FEATURES
from forecast_matched_population_cpr import (
    CUTOFF_YEAR, BATCH_SIZE,
    ORIG_SEQ_DIR, TRAIL_SEQ_DIR, MULTIOBS_SEQ_DIR,
    RAW_PASS_CACHE,
    score_multiobs_model, filter_to_population, cached_read_coupon_and_realized,
    mean_h_adj_by_coupon, pooled_comparison,
)
from forecast_rolling_cpr import aggregate

IPW_LABEL_SUFFIX = '_multiobs_k5_h1_ipw'
IPW_OUT_DIR = os.path.join(BASE, f'outputs/rolling/cutoff_{CUTOFF_YEAR}{IPW_LABEL_SUFFIX}')

EXPECTED_POPULATION_N = 365146
COUPON_LO, COUPON_HI, MIN_N = 2.0, 5.0, 5000

# Epoch-1 numbers from last night's run (job 17099441), quoted in the module
# docstring above -- reprinted here as a live comparison target, not
# recomputed (that checkpoint's own directory was moved aside, not deleted).
EPOCH1_DISPERSION   = (1.412, 0.734)
EPOCH1_POOLED_RATIO = 1.1195
EPOCH1_H_RANGE      = (0.0158, 0.0455)
EPOCH1_N_SATURATING = 0


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
    out_path = os.path.join(IPW_OUT_DIR, f'rolling_cpr_forecast_matched_{csv_suffix}.csv')
    result.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    print(result.to_string(index=False))

    pc = pooled_comparison(result, min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))
    h_tab = mean_h_adj_by_coupon(ids, h, coupon_map, active_set, logit_offset=0.0)
    h_tab = h_tab[(h_tab['coupon'] >= COUPON_LO) & (h_tab['coupon'] <= COUPON_HI)
                  & (h_tab['n_loans_check'] >= MIN_N)]

    h_lo, h_hi = float(h_tab['h_t_mean_monthly'].min()), float(h_tab['h_t_mean_monthly'].max())
    n_saturating = int((h_tab['h_t_mean_monthly'] > 0.10).sum())
    n_coupons = len(h_tab)

    print(f"\ncoupon  h_t_mean_monthly  n_loans")
    for _, row in h_tab.sort_values('coupon').iterrows():
        print(f"  {row['coupon']:.1f}        {row['h_t_mean_monthly']:.4f}          {int(row['n_loans_check'])}")

    print(f"\n  dispersion (max..min): {pc['dispersion_max']:.3f}..{pc['dispersion_min']:.3f}   "
          f"pooled_ratio: {pc['pooled_ratio']:.4f}")
    print(f"  h_t range: {h_lo:.4f}-{h_hi:.4f}   ({n_saturating} of {n_coupons} coupons above 0.10)")

    return {
        'dispersion': (pc['dispersion_max'], pc['dispersion_min']),
        'pooled_ratio': pc['pooled_ratio'],
        'h_range': (h_lo, h_hi),
        'n_saturating': n_saturating,
        'n_coupons': n_coupons,
        'table': result,
        'h_table': h_tab,
    }


LAST_NIGHT_BEST_PATH = os.path.join(
    BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_epoch1_20260907/hazard_best.pt')


def compare_checkpoints_identical(path_a: str, path_b: str) -> None:
    """Tensor-compare two checkpoints' state_dicts. Does not assume
    same-seed training is bit-reproducible -- reports the finding, does not
    hard-code a verdict."""
    print(f'\n{"=" * 100}')
    print('CHECKPOINT IDENTITY CHECK -- this run\'s epoch-1 (hazard_best.pt) vs. '
          'last night\'s epoch-1 (moved aside, job 17099441)')
    print(f'{"=" * 100}')
    print(f'  A (this run):    {path_a}')
    print(f'  B (last night):  {path_b}')
    if not os.path.exists(path_b):
        print(f'  B does not exist at that path -- cannot compare. STOP, do not assume identity.')
        return
    ckpt_a = torch.load(path_a, map_location='cpu')
    ckpt_b = torch.load(path_b, map_location='cpu')
    print(f'  A: epoch={ckpt_a.get("epoch","?")} auc={ckpt_a.get("auc","?")}')
    print(f'  B: epoch={ckpt_b.get("epoch","?")} auc={ckpt_b.get("auc","?")}')

    sd_a, sd_b = ckpt_a['model_state'], ckpt_b['model_state']
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    if keys_a != keys_b:
        print(f'  Parameter key sets DIFFER: only in A: {keys_a - keys_b} | only in B: {keys_b - keys_a}')
        print('  VERDICT: NOT comparable -- architecture/state_dict mismatch.')
        return

    all_identical = True
    max_diff_overall = 0.0
    for k in sorted(keys_a):
        ta, tb = sd_a[k], sd_b[k]
        if ta.shape != tb.shape:
            print(f'  {k}: SHAPE MISMATCH {tuple(ta.shape)} vs {tuple(tb.shape)}')
            all_identical = False
            continue
        identical = torch.equal(ta, tb)
        if not identical:
            all_identical = False
            diff = (ta.float() - tb.float()).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            print(f'  {k}: NOT identical, max abs diff = {diff:.3e}')

    print(f'\n  All {len(keys_a)} parameter tensors bit-identical: {all_identical}')
    if all_identical:
        print('  VERDICT: this run\'s epoch-1 checkpoint IS numerically identical to last night\'s. '
              'Same-seed training reproduced bit-for-bit; the near-equal AUCs (0.7124 vs 0.7127) '
              'reflect the same underlying weights, not a coincidence.')
    else:
        print(f'  VERDICT: this run\'s epoch-1 checkpoint is NOT bit-identical to last night\'s '
              f'(max abs param diff = {max_diff_overall:.3e}). The near-equal AUCs (0.7124 vs 0.7127) '
              'are two independent draws landing close, not reproducibility -- do not treat last '
              'night\'s epoch-1 CPR numbers (dispersion 1.412..0.734, pooled 1.1195, h_t '
              '0.0158-0.0455) as interchangeable with this run\'s BEST-checkpoint numbers below; '
              'they are correlated but distinct results.')


def main():
    print(f'Device: {DEVICE}', flush=True)

    compare_checkpoints_identical(
        os.path.join(IPW_OUT_DIR, 'hazard_best.pt'), LAST_NIGHT_BEST_PATH)

    orig_ids_raw  = np.load(os.path.join(ORIG_SEQ_DIR,     'test_loan_ids.npy'), allow_pickle=True)
    trail_ids_raw = np.load(os.path.join(TRAIL_SEQ_DIR,    'test_loan_ids.npy'), allow_pickle=True)
    multi_ids_raw = np.load(os.path.join(MULTIOBS_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    population = set(orig_ids_raw.tolist()) & set(trail_ids_raw.tolist()) & set(multi_ids_raw.tolist())
    print(f'Matched population: {len(population):,} loans (expected {EXPECTED_POPULATION_N:,}).', flush=True)
    assert len(population) == EXPECTED_POPULATION_N, (
        f'Matched population is {len(population):,}, not the expected {EXPECTED_POPULATION_N:,}.')

    coupon_map, active_set, prepaid_set = cached_read_coupon_and_realized(
        CUTOFF_YEAR, population, RAW_PASS_CACHE)

    best_path  = os.path.join(IPW_OUT_DIR, 'hazard_best.pt')
    final_path = os.path.join(IPW_OUT_DIR, 'hazard_final.pt')

    best_stats  = score_and_report('BEST checkpoint (epoch-1-style, whichever epoch actually won)',
                                    best_path,  population, coupon_map, active_set, prepaid_set, 'best')
    final_stats = score_and_report('FINAL checkpoint (last training epoch, unconditional save)',
                                    final_path, population, coupon_map, active_set, prepaid_set, 'final')

    print(f'\n{"=" * 100}')
    print('SUMMARY: does the tight, monotonic epoch-1 calibration hold at the final epoch?')
    print(f'{"=" * 100}')
    def _fmt(stats):
        disp = f"{stats['dispersion'][0]:.3f}..{stats['dispersion'][1]:.3f}"
        hrng = f"{stats['h_range'][0]:.4f}-{stats['h_range'][1]:.4f}"
        sat  = f"{stats['n_saturating']}/{stats['n_coupons']}"
        return disp, stats['pooled_ratio'], hrng, sat

    print(f"{'':<10}{'dispersion (max..min)':>26}{'pooled_ratio':>16}{'h_t range':>20}{'n_saturating':>16}")
    epoch1_disp = f"{EPOCH1_DISPERSION[0]:.3f}..{EPOCH1_DISPERSION[1]:.3f}"
    epoch1_hrng = f"{EPOCH1_H_RANGE[0]:.4f}-{EPOCH1_H_RANGE[1]:.4f}"
    epoch1_sat  = f"{EPOCH1_N_SATURATING}/7"
    print(f"{'epoch1*':<10}{epoch1_disp:>26}{EPOCH1_POOLED_RATIO:>16.4f}{epoch1_hrng:>20}{epoch1_sat:>16}")

    best_disp, best_pr, best_hrng, best_sat = _fmt(best_stats)
    print(f"{'best':<10}{best_disp:>26}{best_pr:>16.4f}{best_hrng:>20}{best_sat:>16}")

    final_disp, final_pr, final_hrng, final_sat = _fmt(final_stats)
    print(f"{'final':<10}{final_disp:>26}{final_pr:>16.4f}{final_hrng:>20}{final_sat:>16}")
    print("\n* epoch1 row is from last night's run (job 17099441), reprinted from the module "
          "docstring/log, NOT recomputed in this invocation.")

    print('\nDone.')


if __name__ == '__main__':
    main()
