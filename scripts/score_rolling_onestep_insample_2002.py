"""score_rolling_onestep_insample_2002.py -- in-sample one-step-ahead
scoring for the cutoff_2002 model, reference months Jan2001 through
Nov2002 (all strictly INSIDE the training period, unlike the Dec2002-
Nov2003 forecast window scored by score_rolling_one_step.py).

Answers: does the cutoff_2002 model under-respond to incentive even within
its own training period, or only out-of-sample in 2003?

Reuses score_rolling_one_step.py's build_rolling_eligible()/month_result()
(same eligibility, same forward-adjacency filter, same right-aligned
explicit-obs gather -- the "obs=None override" path, not select_observations()
sampling -- same frozen scaler) and score_multiobs_dec_window.py's
build_combined_pass(). Calling build_combined_pass with cutoff_year=2002
hits the SAME cached combined pass already built for the 2003 test (same
cutoff_year/map_era/test_ids population_hash), so this needs no new raw
scan.

DEC-2002 ANCHOR CHECK: this file is new pipeline code reusing month_result()
over a month range it wasn't originally written for, so month 200212 is
scored here too (then dropped from the analysis output) purely to assert
per-loan h equality against the existing frozen run's
outputs/rolling/rolling_onestep_cutoff_2002_seed{seed}/rolling_month_200212.csv
-- same full_df, same elig, same function, so it must match exactly
(< 1e-6). This is the "test the paths against each other" convention
applied to this script, not a re-run of the standing consistency test.

Usage:
    python scripts/score_rolling_onestep_insample_2002.py \\
        --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist \\
        --ckpt_path outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist/hazard_best.pt \\
        --map_era fixed --include_pre2013 --seed_label seed42 \\
        --anchor_scores outputs/rolling/rolling_onestep_cutoff_2002_seed42/rolling_month_200212.csv \\
        --out_dir outputs/rolling/rolling_onestep_insample_cutoff_2002_seed42
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_multiobs_dec_window import build_combined_pass, load_checkpoint, verify_checkpoint, BASE
from score_rolling_one_step import build_rolling_eligible, month_result, rate_table, MIN_N

CUTOFF_YEAR = 2002
ANCHOR_YM = 200212
CONSISTENCY_TOL = 1e-6


def reference_months_insample():
    """Jan2001-Dec2001, Jan2002-Nov2002 (23 months), plus the Dec-2002
    anchor appended last so month_result()'s checkpoint files for the
    analysis range are written first / independent of the anchor."""
    months = [2001 * 100 + m for m in range(1, 13)] + [2002 * 100 + m for m in range(1, 12)]
    return months + [ANCHOR_YM]


def anchor_check(anchor_month_df: pd.DataFrame, anchor_scores_path: str):
    anchor = pd.read_csv(anchor_scores_path)
    merged = anchor_month_df.merge(anchor[['loan_id', 'h']], on='loan_id', how='inner', suffixes=('', '_frozen'))
    n_new, n_frozen, n_common = len(anchor_month_df), len(anchor), len(merged)
    max_abs_diff = float((merged['h'] - merged['h_frozen']).abs().max()) if n_common else float('nan')
    print(f'\nDEC-2002 ANCHOR CHECK (vs. {anchor_scores_path}):', flush=True)
    print(f'  this run population: {n_new:,}  frozen population: {n_frozen:,}  shared: {n_common:,}', flush=True)
    print(f'  max abs diff on shared loans: {max_abs_diff:.3e}', flush=True)
    assert n_common > 0, 'Zero loans in common with the frozen Dec-2002 anchor run -- STOP.'
    assert max_abs_diff < CONSISTENCY_TOL, (
        f'Anchor check FAILED: max abs diff {max_abs_diff:.3e} >= {CONSISTENCY_TOL:.0e} -- '
        f'this script does not reproduce the existing frozen path. STOP, do not trust its output.')
    print(f'  PASSED (< {CONSISTENCY_TOL:.0e}).', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_dir', type=str, required=True)
    ap.add_argument('--ckpt_path', type=str, required=True)
    ap.add_argument('--map_era', choices=['fixed', 'prefix'], required=True)
    ap.add_argument('--include_pre2013', action='store_true')
    ap.add_argument('--seed_label', type=str, required=True)
    ap.add_argument('--anchor_scores', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default=None)
    ap.add_argument('--batch_size', type=int, default=8192)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')

    with open(os.path.join(args.seq_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    test_ids = np.load(os.path.join(args.seq_dir, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())

    full_df = build_combined_pass(CUTOFF_YEAR, args.include_pre2013, args.map_era, test_ids_set, cache_dir)
    elig = build_rolling_eligible(full_df)

    model = load_checkpoint(args.ckpt_path)
    results_json_path = os.path.join(os.path.dirname(args.ckpt_path), 'results.json')
    ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
    verify_checkpoint(model, args.seq_dir, results_json_path, ckpt_dict, args.batch_size)

    ref_months = reference_months_insample()
    monthly_rows = []
    all_months = []
    anchor_month_df = None
    for ref_ym in ref_months:
        m = month_result(full_df, elig, scaler, ref_ym, model, args.batch_size, args.out_dir)
        if ref_ym == ANCHOR_YM:
            anchor_month_df = m
            continue
        all_months.append(m)
        monthly_rows.append({
            'ref_month': ref_ym, 'n_active': len(m),
            'predicted_rate_monthly': m['h'].mean(), 'realized_rate_monthly': m['realized_event'].mean(),
            'n_excluded_by_forward_gap': int(m['n_excluded_by_forward_gap'].iloc[0]),
        })

    anchor_check(anchor_month_df, args.anchor_scores)

    full = pd.concat(all_months, ignore_index=True)
    full.to_csv(os.path.join(args.out_dir, 'rolling_all_months.csv'), index=False)
    full['coupon'] = ((full['note_rate'] - 0.5) * 2).round() / 2

    print(f'\n{"=" * 100}\nPredicted vs realized, by coupon, summed over Jan2001-Nov2002 -- COUNT-weighted\n{"=" * 100}')
    print(rate_table(full, 'coupon', None, MIN_N).to_string(index=False))

    e_tab = pd.DataFrame(monthly_rows)
    e_tab.to_csv(os.path.join(args.out_dir, 'monthly_pooled.csv'), index=False)
    print('\n=== MONTH-BY-MONTH ===')
    print(e_tab.to_string(index=False))
    print('\nDone.')


if __name__ == '__main__':
    main()
