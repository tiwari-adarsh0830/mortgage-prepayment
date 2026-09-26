"""
alignment_effect_vs_realized.py -- adds realized CY2021 prepay rate (count-
weighted, same active_set/prepaid_set definition as
forecast_rolling_cpr.read_coupon_and_realized/aggregate: realized=1 iff
zero_balance_code_actual==1 in any forecast-year row, else 0) to
measure_alignment_effect.py's per-coupon full/partial-history tables.

Reuses the cached combined pass (no new raw-file scan) -- same
cutoff_year=2020, map_era='prefix', test_ids_set as the alignment job, so
build_combined_pass hits the existing cache file.

Usage: python scripts/diag/alignment_effect_vs_realized.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_multiobs_dec_window import build_combined_pass, BASE

ALIGNMENT_CSV = os.path.join(BASE, 'outputs/rolling/alignment_effect_cutoff_2020_seed42/alignment_effect_loans.csv')
CONTROL_SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP')
CACHE_DIR = os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')


def main():
    df = pd.read_csv(ALIGNMENT_CSV)
    print(f'Loaded {len(df):,} loans from {ALIGNMENT_CSV}', flush=True)

    test_ids = np.load(os.path.join(CONTROL_SEQ_DIR, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    full_df = build_combined_pass(2020, False, 'prefix', test_ids_set, CACHE_DIR)

    fy_lo, fy_hi = 202101, 202112
    fy_window = full_df[(full_df['yyyymm'] >= fy_lo) & (full_df['yyyymm'] <= fy_hi)]
    prepaid_set = set(fy_window.loc[fy_window['zero_balance_code_actual'] == 1.0, 'loan_id'].unique().tolist())
    df['realized_prepay'] = df['loan_id'].isin(prepaid_set).astype(int)

    print(f'\n{"=" * 100}\nPer-coupon: cpr_left vs cpr_right vs realized_cpr (count-weighted), '
          f'split by history\n{"=" * 100}')
    for label, g in [('full_history (33mo)', df[df['full_history']]),
                      ('partial_history (<33mo)', df[~df['full_history']])]:
        rows = []
        for coupon, gg in g.groupby('coupon'):
            realized_cpr = round(gg['realized_prepay'].mean() * 100, 4)
            cpr_left = round(gg['annual_pp_left'].mean() * 100, 4)
            cpr_right = round(gg['annual_pp_right'].mean() * 100, 4)
            closer = 'left' if abs(cpr_left - realized_cpr) < abs(cpr_right - realized_cpr) else 'right'
            rows.append({'coupon': coupon, 'n_loans': len(gg), 'cpr_left': cpr_left,
                         'cpr_right': cpr_right, 'realized_cpr': realized_cpr,
                         'abs_err_left': round(abs(cpr_left - realized_cpr), 4),
                         'abs_err_right': round(abs(cpr_right - realized_cpr), 4),
                         'closer_to_realized': closer})
        t = pd.DataFrame(rows).sort_values('coupon')
        print(f'\n{label}:')
        print(t.to_string(index=False))


if __name__ == '__main__':
    main()
