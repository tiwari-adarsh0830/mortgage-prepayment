"""
upb_blank_check.py -- for both the cutoff_2002 primary population and the
cutoff_2020 control population, reports the share of scored Dec-window rows
where current_actual_upb is blank (NaN) or zero. No imputation -- read-only
diagnostic, runs the SAME build path score_multiobs_dec_window.py uses (so
this also warms its raw-pass cache), but never loads a checkpoint or touches
the GPU.

Usage: python scripts/diag/upb_blank_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_multiobs_dec_window import build_combined_pass, build_dec_window_obs, BASE

CACHE_DIR = os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')


def check(label, cutoff_year, include_pre2013, map_era, seq_dir):
    test_ids = np.load(os.path.join(seq_dir, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    print(f'\n=== {label}: {len(test_ids_set):,} test loans ===', flush=True)

    full_df = build_combined_pass(cutoff_year, include_pre2013, map_era, test_ids_set, CACHE_DIR)
    _obs, dec_rows, diag = build_dec_window_obs(full_df, cutoff_year)
    print(f'  population diagnostics: {diag}', flush=True)

    upb = dec_rows['current_actual_upb']
    n = len(dec_rows)
    n_blank = int(upb.isna().sum())
    n_zero  = int((upb == 0).sum())
    n_bad   = int((upb.isna() | (upb == 0)).sum())
    print(f'  scored Dec-window rows: {n:,}')
    print(f'  current_actual_upb blank (NaN): {n_blank:,} ({100*n_blank/n:.2f}%)')
    print(f'  current_actual_upb == 0:        {n_zero:,} ({100*n_zero/n:.2f}%)')
    print(f'  blank OR zero (excluded from UPB-weighted metrics): {n_bad:,} ({100*n_bad/n:.2f}%)')


if __name__ == '__main__':
    check('cutoff_2002 primary', 2002, True, 'fixed',
          os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist'))
    check('cutoff_2020 control', 2020, False, 'prefix',
          os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP'))
