"""build_zip3_map_2002.py -- builds a loan_id -> zip3 lookup for the
cutoff_2002 test population, for the house-price/ZHVI-growth test.

zip3 is read from raw (already in _USECOLS, needed transiently for the
current_ltv ZHVI join) but dropped before load_vintage_filtered's returned
frame unless requested via extra_keep_cols. This script asks for it via
build_combined_pass's new extra_keep_cols parameter, using a SEPARATE
cache_dir from the production _dec_window_raw_cache (own schema tag too --
see build_combined_pass's docstring) so nothing about the existing,
already-validated production cache is touched or re-scanned.

Deliberately does NOT re-score any checkpoint. h (predicted probability)
for every loan-month already comes from the existing, checkpoint-verified
rolling_all_months.csv outputs -- this script only supplies the (loan_id ->
zip3) join key, sidestepping any need to trust that a fresh raw scan
reproduces the same features/predictions (zip3 is not itself a FEATURE_COLS
entry, so it cannot affect any prediction; the join is purely additive).

Output: outputs/rolling/loan_zip3_map_cutoff_2002.csv (loan_id, zip3)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_multiobs_dec_window import build_combined_pass, BASE

CUTOFF_YEAR = 2002
INCLUDE_PRE2013 = True
MAP_ERA = 'fixed'
SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist')
CACHE_DIR = os.path.join(BASE, 'outputs/rolling/_zip3_raw_cache')   # separate from production cache
OUT_PATH = os.path.join(BASE, 'outputs/rolling/loan_zip3_map_cutoff_2002.csv')


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    test_ids = np.load(os.path.join(SEQ_DIR, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    print(f'Test split loans: {len(test_ids_set):,}', flush=True)

    full_df = build_combined_pass(CUTOFF_YEAR, INCLUDE_PRE2013, MAP_ERA, test_ids_set, CACHE_DIR,
                                   extra_keep_cols=['current_actual_upb', 'zip3'])
    print(f'Combined pass rows: {len(full_df):,}  unique loans: {full_df["loan_id"].nunique():,}',
          flush=True)
    assert 'zip3' in full_df.columns, 'zip3 not retained -- extra_keep_cols wiring broken.'

    n_null_zip3 = int(full_df['zip3'].isna().sum())
    print(f'Rows with null zip3: {n_null_zip3:,}/{len(full_df):,}', flush=True)

    zip3_map = (full_df.dropna(subset=['zip3'])
                        .groupby('loan_id')['zip3'].first()
                        .reset_index())
    zip3_map['zip3'] = zip3_map['zip3'].astype(int)
    print(f'loan_id -> zip3 map: {len(zip3_map):,} loans, '
          f'{zip3_map["zip3"].nunique():,} distinct zip3s', flush=True)

    # Sanity: a loan's zip3 should not change across its own rows (it's a
    # property attribute, not time-varying) -- confirm before trusting "first".
    per_loan_nunique = full_df.dropna(subset=['zip3']).groupby('loan_id')['zip3'].nunique()
    n_multi = int((per_loan_nunique > 1).sum())
    print(f'Loans with >1 distinct zip3 across their own rows: {n_multi:,} '
          f'(expected 0 -- zip3 is a static property attribute)', flush=True)

    zip3_map.to_csv(OUT_PATH, index=False)
    print(f'Saved: {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
