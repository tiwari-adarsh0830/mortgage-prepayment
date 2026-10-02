"""old_pass_postmod_effect.py -- population-level measurement of how many
2003 one-step-ahead loan-months the OLD cutoff_2002 scoring population
loses when rebuilt under the CURRENT loader (term_filter=360 + post-mod
drop, added in bb8dece, after the Sep 24-26 old one-step runs were
originally scored). Compares the as-run old ensemble CSV's 374,850 pairs
against a fresh rebuild's eligible+forward_adjacent population, using the
already-rebuilt fresh old-pass cache (outputs/rolling/_dec_window_raw_cache_old_pass_fresh).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from score_multiobs_dec_window import build_combined_pass, BASE
from score_rolling_one_step import build_rolling_eligible, reference_months

SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist')
CACHE_DIR = os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache_old_pass_fresh')
OLD_CSV = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv')


def main():
    old = pd.read_csv(OLD_CSV, usecols=['loan_id', 'ref_month'])
    old_pairs = set(zip(old['loan_id'], old['ref_month']))
    print(f'As-run old population: {len(old_pairs):,} (loan_id, ref_month) pairs', flush=True)

    test_ids = np.load(os.path.join(SEQ_DIR, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    full_df = build_combined_pass(2002, True, 'fixed', test_ids_set, cache_dir=CACHE_DIR)
    elig = build_rolling_eligible(full_df)

    fresh_pairs = set()
    for ref_ym in reference_months(2002):
        month_rows = elig[(elig['yyyymm'] == ref_ym) & elig['forward_adjacent']]
        fresh_pairs.update(zip(month_rows['loan_id'], month_rows['yyyymm']))
    print(f'Fresh old-pass (current loader) eligible population: {len(fresh_pairs):,} pairs', flush=True)

    missing = old_pairs - fresh_pairs
    added = fresh_pairs - old_pairs
    print(f'Pairs in as-run old CSV but NOT in fresh old-pass (removed by current loader): '
          f'{len(missing):,} ({len(missing) / len(old_pairs):.4%} of old)', flush=True)
    print(f'Pairs in fresh old-pass but NOT in as-run old CSV (newly eligible?): '
          f'{len(added):,} ({len(added) / len(old_pairs):.4%} of old)', flush=True)

    # realized_event label drift on shared pairs
    shared = old_pairs & fresh_pairs
    old_lbl = pd.read_csv(OLD_CSV, usecols=['loan_id', 'ref_month', 'realized_event'])
    old_lbl_map = {(r.loan_id, r.ref_month): r.realized_event for r in old_lbl.itertuples()}
    fresh_lbl_map = {}
    for ref_ym in reference_months(2002):
        month_rows = elig[(elig['yyyymm'] == ref_ym) & elig['forward_adjacent']]
        for lid, nz in zip(month_rows['loan_id'], month_rows['next_zbc']):
            fresh_lbl_map[(lid, ref_ym)] = int(nz == 1.0)
    n_diff = sum(1 for p in shared if old_lbl_map.get(p) != fresh_lbl_map.get(p))
    print(f'realized_event label drift on {len(shared):,} shared pairs: {n_diff:,} differ', flush=True)


if __name__ == '__main__':
    main()
