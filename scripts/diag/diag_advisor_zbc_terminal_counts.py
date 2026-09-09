"""
diag_advisor_zbc_terminal_counts.py -- terminal zero_balance_code_actual counts
on the cutoff_2020 matched test population (365,146 loans: intersection of the
origination/trailing/multiobs test sets), requested by the advisor.

Answers: prepare_sequences_multiobs_zbc.py's _prepare_panel() only treats
zbc==1 as is_prepaid=True (lines 466-470); every other terminal code falls
through to term_t=L-1, the same branch an ordinary still-performing loan
takes. This script counts how many matched-population loans actually carry
each terminal code as of the Dec-2020 cutoff, so that branch statement is
backed by real numbers instead of just the code path.

For each loan, "terminal" here means the row with the maximum yyyymm <=
Dec-2020 across all vintages <= cutoff_year -- mirrors read_coupon_and_realized()'s
own vintage-relevance filter (ALL_VINTAGES with year <= cutoff_year).

Usage:
    python scripts/diag/diag_advisor_zbc_terminal_counts.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast_rolling_cpr import DATA_DIR, _ALL_COLS, ALL_VINTAGES, mmyyyy_to_yyyymm

CUTOFF_YEAR = 2020
CUTOFF_YYYYMM = CUTOFF_YEAR * 100 + 12
BASE = '/scratch/at7095/mortgage_prepayment'


def main():
    orig  = set(np.load(os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc/test_loan_ids.npy'), allow_pickle=True).tolist())
    trail = set(np.load(os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_trail/test_loan_ids.npy'), allow_pickle=True).tolist())
    multi = set(np.load(os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1/test_loan_ids.npy'), allow_pickle=True).tolist())
    population = orig & trail & multi
    print(f'Matched population: {len(population):,}', flush=True)

    usecols = dict(sorted({
        _ALL_COLS.index('loan_id') + 1: 'loan_id',
        _ALL_COLS.index('monthly_reporting_period') + 1: 'monthly_reporting_period',
        43: 'zero_balance_code_actual',
    }.items()))

    relevant = [v for v in ALL_VINTAGES if int(v[:4]) <= CUTOFF_YEAR]
    print(f'{len(relevant)} relevant vintages: {relevant}', flush=True)

    # per-loan running max yyyymm seen <= cutoff, and its zbc code
    best_yyyymm = {}
    best_zbc = {}

    for vintage in relevant:
        path = os.path.join(DATA_DIR, f'{vintage}.csv')
        if not os.path.exists(path):
            print(f'  MISSING {vintage}', flush=True)
            continue
        n_rows_seen = 0
        for chunk in pd.read_csv(path, sep='|', header=None, usecols=list(usecols.keys()),
                                  low_memory=False, chunksize=1_000_000):
            chunk.columns = list(usecols.values())
            chunk = chunk[chunk['loan_id'].isin(population)]
            if chunk.empty:
                del chunk
                continue
            chunk['monthly_reporting_period'] = pd.to_numeric(chunk['monthly_reporting_period'], errors='coerce')
            chunk = chunk[chunk['monthly_reporting_period'].notna()]
            chunk['yyyymm'] = chunk['monthly_reporting_period'].astype(np.int64).map(mmyyyy_to_yyyymm)
            chunk = chunk[chunk['yyyymm'] <= CUTOFF_YYYYMM]
            if chunk.empty:
                del chunk
                continue
            chunk['zero_balance_code_actual'] = pd.to_numeric(chunk['zero_balance_code_actual'], errors='coerce')
            n_rows_seen += len(chunk)
            for lid, ym, zbc in zip(chunk['loan_id'].values, chunk['yyyymm'].values, chunk['zero_balance_code_actual'].values):
                prev = best_yyyymm.get(lid, -1)
                if ym >= prev:
                    best_yyyymm[lid] = ym
                    best_zbc[lid] = zbc
            del chunk
        print(f'  {vintage}: cumulative matched loans so far={len(best_yyyymm):,} (rows this vintage={n_rows_seen:,})', flush=True)

    print(f'\nTotal loans with any row <= cutoff: {len(best_yyyymm):,} (population={len(population):,})', flush=True)

    zbc_series = pd.Series(best_zbc)
    counts = zbc_series.value_counts(dropna=False).sort_index()
    print('\nTerminal (last-observed-<=cutoff) zero_balance_code_actual counts, matched population:')
    print(counts)

    missing = population - set(best_yyyymm.keys())
    print(f'\nLoans in population with ZERO rows <= cutoff (should be 0): {len(missing):,}')

    out_dir = os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'terminal_zbc_matched_population.csv')
    out = pd.DataFrame({'loan_id': list(best_zbc.keys()), 'zbc': list(best_zbc.values())})
    out.to_csv(out_path, index=False)
    print(f'Saved per-loan terminal zbc to {out_path}')


if __name__ == '__main__':
    main()
