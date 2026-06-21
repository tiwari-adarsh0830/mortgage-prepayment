"""
diag_zbc_column.py — Locate the real zero_balance_code column in raw Fannie files.

Symptom: rolling prep shows EXACTLY 0.00% prepay on 2013-2016 vintages, meaning
the column mapped to zero_balance_code_actual never equals 1.0 in those files.
Hypothesis: pre-2017 files have a different column structure than 2018-2019,
so the extra_13+1=106 offset lands on a date column (never ==1).

This reads a sample of each file with NO column assumptions and reports:
  - total column count
  - for candidate columns near 106, the value distribution
  - which column actually contains zero-balance codes {1,2,3,6,9,15,16}
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = '/scratch/at7095/mortgage_prepayment/data/raw'
ZBC_CODES = {1, 2, 3, 6, 9, 15, 16, 96}  # valid Fannie zero-balance codes

def inspect(vintage, nrows=200_000):
    path = os.path.join(DATA_DIR, f'{vintage}.csv')
    if not os.path.exists(path):
        print(f'{vintage}: FILE NOT FOUND at {path}')
        return
    # Read with NO usecols — get every column as string
    df = pd.read_csv(path, sep='|', header=None, nrows=nrows,
                     dtype=str, low_memory=False)
    ncols = df.shape[1]
    print(f'\n========== {vintage} ==========')
    print(f'  total columns: {ncols}  (leading pipe → col 0 should be empty)')
    print(f'  col 0 sample values: {df[0].dropna().unique()[:3].tolist()}')

    # Scan every column: which ones contain values that look like zero-balance codes?
    print('  Scanning for the zero-balance-code column (values in {1,2,3,6,9,15,16}):')
    candidates = []
    for c in range(ncols):
        vals = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        uniq = set(vals.unique().astype(int).tolist()) if vals.dtype != object else set()
        # zbc column: small integer set, mostly subset of valid codes, includes a 1
        nonzero = vals[vals != 0]
        if len(nonzero) > 0:
            small = set(nonzero.unique().astype(int).tolist())
            if small and small.issubset(ZBC_CODES) and 1 in small:
                n_ones = int((vals == 1).sum())
                candidates.append((c, sorted(small), n_ones))

    if candidates:
        for c, codes, n_ones in candidates:
            print(f'    col {c}: codes={codes}  n(==1)={n_ones:,}  <-- ZBC candidate')
    else:
        print('    NO column found with clean zero-balance codes in this sample')

    # Specifically report what col 106 (current mapping) contains
    for probe in (104, 105, 106, 107, 108):
        if probe < ncols:
            v = pd.to_numeric(df[probe], errors='coerce').dropna()
            samp = v.unique()[:6]
            n1 = int((v == 1).sum()) if len(v) else 0
            print(f'    [probe] col {probe}: n(==1)={n1:,}  sample={samp[:6].tolist()}')

for v in ['2013Q1', '2015Q1', '2018Q1', '2020Q1']:
    inspect(v)
