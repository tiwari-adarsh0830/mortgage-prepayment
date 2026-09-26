"""
Term/maturity distribution scan for two quarterly files (2002Q3, 2011Q2).

Column positions follow the verified mapping in prepare_sequences_multiobs_zbc.py
(_ALL_COLS.index(name) + 1, pandas 0-indexed usecols). Only pre-drift columns are
used here (loan_id, monthly_reporting_period, original_loan_term, origination_date,
loan_age, all at or before position 15) plus the hardcoded, independently verified
zero_balance_code_actual at position 43 -- the same constant used throughout the
pipeline, NOT the naive index-derived 42 (see that file's comment for why).

AGE-AT-TERMINATION -- loan_age (raw column 15) is blank on every terminal
(zero-balance) row, verified by hand against three loans across both files
(e.g. loan 100003717196 in 2002Q3: rows 082002..032003 have loan_age -1..6,
row 042003 has zero_balance_code=01 and loan_age blank). It must be derived
from monthly_reporting_period and origination_date instead. The raw loan_age
column itself (on non-terminal rows) was verified to equal
  month_diff(mrp, orig) - 1, clipped at 0
not a bare (mrp - orig) diff -- e.g. same-month mrp/orig gives raw age -1,
confirming both the "-1" term and the "clipped at 0" floor. Age-at-termination
is computed with that identical formula so it is continuous with the raw
age progression on the loan's preceding rows.

For each file, per loan, tracks:
  - original_loan_term (first non-null value seen)
  - origination_date (first non-null value seen)
  - age-at-termination, derived from the terminal row's own monthly_reporting_period
    and the loan's origination_date (zbc is set once, at the terminal record)
  - the zbc value itself (terminal code)
  - max monthly_reporting_period observed anywhere in the file (last reporting period)

"Active" = loans with no non-null zbc anywhere in the file (still open as of the
file's last reporting period). "Terminal" = any non-null zbc. "Code-01" = zbc==1.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

DATA = '/scratch/at7095/mortgage_prepayment/data_pre2013_raw'
FILES = ['2002Q3.csv', '2011Q2.csv']

COL_LOAN, COL_MRP, COL_TERM, COL_ORIG, COL_AGE, COL_ZBC = 1, 2, 12, 13, 15, 43
CHUNK = 1_000_000


def mmyyyy_to_ym(v):
    """MMYYYY (Fannie convention, month NOT zero-padded) -> (year, month)."""
    s = str(int(v))
    if len(s) == 5:
        mm, yyyy = int(s[0]), int(s[1:])
    elif len(s) == 6:
        mm, yyyy = int(s[:2]), int(s[2:])
    else:
        raise ValueError('unexpected MMYYYY length for %r: %r' % (v, s))
    return yyyy, mm


def age_from_dates(mrp, orig):
    """Reproduces this dataset's raw loan_age convention: month_diff - 1, clipped at 0."""
    y1, m1 = mmyyyy_to_ym(mrp)
    y2, m2 = mmyyyy_to_ym(orig)
    diff = (y1 - y2) * 12 + (m1 - m2)
    return max(diff - 1, 0)

TERM_BUCKETS = [(0, 120, '<=120 (10y)'), (121, 180, '121-180 (15y)'),
                (181, 240, '181-240 (20y)'), (241, 300, '241-300 (25y)'),
                (301, 360, '301-360 (30y)'), (361, 10**6, '>360')]


def bucket_term(t):
    for lo, hi, label in TERM_BUCKETS:
        if lo <= t <= hi:
            return label
    return 'unknown'


def scan_file(path):
    term = {}          # loan_id -> original_loan_term
    orig = {}           # loan_id -> origination_date (MMYYYY)
    zbc = {}             # loan_id -> terminal zero_balance_code
    age_at_term = {}      # loan_id -> derived age at the terminal row
    last_mrp = 0

    for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=[COL_LOAN, COL_MRP, COL_TERM, COL_ORIG, COL_AGE, COL_ZBC],
            names=['loan_id', 'mrp', 'term', 'orig', 'age', 'zbc'],
            chunksize=CHUNK, low_memory=False):
        chunk['mrp'] = pd.to_numeric(chunk['mrp'], errors='coerce')
        chunk['term'] = pd.to_numeric(chunk['term'], errors='coerce')
        chunk['orig'] = pd.to_numeric(chunk['orig'], errors='coerce')
        chunk['zbc'] = pd.to_numeric(chunk['zbc'], errors='coerce')
        # 'age' (raw loan_age, col 15) is NOT used for age-at-termination --
        # verified blank on every terminal row. Kept only as an unused column
        # here for clarity; age-at-termination is derived below.

        mx = chunk['mrp'].max(skipna=True)
        if pd.notna(mx):
            last_mrp = max(last_mrp, int(mx))

        first = chunk.dropna(subset=['loan_id']).drop_duplicates('loan_id')
        for lid, t, o in zip(first['loan_id'], first['term'], first['orig']):
            if lid not in term and pd.notna(t):
                term[lid] = float(t)
            if lid not in orig and pd.notna(o):
                orig[lid] = float(o)

        termd = chunk.dropna(subset=['zbc'])
        for lid, z, o, m in zip(termd['loan_id'], termd['zbc'], termd['orig'], termd['mrp']):
            if lid not in zbc:
                zbc[lid] = float(z)
                # prefer the terminal row's own origination_date (static field,
                # present on every row); fall back to the first-seen value
                o_use = o if pd.notna(o) else orig.get(lid)
                if pd.notna(m) and o_use is not None and pd.notna(o_use):
                    age_at_term[lid] = float(age_from_dates(m, o_use))
                else:
                    age_at_term[lid] = None

    return term, zbc, age_at_term, last_mrp


def main():
    for f in FILES:
        path = os.path.join(DATA, f)
        print('=' * 70, flush=True)
        print(f'FILE: {f}', flush=True)
        term, zbc, age_at_term, last_mrp = scan_file(path)

        n_total = len(term)
        n_terminal = len(zbc)
        n_active = n_total - n_terminal
        n_code01 = sum(1 for v in zbc.values() if v == 1.0)

        print(f'last reporting period (monthly_reporting_period, raw MMYYYY): {last_mrp}', flush=True)
        print(f'loans total={n_total} terminal={n_terminal} code01={n_code01} active={n_active}', flush=True)

        # original term distribution (all loans)
        print('\noriginal_loan_term distribution (all loans):', flush=True)
        term_bucket_counts = defaultdict(int)
        for t in term.values():
            term_bucket_counts[bucket_term(t)] += 1
        for lo, hi, label in TERM_BUCKETS:
            print(f'  {label:16s} {term_bucket_counts.get(label, 0)}', flush=True)

        # code-01 split by term bucket
        print('\ncode-01 (prepaid/matured) loans split by original_loan_term bucket:', flush=True)
        code01_bucket_counts = defaultdict(int)
        for lid, z in zbc.items():
            if z == 1.0 and lid in term:
                code01_bucket_counts[bucket_term(term[lid])] += 1
        for lo, hi, label in TERM_BUCKETS:
            print(f'  {label:16s} {code01_bucket_counts.get(label, 0)}', flush=True)

        # age-at-termination distribution for code-01 loans
        ages = np.array([age_at_term[lid] for lid, z in zbc.items()
                          if z == 1.0 and age_at_term.get(lid) is not None])
        print(f'\nage-at-termination (months) distribution for code-01 loans, n={len(ages)}:', flush=True)
        if len(ages):
            qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
            pct = np.percentile(ages, qs)
            for q, v in zip(qs, pct):
                print(f'  p{q:<3d} {v:.1f}', flush=True)
            print(f'  mean {ages.mean():.1f}  min {ages.min():.0f}  max {ages.max():.0f}', flush=True)

        # code-01 loans terminating within ~2 months of their original term
        near_term = 0
        n_code01_with_term = 0
        for lid, z in zbc.items():
            if z != 1.0 or lid not in term or age_at_term.get(lid) is None:
                continue
            n_code01_with_term += 1
            if age_at_term[lid] >= term[lid] - 2:
                near_term += 1
        share = near_term / n_code01_with_term if n_code01_with_term else float('nan')
        print(f'\ncode-01 loans terminating within 2 months of original_loan_term: '
              f'{near_term} of {n_code01_with_term} code-01 loans with known term/age '
              f'({share:.4%})', flush=True)
        print(f'  (as share of ALL code-01 loans in file, n={n_code01}): '
              f'{near_term / n_code01 if n_code01 else float("nan"):.4%}', flush=True)
        print(flush=True)


if __name__ == '__main__':
    main()
