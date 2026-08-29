"""
Prepayment-event counts per vintage-quarter x coupon cell.
Input files are the 2000Q1-2012Q4 acquisition quarters, but vintage_quarter
is derived from each loan's origination_date (col 13), so the output spans
1999Q1-2012Q4: the early files carry loans originated shortly before
acquisition (2000Q1 is 5.3M rows originated 1999 vs 3.9M originated 2000;
the 1999 share decays to 374 rows by 2005Q1). 55 of 729 output cells are
pre-2000, covering 159,981 loans. These are real originations but 1999 is
survivor-selected -- only loans acquired 2000Q1 or later appear.

Needed before the historical sample can be drawn: the spec targets a number
of prepayment EVENTS per cell (roughly 1-5k, floor a few hundred, 2-4M loans
total), so the actual event distribution has to be measured first rather
than the oversampling weights guessed.

COLUMN CHOICE -- this was wrong in the first version, documented here so it
is not repeated. Columns below are pandas 0-indexed (= awk field - 1):

  col 1   loan_id
  col 7   original_interest_rate
  col 13  origination_date (MMYYYY)
  col 43  zero_balance_code        <- the label, for THIS era
  col 106 extra_13                 <- what the modern pipeline uses; NOT
                                      usable pre-2013

The first version used extra_13 (col 106) because that is what
prepare_sequences_rolling.py uses. In the pre-2013 files that column is
almost entirely empty: 2,143 nonempty rows in 2000Q1 against 246,148 rows
with a zero-balance effective date. Counting on it gave 326 events for the
whole 2000 vintage year against a true ~241k for 2000Q1 alone.

zero_balance_code (col 43) is verified for this era: in 2000Q1 it is set for
246,148 of 246,862 distinct loans, once per loan, distributed
  01 prepaid/matured  241,392
  09 foreclosure        2,685
  06 repurchase           980
  02 third-party sale     494
  16 / 03 / 15            597
which is the right shape for 2000-vintage 30y loans, all long terminated.

NOTE: code 01 is "prepaid OR matured". For 2000-2012 vintages of 30y loans
nothing has reached scheduled maturity yet, so 01 is effectively prepayment
here. That will not hold forever and should not be copied forward blindly.

Coupon bucketing follows realized_cpr_v6_upb.py: round(rate*2)/2 on the
NOTE rate. That is the loan's own rate, not the TBA pass-through coupon
(which nets servicing and g-fee, roughly 50bp). Kept consistent with the
existing pipeline deliberately -- flag it as a choice when reporting.

Checkpointed per file with a resume guard: a prior long scan was lost to a
SLURM timeout holding state in memory.
"""
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
DATA = os.path.join(BASE, 'data_pre2013_raw')
OUT = os.path.join(BASE, 'outputs')
CKPT = os.path.join(OUT, 'prepay_event_counts_ckpt.pkl')
FINAL = os.path.join(OUT, 'prepay_event_counts_pre2013.csv')

COL_LOAN, COL_RATE, COL_ORIG, COL_ZBC = 1, 7, 13, 43
PREPAID_CODE = 1.0
CHUNK = 1_000_000


def mmyyyy_to_quarter(v):
    """MMYYYY -> 'YYYYQn'. Convert before any compare (pipeline rule)."""
    try:
        s = str(int(v)).zfill(6)
    except (ValueError, TypeError):
        return None
    mm, yyyy = int(s[:2]), int(s[2:])
    if not (1 <= mm <= 12) or not (1980 <= yyyy <= 2030):
        return None
    return '%dQ%d' % (yyyy, (mm - 1) // 3 + 1)


def scan_file(path):
    """Per-loan vintage, coupon, and zero-balance code for one quarter file."""
    vint, cpn, zbc = {}, {}, {}
    for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=[COL_LOAN, COL_RATE, COL_ORIG, COL_ZBC],
            names=['loan_id', 'rate', 'orig', 'zbc'],
            chunksize=CHUNK, low_memory=False):
        chunk['rate'] = pd.to_numeric(chunk['rate'], errors='coerce')
        chunk['zbc'] = pd.to_numeric(chunk['zbc'], errors='coerce')

        first = chunk.dropna(subset=['loan_id', 'rate', 'orig'])
        first = first.drop_duplicates('loan_id')
        for lid, r, o in zip(first['loan_id'], first['rate'], first['orig']):
            if lid in vint:
                continue
            q = mmyyyy_to_quarter(o)
            if q is None:
                continue
            vint[lid] = q
            cpn[lid] = float(np.round(r * 2) / 2.0)

        term = chunk.dropna(subset=['zbc'])
        for lid, z in zip(term['loan_id'], term['zbc']):
            if lid not in zbc:
                zbc[lid] = float(z)
    return vint, cpn, zbc


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith('.csv'))
    print('found %d quarter files' % len(files), flush=True)

    if os.path.exists(CKPT):
        with open(CKPT, 'rb') as fh:
            start_idx, events, loans, terms = pickle.load(fh)
        print('RESUME from file %d/%d' % (start_idx, len(files)), flush=True)
    else:
        start_idx = 0
        events = defaultdict(int)   # (vq, cpn) -> code 01 count
        loans = defaultdict(int)    # (vq, cpn) -> distinct loans
        terms = defaultdict(int)    # code -> count, for a sanity check

    for fi in range(start_idx, len(files)):
        f = files[fi]
        vint, cpn, zbc = scan_file(os.path.join(DATA, f))

        n_prepaid = 0
        for lid, q in vint.items():
            cell = (q, cpn[lid])
            loans[cell] += 1
            code = zbc.get(lid)
            if code is not None:
                terms[code] += 1
                if code == PREPAID_CODE:
                    events[cell] += 1
                    n_prepaid += 1

        with open(CKPT, 'wb') as fh:
            pickle.dump((fi + 1, events, loans, terms), fh)
        print('[%2d/%2d] %-12s loans=%-9d prepaid=%-9d rate=%.3f'
              % (fi + 1, len(files), f, len(vint), n_prepaid,
                 n_prepaid / max(len(vint), 1)), flush=True)

    rows = []
    for cell, nl in sorted(loans.items()):
        q, c = cell
        rows.append({'vintage_quarter': q, 'coupon': c,
                     'loans': nl, 'prepay_events': events.get(cell, 0)})
    df = pd.DataFrame(rows)
    df['event_rate'] = df['prepay_events'] / df['loans']
    df.to_csv(FINAL, index=False)

    print()
    print('zero-balance code distribution (all files):')
    for code in sorted(terms):
        print('  %5.0f : %d' % (code, terms[code]))
    print()
    print('cells: %d   loans: %d   events: %d   overall rate: %.3f'
          % (len(df), df['loans'].sum(), df['prepay_events'].sum(),
             df['prepay_events'].sum() / max(df['loans'].sum(), 1)))
    print()
    df['year'] = df['vintage_quarter'].str[:4]
    print('by vintage year:')
    print(df.groupby('year')[['loans', 'prepay_events']].sum().to_string())
    print()
    print('cells with >=300 events: %d of %d'
          % ((df['prepay_events'] >= 300).sum(), len(df)))
    print('wrote %s' % FINAL)


if __name__ == '__main__':
    main()
