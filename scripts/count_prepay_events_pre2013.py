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
import argparse
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
# original_loan_term (awk $13 -> usecols 12) and modification_flag
# (awk $42 -> usecols 41, HARDCODED position -- see
# prepare_sequences_multiobs_zbc.py's _COL_MAP comment for why a name
# lookup would drift here; this script has no name list, so the literal
# index is the only form, same as COL_ORIG/COL_ZBC above).
COL_TERM, COL_MOD = 12, 41
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
    """Per-loan vintage, coupon, and zero-balance code for one quarter file.

    30-YEAR FILTER + POST-MOD: a loan's 30-year status is determined by its
    ORIGINATION characteristics, not by any row (advisor's Oct 4 decision).
    This reader has no per-row reporting-period field, so the "earliest
    reporting-period row" used by the other readers is approximated as the
    FIRST row encountered for that loan IN FILE ORDER: quarter files are
    period-ordered (each covers one calendar quarter, rows within it appear
    in reporting-month order), so the first row seen for a loan in a given
    quarter file is its earliest row in that file. A loan whose
    first-encountered original_loan_term != 360 -- including blank/
    unparseable (NaN) -- is dropped from the returned dicts entirely. (This
    replaces the prior any-observed-row rule, which treated a loan whose
    term changes at modification as non-30-year even though it originated
    as one.) A loan with any observed modification_flag=='Y' has its zbc
    entry removed (CONSEQUENCE: build_
    cell_grid_sample_pre2013.py's `zbc.get(lid)` then returns None for it,
    i.e. treated as censored/no event for sampling purposes -- the same
    "termination row dropped -> censored" consequence as the main readers,
    collapsed to a single loan-level flag since this reader only tracks
    loan-level vintage/coupon/zbc, not a full per-row panel). This loan-
    level "ever Y -> censor" rule does NOT depend on modification_flag
    being monotone (no Y->N reversion) -- it does not need chronological
    order at all, since any observed Y anywhere in the file censors the
    loan. That said, monotonicity is NOT a safe assumption: it was found
    FALSE for 2002Q1 (142 loans revert Y->N -- see scripts/diag/
    verify_term_mod_filters.py's decisive decode, 2026-10-01, and
    prepare_sequences_multiobs_zbc.py, whose per-row panel reader reports
    this as a diagnostic rather than asserting it).
    """
    vint, cpn, zbc, orig_term = {}, {}, {}, {}
    ever_y = set()
    # usecols/names MUST be listed in ASCENDING column-index order: pandas
    # selects usecols columns in ascending file-index order regardless of
    # the order they're listed in, then zips the result positionally with
    # `names` -- so an out-of-order usecols list silently scrambles which
    # name gets which column's data (found 2026-10-01: COL_ZBC=43 was
    # listed before COL_TERM=12/COL_MOD=41, so 'orig' actually received
    # original_loan_term, 'zbc' received origination_date, 'term' received
    # modification_flag, and 'mod' received zero_balance_code -- every row's
    # 'orig' value then failed mmyyyy_to_quarter(), silently producing 0
    # loans for every file; never run end-to-end before this was caught).
    # Ascending order here: COL_LOAN=1, COL_RATE=7, COL_TERM=12, COL_ORIG=13,
    # COL_MOD=41, COL_ZBC=43.
    for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=[COL_LOAN, COL_RATE, COL_TERM, COL_ORIG, COL_MOD, COL_ZBC],
            names=['loan_id', 'rate', 'term', 'orig', 'mod', 'zbc'],
            chunksize=CHUNK, low_memory=False):
        chunk['rate'] = pd.to_numeric(chunk['rate'], errors='coerce')
        chunk['zbc'] = pd.to_numeric(chunk['zbc'], errors='coerce')
        chunk['term'] = pd.to_numeric(chunk['term'], errors='coerce')

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

        # Record the origination-row term: the first row encountered for a
        # loan, in file order (see docstring -- files are period-ordered, so
        # this is that loan's earliest row in this file). Never overwritten
        # once set, so a later row's term (e.g. post-mod) cannot change it.
        new_rows = chunk.loc[~chunk['loan_id'].isin(orig_term.keys()),
                              ['loan_id', 'term']].drop_duplicates('loan_id', keep='first')
        for lid, t in zip(new_rows['loan_id'], new_rows['term']):
            orig_term[lid] = t

        y_rows = chunk.loc[chunk['mod'] == 'Y']
        ever_y.update(y_rows['loan_id'].tolist())

    # A loan's origination-row term != 360 -- including blank/unparseable
    # (NaN != 360 is True elementwise) -- disqualifies it. Loan-level: no
    # partial keeping.
    bad_term_loans = {lid for lid, t in orig_term.items() if t != 360}

    n_vint_before = len(vint)
    n_dropped_from_vint = len(bad_term_loans & vint.keys())
    for lid in bad_term_loans:
        vint.pop(lid, None)
        cpn.pop(lid, None)
        zbc.pop(lid, None)

    n_censored = 0
    for lid in ever_y:
        if lid in zbc:
            del zbc[lid]
            n_censored += 1

    print(f'    {os.path.basename(path)}: term filter dropped '
          f'{n_dropped_from_vint:,} of {n_vint_before:,} loans; post-mod censored '
          f'{n_censored:,} zbc entries of {len(ever_y):,} ever-modified loans',
          flush=True)
    return vint, cpn, zbc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=CKPT,
                         help='checkpoint path (resume guard); default matches the '
                              'pre-fix CKPT module constant -- override to point a '
                              'rerun (e.g. with the loan-level term/mod filters) at a '
                              'fresh path so it does not resume from stale state')
    parser.add_argument('--final', default=FINAL,
                         help='output CSV path; default matches the pre-fix FINAL '
                              'module constant')
    args = parser.parse_args()
    ckpt_path, final_path = args.ckpt, args.final

    files = sorted(f for f in os.listdir(DATA) if f.endswith('.csv'))
    print('found %d quarter files' % len(files), flush=True)

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as fh:
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

        with open(ckpt_path, 'wb') as fh:
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
    df.to_csv(final_path, index=False)

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
    print('wrote %s' % final_path)


if __name__ == '__main__':
    main()
