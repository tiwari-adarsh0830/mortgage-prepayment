"""
build_cell_grid_sample_pre2013.py -- draws the historical (pre-2013) loan
population for the 729-cell (vintage_quarter x coupon) grid described in
count_prepay_events_pre2013.py's docstring: target roughly 1,000-5,000
zero_balance_code==01 ("prepay") events per cell, no cell merged, dropped,
or floored.

THIS SCRIPT ANSWERS "WHICH LOANS", NOT "WHICH ROWS OF EACH LOAN'S PANEL".
It selects whole loan_ids into or out of the historical population.
prepare_sequences_multiobs_zbc.py's vintage list / data path is a SEPARATE
follow-up change (extend it to read data_pre2013_raw and this script's
output loan_id set) -- not touched here.

DESIGN
------
TARGET_CAP = 5000, the upper end of the spec's 1,000-5,000 event band.

  - n_events <= TARGET_CAP  ->  keep every loan in the cell. No threshold
    logic needed at selection time for this branch -- see below.
  - n_events >  TARGET_CAP  ->  keep a hash-ranked subset of loans sized so
    the EXPECTED surviving event count lands at TARGET_CAP:
        budget = ceil(TARGET_CAP / n_events * n_loans)
    Selecting loans (not events) uniformly at random preserves the cell's
    observed event RATE in expectation, because the hash used for ranking
    is a function of loan_id only -- independent of zero_balance_code. So
    downsampling to a `budget`-loan subset gives ~TARGET_CAP surviving
    events, not exactly TARGET_CAP (this is a census, not a fresh binomial
    draw against the true rate), but close, and exact-egg-count matching
    was never the ask -- the events count anchors the loan-count budget.

  Cells already inside or below the 1,000-5,000 band (which is most of
  them -- see count_prepay_events_pre2013.py's output: 72 cells sit inside
  [1000, 5000], 400 sit below 1000) are left completely alone. There is no
  floor: the 346 thinnest cells (events < ~300) hold only 15,583 loans
  total across all of them, so keeping every loan in every thin cell is
  nearly free relative to the ~29M-loan corpus.

  budget is computed once from n_loans/n_events already in
  outputs/prepay_event_counts_pre2013.csv (the full 52-quarter census), so
  it is correct regardless of which subset of quarter files this script is
  pointed at for a given run (e.g. the 2-file correctness test below).

SELECTION RULE -- one uniform rule for both branches, no branching at
selection time:
    rank = groupby(cell)['hash'].rank(method='first')   # ascending, 1-based
    selected = rank <= budget
When budget >= n_loans (every keep-everything cell), every row's rank is
<= n_loans <= budget, so every row is selected automatically. The only
place cell size actually matters is when computing budget.

REUSES, DOES NOT REIMPLEMENT
-----------------------------
  - scan_file / mmyyyy_to_quarter / COL_* / PREPAID_CODE from
    count_prepay_events_pre2013.py: this is what guarantees the cell
    assignment here (vintage_quarter from origination date, coupon =
    round(rate*2)/2 on the note rate) is IDENTICAL to what already
    produced outputs/prepay_event_counts_pre2013.csv, not just similar.
  - _loan_base_hash from prepare_sequences_multiobs_zbc.py: the same
    blake2b-digest-of-str(loan_id) deterministic hash the modern pipeline
    uses for its own bottom-k / fixed_fraction loan-month selection.
    Deliberately NOT Python's salted str hash(), NOT rng.choice(seed=...)
    -- reproducible across processes/machines and consistent with the
    convention already established in this repo.

CHECKPOINTING -- per-file resume guard, same discipline as
count_prepay_events_pre2013.py (a prior long scan there was lost to a
SLURM timeout holding state in memory). Not exercised by the 2-file test;
matters for the 52-file full run.

CROSS-FILE LOAN_ID CONVENTION -- same as count_prepay_events_pre2013.py:
no dedup check across quarter files. Each file's loans are attributed to
their cells independently; the "1999 vintage split across several early
acquisition files" case documented there applies here unchanged.

Usage:
    # full run (NOT executed yet -- pending the 2-quarter test below)
    python scripts/build_cell_grid_sample_pre2013.py

    # restricted run, e.g. the 2-quarter correctness test
    python scripts/build_cell_grid_sample_pre2013.py \
        --files 2002Q3.csv,2011Q2.csv --out_prefix outputs/pre2013_cell_sample_TEST
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from count_prepay_events_pre2013 import scan_file, PREPAID_CODE  # noqa: E402
from prepare_sequences_multiobs_zbc import _loan_base_hash        # noqa: E402

BASE = '/scratch/at7095/mortgage_prepayment'
DATA = os.path.join(BASE, 'data_pre2013_raw')
OUT = os.path.join(BASE, 'outputs')
COUNTS_CSV = os.path.join(OUT, 'prepay_event_counts_pre2013.csv')

TARGET_CAP = 5000  # events; cells at/under this keep every loan


def compute_budgets(counts_path=COUNTS_CSV, cap=TARGET_CAP):
    """cell=(vintage_quarter, coupon) -> loan budget, from the full census."""
    counts = pd.read_csv(counts_path)
    budget = {}
    for row in counts.itertuples():
        cell = (row.vintage_quarter, float(row.coupon))
        n_loans, n_events = int(row.loans), int(row.prepay_events)
        if n_events <= cap:
            budget[cell] = n_loans
        else:
            budget[cell] = int(np.ceil(cap / n_events * n_loans))
    return budget


def scan_files(files, data_dir=DATA, ckpt_path=None):
    """Loan-level (vintage_quarter, coupon, zbc) for the given quarter
    files, checkpointed per file exactly like
    count_prepay_events_pre2013.py's main loop. Returns a list of
    (loan_id, vintage_quarter, coupon, zbc) tuples -- one row per loan per
    file, undeduplicated across files (see module docstring)."""
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as fh:
            start_idx, records = pickle.load(fh)
        print('RESUME from file %d/%d' % (start_idx, len(files)), flush=True)
    else:
        start_idx, records = 0, []

    for fi in range(start_idx, len(files)):
        f = files[fi]
        vint, cpn, zbc = scan_file(os.path.join(data_dir, f))
        for lid, q in vint.items():
            records.append((lid, q, cpn[lid], zbc.get(lid)))
        print('[%2d/%2d] %-12s loans=%-9d' % (fi + 1, len(files), f, len(vint)), flush=True)
        if ckpt_path:
            with open(ckpt_path, 'wb') as fh:
                pickle.dump((fi + 1, records), fh)

    return records


def select_loans(records, budget):
    """records: list of (loan_id, vintage_quarter, coupon, zbc).
    Returns a DataFrame with one row per input loan, hash/rank/budget/
    selected columns attached."""
    df = pd.DataFrame(records, columns=['loan_id', 'vintage_quarter', 'coupon', 'zbc'])
    df['coupon'] = df['coupon'].astype(float)

    base_hash_map = {lid: _loan_base_hash(lid) for lid in df['loan_id'].unique()}
    df['hash'] = df['loan_id'].map(base_hash_map)

    cell = list(zip(df['vintage_quarter'], df['coupon']))
    df['cell_budget'] = [budget.get(c, 0) for c in cell]
    df['cell_n_loans'] = df.groupby(['vintage_quarter', 'coupon'])['loan_id'].transform('size')
    df['rank_in_cell'] = df.groupby(['vintage_quarter', 'coupon'])['hash'].rank(method='first').astype(int)
    df['selected'] = df['rank_in_cell'] <= df['cell_budget']
    return df


def summarize(df):
    g = df.groupby(['vintage_quarter', 'coupon'])
    summary = g.apply(lambda d: pd.Series({
        'n_loans': len(d),
        'cell_budget': d['cell_budget'].iloc[0],
        'n_selected': int(d['selected'].sum()),
        'n_events_selected': int((d.loc[d['selected'], 'zbc'] == PREPAID_CODE).sum()),
    }), include_groups=False).reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', default=None,
                         help='comma-separated quarter filenames under data_pre2013_raw/ '
                              '(default: all files found there)')
    parser.add_argument('--out_prefix', default=os.path.join(OUT, 'pre2013_cell_sample'))
    parser.add_argument('--cap', type=int, default=TARGET_CAP)
    args = parser.parse_args()

    if args.files:
        files = args.files.split(',')
    else:
        files = sorted(f for f in os.listdir(DATA) if f.endswith('.csv'))
    print('files: %d' % len(files), flush=True)

    budget = compute_budgets(cap=args.cap)
    ckpt_path = args.out_prefix + '_scan_ckpt.pkl'
    records = scan_files(files, ckpt_path=ckpt_path)
    print('records: %d' % len(records), flush=True)

    df = select_loans(records, budget)
    summary = summarize(df)

    sel_path = args.out_prefix + '_loans.csv'
    sum_path = args.out_prefix + '_summary.csv'
    df.loc[df['selected'], ['loan_id', 'vintage_quarter', 'coupon', 'zbc', 'hash', 'rank_in_cell',
                             'cell_n_loans', 'cell_budget']].to_csv(sel_path, index=False)
    summary.to_csv(sum_path, index=False)

    print('selected %d of %d loans (%.1f%%)'
          % (df['selected'].sum(), len(df), 100 * df['selected'].mean()), flush=True)
    print('cells touched: %d' % len(summary), flush=True)
    print('cells downsampled (n_loans > cell_budget): %d'
          % (summary['n_loans'] > summary['cell_budget']).sum(), flush=True)
    print('wrote %s' % sel_path, flush=True)
    print('wrote %s' % sum_path, flush=True)


if __name__ == '__main__':
    main()
