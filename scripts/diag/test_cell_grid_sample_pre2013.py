"""test_cell_grid_sample_pre2013.py -- correctness check for
build_cell_grid_sample_pre2013.py, run BEFORE the 52-quarter selection
(that is a real cost commitment; this is not).

Uses REAL data (2002Q3.csv, 2011Q2.csv under data_pre2013_raw/ -- the two
quarters already scanned by the term/maturity diagnostic), unlike the
synthetic-panel tests in this directory, because the thing under test is
cell ASSIGNMENT against real note rates and origination dates, which a
hand-built synthetic panel can't exercise honestly.

Checks:
  1. Cell assignment (vintage_quarter, coupon) tallied by
     build_cell_grid_sample_pre2013.scan_files/select_loans for these two
     files reproduces the SAME per-cell (n_loans, n_events) as an
     independent replica of count_prepay_events_pre2013.py's own main-loop
     accumulation logic run over the identical two files. Both call
     scan_file(), but the accumulation step is written twice, independently,
     specifically so a bug in one accumulation loop isn't invisible to
     the other.
  2. Determinism: running the full scan -> hash -> rank -> select pipeline
     twice over the same two files gives an identical selected loan_id set
     (order-independent) both times.

Run:
    cd /scratch/at7095/mortgage_prepayment
    python scripts/diag/test_cell_grid_sample_pre2013.py
"""
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from count_prepay_events_pre2013 import scan_file, PREPAID_CODE
import build_cell_grid_sample_pre2013 as bcs

BASE = '/scratch/at7095/mortgage_prepayment'
DATA = os.path.join(BASE, 'data_pre2013_raw')
FILES = ['2002Q3.csv', '2011Q2.csv']

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail and not cond else ''))
    if not cond:
        FAILURES.append(name)


def independent_replica_tally(files):
    """Reproduces count_prepay_events_pre2013.py's main() accumulation
    loop verbatim (persistent loans/events defaultdicts across files,
    scan_file per file) -- NOT by calling anything in
    build_cell_grid_sample_pre2013, so this is a true second
    implementation of the aggregation step, not a re-run of the first."""
    loans = defaultdict(int)
    events = defaultdict(int)
    for f in files:
        vint, cpn, zbc = scan_file(os.path.join(DATA, f))
        for lid, q in vint.items():
            cell = (q, cpn[lid])
            loans[cell] += 1
            code = zbc.get(lid)
            if code is not None and code == PREPAID_CODE:
                events[cell] += 1
    rows = [{'vintage_quarter': q, 'coupon': c, 'n_loans': n, 'n_events': events.get((q, c), 0)}
             for (q, c), n in loans.items()]
    return pd.DataFrame(rows)


def main():
    print('=== Check 1: cell assignment matches an independent replica ===', flush=True)
    replica = independent_replica_tally(FILES)

    records = bcs.scan_files(FILES, ckpt_path=None)
    df = pd.DataFrame(records, columns=['loan_id', 'vintage_quarter', 'coupon', 'zbc'])
    df['coupon'] = df['coupon'].astype(float)
    sampler_tally = df.groupby(['vintage_quarter', 'coupon']).agg(
        n_loans=('loan_id', 'size'),
        n_events=('zbc', lambda s: (s == PREPAID_CODE).sum()),
    ).reset_index()

    merged = replica.merge(sampler_tally, on=['vintage_quarter', 'coupon'],
                            how='outer', suffixes=('_replica', '_sampler'))
    check('same set of (vintage_quarter, coupon) cells touched',
          merged[['n_loans_replica', 'n_loans_sampler']].isna().sum().sum() == 0,
          detail=f'{merged[merged.isna().any(axis=1)]}')
    loans_match = (merged['n_loans_replica'] == merged['n_loans_sampler']).all()
    events_match = (merged['n_events_replica'] == merged['n_events_sampler']).all()
    check('n_loans per cell matches independent replica, all cells', loans_match,
          detail=f'{(merged["n_loans_replica"] != merged["n_loans_sampler"]).sum()} mismatched cells')
    check('n_events per cell matches independent replica, all cells', events_match,
          detail=f'{(merged["n_events_replica"] != merged["n_events_sampler"]).sum()} mismatched cells')
    check('cell count matches count_prepay_events_pre2013.py output for these two files',
          len(merged) == len(replica) == len(sampler_tally),
          detail=f'replica={len(replica)} sampler={len(sampler_tally)} merged={len(merged)}')

    print(f'\ncells touched by these two files: {len(merged)}', flush=True)
    print(f'total loans: {merged["n_loans_replica"].sum():,}  '
          f'total events: {merged["n_events_replica"].sum():,}', flush=True)

    print('\n=== Check 2: determinism (run selection twice, compare loan_ids) ===', flush=True)
    budget = bcs.compute_budgets()
    sel1 = bcs.select_loans(records, budget)
    sel2 = bcs.select_loans(records, budget)  # same records, fresh hash/rank/select computation

    ids1 = sorted(sel1.loc[sel1['selected'], 'loan_id'].tolist())
    ids2 = sorted(sel2.loc[sel2['selected'], 'loan_id'].tolist())
    check('identical selected loan_id count across two runs', len(ids1) == len(ids2),
          detail=f'{len(ids1)} vs {len(ids2)}')
    check('identical selected loan_id set across two runs', ids1 == ids2)

    # Independent re-scan of the raw files (not reusing `records`) to also
    # confirm the hash itself is stable across a fresh scan_file() call,
    # not just stable given the same in-memory records.
    records3 = bcs.scan_files(FILES, ckpt_path=None)
    sel3 = bcs.select_loans(records3, budget)
    ids3 = sorted(sel3.loc[sel3['selected'], 'loan_id'].tolist())
    check('identical selected loan_id set after a fresh scan_file() re-scan', ids1 == ids3)

    n_downsampled_cells = int((sel1.groupby(['vintage_quarter', 'coupon'])
                                .apply(lambda d: d['cell_n_loans'].iloc[0] > d['cell_budget'].iloc[0],
                                       include_groups=False)).sum())
    print(f'\ncells among these two files where the FULL-CENSUS budget triggers '
          f'downsampling: {n_downsampled_cells}', flush=True)
    print(f'selected {int(sel1["selected"].sum()):,} of {len(sel1):,} loans from these two files '
          f'({100 * sel1["selected"].mean():.1f}%)', flush=True)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S): {FAILURES}')
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
