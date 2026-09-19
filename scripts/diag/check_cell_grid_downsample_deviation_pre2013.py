"""check_cell_grid_downsample_deviation_pre2013.py -- empirical check that
hash-based loan downsampling in build_cell_grid_sample_pre2013.py actually
lands near the TARGET_CAP=5000-event target, not just in expectation.

Scope: the 2002Q3.csv / 2011Q2.csv two-file test population only (same
files as test_cell_grid_sample_pre2013.py). budget per cell is still
computed from the FULL 52-quarter census
(outputs/prepay_event_counts_pre2013.csv), as build_cell_grid_sample_pre2013
does for a real run -- this script does not change that. CAVEAT: for a
cell like (2002Q3, 6.0), the two-file scan's loan population is presumably
the large majority but not necessarily 100% of that cell's full-census
population (a small tail of same-vintage loans can be acquired in
adjacent-quarter files, as documented in
count_prepay_events_pre2013.py's docstring for early vintages). So this
checks "does selecting `budget` loans out of the loans we actually have
here land near 5,000 events", which is the available empirical proxy for
"does it land near 5,000 events at full scale" -- not a substitute for
checking the full 52-file run itself.

For every cell where this two-file population already exceeds its budget
(i.e. downsampling actually triggers here), reports:
  budget, n_selected (== budget, by construction of rank<=budget),
  n_events_selected (the actual code-01 count within the selected subset),
  deviation from 5000 in absolute count and percent.

Run via sbatch (see run_check_cell_grid_downsample_deviation.sbatch) --
the login node OOM-kills a python process that scans these files directly.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from count_prepay_events_pre2013 import PREPAID_CODE
import build_cell_grid_sample_pre2013 as bcs

FILES = ['2002Q3.csv', '2011Q2.csv']
CAP = bcs.TARGET_CAP


def main():
    budget = bcs.compute_budgets()
    records = bcs.scan_files(FILES, ckpt_path=None)
    df = bcs.select_loans(records, budget)

    g = df.groupby(['vintage_quarter', 'coupon'])
    rows = []
    for (q, c), d in g:
        n_loans = len(d)
        cell_budget = int(d['cell_budget'].iloc[0])
        if n_loans <= cell_budget:
            continue  # not downsampled within this two-file population
        n_selected = int(d['selected'].sum())
        n_events_selected = int((d.loc[d['selected'], 'zbc'] == PREPAID_CODE).sum())
        dev_abs = n_events_selected - CAP
        dev_pct = 100.0 * dev_abs / CAP
        rows.append({
            'vintage_quarter': q, 'coupon': c,
            'n_loans_2file': n_loans, 'budget': cell_budget,
            'n_selected': n_selected, 'n_events_selected': n_events_selected,
            'target': CAP, 'dev_abs': dev_abs, 'dev_pct': dev_pct,
        })

    out = pd.DataFrame(rows).sort_values('dev_pct', key=lambda s: s.abs(), ascending=False)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', None)
    print(f'{len(out)} cells downsampled within the 2002Q3/2011Q2 two-file population\n')
    print(out.to_string(index=False))

    print()
    print(f'max |dev_pct|: {out["dev_pct"].abs().max():.2f}%')
    print(f'mean |dev_pct|: {out["dev_pct"].abs().mean():.2f}%')
    n_over_10 = (out['dev_pct'].abs() > 10).sum()
    n_over_15 = (out['dev_pct'].abs() > 15).sum()
    print(f'cells with |dev_pct| > 10%: {n_over_10}')
    print(f'cells with |dev_pct| > 15%: {n_over_15}')

    out_path = os.path.join(bcs.OUT, 'cell_grid_downsample_deviation_TEST_2q.csv')
    out.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
