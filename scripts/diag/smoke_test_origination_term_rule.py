"""
One-off smoke test for the origination-row term rule patch (advisor's Oct 4
decision, Part 1 of the Oct 5 prompt). Runs each patched reader on exactly
one quarter file (2002Q1 for the pre-2013 readers, 2018Q1 for the others)
and prints kept/dropped loan counts next to the old (any-row) rule's
reference numbers (594,424 kept / 367,395 dropped, as given in the prompt),
so the difference can be checked against the known 3 mixed-term loans.

Not a permanent test -- ad hoc, run once, not wired into any gate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import prepare_sequences_multiobs_zbc as multiobs_zbc
import prepare_sequences_rolling_zbc as rolling_zbc
import prepare_sequences_trailing_zbc as trailing_zbc
import count_prepay_events_pre2013 as pre2013
import realized_cpr_v6
import realized_cpr_v6_upb

BASE = '/scratch/at7095/mortgage_prepayment'
F2002Q1_PRE2013 = os.path.join(BASE, 'data_pre2013_raw', '2002Q1.csv')
F2018Q1_MODERN = os.path.join(BASE, 'data', 'raw', '2018Q1.csv')

REF_KEPT, REF_DROPPED = 594_424, 367_395


def report(label, kept, dropped):
    diff_kept = kept - REF_KEPT
    diff_dropped = dropped - REF_DROPPED
    print(f'[{label}] kept={kept:,} dropped={dropped:,}  '
          f'(vs ref kept={REF_KEPT:,} dropped={REF_DROPPED:,}: '
          f'diff_kept={diff_kept:+,} diff_dropped={diff_dropped:+,})', flush=True)


print('=== prepare_sequences_multiobs_zbc.load_vintage_filtered, 2002Q1 (pre-2013, cell-gated) ===', flush=True)
pmms = multiobs_zbc.load_pmms()
zhvi = multiobs_zbc.load_zhvi()
df = multiobs_zbc.load_vintage_filtered('2002Q1', pmms, zhvi, cutoff_yyyymm=202012, keep_ids=None)
n_loans = df['loan_id'].nunique() if df is not None else 0
print(f'[multiobs_zbc 2002Q1] loader output: {n_loans:,} loans survive to the end of '
      f'load_vintage_filtered (cell-gated -- not directly comparable to the ungated '
      f'reference counts; the printed "term filter" line above is the comparable number)',
      flush=True)

print()
print('=== prepare_sequences_multiobs_zbc.load_vintage_filtered, 2018Q1 (modern, ungated) ===', flush=True)
df2 = multiobs_zbc.load_vintage_filtered('2018Q1', pmms, zhvi, cutoff_yyyymm=202412, keep_ids=None)
n_loans2 = df2['loan_id'].nunique() if df2 is not None else 0
print(f'[multiobs_zbc 2018Q1] loader output: {n_loans2:,} loans survive to the end of '
      f'load_vintage_filtered', flush=True)

print()
print('=== prepare_sequences_rolling_zbc.load_vintage_filtered, 2018Q1 ===', flush=True)
df3 = rolling_zbc.load_vintage_filtered('2018Q1', pmms, zhvi, cutoff_yyyymm=202412, keep_ids=None)
n_loans3 = df3['loan_id'].nunique() if df3 is not None else 0
print(f'[rolling_zbc 2018Q1] loader output: {n_loans3:,} loans', flush=True)

print()
print('=== prepare_sequences_trailing_zbc.load_vintage_filtered, 2018Q1 ===', flush=True)
df4 = trailing_zbc.load_vintage_filtered('2018Q1', pmms, zhvi, cutoff_yyyymm=202412, keep_ids=None)
n_loans4 = df4['loan_id'].nunique() if df4 is not None else 0
print(f'[trailing_zbc 2018Q1] loader output: {n_loans4:,} loans', flush=True)

print()
print('=== count_prepay_events_pre2013.scan_file, 2002Q1 ===', flush=True)
vint, cpn, zbc = pre2013.scan_file(F2002Q1_PRE2013)
print(f'[pre2013 scan_file 2002Q1] {len(vint):,} loans survive (vint dict size)', flush=True)

print()
print('=== realized_cpr_v6.pass0_global_last, 2018Q1 ===', flush=True)
prepay_month, rate_map, first_mod_ym = realized_cpr_v6.pass0_global_last([F2018Q1_MODERN])
print(f'[realized_cpr_v6 2018Q1] {len(rate_map):,} loans survive in rate_map after term filter',
      flush=True)

print()
print('=== realized_cpr_v6_upb.pass0_global_top2, 2018Q1 ===', flush=True)
prepay_month_u, rate_map_u, payoff_balance_u, first_mod_ym_u = \
    realized_cpr_v6_upb.pass0_global_top2([F2018Q1_MODERN])
print(f'[realized_cpr_v6_upb 2018Q1] {len(rate_map_u):,} loans survive in rate_map after term filter',
      flush=True)
