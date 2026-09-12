"""check_sampler_vs_census_by_vintage.py -- robustness check requested after
17308661 (check_sampler_vs_census.py) passed at the pooled (all-41-vintage)
level: verify no individual vintage is failing tolerance while canceling
out in the aggregate (e.g. one vintage over-representing offset by another
under-representing).

Re-aggregates the SAME two already-computed checkpoint sets used by the
overnight job -- no new sampling or census enumeration is run:

  - outputs/census_panel_baseline_checkpoints/cutoff_2020/{vintage}.pkl
    (ground truth: eligible_months_by_coupon, prepay_events_by_coupon,
    summed here across coupon to get a per-vintage total)
  - outputs/check_sampler_vs_census_checkpoints/cutoff_2020_f0.2/{vintage}.pkl
    (IPW-weighted sampler estimate: by_coupon['sum_weight'],
    by_coupon['sum_weight_label'], summed the same way)

Same two checks, same 1% tolerance, as check_sampler_vs_census.py, but
broken out by VINTAGE instead of pooled across all 41.

Usage:
    python scripts/diag/check_sampler_vs_census_by_vintage.py
"""
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare_sequences_multiobs_zbc as m

BASE = '/scratch/at7095/mortgage_prepayment'
CUTOFF_YEAR = 2020
FRAC_DRAWS = 0.2
TOL = 0.01

CENSUS_CKPT_DIR = os.path.join(BASE, 'outputs', 'census_panel_baseline_checkpoints',
                                f'cutoff_{CUTOFF_YEAR}')
SAMPLER_CKPT_DIR = os.path.join(BASE, 'outputs', 'check_sampler_vs_census_checkpoints',
                                 f'cutoff_{CUTOFF_YEAR}_f{FRAC_DRAWS}')


def main():
    rows = []
    for v in m.ALL_VINTAGES:
        census_path = os.path.join(CENSUS_CKPT_DIR, f'{v}.pkl')
        sampler_path = os.path.join(SAMPLER_CKPT_DIR, f'{v}.pkl')
        if not os.path.exists(census_path) or not os.path.exists(sampler_path):
            print(f'  {v}: missing checkpoint(s), skipping', flush=True)
            continue

        with open(census_path, 'rb') as f:
            cd = pickle.load(f)
        with open(sampler_path, 'rb') as f:
            sd = pickle.load(f)

        # Vintages originated after the cutoff (2021Q1-2023Q1, cutoff=Dec 2020)
        # have zero pre-cutoff history by construction -- both census.py and
        # check_sampler_vs_census.py write result=None for these (df is
        # None/empty before any eligibility/sampling step runs), not a
        # computation failure. Treat as a trivial 0-vs-0 pass, not NaN.
        if cd is None or sd is None:
            census_months = census_events = 0
            ipw_months = ipw_events = 0.0
        else:
            census_months = int(cd['eligible_months_by_coupon'].sum())
            census_events = int(cd['prepay_events_by_coupon'].sum())
            bc = sd['by_coupon']
            if bc is None:
                ipw_months = 0.0
                ipw_events = 0.0
            else:
                ipw_months = float(bc['sum_weight'].sum())
                ipw_events = float(bc['sum_weight_label'].sum())

        census_rate = (census_events / census_months) if census_months else float('nan')
        ipw_rate = (ipw_events / ipw_months) if ipw_months > 0 else float('nan')

        if census_months == 0:
            dev_months = 0.0 if ipw_months == 0 else float('inf')
        else:
            dev_months = (ipw_months - census_months) / census_months
        if census_events == 0 or census_months == 0:
            dev_rate = 0.0 if (ipw_events == 0 or ipw_months == 0) else float('inf')
        else:
            dev_rate = (ipw_rate - census_rate) / census_rate

        rows.append({
            'vintage': v,
            'n_eligible_loan_months_census': census_months,
            'n_eligible_loan_months_ipw_hat': ipw_months,
            'rel_dev_n_eligible': dev_months,
            'pass_n_eligible_1pct': bool(abs(dev_months) <= TOL),
            'n_prepay_events_census': census_events,
            'n_prepay_events_ipw_hat': ipw_events,
            'raw_prepay_rate_census': census_rate,
            'ipw_prepay_rate_hat': ipw_rate,
            'rel_dev_prepay_rate': dev_rate,
            'pass_prepay_rate_1pct': bool(abs(dev_rate) <= TOL),
        })

    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(BASE, 'outputs', f'check_sampler_vs_census_by_vintage_f{FRAC_DRAWS}.csv')
    out_df.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}\n')

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 50)
    print(f'=== Per-vintage IPW-weighted totals vs. census (tolerance = {TOL*100:.0f}%) ===')
    print(out_df.to_string(index=False))

    n_total = len(out_df)
    fail_months = out_df[~out_df['pass_n_eligible_1pct']]
    fail_rate = out_df[~out_df['pass_prepay_rate_1pct']]
    n_pass_both = int(((out_df['pass_n_eligible_1pct']) & (out_df['pass_prepay_rate_1pct'])).sum())

    print(f'\n{n_pass_both} / {n_total} vintages pass BOTH checks individually.')

    print(f'\nVintages failing n_eligible_loan_months tolerance ({TOL*100:.0f}%):')
    if fail_months.empty:
        print('  none')
    else:
        for _, r in fail_months.iterrows():
            direction = 'OVER-represents' if r['rel_dev_n_eligible'] > 0 else 'UNDER-represents'
            print(f"  {r['vintage']}: rel_dev={r['rel_dev_n_eligible']:.4%}, sample {direction} census "
                  f"(census={r['n_eligible_loan_months_census']:.0f}, ipw_hat={r['n_eligible_loan_months_ipw_hat']:.2f})")

    print(f'\nVintages failing prepay-rate tolerance ({TOL*100:.0f}%):')
    if fail_rate.empty:
        print('  none')
    else:
        for _, r in fail_rate.iterrows():
            direction = 'OVER-represents' if r['rel_dev_prepay_rate'] > 0 else 'UNDER-represents'
            print(f"  {r['vintage']}: rel_dev={r['rel_dev_prepay_rate']:.4%}, sample {direction} census "
                  f"(census={r['raw_prepay_rate_census']:.6f}, ipw_hat={r['ipw_prepay_rate_hat']:.6f})")

    if n_pass_both == n_total:
        print(f'\nPASS: all {n_total} vintages pass both checks individually -- the pooled '
              f'(all-41-vintage) pass is not hiding any offsetting per-vintage errors.')
    else:
        print(f'\n{n_total - n_pass_both} of {n_total} vintages fail at least one check '
              f'individually despite the pooled aggregate passing -- see deviations above.')


if __name__ == '__main__':
    main()
