"""inspect_fixed_fraction_examples.py -- verify the fixed_fraction sampler's
mechanism directly, per-loan, rather than inferring it from the aggregate
per-coupon table in fixed_fraction_coupon_preview.py.

Loads ONE vintage (2013Q1), runs select_observations() in fixed_fraction
mode exactly as fixed_fraction_coupon_preview.py does, then for coupon 2.0
(small n_eligible) and coupon 3.5 (large n_eligible) prints a sample of
per-loan (n_eligible, k_actual, budget_expected=ceil(0.2*n_eligible),
has_terminal) rows plus the non-terminal incl_prob distribution, so the
floor/ceil effect at small n_eligible is visible directly instead of
inferred from the coupon-level incl_prob_mean shape.

Usage:
    python scripts/diag/inspect_fixed_fraction_examples.py
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_sequences_multiobs_zbc as m
import census_panel_baseline as cb

VINTAGE = '2013Q1'


def main():
    cutoff_ym = m.dec_yyyymm(2020)
    pmms_rates = m.load_pmms()
    zhvi_df = m.load_zhvi()

    df = m.load_vintage_filtered(VINTAGE, pmms_rates, zhvi_df, cutoff_ym, keep_ids=None)
    print('rows loaded:', len(df), flush=True)

    obs = m.select_observations(
        df, k_draws=0, H=cb.H, min_hist=cb.MIN_HIST, draw_scheme='uniform',
        sampling_mode='fixed_fraction', frac_draws=0.2)
    print('obs rows:', len(obs), flush=True)

    coupon_map = df.drop_duplicates('loan_id').set_index('loan_id')['original_interest_rate']
    obs = obs.copy()
    obs['coupon'] = cb.coupon_bucket(obs['loan_id'].map(coupon_map))

    per_loan = obs.groupby('loan_id').agg(
        coupon=('coupon', 'first'),
        n_eligible=('n_eligible', 'first'),
        k_actual=('k_actual', 'first'),
        has_terminal=('is_terminal', 'max'),
    ).reset_index()
    per_loan['budget_expected'] = np.ceil(0.2 * per_loan['n_eligible']).astype(int)

    for coupon_val in [2.0, 3.5]:
        sub = per_loan[per_loan['coupon'] == coupon_val]
        print(f'\n=== coupon {coupon_val}: n_loans={len(sub)}, '
              f'median n_eligible={sub["n_eligible"].median()} ===', flush=True)
        print(sub[['loan_id', 'n_eligible', 'k_actual', 'has_terminal', 'budget_expected']]
              .sample(min(10, len(sub)), random_state=1).to_string(index=False), flush=True)
        non_term_obs = obs[(obs['coupon'] == coupon_val) & (~obs['is_terminal'])]
        print('non-terminal incl_prob stats:',
              non_term_obs['incl_prob'].describe()[['mean', 'min', 'max']].to_dict(), flush=True)


if __name__ == '__main__':
    main()
