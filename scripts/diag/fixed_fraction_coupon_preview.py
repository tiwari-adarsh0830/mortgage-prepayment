"""fixed_fraction_coupon_preview.py -- per-coupon preview of the
fixed_fraction sampler against the Step 1 census baseline, run BEFORE the
formal check_sampler_vs_census.py exists (that's Step 3 -- not yet
written). This is "by inspection," not the formal statistical check: does
the length-bias fix show up as a roughly CONSTANT incl_prob and a roughly
CONSTANT n_sampled/n_eligible ratio across coupons?

That contrast matters because Step 1's census baseline
(outputs/census_panel_baseline_cutoff_2020.csv) found n_eligible-per-loan
is strongly coupon-correlated: coupon 2.0's median is 3, coupon 3.5's is
35. A fixed k draws roughly the same OBSERVATION COUNT per loan regardless
of that difference, so it samples a much LARGER share of a low-coupon
loan's eligible months than a high-coupon loan's -- exactly the length
bias fixed_fraction exists to remove.

Runs select_observations() directly (NOT the full build_sequences_multiobs
pipeline -- no scaler fit, no sequence arrays, no train/test split) over
the FULL cutoff_2020 population, same vintages / H / min_hist as
census_panel_baseline.py, so the per-coupon numbers are directly
comparable to its output. Checkpointed per vintage, same resume-guard
discipline.

Usage:
    python scripts/diag/fixed_fraction_coupon_preview.py --cutoff_year 2020 --frac_draws 0.2
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_sequences_multiobs_zbc as m
import census_panel_baseline as cb   # reuse coupon_bucket(), H, MIN_HIST -- do not redefine

BASE = '/scratch/at7095/mortgage_prepayment'


def process_vintage(vintage, cutoff_ym, pmms_rates, zhvi_df, frac_draws, ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, f'{vintage}.pkl')
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            result = pickle.load(f)
        print(f'  {vintage}: loaded from checkpoint', flush=True)
        return result

    df = m.load_vintage_filtered(vintage, pmms_rates, zhvi_df, cutoff_ym, keep_ids=None)
    if df is None or df.empty:
        result = None
    else:
        obs = m.select_observations(
            df, k_draws=0, H=cb.H, min_hist=cb.MIN_HIST, draw_scheme='uniform',
            sampling_mode='fixed_fraction', frac_draws=frac_draws)

        if obs.empty:
            empty_f = pd.Series(dtype=np.float64)
            empty_i = pd.Series(dtype=np.int64)
            result = {
                'vintage': vintage,
                'n_sampled_by_coupon':      empty_i,
                'incl_prob_sum_by_coupon':  empty_f,
                'incl_prob_n_by_coupon':    empty_i,
                'incl_prob_min_by_coupon':  empty_f,
                'incl_prob_max_by_coupon':  empty_f,
            }
        else:
            coupon_map = df.drop_duplicates('loan_id').set_index('loan_id')['original_interest_rate']
            obs = obs.copy()
            obs['coupon'] = cb.coupon_bucket(obs['loan_id'].map(coupon_map))

            non_term = obs[~obs['is_terminal']]
            result = {
                'vintage': vintage,
                'n_sampled_by_coupon':     obs.groupby('coupon').size(),
                'incl_prob_sum_by_coupon': non_term.groupby('coupon')['incl_prob'].sum(),
                'incl_prob_n_by_coupon':   non_term.groupby('coupon')['incl_prob'].size(),
                'incl_prob_min_by_coupon': non_term.groupby('coupon')['incl_prob'].min(),
                'incl_prob_max_by_coupon': non_term.groupby('coupon')['incl_prob'].max(),
            }

    with open(ckpt_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'  {vintage}: checkpoint written', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year', type=int, required=True)
    parser.add_argument('--frac_draws', type=float, required=True)
    args = parser.parse_args()

    cutoff_ym = m.dec_yyyymm(args.cutoff_year)
    ckpt_dir = os.path.join(BASE, 'outputs', 'fixed_fraction_coupon_preview_checkpoints',
                             f'cutoff_{args.cutoff_year}_f{args.frac_draws}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Fixed-fraction coupon preview | cutoff = Dec {args.cutoff_year} '
          f'(YYYYMM={cutoff_ym}) | frac_draws={args.frac_draws} | H={cb.H} '
          f'min_hist={cb.MIN_HIST} (matches census_panel_baseline.py)', flush=True)
    print(f'Checkpoint dir: {ckpt_dir}', flush=True)

    pmms_rates = m.load_pmms()
    zhvi_df    = m.load_zhvi()

    n_sampled_total     = pd.Series(dtype=np.int64)
    incl_prob_sum_total = pd.Series(dtype=np.float64)
    incl_prob_n_total   = pd.Series(dtype=np.int64)
    incl_prob_min_total = pd.Series(dtype=np.float64)   # running elementwise min
    incl_prob_max_total = pd.Series(dtype=np.float64)   # running elementwise max

    for v in m.ALL_VINTAGES:
        result = process_vintage(v, cutoff_ym, pmms_rates, zhvi_df, args.frac_draws, ckpt_dir)
        if result is None:
            continue
        n_sampled_total     = n_sampled_total.add(result['n_sampled_by_coupon'], fill_value=0)
        incl_prob_sum_total = incl_prob_sum_total.add(result['incl_prob_sum_by_coupon'], fill_value=0)
        incl_prob_n_total   = incl_prob_n_total.add(result['incl_prob_n_by_coupon'], fill_value=0)
        incl_prob_min_total = pd.concat([incl_prob_min_total, result['incl_prob_min_by_coupon']],
                                         axis=1).min(axis=1)
        incl_prob_max_total = pd.concat([incl_prob_max_total, result['incl_prob_max_by_coupon']],
                                         axis=1).max(axis=1)

    incl_prob_mean = incl_prob_sum_total / incl_prob_n_total

    out = pd.DataFrame({
        'coupon':               n_sampled_total.index,
        'n_sampled_loan_months': n_sampled_total.astype(int).values,
    }).set_index('coupon')
    out['incl_prob_mean'] = incl_prob_mean
    out['incl_prob_min']  = incl_prob_min_total
    out['incl_prob_max']  = incl_prob_max_total
    out = out.reset_index().sort_values('coupon')

    census_path = os.path.join(BASE, 'outputs', f'census_panel_baseline_cutoff_{args.cutoff_year}.csv')
    census = pd.read_csv(census_path)
    census = census[census['coupon'] != 'ALL'].copy()
    census['coupon'] = census['coupon'].astype(float)

    merged = out.merge(
        census[['coupon', 'n_eligible_loan_months', 'n_eligible_median']],
        on='coupon', how='outer'
    ).sort_values('coupon')
    merged['sampled_over_eligible_ratio'] = (
        merged['n_sampled_loan_months'] / merged['n_eligible_loan_months']
    )

    out_path = os.path.join(BASE, 'outputs',
                             f'fixed_fraction_coupon_preview_cutoff_{args.cutoff_year}_f{args.frac_draws}.csv')
    merged.to_csv(out_path, index=False)
    print(f'\nWrote {out_path}', flush=True)
    print('\n' + merged.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
