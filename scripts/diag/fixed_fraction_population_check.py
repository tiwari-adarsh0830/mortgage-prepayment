"""fixed_fraction_population_check.py -- population-wide confirmation that
incl_prob for a non-terminal fixed_fraction draw is an EXACT deterministic
function of n_pool (incl_prob = ceil(frac_draws * n_pool) / n_pool), not
merely consistent with it on a handful of spot-checked loans.

Reruns select_observations() over every vintage (same population as
fixed_fraction_coupon_preview.py / census_panel_baseline.py), but groups
non-terminal rows by n_pool (the `n_eligible` field) instead of by coupon,
and compares the OBSERVED incl_prob against the FORMULA prediction for
that n_pool. Also retains a (coupon, n_pool) breakdown so a coupon's
observed incl_prob_mean can be checked against what its own n_pool
distribution alone would predict -- i.e. whether any coupon's incl_prob
is explained by something other than its n_pool mix.

Checkpointed per vintage, same resume-guard discipline as the other diag
scripts in this directory.

Usage:
    python scripts/diag/fixed_fraction_population_check.py --frac_draws 0.2
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
import census_panel_baseline as cb   # reuse coupon_bucket(), H, MIN_HIST

BASE = '/scratch/at7095/mortgage_prepayment'
CUTOFF_YEAR = 2020


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

        non_term = obs[~obs['is_terminal']].copy()
        if non_term.empty:
            result = {'vintage': vintage, 'by_npool': None, 'by_coupon_npool': None}
        else:
            coupon_map = df.drop_duplicates('loan_id').set_index('loan_id')['original_interest_rate']
            non_term['coupon'] = cb.coupon_bucket(non_term['loan_id'].map(coupon_map))

            n_pool = non_term['n_eligible'].astype(np.int64)
            expected = np.ceil(frac_draws * n_pool) / n_pool
            non_term['expected_incl_prob'] = expected
            non_term['abs_dev'] = (non_term['incl_prob'] - expected).abs()

            by_npool = non_term.groupby('n_eligible').agg(
                n_rows=('incl_prob', 'size'),
                incl_prob_sum=('incl_prob', 'sum'),
                incl_prob_min=('incl_prob', 'min'),
                incl_prob_max=('incl_prob', 'max'),
                expected_incl_prob=('expected_incl_prob', 'first'),
                max_abs_dev=('abs_dev', 'max'),
            )
            by_coupon_npool = non_term.groupby(['coupon', 'n_eligible']).agg(
                n_rows=('incl_prob', 'size'),
                incl_prob_sum=('incl_prob', 'sum'),
            )
            result = {'vintage': vintage, 'by_npool': by_npool, 'by_coupon_npool': by_coupon_npool}

    with open(ckpt_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'  {vintage}: checkpoint written', flush=True)
    return result


def combine_by_npool(total, part):
    if part is None:
        return total
    if total is None:
        return part.copy()
    out = total.join(part, how='outer', lsuffix='_a', rsuffix='_b')
    out['n_rows'] = out[['n_rows_a', 'n_rows_b']].sum(axis=1, skipna=True)
    out['incl_prob_sum'] = out[['incl_prob_sum_a', 'incl_prob_sum_b']].sum(axis=1, skipna=True)
    out['incl_prob_min'] = out[['incl_prob_min_a', 'incl_prob_min_b']].min(axis=1, skipna=True)
    out['incl_prob_max'] = out[['incl_prob_max_a', 'incl_prob_max_b']].max(axis=1, skipna=True)
    out['expected_incl_prob'] = out[['expected_incl_prob_a', 'expected_incl_prob_b']].bfill(axis=1).iloc[:, 0]
    out['max_abs_dev'] = out[['max_abs_dev_a', 'max_abs_dev_b']].max(axis=1, skipna=True)
    return out[['n_rows', 'incl_prob_sum', 'incl_prob_min', 'incl_prob_max',
                'expected_incl_prob', 'max_abs_dev']]


def combine_by_coupon_npool(total, part):
    if part is None:
        return total
    if total is None:
        return part.copy()
    out = total.add(part, fill_value=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac_draws', type=float, required=True)
    args = parser.parse_args()

    cutoff_ym = m.dec_yyyymm(CUTOFF_YEAR)
    ckpt_dir = os.path.join(BASE, 'outputs', 'fixed_fraction_population_check_checkpoints',
                             f'cutoff_{CUTOFF_YEAR}_f{args.frac_draws}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Fixed-fraction population check | cutoff = Dec {CUTOFF_YEAR} '
          f'(YYYYMM={cutoff_ym}) | frac_draws={args.frac_draws} | H={cb.H} '
          f'min_hist={cb.MIN_HIST}', flush=True)
    print(f'Checkpoint dir: {ckpt_dir}', flush=True)

    pmms_rates = m.load_pmms()
    zhvi_df = m.load_zhvi()

    total_by_npool = None
    total_by_coupon_npool = None

    for v in m.ALL_VINTAGES:
        result = process_vintage(v, cutoff_ym, pmms_rates, zhvi_df, args.frac_draws, ckpt_dir)
        if result is None:
            continue
        total_by_npool = combine_by_npool(total_by_npool, result['by_npool'])
        total_by_coupon_npool = combine_by_coupon_npool(total_by_coupon_npool, result['by_coupon_npool'])

    # ---- Check 1: population-wide deviation from the formula, by n_pool ----
    total_by_npool = total_by_npool.sort_index()
    total_by_npool['incl_prob_mean'] = total_by_npool['incl_prob_sum'] / total_by_npool['n_rows']
    total_by_npool['mean_vs_expected_dev'] = (
        total_by_npool['incl_prob_mean'] - total_by_npool['expected_incl_prob']
    ).abs()

    npool_path = os.path.join(BASE, 'outputs', f'fixed_fraction_population_check_by_npool_f{args.frac_draws}.csv')
    total_by_npool.reset_index().to_csv(npool_path, index=False)
    print(f'\nWrote {npool_path}', flush=True)

    max_dev = total_by_npool['max_abs_dev'].max()
    max_mean_dev = total_by_npool['mean_vs_expected_dev'].max()
    print(f'\n=== CHECK 1: incl_prob vs ceil(f*n_pool)/n_pool, by n_pool, over full population ===')
    print(f'n_pool groups: {len(total_by_npool)}')
    print(f'max |incl_prob_min/max - expected| across all rows/groups: {max_dev:.3e}')
    print(f'max |incl_prob_mean(n_pool) - expected(n_pool)|:           {max_mean_dev:.3e}')
    worst = total_by_npool.sort_values('max_abs_dev', ascending=False).head(10)
    print('\nTop 10 n_pool groups by max_abs_dev:')
    print(worst[['n_rows', 'incl_prob_mean', 'expected_incl_prob', 'incl_prob_min',
                 'incl_prob_max', 'max_abs_dev']].to_string())

    # ---- Check 2: within each coupon, is incl_prob_mean explained purely ----
    # ---- by that coupon's n_pool distribution via the formula?          ----
    tbcn = total_by_coupon_npool.reset_index()
    tbcn['expected_incl_prob'] = np.ceil(args.frac_draws * tbcn['n_eligible']) / tbcn['n_eligible']
    tbcn['expected_sum'] = tbcn['expected_incl_prob'] * tbcn['n_rows']

    by_coupon = tbcn.groupby('coupon').apply(
        lambda g: pd.Series({
            'n_rows': g['n_rows'].sum(),
            'incl_prob_mean_actual': g['incl_prob_sum'].sum() / g['n_rows'].sum(),
            'incl_prob_mean_predicted_from_npool_mix': g['expected_sum'].sum() / g['n_rows'].sum(),
            'mean_n_pool': (g['n_eligible'] * g['n_rows']).sum() / g['n_rows'].sum(),
        }), include_groups=False
    ).reset_index()
    by_coupon['actual_vs_predicted_dev'] = (
        by_coupon['incl_prob_mean_actual'] - by_coupon['incl_prob_mean_predicted_from_npool_mix']
    ).abs()
    by_coupon = by_coupon.sort_values('mean_n_pool')

    coupon_path = os.path.join(BASE, 'outputs', f'fixed_fraction_population_check_by_coupon_f{args.frac_draws}.csv')
    by_coupon.to_csv(coupon_path, index=False)
    print(f'\nWrote {coupon_path}', flush=True)

    print(f'\n=== CHECK 2: per-coupon incl_prob_mean vs formula-predicted from that coupon\'s n_pool mix ===')
    print(by_coupon.to_string(index=False))
    print(f'\nmax |actual - predicted-from-n_pool-mix| across coupons: '
          f'{by_coupon["actual_vs_predicted_dev"].max():.3e}')

    # Monotonicity: as mean_n_pool increases, incl_prob_mean should be
    # non-increasing until it flattens near frac_draws.
    sorted_by_npool = by_coupon.sort_values('mean_n_pool')
    diffs = sorted_by_npool['incl_prob_mean_actual'].diff().dropna()
    violations = sorted_by_npool.loc[diffs[diffs > 1e-9].index]
    print(f'\nCoupons where incl_prob_mean INCREASED despite mean_n_pool increasing '
          f'(non-monotonic, tol 1e-9):')
    if violations.empty:
        print('  none')
    else:
        print(violations[['coupon', 'mean_n_pool', 'incl_prob_mean_actual']].to_string(index=False))


if __name__ == '__main__':
    main()
