"""check_sampler_vs_census.py -- the validation assertion the advisor asked
for: "in terms of validation, let's always compare the validation to the
census panel you are sampling from." Everything checked so far (the
fixed_fraction docstring formula, the per-loan spot check, and the
population-wide incl_prob-vs-formula check in
fixed_fraction_population_check.py) verified the sampler is INTERNALLY
consistent with its own ceil(f*n_pool) formula. None of that checks
whether the sampler, once IPW-corrected, actually reproduces the real
population it is drawing from. This script closes that gap.

Ground truth: outputs/census_panel_baseline_cutoff_2020.{json,csv} (Step 1
-- the FULL eligible (loan_id, ref_month) population, no sampling, built
from the same _eligible_candidates() the sampler draws from).

Sampler side: reruns select_observations() in fixed_fraction mode
(frac_draws=0.2) over all 41 vintages -- NOT reused from
fixed_fraction_coupon_preview.py's or fixed_fraction_population_check.py's
saved checkpoints, because neither retains the `label` field or an
all-rows (terminal + non-terminal) weight sum by coupon; both are needed
here and are cheap to recompute alongside a fresh pass.

THREE ASSERTIONS, per coupon and overall:

  1. IPW-weighted total loan-months (sum of 1/incl_prob over every sampled
     row, terminal + non-terminal) vs. census n_eligible_loan_months.
     For H=1 specifically, this is expected to match EXACTLY (not just
     within a tolerance), not approximately: the non-mandatory budget
     ceil(f*n_pool) is a DETERMINISTIC function of n_pool (see
     fixed_fraction_population_check.py's exact-match result), and every
     selected pool row gets the SAME weight n_pool/budget, so
     sum(weight over the budget selected pool rows) = budget *
     (n_pool/budget) = n_pool exactly, regardless of WHICH pool rows the
     hash-rank draw happens to pick. Add the mandatory row's weight (1.0,
     deterministic) and the per-loan HT total is exactly n_pool + 1 =
     that loan's full eligible-month count, with zero sampling variance.
     This is a prediction to VERIFY empirically, not an assumption -- if
     it does not hold exactly, that is itself the finding.

  2. IPW-weighted prepay rate vs. census raw_prepay_rate. For H=1, the
     label formula (is_prepaid & term_t - row_idx <= H) is 1 only at
     row_idx = term_t - 1 -- the SAME row_idx as the mandatory draw
     (row_idx = term_t - H = term_t - 1). So the one candidate row that
     can ever be labeled 1 is always the deterministically-sampled
     mandatory row, never a stochastically-drawn pool row. The prepay
     count should therefore also reproduce exactly, not approximately.
     Also verified empirically, not assumed.

  3. Weight direction: every non-terminal weight (1/incl_prob) should be
     >= 1, since incl_prob <= 1 always. The historical bug class this
     guards against (scripts/prepare_sequences_multiobs_zbc.py commit
     history: train_hazard_multiobs.py weighting by incl_prob directly
     instead of 1/incl_prob) would show up as weights systematically BELOW
     1 for the bulk of the distribution. A literal ">1 strictly" assertion
     is known, from fixed_fraction_population_check.py's own by-n_pool
     table, to be FALSE at n_pool=1 (ceil(0.2*1)/1 = 1.0 exactly, weight
     = 1.0 exactly, not > 1) -- that is an already-diagnosed, expected
     edge case, not the bug class being guarded against. This script
     therefore checks weight >= 1 (allowing the known n_pool=1 boundary)
     AND separately confirms every row sitting exactly at that boundary
     really is an n_pool=1 row (not a different, unexplained cause) --
     which is the real thing that would distinguish "known edge case"
     from "new bug."

TOLERANCE: 1% relative deviation on checks 1 and 2, chosen as a generous
sanity margin given the analysis above predicts EXACT (not approximately
close) agreement for H=1 -- if the true deviation is near 0 the 1% bound
is not doing much work, and if it is not near 0 that itself needs
explaining rather than being tolerance-hidden. For the two degenerate
low-n_pool coupons (1.0, 1.5), the census event counts are tiny
(n_prepay_events = 2 and 101 respectively) -- a percent-relative tolerance
on a rate built from 2 events is not a statistically meaningful bar
(one event of sampling difference is a 50% swing), so those two coupons
are reported with actual counts and flagged rather than scored PASS/FAIL
on the same 1% rule as the main coupons.

Checkpointed per vintage, same resume-guard discipline as this repo's
other diag scripts.

Usage:
    python scripts/diag/check_sampler_vs_census.py --frac_draws 0.2
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_sequences_multiobs_zbc as m
import census_panel_baseline as cb

BASE = '/scratch/at7095/mortgage_prepayment'
CUTOFF_YEAR = 2020
TOL = 0.01
DEGENERATE_COUPONS = {1.0, 1.5}


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
            result = {'by_coupon': None, 'weight_check': None}
        else:
            coupon_map = df.drop_duplicates('loan_id').set_index('loan_id')['original_interest_rate']
            obs = obs.copy()
            obs['coupon'] = cb.coupon_bucket(obs['loan_id'].map(coupon_map))
            obs['weight'] = 1.0 / obs['incl_prob']
            obs['weight_label'] = obs['weight'] * obs['label']

            by_coupon = obs.groupby('coupon').agg(
                sum_weight=('weight', 'sum'),
                sum_weight_label=('weight_label', 'sum'),
                n_rows=('weight', 'size'),
            )

            non_term = obs[~obs['is_terminal']]
            if non_term.empty:
                weight_check = None
            else:
                boundary = non_term[non_term['incl_prob'] >= 1.0 - 1e-12]
                n_boundary_not_pool1 = int((boundary['n_eligible'] != 1).sum())
                n_weight_lt_1_strict = int((non_term['weight'] < 1.0 - 1e-9).sum())
                weight_check = {
                    'n_nonterminal_rows': int(len(non_term)),
                    'n_boundary_incl_prob_eq1': int(len(boundary)),
                    'n_boundary_not_explained_by_npool1': n_boundary_not_pool1,
                    'n_weight_lt_1_strict': n_weight_lt_1_strict,
                    'min_incl_prob_nonterminal': float(non_term['incl_prob'].min()),
                    'min_weight_nonterminal': float(non_term['weight'].min()),
                }
            result = {'by_coupon': by_coupon, 'weight_check': weight_check}

    with open(ckpt_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'  {vintage}: checkpoint written', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac_draws', type=float, required=True)
    args = parser.parse_args()

    cutoff_ym = m.dec_yyyymm(CUTOFF_YEAR)
    ckpt_dir = os.path.join(BASE, 'outputs', 'check_sampler_vs_census_checkpoints',
                             f'cutoff_{CUTOFF_YEAR}_f{args.frac_draws}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Sampler-vs-census check | cutoff = Dec {CUTOFF_YEAR} (YYYYMM={cutoff_ym}) | '
          f'frac_draws={args.frac_draws} | H={cb.H} min_hist={cb.MIN_HIST}', flush=True)
    print(f'Checkpoint dir: {ckpt_dir}', flush=True)

    pmms_rates = m.load_pmms()
    zhvi_df = m.load_zhvi()

    total_by_coupon = None
    wc_n_nonterminal = 0
    wc_n_boundary = 0
    wc_n_boundary_unexplained = 0
    wc_n_weight_lt_1_strict = 0
    wc_min_incl_prob = None
    wc_min_weight = None

    for v in m.ALL_VINTAGES:
        result = process_vintage(v, cutoff_ym, pmms_rates, zhvi_df, args.frac_draws, ckpt_dir)
        if result is None:
            continue
        bc = result['by_coupon']
        if bc is not None:
            total_by_coupon = bc.copy() if total_by_coupon is None else total_by_coupon.add(bc, fill_value=0)
        wc = result['weight_check']
        if wc is not None:
            wc_n_nonterminal += wc['n_nonterminal_rows']
            wc_n_boundary += wc['n_boundary_incl_prob_eq1']
            wc_n_boundary_unexplained += wc['n_boundary_not_explained_by_npool1']
            wc_n_weight_lt_1_strict += wc['n_weight_lt_1_strict']
            wc_min_incl_prob = wc['min_incl_prob_nonterminal'] if wc_min_incl_prob is None \
                else min(wc_min_incl_prob, wc['min_incl_prob_nonterminal'])
            wc_min_weight = wc['min_weight_nonterminal'] if wc_min_weight is None \
                else min(wc_min_weight, wc['min_weight_nonterminal'])

    # ---- Load census ground truth ----
    with open(os.path.join(BASE, 'outputs', f'census_panel_baseline_cutoff_{CUTOFF_YEAR}.json')) as f:
        census = json.load(f)

    rows = []
    coupon_keys = [('ALL', census['overall'])] + \
                  [(c, census['by_coupon'][c]) for c in census['by_coupon']]
    for key, stats in coupon_keys:
        census_months = stats['n_eligible_loan_months']
        census_events = stats['n_prepay_events']
        census_rate   = stats['raw_prepay_rate']

        if key == 'ALL':
            ipw_months = total_by_coupon['sum_weight'].sum()
            ipw_events = total_by_coupon['sum_weight_label'].sum()
        else:
            c = float(key)
            if c not in total_by_coupon.index:
                ipw_months = 0.0
                ipw_events = 0.0
            else:
                ipw_months = total_by_coupon.loc[c, 'sum_weight']
                ipw_events = total_by_coupon.loc[c, 'sum_weight_label']

        ipw_rate = (ipw_events / ipw_months) if ipw_months > 0 else np.nan

        dev_months = (ipw_months - census_months) / census_months if census_months else np.nan
        dev_rate   = (ipw_rate - census_rate) / census_rate if census_rate else np.nan

        is_degenerate = (key != 'ALL') and (float(key) in DEGENERATE_COUPONS)
        pass_months = abs(dev_months) <= TOL
        pass_rate   = abs(dev_rate) <= TOL

        rows.append({
            'coupon': key,
            'degenerate_low_n': is_degenerate,
            'n_eligible_loan_months_census': census_months,
            'n_eligible_loan_months_ipw_hat': ipw_months,
            'rel_dev_n_eligible': dev_months,
            'pass_n_eligible_1pct': (None if is_degenerate else bool(pass_months)),
            'n_prepay_events_census': census_events,
            'n_prepay_events_ipw_hat': ipw_events,
            'raw_prepay_rate_census': census_rate,
            'ipw_prepay_rate_hat': ipw_rate,
            'rel_dev_prepay_rate': dev_rate,
            'pass_prepay_rate_1pct': (None if is_degenerate else bool(pass_rate)),
        })

    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(BASE, 'outputs', f'check_sampler_vs_census_f{args.frac_draws}.csv')
    out_df.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}', flush=True)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    print('\n=== CHECK 1 & 2: per-coupon IPW-weighted totals vs. census (tolerance = '
          f'{TOL*100:.0f}%) ===')
    print(out_df.to_string(index=False))

    non_degenerate = out_df[~out_df['degenerate_low_n'] & (out_df['coupon'] != 'ALL')]
    failed_months = non_degenerate[non_degenerate['pass_n_eligible_1pct'] == False]
    failed_rate = non_degenerate[non_degenerate['pass_prepay_rate_1pct'] == False]

    print(f'\nCoupons failing n_eligible_loan_months tolerance ({TOL*100:.0f}%):')
    if failed_months.empty:
        print('  none')
    else:
        for _, r in failed_months.iterrows():
            direction = 'OVER-represents' if r['rel_dev_n_eligible'] > 0 else 'UNDER-represents'
            print(f"  coupon {r['coupon']}: rel_dev={r['rel_dev_n_eligible']:.4%}, sample {direction} census "
                  f"(census={r['n_eligible_loan_months_census']:.0f}, ipw_hat={r['n_eligible_loan_months_ipw_hat']:.2f})")

    print(f'\nCoupons failing prepay-rate tolerance ({TOL*100:.0f}%):')
    if failed_rate.empty:
        print('  none')
    else:
        for _, r in failed_rate.iterrows():
            direction = 'OVER-represents' if r['rel_dev_prepay_rate'] > 0 else 'UNDER-represents'
            print(f"  coupon {r['coupon']}: rel_dev={r['rel_dev_prepay_rate']:.4%}, sample {direction} census "
                  f"(census={r['raw_prepay_rate_census']:.6f}, ipw_hat={r['ipw_prepay_rate_hat']:.6f})")

    degenerate_rows = out_df[out_df['degenerate_low_n']]
    print('\nDegenerate low-n_pool coupons (1.0, 1.5) -- reported, not scored PASS/FAIL:')
    print(degenerate_rows[['coupon', 'n_eligible_loan_months_census', 'n_eligible_loan_months_ipw_hat',
                            'rel_dev_n_eligible', 'n_prepay_events_census', 'n_prepay_events_ipw_hat',
                            'raw_prepay_rate_census', 'ipw_prepay_rate_hat', 'rel_dev_prepay_rate']]
          .to_string(index=False))
    print('n_prepay_events_census is 2 (coupon 1.0) and 101 (coupon 1.5) -- a single-event '
          'difference is a 50% / ~1% swing respectively in the rate; the 1% tolerance used for '
          'the main coupons is not statistically meaningful at these event counts and is not '
          'applied here.')

    print('\n=== CHECK 3: weight direction (non-terminal rows) ===')
    print(f'Total non-terminal sampled rows (all vintages): {wc_n_nonterminal:,}')
    print(f'Rows with incl_prob >= 1 - 1e-12 (weight <= 1, the n_pool=1 boundary): {wc_n_boundary:,}')
    print(f'  of which NOT explained by n_eligible==1 (i.e. unexplained boundary rows): '
          f'{wc_n_boundary_unexplained:,}')
    print(f'Rows with weight < 1 - 1e-9 STRICTLY (the historical inverted-weight bug signature): '
          f'{wc_n_weight_lt_1_strict:,}')
    print(f'min(incl_prob) over all non-terminal rows: {wc_min_incl_prob}')
    print(f'min(weight) over all non-terminal rows: {wc_min_weight}')
    if wc_n_weight_lt_1_strict == 0 and wc_n_boundary_unexplained == 0:
        print('PASS: every non-terminal weight is >= 1; every row sitting exactly at the =1 '
              'boundary is explained by n_pool=1 (the known ceil() edge case from '
              'fixed_fraction_population_check.py), not by an inverted-weight bug.')
    else:
        print('FAIL: see counts above -- do not assume the cause, investigate the unexplained '
              'rows directly.')


if __name__ == '__main__':
    main()
