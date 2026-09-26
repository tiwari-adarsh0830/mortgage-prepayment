"""
score_matched_population_right_aligned.py -- like-for-like correction of the
Sep 6-7/Sep 8 README numbers for the cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a
checkpoint (hazard_best.pt=epoch4, hazard_final.pt=epoch50), which were
computed on LEFT-aligned TRAIL_SEQ_DIR sequences (condition-4 bug). Recomputes
each one RIGHT-aligned (one window per loan ending exactly Dec 2020, via
build_sequences_multiobs(obs=...)), on the SAME 365,146-loan matched
population, using this checkpoint's OWN training scaler
(data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1/scaler.pkl, confirmed
untouched since 2026-09-03, pre-fix category codes {0,1,2,3} only) and the
pre-fix CP/U maps (this checkpoint trained 2026-09-07, before commit
2a5b283 on 2026-09-19).

Also RECOMPUTES the left-aligned numbers fresh in this same run (not just
quoting README) so the old/new comparison is apples-to-apples on identical
population/cache, and doubles as a self-check against the already-published
README values.

Population note: forecast_matched_population_cpr.py's own orig_set &
trail_set & multi_set intersection is used directly (374,182 orig, 374,182
trail, 365,146 multi -> 365,146 intersection), matching the README's stated
365,146 exactly by construction, BEFORE the Dec-window eligibility filter
(active-in-2021 / has-Dec-2020-row / min_hist / calendar-gap) is applied for
the right-aligned build -- that filter can (and does) drop a few more loans;
the drop count is reported explicitly, not silently absorbed.

Usage:
    python scripts/score_matched_population_right_aligned.py --build_cache_only   # CPU build job
    python scripts/score_matched_population_right_aligned.py                     # GPU scoring job
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag'))

from score_multiobs_dec_window import (
    build_combined_pass, build_dec_window_obs, score, verify_windows, BASE,
)
from prepare_sequences_multiobs_zbc import build_sequences_multiobs
from forecast_matched_population_cpr import (
    CUTOFF_YEAR, BATCH_SIZE, TRAIL_SEQ_DIR, RAW_PASS_CACHE,
    score_multiobs_model, filter_to_population, cached_read_coupon_and_realized,
    mean_h_adj_by_coupon, pooled_comparison,
)
from forecast_rolling_cpr import aggregate, DEVICE
from forecast_multiobs_ipw_seedcheck_epoch_compare import SEEDCHECK_DIR, load_model_from_checkpoint
from diag_advisor_unannualized_epoch4 import (
    read_active_prepaid_single_month, realized_1mo_by_coupon, TARGET_YYYYMM,
)

K5H1_SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1')
COUPON_LO, COUPON_HI, MIN_N = 2.0, 5.0, 5000
CACHE_DIR = os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/matched_population_right_aligned_seedcheck_a')


def matched_population() -> set:
    orig  = np.load(os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc/test_loan_ids.npy'), allow_pickle=True)
    trail = np.load(os.path.join(TRAIL_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    multi = np.load(os.path.join(K5H1_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    pop = set(orig.tolist()) & set(trail.tolist()) & set(multi.tolist())
    print(f'Matched population: orig={len(orig):,} trail={len(trail):,} multi={len(multi):,} '
          f'intersection={len(pop):,} (README states 365,146)', flush=True)
    return pop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build_cache_only', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    population = matched_population()

    if args.build_cache_only:
        build_combined_pass(2020, False, 'prefix', population, CACHE_DIR)
        print('--build_cache_only: cache populated, exiting.', flush=True)
        return

    # ── RIGHT-aligned build ──────────────────────────────────────────────
    full_df = build_combined_pass(2020, False, 'prefix', population, CACHE_DIR)
    obs, dec_rows, diag = build_dec_window_obs(full_df, 2020)
    print('Right-aligned population diagnostics:', flush=True)
    for k, v in diag.items():
        print(f'  {k}: {v:,}', flush=True)
    n_dropped_by_eligibility = len(population) - diag['final_population']
    print(f'Population difference vs. the 365,146-loan matched population: '
          f'{n_dropped_by_eligibility:,} loans dropped by the Dec-window active/eligibility filter '
          f'({diag["final_population"]:,} scored).', flush=True)

    with open(os.path.join(K5H1_SEQ_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    sequences, masks, _l, _p, loan_ids_out, _extras = build_sequences_multiobs(
        full_df, scaler, k_draws=1, H=1, min_hist=1, obs=obs)
    verify_windows(sequences, masks, obs, dec_rows, scaler)

    fy_lo, fy_hi = 202101, 202112
    fy_window = full_df[(full_df['yyyymm'] >= fy_lo) & (full_df['yyyymm'] <= fy_hi)]
    right_active_set  = set(fy_window['loan_id'].unique().tolist())
    right_prepaid_set = set(fy_window.loc[fy_window['zero_balance_code_actual'] == 1.0, 'loan_id'].unique().tolist())
    coupon_map_right = dec_rows.set_index('loan_id').loc[loan_ids_out, 'original_interest_rate'].to_dict()

    # ── LEFT-aligned reproduction, fresh, same population, cached raw pass ──
    coupon_map, active_set, prepaid_set = cached_read_coupon_and_realized(
        CUTOFF_YEAR, population, RAW_PASS_CACHE)

    def score_both(ckpt_path, label):
        model = load_model_from_checkpoint(ckpt_path)
        # LEFT
        ids_l, h_l = score_multiobs_model(model, TRAIL_SEQ_DIR, BATCH_SIZE)
        ids_l, h_l = filter_to_population(ids_l, h_l, population)
        assert len(ids_l) == len(population), f'LEFT filter did not land on {len(population):,}'
        result_left = aggregate(ids_l, h_l, coupon_map, active_set, prepaid_set, logit_offset=0.0)
        pc_left = pooled_comparison(result_left, min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))

        # RIGHT
        h_r = score(model, sequences, masks, BATCH_SIZE)
        result_right = aggregate(loan_ids_out, h_r, coupon_map_right, right_active_set, right_prepaid_set,
                                  logit_offset=0.0)
        pc_right = pooled_comparison(result_right, min_n=MIN_N, coupon_range=(COUPON_LO, COUPON_HI))

        print(f'\n{"=" * 100}\n{label}\n{"=" * 100}')
        merged = pc_left['table'].merge(pc_right['table'], on='coupon', suffixes=('_LEFT', '_RIGHT'))
        print(merged.to_string(index=False))
        print(f'\nLEFT  dispersion {pc_left["dispersion_max"]:.3f}..{pc_left["dispersion_min"]:.3f}   '
              f'pooled_ratio {pc_left["pooled_ratio"]:.4f}')
        print(f'RIGHT dispersion {pc_right["dispersion_max"]:.3f}..{pc_right["dispersion_min"]:.3f}   '
              f'pooled_ratio {pc_right["pooled_ratio"]:.4f}')
        return model, h_l, ids_l, h_r

    model_best, h_l_best, ids_l_best, h_r_best = score_both(
        os.path.join(SEEDCHECK_DIR, 'hazard_best.pt'),
        'EPOCH 4 (hazard_best.pt) -- annualized, coupons 2.0-5.0, n>=5000')
    score_both(os.path.join(SEEDCHECK_DIR, 'hazard_final.pt'),
               'EPOCH 50 (hazard_final.pt) -- annualized, coupons 2.0-5.0, n>=5000')

    # ── Un-annualized Jan-2021-only comparison, epoch-4/hazard_best.pt only ──
    print(f'\n{"=" * 100}\nEPOCH 4 (hazard_best.pt) -- UN-ANNUALIZED, January 2021 only, coupons 2.0-5.0, n>=5000\n{"=" * 100}')
    h_tab_left = mean_h_adj_by_coupon(ids_l_best, h_l_best, coupon_map, active_set, logit_offset=0.0)
    active_1mo, prepaid_1mo = read_active_prepaid_single_month(CUTOFF_YEAR, population, TARGET_YYYYMM)
    r1_tab = realized_1mo_by_coupon(population, coupon_map, active_1mo, prepaid_1mo)
    left_1mo = h_tab_left.merge(r1_tab, on='coupon', how='inner')
    left_1mo = left_1mo[(left_1mo['coupon'] >= COUPON_LO) & (left_1mo['coupon'] <= COUPON_HI)
                         & (left_1mo['n_loans_check'] >= MIN_N)].sort_values('coupon')
    left_1mo['ratio_LEFT'] = left_1mo['h_t_mean_monthly'] / left_1mo['realized_1mo_rate']

    df_r = pd.DataFrame({'loan_id': loan_ids_out, 'h_t': h_r_best})
    df_r['note_rate'] = df_r['loan_id'].map(coupon_map_right)
    df_r['coupon'] = ((df_r['note_rate'] - 0.5) * 2).round() / 2
    right_1mo_active = active_1mo & set(loan_ids_out.tolist())
    right_1mo_prepaid = prepaid_1mo & set(loan_ids_out.tolist())
    df_r_active = df_r[df_r['loan_id'].isin(right_1mo_active)]
    h_tab_right = df_r_active.groupby('coupon')['h_t'].agg(['mean', 'count']).reset_index()
    h_tab_right.columns = ['coupon', 'h_t_mean_monthly_RIGHT', 'n_loans_check_RIGHT']
    r1_tab_right = realized_1mo_by_coupon(set(loan_ids_out.tolist()), coupon_map_right, active_1mo, prepaid_1mo)
    right_1mo = h_tab_right.merge(r1_tab_right, on='coupon', how='inner')
    right_1mo = right_1mo[(right_1mo['coupon'] >= COUPON_LO) & (right_1mo['coupon'] <= COUPON_HI)
                           & (right_1mo['n_loans_check_RIGHT'] >= MIN_N)].sort_values('coupon')
    right_1mo['ratio_RIGHT'] = right_1mo['h_t_mean_monthly_RIGHT'] / right_1mo['realized_1mo_rate']

    merged_1mo = left_1mo[['coupon', 'n_loans_check', 'h_t_mean_monthly', 'realized_1mo_rate', 'ratio_LEFT']].merge(
        right_1mo[['coupon', 'n_loans_check_RIGHT', 'h_t_mean_monthly_RIGHT', 'ratio_RIGHT']], on='coupon')
    print(merged_1mo.to_string(index=False))

    out_path = os.path.join(OUT_DIR, 'left_vs_right_comparison.csv')
    merged_1mo.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
