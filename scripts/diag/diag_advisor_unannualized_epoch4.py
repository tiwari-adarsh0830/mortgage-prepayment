"""
diag_advisor_unannualized_epoch4.py -- un-annualized coupon-level comparison
for the epoch-4 seedcheck_a checkpoint, requested by the advisor.

Question: does the 4.5-5.0 coupon under-forecast seen in the ANNUALIZED
comparison (rolling_cpr_forecast_matched_best.csv: forecast_cpr < realized_cpr
at 4.5/5.0) persist un-annualized, or is it purely an artifact of
1-(1-h)^12 saturation?

Compares, per coupon, on the SAME 365,146-loan matched population used
throughout this investigation (intersection of origination/trailing/multiobs
test sets):
  h_t           = mean monthly hazard from hazard_best.pt (epoch 4,
                  cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a), scored via
                  score_multiobs_model on the TRAILING test sequences
                  (mean_h_adj_by_coupon, logit_offset=0.0 -- unchanged from
                  forecast_multiobs_ipw_seedcheck_epoch_compare.py).
  realized_1mo  = prepayments in January 2021 (the single calendar month
                  immediately following the Dec-2020 cutoff, matching the
                  model's H=1 one-month-ahead horizon) / loans active that
                  month, same population. NOT the annual realized_cpr.

Does not touch forecast_rolling_cpr.py, forecast_matched_population_cpr.py,
or any existing output file. Writes one new CSV next to the existing
matched_best.csv.

Usage:
    python scripts/diag/diag_advisor_unannualized_epoch4.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast_rolling_cpr import (
    DATA_DIR, _ALL_COLS, ALL_VINTAGES, mmyyyy_to_yyyymm, DEVICE, BASE,
)
from forecast_matched_population_cpr import (
    CUTOFF_YEAR, BATCH_SIZE, TRAIL_SEQ_DIR, RAW_PASS_CACHE,
    score_multiobs_model, filter_to_population, cached_read_coupon_and_realized,
    mean_h_adj_by_coupon,
)
from forecast_multiobs_ipw_seedcheck_epoch_compare import (
    SEEDCHECK_DIR, EXPECTED_POPULATION_N, load_model_from_checkpoint,
)

COUPON_LO, COUPON_HI, MIN_N = 2.0, 5.0, 5000
TARGET_YYYYMM = 202101  # January 2021 -- the single month immediately after
                         # the Dec-2020 cutoff, matching the model's H=1 horizon.


def read_active_prepaid_single_month(cutoff_year: int, test_id_set: set, target_yyyymm: int):
    """Mirrors forecast_rolling_cpr.read_coupon_and_realized()'s raw-pass
    structure EXACTLY, but restricts active/prepaid membership to ONE
    calendar month instead of the whole forecast year. Not a reuse of that
    function because its ym_start/ym_end are hardcoded to the full FY;
    copied here rather than parameterizing the original (which the advisor's
    task instructions say not to touch)."""
    active_set  = set()
    prepaid_set = set()

    usecols = dict(sorted({
        _ALL_COLS.index('loan_id') + 1: 'loan_id',
        _ALL_COLS.index('monthly_reporting_period') + 1: 'monthly_reporting_period',
        43: 'zero_balance_code_actual',
    }.items()))

    relevant = [v for v in ALL_VINTAGES if int(v[:4]) <= cutoff_year]
    print(f'Raw pass over {len(relevant)} vintages for single month {target_yyyymm}...', flush=True)

    for vintage in relevant:
        path = os.path.join(DATA_DIR, f'{vintage}.csv')
        if not os.path.exists(path):
            continue
        for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=list(usecols.keys()), low_memory=False, chunksize=1_000_000,
        ):
            chunk.columns = list(usecols.values())
            chunk = chunk[chunk['loan_id'].isin(test_id_set)]
            if chunk.empty:
                del chunk; continue
            chunk['monthly_reporting_period'] = pd.to_numeric(
                chunk['monthly_reporting_period'], errors='coerce')
            chunk = chunk[chunk['monthly_reporting_period'].notna()]
            chunk['yyyymm'] = chunk['monthly_reporting_period'].astype(np.int64).map(
                mmyyyy_to_yyyymm)
            chunk = chunk[chunk['yyyymm'] == target_yyyymm]
            if chunk.empty:
                del chunk; continue
            chunk['zero_balance_code_actual'] = pd.to_numeric(
                chunk['zero_balance_code_actual'], errors='coerce')
            active_set.update(chunk['loan_id'].tolist())
            prepaid_set.update(
                chunk.loc[chunk['zero_balance_code_actual'] == 1.0, 'loan_id'].tolist())
            del chunk
        print(f'  {vintage}: active={len(active_set):,} prepaid={len(prepaid_set):,}', flush=True)

    print(f'Done. active={len(active_set):,} prepaid={len(prepaid_set):,}', flush=True)
    return active_set, prepaid_set


def realized_1mo_by_coupon(population, coupon_map, active_1mo, prepaid_1mo):
    df = pd.DataFrame({'loan_id': list(population)})
    df = df[df['loan_id'].isin(active_1mo)].copy()
    df['note_rate'] = df['loan_id'].map(coupon_map)
    df = df.dropna(subset=['note_rate'])
    df['coupon']   = ((df['note_rate'] - 0.5) * 2).round() / 2
    df['prepaid_1mo'] = df['loan_id'].isin(prepaid_1mo).astype(int)
    out = df.groupby('coupon').agg(
        n_active_1mo=('loan_id', 'size'),
        n_prepaid_1mo=('prepaid_1mo', 'sum'),
    ).reset_index()
    out['realized_1mo_rate'] = out['n_prepaid_1mo'] / out['n_active_1mo']
    return out


def main():
    print(f'Device: {DEVICE}', flush=True)

    orig_ids_raw  = np.load(os.path.join(BASE, f'data/sequences_rolling/cutoff_{CUTOFF_YEAR}_zbc/test_loan_ids.npy'), allow_pickle=True)
    trail_ids_raw = np.load(os.path.join(TRAIL_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    multi_ids_raw = np.load(os.path.join(BASE, f'data/sequences_rolling/cutoff_{CUTOFF_YEAR}_zbc_multiobs_k5_h1/test_loan_ids.npy'), allow_pickle=True)
    population = set(orig_ids_raw.tolist()) & set(trail_ids_raw.tolist()) & set(multi_ids_raw.tolist())
    print(f'Matched population: {len(population):,} (expected {EXPECTED_POPULATION_N:,})', flush=True)
    assert len(population) == EXPECTED_POPULATION_N

    coupon_map, active_set, prepaid_set = cached_read_coupon_and_realized(
        CUTOFF_YEAR, population, RAW_PASS_CACHE)

    print(f'\n[1/2] Scoring epoch-4 (hazard_best.pt) on trailing test sequences...', flush=True)
    best_path = os.path.join(SEEDCHECK_DIR, 'hazard_best.pt')
    model = load_model_from_checkpoint(best_path)
    ids, h = score_multiobs_model(model, TRAIL_SEQ_DIR, BATCH_SIZE)
    ids, h = filter_to_population(ids, h, population)
    assert len(ids) == len(population)

    h_tab = mean_h_adj_by_coupon(ids, h, coupon_map, active_set, logit_offset=0.0)

    print(f'\n[2/2] Raw pass for single-month ({TARGET_YYYYMM}) active/prepaid...', flush=True)
    active_1mo, prepaid_1mo = read_active_prepaid_single_month(CUTOFF_YEAR, population, TARGET_YYYYMM)
    r1_tab = realized_1mo_by_coupon(population, coupon_map, active_1mo, prepaid_1mo)

    full = h_tab.merge(r1_tab, on='coupon', how='inner')
    full = full[(full['coupon'] >= COUPON_LO) & (full['coupon'] <= COUPON_HI)
                & (full['n_loans_check'] >= MIN_N)].sort_values('coupon')
    full['ratio_h_over_realized1mo'] = full['h_t_mean_monthly'] / full['realized_1mo_rate']

    out_path = os.path.join(SEEDCHECK_DIR, f'unannualized_comparison_epoch4_{TARGET_YYYYMM}.csv')
    full.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    print(f"\n{'coupon':>7}{'n_loans':>10}{'n_active_1mo':>14}{'n_prepaid_1mo':>15}"
          f"{'mean_h_t':>11}{'realized_1mo':>14}{'ratio':>9}")
    for _, row in full.iterrows():
        print(f"{row['coupon']:>7.1f}{int(row['n_loans_check']):>10,}{int(row['n_active_1mo']):>14,}"
              f"{int(row['n_prepaid_1mo']):>15,}{row['h_t_mean_monthly']:>11.5f}"
              f"{row['realized_1mo_rate']:>14.5f}{row['ratio_h_over_realized1mo']:>9.4f}")

    print('\nDone.')


if __name__ == '__main__':
    main()
