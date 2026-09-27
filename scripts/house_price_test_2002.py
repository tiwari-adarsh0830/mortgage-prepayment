"""house_price_test_2002.py -- does the cutoff_2002 one-step-ahead forecast
error line up with zip3 house-price growth over the prior 12 months?

Low-incentive loan-months: contemporaneous incentive_at_ref <= 0.5.
Placebo loan-months (expect no relationship): incentive_at_ref >= 1.5.

For each zip3 with >=100 loan-months in the group:
    error_i = realized_event_i - h_i (ensemble-averaged h), per loan-month
    y_z     = mean(error_i) over zip3 z's loan-months in the group
              (count version: simple mean == count-weighted; UPB version:
              UPB-weighted mean, blank/zero UPB rows excluded and counted)
    x_z     = mean, over zip3 z's loan-months, of that loan-month's own
              trailing-12-month ZHVI growth for zip3 z (ZHVI(ref_month) /
              ZHVI(ref_month - 12mo) - 1, in percent)

Statistic: weighted least squares slope of y_z on x_z across zip3s (weights
= loan-month count for the primary count-weighted version, weights = total
zip3 UPB for the secondary UPB-weighted version), with the slope's standard
error, plus the weighted correlation. Reports n_zip3s and n_loan_months used
in each group.

ZHVI DATE KEY WARNING: zhvi_zip3.csv's reporting_period is MMYYYY
(month*10000+year, e.g. Jan 2003 -> 12003), matching load_pmms()'s
convention, NOT YYYYMM -- this repo's own mistakes_and_lessons.md documents
this exact format tripping up code before. ref_month here is YYYYMM.
12-months-prior of (year, month) is ALWAYS (year-1, month) -- no month
rollover needed -- so mmyyyy_prior = month*10000 + (year-1), never computed
via arithmetic on the YYYYMM or MMYYYY integers directly.

Requires:
    outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv
    outputs/rolling/loan_zip3_map_cutoff_2002.csv
    data/zhvi_zip3.csv

Run:
    python scripts/house_price_test_2002.py
"""
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
MERGED_PATH = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv')
ZIP3_MAP_PATH = os.path.join(BASE, 'outputs/rolling/loan_zip3_map_cutoff_2002.csv')
ZHVI_PATH = os.path.join(BASE, 'data/zhvi_zip3.csv')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002')

LOW_INCENTIVE_MAX = 0.5
PLACEBO_MIN = 1.5
MIN_LOAN_MONTHS_PER_ZIP3 = 100


def yyyymm_to_mmyyyy(yyyymm: int) -> int:
    year, month = yyyymm // 100, yyyymm % 100
    return month * 10000 + year


def trailing_12mo_growth(zhvi_lookup: dict, zip3: int, ref_month_yyyymm: int):
    year, month = ref_month_yyyymm // 100, ref_month_yyyymm % 100
    mmyyyy_now = month * 10000 + year
    mmyyyy_prior = month * 10000 + (year - 1)   # 12 months prior of (Y, M) is always (Y-1, M)
    now = zhvi_lookup.get((zip3, mmyyyy_now))
    prior = zhvi_lookup.get((zip3, mmyyyy_prior))
    if now is None or prior is None or prior == 0:
        return None
    return 100.0 * (now / prior - 1.0)


def wls_fit(x, y, w):
    """Weighted least squares y = a + b*x. Returns (slope, se, weighted_corr, n)."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    n = len(x)
    if n < 3:
        return float('nan'), float('nan'), float('nan'), n
    wsum = w.sum()
    xbar, ybar = (w * x).sum() / wsum, (w * y).sum() / wsum
    sxx = (w * (x - xbar) ** 2).sum()
    syy = (w * (y - ybar) ** 2).sum()
    sxy = (w * (x - xbar) * (y - ybar)).sum()
    if sxx == 0 or syy == 0:
        return float('nan'), float('nan'), float('nan'), n
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    dof = n - 2
    sigma2 = (w * resid ** 2).sum() / dof if dof > 0 else float('nan')
    se = np.sqrt(sigma2 / sxx) if dof > 0 else float('nan')
    corr = sxy / np.sqrt(sxx * syy)
    return float(slope), float(se), float(corr), n


def run_group(df_group, zhvi_lookup, group_name):
    results = {}
    for wlabel in ['count', 'upb']:
        d = df_group.copy()
        n_blank = 0
        if wlabel == 'upb':
            n_blank = int((d['current_actual_upb'].isna() | (d['current_actual_upb'] == 0)).sum())
            d = d[d['current_actual_upb'].notna() & (d['current_actual_upb'] != 0)]

        zip3_stats = []
        for z, g in d.groupby('zip3'):
            if len(g) < MIN_LOAN_MONTHS_PER_ZIP3:
                continue
            if wlabel == 'count':
                y_z = g['error'].mean()
                weight_z = len(g)
            else:
                w = g['current_actual_upb']
                y_z = (g['error'] * w).sum() / w.sum()
                weight_z = w.sum()
            x_z = g['growth_12mo'].mean()
            zip3_stats.append({'zip3': z, 'n': len(g), 'x': x_z, 'y': y_z, 'weight': weight_z})

        zdf = pd.DataFrame(zip3_stats)
        slope, se, corr, n_zip3 = wls_fit(zdf['x'], zdf['y'], zdf['weight']) if len(zdf) else (float('nan'),) * 3 + (0,)
        results[wlabel] = {
            'group': group_name, 'weight': wlabel,
            'n_zip3s': n_zip3,
            'n_loan_months': int(zdf['n'].sum()) if len(zdf) else 0,
            'n_blank_upb_excluded': n_blank,
            'wls_slope': slope, 'wls_se': se, 'weighted_corr': corr,
        }
        zdf.to_csv(os.path.join(OUT_DIR, f'house_price_zip3_{group_name}_{wlabel}.csv'), index=False)
    return results


def main():
    merged = pd.read_csv(MERGED_PATH)
    zip3_map = pd.read_csv(ZIP3_MAP_PATH)
    zhvi = pd.read_csv(ZHVI_PATH)
    zhvi_lookup = dict(zip(zip(zhvi['zip3'].astype(int), zhvi['reporting_period'].astype(int)), zhvi['zhvi']))

    df = merged.merge(zip3_map, on='loan_id', how='left')
    n_no_zip3 = int(df['zip3'].isna().sum())
    print(f'Loan-months with no zip3 (test loan not in zip3 map -- excluded): {n_no_zip3:,}/{len(df):,}', flush=True)
    df = df.dropna(subset=['zip3']).copy()
    df['zip3'] = df['zip3'].astype(int)

    df['error'] = df['realized_event'] - df['h']
    df['growth_12mo'] = df.apply(
        lambda r: trailing_12mo_growth(zhvi_lookup, r['zip3'], int(r['ref_month'])), axis=1)
    n_no_growth = int(df['growth_12mo'].isna().sum())
    print(f'Loan-months with no computable 12mo ZHVI growth (missing zip3/period -- excluded): '
          f'{n_no_growth:,}/{len(df):,}', flush=True)
    df = df.dropna(subset=['growth_12mo']).copy()

    all_results = []
    low = df[df['incentive_at_ref'] <= LOW_INCENTIVE_MAX]
    print(f'\nLow-incentive group (incentive_at_ref <= {LOW_INCENTIVE_MAX}): {len(low):,} loan-months', flush=True)
    all_results += list(run_group(low, zhvi_lookup, 'low_incentive').values())

    placebo = df[df['incentive_at_ref'] >= PLACEBO_MIN]
    print(f'Placebo group (incentive_at_ref >= {PLACEBO_MIN}): {len(placebo):,} loan-months', flush=True)
    all_results += list(run_group(placebo, zhvi_lookup, 'placebo').values())

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(os.path.join(OUT_DIR, 'house_price_test_results.csv'), index=False)
    print('\n=== HOUSE-PRICE TEST RESULTS ===')
    print(result_df.to_string(index=False))
    print(f'\nSaved: {os.path.join(OUT_DIR, "house_price_test_results.csv")}')


if __name__ == '__main__':
    main()
