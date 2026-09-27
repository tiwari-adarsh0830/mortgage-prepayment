"""responsiveness_extras_2020_control.py -- follow-ups on
responsiveness_2020_control.py, mirroring responsiveness_extras_2002.py:
(a) SE on each coupon's 12-point realized/predicted slope, and the
    predicted/realized ratio's approximate 95% CI via the DELTA METHOD --
    Var(b_pred), Var(b_real) from standard OLS SE formulas, Cov(b_pred,
    b_real) estimated from the cross-product of the two regressions'
    month-by-month residuals (both regressions share the same x/months, so
    this captures within-month comovement of the two error series; it is a
    plug-in/empirical covariance from 12 points, not a large-sample
    guarantee).
(b) 1-month-LAG version: month t's realized/predicted rate regressed on
    month (t-1)'s mean incentive, same coupon. 11 usable months (Jan2021-
    Nov2021, paired with Dec2020-Oct2021 incentive) instead of 12.
(c) In-the-money crossover, Dec2020 vs Jun2021: share of loan-months with
    incentive_at_ref <= 0. DEVIATION FROM responsiveness_extras_2002.py:
    that script hardcoded a manually-picked coupon subset ([5.0..7.0], the
    coupons "near the money" for the 2002-03 rate environment). Rather than
    re-guess an analogous subset for the very different 2020-21 rate
    environment (which would be a new judgment call, not a mirror), this
    version reuses the SAME qualifying_coupons list (>=5,000 loan-months)
    that responsiveness_2020_control.py's main table already establishes --
    i.e. every coupon covered by that table, not a hand-picked slice of it.

Run:
    python scripts/responsiveness_extras_2020_control.py
"""
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
MERGED_PATH = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2020_control/ensemble_merged_all_months.csv')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2020_control')
MIN_LOAN_MONTHS_PER_COUPON = 5000
Z975 = 1.959963985


def ols_slope_se(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    xbar = x.mean()
    sxx = ((x - xbar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    dof = n - 2
    sigma2 = (resid ** 2).sum() / dof if dof > 0 else float('nan')
    se = np.sqrt(sigma2 / sxx) if dof > 0 and sxx > 0 else float('nan')
    return float(slope), float(se), resid, n


def monthly_points(df, coupon):
    d = df[df['coupon'] == coupon]
    pts = d.groupby('ref_month').agg(
        mean_incentive=('incentive_at_ref', 'mean'),
        realized_rate=('realized_event', 'mean'),
        predicted_rate=('h', 'mean'),
        n=('h', 'size'),
    ).reset_index().sort_values('ref_month')
    return pts


def part_a(merged, qualifying_coupons):
    rows = []
    for coupon in qualifying_coupons:
        pts = monthly_points(merged, coupon)
        real_slope, real_se, real_resid, n = ols_slope_se(pts['mean_incentive'], pts['realized_rate'])
        pred_slope, pred_se, pred_resid, _ = ols_slope_se(pts['mean_incentive'], pts['predicted_rate'])

        ratio = pred_slope / real_slope if real_slope != 0 else float('nan')
        # delta method: Cov(b_pred, b_real) via cross-product of the two
        # regressions' residuals (same x/months for both).
        cov_pred_real = float((real_resid * pred_resid).sum()) / (n - 2) / \
            (((pts['mean_incentive'] - pts['mean_incentive'].mean()) ** 2).sum())
        d_dpred = 1.0 / real_slope
        d_dreal = -pred_slope / (real_slope ** 2)
        var_ratio = (d_dpred ** 2) * (pred_se ** 2) + (d_dreal ** 2) * (real_se ** 2) + \
            2 * d_dpred * d_dreal * cov_pred_real
        se_ratio = np.sqrt(var_ratio) if var_ratio >= 0 else float('nan')
        ci_lo, ci_hi = ratio - Z975 * se_ratio, ratio + Z975 * se_ratio

        rows.append({
            'coupon': coupon, 'n_months': n,
            'realized_slope': real_slope, 'realized_se': real_se,
            'predicted_slope': pred_slope, 'predicted_se': pred_se,
            'ratio_pred_over_real': ratio, 'ratio_se_delta': se_ratio,
            'ratio_ci95_lo': ci_lo, 'ratio_ci95_hi': ci_hi,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'responsiveness_slope_se_ci.csv'), index=False)
    print('\n=== 3a: SLOPE SE + RATIO 95% CI (delta method) ===')
    print(df.to_string(index=False))
    return df


def part_b(merged, qualifying_coupons):
    rows = []
    for coupon in qualifying_coupons:
        pts = monthly_points(merged, coupon).sort_values('ref_month').reset_index(drop=True)
        # no-lag (contemporaneous), for side-by-side comparison
        real_slope0, _, _, n0 = ols_slope_se(pts['mean_incentive'], pts['realized_rate'])
        pred_slope0, _, _, _ = ols_slope_se(pts['mean_incentive'], pts['predicted_rate'])
        ratio0 = pred_slope0 / real_slope0 if real_slope0 != 0 else float('nan')

        # lag-1: month t's rate vs month (t-1)'s mean_incentive
        lag_incentive = pts['mean_incentive'].to_numpy()[:-1]
        real_t = pts['realized_rate'].to_numpy()[1:]
        pred_t = pts['predicted_rate'].to_numpy()[1:]
        real_slope1, real_se1, _, n1 = ols_slope_se(lag_incentive, real_t)
        pred_slope1, pred_se1, _, _ = ols_slope_se(lag_incentive, pred_t)
        ratio1 = pred_slope1 / real_slope1 if real_slope1 != 0 else float('nan')

        rows.append({
            'coupon': coupon,
            'no_lag_n_months': n0, 'no_lag_realized_slope': real_slope0,
            'no_lag_predicted_slope': pred_slope0, 'no_lag_ratio': ratio0,
            'lag1_n_months': n1, 'lag1_realized_slope': real_slope1, 'lag1_realized_se': real_se1,
            'lag1_predicted_slope': pred_slope1, 'lag1_predicted_se': pred_se1, 'lag1_ratio': ratio1,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'responsiveness_lag_check.csv'), index=False)
    print('\n=== 3b: 1-MONTH LAG CHECK (ratio with vs. without lag) ===')
    print(df.to_string(index=False))
    return df


def part_c(merged, qualifying_coupons):
    rows = []
    for coupon in qualifying_coupons:
        d = merged[merged['coupon'] == coupon]
        dec = d[d['ref_month'] == 202012]
        jun = d[d['ref_month'] == 202106]
        share_dec = (dec['incentive_at_ref'] <= 0).mean() if len(dec) else float('nan')
        share_jun = (jun['incentive_at_ref'] <= 0).mean() if len(jun) else float('nan')
        rows.append({
            'coupon': coupon,
            'n_dec2020': len(dec), 'share_itm_dec2020': share_dec,
            'n_jun2021': len(jun), 'share_itm_jun2021': share_jun,
            'delta_share': share_jun - share_dec if len(dec) and len(jun) else float('nan'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'responsiveness_itm_crossover.csv'), index=False)
    print('\n=== 3c: SHARE WITH incentive <= 0 (in-the-money), Dec2020 vs Jun2021 ===')
    print(df.to_string(index=False))
    return df


def main():
    merged = pd.read_csv(MERGED_PATH)
    coupon_counts = merged.groupby('coupon').size()
    qualifying_coupons = sorted(coupon_counts[coupon_counts >= MIN_LOAN_MONTHS_PER_COUPON].index.tolist())
    print(f'Qualifying coupons (>= {MIN_LOAN_MONTHS_PER_COUPON:,} loan-months): {qualifying_coupons}')

    part_a(merged, qualifying_coupons)
    part_b(merged, qualifying_coupons)
    part_c(merged, qualifying_coupons)


if __name__ == '__main__':
    main()
