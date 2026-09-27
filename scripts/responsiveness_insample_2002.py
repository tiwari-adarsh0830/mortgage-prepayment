"""responsiveness_insample_2002.py -- same fixed method as
responsiveness_extras_2002.py's part_a (SE on the realized/predicted slope,
delta-method 95% CI on the predicted/realized ratio), applied to the
cutoff_2002 IN-SAMPLE one-step-ahead population (Jan2001-Nov2002, TEST-split
loans, ensemble of 5 seeds) instead of the 2003 forecast population.

For each coupon with >=5,000 loan-months: monthly points of (mean
contemporaneous incentive, realized rate, predicted rate); OLS slopes of
realized and predicted on incentive with SEs; predicted/realized slope ratio
with a 95% CI (delta method, Cov(b_pred,b_real) from the cross-product of
the two regressions' residuals); pooled predicted/realized per coupon;
n_months_used per coupon, flagged if <12.

Requires scripts/ensemble_onestep_insample_2002.py's output.

Run:
    python scripts/responsiveness_insample_2002.py
"""
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
MERGED_PATH = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_insample_cutoff_2002/ensemble_merged_all_months.csv')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_insample_cutoff_2002')
MIN_LOAN_MONTHS_PER_COUPON = 5000
MIN_MONTHS_FLAG = 12
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


def main():
    merged = pd.read_csv(MERGED_PATH)
    print(f'Loaded {len(merged):,} loan-months.', flush=True)

    coupon_counts = merged.groupby('coupon').size()
    qualifying_coupons = sorted(coupon_counts[coupon_counts >= MIN_LOAN_MONTHS_PER_COUPON].index.tolist())
    print(f'Coupons with >={MIN_LOAN_MONTHS_PER_COUPON:,} loan-months (Jan2001-Nov2002): {qualifying_coupons}',
          flush=True)

    rows = []
    pts_path = os.path.join(OUT_DIR, 'responsiveness_insample_monthly_points.csv')
    if os.path.exists(pts_path):
        os.remove(pts_path)

    for coupon in qualifying_coupons:
        d = merged[merged['coupon'] == coupon]
        n_total = len(d)
        pooled_pred = d['h'].mean()
        pooled_real = d['realized_event'].mean()

        pts = monthly_points(merged, coupon)
        real_slope, real_se, real_resid, n_months = ols_slope_se(pts['mean_incentive'], pts['realized_rate'])
        pred_slope, pred_se, pred_resid, _ = ols_slope_se(pts['mean_incentive'], pts['predicted_rate'])

        ratio = pred_slope / real_slope if real_slope != 0 else float('nan')
        cov_pred_real = float((real_resid * pred_resid).sum()) / (n_months - 2) / \
            (((pts['mean_incentive'] - pts['mean_incentive'].mean()) ** 2).sum())
        d_dpred = 1.0 / real_slope
        d_dreal = -pred_slope / (real_slope ** 2)
        var_ratio = (d_dpred ** 2) * (pred_se ** 2) + (d_dreal ** 2) * (real_se ** 2) + \
            2 * d_dpred * d_dreal * cov_pred_real
        se_ratio = np.sqrt(var_ratio) if var_ratio >= 0 else float('nan')
        ci_lo, ci_hi = ratio - Z975 * se_ratio, ratio + Z975 * se_ratio

        rows.append({
            'coupon': coupon, 'n_loan_months': n_total, 'n_months_used': n_months,
            'flag_lt_12_months': n_months < MIN_MONTHS_FLAG,
            'realized_slope': real_slope, 'realized_se': real_se,
            'predicted_slope': pred_slope, 'predicted_se': pred_se,
            'ratio_pred_over_real': ratio, 'ratio_se_delta': se_ratio,
            'ratio_ci95_lo': ci_lo, 'ratio_ci95_hi': ci_hi,
            'pooled_predicted_rate_monthly': pooled_pred, 'pooled_realized_rate_monthly': pooled_real,
            'pooled_ratio_pred_over_real': pooled_pred / pooled_real,
        })

        pts.insert(0, 'coupon', coupon)
        pts.to_csv(pts_path, mode='a', header=not os.path.exists(pts_path), index=False)

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUT_DIR, 'responsiveness_insample_by_coupon.csv'), index=False)
    print('\n=== IN-SAMPLE (Jan2001-Nov2002) RESPONSIVENESS BY COUPON ===')
    print(result.to_string(index=False))
    flagged = result[result['flag_lt_12_months']]
    if len(flagged):
        print(f'\nFLAGGED (n_months_used < {MIN_MONTHS_FLAG}): {flagged["coupon"].tolist()}')
    print(f'\nSaved: {os.path.join(OUT_DIR, "responsiveness_insample_by_coupon.csv")}')
    print(f'Saved: {pts_path}')


if __name__ == '__main__':
    main()
