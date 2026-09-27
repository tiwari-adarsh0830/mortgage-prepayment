"""responsiveness_same_incentive_2002.py -- follow-up to
responsiveness_insample_2002.py: the coupon-level 2001-02 vs 2003
comparison is confounded (rates fell ~1.5pp between the two windows, so
the same coupon sits at very different incentive levels in each period).
This compares at the SAME incentive instead.

Reuses the two already-scored ensemble merges (no new model scoring):
  - outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv       (2003: Dec2002-Nov2003, 12 months)
  - outputs/rolling/ensemble_onestep_insample_cutoff_2002/ensemble_merged_all_months.csv (2001-02: Jan2001-Nov2002, 23 months)

(a) Per-coupon loan-months, n_months, and mean-incentive range covered, for
    both periods -- quantifies the confound.
(b) Per-incentive-bin (same edges as the 2003 report) pooled predicted/
    realized/ratio, both periods side by side, with a bootstrap-over-loans
    95% CI on the ratio (500 resamples, fixed seed, block bootstrap: each
    resampled loan contributes ALL its loan-months in that bin, respecting
    within-loan correlation across months).
(c) Within incentive (-0.5, 1.5]: monthly OLS slope of realized/predicted
    on mean incentive, both periods, with SEs and a ratio CI (delta
    method, same formula as responsiveness_extras_2002.py). Also reports
    which coupons supply the loan-months in that range, each period.

Run:
    python scripts/responsiveness_same_incentive_2002.py
"""
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
PATH_2003 = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv')
PATH_INSAMPLE = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_insample_cutoff_2002/ensemble_merged_all_months.csv')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_insample_cutoff_2002')

BIN_EDGES = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
N_BOOT = 500
BOOT_SEED = 20260927
Z975 = 1.959963985
SLOPE_RANGE = (-0.5, 1.5)


def load(path):
    return pd.read_csv(path)


def part_a(insample, y2003):
    def summary(df, label):
        rows = []
        for coupon, g in df.groupby('coupon'):
            monthly = g.groupby('ref_month')['incentive_at_ref'].mean()
            rows.append({
                'period': label, 'coupon': coupon, 'n_loan_months': len(g),
                'n_months': g['ref_month'].nunique(),
                'mean_incentive_min': monthly.min(), 'mean_incentive_max': monthly.max(),
            })
        return pd.DataFrame(rows)
    combo = pd.concat([summary(insample, '2001-02'), summary(y2003, '2003')]).sort_values(['coupon', 'period'])
    combo.to_csv(os.path.join(OUT_DIR, 'same_incentive_coupon_ranges.csv'), index=False)
    print('=== (a) PER-COUPON LOAN-MONTHS / MONTHS / MEAN-INCENTIVE RANGE, BOTH PERIODS ===')
    print(combo.to_string(index=False))
    return combo


def bootstrap_ratio_ci(d, n_boot, seed):
    """Block bootstrap over loans: resample loan_ids with replacement, each
    resampled loan contributes ALL its (already bin-filtered) loan-months."""
    per_loan = d.groupby('loan_id').agg(n=('h', 'size'), sum_h=('h', 'sum'), sum_real=('realized_event', 'sum'))
    n_arr, h_arr, r_arr = per_loan['n'].to_numpy(), per_loan['sum_h'].to_numpy(), per_loan['sum_real'].to_numpy()
    n_loans = len(per_loan)
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_loans, size=n_loans)
        w = np.bincount(idx, minlength=n_loans)
        n_tot = w @ n_arr
        pred = (w @ h_arr) / n_tot
        real = (w @ r_arr) / n_tot
        ratios[i] = pred / real if real != 0 else np.nan
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def part_b(insample, y2003):
    rows = []
    for label, df in [('2001-02', insample), ('2003', y2003)]:
        d = df.copy()
        d['incentive_bin'] = pd.cut(d['incentive_at_ref'], bins=BIN_EDGES)
        for b, g in d.dropna(subset=['incentive_bin']).groupby('incentive_bin', observed=True):
            n = len(g)
            pred = g['h'].mean()
            real = g['realized_event'].mean()
            ratio = pred / real if real != 0 else float('nan')
            ci_lo, ci_hi = bootstrap_ratio_ci(g, N_BOOT, BOOT_SEED) if n > 0 else (float('nan'), float('nan'))
            rows.append({
                'incentive_bin': str(b), 'period': label, 'n_loan_months': n,
                'predicted_rate_monthly': pred, 'realized_rate_monthly': real,
                'ratio_pred_over_real': ratio, 'ratio_ci95_lo': ci_lo, 'ratio_ci95_hi': ci_hi,
            })
    result = pd.DataFrame(rows)
    # order bins low-to-high by left edge, period 2001-02 then 2003 within each bin
    result['_bin_left'] = result['incentive_bin'].str.extract(r'\((-?[\d.]+),')[0].astype(float)
    result = result.sort_values(['_bin_left', 'period']).drop(columns='_bin_left')
    result.to_csv(os.path.join(OUT_DIR, 'same_incentive_bin_comparison.csv'), index=False)
    print(f'\n=== (b) PER-INCENTIVE-BIN, POOLED, BOTH PERIODS ({N_BOOT} bootstrap draws over loans, seed={BOOT_SEED}) ===')
    print(result.to_string(index=False))
    return result


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


def part_c(insample, y2003):
    rows = []
    coupon_rows = []
    for label, df in [('2001-02', insample), ('2003', y2003)]:
        d = df[(df['incentive_at_ref'] > SLOPE_RANGE[0]) & (df['incentive_at_ref'] <= SLOPE_RANGE[1])]
        pts = d.groupby('ref_month').agg(
            mean_incentive=('incentive_at_ref', 'mean'),
            realized_rate=('realized_event', 'mean'),
            predicted_rate=('h', 'mean'),
            n=('h', 'size'),
        ).reset_index().sort_values('ref_month')

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
            'period': label, 'n_months': n_months, 'n_loan_months': len(d),
            'realized_slope': real_slope, 'realized_se': real_se,
            'predicted_slope': pred_slope, 'predicted_se': pred_se,
            'ratio_pred_over_real': ratio, 'ratio_se_delta': se_ratio,
            'ratio_ci95_lo': ci_lo, 'ratio_ci95_hi': ci_hi,
        })

        by_coupon = d.groupby('coupon').size().sort_values(ascending=False)
        for coupon, n in by_coupon.items():
            coupon_rows.append({'period': label, 'coupon': coupon, 'n_loan_months': int(n)})

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUT_DIR, 'same_incentive_slope_comparison.csv'), index=False)
    print(f'\n=== (c) TIME-SERIES SLOPE WITHIN incentive in ({SLOPE_RANGE[0]}, {SLOPE_RANGE[1]}], BOTH PERIODS ===')
    print(result.to_string(index=False))

    coupon_df = pd.DataFrame(coupon_rows)
    coupon_df.to_csv(os.path.join(OUT_DIR, 'same_incentive_slope_coupon_breakdown.csv'), index=False)
    print(f'\n=== Coupons supplying loan-months in ({SLOPE_RANGE[0]}, {SLOPE_RANGE[1]}] ===')
    print(coupon_df.to_string(index=False))
    return result, coupon_df


def main():
    insample = load(PATH_INSAMPLE)
    y2003 = load(PATH_2003)
    part_a(insample, y2003)
    part_b(insample, y2003)
    part_c(insample, y2003)


if __name__ == '__main__':
    main()
