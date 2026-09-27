"""responsiveness_2002.py -- is the cutoff_2002 model too low in level, or
not responsive enough to rates, within vs. beyond the incentive range it was
trained on?

Training incentive range: cutoff_2002 training set's own 1st-99th percentile
of incentive_at_ref = [-1.306, 2.967] (computed from
data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist/train_incentive_at_ref.npy).

For each coupon with >=5,000 loan-months in the 2003 one-step-ahead
population: build 12 monthly points of (mean contemporaneous incentive,
mean realized rate, mean ensemble-predicted rate), one point per forecast
month; slope of realized-on-incentive and predicted-on-incentive by OLS
across those monthly points; ratio predicted/realized slope. Done twice --
once over ALL loan-months, once restricted to WITHIN-RANGE loan-months only
(incentive_at_ref in [-1.306, 2.967]) -- the within-range version can use
fewer than 12 months if a whole month's coupon population falls outside the
range, so n_months_used is reported alongside every slope, not just the
number (a 3-point slope is not a 12-point slope).

Also: a level check (pooled predicted/realized per coupon, no slope), and
the share of each coupon's loan-months that fall outside the training range.

Requires scripts/ensemble_onestep_2002.py's output
(outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv).

Run:
    python scripts/responsiveness_2002.py
"""
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
MERGED_PATH = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002')
TRAIN_RANGE = (-1.306, 2.967)
MIN_LOAN_MONTHS_PER_COUPON = 5000


def ols_slope(x, y):
    """Returns (slope, n). n<2 -> slope=nan (undefined)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float('nan'), len(x)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope), len(x)


def monthly_points(df, coupon, restrict_range: bool):
    d = df[df['coupon'] == coupon]
    if restrict_range:
        d = d[(d['incentive_at_ref'] >= TRAIN_RANGE[0]) & (d['incentive_at_ref'] <= TRAIN_RANGE[1])]
    pts = d.groupby('ref_month').agg(
        mean_incentive=('incentive_at_ref', 'mean'),
        realized_rate=('realized_event', 'mean'),
        predicted_rate=('h', 'mean'),
        n=('h', 'size'),
    ).reset_index()
    return pts


def main():
    merged = pd.read_csv(MERGED_PATH)
    print(f'Loaded {len(merged):,} loan-months.', flush=True)

    n_outside = ((merged['incentive_at_ref'] < TRAIN_RANGE[0]) |
                 (merged['incentive_at_ref'] > TRAIN_RANGE[1])).sum()
    print(f'2003 loan-months outside training range {TRAIN_RANGE}: '
          f'{n_outside:,}/{len(merged):,} ({100*n_outside/len(merged):.2f}%)', flush=True)

    coupon_counts = merged.groupby('coupon').size()
    qualifying_coupons = sorted(coupon_counts[coupon_counts >= MIN_LOAN_MONTHS_PER_COUPON].index.tolist())
    print(f'Coupons with >={MIN_LOAN_MONTHS_PER_COUPON:,} loan-months in 2003: {qualifying_coupons}', flush=True)

    rows = []
    for coupon in qualifying_coupons:
        d = merged[merged['coupon'] == coupon]
        n_total = len(d)
        n_out = ((d['incentive_at_ref'] < TRAIN_RANGE[0]) | (d['incentive_at_ref'] > TRAIN_RANGE[1])).sum()
        share_out = n_out / n_total

        # level check (pooled, no slope)
        pooled_pred = d['h'].mean()
        pooled_real = d['realized_event'].mean()

        row = {
            'coupon': coupon, 'n_loan_months': n_total,
            'share_outside_range': share_out,
            'pooled_predicted_rate_monthly': pooled_pred,
            'pooled_realized_rate_monthly': pooled_real,
            'pooled_ratio_pred_over_real': pooled_pred / pooled_real,
        }

        for tag, restrict in [('all', False), ('within_range', True)]:
            pts = monthly_points(merged, coupon, restrict)
            real_slope, n_months = ols_slope(pts['mean_incentive'], pts['realized_rate'])
            pred_slope, _ = ols_slope(pts['mean_incentive'], pts['predicted_rate'])
            row[f'{tag}_n_months_used'] = n_months
            row[f'{tag}_realized_slope'] = real_slope
            row[f'{tag}_predicted_slope'] = pred_slope
            row[f'{tag}_slope_ratio_pred_over_real'] = (
                pred_slope / real_slope if real_slope and not np.isnan(real_slope) and real_slope != 0 else float('nan')
            )
            # keep the raw monthly points for auditability
            pts.insert(0, 'coupon', coupon)
            pts.insert(1, 'range', tag)
            rows_pts_path = os.path.join(OUT_DIR, 'responsiveness_monthly_points.csv')
            pts.to_csv(rows_pts_path, mode='a', header=not os.path.exists(rows_pts_path), index=False)

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUT_DIR, 'responsiveness_by_coupon.csv'), index=False)
    print('\n=== RESPONSIVENESS BY COUPON ===')
    print(result.to_string(index=False))
    print(f'\nSaved: {os.path.join(OUT_DIR, "responsiveness_by_coupon.csv")}')
    print(f'Saved: {os.path.join(OUT_DIR, "responsiveness_monthly_points.csv")}')


if __name__ == '__main__':
    if os.path.exists(os.path.join(OUT_DIR, 'responsiveness_monthly_points.csv')):
        os.remove(os.path.join(OUT_DIR, 'responsiveness_monthly_points.csv'))
    main()
