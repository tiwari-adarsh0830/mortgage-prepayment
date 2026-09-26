"""
report_multiobs_dec_window.py -- metrics (a)-(d) over score_multiobs_dec_window.py's
per-run CSVs, for the cutoff_2002 primary run and the cutoff_2020 matched control.

Metrics, confirmed per the review-gate conversation:
  (a) per-coupon table: forecast_cpr/realized_cpr/n_loans, COUNT-weighted
      and UPB-weighted separately (UPB-weighted excludes blank/zero
      current_actual_upb rows -- no imputation -- and states that excluded
      share next to the table).
  (b) PRIMARY: max/min of forecast_cpr and max/min of realized_cpr
      separately (each a dispersion ratio across coupons with n_loans>=
      min_n), plus their ratio (forecast_dispersion / realized_dispersion).
      SECONDARY (labeled): max-min spread of forecast_cpr / max-min spread
      of realized_cpr -- kept for continuity with the earlier draft, not
      the benchmark-matching definition. Both computed for COUNT- and
      UPB-weighted (a) tables.
  (c) incentive-bin slope, the PRIMARY cross-cutoff comparison: bins by
      UNSCALED incentive_at_ref using the SAME edges as
      incentive_scurve_check_cutoff_2002.py ([-6,-4,-3,-2,-1,-0.5,0,0.5,1,
      1.5,2,3,4,6] -- quoted directly from that file's `bins =` line).
      Values outside [-6,6] are counted and reported explicitly (pd.cut
      leaves them NaN/dropped from the bin table -- nothing drops
      silently). ALL in-range bins are reported regardless of n; the
      PRIMARY slope uses only the highest- and lowest-incentive bins with
      n>=min_n (default 1000), same two bins for forecast and realized
      since they're the same underlying rows. A SECONDARY polyfit-based
      slope (linear fit across all qualifying bins) is also reported,
      labeled secondary.
  (d) pooled level: n_loans-weighted (count) and current_actual_upb-weighted
      (blank/zero excluded, share reported), each reported BOTH
      unweighted-by-cell and cell-reweighted (weight =
      cell_n_loans/cell_budget from outputs/pre2013_cell_sample_loans.csv,
      i.e. 1/selection-fraction, from COUNTS not a fitted model). Applies
      only to cutoff_2002 (historical-era cell-grid sampler); cutoff_2020 has
      no cell-grid sampling, so its cell-reweighted column is identical to
      unweighted by construction and reported as such, not omitted.

Usage:
    python scripts/report_multiobs_dec_window.py \\
        --cutoff_year 2002 --scores outputs/rolling/dec_window_cutoff_2002_seed42/dec_window_scores_seed42.csv \\
                                     outputs/rolling/dec_window_cutoff_2002_seed7/dec_window_scores_seed7.csv \\
        --cell_sample outputs/pre2013_cell_sample_loans.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Matches scripts/diag/incentive_scurve_check_cutoff_2002.py's bins exactly
# (same UNSCALED refi_incentive field, same edges), per review-gate decision.
INCENTIVE_EDGES = [-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]


def coupon_table(df: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    """Metric (a): per-0.5pt-coupon forecast_cpr / realized_cpr / n_loans,
    same binning convention as forecast_rolling_cpr.aggregate(). weight_col:
    None -> count-weighted (equal per loan); 'current_actual_upb' -> UPB-
    weighted, with blank/zero-UPB rows dropped first (no imputation) --
    caller reports the dropped share separately."""
    d = df.copy()
    d['coupon'] = ((d['note_rate'] - 0.5) * 2).round() / 2
    if weight_col is not None:
        d = d[d[weight_col].notna() & (d[weight_col] != 0)].copy()
        d['_w'] = d[weight_col]
    else:
        d['_w'] = 1.0
    rows = []
    for coupon, g in d.groupby('coupon'):
        rows.append({
            'coupon':       coupon,
            'forecast_cpr': round(np.average(g['annual_pp'], weights=g['_w']) * 100, 4),
            'realized_cpr': round(np.average(g['realized_prepay'], weights=g['_w']) * 100, 4),
            'n_loans':      len(g),
        })
    return pd.DataFrame(rows).sort_values('coupon').reset_index(drop=True)


def spread_ratio(table: pd.DataFrame, min_n: int) -> dict:
    """Metric (b). PRIMARY (benchmark definition): max/min of forecast_cpr
    and max/min of realized_cpr, each reported, plus their ratio.
    SECONDARY: max-min spread of each, and that ratio -- kept for
    continuity, explicitly labeled, not the primary number."""
    f = table[table['n_loans'] >= min_n]
    forecast_dispersion = f['forecast_cpr'].max() / f['forecast_cpr'].min()
    realized_dispersion = f['realized_cpr'].max() / f['realized_cpr'].min()
    f_spread = f['forecast_cpr'].max() - f['forecast_cpr'].min()
    r_spread = f['realized_cpr'].max() - f['realized_cpr'].min()
    return {
        'forecast_dispersion_max_over_min': forecast_dispersion,
        'realized_dispersion_max_over_min': realized_dispersion,
        'dispersion_ratio': forecast_dispersion / realized_dispersion if realized_dispersion else float('nan'),
        'secondary_forecast_spread_maxminus_min': f_spread,
        'secondary_realized_spread_maxminus_min': r_spread,
        'secondary_spread_ratio': f_spread / r_spread if r_spread else float('nan'),
        'n_coupons': len(f),
    }


def incentive_slope(df: pd.DataFrame, min_n: int) -> dict:
    """Metric (c), the PRIMARY cross-cutoff comparison. Bins by UNSCALED
    incentive_at_ref using INCENTIVE_EDGES (matches
    incentive_scurve_check_cutoff_2002.py). Reports EVERY bin with its n,
    regardless of min_n. slope = rate in the highest bin with n>=min_n
    divided by rate in the lowest bin with n>=min_n, computed separately
    for forecast and realized but using the SAME two bins for both (they
    are the same rows either way, so bin qualification is identical)."""
    d = df.copy()
    d['bin'] = pd.cut(d['incentive_at_ref'], bins=INCENTIVE_EDGES)
    n_outside = int(d['bin'].isna().sum())   # values < -6 or > 6 -- pd.cut leaves these NaN
    rows = []
    for b, g in d.groupby('bin', observed=True):
        rows.append({'bin': str(b), 'mid': b.mid, 'n': len(g),
                     'forecast_rate': g['annual_pp'].mean() * 100,
                     'realized_rate': g['realized_prepay'].mean() * 100})
    t = pd.DataFrame(rows).sort_values('mid').reset_index(drop=True)

    qualifying = t[t['n'] >= min_n]
    if len(qualifying) < 2:
        return {'table': t, 'n_outside_edges': n_outside, 'lowest_bin': None, 'highest_bin': None,
                'forecast_slope': float('nan'), 'realized_slope': float('nan'),
                'slope_ratio': float('nan'),
                'secondary_polyfit_forecast_slope': float('nan'),
                'secondary_polyfit_realized_slope': float('nan')}
    lowest, highest = qualifying.iloc[0], qualifying.iloc[-1]
    f_slope = highest['forecast_rate'] / lowest['forecast_rate'] if lowest['forecast_rate'] else float('nan')
    r_slope = highest['realized_rate'] / lowest['realized_rate'] if lowest['realized_rate'] else float('nan')
    poly_f = np.polyfit(qualifying['mid'], qualifying['forecast_rate'], 1)[0]
    poly_r = np.polyfit(qualifying['mid'], qualifying['realized_rate'], 1)[0]
    return {
        'table': t,   # ALL in-range bins, regardless of n
        'n_outside_edges': n_outside,
        'lowest_bin': lowest['bin'], 'lowest_n': int(lowest['n']),
        'highest_bin': highest['bin'], 'highest_n': int(highest['n']),
        'forecast_slope': f_slope, 'realized_slope': r_slope,
        'slope_ratio': f_slope / r_slope if r_slope else float('nan'),
        'secondary_polyfit_forecast_slope': float(poly_f),
        'secondary_polyfit_realized_slope': float(poly_r),
    }


def pooled_level(df: pd.DataFrame, cell_weights: pd.Series | None) -> dict:
    """Metric (d): pooled forecast/realized, count-weighted and
    UPB-weighted, each unweighted-by-cell and cell-reweighted. UPB rows that
    are blank/zero are excluded (not imputed) from the 'upb' variants only --
    the 'count' variants use the full population."""
    d = df.copy()
    if cell_weights is not None:
        missing = set(d['loan_id']) - set(cell_weights.index)
        assert not missing, (
            f'{len(missing)} scored loans have no cell weight (e.g. {list(missing)[:5]}) -- '
            f'cell_sample coverage does not match the scored population. Investigate before '
            f'trusting any cell_reweighted number; do not silently default to weight=1.')
        d['cell_w'] = d['loan_id'].map(cell_weights)
    else:
        d['cell_w'] = 1.0

    d_upb = d[d['current_actual_upb'].notna() & (d['current_actual_upb'] != 0)]

    out = {}
    for label, dd, base_w in [('count', d,     pd.Series(1.0, index=d.index)),
                               ('upb',   d_upb, d_upb['current_actual_upb'])]:
        for reweight, w in [('unweighted', base_w), ('cell_reweighted', base_w * dd['cell_w'])]:
            wt = w.to_numpy()
            out[f'{label}_{reweight}_forecast'] = float(np.average(dd['annual_pp'] * 100, weights=wt))
            out[f'{label}_{reweight}_realized']  = float(np.average(dd['realized_prepay'] * 100, weights=wt))
    out['upb_n_excluded_blank_or_zero'] = int(len(d) - len(d_upb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff_year', type=int, required=True)
    ap.add_argument('--scores', nargs='+', required=True, help='One CSV per seed.')
    ap.add_argument('--min_n', type=int, default=1000)
    ap.add_argument('--cell_sample', type=str, default=None,
                     help='outputs/pre2013_cell_sample_loans.csv -- cutoff_2002 only.')
    args = ap.parse_args()

    cell_weights = None
    if args.cell_sample:
        cs = pd.read_csv(args.cell_sample)
        cs['weight'] = cs['cell_n_loans'] / cs['cell_budget']
        cell_weights = cs.set_index('loan_id')['weight']

    for path in args.scores:
        seed_label = os.path.basename(path).replace('dec_window_scores_', '').replace('.csv', '')
        df = pd.read_csv(path)
        print(f'\n{"=" * 90}\ncutoff_{args.cutoff_year} | {seed_label} | n={len(df):,}\n{"=" * 90}')

        n_upb_bad = int((df['current_actual_upb'].isna() | (df['current_actual_upb'] == 0)).sum())
        upb_share_str = (f'{n_upb_bad:,}/{len(df):,} ({100 * n_upb_bad / len(df):.2f}%) '
                          f'blank/zero, excluded (not imputed)')

        table_count = coupon_table(df, weight_col=None)
        print('\n(a) per-coupon table, COUNT-weighted:')
        print(table_count.to_string(index=False))
        print('\n(b) spread, COUNT-weighted (PRIMARY = max/min dispersion; secondary = max-min):')
        print(spread_ratio(table_count, args.min_n))

        table_upb = coupon_table(df, weight_col='current_actual_upb')
        print(f'\n(a) per-coupon table, UPB-weighted  [UPB excluded: {upb_share_str}]:')
        print(table_upb.to_string(index=False))
        print(f'\n(b) spread, UPB-weighted  [UPB excluded: {upb_share_str}]:')
        print(spread_ratio(table_upb, args.min_n))

        print('\n(c) incentive-bin slope (PRIMARY cross-cutoff comparison):')
        isl = incentive_slope(df, args.min_n)
        print(isl['table'].to_string(index=False))
        print(f'  values outside [-6,6] (dropped from the bin table above, counted not silent): '
              f'{isl["n_outside_edges"]:,}')
        print({k: v for k, v in isl.items() if k not in ('table', 'n_outside_edges')})

        print(f'\n(d) pooled level (count/UPB x unweighted/cell-reweighted)  '
              f'[UPB excluded: {upb_share_str}]:')
        for k, v in pooled_level(df, cell_weights).items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
