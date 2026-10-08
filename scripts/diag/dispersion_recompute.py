"""dispersion_recompute.py -- recompute per-coupon dispersion from an ensemble
one-step run's ensemble_merged_all_months.csv, with an explicit min_n and a
zero-realized guard.

Why: score_rolling_one_step.dispersion_stats takes max/min of the realized
per-coupon rate over every coupon with n_loan_months >= min_n, and
ensemble_onestep_seq.py calls it at min_n=100. For cutoffs 2004 and 2005 _seq
some thin coupons (2.5, 3.0/3.5, 10.5) passed that filter with zero realized
prepayments, so realized dispersion was max/0 = inf and the dispersion ratio
0.0 (Oct 7, 2026). This script is the authoritative dispersion number for
cutoffs 2002-2005 _seq; the scorer guard covers 2006 onward.

For each --run_dir:
  - read ensemble_merged_all_months.csv (columns printed first);
  - by coupon, over all ref_months in the file: loan-months, predicted
    events (sum of the ensemble hazard `h`), realized events (sum of
    `realized_event`), monthly rates = events / loan-months;
  - drop coupons with loan-months < --min_n, then drop (and list) coupons
    with zero realized events;
  - print the per-coupon predicted/realized ratio table, predicted
    dispersion (max/min predicted rate), realized dispersion (max/min
    realized rate), their ratio, and the number of coupons kept.

Usage:
  python scripts/diag/dispersion_recompute.py \
      --run_dir outputs/rolling/ensemble_onestep_cutoff_2003_seq \
      --run_dir outputs/rolling/ensemble_onestep_cutoff_2004_seq --min_n 1000
"""
import argparse
import os

import pandas as pd

COUPON_COL = 'coupon'
PRED_COL = 'h'                 # ensemble mean hazard (h_seed* are the per-seed columns)
REAL_COL = 'realized_event'


def recompute(run_dir: str, min_n: int) -> dict:
    path = os.path.join(run_dir, 'ensemble_merged_all_months.csv')
    cols = list(pd.read_csv(path, nrows=0).columns)
    print(f'\n######## {run_dir}  (min_n={min_n:,})')
    print(f'columns: {cols}')
    for c in (COUPON_COL, PRED_COL, REAL_COL):
        if c not in cols:
            raise SystemExit(f'column {c!r} missing from {path}')
    df = pd.read_csv(path, usecols=[COUPON_COL, PRED_COL, REAL_COL])
    print(f'rows (loan-months): {len(df):,}')

    g = df.groupby(COUPON_COL).agg(n_loan_months=(REAL_COL, 'size'),
                                   pred_events=(PRED_COL, 'sum'),
                                   real_events=(REAL_COL, 'sum')).reset_index()
    g['pred_rate'] = g['pred_events'] / g['n_loan_months']
    g['real_rate'] = g['real_events'] / g['n_loan_months']

    thin = g[g['n_loan_months'] < min_n]
    kept = g[g['n_loan_months'] >= min_n]
    zero = kept[kept['real_events'] == 0]
    kept = kept[kept['real_events'] > 0].copy()
    print(f'dropped for n_loan_months < {min_n:,}: {len(thin)} coupons '
          f'{thin[COUPON_COL].tolist()}')
    if len(zero):
        print(f'dropped for zero realized events: {len(zero)} coupons')
        print(zero[[COUPON_COL, 'n_loan_months', 'pred_events', 'real_events']].to_string(index=False))
    else:
        print('dropped for zero realized events: none')

    kept['ratio'] = kept['pred_rate'] / kept['real_rate']
    print(kept[[COUPON_COL, 'n_loan_months', 'pred_events', 'real_events', 'pred_rate',
                'real_rate', 'ratio']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    pred_disp = kept['pred_rate'].max() / kept['pred_rate'].min()
    real_disp = kept['real_rate'].max() / kept['real_rate'].min()
    out = {'run_dir': run_dir, 'min_n': min_n, 'n_coupons_kept': len(kept),
           'n_zero_realized_dropped': len(zero),
           'coupons_kept': f'{kept[COUPON_COL].min()}-{kept[COUPON_COL].max()}',
           'pred_disp': pred_disp, 'real_disp': real_disp,
           'disp_ratio': pred_disp / real_disp}
    print(f'predicted dispersion (max/min pred_rate): {pred_disp:.3f}')
    print(f'realized dispersion  (max/min real_rate): {real_disp:.3f}')
    print(f'dispersion ratio (pred/real):             {pred_disp / real_disp:.3f}')
    print(f'coupons kept: {len(kept)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', action='append', required=True)
    ap.add_argument('--min_n', type=int, default=1000)
    args = ap.parse_args()
    rows = [recompute(d, args.min_n) for d in args.run_dir]
    print(f'\n=== SUMMARY (min_n={args.min_n:,}) ===')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
