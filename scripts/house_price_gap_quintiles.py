"""house_price_gap_quintiles.py -- for a given cutoff year's one-step-ahead
ensemble forecast, computes (a) the pooled ensemble realized-minus-predicted
monthly rate gap over the low-incentive group (incentive_at_ref <= 0.5) and
the placebo group (incentive_at_ref >= 1.5) -- the SAME loan-month
population used by house_price_test_{2002,2020_control}.py, before any
per-zip3 >=100-loan-month filter -- and (b) a zip3-quintile breakdown (by
each zip3's mean trailing-12mo ZHVI growth x) of pooled predicted/realized
rates within each group, restricted to zip3s with >=100 loan-months (the
same zip3 set the WLS regression in house_price_test_*.py actually uses).

Does not re-run any model; reads ensemble_merged_all_months.csv, the zip3
map, and zhvi_zip3.csv exactly as house_price_test_*.py does.

Run:
    python scripts/house_price_gap_quintiles.py --year 2002
    python scripts/house_price_gap_quintiles.py --year 2020_control
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
LOW_INCENTIVE_MAX = 0.5
PLACEBO_MIN = 1.5
MIN_LOAN_MONTHS_PER_ZIP3 = 100
N_QUINTILES = 5


def paths_for(year):
    if year == '2002':
        return dict(
            merged=os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv'),
            zip3_map=os.path.join(BASE, 'outputs/rolling/loan_zip3_map_cutoff_2002.csv'),
            out_dir=os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2002'),
        )
    elif year == '2020_control':
        return dict(
            merged=os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2020_control/ensemble_merged_all_months.csv'),
            zip3_map=os.path.join(BASE, 'outputs/rolling/loan_zip3_map_cutoff_2020.csv'),
            out_dir=os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2020_control'),
        )
    raise ValueError(year)


def trailing_12mo_growth(zhvi_lookup, zip3, ref_month_yyyymm):
    year, month = ref_month_yyyymm // 100, ref_month_yyyymm % 100
    mmyyyy_now = month * 10000 + year
    mmyyyy_prior = month * 10000 + (year - 1)
    now = zhvi_lookup.get((zip3, mmyyyy_now))
    prior = zhvi_lookup.get((zip3, mmyyyy_prior))
    if now is None or prior is None or prior == 0:
        return None
    return 100.0 * (now / prior - 1.0)


def pooled_gap(d, wlabel, zip3_filter=None):
    """Pooled realized/predicted monthly rate and their gap, count- or
    UPB-weighted. If zip3_filter is given, restricts to that zip3 set
    (the same >=100-loan-month zip3s house_price_test_*.py's WLS regression
    actually uses) so n_loan_months matches house_price_test_results.csv
    exactly; otherwise pools over every loan-month in the incentive-defined
    group regardless of its zip3's size."""
    n_blank = 0
    if wlabel == 'upb':
        n_blank = int((d['current_actual_upb'].isna() | (d['current_actual_upb'] == 0)).sum())
        d = d[d['current_actual_upb'].notna() & (d['current_actual_upb'] != 0)]
    if zip3_filter is not None:
        d = d[d['zip3'].isin(zip3_filter)]
    if wlabel == 'upb':
        w = d['current_actual_upb'].to_numpy(float)
    else:
        w = np.ones(len(d))
    wsum = w.sum()
    pred = float((w * d['h'].to_numpy(float)).sum() / wsum)
    real = float((w * d['realized_event'].to_numpy(float)).sum() / wsum)
    return {
        'weight': wlabel, 'n_loan_months': int(len(d)), 'n_blank_upb_excluded': n_blank,
        'pooled_predicted_rate': pred, 'pooled_realized_rate': real, 'gap_realized_minus_predicted': real - pred,
    }


def zip3_table(d, wlabel):
    """Per-zip3 x/predicted/realized/n/weight, restricted to zip3s with
    >= MIN_LOAN_MONTHS_PER_ZIP3, for the quintile breakdown."""
    n_blank = 0
    if wlabel == 'upb':
        n_blank = int((d['current_actual_upb'].isna() | (d['current_actual_upb'] == 0)).sum())
        d = d[d['current_actual_upb'].notna() & (d['current_actual_upb'] != 0)]

    rows = []
    for z, g in d.groupby('zip3'):
        if len(g) < MIN_LOAN_MONTHS_PER_ZIP3:
            continue
        if wlabel == 'upb':
            w = g['current_actual_upb'].to_numpy(float)
        else:
            w = np.ones(len(g))
        wsum = w.sum()
        pred_z = float((w * g['h'].to_numpy(float)).sum() / wsum)
        real_z = float((w * g['realized_event'].to_numpy(float)).sum() / wsum)
        x_z = float(g['growth_12mo'].mean())
        rows.append({'zip3': z, 'n': len(g), 'weight': wsum, 'x': x_z, 'predicted': pred_z, 'realized': real_z})
    return pd.DataFrame(rows), n_blank


def quintile_breakdown(zdf, n_quintiles=N_QUINTILES):
    zdf = zdf.sort_values('x').reset_index(drop=True)
    zdf['quintile'] = pd.qcut(zdf['x'], n_quintiles, labels=False, duplicates='drop')
    rows = []
    for q, g in zdf.groupby('quintile'):
        w = g['weight'].to_numpy(float)
        wsum = w.sum()
        pred = float((w * g['predicted'].to_numpy(float)).sum() / wsum)
        real = float((w * g['realized'].to_numpy(float)).sum() / wsum)
        rows.append({
            'quintile': int(q) + 1, 'x_min': float(g['x'].min()), 'x_max': float(g['x'].max()),
            'n_zip3s': int(len(g)), 'n_loan_months': int(g['n'].sum()),
            'pooled_predicted_rate': pred, 'pooled_realized_rate': real,
            'gap_realized_minus_predicted': real - pred,
        })
    return pd.DataFrame(rows).sort_values('quintile').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', required=True, choices=['2002', '2020_control'])
    args = ap.parse_args()
    p = paths_for(args.year)

    merged = pd.read_csv(p['merged'])
    zip3_map = pd.read_csv(p['zip3_map'])
    zhvi = pd.read_csv(os.path.join(BASE, 'data/zhvi_zip3.csv'))
    zhvi_lookup = dict(zip(zip(zhvi['zip3'].astype(int), zhvi['reporting_period'].astype(int)), zhvi['zhvi']))

    df = merged.merge(zip3_map, on='loan_id', how='left')
    df = df.dropna(subset=['zip3']).copy()
    df['zip3'] = df['zip3'].astype(int)
    df['growth_12mo'] = df.apply(
        lambda r: trailing_12mo_growth(zhvi_lookup, r['zip3'], int(r['ref_month'])), axis=1)
    df = df.dropna(subset=['growth_12mo']).copy()

    print(f'=== YEAR {args.year} ===', flush=True)
    gap_rows = []
    quint_rows = []
    for group_name, mask in [
        ('low_incentive', df['incentive_at_ref'] <= LOW_INCENTIVE_MAX),
        ('placebo', df['incentive_at_ref'] >= PLACEBO_MIN),
    ]:
        d = df[mask]
        print(f'\n--- {group_name}: {len(d):,} loan-months (raw incentive filter, any zip3 size) ---', flush=True)
        for wlabel in ['count', 'upb']:
            zdf, n_blank = zip3_table(d, wlabel)
            zip3_filter = set(zdf['zip3'])

            g_raw = pooled_gap(d, wlabel)
            g_raw['group'], g_raw['scope'] = group_name, 'raw_incentive_filter'
            gap_rows.append(g_raw)
            g_reg = pooled_gap(d, wlabel, zip3_filter=zip3_filter)
            g_reg['group'], g_reg['scope'] = group_name, 'regression_eligible_zip3s'
            gap_rows.append(g_reg)
            print(f'  [{wlabel}] RAW (all zip3 sizes):        pooled_predicted={g_raw["pooled_predicted_rate"]:.6f} '
                  f'pooled_realized={g_raw["pooled_realized_rate"]:.6f} gap={g_raw["gap_realized_minus_predicted"]:.6f} '
                  f'(n={g_raw["n_loan_months"]:,})', flush=True)
            print(f'  [{wlabel}] REGRESSION-ELIGIBLE (>=100/zip3): pooled_predicted={g_reg["pooled_predicted_rate"]:.6f} '
                  f'pooled_realized={g_reg["pooled_realized_rate"]:.6f} gap={g_reg["gap_realized_minus_predicted"]:.6f} '
                  f'(n={g_reg["n_loan_months"]:,}, matches house_price_test_results.csv)', flush=True)

            qdf = quintile_breakdown(zdf)
            qdf.insert(0, 'weight', wlabel)
            qdf.insert(0, 'group', group_name)
            quint_rows.append(qdf)
            print(f'  [{wlabel}] quintile table ({len(zdf)} zip3s used):')
            print(qdf.to_string(index=False), flush=True)

    gap_df = pd.DataFrame(gap_rows)
    quint_df = pd.concat(quint_rows, ignore_index=True)
    gap_out = os.path.join(p['out_dir'], 'house_price_pooled_gap.csv')
    quint_out = os.path.join(p['out_dir'], 'house_price_zip3_quintiles.csv')
    gap_df.to_csv(gap_out, index=False)
    quint_df.to_csv(quint_out, index=False)
    print(f'\nSaved: {gap_out}')
    print(f'Saved: {quint_out}')


if __name__ == '__main__':
    main()
