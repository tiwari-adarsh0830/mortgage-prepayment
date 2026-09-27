"""ensemble_onestep_2020_control.py -- five-seed ensemble of the cutoff_2020
(MATCHED CONTROL) one-step-ahead 2021 forecast. Mirrors ensemble_onestep_2002.py
exactly; only SEED_PATH_TAG/OUT_DIR differ (rolling_onestep_cutoff_2020_control_seed{s}
instead of rolling_onestep_cutoff_2002_seed{s} -- all 5 seeds here are the
GOLDEN_BACKUP-consistent checkpoints: seed42/7/123 trained pre-fix originally,
seed1001/2026 RETRAINED pre-fix via explicit --seq_dir GOLDEN_BACKUP after the
POSTFIX_MISMATCH was found and fixed 2026-09-26 -- see config/ensemble_seeds.json).

Averages raw monthly h (NOT annualized/transformed) per loan-month across
seeds 42, 7, 123, 1001, 2026, then reports the same statistics (pooled,
per-coupon, per-incentive-bin, dispersion, month-by-month) for the ensemble
AND for each individual seed.

BEFORE averaging, asserts the five seeds' (loan_id, ref_month) populations
are IDENTICAL (eligibility/forward-adjacency is model-independent, so they
should be -- checked, not assumed).

DISAGREEMENT is the spread of each seed's OWN ratio across the five
individual seeds (min/max/spread of 5 numbers), not any per-loan-month
spread -- kept as a distinct computation from the ensemble-averaged column's
own ratios.

Run:
    python scripts/ensemble_onestep_2020_control.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_rolling_one_step import rate_table, dispersion_stats, INCENTIVE_EDGES

BASE = '/scratch/at7095/mortgage_prepayment'
SEEDS = [42, 7, 123, 1001, 2026]
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2020_control')
os.makedirs(OUT_DIR, exist_ok=True)


def load_seed(seed):
    path = os.path.join(BASE, f'outputs/rolling/rolling_onestep_cutoff_2020_control_seed{seed}/rolling_all_months.csv')
    df = pd.read_csv(path)
    df['coupon'] = ((df['note_rate'] - 0.5) * 2).round() / 2
    return df


def pooled_stats(df, label):
    rows = []
    for wlabel, wcol in [('count', None), ('upb', 'current_actual_upb')]:
        d = df if wcol is None else df[df[wcol].notna() & (df[wcol] != 0)]
        w = pd.Series(1.0, index=d.index) if wcol is None else d[wcol]
        pred = float((d['h'] * w).sum() / w.sum())
        real = float((d['realized_event'] * w).sum() / w.sum())
        rows.append({
            'model': label, 'weight': wlabel,
            'predicted_rate_monthly': pred, 'realized_rate_monthly': real,
            'predicted_rate_annualized': 1 - (1 - pred) ** 12,
            'realized_rate_annualized': 1 - (1 - real) ** 12,
            'ratio': pred / real,
        })
    return rows


def main():
    seed_dfs = {s: load_seed(s) for s in SEEDS}

    # --- population identity check across all five seeds ---
    keys = {s: set(zip(d['loan_id'], d['ref_month'])) for s, d in seed_dfs.items()}
    base_key = keys[SEEDS[0]]
    for s in SEEDS[1:]:
        assert keys[s] == base_key, (
            f'Seed {s} population differs from seed {SEEDS[0]}: '
            f'{len(keys[s] - base_key)} extra, {len(base_key - keys[s])} missing -- STOP.')
    print(f'Population identity check PASSED: all 5 seeds share the same '
          f'{len(base_key):,} (loan_id, ref_month) pairs.', flush=True)

    # --- merge on (loan_id, ref_month), average h ---
    merged = seed_dfs[SEEDS[0]][['loan_id', 'ref_month', 'note_rate', 'incentive_at_ref',
                                  'current_actual_upb', 'realized_event', 'coupon']].copy()
    h_cols = []
    for s in SEEDS:
        col = f'h_seed{s}'
        merged[col] = seed_dfs[s].set_index(['loan_id', 'ref_month']).loc[
            list(zip(merged['loan_id'], merged['ref_month'])), 'h'].to_numpy()
        h_cols.append(col)
    merged['h'] = merged[h_cols].mean(axis=1)
    merged.to_csv(os.path.join(OUT_DIR, 'ensemble_merged_all_months.csv'), index=False)
    print(f'Merged {len(merged):,} loan-months, {len(h_cols)} seed columns + ensemble mean.', flush=True)

    # --- pooled: ensemble + each seed ---
    all_pooled = []
    all_pooled += pooled_stats(merged, 'ensemble')
    for s in SEEDS:
        d = merged.copy(); d['h'] = merged[f'h_seed{s}']
        all_pooled += pooled_stats(d, f'seed{s}')
    pooled_df = pd.DataFrame(all_pooled)
    pooled_df.to_csv(os.path.join(OUT_DIR, 'pooled_stats.csv'), index=False)
    print('\n=== POOLED (ensemble + per-seed) ===')
    print(pooled_df.to_string(index=False))

    # --- disagreement across seeds: pooled ratio ---
    disagree_rows = []
    for wlabel in ['count', 'upb']:
        seed_ratios = pooled_df[(pooled_df['model'] != 'ensemble') & (pooled_df['weight'] == wlabel)]['ratio']
        disagree_rows.append({
            'stat': 'pooled_ratio', 'weight': wlabel,
            'min': seed_ratios.min(), 'max': seed_ratios.max(), 'spread': seed_ratios.max() - seed_ratios.min(),
        })

    # --- per-coupon: ensemble + each seed, count-weighted (min_n relaxed since 2021 data is smaller) ---
    coupon_min_n = 100
    coupon_tables = {'ensemble': rate_table(merged, 'coupon', None, coupon_min_n)}
    for s in SEEDS:
        d = merged.copy(); d['h'] = merged[f'h_seed{s}']
        coupon_tables[f'seed{s}'] = rate_table(d, 'coupon', None, coupon_min_n)
    print('\n=== PER-COUPON (ensemble), count-weighted, min_n=100 ===')
    print(coupon_tables['ensemble'].to_string(index=False))

    coupon_ratio_wide = coupon_tables['ensemble'][['coupon', 'n_loan_months']].copy()
    for s in SEEDS:
        t = coupon_tables[f'seed{s}'].set_index('coupon')['predicted_rate_monthly'] / \
            coupon_tables[f'seed{s}'].set_index('coupon')['realized_rate_monthly']
        coupon_ratio_wide[f'ratio_seed{s}'] = coupon_ratio_wide['coupon'].map(t)
    coupon_ratio_wide['ensemble_ratio'] = (coupon_tables['ensemble'].set_index('coupon')['predicted_rate_monthly'] /
                                            coupon_tables['ensemble'].set_index('coupon')['realized_rate_monthly']).values
    seed_ratio_cols = [f'ratio_seed{s}' for s in SEEDS]
    coupon_ratio_wide['min'] = coupon_ratio_wide[seed_ratio_cols].min(axis=1)
    coupon_ratio_wide['max'] = coupon_ratio_wide[seed_ratio_cols].max(axis=1)
    coupon_ratio_wide['spread'] = coupon_ratio_wide['max'] - coupon_ratio_wide['min']
    coupon_ratio_wide.to_csv(os.path.join(OUT_DIR, 'per_coupon_ratio_disagreement.csv'), index=False)
    print('\n=== PER-COUPON ratio disagreement across seeds ===')
    print(coupon_ratio_wide.to_string(index=False))

    # --- per-incentive-bin: ensemble + each seed ---
    bin_min_n = 100
    merged_binned = merged.copy()
    merged_binned['incentive_bin'] = pd.cut(merged_binned['incentive_at_ref'], bins=INCENTIVE_EDGES)
    bin_tables = {'ensemble': rate_table(merged_binned.dropna(subset=['incentive_bin']), 'incentive_bin', None, bin_min_n)}
    for s in SEEDS:
        d = merged_binned.copy(); d['h'] = merged_binned[f'h_seed{s}']
        bin_tables[f'seed{s}'] = rate_table(d.dropna(subset=['incentive_bin']), 'incentive_bin', None, bin_min_n)
    print('\n=== PER-INCENTIVE-BIN (ensemble), count-weighted, min_n=100 ===')
    print(bin_tables['ensemble'].to_string(index=False))

    bin_ratio_wide = bin_tables['ensemble'][['incentive_bin', 'n_loan_months']].copy()
    for s in SEEDS:
        t = bin_tables[f'seed{s}'].set_index('incentive_bin')['predicted_rate_monthly'] / \
            bin_tables[f'seed{s}'].set_index('incentive_bin')['realized_rate_monthly']
        bin_ratio_wide[f'ratio_seed{s}'] = bin_ratio_wide['incentive_bin'].map(t)
    bin_ratio_wide['ensemble_ratio'] = (bin_tables['ensemble'].set_index('incentive_bin')['predicted_rate_monthly'] /
                                         bin_tables['ensemble'].set_index('incentive_bin')['realized_rate_monthly']).values
    bin_ratio_wide['min'] = bin_ratio_wide[seed_ratio_cols].min(axis=1)
    bin_ratio_wide['max'] = bin_ratio_wide[seed_ratio_cols].max(axis=1)
    bin_ratio_wide['spread'] = bin_ratio_wide['max'] - bin_ratio_wide['min']
    bin_ratio_wide.to_csv(os.path.join(OUT_DIR, 'per_bin_ratio_disagreement.csv'), index=False)
    print('\n=== PER-INCENTIVE-BIN ratio disagreement across seeds ===')
    print(bin_ratio_wide.to_string(index=False))

    # --- dispersion: ensemble + each seed ---
    disp_rows = []
    for label, d in [('ensemble', merged)] + [(f'seed{s}', None) for s in SEEDS]:
        if d is None:
            s = int(label.replace('seed', ''))
            d = merged.copy(); d['h'] = merged[f'h_seed{s}']
        t = rate_table(d, 'coupon', None, coupon_min_n)
        stats = dispersion_stats(t, 'predicted_rate_monthly', 'realized_rate_monthly', 'n_loan_months', coupon_min_n)
        stats['model'] = label
        disp_rows.append(stats)
    disp_df = pd.DataFrame(disp_rows)
    disp_df.to_csv(os.path.join(OUT_DIR, 'dispersion_stats.csv'), index=False)
    print('\n=== DISPERSION (per-coupon, count-weighted), ensemble + per-seed ===')
    print(disp_df.to_string(index=False))
    seed_disp = disp_df[disp_df['model'] != 'ensemble']
    for col in ['PRIMARY_predicted_dispersion_max_over_min', 'PRIMARY_realized_dispersion_max_over_min',
                'PRIMARY_dispersion_ratio', 'SECONDARY_ratio_max', 'SECONDARY_ratio_min']:
        disagree_rows.append({
            'stat': col, 'weight': 'count',
            'min': seed_disp[col].min(), 'max': seed_disp[col].max(),
            'spread': seed_disp[col].max() - seed_disp[col].min(),
        })

    disagree_df = pd.DataFrame(disagree_rows)
    disagree_df.to_csv(os.path.join(OUT_DIR, 'disagreement_summary.csv'), index=False)
    print('\n=== DISAGREEMENT ACROSS THE 5 INDIVIDUAL SEEDS (min/max/spread) ===')
    print(disagree_df.to_string(index=False))

    # --- month-by-month: ensemble + each seed ---
    month_rows = []
    for label, d in [('ensemble', merged)] + [(f'seed{s}', None) for s in SEEDS]:
        if d is None:
            s = int(label.replace('seed', ''))
            d = merged.copy(); d['h'] = merged[f'h_seed{s}']
        for ref_ym, g in d.groupby('ref_month'):
            pred = g['h'].mean(); real = g['realized_event'].mean()
            month_rows.append({
                'model': label, 'ref_month': ref_ym, 'n_active': len(g),
                'predicted_rate_annualized': 1 - (1 - pred) ** 12,
                'realized_rate_annualized': 1 - (1 - real) ** 12,
            })
    month_df = pd.DataFrame(month_rows).sort_values(['model', 'ref_month'])
    month_df.to_csv(os.path.join(OUT_DIR, 'month_by_month.csv'), index=False)
    print('\n=== MONTH-BY-MONTH (ensemble) ===')
    print(month_df[month_df['model'] == 'ensemble'].to_string(index=False))

    print(f'\nAll outputs saved under: {OUT_DIR}')


if __name__ == '__main__':
    main()
