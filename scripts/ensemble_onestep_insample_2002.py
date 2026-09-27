"""ensemble_onestep_insample_2002.py -- five-seed ensemble of the cutoff_2002
in-sample one-step-ahead scoring (Jan2001-Nov2002), analogue of
ensemble_onestep_2002.py for the 2003 forecast window.

BEFORE averaging, asserts the five seeds' (loan_id, ref_month) populations
are IDENTICAL, same as ensemble_onestep_2002.py.

Run:
    python scripts/ensemble_onestep_insample_2002.py
"""
import os

import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
SEEDS = [42, 7, 123, 1001, 2026]
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_insample_cutoff_2002')
os.makedirs(OUT_DIR, exist_ok=True)


def load_seed(seed):
    path = os.path.join(BASE, f'outputs/rolling/rolling_onestep_insample_cutoff_2002_seed{seed}/rolling_all_months.csv')
    df = pd.read_csv(path)
    df['coupon'] = ((df['note_rate'] - 0.5) * 2).round() / 2
    return df


def main():
    seed_dfs = {s: load_seed(s) for s in SEEDS}

    keys = {s: set(zip(d['loan_id'], d['ref_month'])) for s, d in seed_dfs.items()}
    base_key = keys[SEEDS[0]]
    for s in SEEDS[1:]:
        assert keys[s] == base_key, (
            f'Seed {s} population differs from seed {SEEDS[0]}: '
            f'{len(keys[s] - base_key)} extra, {len(base_key - keys[s])} missing -- STOP.')
    print(f'Population identity check PASSED: all 5 seeds share the same '
          f'{len(base_key):,} (loan_id, ref_month) pairs.', flush=True)

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

    print(f'Reference months present: {sorted(merged["ref_month"].unique())}')
    print(f'\nAll outputs saved under: {OUT_DIR}')


if __name__ == '__main__':
    main()
