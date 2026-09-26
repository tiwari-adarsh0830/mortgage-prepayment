"""
measure_alignment_effect.py -- measures the magnitude of the LEFT- vs
RIGHT-aligned scoring mismatch found in condition 4 (see
docs/mistakes_and_lessons.md, "forecast_matched_population_cpr.py scores a
right-aligned-trained multiobs model on left-aligned trailing sequences").

For the cutoff_2020 f0.2/L33/IPW seed-42 checkpoint, scores the SAME loans
two ways:
  LEFT-aligned  : TRAIL_SEQ_DIR's existing test_seq.npy/test_mask.npy
                  (prepare_sequences_trailing_zbc.py's convention), exactly
                  the path forecast_matched_population_cpr.py already uses.
  RIGHT-aligned : the new Dec-2020 control-population build
                  (score_multiobs_dec_window.py's build_combined_pass +
                  build_dec_window_obs + build_sequences_multiobs(obs=...)),
                  pre-fix CP/U maps, GOLDEN_BACKUP scaler -- matches how this
                  checkpoint was actually trained.

Restricted to the INTERSECTION of both populations. Split by full-history
(mask.sum()==33, where the two alignments coincide by construction) vs
partial (<33, where they can differ). Full-history loans are expected to
score identically (or near-identically -- float/order-of-summation noise
only) either way; if they don't, that means the two paths differ in
something besides alignment, and this script says so explicitly instead of
reporting the rest of the numbers as if the comparison were clean.

Usage:
    python scripts/measure_alignment_effect.py \\
        --out_dir outputs/rolling/alignment_effect_cutoff_2020_seed42
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_multiobs_dec_window import (
    build_combined_pass, build_dec_window_obs, load_checkpoint, score, BASE,
)
from prepare_sequences_multiobs_zbc import build_sequences_multiobs

TRAIL_SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_trail')
CONTROL_SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP')
CKPT_PATH = os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33/hazard_best.pt')
FULL_HISTORY_LEN = 33
FULL_HISTORY_TOL = 1e-4   # relative tolerance for the full-history sanity check


def score_left_aligned(model, batch_size=8192):
    seqs  = np.load(os.path.join(TRAIL_SEQ_DIR, 'test_seq.npy'),      mmap_mode='r')
    masks = np.load(os.path.join(TRAIL_SEQ_DIR, 'test_mask.npy'),     mmap_mode='r')
    ids   = np.load(os.path.join(TRAIL_SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    h = score(model, np.asarray(seqs), np.asarray(masks), batch_size)
    n_months = np.asarray(masks).sum(axis=1)
    return pd.DataFrame({'loan_id': ids, 'h_left': h, 'n_months': n_months})


def score_right_aligned(model, batch_size=8192):
    with open(os.path.join(CONTROL_SEQ_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    test_ids = np.load(os.path.join(CONTROL_SEQ_DIR, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())

    cache_dir = os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')
    full_df = build_combined_pass(2020, False, 'prefix', test_ids_set, cache_dir)
    obs, dec_rows, diag = build_dec_window_obs(full_df, 2020)
    print(f'Right-aligned population diagnostics: {diag}', flush=True)

    sequences, masks, _l, _p, loan_ids_out, _extras = build_sequences_multiobs(
        full_df, scaler, k_draws=1, H=1, min_hist=1, obs=obs)
    h = score(model, sequences, masks, batch_size)
    note_rate = dec_rows.set_index('loan_id').loc[loan_ids_out, 'original_interest_rate'].to_numpy()
    return pd.DataFrame({'loan_id': loan_ids_out, 'h_right': h, 'note_rate': note_rate})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=8192)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_checkpoint(CKPT_PATH)

    print('Scoring LEFT-aligned (TRAIL_SEQ_DIR)...', flush=True)
    left = score_left_aligned(model, args.batch_size)
    print('Scoring RIGHT-aligned (new control path, pre-fix maps)...', flush=True)
    right = score_right_aligned(model, args.batch_size)

    df = left.merge(right, on='loan_id', how='inner')
    print(f'Intersection: {len(df):,} loans (left n={len(left):,}, right n={len(right):,})', flush=True)
    df['ratio'] = df['h_right'] / df['h_left']
    df['annual_pp_left']  = 1.0 - (1.0 - np.clip(df['h_left'],  1e-7, 1 - 1e-7)) ** 12
    df['annual_pp_right'] = 1.0 - (1.0 - np.clip(df['h_right'], 1e-7, 1 - 1e-7)) ** 12
    df['full_history'] = df['n_months'] == FULL_HISTORY_LEN
    df['coupon'] = ((df['note_rate'] - 0.5) * 2).round() / 2

    out_path = os.path.join(args.out_dir, 'alignment_effect_loans.csv')
    df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}', flush=True)

    full = df[df['full_history']]
    if len(full):
        rel_diff = (full['h_right'] - full['h_left']).abs() / full['h_left'].clip(lower=1e-9)
        n_bad = int((rel_diff > FULL_HISTORY_TOL).sum())
        print(f'\nFull-history ({FULL_HISTORY_LEN}mo) sanity check: {n_bad:,}/{len(full):,} loans '
              f'differ by more than {FULL_HISTORY_TOL:.0e} relative -- STOP and investigate before '
              f'trusting the partial-history numbers below if this is not ~0.' if n_bad else
              f'\nFull-history ({FULL_HISTORY_LEN}mo) sanity check PASSED: all {len(full):,} loans '
              f'agree within {FULL_HISTORY_TOL:.0e} relative (as expected -- the two alignments '
              f'coincide when the window is completely full).', flush=True)

    print(f'\n{"=" * 90}\nn / mean h_left / mean h_right / mean ratio / median ratio, split by history\n{"=" * 90}')
    for label, g in [('full_history (33mo)', df[df['full_history']]),
                      ('partial_history (<33mo)', df[~df['full_history']])]:
        if len(g) == 0:
            print(f'{label}: no loans')
            continue
        print(f'{label}: n={len(g):,}  mean_h_left={g["h_left"].mean():.5f}  '
              f'mean_h_right={g["h_right"].mean():.5f}  mean_ratio={g["ratio"].mean():.4f}  '
              f'median_ratio={g["ratio"].median():.4f}')

    print(f'\n{"=" * 90}\nPer-coupon annualized CPR, both alignments, split by history\n{"=" * 90}')
    for label, g in [('full_history (33mo)', df[df['full_history']]),
                      ('partial_history (<33mo)', df[~df['full_history']])]:
        if len(g) == 0:
            continue
        rows = []
        for coupon, gg in g.groupby('coupon'):
            rows.append({
                'coupon': coupon, 'n_loans': len(gg),
                'cpr_left':  round(gg['annual_pp_left'].mean() * 100, 4),
                'cpr_right': round(gg['annual_pp_right'].mean() * 100, 4),
            })
        t = pd.DataFrame(rows).sort_values('coupon')
        print(f'\n{label}:')
        print(t.to_string(index=False))


if __name__ == '__main__':
    main()
