"""score_matched_intersection_2003.py -- model-vs-population decomposition
for the cutoff_2002 2003 one-step-ahead forecast.

Scores a FIXED matched intersection of (loan_id, ref_month) pairs --
360-term loans in the TEST split of BOTH the old (_hist) and new (_30y)
cutoff_2002 builds, with reference months common to both -- under one of
4 combinations of {old,new} x {checkpoint set, population/pass}:

  --ckpts {old,new}: which 5-seed checkpoint set (+ its own scaler,
      map_era='fixed' for both -- no map confound) scores the windows.
  --pass_set {old,new}: which raw combined-pass (test_ids_set/cell_sample/
      cache_dir) builds the full_df the windows are gathered from.

Both passes run through the SAME current load_vintage_filtered -- for
loans already in the intersection, the only way "old pass" vs "new pass"
can differ is a STALE combined-pass cache. The old pass's default cache
(_raw_combined_pass_50d476ff0e0bb158_trunc200312_v2_upb.pkl, built
2026-09-24 22:09) predates bb8dece (2026-10-01 22:03, added the term/
post-mod fix to load_vintage_filtered) and CACHE_VERSION was never bumped
across that commit -- so --pass_set old points at a FRESH --cache_dir,
not the default, to force a genuine rebuild under the current loader.

Usage:
    python scripts/score_matched_intersection_2003.py --ckpts old --pass_set old --out_dir ...
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_multiobs_dec_window import (
    build_combined_pass, load_checkpoint, verify_checkpoint, score, BASE,
)
from score_rolling_one_step import build_rolling_eligible, reference_months
from prepare_sequences_multiobs_zbc import build_sequences_multiobs, PRE2013_CELL_SAMPLE_PATH

H = 1
MIN_HIST = 1
INTERSECTION_CSV = os.path.join(BASE, 'outputs/diag/matched_intersection_2003_pairs.csv')

SEQ_DIR = {
    'old': os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist'),
    'new': os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y'),
}
CELL_SAMPLE = {
    'old': PRE2013_CELL_SAMPLE_PATH,
    'new': os.path.join(BASE, 'outputs/pre2013_cell_sample_30y_loans.csv'),
}
CACHE_DIR = {
    'old': os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache_old_pass_fresh'),
    'new': os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache'),
}
CKPT_DIRS = {
    'old': {
        42:   'cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist',
        7:    'cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed7',
        123:  'cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed123',
        1001: 'cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed1001',
        2026: 'cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed2026',
    },
    'new': {
        42:   'cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s42',
        7:    'cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s7',
        123:  'cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s123',
        1001: 'cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s1001',
        2026: 'cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s2026',
    },
}
SEEDS = [42, 7, 123, 1001, 2026]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', choices=['old', 'new'], required=True)
    ap.add_argument('--pass_set', choices=['old', 'new'], required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=8192)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Cell: ckpts={args.ckpts}  pass_set={args.pass_set}', flush=True)

    pairs_df = pd.read_csv(INTERSECTION_CSV)
    intersection_pairs = set(zip(pairs_df['loan_id'], pairs_df['ref_month']))
    print(f'Matched intersection: {len(intersection_pairs):,} (loan_id, ref_month) pairs '
          f'from {INTERSECTION_CSV}', flush=True)

    import pickle
    with open(os.path.join(SEQ_DIR[args.ckpts], 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    pass_seq_dir = SEQ_DIR[args.pass_set]
    test_ids = np.load(os.path.join(pass_seq_dir, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    print(f'Pass test split ({args.pass_set}, {pass_seq_dir}): {len(test_ids_set):,} loans', flush=True)

    os.makedirs(CACHE_DIR[args.pass_set], exist_ok=True)
    full_df = build_combined_pass(2002, True, 'fixed', test_ids_set,
                                   cache_dir=CACHE_DIR[args.pass_set],
                                   cell_sample_path=CELL_SAMPLE[args.pass_set])
    elig = build_rolling_eligible(full_df)

    models = {}
    for seed in SEEDS:
        ckpt_dir = os.path.join(BASE, 'outputs/rolling', CKPT_DIRS[args.ckpts][seed])
        ckpt_path = os.path.join(ckpt_dir, 'hazard_best.pt')
        results_json_path = os.path.join(ckpt_dir, 'results.json')
        model = load_checkpoint(ckpt_path)
        ckpt_dict = torch.load(ckpt_path, map_location='cpu')
        verify_checkpoint(model, SEQ_DIR[args.ckpts], results_json_path, ckpt_dict, args.batch_size)
        models[seed] = model

    all_months = []
    n_expected_total = 0
    n_scored_total = 0
    for ref_ym in reference_months(2002):
        wanted = {lid for (lid, rm) in intersection_pairs if rm == ref_ym}
        n_expected_total += len(wanted)
        month_rows_all = elig[(elig['yyyymm'] == ref_ym) & (elig['loan_id'].isin(wanted))]
        month_rows = month_rows_all[month_rows_all['forward_adjacent']].copy()
        n_dropped = len(wanted) - len(month_rows)
        print(f'  {ref_ym}: intersection wants {len(wanted):,}; eligible+forward_adjacent in this '
              f'pass: {len(month_rows):,}  (dropped {n_dropped:,})', flush=True)
        if month_rows.empty:
            continue

        obs = pd.DataFrame({
            'loan_id':          month_rows['loan_id'].to_numpy(),
            't':                month_rows['row_idx'].astype(int).to_numpy(),
            'ref_month':        month_rows['yyyymm'].astype(int).to_numpy(),
            'label':            np.zeros(len(month_rows), dtype=np.float32),
            'is_terminal':      np.zeros(len(month_rows), dtype=bool),
            'incl_prob':        np.ones(len(month_rows), dtype=np.float32),
            'n_eligible':       np.ones(len(month_rows), dtype=np.int32),
            'k_actual':         np.ones(len(month_rows), dtype=np.int32),
            'age_at_ref':       month_rows['loan_age_months'].astype(float).to_numpy(),
            'incentive_at_ref': month_rows['refi_incentive'].astype(float).to_numpy(),
        })
        sequences, masks, _l, _p, loan_ids_out, _extras = build_sequences_multiobs(
            full_df, scaler, k_draws=1, H=H, min_hist=MIN_HIST, obs=obs)

        h_per_seed = {}
        for seed in SEEDS:
            h_per_seed[f'h_seed{seed}'] = score(models[seed], sequences, masks, args.batch_size)
        h_ensemble = np.mean(np.stack(list(h_per_seed.values()), axis=0), axis=0)

        mr = month_rows.set_index('loan_id')
        realized_event = (mr.loc[loan_ids_out, 'next_zbc'] == 1.0).to_numpy()
        month_out = pd.DataFrame({
            'loan_id':            loan_ids_out,
            'ref_month':          ref_ym,
            'note_rate':          mr.loc[loan_ids_out, 'original_interest_rate'].to_numpy(),
            'incentive_at_ref':   mr.loc[loan_ids_out, 'refi_incentive'].to_numpy(),
            'current_actual_upb': mr.loc[loan_ids_out, 'current_actual_upb'].to_numpy(),
            'realized_event':     realized_event.astype(int),
            **h_per_seed,
            'h':                  h_ensemble,
        })
        month_out['coupon'] = ((month_out['note_rate'] - 0.5) * 2).round() / 2
        all_months.append(month_out)
        n_scored_total += len(month_out)

    full = pd.concat(all_months, ignore_index=True)
    print(f'\nTotal: intersection wanted {n_expected_total:,} pairs across 12 months; '
          f'scored {n_scored_total:,} ({n_expected_total - n_scored_total:,} dropped by '
          f'eligibility/forward-adjacency in this pass).', flush=True)
    out_path = os.path.join(args.out_dir, 'all_months.csv')
    full.to_csv(out_path, index=False)
    print(f'Saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
