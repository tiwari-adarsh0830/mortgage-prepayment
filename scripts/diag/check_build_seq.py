"""check_build_seq.py -- generic sanity gate for a cutoff_{year}..._seq multiobs
build (any cutoff year, the 23-cutoff x 10-seed yearly-sequence chain -- Part
3/4 of the Oct 5, 2026 prompt). Generalizes check_build_30y.py/
check_build_2020_30y.py (both hardcoded to one cutoff) to an arbitrary
--build_dir/--cutoff_year, since most cutoffs in this chain have no prior
build to hardcode a reference against.

Checks:
  - train/test seq arrays load, L (seq.shape[1]) == --expected_l (default 33)
  - n_features (seq.shape[2]) == 10; column 9 (harp_eligible) is all zero
    after unscaling (advisor's Oct 4 decision -- see
    scripts/prepare_sequences_multiobs_zbc.py's FEATURE_COLS)
  - distinct property_type_enc codes subset of {0,1,2,3,4}; loan_purpose_enc
    subset of {0,1,2,3} (the CP/U-fixed maps)
  - scaler.pkl exists
  - n_loans (distinct loan_ids) / n_obs (seq.shape[0]), train+test, reported
    informationally -- NO old-build comparison, unlike check_build_30y.py:
    most cutoffs in this chain are first-time builds with nothing to compare
    against. (cutoff_2002's specific comparison against the existing _30y
    five-seed 0.898 AUC baseline happens after training, not here.)
  - greps --build_log for "Loaded pre-2013 cell-grid sample from <path>: N
    loan_ids" and checks path/count match the fixed _30y cell sample -- every
    cutoff in this chain uses --include_pre2013 with the SAME cell-sample
    file, so this check is cutoff-independent.
  - count of observations with is_terminal & label==1

Exits 1 on any failure, so a dependent sbatch job submitted with
--dependency=afterok only runs if this gate passes.

Usage:
    python scripts/diag/check_build_seq.py \\
        --build_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_seq \\
        --build_log logs/multiobs_2002_seq_<jobid>.out
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np

BASE = '/scratch/at7095/mortgage_prepayment'
EXPECTED_N_FEATURES = 10
EXPECTED_PT_CODES = {0, 1, 2, 3, 4}
EXPECTED_LP_CODES = {0, 1, 2, 3}
PT_IDX, LP_IDX = 8, 7   # FEATURE_COLS indices, see prepare_sequences_multiobs_zbc.py
HARP_IDX = 9             # harp_eligible -- all zeros, no eligibility logic yet
EXPECTED_CELL_SAMPLE_PATH = os.path.join(BASE, 'outputs', 'pre2013_cell_sample_30y_loans.csv')
EXPECTED_N_IDS = 1_482_004


def fail(msg):
    print(f'FAIL: {msg}', flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build_dir', type=str, required=True)
    ap.add_argument('--build_log', type=str, required=True,
                     help='Build job stdout log, grepped for the cell-sample loader line.')
    ap.add_argument('--expected_l', type=int, default=33)
    ap.add_argument('--skip_cell_sample_check', action='store_true',
                     help='Pass if this build did not use --include_pre2013 (no cell-sample '
                          'loader line will exist in --build_log).')
    args = ap.parse_args()

    print(f'Checking build dir: {args.build_dir}', flush=True)
    if not os.path.isdir(args.build_dir):
        fail(f'{args.build_dir} does not exist')

    with open(os.path.join(args.build_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print(f'scaler.pkl loaded. property_type_enc (scaler.mean_[{PT_IDX}]) mean: '
          f'{scaler.mean_[PT_IDX]:.6f}', flush=True)

    n_loans_all, n_obs_total = set(), 0
    n_terminal_prepay = 0

    for split in ('train', 'test'):
        seq   = np.load(os.path.join(args.build_dir, f'{split}_seq.npy'), mmap_mode='r')
        mask  = np.load(os.path.join(args.build_dir, f'{split}_mask.npy'), mmap_mode='r')
        lbl   = np.load(os.path.join(args.build_dir, f'{split}_labels.npy'))
        term  = np.load(os.path.join(args.build_dir, f'{split}_is_terminal.npy'))
        lids  = np.load(os.path.join(args.build_dir, f'{split}_loan_ids.npy'), allow_pickle=True)

        print(f'{split}_seq.npy shape: {seq.shape}', flush=True)
        if seq.shape[1] != args.expected_l:
            fail(f'{split}_seq.npy L={seq.shape[1]}, expected {args.expected_l}')
        if seq.shape[2] != EXPECTED_N_FEATURES:
            fail(f'{split}_seq.npy n_features={seq.shape[2]}, expected {EXPECTED_N_FEATURES}')

        mask_np = np.asarray(mask)
        seq_np  = np.asarray(seq)

        valid_harp = seq_np[..., HARP_IDX][mask_np]
        harp_unscaled = valid_harp * scaler.scale_[HARP_IDX] + scaler.mean_[HARP_IDX]
        if not np.allclose(harp_unscaled, 0.0):
            fail(f'{split} harp_eligible (col {HARP_IDX}) not all zero: '
                 f'max abs {np.abs(harp_unscaled).max():.6g}')

        valid_pt = seq_np[..., PT_IDX][mask_np]
        valid_lp = seq_np[..., LP_IDX][mask_np]
        pt_codes = np.unique(np.round(valid_pt * scaler.scale_[PT_IDX] + scaler.mean_[PT_IDX])).astype(int)
        lp_codes = np.unique(np.round(valid_lp * scaler.scale_[LP_IDX] + scaler.mean_[LP_IDX])).astype(int)
        print(f'{split}: distinct property_type_enc codes: {sorted(pt_codes.tolist())}', flush=True)
        print(f'{split}: distinct loan_purpose_enc codes: {sorted(lp_codes.tolist())}', flush=True)
        if not set(pt_codes.tolist()) <= EXPECTED_PT_CODES:
            fail(f'{split} property_type_enc codes {sorted(pt_codes.tolist())} not subset of {EXPECTED_PT_CODES}')
        if not set(lp_codes.tolist()) <= EXPECTED_LP_CODES:
            fail(f'{split} loan_purpose_enc codes {sorted(lp_codes.tolist())} not subset of {EXPECTED_LP_CODES}')

        n_loans_all.update(lids.tolist())
        n_obs_total += seq.shape[0]
        n_terminal_prepay += int((term & (lbl == 1)).sum())

        print(f'{split}: n_obs={seq.shape[0]:,}  n_terminal&prepay={int((term & (lbl == 1)).sum()):,}',
              flush=True)

    n_loans = len(n_loans_all)
    print(f'\nn_loans (distinct, train+test): {n_loans:,}', flush=True)
    print(f'n_obs (train+test):              {n_obs_total:,}', flush=True)
    print(f'observations with is_terminal & is_prepaid (label==1), train+test: '
          f'{n_terminal_prepay:,}', flush=True)

    if not os.path.exists(args.build_log):
        fail(f'--build_log {args.build_log} does not exist')
    with open(args.build_log) as f:
        log_text = f.read()

    if not args.skip_cell_sample_check:
        m = re.search(r'Loaded pre-2013 cell-grid sample from (\S+): ([\d,]+) loan_ids', log_text)
        if not m:
            fail(f'no cell-sample loader line found in {args.build_log} '
                 f'(pass --skip_cell_sample_check if this build did not use --include_pre2013)')
        log_path, log_count = m.group(1), int(m.group(2).replace(',', ''))
        print(f'\nBuild log loader line: path={log_path} count={log_count:,}', flush=True)
        if os.path.abspath(log_path) != os.path.abspath(EXPECTED_CELL_SAMPLE_PATH):
            fail(f'build log cell-sample path {log_path} != expected {EXPECTED_CELL_SAMPLE_PATH}')
        if log_count != EXPECTED_N_IDS:
            fail(f'build log cell-sample count {log_count:,} != expected {EXPECTED_N_IDS:,}')

    print('\nALL CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
