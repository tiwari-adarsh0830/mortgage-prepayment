"""check_build_30y.py -- sanity gate on the cutoff_2002 _30y multiobs build
(data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y, built by
prepare_sequences_multiobs_zbc.py --cell_sample outputs/pre2013_cell_sample_30y_loans.csv
--run_tag _30y). Exits 1 on any failure, so a dependent sbatch job submitted
with --dependency=afterok only runs if this gate passes.

Checks:
  - train/test seq arrays load, L (seq.shape[1]) == 33
  - distinct property_type_enc codes (unscaled via scaler.mean_/scale_,
    masked to valid timesteps) are a subset of {0,1,2,3,4}
  - distinct loan_purpose_enc codes are a subset of {0,1,2,3} (the CP/U-fixed
    map: R=0, C=1, P=2, U=3)
  - scaler.pkl exists; prints its property_type_enc mean (scaler.mean_[8])
  - n_loans (distinct loan_ids) and n_obs (seq.shape[0]), train+test, printed
    next to the old (pre-30y) build's reference figures (304,447 / 906,877,
    README job 17980942)
  - count of observations with is_terminal & label==1 (mandatory/terminal
    draws that are themselves the prepay event)
  - greps --build_log for the "Loaded pre-2013 cell-grid sample from <path>"
    line and checks it names the _30y path with 1,482,004 ids

Usage:
    python scripts/diag/check_build_30y.py --build_log logs/multiobs_2002_f0.2_L33_hist_30y_<jobid>.out
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np

BASE = '/scratch/at7095/mortgage_prepayment'
BUILD_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y')
EXPECTED_L = 33
EXPECTED_PT_CODES = {0, 1, 2, 3, 4}
EXPECTED_LP_CODES = {0, 1, 2, 3}
PT_IDX, LP_IDX = 8, 7   # FEATURE_COLS indices, see prepare_sequences_multiobs_zbc.py
OLD_N_LOANS, OLD_N_OBS = 304_447, 906_877   # pre-30y build reference (README job 17980942)
EXPECTED_CELL_SAMPLE_PATH = os.path.join(BASE, 'outputs', 'pre2013_cell_sample_30y_loans.csv')
EXPECTED_N_IDS = 1_482_004


def fail(msg):
    print(f'FAIL: {msg}', flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build_log', type=str, required=True,
                     help='Job B stdout log, grepped for the cell-sample loader line.')
    args = ap.parse_args()

    print(f'Checking build dir: {BUILD_DIR}', flush=True)
    if not os.path.isdir(BUILD_DIR):
        fail(f'{BUILD_DIR} does not exist')

    with open(os.path.join(BUILD_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print(f'scaler.pkl loaded. property_type_enc (scaler.mean_[{PT_IDX}]) mean: '
          f'{scaler.mean_[PT_IDX]:.6f}', flush=True)

    n_loans_all, n_obs_total = set(), 0
    n_terminal_prepay = 0

    for split in ('train', 'test'):
        seq   = np.load(os.path.join(BUILD_DIR, f'{split}_seq.npy'), mmap_mode='r')
        mask  = np.load(os.path.join(BUILD_DIR, f'{split}_mask.npy'), mmap_mode='r')
        lbl   = np.load(os.path.join(BUILD_DIR, f'{split}_labels.npy'))
        term  = np.load(os.path.join(BUILD_DIR, f'{split}_is_terminal.npy'))
        lids  = np.load(os.path.join(BUILD_DIR, f'{split}_loan_ids.npy'), allow_pickle=True)

        print(f'{split}_seq.npy shape: {seq.shape}', flush=True)
        if seq.shape[1] != EXPECTED_L:
            fail(f'{split}_seq.npy L={seq.shape[1]}, expected {EXPECTED_L}')

        mask_np = np.asarray(mask)
        seq_np  = np.asarray(seq)
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
    print(f'\nn_loans (distinct, train+test): {n_loans:,}  (old build: {OLD_N_LOANS:,})', flush=True)
    print(f'n_obs (train+test):              {n_obs_total:,}  (old build: {OLD_N_OBS:,})', flush=True)
    print(f'observations with is_terminal & is_prepaid (label==1), train+test: '
          f'{n_terminal_prepay:,}', flush=True)

    if not os.path.exists(args.build_log):
        fail(f'--build_log {args.build_log} does not exist')
    with open(args.build_log) as f:
        log_text = f.read()
    m = re.search(r'Loaded pre-2013 cell-grid sample from (\S+): ([\d,]+) loan_ids', log_text)
    if not m:
        fail(f'no cell-sample loader line found in {args.build_log}')
    log_path, log_count = m.group(1), int(m.group(2).replace(',', ''))
    print(f'\nBuild log loader line: path={log_path} count={log_count:,}', flush=True)
    if log_path != EXPECTED_CELL_SAMPLE_PATH:
        fail(f'build log cell-sample path {log_path} != expected {EXPECTED_CELL_SAMPLE_PATH}')
    if log_count != EXPECTED_N_IDS:
        fail(f'build log cell-sample count {log_count:,} != expected {EXPECTED_N_IDS:,}')

    print('\nALL CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
