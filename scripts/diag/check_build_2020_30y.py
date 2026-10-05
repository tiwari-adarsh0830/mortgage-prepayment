"""check_build_2020_30y.py -- sanity gate on the cutoff_2020 _30y multiobs build
(data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_30y, built by
prepare_sequences_multiobs_zbc.py --cutoff_year 2020 --sample_frac 0.1 --run_tag _30y).
Same pattern as check_build_30y.py (cutoff_2002), adapted for a modern-only
(no --include_pre2013) cutoff: there is no pre-2013 cell-grid sample to check
a loader path/count against here -- cutoff_2020's population is instead a
10% uniform loan subsample drawn INSIDE load_vintage_filtered itself
(Pass 1 discovery, seeded np.random.default_rng(42)), with no separate
loan-id-list file. The "2020 expected loader line" check below is therefore
the "Total loans: N | Prepay rate: P%" line prepare_sequences_multiobs_zbc.py
prints after Pass 1 -- checked for presence/parseability (a missing or
unparseable line means Pass 1 itself failed), not for exact equality against
the pre-fix reference figures (the term/post-mod filters change the
population, so the new count is EXPECTED to differ from the old one; both
are printed side by side, informationally, same as check_build_30y.py's
n_loans/n_obs convention).

Exits 1 on any failure, so a dependent sbatch job submitted with
--dependency=afterok only runs if this gate passes.

Checks:
  - train/test seq arrays load, L (seq.shape[1]) == 33
  - distinct property_type_enc codes are a subset of {0,1,2,3,4} (map_era=fixed)
  - distinct loan_purpose_enc codes are a subset of {0,1,2,3}
  - scaler.pkl exists; prints its property_type_enc mean
  - n_loans/n_obs (train+test), printed next to the pre-fix GOLDEN_BACKUP
    build's reference figures (1,870,909 loans / 14,039,222 obs)
  - count of observations with is_terminal & label==1
  - greps --build_log for the "Total loans: N | Prepay rate: P%" line and
    checks it parses (count NOT asserted equal to the old reference)

Usage:
    python scripts/diag/check_build_2020_30y.py --build_log logs/multiobs_2020_30y_<jobid>.out
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np

BASE = '/scratch/at7095/mortgage_prepayment'
BUILD_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_30y')
EXPECTED_L = 33
EXPECTED_N_FEATURES = 10
EXPECTED_PT_CODES = {0, 1, 2, 3, 4}
EXPECTED_LP_CODES = {0, 1, 2, 3}
PT_IDX, LP_IDX = 8, 7   # FEATURE_COLS indices, see prepare_sequences_multiobs_zbc.py
HARP_IDX = 9             # harp_eligible -- all zeros, no eligibility logic yet
OLD_N_LOANS, OLD_N_OBS = 1_870_909, 14_039_222   # pre-fix GOLDEN_BACKUP build reference


def fail(msg):
    print(f'FAIL: {msg}', flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build_log', type=str, required=True,
                     help='Job B stdout log, grepped for the "Total loans" Pass-1 line.')
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
    print(f'\nn_loans (distinct, train+test): {n_loans:,}  (pre-fix build: {OLD_N_LOANS:,})', flush=True)
    print(f'n_obs (train+test):              {n_obs_total:,}  (pre-fix build: {OLD_N_OBS:,})', flush=True)
    print(f'observations with is_terminal & is_prepaid (label==1), train+test: '
          f'{n_terminal_prepay:,}', flush=True)

    if not os.path.exists(args.build_log):
        fail(f'--build_log {args.build_log} does not exist')
    with open(args.build_log) as f:
        log_text = f.read()
    m = re.search(r'Total loans: ([\d,]+) \| Prepay rate: ([\d.]+)%', log_text)
    if not m:
        fail(f'no "Total loans" Pass-1 line found in {args.build_log}')
    log_loans, log_rate = int(m.group(1).replace(',', '')), float(m.group(2))
    print(f'\nBuild log Pass-1 line: Total loans={log_loans:,}  Prepay rate={log_rate}%', flush=True)

    print('\nALL CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
