"""
slice_l33_to_l1.py — build the max_seq_len=1 "no-history" control dataset by
slicing the last timestep off the already-built L33 multiobs data, instead of
running a fresh prepare_sequences_multiobs_zbc.py --max_seq_len 1 job.

Why this is valid (see scoping discussion, not repeated in train_hazard_multiobs.py):
build_sequences_multiobs is RIGHT-ALIGNED -- index MAX_SEQ_LEN-1 is always
ref_month itself, for every observation, confirmed empirically (L33 test_mask
last column is 100% True across 2,808,691 rows). Eligibility/selection/labels/
incl_prob have zero dependence on MAX_SEQ_LEN (the window-gap filter is the
only MAX_SEQ_LEN-dependent term, and it's vacuously False at L=1 anyway). So
train_seq[:, -1:, :] IS the --max_seq_len 1 build for every observation L33
has, with labels/incl_prob unchanged -- not an approximation.

Source: data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1   (L33)
Dest:   data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_L1
"""

import os
import shutil
import numpy as np

BASE = '/scratch/at7095/mortgage_prepayment'
SRC  = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1')
DST  = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_L1')

os.makedirs(DST, exist_ok=True)

for split in ('train', 'test'):
    seq  = np.load(os.path.join(SRC, f'{split}_seq.npy'),  mmap_mode='r')
    mask = np.load(os.path.join(SRC, f'{split}_mask.npy'), mmap_mode='r')

    assert mask[:, -1].all(), f'{split}: last-timestep mask not all True -- abort, assumption violated'

    seq_out  = np.ascontiguousarray(seq[:, -1:, :])
    mask_out = np.ascontiguousarray(mask[:, -1:])
    np.save(os.path.join(DST, f'{split}_seq.npy'),  seq_out)
    np.save(os.path.join(DST, f'{split}_mask.npy'), mask_out)
    print(f'{split}_seq:  {seq.shape} -> {seq_out.shape}', flush=True)
    print(f'{split}_mask: {mask.shape} -> {mask_out.shape}', flush=True)
    del seq, mask, seq_out, mask_out

# labels/incl_prob unchanged -- no MAX_SEQ_LEN dependence, copy as-is.
for fname in ('train_labels.npy', 'train_incl_prob.npy', 'test_labels.npy'):
    shutil.copy2(os.path.join(SRC, fname), os.path.join(DST, fname))
    print(f'copied {fname}', flush=True)

print('Done.', flush=True)
