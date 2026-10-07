"""
compare_seq_30y_vs_seq_2002.py -- identity check between the two cutoff_2002 builds:

  _30y : data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y   (9 feature columns)
  _seq : data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_seq   (10 feature columns,
         harp_eligible appended as the last column, all zeros)

Claim under test: seq[:, :, :9] is identical to the _30y sequence array, and
every non-sequence array (mask, labels, loan_ids, ref_month, incl_prob, ...) is
identical. If so, the only input difference between the two training runs is the
constant 10th column, so the AUC difference on the same seeds can only come from
the changed parameter shapes (initialization / RNG stream), not from the data.

Reads with np.load(mmap_mode='r'); sequence arrays are compared in chunks of
50,000 rows. Prints one line per array. Exit code is 0 either way -- the
VERDICT line at the end is the result.
"""
import os
import sys

import numpy as np

BASE = '/scratch/at7095/mortgage_prepayment/data/sequences_rolling'
DIR_30Y = os.path.join(BASE, 'cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y')
DIR_SEQ = os.path.join(BASE, 'cutoff_2002_zbc_multiobs_f0.2_h1_hist_seq')
CHUNK = 50_000


def _load(path):
    try:
        return np.load(path, mmap_mode='r')
    except ValueError:  # object arrays (e.g. *_loan_ids_split.npy) need pickle
        return np.load(path, allow_pickle=True)


def _eq(a, b):
    if a.dtype.kind == 'f' and b.dtype.kind == 'f':
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _maxabs(a, b):
    if a.dtype.kind not in 'fiub' or b.dtype.kind not in 'fiub':
        return float('nan')
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    d = d[~np.isnan(d)]
    return float(d.max()) if d.size else 0.0


def compare_seq(split):
    name = f'{split}_seq.npy'
    a = _load(os.path.join(DIR_30Y, name))
    b = _load(os.path.join(DIR_SEQ, name))
    print(f'{name}: _30y shape={a.shape} dtype={a.dtype} | _seq shape={b.shape} dtype={b.dtype}', flush=True)
    assert a.ndim == b.ndim == 3, f'{name}: expected 3-D arrays'
    assert a.shape[:2] == b.shape[:2], f'{name}: first two axes differ {a.shape[:2]} vs {b.shape[:2]}'
    assert a.shape[2] == 9 and b.shape[2] == 10, f'{name}: last axis {a.shape[2]} vs {b.shape[2]}, expected 9 vs 10'

    n = a.shape[0]
    overall_max, n_bad = 0.0, 0
    tenth_min, tenth_max = np.inf, -np.inf
    for s in range(0, n, CHUNK):
        ca = np.asarray(a[s:s + CHUNK])
        cb = np.asarray(b[s:s + CHUNK, :, :9])
        if not _eq(ca, cb):
            md = _maxabs(ca, cb)
            overall_max = max(overall_max, md)
            n_bad += 1
            print(f'  chunk rows {s}:{s + len(ca)} DIFFERS, max abs diff = {md:.6e}', flush=True)
        c10 = np.asarray(b[s:s + CHUNK, :, 9])
        tenth_min, tenth_max = min(tenth_min, float(c10.min())), max(tenth_max, float(c10.max()))
    ident = n_bad == 0
    print(f'{name} [:, :, :9] vs _30y: {"IDENTICAL" if ident else "NOT identical"}  '
          f'max abs diff={overall_max:.6e}  bad chunks={n_bad}/{(n + CHUNK - 1) // CHUNK}  '
          f'| _seq 10th column range=[{tenth_min}, {tenth_max}]', flush=True)
    return ident


def main():
    files_30y = sorted(f for f in os.listdir(DIR_30Y) if f.endswith('.npy'))
    files_seq = sorted(f for f in os.listdir(DIR_SEQ) if f.endswith('.npy'))
    only_30y, only_seq = set(files_30y) - set(files_seq), set(files_seq) - set(files_30y)
    print(f'.npy files: _30y={len(files_30y)} _seq={len(files_seq)} '
          f'only_in_30y={sorted(only_30y)} only_in_seq={sorted(only_seq)}', flush=True)

    results = {}
    for split in ('train', 'test'):
        results[f'{split}_seq.npy'] = compare_seq(split)

    for f in sorted(set(files_30y) & set(files_seq)):
        if f in ('train_seq.npy', 'test_seq.npy'):
            continue
        a, b = _load(os.path.join(DIR_30Y, f)), _load(os.path.join(DIR_SEQ, f))
        if a.shape != b.shape:
            print(f'{f}: shape differs {a.shape} vs {b.shape} -> NOT identical', flush=True)
            results[f] = False
            continue
        if a.dtype == object or b.dtype == object:
            ident = bool(np.array_equal(a, b))
            md = float('nan')
        else:
            ident = _eq(np.asarray(a), np.asarray(b))
            md = 0.0 if ident else _maxabs(np.asarray(a), np.asarray(b))
        print(f'{f}: {"identical" if ident else "NOT identical"}  max abs diff={md}', flush=True)
        results[f] = ident

    bad = [k for k, v in results.items() if not v] + sorted(only_30y | only_seq)
    print('\nVERDICT: ' + ('ALL arrays identical (seq[:,:,:9] == _30y; all other arrays equal; '
                           'no file present in only one directory)'
                           if not bad else f'DIFFERENCES in: {bad}'), flush=True)


if __name__ == '__main__':
    sys.exit(main())
