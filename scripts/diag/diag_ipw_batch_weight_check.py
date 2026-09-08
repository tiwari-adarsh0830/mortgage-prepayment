"""
diag_ipw_batch_weight_check.py -- four-check diagnostic that located the
inverted-weight bug in train_hazard_multiobs.py's --use_ipw loss (caught
2026-09-07). Runs entirely on real training data with a fixed seed, no GPU
needed -- every number it prints is reproducible by rerunning this file.

Background
----------
The --use_ipw retrain (job 17068340) produced calibration that was WORSE
than the uncorrected run, not better (see README "IPW correction and
reproducibility" section) -- the wrong direction for a correction. This
script isolates why, checking four independent things so the cause is
demonstrated rather than assumed:

  CHECK 4 (premise) -- is the incl_prob distribution actually different
    between label=1 and label=0 observations in a way that WOULD matter if
    weighted the wrong way? Confirms every positive observation has
    incl_prob == 1.0 exactly (the mandatory/terminal draw is the only draw
    that can carry a positive label under this builder's H=1 window, and it
    is always included when eligible), while negatives span a wide range.

  CHECK 1 -- the actual sample_batch() output on the EXACT code path
    train_and_evaluate uses (same seed, same ObservationSampler, same
    batch_size): does the weight it hands to the loss favor positives or
    negatives, and does that match or contradict the intended IPW
    direction (rare draws upweighted)?

  CHECK 1b -- what the weight WOULD be on the same batch if correctly
    inverted (1/incl_prob instead of incl_prob directly) -- the
    counterfactual that shows the fix's actual effect size, not just its
    direction.

  CHECK 2 -- index alignment: does bweight/blabels/bmask returned by
    sample_batch() match direct fancy-indexing of the source arrays at the
    same idx? Rules out a batch-assembly bug as an alternative explanation
    before blaming the weight formula.

  CHECK 3 -- loss magnitude: is the uncorrected-vs-IPW loss gap (0.244 vs
    0.268) itself informative? No -- loss = (per_sample * w).sum() /
    w.sum() is a weighted MEAN regardless of whether w is incl_prob or
    1/incl_prob, so comparable magnitude is expected either way and does
    not distinguish a correct weighting from an inverted one.

Usage:
    python scripts/diag/diag_ipw_batch_weight_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from train_hazard_multiobs import ObservationSampler, BATCH_SIZE

BASE = '/scratch/at7095/mortgage_prepayment'
SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1')


def main():
    train_labels    = np.load(os.path.join(SEQ_DIR, 'train_labels.npy'))
    train_incl_prob = np.load(os.path.join(SEQ_DIR, 'train_incl_prob.npy'))
    train_seq  = np.load(os.path.join(SEQ_DIR, 'train_seq.npy'),  mmap_mode='r')
    train_mask = np.load(os.path.join(SEQ_DIR, 'train_mask.npy'), mmap_mode='r')

    print('=' * 90)
    print('CHECK 4 (premise): incl_prob distribution split by label, FULL TRAIN SET')
    print('=' * 90)
    for lbl in (0, 1):
        p = train_incl_prob[train_labels == lbl]
        print(f'  label={lbl}  n={len(p):,}  incl_prob: min={p.min():.5f} mean={p.mean():.5f} '
              f'max={p.max():.5f}  frac==1.0={float((p == 1.0).mean())*100:.2f}%')
    n_pos_not_mandatory = int(((train_labels == 1) & (train_incl_prob != 1.0)).sum())
    n_pos = int((train_labels == 1).sum())
    print(f'  positives with incl_prob != 1.0: {n_pos_not_mandatory:,} / {n_pos:,} positives')
    print()

    print('=' * 90)
    print('CHECK 1: actual sample_batch() output -- exact code path used by train_and_evaluate '
          '(pos_ratio=None, seed=42, batch_size={})'.format(BATCH_SIZE))
    print('=' * 90)
    sampler = ObservationSampler(train_seq, train_mask, train_labels, train_incl_prob, pos_ratio=None)
    rng = np.random.default_rng(42)
    bseq, bmask, blabels, bweight = sampler.sample_batch(BATCH_SIZE, rng)
    print('  batch idx dtype/shape check -- bweight IS incl_prob[idx] directly (see sample_batch '
          'code, no 1/incl_prob anywhere in ObservationSampler or train_and_evaluate).')
    print(f'  batch label counts: {np.bincount(blabels.astype(int))}')
    w1 = bweight[blabels == 1]
    w0 = bweight[blabels == 0]
    print(f'  weight for label==1 observations: n={len(w1)}  mean={w1.mean():.4f}  '
          f'min={w1.min():.4f}  max={w1.max():.4f}')
    print(f'  weight for label==0 observations: n={len(w0)}  mean={w0.mean():.4f}  '
          f'min={w0.min():.4f}  max={w0.max():.4f}')
    print(f'  ACTUAL DIRECTION: label==1 mean weight = {w1.mean():.4f}, '
          f'label==0 mean weight = {w0.mean():.4f}')
    if w1.mean() > w0.mean():
        print('  => positives get HIGHER weight than negatives. If code intended IPW '
              '(1/incl_prob), this is BACKWARDS (negatives, being the rare pool draws, should '
              'get the higher weight).')
    else:
        print('  => negatives get higher weight than positives -- matches correct IPW direction.')
    print()

    print('=' * 90)
    print('CHECK 1b: what the weight WOULD be if correctly inverted (1/incl_prob), same batch')
    print('=' * 90)
    w_correct = 1.0 / bweight
    w1c = w_correct[blabels == 1]
    w0c = w_correct[blabels == 0]
    print(f'  CORRECT (1/incl_prob) weight for label==1: mean={w1c.mean():.4f}')
    print(f'  CORRECT (1/incl_prob) weight for label==0: mean={w0c.mean():.4f}')
    print()

    print('=' * 90)
    print('CHECK 2: index alignment in ObservationSampler.sample_batch')
    print('=' * 90)
    rng2 = np.random.default_rng(42)
    idx = rng2.integers(0, sampler.n, size=BATCH_SIZE)
    labels_direct = train_labels[idx]
    weight_direct = train_incl_prob[idx]
    mask_direct   = np.asarray(train_mask[idx])
    print(f'  batch_labels matches train_labels[idx] exactly: '
          f'{np.array_equal(blabels, labels_direct)}')
    print(f'  batch_weight matches train_incl_prob[idx] exactly: '
          f'{np.array_equal(bweight, weight_direct)}')
    print(f'  batch_mask matches train_mask[idx] exactly: '
          f'{np.array_equal(np.asarray(bmask), mask_direct)}')
    print()

    print('=' * 90)
    print('CHECK 3: loss normalization / magnitude comparison')
    print('=' * 90)
    print('  uncorrected run (job 16964657) final epoch 50 loss: 0.24411  (unweighted mean BCE)')
    print('  IPW run         (job 17068340) final epoch 50 loss: 0.26755  (weighted mean BCE, '
          'buggy direct-incl_prob weight)')
    print(f'  this batch: mean(bweight) = {bweight.mean():.4f}  (used as w in the weighted-mean '
          f'formula)')
    print('  loss = (per_sample * w).sum() / w.sum() is a WEIGHTED MEAN regardless of whether w '
          'is incl_prob or 1/incl_prob -- normalization by w.sum() keeps the loss on the same '
          'BCE scale either way, so comparable magnitude (0.244 vs 0.268) is expected and '
          'uninformative about weight DIRECTION -- it only rules out a missing/extra scale '
          'factor, not a sign/inversion bug.')


if __name__ == '__main__':
    main()
