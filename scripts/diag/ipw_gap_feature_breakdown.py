"""ipw_gap_feature_breakdown.py -- diagnosis only, no offsets applied to any
model. Extends ipw_consistent_gap.py (which showed the pooled IPW-weighted
gap is +0.0269/+0.0693 logit for cutoff_2020/cutoff_2002 test, seed 42) along
two axes:

1. --split {test,train}: same pooled IPW-weighted gap computed on the
   training set instead of test, to distinguish a fit problem (overshoot on
   train too) from sampling noise (train matches, test doesn't).
2. --feature_breakdown: groups the IPW-weighted pooled gap by two things the
   MODEL CAN SEE at scoring time -- months before the rolling cutoff
   (cutoff_ym - ref_month, in months) and loan_age (age_at_ref, unscaled) --
   as opposed to grouping by row_type (pool / mandatory_prepaid /
   mandatory_censored), which is an OUTCOME-defined split (realized is 0 or 1
   within each group by construction) and therefore cannot be used to argue
   censoring biases the model; do not repeat that reasoning here.
3. --out_dir override lets the same script point at a seed-7 (or other
   seed) checkpoint instead of the CFG default, to check whether the pooled
   gap is seed-specific.

cutoff_ym = December of the cutoff year (dec_yyyymm() in
prepare_sequences_multiobs_zbc.py: year*100+12), matching --cutoff_year at
build time. ref_month is stored as YYYYMM (builder's out['ref_month'] =
out['yyyymm']). Label horizon H=1 means the outcome window is
(ref_month, ref_month+1], so the max eligible ref_month is cutoff_ym-1 month
-- i.e. months_before_cutoff >= 1 for every row, not >= 0.

Usage:
    python scripts/diag/ipw_gap_feature_breakdown.py --cutoff 2020 --split test --feature_breakdown
    python scripts/diag/ipw_gap_feature_breakdown.py --cutoff 2002 --split test --feature_breakdown
    python scripts/diag/ipw_gap_feature_breakdown.py --cutoff 2002 --split test \
        --out_dir outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed7
    python scripts/diag/ipw_gap_feature_breakdown.py --cutoff 2020 --split train
    python scripts/diag/ipw_gap_feature_breakdown.py --cutoff 2002 --split train
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_hazard_multiobs import PrepaymentTransformer, DEVICE  # noqa: E402

BASE = '/scratch/at7095/mortgage_prepayment'

CFG = {
    2020: dict(
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1'),
        out_dir=os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33'),
        batch_size=8192, expected_n_test=2808691, expected_n_train=11230531,
        cutoff_ym=202012,
    ),
    2002: dict(
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist'),
        out_dir=os.path.join(BASE, 'outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist'),
        batch_size=2048, expected_n_test=227078, expected_n_train=906877,
        cutoff_ym=200212,
    ),
}

MONTHS_BEFORE_BINS = [(1, 1), (2, 3), (4, 12), (13, None)]
AGE_BINS = [(0, 6), (7, 12), (13, 24), (25, None)]


def logit(p):
    if p <= 0.0 or p >= 1.0:
        return float('nan')
    return float(np.log(p / (1 - p)))


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def yyyymm_to_months(ym):
    return (ym // 100) * 12 + (ym % 100)


def print_gap_row(label, n, sw, pw, rw):
    gap = logit(pw) - logit(rw)
    ratio = pw / rw if rw > 0 else float('nan')
    print(f'{label:<18}{n:>12,}{sw:>16.1f}{pw:>14.6f}{rw:>14.6f}{ratio:>10.4f}{gap:>+12.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', type=int, required=True, choices=[2002, 2020])
    ap.add_argument('--split', choices=['test', 'train'], default='test')
    ap.add_argument('--out_dir', default=None, help='override CFG out_dir (e.g. seed7 checkpoint)')
    ap.add_argument('--feature_breakdown', action='store_true')
    args = ap.parse_args()
    cfg = CFG[args.cutoff]

    print(f'Device: {DEVICE}', flush=True)
    seq_dir = cfg['seq_dir']
    out_dir = args.out_dir if args.out_dir else cfg['out_dir']
    assert os.path.isdir(seq_dir), seq_dir
    assert os.path.isdir(out_dir), out_dir
    print(f'seq_dir={seq_dir}', flush=True)
    print(f'out_dir={out_dir}', flush=True)

    with open(os.path.join(out_dir, 'results.json')) as f:
        results = json.load(f)
    print(f"use_ipw={results['use_ipw']} seed={results.get('seed')} best_auc={results['best_auc']:.4f}", flush=True)

    sp = args.split
    expected_n = cfg['expected_n_test'] if sp == 'test' else cfg['expected_n_train']
    if sp == 'test':
        assert results['n_test'] == expected_n

    seq         = np.load(os.path.join(seq_dir, f'{sp}_seq.npy'),         mmap_mode='r')
    mask        = np.load(os.path.join(seq_dir, f'{sp}_mask.npy'),        mmap_mode='r')
    labels      = np.load(os.path.join(seq_dir, f'{sp}_labels.npy'))
    incl_prob   = np.load(os.path.join(seq_dir, f'{sp}_incl_prob.npy'))
    ref_month   = np.load(os.path.join(seq_dir, f'{sp}_ref_month.npy'))
    age_at_ref  = np.load(os.path.join(seq_dir, f'{sp}_age_at_ref.npy'))
    n = len(labels)
    assert n == expected_n, f'{sp} n mismatch: loaded {n:,}, expected {expected_n:,}'
    print(f'Loaded {n:,} {sp} observations', flush=True)

    ckpt = torch.load(os.path.join(out_dir, 'hazard_best.pt'), map_location=DEVICE, weights_only=False)
    mcfg = ckpt['config']
    model = PrepaymentTransformer(
        input_dim=mcfg['input_dim'], d_model=mcfg['d_model'], n_heads=mcfg['n_heads'],
        n_layers=mcfg['n_layers'], dim_ff=mcfg['dim_ff'], max_seq=mcfg['max_seq'],
        dropout=mcfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded hazard_best.pt: epoch={ckpt['epoch']}, auc={ckpt['auc']:.4f}", flush=True)

    bs_ = cfg['batch_size']
    raw_scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, bs_):
            bseq = torch.tensor(np.asarray(seq[i:i + bs_]),  device=DEVICE)
            bmsk = torch.tensor(np.asarray(mask[i:i + bs_]), device=DEVICE)
            logits = model(bseq, mask=bmsk)
            raw_scores[i:i + bs_] = torch.sigmoid(logits).cpu().numpy()
            if (i // bs_) % 200 == 0:
                print(f'  scored {i + bmsk.shape[0]:,}/{n:,}', flush=True)
    from sklearn.metrics import roc_auc_score
    auc_check = roc_auc_score(labels, raw_scores)
    print(f'Re-scored AUC ({sp}): {auc_check:.4f} (results.json best_auc, test-set: {results["best_auc"]:.4f})', flush=True)
    if sp == 'test':
        assert abs(auc_check - results['best_auc']) < 1e-3, 'wrong checkpoint/scoring path, STOP'

    w = 1.0 / incl_prob
    months_before = yyyymm_to_months(cfg['cutoff_ym']) - yyyymm_to_months(ref_month)
    df = pd.DataFrame({
        'label': labels, 'w': w, 'raw_score': raw_scores,
        'months_before_cutoff': months_before,
        'loan_age': age_at_ref,
    })

    print('\n' + '=' * 100)
    print(f'POOLED IPW-WEIGHTED GAP -- cutoff={args.cutoff} split={sp} out_dir={os.path.basename(out_dir)}')
    print('=' * 100)
    pred_w = wmean(df['raw_score'], df['w'])
    real_w = wmean(df['label'], df['w'])
    print(f'n={len(df):,}  ipw_weighted_mean_pred={pred_w:.6f}  ipw_weighted_realized={real_w:.6f}  '
          f'ratio={pred_w / real_w:.4f}  logit_gap={logit(pred_w) - logit(real_w):+.4f}', flush=True)

    if args.feature_breakdown:
        print('\n' + '=' * 100)
        print('BY MONTHS BEFORE CUTOFF (cutoff_ym - ref_month, months) -- feature the model can see')
        print('=' * 100)
        print(f"{'bucket':<18}{'n':>12}{'sum_w':>16}{'pred_w':>14}{'real_w':>14}{'ratio':>10}{'logit_gap':>12}")
        for lo, hi in MONTHS_BEFORE_BINS:
            if hi is None:
                sel = df['months_before_cutoff'] >= lo
                label = f'{lo}+'
            else:
                sel = (df['months_before_cutoff'] >= lo) & (df['months_before_cutoff'] <= hi)
                label = f'{lo}-{hi}' if lo != hi else f'{lo}'
            g = df[sel]
            if len(g) == 0:
                print(f'{label:<18}{"(empty)":>12}')
                continue
            pw = wmean(g['raw_score'], g['w'])
            rw = wmean(g['label'], g['w'])
            print_gap_row(label, len(g), g['w'].sum(), pw, rw)

        print('\n' + '=' * 100)
        print('BY LOAN_AGE (age_at_ref, months) -- feature the model can see')
        print('=' * 100)
        print(f"{'bucket':<18}{'n':>12}{'sum_w':>16}{'pred_w':>14}{'real_w':>14}{'ratio':>10}{'logit_gap':>12}")
        for lo, hi in AGE_BINS:
            if hi is None:
                sel = df['loan_age'] >= lo
                label = f'{lo}+'
            else:
                sel = (df['loan_age'] >= lo) & (df['loan_age'] <= hi)
                label = f'{lo}-{hi}'
            g = df[sel]
            if len(g) == 0:
                print(f'{label:<18}{"(empty)":>12}')
                continue
            pw = wmean(g['raw_score'], g['w'])
            rw = wmean(g['label'], g['w'])
            print_gap_row(label, len(g), g['w'].sum(), pw, rw)

    print('\nDone.')


if __name__ == '__main__':
    main()
