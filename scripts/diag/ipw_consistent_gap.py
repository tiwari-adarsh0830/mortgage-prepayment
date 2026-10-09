"""ipw_consistent_gap.py -- follow-up to calibration_check_cutoff_2002.py /
calibration_check_cutoff_2020_ipw_L33.py: those scripts compare an UNWEIGHTED
mean prediction (forecast_raw_h1 = g['raw_score'].mean()) against an
incl_prob-WEIGHTED realized rate (ipw_debiased_rate_h1 = sum(label*w)/sum(w),
w=1/incl_prob) -- an apples-to-oranges comparison, since the sample itself
over-represents incl_prob=1 mandatory/terminal rows relative to their true
population share. This script reweights BOTH sides by the same w=1/incl_prob
and re-derives the ratio/logit-gap, then breaks the weighted gap out by row
type: pool rows (is_terminal==False), mandatory rows of loans that prepaid
(is_terminal & label==1), and mandatory rows of censored loans (is_terminal &
label==0) -- exploiting that for any mandatory row label==is_prepaid exactly
(row_idx = term_t - H, so term_t - row_idx == H, so label = is_prepaid &
True). These row-type groups are OUTCOME-defined, not feature-defined: within
each mandatory group, realized is 0 or 1 by construction (mandatory_prepaid
always has label==1, mandatory_censored always has label==0), so a large gap
or w_share in one of those groups is arithmetic, not evidence that censoring
biases the model. It is not a valid basis for a "censored rows cause the gap"
claim; see scripts/diag/ipw_gap_feature_breakdown.py for a breakdown by
features the model can actually see (months before cutoff, loan_age) instead.
No offsets are applied to any model here -- diagnosis only.

Usage:
    python scripts/diag/ipw_consistent_gap.py --cutoff 2020
    python scripts/diag/ipw_consistent_gap.py --cutoff 2002
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_hazard_multiobs import PrepaymentTransformer, DEVICE  # noqa: E402
from prepare_sequences_multiobs_zbc import load_pmms, mmyyyy_to_yyyymm  # noqa: E402

BASE = '/scratch/at7095/mortgage_prepayment'

CFG = {
    2020: dict(
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1'),
        out_dir=os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33'),
        batch_size=8192, expected_n_test=2808691,
        coupon_lo=2.0, coupon_hi=5.0, min_n=5000,
        census_rate=7155149 / 592398205,
    ),
    2002: dict(
        seq_dir=os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist'),
        out_dir=os.path.join(BASE, 'outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist'),
        batch_size=2048, expected_n_test=227078,
        coupon_lo=5.5, coupon_hi=9.5, min_n=1000,
        census_rate=None,
    ),
}

# _seq cutoffs 2003-2008 (added 2026-10-08), 2009-2011 (added 2026-10-09): seq_dir/ckpt_dir must be passed on
# the command line, so the hardcoded-build asserts are skipped. The filtered
# block uses coupons 4.0-9.0 at n_obs>=1000 (the range the 2006-2008 one-step
# dispersion recompute kept); the UNFILTERED pooled ratio is the figure to read.
for _y in range(2003, 2012):
    CFG[_y] = dict(
        seq_dir=None, out_dir=None, batch_size=2048, expected_n_test=None,
        coupon_lo=4.0, coupon_hi=9.0, min_n=1000, census_rate=None,
    )


def logit(p):
    if p <= 0.0 or p >= 1.0:
        return float('nan')  # degenerate group (e.g. mandatory_prepaid, realized rate
                              # is trivially 1.0 by construction -- not a hazard estimate)
    return float(np.log(p / (1 - p)))


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', type=int, required=True, choices=sorted(CFG))
    ap.add_argument('--seq_dir', type=str, default=None,
                     help='Override CFG[cutoff]["seq_dir"] (e.g. a _30y build dir). When given, '
                          'the expected_n_test assert is skipped (that reference figure is only '
                          'valid for the hardcoded CFG build it was measured on).')
    ap.add_argument('--ckpt_file', type=str, default='hazard_best.pt',
                     help='Checkpoint filename inside ckpt_dir (e.g. hazard_final.pt for the last epoch).')
    ap.add_argument('--ckpt_dir', type=str, default=None,
                     help='Override CFG[cutoff]["out_dir"] (dir containing hazard_best.pt and '
                          'results.json) -- e.g. a specific seed\'s _30y checkpoint dir, so this '
                          'script can be run per-seed instead of only the CFG-hardcoded seed42.')
    args = ap.parse_args()
    cfg = CFG[args.cutoff]
    if cfg['seq_dir'] is None:
        assert args.seq_dir and args.ckpt_dir, f'cutoff {args.cutoff} needs --seq_dir and --ckpt_dir'
    overridden = args.seq_dir is not None or args.ckpt_dir is not None

    print(f'Device: {DEVICE}', flush=True)
    seq_dir = args.seq_dir or cfg['seq_dir']
    out_dir = args.ckpt_dir or cfg['out_dir']
    assert os.path.isdir(seq_dir), seq_dir
    assert os.path.isdir(out_dir), out_dir

    with open(os.path.join(out_dir, 'results.json')) as f:
        results = json.load(f)
    if not overridden:
        assert results['n_test'] == cfg['expected_n_test']
    print(f"use_ipw={results['use_ipw']} best_auc={results['best_auc']:.4f}", flush=True)

    seq        = np.load(os.path.join(seq_dir, 'test_seq.npy'),        mmap_mode='r')
    mask       = np.load(os.path.join(seq_dir, 'test_mask.npy'),       mmap_mode='r')
    labels     = np.load(os.path.join(seq_dir, 'test_labels.npy'))
    incl_prob  = np.load(os.path.join(seq_dir, 'test_incl_prob.npy'))
    is_terminal = np.load(os.path.join(seq_dir, 'test_is_terminal.npy'))
    ref_month  = np.load(os.path.join(seq_dir, 'test_ref_month.npy'))
    loan_ids   = np.load(os.path.join(seq_dir, 'test_loan_ids.npy'), allow_pickle=True)
    n = len(labels)
    if not overridden:
        assert n == cfg['expected_n_test']
    print(f'Loaded {n:,} test observations', flush=True)

    ckpt = torch.load(os.path.join(out_dir, args.ckpt_file), map_location=DEVICE, weights_only=False)
    mcfg = ckpt['config']
    model = PrepaymentTransformer(
        input_dim=mcfg['input_dim'], d_model=mcfg['d_model'], n_heads=mcfg['n_heads'],
        n_layers=mcfg['n_layers'], dim_ff=mcfg['dim_ff'], max_seq=mcfg['max_seq'],
        dropout=mcfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded {args.ckpt_file}: epoch={ckpt['epoch']}, auc={ckpt['auc']:.4f}", flush=True)

    bs_ = cfg['batch_size']
    raw_scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, bs_):
            bseq = torch.tensor(np.asarray(seq[i:i + bs_]),  device=DEVICE)
            bmsk = torch.tensor(np.asarray(mask[i:i + bs_]), device=DEVICE)
            logits = model(bseq, mask=bmsk)
            raw_scores[i:i + bs_] = torch.sigmoid(logits).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    auc_check = roc_auc_score(labels, raw_scores)
    ref_auc = results['best_auc'] if args.ckpt_file == 'hazard_best.pt' else ckpt['auc']
    print(f'Re-scored AUC: {auc_check:.4f} ({"results.json" if args.ckpt_file == "hazard_best.pt" else "checkpoint"}: {ref_auc:.4f})', flush=True)
    assert abs(auc_check - ref_auc) < 1e-3, 'wrong checkpoint/scoring path, STOP'

    with open(os.path.join(seq_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean0, scale0 = scaler.mean_[0], scaler.scale_[0]
    refi_incentive_unscaled = np.asarray(seq[:, -1, 0]) * scale0 + mean0

    pmms_rates_raw = load_pmms()
    pmms_rates = {mmyyyy_to_yyyymm(k): v for k, v in pmms_rates_raw.items()}
    market_rate = pd.Series(ref_month).map(pmms_rates).to_numpy()
    note_rate = refi_incentive_unscaled + market_rate
    coupon = np.round((note_rate - 0.5) * 2) / 2

    w = 1.0 / incl_prob
    df = pd.DataFrame({
        'loan_id': loan_ids, 'coupon': coupon, 'label': labels,
        'incl_prob': incl_prob, 'w': w, 'raw_score': raw_scores,
        'is_terminal': is_terminal.astype(bool),
    })
    df = df.dropna(subset=['coupon'])
    print(f'Dropped {n - len(df):,} obs with unrecoverable coupon', flush=True)

    # row_type: for any mandatory (is_terminal) row, label == is_prepaid exactly
    # (row_idx = term_t - H => term_t - row_idx == H => label = is_prepaid).
    def row_type(r):
        if not r['is_terminal']:
            return 'pool'
        return 'mandatory_prepaid' if r['label'] == 1 else 'mandatory_censored'
    df['row_type'] = np.where(~df['is_terminal'], 'pool',
                        np.where(df['label'] == 1, 'mandatory_prepaid', 'mandatory_censored'))

    print('\n' + '=' * 100)
    print('UNFILTERED (all coupons) -- IPW-weighted BOTH sides, full test set')
    print('=' * 100)
    pred_w_full = wmean(df['raw_score'], df['w'])
    real_w_full = wmean(df['label'], df['w'])
    print(f'n={len(df):,}  ipw_weighted_mean_pred={pred_w_full:.6f}  '
          f'ipw_weighted_realized={real_w_full:.6f}  ratio={pred_w_full / real_w_full:.4f}  '
          f'logit_gap={logit(pred_w_full) - logit(real_w_full):+.4f}', flush=True)
    if cfg['census_rate'] is not None:
        cr = cfg['census_rate']
        print(f'census_rate={cr:.6f}  '
              f'ipw_weighted_mean_pred / census_rate = {pred_w_full / cr:.4f}  '
              f'logit_gap(pred, census) = {logit(pred_w_full) - logit(cr):+.4f}  '
              f'logit_gap(ipw_realized, census) = {logit(real_w_full) - logit(cr):+.4f}', flush=True)

    lo, hi, min_n = cfg['coupon_lo'], cfg['coupon_hi'], cfg['min_n']
    n_by_coupon = df.groupby('coupon').size()
    keep_coupons = n_by_coupon[(n_by_coupon >= min_n)].index
    filt = df[(df['coupon'] >= lo) & (df['coupon'] <= hi) & (df['coupon'].isin(keep_coupons))]
    print('\n' + '=' * 100)
    print(f'FILTERED (coupon {lo}-{hi}, n_obs>={min_n} per coupon) -- matches old primary-comparison population')
    print('=' * 100)
    pred_w = wmean(filt['raw_score'], filt['w'])
    real_w = wmean(filt['label'], filt['w'])
    old_unweighted_pred = filt['raw_score'].mean()
    print(f'n={len(filt):,}')
    print(f'OLD (unweighted pred vs IPW-weighted realized): '
          f'pred_mean={old_unweighted_pred:.6f}  realized_w={real_w:.6f}  '
          f'ratio={old_unweighted_pred / real_w:.4f}  logit_gap={logit(old_unweighted_pred) - logit(real_w):+.4f}')
    print(f'NEW (IPW-weighted pred vs IPW-weighted realized): '
          f'pred_w={pred_w:.6f}  realized_w={real_w:.6f}  '
          f'ratio={pred_w / real_w:.4f}  logit_gap={logit(pred_w) - logit(real_w):+.4f}')

    print('\n' + '=' * 100)
    print('BREAKDOWN BY ROW TYPE (filtered population, IPW-weighted both sides) -- '
          'OUTCOME-defined groups: real_w is 0 or 1 by construction in the two '
          'mandatory groups, so this is not evidence about the model')
    print('=' * 100)
    total_gap_num = None
    rows_out = []
    for rt, g in filt.groupby('row_type'):
        if g['w'].sum() == 0 or len(g) == 0:
            continue
        pw = wmean(g['raw_score'], g['w'])
        rw = wmean(g['label'], g['w'])
        gap = logit(pw) - logit(rw)
        weight_share = g['w'].sum() / filt['w'].sum()
        rows_out.append((rt, len(g), g['w'].sum(), pw, rw, gap, weight_share))
    print(f"{'row_type':<20}{'n':>10}{'sum_w':>16}{'pred_w':>12}{'real_w':>12}{'logit_gap':>12}{'w_share':>10}")
    for rt, n_, sw, pw, rw, gap, ws in rows_out:
        print(f'{rt:<20}{n_:>10,}{sw:>16.1f}{pw:>12.6f}{rw:>12.6f}{gap:>+12.4f}{ws:>10.4f}')

    # Sanity check only, NOT a causal decomposition: express the pooled pred
    # and pooled realized as w-share-weighted sums of the per-group pred/
    # realized, and confirm they reconstruct the pooled values exactly (true
    # by construction of a weighted mean, for any grouping). This does not
    # "attribute" the gap to any group in a meaningful sense -- the row-type
    # groups above are OUTCOME-defined (real_w trivially 0 or 1 in the two
    # mandatory groups), so a large per-group gap or weight share reflects
    # that construction, not something the model got wrong for that group.
    total_w = filt['w'].sum()
    recon_pred = sum(sw * pw for _, _, sw, pw, rw, gap, ws in rows_out) / total_w
    recon_real = sum(sw * rw for _, _, sw, pw, rw, gap, ws in rows_out) / total_w
    print(f'\nSanity check -- reconstructed pooled pred={recon_pred:.6f} (actual {pred_w:.6f}), '
          f'reconstructed pooled realized={recon_real:.6f} (actual {real_w:.6f})')

    print('\nDone.')


if __name__ == '__main__':
    main()
