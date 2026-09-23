"""
incentive_scurve_check_cutoff_2002.py -- direct check of whether cutoff_2002
risks the Phase 15 "pre-2020-only model learns an INVERTED refi S-curve"
failure mode (README Phase 15, June 18-19 2026): that failure was root-caused
to the 2013-2019 TRAINING WINDOW containing no refi boom (PMMS essentially
flat/tight that whole period), so the model never saw real prepay response to
positive incentive and collapsed to ~0 hazard everywhere.

This script checks the analogous premise for cutoff_2002 on REAL data (not a
synthetic sweep): (1) how much PMMS rate variation actually occurred inside
the training window itself (calendar-truncated, same discipline as everywhere
else in this pipeline), (2) the resulting refi_incentive distribution across
real sampled observations -- in particular whether meaningfully positive
(in-the-money) incentive occurs WITHIN the window, not just after it, and (3)
whether the trained model's raw hazard actually responds to that incentive on
real data (bucketed, like Phase 15's refi-incentive-by-bin realized-CPR
check), instead of assuming the expanding-window framing rules the pathology
out by construction.

Uses TEST set only (227,078 obs, already validated in
calibration_check_cutoff_2002.py) -- same hazard_best.pt, same coupon/
incentive recovery method (last-timestep feature 0, unscaled, see that
script's docstring for why the last timestep is guaranteed = ref_month).

Usage:
    python scripts/diag/incentive_scurve_check_cutoff_2002.py
"""
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
SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist')
BATCH_SIZE = 2048


def main():
    print(f'Device: {DEVICE}', flush=True)

    # ── Part 1: PMMS variation actually present inside the training window ──
    pmms_rates_raw = load_pmms()
    pmms_rates = {mmyyyy_to_yyyymm(k): v for k, v in pmms_rates_raw.items()}
    window = {k: v for k, v in pmms_rates.items() if 200001 <= k <= 200212}
    ks = sorted(window)
    vs = [window[k] for k in ks]
    print(f'\n{"=" * 90}\nPART 1: PMMS rate trajectory WITHIN the cutoff_2002 training window '
          f'(2000-01..2002-12)\n{"=" * 90}')
    print(f'n_months={len(ks)}  min={min(vs):.3f}  max={max(vs):.3f}  '
          f'range={max(vs) - min(vs):.3f}pp')
    print(f'first 3 months: {list(zip(ks[:3], vs[:3]))}')
    print(f'last 3 months:  {list(zip(ks[-3:], vs[-3:]))}')
    print('(Phase 15 comparison point: the 2013-2019 window that produced the inverted/')
    print(' collapsed S-curve had essentially no refi boom / no comparable rate decline.)')

    # ── Load test set + model (identical to calibration_check_cutoff_2002.py) ──
    with open(os.path.join(OUT_DIR, 'results.json')) as f:
        results = json.load(f)
    seq       = np.load(os.path.join(SEQ_DIR, 'test_seq.npy'),   mmap_mode='r')
    mask      = np.load(os.path.join(SEQ_DIR, 'test_mask.npy'),  mmap_mode='r')
    labels    = np.load(os.path.join(SEQ_DIR, 'test_labels.npy'))
    incl_prob = np.load(os.path.join(SEQ_DIR, 'test_incl_prob.npy'))
    ref_month = np.load(os.path.join(SEQ_DIR, 'test_ref_month.npy'))
    n = len(labels)
    print(f'\nLoaded test set: {n:,} observations', flush=True)

    ckpt = torch.load(os.path.join(OUT_DIR, 'hazard_best.pt'), map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    model = PrepaymentTransformer(
        input_dim=cfg['input_dim'], d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], dim_ff=cfg['dim_ff'], max_seq=cfg['max_seq'],
        dropout=cfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    raw_scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            bs = torch.tensor(np.asarray(seq[i:i + BATCH_SIZE]),  device=DEVICE)
            bm = torch.tensor(np.asarray(mask[i:i + BATCH_SIZE]), device=DEVICE)
            logits = model(bs, mask=bm)
            raw_scores[i:i + BATCH_SIZE] = torch.sigmoid(logits).cpu().numpy()

    with open(os.path.join(SEQ_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean0, scale0 = scaler.mean_[0], scaler.scale_[0]
    refi_incentive = np.asarray(seq[:, -1, 0]) * scale0 + mean0

    # ── Part 2: refi_incentive distribution actually present at ref_month ──
    # (ref_month is WITHIN the window by construction -- calendar cutoff filter
    # already guarantees this, see load_vintage_filtered() -- this just measures
    # what that distribution actually looks like, not whether it's in-window.)
    print(f'\n{"=" * 90}\nPART 2: refi_incentive distribution, cutoff_2002 TEST observations '
          f'(all within-window by construction)\n{"=" * 90}')
    print(f'mean={refi_incentive.mean():.3f}  min={refi_incentive.min():.3f}  '
          f'max={refi_incentive.max():.3f}')
    pct_in_money = (refi_incentive > 0).mean() * 100
    print(f'% in-the-money (incentive > 0): {pct_in_money:.2f}%')
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        print(f'  p{int(q*100)}: {np.quantile(refi_incentive, q):.3f}')

    # ── Part 3: does the trained model's raw hazard actually rise with incentive? ──
    df = pd.DataFrame({
        'incentive': refi_incentive,
        'label': labels,
        'incl_prob': incl_prob,
        'raw_score': raw_scores,
        'ipw_weight': 1.0 / incl_prob,
    })
    bins = [-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
    df['bin'] = pd.cut(df['incentive'], bins=bins)

    def per_bin(g):
        n_obs = len(g)
        raw_sampled_rate = g['label'].mean()
        sum_w = g['ipw_weight'].sum()
        ipw_debiased_rate = (g['label'] * g['ipw_weight']).sum() / sum_w
        # model_raw_h is IPW-weighted by the same w=1/incl_prob as
        # ipw_debiased_rate above, NOT a plain .mean() -- same fix as
        # calibration_check_cutoff_2002.py/calibration_check_cutoff_2020_ipw_L33.py
        # (see scripts/diag/ipw_consistent_gap.py). An earlier unweighted
        # version of this line is what produced the "~29x" figure reported in
        # README's Sep 21 2026 S-curve section; rerun with this fix (job
        # 18376180) gives about 27.3x; shape conclusion unchanged.
        model_raw_h = (g['raw_score'] * g['ipw_weight']).sum() / sum_w
        return pd.Series({
            'n_obs': n_obs,
            'sum_w': sum_w,
            'raw_sampled_rate_h1': raw_sampled_rate,
            'ipw_debiased_rate_h1': ipw_debiased_rate,
            'model_raw_h_t': model_raw_h,
        })

    table = df.groupby('bin', observed=True).apply(per_bin, include_groups=False).reset_index()
    print(f'\n{"=" * 90}\nPART 3: model raw hazard vs incentive bin, real cutoff_2002 test data '
          f'(H=1 hazard-scale, NOT annualized)\n{"=" * 90}')
    print(table.to_string(index=False))

    n_bins = len(table)
    n_monotone_violations = int((table['model_raw_h_t'].diff().dropna() < 0).sum())
    print(f'\nOf {n_bins - 1} consecutive bin-to-bin steps in model_raw_h_t, '
          f'{n_monotone_violations} are DECREASES (non-monotone steps).')
    print(f'model_raw_h_t range: {table["model_raw_h_t"].min():.5f} - {table["model_raw_h_t"].max():.5f}')
    print(f'(Phase 15\'s collapsed-S-curve failure mode: hazard pinned near 0 (<0.001%) at every '
          f'positive-incentive bin, regardless of monotonicity elsewhere.)')

    print('\nDone.')


if __name__ == '__main__':
    main()
