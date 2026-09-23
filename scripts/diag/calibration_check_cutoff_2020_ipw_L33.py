"""
calibration_check_cutoff_2020_ipw_L33.py -- IDENTICAL methodology to
calibration_check_cutoff_2002.py (same file, see its module docstring for the
full rationale on why this is an IN-SAMPLE check, the two "realized" rate
definitions, and the coupon-recovery method), pointed instead at
outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33/ -- the main,
already-reported, non-buggy modern-era multiobs+IPW result (f0.2, L33,
seed=42 default, NOT the ipw_buggy run).

Purpose: determine whether the 3-6x systematic Platt-forecast-vs-IPW-debiased
overshoot found in cutoff_2002 is specific to that run, or was already present
in this previously-reported modern-era result. No methodology changes from
the cutoff_2002 script -- only SEQ_DIR/OUT_DIR and the coupon range (this
population's own note-rate distribution, not assumed).

Usage:
    python scripts/diag/calibration_check_cutoff_2020_ipw_L33.py
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
SEQ_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1')
OUT_DIR = os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33')
BATCH_SIZE = 4096
EXPECTED_N_TEST = 2808691
# Coupon range NOT copied blind from the cutoff_2002 script or from
# forecast_matched_population_cpr.py -- picked after inspecting this run's
# own ALL-BUCKETS table printed below (n_obs >= threshold).
COUPON_LO, COUPON_HI, MIN_N = 2.0, 5.0, 5000


def main():
    print(f'Device: {DEVICE}', flush=True)
    assert 'STALE' not in SEQ_DIR
    assert os.path.isdir(SEQ_DIR), f'missing seq dir: {SEQ_DIR}'

    with open(os.path.join(OUT_DIR, 'results.json')) as f:
        results = json.load(f)
    platt_a, platt_b = results['platt_a'], results['platt_b']
    assert results['n_test'] == EXPECTED_N_TEST, (
        f"n_test mismatch: results.json says {results['n_test']:,}, expected {EXPECTED_N_TEST:,}")
    print(f"Loaded results.json: platt_a={platt_a}, platt_b={platt_b}, "
          f"n_test={results['n_test']:,}, use_ipw={results['use_ipw']}, "
          f"best_auc={results['best_auc']:.4f}", flush=True)

    seq        = np.load(os.path.join(SEQ_DIR, 'test_seq.npy'),        mmap_mode='r')
    mask       = np.load(os.path.join(SEQ_DIR, 'test_mask.npy'),       mmap_mode='r')
    labels     = np.load(os.path.join(SEQ_DIR, 'test_labels.npy'))
    incl_prob  = np.load(os.path.join(SEQ_DIR, 'test_incl_prob.npy'))
    ref_month  = np.load(os.path.join(SEQ_DIR, 'test_ref_month.npy'))
    loan_ids   = np.load(os.path.join(SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    n = len(labels)
    assert n == EXPECTED_N_TEST, f'n_test mismatch: loaded {n:,}, expected {EXPECTED_N_TEST:,}'
    print(f'Loaded test set: {n:,} observations, {len(set(loan_ids.tolist())):,} unique loans', flush=True)

    ckpt = torch.load(os.path.join(OUT_DIR, 'hazard_best.pt'), map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    model = PrepaymentTransformer(
        input_dim=cfg['input_dim'], d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], dim_ff=cfg['dim_ff'], max_seq=cfg['max_seq'],
        dropout=cfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded hazard_best.pt: epoch={ckpt['epoch']}, auc={ckpt['auc']:.4f}", flush=True)

    raw_scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            bs = torch.tensor(np.asarray(seq[i:i + BATCH_SIZE]),  device=DEVICE)
            bm = torch.tensor(np.asarray(mask[i:i + BATCH_SIZE]), device=DEVICE)
            logits = model(bs, mask=bm)
            raw_scores[i:i + BATCH_SIZE] = torch.sigmoid(logits).cpu().numpy()
            if (i // BATCH_SIZE) % 50 == 0:
                print(f'  scored {i + bs.shape[0]:,}/{n:,}', flush=True)
    from sklearn.metrics import roc_auc_score
    auc_check = roc_auc_score(labels, raw_scores)
    print(f'Re-scored AUC: {auc_check:.4f} (results.json best_auc: {results["best_auc"]:.4f})', flush=True)
    assert abs(auc_check - results['best_auc']) < 1e-3, (
        'Re-scored AUC does not match results.json best_auc -- wrong checkpoint or '
        'scoring path, STOP before trusting anything below.')

    platt_score = 1.0 / (1.0 + np.exp(-(platt_a * raw_scores + platt_b)))
    print(f'raw_scores:   mean={raw_scores.mean():.5f}  min={raw_scores.min():.5f}  max={raw_scores.max():.5f}', flush=True)
    print(f'platt_score:  mean={platt_score.mean():.5f}  min={platt_score.min():.5f}  max={platt_score.max():.5f}', flush=True)
    print(f'label rate (sampled, raw): {labels.mean():.5f}', flush=True)

    with open(os.path.join(SEQ_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean0, scale0 = scaler.mean_[0], scaler.scale_[0]

    last_feat0_scaled = np.asarray(seq[:, -1, 0])
    refi_incentive_unscaled = last_feat0_scaled * scale0 + mean0

    pmms_rates_raw = load_pmms()
    pmms_rates = {mmyyyy_to_yyyymm(k): v for k, v in pmms_rates_raw.items()}
    market_rate = pd.Series(ref_month).map(pmms_rates).to_numpy()
    n_missing_pmms = np.isnan(market_rate).sum()
    print(f'PMMS lookup: {n_missing_pmms:,} / {n:,} ref_months missing from PMMS table', flush=True)

    note_rate = refi_incentive_unscaled + market_rate
    coupon = np.round((note_rate - 0.5) * 2) / 2

    df = pd.DataFrame({
        'loan_id':      loan_ids,
        'coupon':       coupon,
        'label':        labels,
        'incl_prob':    incl_prob,
        'raw_score':    raw_scores,
        'platt_score':  platt_score,
        'ipw_weight':   1.0 / incl_prob,
    })
    df = df.dropna(subset=['coupon'])
    print(f'Dropped {n - len(df):,} obs with unrecoverable coupon (missing PMMS rate)', flush=True)
    assert not df.empty, 'All observations dropped -- coupon recovery is broken, STOP.'

    def per_coupon(g):
        n_obs = len(g)
        n_loans = g['loan_id'].nunique()
        raw_sampled_rate = g['label'].mean()
        ipw_debiased_rate = (g['label'] * g['ipw_weight']).sum() / g['ipw_weight'].sum()
        sum_w = g['ipw_weight'].sum()
        # forecast_raw/forecast_platt are IPW-weighted by the same w=1/incl_prob
        # as ipw_debiased_rate above, NOT a plain .mean(). An earlier version of
        # these two lines used g['platt_score'].mean() / g['raw_score'].mean()
        # (unweighted) while comparing against the IPW-weighted ipw_debiased_rate
        # -- an apples-to-oranges comparison, since the sampled test set
        # over-represents incl_prob=1 mandatory/terminal rows relative to their
        # true population share. That mismatch is what produced the reported
        # 1.16x (cutoff_2020) / 1.23x (cutoff_2002) overshoot figures; weighting
        # both sides the same way drops the pooled gap to ~1.03x/1.07x (see
        # scripts/diag/ipw_consistent_gap.py). See README's "cutoff_2002: seed
        # replication..." section (Sep 21-23, 2026).
        forecast_platt = (g['platt_score'] * g['ipw_weight']).sum() / sum_w
        forecast_raw = (g['raw_score'] * g['ipw_weight']).sum() / sum_w
        return pd.Series({
            'n_obs': n_obs, 'n_loans': n_loans,
            'sum_w': sum_w,
            'raw_sampled_rate_h1': raw_sampled_rate,
            'ipw_debiased_rate_h1': ipw_debiased_rate,
            'forecast_platt_h1': forecast_platt,
            'forecast_raw_h1': forecast_raw,
        })

    table = df.groupby('coupon').apply(per_coupon, include_groups=False).reset_index()
    table['ratio_platt_vs_ipw'] = table['forecast_platt_h1'] / table['ipw_debiased_rate_h1']
    table['ratio_platt_vs_raw_sampled'] = table['forecast_platt_h1'] / table['raw_sampled_rate_h1']

    print(f'\n{"=" * 100}')
    print('PER-COUPON, ALL BUCKETS (H=1 month hazard-scale probabilities, NOT annualized)')
    print(f'{"=" * 100}')
    print(table.to_string(index=False))

    filt = table[(table['coupon'] >= COUPON_LO) & (table['coupon'] <= COUPON_HI) & (table['n_obs'] >= MIN_N)]
    print(f'\n{"=" * 100}')
    print(f'FILTERED (coupon {COUPON_LO}-{COUPON_HI}, n_obs >= {MIN_N}) -- primary comparison')
    print(f'{"=" * 100}')
    print(filt.to_string(index=False))

    if not filt.empty:
        disp_ipw = filt['ratio_platt_vs_ipw']
        disp_raw = filt['ratio_platt_vs_raw_sampled']
        print(f'\nDispersion (Platt-forecast / IPW-debiased-realized), coupon-level ratio: '
              f'max={disp_ipw.max():.4f} min={disp_ipw.min():.4f}', flush=True)
        print(f'Dispersion (Platt-forecast / raw-sampled-realized), coupon-level ratio '
              f'[NOT a population estimate]: '
              f'max={disp_raw.max():.4f} min={disp_raw.min():.4f}', flush=True)

        n_total = int(filt['n_obs'].sum())
        total_w = float(filt['sum_w'].sum())
        # Pooled by sum_w (total ipw_weight per coupon), not n_obs -- consistent
        # with the per-coupon forecast_platt_h1/ipw_debiased_rate_h1 weighting
        # above. n_obs-weighting here would silently reintroduce the same
        # unweighted-vs-weighted mismatch one level up (across coupons instead
        # of within one).
        pooled_platt = float((filt['forecast_platt_h1'] * filt['sum_w']).sum() / total_w)
        pooled_ipw = float((filt['ipw_debiased_rate_h1'] * filt['sum_w']).sum() / total_w)
        print(f'\nPooled (ipw_weight-weighted): forecast_platt_h1={pooled_platt:.5f}  '
              f'ipw_debiased_h1={pooled_ipw:.5f}  ratio={pooled_platt / pooled_ipw:.4f}', flush=True)

        ann = filt.copy()
        ann['forecast_platt_annual_cpr_pct'] = (1 - (1 - ann['forecast_platt_h1']) ** 12) * 100
        ann['ipw_debiased_annual_cpr_pct']    = (1 - (1 - ann['ipw_debiased_rate_h1']) ** 12) * 100
        ann['annual_ratio'] = ann['forecast_platt_annual_cpr_pct'] / ann['ipw_debiased_annual_cpr_pct']
        print(f'\n{"=" * 100}')
        print('ANNUALIZED (1-(1-h)^12), Platt-forecast vs IPW-debiased-realized ONLY '
              '(raw sampled rate excluded -- inflated by design)')
        print(f'{"=" * 100}')
        print(ann[['coupon', 'n_obs', 'n_loans', 'forecast_platt_annual_cpr_pct',
                    'ipw_debiased_annual_cpr_pct', 'annual_ratio']].to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
