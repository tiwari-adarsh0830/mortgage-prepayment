"""
calibration_check_cutoff_2002.py -- in-sample calibration check for the
seed-42 cutoff_2002 (--include_pre2013, f0.2, L33, --use_ipw) run, in the
style of forecast_matched_population_cpr.py's per-coupon forecast-vs-realized
comparison that caught the ipw_buggy direction bug (README "IPW correction
and reproducibility" section: 90.3-99.3% forecast vs 12.2-36.0% realized).

Why this is NOT a repeat of that check
---------------------------------------
That check compared a FORECAST (from the cutoff-year model) against REALIZED
CPR in the FORECAST YEAR (cutoff_year + 1), read fresh from raw vintage files
via forecast_rolling_cpr.read_coupon_and_realized(). For cutoff_2002 that
would mean a raw pass over calendar-year-2003 pre-2013 rows -- but
read_coupon_and_realized() is hardcoded to forecast_rolling_cpr.ALL_VINTAGES
(2013Q1-2023Q1, modern-era file format only) and forecast_year = cutoff_year
+ 1 with no pre-2013 raw-file support. Building that out IS the "predict the
2003 wave" step this run's own README section (and the separate n_test
question) explicitly calls open, undone work. Doing it here would silently
collapse that open question into this one.

What this script does instead: an IN-SAMPLE check
-----------------------------------------------------
Scores the multiobs test set (test_seq.npy/test_mask.npy, cutoff_2002_zbc_
multiobs_f0.2_h1_hist -- the population Platt was actually fit on) with
evaluate()'s exact scoring path (masked-mean-pool + classifier, sigmoid),
applies the run's own Platt calibration (a=21.723925491937663,
b=-3.233141524562801, read from results.json, not re-fit here), and compares
per-coupon against the test set's OWN realized outcome -- no raw-file pass,
no out-of-window data.

Two "realized" rates are reported, not one, because they answer different
questions (see module README's IPW section on why incl_prob=1.0 for the
mandatory/terminal draw inflates the raw pooled rate):
  - raw sampled rate:   mean(label) per coupon over the AS-SAMPLED test
                         observations. Inflated above the true population
                         hazard because every prepaid loan's mandatory/
                         terminal draw is included with incl_prob=1.0 while
                         most non-mandatory (mostly negative) draws are
                         downsampled -- NOT a population estimate.
  - IPW-debiased rate:  Horvitz-Thompson, sum(label/incl_prob) /
                         sum(1/incl_prob) per coupon, undoing that inclusion
                         bias using the run's own stored incl_prob (same
                         1/incl_prob convention as train_hazard_multiobs.py's
                         ipw_weight(), after the inverted-weight bug fix).
                         This is the population-level analog of "realized
                         CPR" available without any new raw pass.

Units: H=1 label horizon means both label rate and Platt-calibrated score are
already ~1-month-hazard-scale probabilities, not annual CPR. The raw sampled
rate is NOT annualized here (1-(1-h)^12 on an inflated input would compound
the inflation into something meaningless, e.g. ~9%/mo -> ~68%/yr). Both the
Platt-calibrated forecast and the IPW-debiased realized rate ARE legitimate
population-level monthly hazards, so an annualized column is also reported
for those two only.

Coupon derivation (no raw-file lookup needed): the builder's own docstring
(build_sequences_multiobs, prepare_sequences_multiobs_zbc.py) guarantees the
LAST timestep of every window is ref_month itself. feature[0] there is
refi_incentive = original_interest_rate - PMMS_rate(ref_month), SCALED by
this run's own scaler.pkl. Unscale with scaler.mean_[0]/scale_[0], add back
PMMS_rate(ref_month) (load_pmms(), same CSV the builder used), recover
original_interest_rate, bucket into 0.5-wide coupon bins exactly as
aggregate()/mean_h_adj_by_coupon() do: round((rate-0.5)*2)/2.

Usage:
    python scripts/diag/calibration_check_cutoff_2002.py
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
# NOTE: 2.0-5.0 (the cutoff_2020/modern-era coupon range used elsewhere in
# this repo) does NOT apply here -- this is a 2000-2002 origination cohort,
# where note rates sit far higher (~5.5-9.5%, see the full per-coupon table
# printed below). Range picked from that table's own n_obs>=1000 buckets,
# not copied from the modern-era scripts.
COUPON_LO, COUPON_HI, MIN_N = 5.5, 9.5, 1000


def main():
    print(f'Device: {DEVICE}', flush=True)

    assert 'STALE' not in SEQ_DIR
    assert os.path.isdir(SEQ_DIR), f'missing seq dir: {SEQ_DIR}'

    with open(os.path.join(OUT_DIR, 'results.json')) as f:
        results = json.load(f)
    platt_a, platt_b = results['platt_a'], results['platt_b']
    assert results['seed'] == 42
    assert results['n_test'] == 227078
    print(f"Loaded results.json: platt_a={platt_a}, platt_b={platt_b}, "
          f"n_test={results['n_test']:,}, use_ipw={results['use_ipw']}", flush=True)

    # ── Load test arrays ──────────────────────────────────────────────────
    seq        = np.load(os.path.join(SEQ_DIR, 'test_seq.npy'),        mmap_mode='r')
    mask       = np.load(os.path.join(SEQ_DIR, 'test_mask.npy'),       mmap_mode='r')
    labels     = np.load(os.path.join(SEQ_DIR, 'test_labels.npy'))
    incl_prob  = np.load(os.path.join(SEQ_DIR, 'test_incl_prob.npy'))
    ref_month  = np.load(os.path.join(SEQ_DIR, 'test_ref_month.npy'))
    loan_ids   = np.load(os.path.join(SEQ_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    n = len(labels)
    assert n == 227078, f'n_test mismatch: loaded {n:,}, results.json says 227,078'
    print(f'Loaded test set: {n:,} observations, {len(set(loan_ids.tolist())):,} unique loans', flush=True)

    # ── Load checkpoint (hazard_best.pt -- Platt was fit on best_scores) ────
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

    # ── Score exactly like evaluate() ────────────────────────────────────
    raw_scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            bs = torch.tensor(np.asarray(seq[i:i + BATCH_SIZE]),  device=DEVICE)
            bm = torch.tensor(np.asarray(mask[i:i + BATCH_SIZE]), device=DEVICE)
            logits = model(bs, mask=bm)
            raw_scores[i:i + BATCH_SIZE] = torch.sigmoid(logits).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    auc_check = roc_auc_score(labels, raw_scores)
    print(f'Re-scored AUC: {auc_check:.4f} (results.json best_auc: {results["best_auc"]:.4f})', flush=True)
    assert abs(auc_check - results['best_auc']) < 1e-3, (
        'Re-scored AUC does not match results.json best_auc -- wrong checkpoint or '
        'scoring path, STOP before trusting anything below.')

    # ── Platt calibration (run's own a/b, NOT re-fit) ────────────────────
    platt_score = 1.0 / (1.0 + np.exp(-(platt_a * raw_scores + platt_b)))
    print(f'raw_scores:   mean={raw_scores.mean():.5f}  min={raw_scores.min():.5f}  max={raw_scores.max():.5f}', flush=True)
    print(f'platt_score:  mean={platt_score.mean():.5f}  min={platt_score.min():.5f}  max={platt_score.max():.5f}', flush=True)
    print(f'label rate (sampled, raw): {labels.mean():.5f}', flush=True)

    # ── Recover coupon from the last timestep's refi_incentive feature ──
    with open(os.path.join(SEQ_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean0, scale0 = scaler.mean_[0], scaler.scale_[0]

    last_feat0_scaled = np.asarray(seq[:, -1, 0])
    refi_incentive_unscaled = last_feat0_scaled * scale0 + mean0

    # load_pmms() keys are raw Fannie MMYYYY (e.g. 122021 = Dec 2021), but
    # ref_month is stored as YYYYMM (builder's out['ref_month'] =
    # out['yyyymm'], see prepare_sequences_multiobs_zbc.py L812). Re-key the
    # PMMS dict to YYYYMM via the same mmyyyy_to_yyyymm() the builder itself
    # uses, instead of converting each ref_month back to MMYYYY.
    pmms_rates_raw = load_pmms()
    pmms_rates = {mmyyyy_to_yyyymm(k): v for k, v in pmms_rates_raw.items()}
    market_rate = np.array([pmms_rates.get(int(ym), np.nan) for ym in ref_month])
    n_missing_pmms = np.isnan(market_rate).sum()
    print(f'PMMS lookup: {n_missing_pmms:,} / {n:,} ref_months missing from PMMS table', flush=True)

    note_rate = refi_incentive_unscaled + market_rate
    coupon = np.round((note_rate - 0.5) * 2) / 2

    # ── Build per-observation frame ──────────────────────────────────────
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
        # DIAGNOSTIC ONLY -- do not wire this into an actual CPR forecast without
        # re-deriving it from scratch. This pools across DIFFERENT loans within a
        # coupon bucket (the same axis aggregate() already pools realized_cpr on),
        # which is NOT the per-loan lifetime-average IPW pooling README's "IPW
        # correction and reproducibility" section abandoned as the wrong estimand --
        # but it is adjacent enough (same incl_prob, same Horvitz-Thompson math) to
        # be mistaken for a validated correction by a future reader who only skims
        # this file. It exists here solely to identify which Platt calibration (if
        # any) the reporting pipeline should be checked against -- see
        # docs/mistakes_and_lessons.md, Sep 21 2026 entry, for the full reasoning.
        ipw_debiased_rate = (g['label'] * g['ipw_weight']).sum() / g['ipw_weight'].sum()
        sum_w = g['ipw_weight'].sum()
        # forecast_raw/forecast_platt are IPW-weighted by the same w=1/incl_prob
        # as ipw_debiased_rate above, NOT a plain .mean(). An earlier version of
        # these two lines used g['platt_score'].mean() / g['raw_score'].mean()
        # (unweighted) while comparing against the IPW-weighted ipw_debiased_rate
        # -- an apples-to-oranges comparison, since the sampled test set
        # over-represents incl_prob=1 mandatory/terminal rows relative to their
        # true population share. That mismatch is what produced the reported
        # 1.23x (cutoff_2002) / 1.16x (cutoff_2020) overshoot figures; weighting
        # both sides the same way drops the pooled gap to ~1.07x/1.03x (see
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
              f'[NOT a population estimate, see module docstring]: '
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

        # Annualized view -- ONLY for the two legitimate population-level rates.
        ann = filt.copy()
        ann['forecast_platt_annual_cpr_pct'] = (1 - (1 - ann['forecast_platt_h1']) ** 12) * 100
        ann['ipw_debiased_annual_cpr_pct']    = (1 - (1 - ann['ipw_debiased_rate_h1']) ** 12) * 100
        ann['annual_ratio'] = ann['forecast_platt_annual_cpr_pct'] / ann['ipw_debiased_annual_cpr_pct']
        print(f'\n{"=" * 100}')
        print('ANNUALIZED (1-(1-h)^12), Platt-forecast vs IPW-debiased-realized ONLY '
              '(raw sampled rate excluded -- inflated by design, see module docstring)')
        print(f'{"=" * 100}')
        print(ann[['coupon', 'n_obs', 'n_loans', 'forecast_platt_annual_cpr_pct',
                    'ipw_debiased_annual_cpr_pct', 'annual_ratio']].to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
