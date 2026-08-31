"""
forecast_rolling_cpr.py — Rolling t→t+1 CPR forecast (GPU, test-set, no raw panel).

DESIGN (addresses every prior failure mode):
  * Inference runs on the ALREADY-BUILT test_seq.npy from prep — no CSV reading,
    no panel, no merge, no concat.  -> cannot OOM on the inference side.
  * Test set only (held-out loans) — correct OOS population AND ~4x less data.
  * Runs on GPU if available (the model is a Transformer; CPU is the wrong tool).
  * ONE minimal raw pass (4 columns) filtered to forecast-year months + test
    loan IDs, aggregated incrementally into dict/set — never holds raw rows.
  * Vectorized last-timestep gather; progress logged by batch counter.

Methodology:
  forecast population = test loans active during the forecast year (cutoff_year+1)
  CPR_forecast(coupon) = mean over loans of [1-(1-h_t)^12] * 100
  CPR_realized(coupon) = (# loans that prepaid in the year) / (# active) * 100

Usage:
  python forecast_rolling_cpr.py --cutoff_year 2020
"""

import argparse
import os
import gc
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Reused, not redefined: prepare_sequences_rolling_zbc.py already verified these
# column positions and PMMS/ZHVI loading against real data during the label fix.
from prepare_sequences_rolling_zbc import load_pmms, load_zhvi

BASE     = '/scratch/at7095/mortgage_prepayment'
DATA_DIR = os.path.join(BASE, 'data/raw')

MAX_SEQ    = 33
N_FEATURES = 9
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Fannie column schema (file-position +1 for leading pipe) ─────────────────
_BASE_COLS = [
    'loan_id','monthly_reporting_period','channel','seller_name','servicer_name',
    'master_servicer','original_interest_rate','current_interest_rate','original_upb',
    'issuance_upb','current_actual_upb','original_loan_term','origination_date',
    'first_payment_date','loan_age','remaining_months_to_legal_maturity',
    'remaining_months_to_maturity','maturity_date','original_ltv','original_cltv',
    'number_of_borrowers','dti','borrower_credit_score','coborrower_credit_score',
    'first_time_homebuyer','loan_purpose','property_type','number_of_units',
    'occupancy_status','property_state','msa','zip','mortgage_insurance_percentage',
    'product_type','prepayment_penalty','interest_only',
    'first_principal_and_interest_payment_date','months_to_amortization',
    'current_loan_delinquency_status','loan_holdback','loan_holdback_effective_date',
    'zero_balance_code','zero_balance_effective_date','last_paid_installment_date',
    'foreclosure_date','disposition_date','foreclosure_costs',
    'property_preservation_repair_costs','asset_recovery_costs','misc_holding_expenses',
    'associated_taxes','net_sales_proceeds','credit_enhancement_proceeds',
    'repurchase_make_whole_proceeds','other_foreclosure_proceeds',
    'non_interest_bearing_upb','principal_forgiveness_amount',
    'repurchase_make_whole_proceedings_flag','foreclosure_principal_write_off_amount',
    'servicing_activity_indicator','current_deferred_upb','loan_due_date',
    'mi_recoveries','net_proceeds','total_expenses','legal_costs',
    'maintenance_preservation_costs','taxes_insurance','misc_expenses',
    'actual_loss','modification_flag','step_modification_flag',
    'payment_deferral','estimated_ltv','zero_balance_removal_upb',
    'delinquent_accrued_interest','disaster_related_assistance',
    'borrower_assistance_status','month_borrower_paid_through_date',
    'high_balance_loan','property_inspection_waiver','business_purpose_loan',
    'hi_ltv_refi_option','relief_refi','hltv_relief_refi',
    'unverified_income','loan_holdback_indicator','mi_type','relocation_mortgage',
    'high_ltv_refi_original_ltv','alternative_delinquency_resolution',
    'alternative_delinquency_resolution_count','total_deferral_amount',
]
_ALL_COLS = _BASE_COLS + [f'extra_{i}' for i in range(1, 17)]

# 4 columns only: loan_id, reporting month, note rate (for coupon), zbc (realized)
_RAW_COL_MAP = dict(sorted({
    _ALL_COLS.index('loan_id') + 1:                  'loan_id',
    _ALL_COLS.index('monthly_reporting_period') + 1: 'monthly_reporting_period',
    _ALL_COLS.index('original_interest_rate') + 1:   'original_interest_rate',
    # LABEL COLUMN -- hardcoded, do NOT switch to a name lookup.
    # extra_13 (usecols 106) is NOT the zero-balance code -- verified
    # against raw field values across 2000Q1/2012Q4/2018Q1 (see
    # prepare_sequences_rolling_zbc.py comment for the same finding).
    # The real zero-balance code is usecols 43.
    43:                                              'zero_balance_code_actual',
    # zip3 and origination_date: needed only by the --time_varying path
    # (recomputing current_ltv/loan_age per forecast month). Both sit
    # BEFORE the drift point that forced the zero_balance_code hardcode
    # above (verified in prepare_sequences_rolling_zbc.py's _COL_MAP,
    # which uses these same name-lookups to build training features) --
    # but read_coupon_and_realized() below still prints and range-checks
    # sampled values before trusting them, same discipline as the label fix.
    _ALL_COLS.index('zip') + 1:                      'zip3',
    _ALL_COLS.index('origination_date') + 1:         'origination_date',
}.items()))

ALL_VINTAGES = [
    '2013Q1','2013Q2','2013Q3','2013Q4','2014Q1','2014Q2','2014Q3','2014Q4',
    '2015Q1','2015Q2','2015Q3','2015Q4','2016Q1','2016Q2','2016Q3','2016Q4',
    '2017Q1','2017Q2','2017Q3','2017Q4','2018Q1','2018Q2','2018Q3','2018Q4',
    '2019Q1','2019Q2','2019Q3','2019Q4','2020Q1','2020Q2','2020Q3','2020Q4',
    '2021Q1','2021Q2','2021Q3','2021Q4','2022Q1','2022Q2','2022Q3','2022Q4','2023Q1',
]


def mmyyyy_to_yyyymm(v: int) -> int:
    s = str(int(v))
    if len(s) == 5:
        return int(s[1:]) * 100 + int(s[0])
    return int(s[2:]) * 100 + int(s[:2])


# ── Model (identical architecture to training) ───────────────────────────────
class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=N_FEATURES, d_model=64, n_heads=4, n_layers=2,
                 dim_ff=256, max_seq=MAX_SEQ, dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
              dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, mask=None, return_per_timestep=False):
        B, T, _ = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out  = self.input_proj(x) + self.pos_embedding(pos)
        pmask = ~mask if mask is not None else None
        out  = self.transformer(out, src_key_padding_mask=pmask)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)


def load_model(cutoff_year: int, label_suffix: str = '') -> PrepaymentTransformer:
    path = os.path.join(BASE, f'outputs/rolling/cutoff_{cutoff_year}{label_suffix}/hazard_best.pt')
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get('config', {})
    m = PrepaymentTransformer(
        input_dim=cfg.get('input_dim', N_FEATURES), d_model=cfg.get('d_model', 64),
        n_heads=cfg.get('n_heads', 4), n_layers=cfg.get('n_layers', 2),
        dim_ff=cfg.get('dim_ff', 256), dropout=cfg.get('dropout', 0.1),
        max_seq=cfg.get('max_seq', MAX_SEQ),
    ).to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    print(f'Loaded model cutoff={cutoff_year} AUC={ckpt.get("auc","?"):.4f} '
          f'device={DEVICE}', flush=True)
    return m


# ── Step 1: inference on prep TEST sequences (mmap, GPU, vectorized) ──────────
def infer_test_set(cutoff_year: int, model, batch_size: int = 8192, label_suffix: str = ''):
    seq_dir = os.path.join(BASE, f'data/sequences_rolling/cutoff_{cutoff_year}{label_suffix}')
    seqs  = np.load(os.path.join(seq_dir, 'test_seq.npy'),       mmap_mode='r')
    masks = np.load(os.path.join(seq_dir, 'test_mask.npy'),      mmap_mode='r')
    ids   = np.load(os.path.join(seq_dir, 'test_loan_ids.npy'),  allow_pickle=True)
    n = len(seqs)
    print(f'Test set: {n:,} loans  ({seqs.shape})', flush=True)

    h_vals = np.zeros(n, dtype=np.float32)
    n_batches = (n + batch_size - 1) // batch_size
    model.eval()
    with torch.no_grad():
        for b, i in enumerate(range(0, n, batch_size)):
            sb = np.ascontiguousarray(seqs[i:i+batch_size])
            mb = np.ascontiguousarray(masks[i:i+batch_size])
            bs = torch.from_numpy(sb).to(DEVICE)
            bm = torch.from_numpy(mb).to(DEVICE)
            logits = model(bs, mask=bm, return_per_timestep=True)   # (B, T)
            h_pt   = torch.sigmoid(logits)                          # (B, T)
            # last real timestep per loan = mask.sum(1)-1, vectorized gather
            seq_len = bm.sum(dim=1).clamp(min=1)                    # (B,)
            last_t  = (seq_len - 1).long()                         # (B,)
            rows    = torch.arange(bs.shape[0], device=DEVICE)
            h_last  = h_pt[rows, last_t]                            # (B,)
            h_vals[i:i+len(h_last)] = h_last.cpu().numpy()
            if b % 50 == 0 or b == n_batches - 1:
                print(f'  inference batch {b+1}/{n_batches} '
                      f'({i+len(h_last):,}/{n:,})', flush=True)
    print(f'  h_t mean={h_vals.mean():.5f}  max={h_vals.max():.4f}', flush=True)
    return ids, h_vals


def yyyymm_to_mmyyyy(yyyymm: int) -> int:
    """Inverse of mmyyyy_to_yyyymm. PMMS/ZHVI dicts are keyed by the raw
    MMYYYY-style integer (month*10000 + year), same convention Fannie's
    monthly_reporting_period uses -- NOT by the YYYYMM sort key."""
    year, month = yyyymm // 100, yyyymm % 100
    return month * 10000 + year


# ── Step 1b: time-varying inference (--time_varying) ─────────────────────────
def infer_test_set_time_varying(cutoff_year: int, model, coupon_map: dict,
                                 zip3_map: dict, origdate_map: dict,
                                 h_frozen: dict, logit_offset: float,
                                 batch_size: int = 8192, label_suffix: str = ''):
    """Recompute refi_incentive / current_ltv / loan_age per forecast month
    using contemporaneous PMMS + ZHVI, substituted into the sequence's last
    valid (already-masked) timestep -- training data still ends at cutoff,
    only the inference-time features move. Position embeddings are reused
    as-is (no retrain): each of the 12 forecast months is scored at the SAME
    last_t slot with that month's features, not appended positions.

    Falls back to the frozen single-hazard^12 extrapolation (h_frozen) for
    any loan missing zip3, origination_date, or a PMMS/ZHVI value for a given
    month -- reported as a count, never silently dropped or NaN-propagated.

    Returns: loan_ids, annual_pp (12-month cumulative prepay probability),
             n_fallback (count of loans using the frozen path for >=1 month)
    """
    seq_dir = os.path.join(BASE, f'data/sequences_rolling/cutoff_{cutoff_year}{label_suffix}')
    seqs  = np.load(os.path.join(seq_dir, 'test_seq.npy'),      mmap_mode='r')
    masks = np.load(os.path.join(seq_dir, 'test_mask.npy'),     mmap_mode='r')
    ids   = np.load(os.path.join(seq_dir, 'test_loan_ids.npy'), allow_pickle=True)
    n = len(seqs)

    with open(os.path.join(seq_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean_, scale_ = scaler.mean_, scaler.scale_   # index order == FEATURE_COLS
    IDX_INCENTIVE, IDX_LTV_NOW, IDX_LOAN_AGE = 0, 3, 5   # refi_incentive, current_ltv, loan_age_months
    IDX_LTV_ORIG = 2                                      # original_ltv (static, inverse-transform to get raw)

    pmms_rates = load_pmms()
    zhvi_df    = load_zhvi()
    zhvi_lookup = dict(zip(zip(zhvi_df['zip3'].astype(int), zhvi_df['reporting_period'].astype(int)),
                           zhvi_df['zhvi'].values))

    # Per-loan static inputs, aligned to `ids` order
    orig_rate = np.array([coupon_map.get(lid, np.nan) for lid in ids], dtype=np.float64)
    zip3      = np.array([zip3_map.get(lid, np.nan) for lid in ids], dtype=np.float64)
    origdate  = np.array([origdate_map.get(lid, np.nan) for lid in ids], dtype=np.float64)

    n_batches = (n + batch_size - 1) // batch_size
    fy = cutoff_year + 1
    forecast_yyyymms = [fy * 100 + m for m in range(1, 13)]

    surv       = np.ones(n, dtype=np.float64)
    fallback   = np.zeros(n, dtype=bool)
    model.eval()

    with torch.no_grad():
        for b, i in enumerate(range(0, n, batch_size)):
            sb = np.ascontiguousarray(seqs[i:i+batch_size]).astype(np.float32)
            mb = np.ascontiguousarray(masks[i:i+batch_size])
            bsz = sb.shape[0]
            seq_len = mb.sum(axis=1).clip(min=1)
            last_t  = (seq_len - 1).astype(int)
            rows    = np.arange(bsz)

            b_rate  = orig_rate[i:i+bsz]
            b_zip3  = zip3[i:i+bsz]
            b_od    = origdate[i:i+bsz]

            # Raw original_ltv, inverse-transformed from the (already scaled)
            # last-timestep row -- avoids a separate raw pass for a value we
            # already have in the training array.
            b_ltv_scaled = sb[rows, last_t, IDX_LTV_ORIG]
            b_ltv_raw    = b_ltv_scaled * scale_[IDX_LTV_ORIG] + mean_[IDX_LTV_ORIG]

            b_zhvi_orig = np.array([
                zhvi_lookup.get((int(z), int(d)), np.nan) if not (np.isnan(z) or np.isnan(d)) else np.nan
                for z, d in zip(b_zip3, b_od)
            ])
            orig_yyyymm = np.array([
                mmyyyy_to_yyyymm(int(d)) if not np.isnan(d) else np.nan for d in b_od
            ])

            for yyyymm_m in forecast_yyyymms:
                mmyyyy_m = yyyymm_to_mmyyyy(yyyymm_m)
                pmms_m   = pmms_rates.get(mmyyyy_m, np.nan)
                zhvi_now = np.array([
                    zhvi_lookup.get((int(z), mmyyyy_m), np.nan) if not np.isnan(z) else np.nan
                    for z in b_zip3
                ])

                incentive_m = b_rate - pmms_m
                ltv_now_m   = b_ltv_raw * b_zhvi_orig / zhvi_now
                age_m       = np.where(
                    np.isnan(orig_yyyymm), np.nan,
                    np.maximum(
                        (yyyymm_m // 100 - orig_yyyymm // 100) * 12
                        + (yyyymm_m % 100 - orig_yyyymm % 100) - 1, 0)
                )

                missing = (np.isnan(incentive_m) | np.isnan(ltv_now_m) | np.isnan(age_m))
                fallback[i:i+bsz] |= missing

                sb_m = sb.copy()
                # only overwrite where inputs are valid; missing rows keep the
                # original (frozen, cutoff-month) values at this position, and
                # are excluded from the survival product for this batch below
                valid = ~missing
                sb_m[rows[valid], last_t[valid], IDX_INCENTIVE] = (
                    (incentive_m[valid] - mean_[IDX_INCENTIVE]) / scale_[IDX_INCENTIVE])
                sb_m[rows[valid], last_t[valid], IDX_LTV_NOW] = (
                    (ltv_now_m[valid] - mean_[IDX_LTV_NOW]) / scale_[IDX_LTV_NOW])
                sb_m[rows[valid], last_t[valid], IDX_LOAN_AGE] = (
                    (age_m[valid] - mean_[IDX_LOAN_AGE]) / scale_[IDX_LOAN_AGE])

                bt = torch.from_numpy(sb_m).to(DEVICE)
                bm = torch.from_numpy(mb).to(DEVICE)
                logits = model(bt, mask=bm, return_per_timestep=True)
                h_pt   = torch.sigmoid(logits).cpu().numpy()
                h_m    = h_pt[rows, last_t]

                if logit_offset != 0.0:
                    hc = np.clip(h_m, 1e-7, 1 - 1e-7)
                    h_m = 1.0 / (1.0 + np.exp(-(np.log(hc / (1 - hc)) + logit_offset)))

                # only compound for loans with valid inputs this month;
                # for missing-input loans this month contributes nothing yet
                # -- they get the frozen fallback applied after the loop.
                surv[i:i+bsz][valid] *= (1.0 - np.clip(h_m[valid], 0, 1 - 1e-7))

            if b % 20 == 0 or b == n_batches - 1:
                print(f'  time-varying batch {b+1}/{n_batches} '
                      f'({i+bsz:,}/{n:,})', flush=True)

    annual_pp = 1.0 - surv
    # fallback: any loan missing a required static input for ANY forecast
    # month reverts entirely to the frozen single-hazard extrapolation
    for idx, lid in enumerate(ids):
        if fallback[idx]:
            annual_pp[idx] = h_frozen.get(lid, np.nan)

    n_fallback = int(fallback.sum())
    print(f'  time-varying: n={n:,}  fallback(frozen)={n_fallback:,} '
          f'({100*n_fallback/max(n,1):.2f}%)', flush=True)
    return ids, annual_pp, n_fallback


# ── Step 2: single raw pass for coupon + realized (4 cols, Y+1 + test filter) ─
def read_coupon_and_realized(cutoff_year: int, test_id_set: set):
    """One pass over raw files. Keep only forecast-year rows for test loans.

    Returns:
      coupon_map   : {loan_id: original_interest_rate}
      active_set   : test loans appearing in any forecast-year month
      prepaid_set  : test loans with zbc==1 in any forecast-year month
      zip3_map     : {loan_id: zip3}         -- only used by --time_varying
      origdate_map : {loan_id: origination_date (raw MMYYYY)} -- ditto
    """
    fy       = cutoff_year + 1
    ym_start = fy * 100 + 1
    ym_end   = fy * 100 + 12

    coupon_map  = {}
    zip3_map    = {}
    origdate_map = {}
    active_set  = set()
    prepaid_set = set()
    _checked_ranges = False

    # Only vintages whose loans could still be active in the forecast year:
    # originated on or before the cutoff year (test set was built ≤ Dec cutoff).
    relevant = [v for v in ALL_VINTAGES if int(v[:4]) <= cutoff_year]
    print(f'Raw pass over {len(relevant)} vintages for FY {fy} '
          f'[{ym_start}-{ym_end}]...', flush=True)

    for vintage in relevant:
        path = os.path.join(DATA_DIR, f'{vintage}.csv')
        if not os.path.exists(path):
            continue
        for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=list(_RAW_COL_MAP.keys()), low_memory=False, chunksize=1_000_000,
        ):
            chunk.columns = list(_RAW_COL_MAP.values())
            # filter to test loans first (hash membership) — biggest cut
            chunk = chunk[chunk['loan_id'].isin(test_id_set)]
            if chunk.empty:
                del chunk; continue
            chunk['monthly_reporting_period'] = pd.to_numeric(
                chunk['monthly_reporting_period'], errors='coerce')
            chunk = chunk[chunk['monthly_reporting_period'].notna()]
            chunk['yyyymm'] = chunk['monthly_reporting_period'].astype(np.int64).map(
                mmyyyy_to_yyyymm)
            chunk = chunk[(chunk['yyyymm'] >= ym_start) & (chunk['yyyymm'] <= ym_end)]
            if chunk.empty:
                del chunk; continue
            chunk['zero_balance_code_actual'] = pd.to_numeric(
                chunk['zero_balance_code_actual'], errors='coerce')
            chunk['zip3']            = pd.to_numeric(chunk['zip3'], errors='coerce')
            chunk['origination_date'] = pd.to_numeric(chunk['origination_date'], errors='coerce')

            if not _checked_ranges:
                # Sanity-check the two new columns before trusting them anywhere
                # downstream -- same discipline as the label-column fix. zip3
                # should look like a 3-digit prefix (1-999); origination_date
                # should parse as a valid MMYYYY/YYYYMM-ish value with month 1-12.
                zc = chunk['zip3'].dropna()
                oc = chunk['origination_date'].dropna()
                print(f'  [range check] zip3 sample: {zc.head(5).tolist()} '
                      f'min={zc.min()} max={zc.max()}', flush=True)
                print(f'  [range check] origination_date sample: {oc.head(5).tolist()}',
                      flush=True)
                assert zc.between(1, 999).mean() > 0.99, (
                    f'zip3 column looks wrong -- {zc.between(1,999).mean():.3f} '
                    f'of sampled values in [1,999]. STOP and re-derive the column '
                    f'position, do not proceed.')
                _od_month = oc.astype(np.int64).map(mmyyyy_to_yyyymm) % 100
                assert _od_month.between(1, 12).mean() > 0.99, (
                    f'origination_date column looks wrong -- decoded month out '
                    f'of [1,12] for {(1 - (_od_month.between(1,12).mean())):.3f} '
                    f'of sample. STOP and re-derive the column position.')
                _checked_ranges = True
                print('  [range check] PASSED for zip3 and origination_date.', flush=True)

            active_set.update(chunk['loan_id'].tolist())
            # coupon / zip3 / origination_date: one static value per loan (first seen)
            for lid, rate, z3, od in zip(chunk['loan_id'].values,
                                        chunk['original_interest_rate'].values,
                                        chunk['zip3'].values,
                                        chunk['origination_date'].values):
                if lid not in coupon_map:
                    coupon_map[lid] = rate
                if lid not in zip3_map and not pd.isna(z3):
                    zip3_map[lid] = z3
                if lid not in origdate_map and not pd.isna(od):
                    origdate_map[lid] = od
            prepaid_set.update(
                chunk.loc[chunk['zero_balance_code_actual'] == 1.0, 'loan_id'].tolist())
            del chunk; gc.collect()
        print(f'  {vintage}: active={len(active_set):,} prepaid={len(prepaid_set):,}',
              flush=True)

    print(f'Done raw pass. active={len(active_set):,} '
          f'prepaid={len(prepaid_set):,} coupons={len(coupon_map):,} '
          f'zip3={len(zip3_map):,} origdate={len(origdate_map):,}', flush=True)
    return coupon_map, active_set, prepaid_set, zip3_map, origdate_map


# ── Step 3: aggregate to coupon-level CPR ────────────────────────────────────
def prior_shift_offset(seq_dir: str, seed: int = 0) -> float:
    """Logit offset undoing the training sampler's oversampling of prepayments.

    HazardSampler draws half its loans from prepaid_idx, then picks ONE timestep
    uniformly per loan and labels it (t == prepay_t). Most draws in the positive
    half therefore land on non-event timesteps, so the effective positive rate is
    well below 0.5 and must be measured, not assumed.

    Returns log-odds(p_true) - log-odds(p_train), where p_true is the empirical
    per-person-month hazard on the same training arrays. No free parameters and
    nothing fitted to realized CPR -- this undoes a known sampling distortion, it
    does not fit the forecast to its target.
    """
    mask = np.asarray(np.load(os.path.join(seq_dir, 'train_mask.npy'), mmap_mode='r'))
    pt   = np.load(os.path.join(seq_dir, 'train_prepay_timestep.npy'))
    L     = mask.sum(axis=1).astype(np.int32)
    max_t = np.where(pt >= 0, pt, L - 1).astype(np.int32)
    valid = np.where(max_t >= 0)[0]
    prep  = np.where(pt >= 0)[0]

    rng = np.random.default_rng(seed)
    B   = 2_000_000
    li  = np.concatenate([rng.choice(prep, B // 2), rng.choice(valid, B // 2)])
    mt  = max_t[li]
    ts  = np.clip((rng.random(len(li)) * (mt + 1)).astype(np.int32), 0, mt)
    p_train = float((ts == pt[li]).mean())

    p_true = float((pt >= 0).sum() / L.sum())

    assert 0 < p_true < p_train < 1, f'p_true={p_true} p_train={p_train}'
    off = np.log(p_true / (1 - p_true)) - np.log(p_train / (1 - p_train))
    print(f'  prior shift: p_train={p_train:.5f} p_true={p_true:.5f} '
          f'offset={off:+.4f}', flush=True)
    return off


def aggregate(loan_ids, h_vals, coupon_map, active_set, prepaid_set,
              logit_offset: float = 0.0, already_annual: bool = False):
    """already_annual=True: h_vals is already a per-loan 12-month cumulative
    probability (the --time_varying path, offset applied per-month inside
    infer_test_set_time_varying) -- used as-is, no further (1-h)^12 or offset."""
    df = pd.DataFrame({'loan_id': loan_ids, 'h_t': h_vals})
    # restrict to loans active in the forecast year
    df = df[df['loan_id'].isin(active_set)].copy()
    df['note_rate'] = df['loan_id'].map(coupon_map)
    df = df.dropna(subset=['note_rate'])
    df['coupon']    = ((df['note_rate'] - 0.5) * 2).round() / 2
    df['realized']  = df['loan_id'].isin(prepaid_set).astype(int)
    if already_annual:
        df['annual_pp'] = df['h_t'].clip(0, 1 - 1e-7)
    else:
        # per-loan annual prepay prob from monthly hazard
        _h = df['h_t'].clip(1e-7, 1 - 1e-7)
        if logit_offset != 0.0:
            _h = 1.0 / (1.0 + np.exp(-(np.log(_h / (1 - _h)) + logit_offset)))
        df['h_adj']     = _h
        df['annual_pp'] = 1.0 - (1.0 - _h.clip(0, 1 - 1e-7)) ** 12

    rows = []
    for coupon, g in df.groupby('coupon'):
        n = len(g)
        rows.append({
            'coupon':            coupon,
            'forecast_cpr':      round(g['annual_pp'].mean() * 100, 4),
            'realized_cpr':      round(g['realized'].mean()  * 100, 4),
            'n_loans':           n,
            'n_prepaid':         int(g['realized'].sum()),
        })
    return pd.DataFrame(rows).sort_values('coupon').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff_year', type=int, required=True)
    ap.add_argument('--batch_size',  type=int, default=8192)
    ap.add_argument('--no_prior_shift', action='store_true',
                    help='Disable the sampler prior-shift correction '
                         '(reproduces the pre-2026-08-30 uncorrected path).')
    ap.add_argument('--label_suffix', type=str, default='',
                     help="e.g. '_zbc' to use a corrected-label model/sequences "
                          "without touching the original cutoff dir")
    ap.add_argument('--time_varying', action='store_true',
                     help="Recompute refi_incentive/current_ltv/loan_age per "
                          "forecast month from contemporaneous PMMS+ZHVI, "
                          "substituted into the last valid timestep, instead "
                          "of extrapolating one frozen cutoff-month hazard^12. "
                          "Falls back to the frozen value per-loan where "
                          "zip3/origination_date/PMMS/ZHVI is missing.")
    args = ap.parse_args()

    suffix = args.label_suffix + ('_tv' if args.time_varying else '')
    out_dir  = os.path.join(BASE, f'outputs/rolling/cutoff_{args.cutoff_year}{suffix}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'rolling_cpr_forecast.csv')

    print(f'=== Rolling forecast cutoff={args.cutoff_year}{args.label_suffix} '
          f'→ FY{args.cutoff_year+1} | time_varying={args.time_varying} '
          f'| device={DEVICE} ===', flush=True)

    model = load_model(args.cutoff_year, args.label_suffix)

    print('\n[1/3] Inference on test sequences (frozen baseline, also used '
          'as the time-varying path\'s fallback)...', flush=True)
    loan_ids, h_vals = infer_test_set(args.cutoff_year, model, args.batch_size, args.label_suffix)

    test_id_set = set(loan_ids.tolist())

    print('\n[2/3] Raw pass for coupon + realized...', flush=True)
    coupon_map, active_set, prepaid_set, zip3_map, origdate_map = read_coupon_and_realized(
        args.cutoff_year, test_id_set)

    seq_dir = os.path.join(
        BASE, f'data/sequences_rolling/cutoff_{args.cutoff_year}{args.label_suffix}')
    off = 0.0 if args.no_prior_shift else prior_shift_offset(seq_dir)

    if args.time_varying:
        print('\n[3/4] Building frozen fallback (offset-applied annual_pp per loan)...',
              flush=True)
        _h = np.clip(h_vals, 1e-7, 1 - 1e-7)
        if off != 0.0:
            _h = 1.0 / (1.0 + np.exp(-(np.log(_h / (1 - _h)) + off)))
        annual_pp_frozen = 1.0 - (1.0 - np.clip(_h, 0, 1 - 1e-7)) ** 12
        h_frozen = dict(zip(loan_ids, annual_pp_frozen))

        print('\n[4/4] Time-varying inference (12 monthly forward passes)...',
              flush=True)
        loan_ids, annual_pp, n_fallback = infer_test_set_time_varying(
            args.cutoff_year, model, coupon_map, zip3_map, origdate_map,
            h_frozen, logit_offset=off, batch_size=args.batch_size,
            label_suffix=args.label_suffix)
        result = aggregate(loan_ids, annual_pp, coupon_map, active_set,
                           prepaid_set, already_annual=True)
        result['n_fallback'] = n_fallback
    else:
        print('\n[3/3] Aggregating to coupon-level CPR...', flush=True)
        result = aggregate(loan_ids, h_vals, coupon_map, active_set,
                           prepaid_set, logit_offset=off)

    result['logit_offset']  = off
    result['cutoff_year']   = args.cutoff_year
    result['forecast_year'] = args.cutoff_year + 1
    result['time_varying']  = args.time_varying
    result.to_csv(out_path, index=False)

    print(f'\nSaved: {out_path}', flush=True)
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
