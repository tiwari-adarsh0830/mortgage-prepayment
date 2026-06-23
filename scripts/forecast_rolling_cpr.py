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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    _ALL_COLS.index('extra_13') + 1:                 'zero_balance_code_actual',
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


def load_model(cutoff_year: int) -> PrepaymentTransformer:
    path = os.path.join(BASE, f'outputs/rolling/cutoff_{cutoff_year}/hazard_best.pt')
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get('config', {})
    m = PrepaymentTransformer(
        input_dim=cfg.get('input_dim', N_FEATURES), d_model=cfg.get('d_model', 64),
        n_heads=cfg.get('n_heads', 4), n_layers=cfg.get('n_layers', 2),
        dim_ff=cfg.get('dim_ff', 256), dropout=cfg.get('dropout', 0.1),
    ).to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    print(f'Loaded model cutoff={cutoff_year} AUC={ckpt.get("auc","?"):.4f} '
          f'device={DEVICE}', flush=True)
    return m


# ── Step 1: inference on prep TEST sequences (mmap, GPU, vectorized) ──────────
def infer_test_set(cutoff_year: int, model, batch_size: int = 8192):
    seq_dir = os.path.join(BASE, f'data/sequences_rolling/cutoff_{cutoff_year}')
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


# ── Step 2: single raw pass for coupon + realized (4 cols, Y+1 + test filter) ─
def read_coupon_and_realized(cutoff_year: int, test_id_set: set):
    """One pass over raw files. Keep only forecast-year rows for test loans.

    Returns:
      coupon_map  : {loan_id: original_interest_rate}
      active_set  : test loans appearing in any forecast-year month
      prepaid_set : test loans with zbc==1 in any forecast-year month
    """
    fy       = cutoff_year + 1
    ym_start = fy * 100 + 1
    ym_end   = fy * 100 + 12

    coupon_map  = {}
    active_set  = set()
    prepaid_set = set()

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

            active_set.update(chunk['loan_id'].tolist())
            # coupon: one rate per loan (first seen)
            for lid, rate in zip(chunk['loan_id'].values,
                                 chunk['original_interest_rate'].values):
                if lid not in coupon_map:
                    coupon_map[lid] = rate
            prepaid_set.update(
                chunk.loc[chunk['zero_balance_code_actual'] == 1.0, 'loan_id'].tolist())
            del chunk; gc.collect()
        print(f'  {vintage}: active={len(active_set):,} prepaid={len(prepaid_set):,}',
              flush=True)

    print(f'Done raw pass. active={len(active_set):,} '
          f'prepaid={len(prepaid_set):,} coupons={len(coupon_map):,}', flush=True)
    return coupon_map, active_set, prepaid_set


# ── Step 3: aggregate to coupon-level CPR ────────────────────────────────────
def aggregate(loan_ids, h_vals, coupon_map, active_set, prepaid_set):
    df = pd.DataFrame({'loan_id': loan_ids, 'h_t': h_vals})
    # restrict to loans active in the forecast year
    df = df[df['loan_id'].isin(active_set)].copy()
    df['note_rate'] = df['loan_id'].map(coupon_map)
    df = df.dropna(subset=['note_rate'])
    df['coupon']    = ((df['note_rate'] - 0.5) * 2).round() / 2
    df['realized']  = df['loan_id'].isin(prepaid_set).astype(int)
    # per-loan annual prepay prob from monthly hazard
    df['annual_pp'] = 1.0 - (1.0 - df['h_t'].clip(0, 1 - 1e-7)) ** 12

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
    args = ap.parse_args()

    out_dir  = os.path.join(BASE, f'outputs/rolling/cutoff_{args.cutoff_year}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'rolling_cpr_forecast.csv')

    print(f'=== Rolling forecast cutoff={args.cutoff_year} '
          f'→ FY{args.cutoff_year+1} | device={DEVICE} ===', flush=True)

    model = load_model(args.cutoff_year)

    print('\n[1/3] Inference on test sequences...', flush=True)
    loan_ids, h_vals = infer_test_set(args.cutoff_year, model, args.batch_size)

    test_id_set = set(loan_ids.tolist())

    print('\n[2/3] Raw pass for coupon + realized...', flush=True)
    coupon_map, active_set, prepaid_set = read_coupon_and_realized(
        args.cutoff_year, test_id_set)

    print('\n[3/3] Aggregating to coupon-level CPR...', flush=True)
    result = aggregate(loan_ids, h_vals, coupon_map, active_set, prepaid_set)
    result['cutoff_year']   = args.cutoff_year
    result['forecast_year'] = args.cutoff_year + 1
    result.to_csv(out_path, index=False)

    print(f'\nSaved: {out_path}', flush=True)
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
