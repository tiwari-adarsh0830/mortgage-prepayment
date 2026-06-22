"""
forecast_rolling_cpr.py — Rolling t→t+1 CPR forecaster.

For a model trained through Dec CUTOFF_YEAR, produces actual-loan-level
CPR forecasts for each month in Jan {Y+1} through Dec {Y+1}, and compares
against realized CPR derived from zero_balance_code_actual in the raw panel.

Design notes:
  - Loads raw vintage data once for the forecast window (last 33 months of
    history per loan through Dec Y+1), builds an in-memory panel.
  - For each forecast month t:
      input  = each active loan's feature sequence for the 33 months ending at t
      output = per-timestep hazard h_t at the last real position
    This matches the training objective: model sees history through t and
    predicts P(prepay at t | survived to t-1).
  - Coupon bucket: FNCL coupon ≈ note_rate - 0.5, rounded to nearest 0.5%.
  - CPR from SMM: CPR = 1 - (1 - SMM)^12

Usage:
    python forecast_rolling_cpr.py --cutoff_year 2018

Output (to outputs/rolling/cutoff_{Y}/):
    rolling_cpr_forecast.csv — columns: forecast_yyyymm, coupon, forecast_cpr,
                                         realized_cpr, n_loans_forecast,
                                         n_loans_realized
"""

import argparse
import json
import os
import pickle
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = '/scratch/at7095/mortgage_prepayment'

DATA_DIR  = os.path.join(BASE, 'data/raw')
PMMS_PATH = os.path.join(BASE, 'data/pmms_monthly.csv')
ZHVI_PATH = os.path.join(BASE, 'data/zhvi_zip3.csv')

MAX_SEQ    = 33
N_FEATURES = 9
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_COLS = [
    'refi_incentive', 'borrower_credit_score', 'original_ltv',
    'current_ltv', 'original_upb', 'loan_age_months',
    'dti', 'loan_purpose_enc', 'property_type_enc',
]

# Same column schema as prepare_sequences_rolling.py
_BASE_COLS = [
    'loan_id', 'monthly_reporting_period', 'channel', 'seller_name', 'servicer_name',
    'master_servicer', 'original_interest_rate', 'current_interest_rate', 'original_upb',
    'issuance_upb', 'current_actual_upb', 'original_loan_term', 'origination_date',
    'first_payment_date', 'loan_age', 'remaining_months_to_legal_maturity',
    'remaining_months_to_maturity', 'maturity_date', 'original_ltv', 'original_cltv',
    'number_of_borrowers', 'dti', 'borrower_credit_score', 'coborrower_credit_score',
    'first_time_homebuyer', 'loan_purpose', 'property_type', 'number_of_units',
    'occupancy_status', 'property_state', 'msa', 'zip', 'mortgage_insurance_percentage',
    'product_type', 'prepayment_penalty', 'interest_only',
    'first_principal_and_interest_payment_date', 'months_to_amortization',
    'current_loan_delinquency_status', 'loan_holdback', 'loan_holdback_effective_date',
    'zero_balance_code', 'zero_balance_effective_date', 'last_paid_installment_date',
    'foreclosure_date', 'disposition_date', 'foreclosure_costs',
    'property_preservation_repair_costs', 'asset_recovery_costs', 'misc_holding_expenses',
    'associated_taxes', 'net_sales_proceeds', 'credit_enhancement_proceeds',
    'repurchase_make_whole_proceeds', 'other_foreclosure_proceeds',
    'non_interest_bearing_upb', 'principal_forgiveness_amount',
    'repurchase_make_whole_proceedings_flag', 'foreclosure_principal_write_off_amount',
    'servicing_activity_indicator', 'current_deferred_upb', 'loan_due_date',
    'mi_recoveries', 'net_proceeds', 'total_expenses', 'legal_costs',
    'maintenance_preservation_costs', 'taxes_insurance', 'misc_expenses',
    'actual_loss', 'modification_flag', 'step_modification_flag',
    'payment_deferral', 'estimated_ltv', 'zero_balance_removal_upb',
    'delinquent_accrued_interest', 'disaster_related_assistance',
    'borrower_assistance_status', 'month_borrower_paid_through_date',
    'high_balance_loan', 'property_inspection_waiver', 'business_purpose_loan',
    'hi_ltv_refi_option', 'relief_refi', 'hltv_relief_refi',
    'unverified_income', 'loan_holdback_indicator', 'mi_type', 'relocation_mortgage',
    'high_ltv_refi_original_ltv', 'alternative_delinquency_resolution',
    'alternative_delinquency_resolution_count', 'total_deferral_amount',
]
_ALL_COLS = _BASE_COLS + [f'extra_{i}' for i in range(1, 17)]
_COL_MAP  = dict(sorted({
    _ALL_COLS.index('loan_id') + 1:                   'loan_id',
    _ALL_COLS.index('monthly_reporting_period') + 1:  'monthly_reporting_period',
    _ALL_COLS.index('original_interest_rate') + 1:    'original_interest_rate',
    _ALL_COLS.index('borrower_credit_score') + 1:     'borrower_credit_score',
    _ALL_COLS.index('original_ltv') + 1:              'original_ltv',
    _ALL_COLS.index('original_upb') + 1:              'original_upb',
    _ALL_COLS.index('loan_age') + 1:                  'loan_age',
    _ALL_COLS.index('origination_date') + 1:          'origination_date',
    _ALL_COLS.index('zip') + 1:                       'zip3',
    _ALL_COLS.index('extra_13') + 1:                  'zero_balance_code_actual',
    _ALL_COLS.index('dti') + 1:                       'dti',
    _ALL_COLS.index('loan_purpose') + 1:              'loan_purpose',
    _ALL_COLS.index('property_type') + 1:             'property_type',
}.items()))


# ── Date helpers ──────────────────────────────────────────────────────────────

def mmyyyy_to_yyyymm(v: int) -> int:
    s = str(int(v))
    if len(s) == 5:
        return int(s[1:]) * 100 + int(s[0])
    return int(s[2:]) * 100 + int(s[:2])


def forecast_months(cutoff_year: int) -> list[int]:
    """Return 12 YYYYMM values for Jan..Dec of cutoff_year+1."""
    y = cutoff_year + 1
    return [y * 100 + m for m in range(1, 13)]


def yyyymm_window_start(cutoff_year: int) -> int:
    """Earliest YYYYMM to load: need up to 33 months of history before Jan Y+1."""
    y = cutoff_year + 1
    # 33 months before Jan Y+1 is roughly Apr Y-2
    start_month = 1 - 33  # offset
    start_y = y + (start_month - 1) // 12
    start_m = ((start_month - 1) % 12) + 1
    return start_y * 100 + start_m


# ── Model ─────────────────────────────────────────────────────────────────────

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
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, mask=None, return_per_timestep=False):
        B, T, _ = x.shape
        pos      = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out      = self.input_proj(x) + self.pos_embedding(pos)
        pad_mask = ~mask if mask is not None else None
        out      = self.transformer(out, src_key_padding_mask=pad_mask)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)


def load_model(cutoff_year: int) -> PrepaymentTransformer:
    ckpt_path = os.path.join(BASE, f'outputs/rolling/cutoff_{cutoff_year}/hazard_best.pt')
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg  = ckpt.get('config', {})
    model = PrepaymentTransformer(
        input_dim=cfg.get('input_dim', N_FEATURES),
        d_model=cfg.get('d_model',   64),
        n_heads=cfg.get('n_heads',   4),
        n_layers=cfg.get('n_layers', 2),
        dim_ff=cfg.get('dim_ff',     256),
        dropout=cfg.get('dropout',   0.1),
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded model: cutoff={cutoff_year} | AUC={ckpt.get("auc", "?"):.4f}',
          flush=True)
    return model


# ── Data loading ──────────────────────────────────────────────────────────────

ALL_VINTAGES = [
    '2013Q1', '2013Q2', '2013Q3', '2013Q4',
    '2014Q1', '2014Q2', '2014Q3', '2014Q4',
    '2015Q1', '2015Q2', '2015Q3', '2015Q4',
    '2016Q1', '2016Q2', '2016Q3', '2016Q4',
    '2017Q1', '2017Q2', '2017Q3', '2017Q4',
    '2018Q1', '2018Q2', '2018Q3', '2018Q4',
    '2019Q1', '2019Q2', '2019Q3', '2019Q4',
    '2020Q1', '2020Q2', '2020Q3', '2020Q4',
    '2021Q1', '2021Q2', '2021Q3', '2021Q4',
    '2022Q1', '2022Q2', '2022Q3', '2022Q4',
    '2023Q1',
]


def _vintage_in_window(vintage: str, win_start: int, win_end: int) -> bool:
    """Skip vintages whose first-33-month sequence cannot reach the forecast window.

    The model was trained on each loan's FIRST 33 months. For consistency at
    inference, we only use loans whose first-33-month window overlaps the forecast
    period. A loan originated in YYYYQ# has its first 33 months ending at roughly
    orig + 32 months. For that to include any forecast month >= win_start:
        orig + 32 >= win_start  →  orig >= win_start - 32

    This cuts the file reads to only vintages originated within ~3 years of the
    forecast window start (e.g. 2018Q1–2021Q4 for a 2021 forecast), vs the old
    design that read all 37 vintage files regardless.
    """
    year = int(vintage[:4])
    q    = int(vintage[5])
    orig_yyyymm = year * 100 + (q - 1) * 3 + 1

    # First 33-month window ends at orig + 32 months (approx)
    end_year  = year + (((q - 1) * 3 + 32) // 12)
    end_month = (((q - 1) * 3 + 32) % 12) + 1
    first_window_end = end_year * 100 + end_month

    # Skip if the vintage's first window ends before our forecast starts,
    # or if the vintage originates after our forecast window ends.
    if first_window_end < win_start:
        return False
    if orig_yyyymm > win_end:
        return False
    return True


def load_panel(cutoff_year: int, pmms_rates: dict, zhvi_df: pd.DataFrame,
               scaler) -> pd.DataFrame:
    """Load raw data for the forecast window, skipping irrelevant vintage files.

    OLD: read all 37 vintage files → filter to window → slow (1hr+)
    NEW: pre-filter vintages by origination year → read only ~6-8 files → fast (<5min)
    """
    win_start = yyyymm_window_start(cutoff_year)
    win_end   = (cutoff_year + 1) * 100 + 12

    relevant = [v for v in ALL_VINTAGES if _vintage_in_window(v, win_start, win_end)]
    print(f'Loading panel for YYYYMM [{win_start}, {win_end}]...', flush=True)
    print(f'  Relevant vintages ({len(relevant)}): {relevant}', flush=True)

    chunks = []
    for vintage in relevant:
        path = os.path.join(DATA_DIR, f'{vintage}.csv')
        if not os.path.exists(path):
            continue
        print(f'  Reading {vintage}...', flush=True)
        for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=list(_COL_MAP.keys()), low_memory=False, chunksize=500_000,
        ):
            chunk.columns = list(_COL_MAP.values())
            chunk['monthly_reporting_period'] = pd.to_numeric(
                chunk['monthly_reporting_period'], errors='coerce'
            )
            chunk = chunk[chunk['monthly_reporting_period'].notna()].copy()
            chunk['yyyymm'] = chunk['monthly_reporting_period'].astype(int).apply(
                mmyyyy_to_yyyymm
            )
            chunk = chunk[
                (chunk['yyyymm'] >= win_start) & (chunk['yyyymm'] <= win_end)
            ]
            if not chunk.empty:
                chunks.append(chunk)
            del chunk; gc.collect()

    if not chunks:
        raise RuntimeError(f'No panel data loaded for cutoff_year={cutoff_year}')

    panel = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    # ── Feature engineering ───────────────────────────────────────────────────
    panel['zip3']                     = pd.to_numeric(panel['zip3'],                     errors='coerce')
    panel['origination_date']         = pd.to_numeric(panel['origination_date'],         errors='coerce')
    panel['zero_balance_code_actual'] = pd.to_numeric(panel['zero_balance_code_actual'], errors='coerce')

    panel['market_rate']    = panel['monthly_reporting_period'].map(pmms_rates)
    panel['refi_incentive'] = panel['original_interest_rate'] - panel['market_rate']

    panel = panel.merge(
        zhvi_df.rename(columns={'reporting_period': 'origination_date', 'zhvi': 'zhvi_orig'}),
        on=['zip3', 'origination_date'], how='left',
    )
    panel = panel.merge(
        zhvi_df.rename(columns={'reporting_period': 'monthly_reporting_period', 'zhvi': 'zhvi_now'}),
        on=['zip3', 'monthly_reporting_period'], how='left',
    )
    panel['original_home_value'] = panel['original_upb'] / (
        (panel['original_ltv'] / 100).replace(0, np.nan)
    )
    panel['price_appreciation'] = panel['zhvi_now'] / panel['zhvi_orig'].replace(0, np.nan)
    panel['current_ltv']        = (
        panel['original_upb'] /
        (panel['original_home_value'] * panel['price_appreciation']).replace(0, np.nan)
    ) * 100
    panel['loan_age_months'] = panel['loan_age'].astype(float)
    panel['dti']             = pd.to_numeric(panel['dti'], errors='coerce')

    panel['loan_purpose_enc']  = panel['loan_purpose'].map(
        {'R': 0, 'C': 1, 'P': 2}
    ).fillna(0).astype(float)
    panel['property_type_enc'] = panel['property_type'].map(
        {'SF': 0, 'PU': 1, 'CO': 2, 'MH': 3}
    ).fillna(0).astype(float)

    panel['coupon'] = (
        ((panel['original_interest_rate'] - 0.5) * 2).round() / 2
    )

    panel = panel.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
    print(f'Panel: {len(panel):,} rows | {panel["loan_id"].nunique():,} loans | '
          f'{panel["yyyymm"].min()}–{panel["yyyymm"].max()}', flush=True)
    return panel



# ── Per-month inference ───────────────────────────────────────────────────────

def infer_month(
    model: PrepaymentTransformer,
    panel: pd.DataFrame,
    scaler,
    forecast_yyyymm: int,
    batch_size: int = 2048,
) -> pd.DataFrame:
    """For one forecast month, extract loan sequences, run inference.

    For each loan with a row at forecast_yyyymm:
      - Build sequence from the last min(loan_history_len, 33) rows ending AT
        forecast_yyyymm (inclusive). The model was trained to predict P(prepay
        at t | sequence through t), so we include month t's features as input.
      - Run return_per_timestep=True, take h at the last real position.
      - 'realized' = 1 if zero_balance_code_actual==1 at that month.

    Returns DataFrame: loan_id, coupon, h_t (forecast hazard), realized
    """
    # Active loans at this month
    active = panel[panel['yyyymm'] == forecast_yyyymm].copy()
    if active.empty:
        return pd.DataFrame()

    loan_ids = active['loan_id'].values
    print(f'  {forecast_yyyymm}: {len(loan_ids):,} active loans', flush=True)

    # Build sequence for each loan: last MAX_SEQ rows ending at forecast_yyyymm
    # We slice from the full panel — group by loan_id, take rows ≤ forecast_yyyymm
    panel_sub  = panel[
        (panel['loan_id'].isin(set(loan_ids))) &
        (panel['yyyymm'] <= forecast_yyyymm)
    ].copy()

    # For each loan, keep last MAX_SEQ rows (chronological — panel is sorted)
    panel_sub['rev_rank'] = panel_sub.groupby('loan_id').cumcount(ascending=False)
    panel_sub = panel_sub[panel_sub['rev_rank'] < MAX_SEQ].copy()

    # Drop rows with NaN features
    valid_feat = panel_sub.dropna(subset=FEATURE_COLS)
    loan_ids_valid = valid_feat['loan_id'].unique()

    # Scale features
    feats_scaled = scaler.transform(valid_feat[FEATURE_COLS].values)
    valid_feat   = valid_feat.copy()
    valid_feat[FEATURE_COLS] = feats_scaled

    # Build padded arrays
    n = len(loan_ids_valid)
    id_to_idx = {lid: i for i, lid in enumerate(loan_ids_valid)}
    valid_feat['_idx'] = valid_feat['loan_id'].map(id_to_idx)

    # Timestep within last MAX_SEQ: compute from rev_rank (rev_rank 0 = most recent)
    # We want timestep 0 = oldest in this window, so timestep = MAX_SEQ - 1 - rev_rank
    # But actual sequence length per loan might be < MAX_SEQ.
    seq_lens = valid_feat.groupby('loan_id')['rev_rank'].max() + 1  # n_months per loan
    valid_feat['_seq_len'] = valid_feat['loan_id'].map(seq_lens)
    valid_feat['_ts']      = (valid_feat['_seq_len'] - 1 - valid_feat['rev_rank']).astype(int)

    sequences = np.zeros((n, MAX_SEQ, N_FEATURES), dtype=np.float32)
    masks     = np.zeros((n, MAX_SEQ), dtype=bool)

    idx_arr = valid_feat['_idx'].values
    ts_arr  = valid_feat['_ts'].values
    sequences[idx_arr, ts_arr, :] = valid_feat[FEATURE_COLS].values.astype(np.float32)
    masks[idx_arr, ts_arr]        = True

    # ── Batch inference ───────────────────────────────────────────────────────
    h_values = np.zeros(n, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bs = torch.tensor(sequences[i:i+batch_size], device=DEVICE)
            bm = torch.tensor(masks[i:i+batch_size],     device=DEVICE)
            logits_pt = model(bs, mask=bm, return_per_timestep=True)  # (B, T)
            h_pt      = torch.sigmoid(logits_pt).cpu().numpy()        # (B, T)
            active_m  = masks[i:i+batch_size]                         # (B, T) bool

            # Take hazard at the LAST real timestep per loan
            # last_t[j] = index of last True in masks[j]
            seq_len_batch = active_m.sum(axis=1)                       # (B,)
            last_t        = (seq_len_batch - 1).clip(min=0)           # (B,)
            for j, lt in enumerate(last_t):
                h_values[i + j] = h_pt[j, lt]

    # ── Realized prepayment ───────────────────────────────────────────────────
    realized_map = dict(
        active.set_index('loan_id')['zero_balance_code_actual'].eq(1.0)
    )
    coupon_map = dict(active.set_index('loan_id')['coupon'])

    result = pd.DataFrame({
        'loan_id':       loan_ids_valid,
        'coupon':        [coupon_map.get(lid, np.nan) for lid in loan_ids_valid],
        'h_t':           h_values,
        'realized':      [int(realized_map.get(lid, False)) for lid in loan_ids_valid],
        'forecast_yyyymm': forecast_yyyymm,
    })
    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

def smm_to_cpr(smm: float) -> float:
    """Monthly single-month mortality to annualised CPR."""
    return (1.0 - (1.0 - smm) ** 12) * 100.0


def aggregate_cpr(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-loan hazard rates to coupon-level CPR.

    CPR_c = (1 - prod_{i in c}(1 - h_{i,t})^{1/n_c})^12 * 100
    Equivalently, SMM_c = 1 - exp(mean(log(1 - h_{i,t}))) ≈ mean(h_{i,t}) for small h.
    We use the log-mean of survival probabilities for consistency with the
    chain-rule derivation (avoids arithmetic-mean overestimation at high h).
    """
    results = []
    for coupon, grp in df.groupby('coupon'):
        h        = grp['h_t'].values.clip(0, 1 - 1e-7)
        n        = len(h)
        # Geometric-mean survival → pool SMM
        log_surv = np.mean(np.log(1.0 - h))
        smm      = 1.0 - np.exp(log_surv)
        cpr      = smm_to_cpr(smm)

        n_real    = grp['realized'].sum()
        smm_real  = n_real / n  # realized monthly prepay rate
        cpr_real  = smm_to_cpr(smm_real)

        results.append({
            'coupon':            coupon,
            'forecast_cpr':      cpr,
            'realized_cpr':      cpr_real,
            'n_loans_forecast':  n,
            'n_loans_realized':  int(n_real),
        })
    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year', type=int, required=True)
    parser.add_argument('--batch_size',  type=int, default=2048)
    args = parser.parse_args()

    OUT_DIR = os.path.join(BASE, f'outputs/rolling/cutoff_{args.cutoff_year}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load model + scaler ───────────────────────────────────────────────────
    model = load_model(args.cutoff_year)

    scaler_path = os.path.join(
        BASE, f'data/sequences_rolling/cutoff_{args.cutoff_year}/scaler.pkl'
    )
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f'Loaded scaler: {scaler_path}', flush=True)

    # ── Load auxiliary data ───────────────────────────────────────────────────
    pmms = pd.read_csv(PMMS_PATH)
    pmms['reporting_period'] = pmms['reporting_period'].astype(int)
    pmms_rates = dict(zip(pmms['reporting_period'], pmms['rate_30yr']))

    zhvi_df = pd.read_csv(ZHVI_PATH)
    zhvi_df['zip3']             = zhvi_df['zip3'].astype(int)
    zhvi_df['reporting_period'] = zhvi_df['reporting_period'].astype(int)

    # ── Load raw panel for forecast window ────────────────────────────────────
    panel = load_panel(args.cutoff_year, pmms_rates, zhvi_df, scaler)

    # ── Forecast each month ───────────────────────────────────────────────────
    f_months = forecast_months(args.cutoff_year)
    print(f'\nForecast months: {f_months[0]}–{f_months[-1]}', flush=True)

    all_rows = []
    for fym in f_months:
        loan_df = infer_month(model, panel, scaler, fym, batch_size=args.batch_size)
        if loan_df.empty:
            print(f'  {fym}: no active loans — skipping', flush=True)
            continue
        agg = aggregate_cpr(loan_df)
        agg['forecast_yyyymm'] = fym
        all_rows.append(agg)

    out_path = os.path.join(OUT_DIR, 'rolling_cpr_forecast.csv')
    write_header = True
    for row_df in all_rows:
        row_df.to_csv(out_path, mode='a', header=write_header, index=False)
        write_header = False

    print(f'\nSaved: {out_path}', flush=True)
    summary = pd.read_csv(out_path)
    print(summary.groupby('coupon')[['forecast_cpr', 'realized_cpr']].mean().round(2))


if __name__ == '__main__':
    main()
