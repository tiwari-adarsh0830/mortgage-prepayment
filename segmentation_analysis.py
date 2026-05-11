import os, json, pickle, glob, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR   = "/scratch/at7095/mortgage_prepayment"
DATA_DIR   = os.path.join(BASE_DIR, "data")
SEQ_DIR    = os.path.join(DATA_DIR, "sequences")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RAW_DIR    = os.path.join(DATA_DIR, "raw")

VINTAGES = [
    "2020Q1","2020Q2","2020Q3","2020Q4",
    "2021Q1","2021Q2","2021Q3","2021Q4","2023Q1",
]

PMMS_PATH = os.path.join(DATA_DIR, "pmms_monthly.csv")
ZHVI_PATH = os.path.join(DATA_DIR, "zhvi_zip3.csv")

FEATURE_COLS = ['refi_incentive', 'borrower_credit_score', 'original_ltv',
                'current_ltv', 'original_upb', 'loan_age_months']

cols = [
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
    'alternative_delinquency_resolution_count', 'total_deferral_amount'
]
extra_cols = [f'extra_{i}' for i in range(1, 17)]
all_cols   = cols + extra_cols


class PrepaymentTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(6, d_model)
        self.pos_embedding = nn.Embedding(33, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        # mask: True=real, False=padding (same as train_transformer.py)
        seq_len = x.shape[1]
        x = self.input_proj(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        padding_mask = ~mask  # invert for pytorch: True=ignore
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        mask_exp = mask.unsqueeze(-1).float()
        x_pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        return self.classifier(x_pooled).squeeze(-1)


def transformer_predict(test_seq, test_mask, model_path, batch_size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Transformer inference on: {device}", flush=True)
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model = PrepaymentTransformer(
        d_model  = cfg.get("d_model", 64),
        n_heads  = cfg.get("n_heads", 4),
        n_layers = cfg.get("n_layers", 2),
        dropout  = cfg.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    X = torch.tensor(test_seq,  dtype=torch.float32)
    M = torch.tensor(test_mask, dtype=torch.bool)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(device)
            mb = M[i:i+batch_size].to(device)
            logits = model(xb, mb)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


def xgboost_from_sequences(test_seq, y_test):
    print("  Loading train sequences for XGBoost...", flush=True)
    train_seq    = np.load(os.path.join(SEQ_DIR, "train_seq.npy"))
    train_labels = np.load(os.path.join(SEQ_DIR, "train_labels.npy"))

    def last_real(seq):
        real = np.any(seq != 0, axis=1)
        idx  = np.where(real)[0]
        return seq[idx[-1]] if len(idx) > 0 else seq[-1]

    print("  Extracting last-obs from train...", flush=True)
    X_train = np.vstack([last_real(s) for s in train_seq])
    print("  Extracting last-obs from test...", flush=True)
    X_test  = np.vstack([last_real(s) for s in test_seq])

    neg = (train_labels == 0).sum()
    pos = (train_labels == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg/pos,
        use_label_encoder=False, eval_metric="auc",
        random_state=42, n_jobs=-1
    )
    print("  Training XGBoost...", flush=True)
    xgb.fit(X_train, train_labels)
    return xgb.predict_proba(X_test)[:, 1]


def load_test_metadata(test_loan_ids):
    print("Loading metadata from raw files...", flush=True)
    test_id_set = set(test_loan_ids)

    col_map = {
        all_cols.index('loan_id') + 1:                  'loan_id',
        all_cols.index('monthly_reporting_period') + 1: 'monthly_reporting_period',
        all_cols.index('original_interest_rate') + 1:   'original_interest_rate',
        all_cols.index('borrower_credit_score') + 1:    'borrower_credit_score',
        all_cols.index('original_ltv') + 1:             'original_ltv',
        all_cols.index('original_upb') + 1:             'original_upb',
        all_cols.index('loan_age') + 1:                 'loan_age',
        all_cols.index('origination_date') + 1:         'origination_date',
        all_cols.index('zip') + 1:                      'zip3',
        all_cols.index('extra_13') + 1:                 'zero_balance_code_actual',
    }
    sorted_map  = dict(sorted(col_map.items()))
    usecols_idx = list(sorted_map.keys())
    col_names   = list(sorted_map.values())

    pmms = pd.read_csv(PMMS_PATH)
    pmms['reporting_period'] = pmms['reporting_period'].astype(int)
    pmms_dict = dict(zip(pmms['reporting_period'], pmms['rate_30yr']))

    zhvi = pd.read_csv(ZHVI_PATH)
    zhvi['zip3'] = zhvi['zip3'].astype(int)
    zhvi['reporting_period'] = zhvi['reporting_period'].astype(int)

    frames = []
    for vintage in VINTAGES:
        path = os.path.join(RAW_DIR, f'{vintage}.csv')
        print(f"  {vintage}...", flush=True)
        chunks = []
        for chunk in pd.read_csv(path, sep='|', header=None,
                                  usecols=usecols_idx, chunksize=500_000, low_memory=False):
            chunk.columns = col_names
            chunk = chunk[chunk['loan_id'].isin(test_id_set)]
            chunks.append(chunk)
        if chunks:
            vdf = pd.concat(chunks, ignore_index=True)
            vdf['vintage'] = vintage
            frames.append(vdf)

    df = pd.concat(frames, ignore_index=True)
    df['monthly_reporting_period'] = pd.to_numeric(df['monthly_reporting_period'], errors='coerce')
    df['zip3']                     = pd.to_numeric(df['zip3'], errors='coerce')
    df['origination_date']         = pd.to_numeric(df['origination_date'], errors='coerce')
    df['zero_balance_code_actual'] = pd.to_numeric(df['zero_balance_code_actual'], errors='coerce')
    df['original_ltv']             = pd.to_numeric(df['original_ltv'], errors='coerce')
    df['borrower_credit_score']    = pd.to_numeric(df['borrower_credit_score'], errors='coerce')
    df['original_upb']             = pd.to_numeric(df['original_upb'], errors='coerce')
    df['original_interest_rate']   = pd.to_numeric(df['original_interest_rate'], errors='coerce')
    df['loan_age']                 = pd.to_numeric(df['loan_age'], errors='coerce')

    df['pmms_rate']      = df['monthly_reporting_period'].map(pmms_dict)
    df['refi_incentive'] = df['original_interest_rate'] - df['pmms_rate']

    # Last observation per loan
    df = df.sort_values('monthly_reporting_period')
    df_last = df.groupby('loan_id').last().reset_index()
    df_last['loan_age_months'] = df_last['loan_age']
    df_last['prepaid'] = (df_last['zero_balance_code_actual'] == 1.0).astype(int)

    return df_last


SEGMENTS = {
    "fico_bucket": {
        "col": "borrower_credit_score",
        "bins": [0, 640, 680, 720, 760, 9999],
        "labels": ["<640","640-680","680-720","720-760","760+"],
    },
    "ltv_bucket": {
        "col": "original_ltv",
        "bins": [0, 60, 70, 80, 90, 200],
        "labels": ["<60","60-70","70-80","80-90","90+"],
    },
    "refi_bucket": {
        "col": "refi_incentive",
        "bins": [-99, -1.5, -0.5, 0.5, 1.5, 99],
        "labels": ["<-150bps","-150to-50bps","-50to+50bps","+50to+150bps",">+150bps"],
    },
    "loan_age_bucket": {
        "col": "loan_age_months",
        "bins": [0, 6, 12, 24, 34],
        "labels": ["0-6m","6-12m","12-24m","24-33m"],
    },
    "vintage": {"col": "vintage"},
}


def auc_by_segment(df_meta, y_true, prob_tf, prob_xgb):
    results = {}
    for seg_name, seg_cfg in SEGMENTS.items():
        col = seg_cfg["col"]
        if col not in df_meta.columns:
            continue
        if "bins" in seg_cfg:
            df_meta[seg_name] = pd.cut(
                df_meta[col], bins=seg_cfg["bins"],
                labels=seg_cfg["labels"], right=False
            )
            groups = seg_cfg["labels"]
        else:
            df_meta[seg_name] = df_meta[col]
            groups = sorted(df_meta[seg_name].dropna().unique())

        seg_rows = []
        print(f"\n  {seg_name}:")
        print(f"  {'Group':<18} {'N':>8} {'Prepay%':>8} {'TF AUC':>9} {'XGB AUC':>9} {'Delta':>8}")
        print("  " + "-"*65)
        for grp in groups:
            mask = df_meta[seg_name] == grp
            if mask.sum() < 50:
                continue
            yt  = y_true[mask.values]
            ptf = prob_tf[mask.values]
            pxg = prob_xgb[mask.values]
            if len(np.unique(yt)) < 2:
                continue
            auc_tf  = roc_auc_score(yt, ptf)
            auc_xgb = roc_auc_score(yt, pxg)
            row = {
                "segment": seg_name, "group": str(grp),
                "n_loans": int(mask.sum()),
                "prepay_pct": round(float(yt.mean() * 100), 2),
                "auc_transformer": round(auc_tf, 4),
                "auc_xgboost":     round(auc_xgb, 4),
                "delta":           round(auc_tf - auc_xgb, 4),
            }
            seg_rows.append(row)
            print(f"  {str(grp):<18} {row['n_loans']:>8,} {row['prepay_pct']:>7.2f}% "
                  f"{auc_tf:>9.4f} {auc_xgb:>9.4f} {auc_tf-auc_xgb:>+8.4f}")
        results[seg_name] = seg_rows
    return results


def main():
    print("Loading test sequences...", flush=True)
    test_seq    = np.load(os.path.join(SEQ_DIR, "test_seq.npy"))
    test_mask   = np.any(test_seq != 0, axis=2)
    y_test      = np.load(os.path.join(SEQ_DIR, "test_labels.npy"))
    test_loan_ids = np.load(os.path.join(SEQ_DIR, "test_loan_ids.npy"), allow_pickle=True)

    print(f"Test set: {len(test_seq):,} loans", flush=True)

    print("\nRunning Transformer inference...", flush=True)
    model_path = os.path.join(OUTPUT_DIR, "transformer_best.pt")
    prob_tf = transformer_predict(test_seq, test_mask, model_path)

    print("\nRunning XGBoost (last-obs snapshot)...", flush=True)
    prob_xgb = xgboost_from_sequences(test_seq, y_test)

    print(f"\nOverall AUC — Transformer: {roc_auc_score(y_test, prob_tf):.4f}  "
          f"XGBoost: {roc_auc_score(y_test, prob_xgb):.4f}", flush=True)

    print("\nLoading test metadata...", flush=True)
    df_meta = load_test_metadata(test_loan_ids)

    # Align metadata to test_loan_ids order
    # test_loan_ids has 981K but y_test has 979K — use only first len(y_test) ids
    # and filter to loans that have metadata
    test_loan_ids = test_loan_ids[:len(y_test)]
    df_meta_indexed = df_meta.set_index("loan_id")
    df_meta = df_meta_indexed.reindex(test_loan_ids).reset_index()

    # Find rows where metadata exists
    valid_mask = df_meta["borrower_credit_score"].notna().values
    df_meta    = df_meta[valid_mask].reset_index(drop=True)
    y_test     = y_test[valid_mask]
    prob_tf    = prob_tf[valid_mask]
    prob_xgb   = prob_xgb[valid_mask]
    print(f"Valid loans with metadata: {valid_mask.sum():,}", flush=True)

    print("\n=== Segmentation Analysis ===", flush=True)
    seg_results = auc_by_segment(df_meta, y_test, prob_tf, prob_xgb)

    out_json = os.path.join(OUTPUT_DIR, "segmentation_results.json")
    with open(out_json, "w") as f:
        json.dump(seg_results, f, indent=2)

    rows = [r for seg in seg_results.values() for r in seg]
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "segmentation_results.csv"), index=False
    )
    print(f"\nSaved → {out_json}", flush=True)


if __name__ == "__main__":
    main()
