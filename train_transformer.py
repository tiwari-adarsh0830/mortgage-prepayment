import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
import json
import gc
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = '/scratch/at7095/mortgage_prepayment/data/raw'
PMMS_PATH  = '/scratch/at7095/mortgage_prepayment/data/pmms_monthly.csv'
ZHVI_PATH  = '/scratch/at7095/mortgage_prepayment/data/zhvi_zip3.csv'
OUTPUT_DIR = '/scratch/at7095/mortgage_prepayment/outputs'

VINTAGES = [
    '2020Q1', '2020Q2', '2020Q3', '2020Q4',
    '2021Q1', '2021Q2', '2021Q3', '2021Q4',
    '2023Q1'
]

MAX_SEQ_LEN  = 33
N_FEATURES   = 6
FEATURE_COLS = ['refi_incentive', 'borrower_credit_score', 'original_ltv',
                'current_ltv', 'original_upb', 'loan_age_months']

# ── Column setup ──────────────────────────────────────────────────────────────
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


# ── Load PMMS monthly rates ───────────────────────────────────────────────────
def load_pmms():
    pmms = pd.read_csv(PMMS_PATH)
    pmms['reporting_period'] = pmms['reporting_period'].astype(int)
    return dict(zip(pmms['reporting_period'], pmms['rate_30yr']))


# ── Load ZHVI zip3-level lookup ───────────────────────────────────────────────
def load_zhvi():
    zhvi = pd.read_csv(ZHVI_PATH)
    zhvi['zip3'] = zhvi['zip3'].astype(int)
    zhvi['reporting_period'] = zhvi['reporting_period'].astype(int)
    return {(row['zip3'], row['reporting_period']): row['zhvi']
            for _, row in zhvi.iterrows()}


# ── Load one vintage keeping ALL monthly rows ─────────────────────────────────
def load_vintage_sequences(vintage, pmms_rates, zhvi_lookup):
    path = os.path.join(DATA_DIR, f'{vintage}.csv')
    print(f'Loading {vintage}...', flush=True)

    col_map = {
        all_cols.index('loan_id') + 1:                   'loan_id',
        all_cols.index('monthly_reporting_period') + 1:  'monthly_reporting_period',
        all_cols.index('original_interest_rate') + 1:    'original_interest_rate',
        all_cols.index('borrower_credit_score') + 1:     'borrower_credit_score',
        all_cols.index('original_ltv') + 1:              'original_ltv',
        all_cols.index('original_upb') + 1:              'original_upb',
        all_cols.index('loan_age') + 1:                  'loan_age',
        all_cols.index('origination_date') + 1:          'origination_date',
        all_cols.index('zip') + 1:                       'zip3',
        all_cols.index('extra_13') + 1:                  'zero_balance_code_actual',
    }
    sorted_cols     = dict(sorted(col_map.items()))
    usecols_idx_raw = list(sorted_cols.keys())
    col_names       = list(sorted_cols.values())

    chunks = []
    for chunk in pd.read_csv(
        path, sep='|', header=None,
        usecols=usecols_idx_raw, low_memory=False, chunksize=500_000
    ):
        chunk.columns = col_names
        chunks.append(chunk)
        del chunk
        gc.collect()

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df = df.sort_values(['loan_id', 'monthly_reporting_period']).reset_index(drop=True)

    df['zip3']                     = pd.to_numeric(df['zip3'], errors='coerce')
    df['origination_date']         = pd.to_numeric(df['origination_date'], errors='coerce')
    df['monthly_reporting_period'] = pd.to_numeric(df['monthly_reporting_period'], errors='coerce')
    df['zero_balance_code_actual'] = pd.to_numeric(df['zero_balance_code_actual'], errors='coerce')

    # Time-varying refi incentive
    df['market_rate']    = df['monthly_reporting_period'].map(pmms_rates)
    df['refi_incentive'] = df['original_interest_rate'] - df['market_rate']

    # Time-varying current LTV — vectorized merge (no apply)
    zhvi_df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in zhvi_lookup.items()],
        columns=['zip3', 'reporting_period', 'zhvi']
    )
    df = df.merge(
        zhvi_df.rename(columns={'reporting_period': 'origination_date', 'zhvi': 'zhvi_orig'}),
        on=['zip3', 'origination_date'], how='left'
    )
    df = df.merge(
        zhvi_df.rename(columns={'reporting_period': 'monthly_reporting_period', 'zhvi': 'zhvi_now'}),
        on=['zip3', 'monthly_reporting_period'], how='left'
    )
    del zhvi_df
    gc.collect()

    df['original_home_value'] = df['original_upb'] / (df['original_ltv'] / 100)
    df['price_appreciation']  = df['zhvi_now'] / df['zhvi_orig']
    # original_upb in numerator — avoids leakage (current_actual_upb = 0 for prepaid loans)
    df['current_ltv']         = (df['original_upb'] / (df['original_home_value'] * df['price_appreciation'])) * 100
    df['loan_age_months']     = df['loan_age'].astype(float)

    # Target — prepaid if any row has zero_balance_code == 1.0
    prepaid_set   = set(df.loc[df['zero_balance_code_actual'] == 1.0, 'loan_id'].unique())
    df['prepaid'] = df['loan_id'].isin(prepaid_set).astype(int)

    keep_cols = ['loan_id', 'monthly_reporting_period', 'prepaid'] + FEATURE_COLS
    df        = df[keep_cols].dropna(subset=FEATURE_COLS)

    n_loans   = df['loan_id'].nunique()
    prepay_rt = df.groupby('loan_id')['prepaid'].first().mean() * 100
    print(f'  -> {n_loans:,} loans | prepay rate: {prepay_rt:.2f}% | rows: {len(df):,}', flush=True)
    return df


# ── Build padded sequences — vectorized ───────────────────────────────────────
def build_sequences(df, scaler):
    """
    Vectorized conversion of panel data to padded 3D arrays.
    Scaler must already be fitted before calling this function.

    Returns:
        sequences:    (n_loans, MAX_SEQ_LEN, N_FEATURES) float32
        masks:        (n_loans, MAX_SEQ_LEN) bool — True=real, False=padding
        labels:       (n_loans,) float32
        loan_ids_out: (n_loans,) array of loan IDs in sequence order
    """
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    # Assign integer index per loan
    loan_ids_unique = df['loan_id'].unique()
    loan_id_to_idx  = {lid: i for i, lid in enumerate(loan_ids_unique)}
    df['loan_idx']  = df['loan_id'].map(loan_id_to_idx)

    # Assign timestep index within each loan (0, 1, 2, ...)
    df['timestep'] = df.groupby('loan_id').cumcount()

    # Clip to MAX_SEQ_LEN
    df = df[df['timestep'] < MAX_SEQ_LEN].copy()

    n_loans   = len(loan_ids_unique)
    sequences = np.zeros((n_loans, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
    masks     = np.zeros((n_loans, MAX_SEQ_LEN), dtype=bool)
    labels    = np.zeros(n_loans, dtype=np.float32)

    # Vectorized fill
    loan_idx  = df['loan_idx'].values
    timesteps = df['timestep'].values
    sequences[loan_idx, timesteps, :] = df[FEATURE_COLS].values.astype(np.float32)
    masks[loan_idx, timesteps]        = True

    # One label per loan
    label_df = df.groupby('loan_idx')['prepaid'].first()
    labels[label_df.index.values] = label_df.values.astype(np.float32)

    return sequences, masks, labels, loan_ids_unique


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class PrepayDataset(Dataset):
    def __init__(self, sequences, masks, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.masks     = torch.BoolTensor(masks)
        self.labels    = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx], self.labels[idx]


# ── Transformer Model ─────────────────────────────────────────────────────────
class PrepayTransformer(nn.Module):
    """
    Transformer encoder for mortgage prepayment prediction.

    Architecture:
        1. Linear projection: N_FEATURES -> d_model
        2. Learnable positional encoding (one embedding per timestep position)
        3. Transformer encoder (n_layers, n_heads)
        4. Mean pooling over real (unmasked) timesteps
        5. Linear classifier -> logit (sigmoid applied externally at inference)
    """
    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()

        self.input_proj    = nn.Linear(N_FEATURES, d_model)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)   # raw logit — BCEWithLogitsLoss applies sigmoid internally
        )

    def forward(self, x, mask):
        """
        x:    (batch, seq_len, N_FEATURES)
        mask: (batch, seq_len) bool — True=real token, False=padding

        PyTorch TransformerEncoder src_key_padding_mask convention:
        True = IGNORE this position (padding), so we invert our mask.
        """
        seq_len = x.shape[1]

        x = self.input_proj(x)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        padding_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Mean pool — clamp denominator to avoid division by zero for all-padding sequences
        mask_exp = mask.unsqueeze(-1).float()
        x_pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)

        return self.classifier(x_pooled).squeeze(-1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    print('Loading PMMS monthly rates...', flush=True)
    pmms_rates = load_pmms()
    print(f'  -> {len(pmms_rates)} monthly rate observations', flush=True)

    print('Loading ZHVI zip3 lookup...', flush=True)
    zhvi_lookup = load_zhvi()
    print(f'  -> {len(zhvi_lookup)} zip3-period observations', flush=True)

    # Load all vintages keeping full sequences
    all_dfs = []
    for v in VINTAGES:
        all_dfs.append(load_vintage_sequences(v, pmms_rates, zhvi_lookup))
        gc.collect()

    df_all = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    total_loans    = df_all['loan_id'].nunique()
    overall_prepay = df_all.groupby('loan_id')['prepaid'].first().mean()
    print(f'\nTotal loans: {total_loans:,} | Prepay rate: {overall_prepay*100:.2f}%', flush=True)

    # Split at loan level — prevents rows from same loan leaking across train/test
    loan_ids       = df_all['loan_id'].unique()
    loan_labels    = df_all.groupby('loan_id')['prepaid'].first()
    aligned_labels = loan_labels.loc[loan_ids].values  # align to loan_ids order

    train_ids, test_ids = train_test_split(
        loan_ids, test_size=0.2, random_state=42, stratify=aligned_labels
    )

    train_df = df_all[df_all['loan_id'].isin(set(train_ids))].copy()
    test_df  = df_all[df_all['loan_id'].isin(set(test_ids))].copy()
    del df_all
    gc.collect()

    print(f'Train loans: {len(train_ids):,} | Test loans: {len(test_ids):,}', flush=True)

    # Fit scaler on training rows only — then apply to both train and test
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])

    # Save scaler for inference
    scaler_path = os.path.join(OUTPUT_DIR, 'transformer_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'Scaler saved to {scaler_path}', flush=True)

    # Build padded sequences
    print('\nBuilding train sequences...', flush=True)
    train_seq, train_mask, train_labels, _ = build_sequences(train_df, scaler)
    del train_df
    gc.collect()
    print(f'  Train shape: {train_seq.shape}', flush=True)

    print('Building test sequences...', flush=True)
    test_seq, test_mask, test_labels, _ = build_sequences(test_df, scaler)
    del test_df
    gc.collect()
    print(f'  Test shape: {test_seq.shape}', flush=True)

    # DataLoaders
    train_loader = DataLoader(
        PrepayDataset(train_seq, train_mask, train_labels),
        batch_size=512, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        PrepayDataset(test_seq, test_mask, test_labels),
        batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model    = PrepayTransformer(d_model=64, n_heads=4, n_layers=2, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel parameters: {n_params:,}', flush=True)

    # Loss — weight positive class for imbalance
    n_neg      = int((train_labels == 0).sum())
    n_pos      = int((train_labels == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print('\nTraining Transformer...', flush=True)
    best_auc = 0.0

    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for seqs, masks, labels in train_loader:
            seqs   = seqs.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(seqs, masks)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for seqs, masks, labels in test_loader:
                seqs  = seqs.to(device)
                masks = masks.to(device)
                probs = torch.sigmoid(model(seqs, masks)).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

        auc      = roc_auc_score(all_labels, all_probs)
        avg_loss = total_loss / len(train_loader)
        is_best  = auc > best_auc

        if is_best:
            best_auc = auc
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'auc':         best_auc,
                'config': {
                    'd_model':  64,
                    'n_heads':  4,
                    'n_layers': 2,
                    'dropout':  0.1,
                }
            }, os.path.join(OUTPUT_DIR, 'transformer_best.pt'))

        print(f'  Epoch {epoch+1:02d}/20 | loss: {avg_loss:.4f} | AUC: {auc:.4f}'
              f'{" [best]" if is_best else ""}', flush=True)

    print('\n' + '='*50)
    print('TRANSFORMER RESULT')
    print('='*50)
    print(f'  Best AUC: {best_auc:.4f}')
    print('='*50)

    results = {'Transformer': best_auc}
    with open(os.path.join(OUTPUT_DIR, 'results_transformer.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_DIR}/results_transformer.json')


if __name__ == '__main__':
    main()
