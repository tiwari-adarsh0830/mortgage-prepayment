import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import gc

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = '/scratch/at7095/mortgage_prepayment/data/raw'
PMMS_PATH   = '/scratch/at7095/mortgage_prepayment/data/pmms_monthly.csv'
ZHVI_PATH   = '/scratch/at7095/mortgage_prepayment/data/zhvi_cbsa.csv'
OUTPUT_DIR  = '/scratch/at7095/mortgage_prepayment/outputs'
VINTAGES = [
    '2020Q1', '2020Q2', '2020Q3', '2020Q4',
    '2021Q1', '2021Q2', '2021Q3', '2021Q4',
    '2023Q1'
]
FEATURES = [
    'refi_incentive', 'borrower_credit_score',
    'original_ltv', 'current_ltv', 'original_upb', 'loan_age_months'
]

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
all_cols   = cols + extra_cols  # 109 total


# ── Load PMMS monthly rates ───────────────────────────────────────────────────
def load_pmms():
    pmms = pd.read_csv(PMMS_PATH)
    pmms = pmms[['reporting_period', 'rate_30yr']].copy()
    pmms['reporting_period'] = pmms['reporting_period'].astype(int)
    return dict(zip(pmms['reporting_period'], pmms['rate_30yr']))


# ── Load ZHVI MSA-level lookup ────────────────────────────────────────────────
def load_zhvi():
    zhvi = pd.read_csv(ZHVI_PATH)
    zhvi['cbsa'] = zhvi['cbsa'].astype(int)
    zhvi['reporting_period'] = zhvi['reporting_period'].astype(int)
    # Return dict keyed by (cbsa, reporting_period) -> zhvi
    return {(row['cbsa'], row['reporting_period']): row['zhvi']
            for _, row in zhvi.iterrows()}


# ── Load one vintage, return only what we need ────────────────────────────────
def load_vintage(vintage, pmms_rates, zhvi_lookup):
    path = os.path.join(DATA_DIR, f'{vintage}.csv')
    print(f'Loading {vintage}...', flush=True)

    usecols_idx_raw = [
        all_cols.index('loan_id') + 1,
        all_cols.index('monthly_reporting_period') + 1,
        all_cols.index('original_interest_rate') + 1,
        all_cols.index('borrower_credit_score') + 1,
        all_cols.index('original_ltv') + 1,
        all_cols.index('original_upb') + 1,
        all_cols.index('loan_age') + 1,
        all_cols.index('origination_date') + 1,
        all_cols.index('msa') + 1,
        all_cols.index('extra_13') + 1,   # zero_balance_code_actual
    ]

    col_names = [
        'loan_id', 'monthly_reporting_period', 'original_interest_rate',
        'borrower_credit_score', 'original_ltv', 'original_upb',
        'loan_age', 'origination_date', 'msa', 'zero_balance_code_actual'
    ]

    chunks = []
    for chunk in pd.read_csv(
        path, sep='|', header=None,
        usecols=usecols_idx_raw,
        low_memory=False,
        chunksize=500_000
    ):
        chunk.columns = col_names
        chunk = chunk.sort_values('monthly_reporting_period').groupby('loan_id').last().reset_index()
        chunks.append(chunk)
        del chunk
        gc.collect()

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df = df.sort_values('monthly_reporting_period').groupby('loan_id').last().reset_index()

    # Target
    df['prepaid'] = (df['zero_balance_code_actual'] == 1.0).astype(int)

    # Time-varying refi incentive
    df['market_rate']    = df['monthly_reporting_period'].map(pmms_rates)
    df['refi_incentive'] = df['original_interest_rate'] - df['market_rate']

    # Dynamic LTV using ZHVI
    df['msa'] = pd.to_numeric(df['msa'], errors='coerce').astype('Int64')
    df['origination_date'] = pd.to_numeric(df['origination_date'], errors='coerce').astype('Int64')

    df['zhvi_orig'] = df.apply(
        lambda r: zhvi_lookup.get((r['msa'], r['origination_date']), np.nan), axis=1
    )
    df['zhvi_last'] = df.apply(
        lambda r: zhvi_lookup.get((r['msa'], r['monthly_reporting_period']), np.nan), axis=1
    )

    # original_home_value = original_upb / (original_ltv / 100)
    # current_home_value  = original_home_value * (zhvi_last / zhvi_orig)
    # current_ltv         = original_upb / current_home_value * 100
    df['original_home_value'] = df['original_upb'] / (df['original_ltv'] / 100)
    df['price_appreciation']  = df['zhvi_last'] / df['zhvi_orig']
    df['current_ltv']         = (df['original_upb'] / (df['original_home_value'] * df['price_appreciation'])) * 100

    df['loan_age_months'] = df['loan_age'].astype(float)
    df['vintage']         = vintage

    keep = FEATURES + ['prepaid', 'vintage']
    df   = df[keep].dropna()

    print(f'  -> {len(df):,} loans | prepay rate: {df["prepaid"].mean()*100:.2f}% | '
          f'avg current_ltv: {df["current_ltv"].mean():.1f}', flush=True)
    return df


# ── MLP ───────────────────────────────────────────────────────────────────────
class PrepayMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load PMMS monthly rates
    print('Loading PMMS monthly rates...', flush=True)
    pmms_rates = load_pmms()
    print(f'  -> {len(pmms_rates)} monthly rate observations loaded', flush=True)

    # Load ZHVI MSA lookup
    print('Loading ZHVI MSA lookup...', flush=True)
    zhvi_lookup = load_zhvi()
    print(f'  -> {len(zhvi_lookup)} MSA-period observations loaded', flush=True)

    # Load all vintages one at a time, keeping only model-ready rows
    chunks = []
    for v in VINTAGES:
        chunks.append(load_vintage(v, pmms_rates, zhvi_lookup))
        gc.collect()

    df_all = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    print(f'\nTotal loans across all vintages: {len(df_all):,}')
    print(f'Overall prepay rate: {df_all["prepaid"].mean()*100:.2f}%')
    print(f'\nPrepay rate by vintage:')
    print(df_all.groupby('vintage')['prepaid'].mean().mul(100).round(2).to_string())

    # Train / test split
    X = df_all[FEATURES]
    y = df_all['prepaid']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression ───────────────────────────────────────────────────
    print('\nTraining Logistic Regression...', flush=True)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    results['Logistic Regression'] = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    print(f'  AUC: {results["Logistic Regression"]:.4f}')

    print('\nLogistic Regression Coefficients:')
    for feat, coef in zip(FEATURES, lr.coef_[0]):
        direction = '✅' if coef > 0 else '❌'
        print(f'  {feat}: {coef:.4f} {direction}')

    # ── Random Forest ─────────────────────────────────────────────────────────
    print('\nTraining Random Forest...', flush=True)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results['Random Forest'] = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f'  AUC: {results["Random Forest"]:.4f}')

    print('\nRandom Forest Feature Importances:')
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f'  {feat}: {imp:.4f}')

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print('\nTraining XGBoost...', flush=True)
    scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42, eval_metric='auc', verbosity=0
    )
    xgb.fit(X_train, y_train)
    results['XGBoost'] = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f'  AUC: {results["XGBoost"]:.4f}')

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print('\nTraining LightGBM...', flush=True)
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    results['LightGBM'] = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
    print(f'  AUC: {results["LightGBM"]:.4f}')

    # ── MLP ───────────────────────────────────────────────────────────────────
    print('\nTraining MLP...', flush=True)
    X_train_t = torch.FloatTensor(X_train_scaled)
    X_test_t  = torch.FloatTensor(X_test_scaled)
    y_train_t = torch.FloatTensor(y_train.values)

    pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mlp        = PrepayMLP()
    optimizer  = torch.optim.Adam(mlp.parameters(), lr=0.001)
    dataset    = TensorDataset(X_train_t, y_train_t)
    loader     = DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(20):
        mlp.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(mlp(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/20 done', flush=True)

    mlp.eval()
    with torch.no_grad():
        y_prob_mlp = mlp(X_test_t).numpy()
    results['MLP'] = roc_auc_score(y_test, y_prob_mlp)
    print(f'  AUC: {results["MLP"]:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*50)
    print('MODEL COMPARISON — Multi-Vintage + Dynamic LTV')
    print('='*50)
    for model, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f'  {model:<25} AUC: {auc:.4f}')
    print('='*50)

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'results_dynamic_ltv.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_DIR}/results_multivintage.json')


if __name__ == '__main__':
    main()
