"""
train_zip3.py — Static models + zip3 as raw covariate
Uses vectorized ZHVI merge (same approach as prepare_sequences.py)
"""
import os, json, pickle, gc, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

DATA_DIR   = '/scratch/at7095/mortgage_prepayment/data/raw'
PMMS_PATH  = '/scratch/at7095/mortgage_prepayment/data/pmms_monthly.csv'
ZHVI_PATH  = '/scratch/at7095/mortgage_prepayment/data/zhvi_zip3.csv'
OUTPUT_DIR = '/scratch/at7095/mortgage_prepayment/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

VINTAGES = [
    '2020Q1','2020Q2','2020Q3','2020Q4',
    '2021Q1','2021Q2','2021Q3','2021Q4','2023Q1'
]

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


def load_vintage(vintage, pmms_rates, zhvi_df):
    path = os.path.join(DATA_DIR, f'{vintage}.csv')
    print(f'Loading {vintage}...', flush=True)
    chunks = []
    for chunk in pd.read_csv(path, sep='|', header=None,
                             usecols=usecols_idx, chunksize=500_000, low_memory=False):
        chunk.columns = col_names
        chunk = chunk.sample(frac=0.5, random_state=42)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    df['zip3']                     = pd.to_numeric(df['zip3'], errors='coerce')
    df['origination_date']         = pd.to_numeric(df['origination_date'], errors='coerce')
    df['monthly_reporting_period'] = pd.to_numeric(df['monthly_reporting_period'], errors='coerce')
    df['zero_balance_code_actual'] = pd.to_numeric(df['zero_balance_code_actual'], errors='coerce')
    df['original_ltv']             = pd.to_numeric(df['original_ltv'], errors='coerce')
    df['original_upb']             = pd.to_numeric(df['original_upb'], errors='coerce')
    df['original_interest_rate']   = pd.to_numeric(df['original_interest_rate'], errors='coerce')
    df['loan_age']                 = pd.to_numeric(df['loan_age'], errors='coerce')

    # PMMS refi incentive
    df['market_rate']    = df['monthly_reporting_period'].map(pmms_rates)
    df['refi_incentive'] = df['original_interest_rate'] - df['market_rate']

    # Vectorized ZHVI merge
    df = df.merge(
        zhvi_df.rename(columns={'reporting_period': 'origination_date', 'zhvi': 'zhvi_orig'}),
        on=['zip3', 'origination_date'], how='left'
    )
    df = df.merge(
        zhvi_df.rename(columns={'reporting_period': 'monthly_reporting_period', 'zhvi': 'zhvi_now'}),
        on=['zip3', 'monthly_reporting_period'], how='left'
    )
    df['original_home_value'] = df['original_upb'] / (df['original_ltv'] / 100)
    df['price_appreciation']  = df['zhvi_now'] / df['zhvi_orig']
    df['current_ltv']         = (df['original_upb'] / (df['original_home_value'] * df['price_appreciation'])) * 100
    df['loan_age_months']     = df['loan_age']

    # Target — label loan as prepaid if ANY row has zero_balance_code==1
    prepaid_set   = set(df.loc[df['zero_balance_code_actual'] == 1.0, 'loan_id'].unique())
    df['prepaid'] = df['loan_id'].isin(prepaid_set).astype(int)

    # Last observation per loan
    df = df.sort_values('monthly_reporting_period')
    df_last = df.groupby('loan_id').last().reset_index()
    df_last['vintage'] = vintage

    print(f'  -> {len(df_last):,} loans | prepay: {df_last["prepaid"].mean()*100:.2f}%', flush=True)
    return df_last


def get_models():
    return {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='auc',
            random_state=42, n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=50,
            early_stopping=True, random_state=42
        ),
    }


def run_models(X_train, X_test, y_train, y_test, label):
    print(f'\n=== {label} ===', flush=True)
    results = {}
    for name, model in get_models().items():
        needs_scale = name in ('LogisticRegression', 'MLP')
        if needs_scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_train)
            Xte = sc.transform(X_test)
        else:
            Xtr, Xte = X_train, X_test

        if name == 'XGBoost':
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            model.set_params(scale_pos_weight=neg/pos)

        model.fit(Xtr, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
        results[name] = round(auc, 4)
        print(f'  {name:<22} AUC: {auc:.4f}', flush=True)
    return results


def main():
    print('Loading PMMS...', flush=True)
    pmms = pd.read_csv(PMMS_PATH)
    pmms['reporting_period'] = pmms['reporting_period'].astype(int)
    pmms_rates = dict(zip(pmms['reporting_period'], pmms['rate_30yr']))

    print('Loading ZHVI...', flush=True)
    zhvi_df = pd.read_csv(ZHVI_PATH)
    zhvi_df['zip3'] = zhvi_df['zip3'].astype(int)
    zhvi_df['reporting_period'] = zhvi_df['reporting_period'].astype(int)

    frames = []
    for v in VINTAGES:
        frames.append(load_vintage(v, pmms_rates, zhvi_df))
        gc.collect()

    df_all = pd.concat(frames, ignore_index=True)
    print(f'\nTotal: {len(df_all):,} loans | Prepay: {df_all["prepaid"].mean()*100:.2f}%', flush=True)

    FEATURES_BASE = ['refi_incentive', 'borrower_credit_score', 'original_ltv',
                     'current_ltv', 'original_upb', 'loan_age_months']
    FEATURES_ZIP3 = FEATURES_BASE + ['zip3']

    # Train/test split at loan level
    train_ids, test_ids = train_test_split(
        df_all['loan_id'].unique(), test_size=0.2, random_state=42
    )
    df_train = df_all[df_all['loan_id'].isin(set(train_ids))]
    df_test  = df_all[df_all['loan_id'].isin(set(test_ids))]
    y_train  = df_train['prepaid'].values
    y_test   = df_test['prepaid'].values

    # Baseline
    mask_tr = df_train[FEATURES_BASE].notna().all(axis=1)
    mask_te = df_test[FEATURES_BASE].notna().all(axis=1)
    res_base = run_models(
        df_train.loc[mask_tr, FEATURES_BASE].values,
        df_test.loc[mask_te, FEATURES_BASE].values,
        y_train[mask_tr], y_test[mask_te], 'Baseline (no zip3)'
    )

    # With zip3
    mask_tr3 = df_train[FEATURES_ZIP3].notna().all(axis=1)
    mask_te3 = df_test[FEATURES_ZIP3].notna().all(axis=1)
    res_zip3 = run_models(
        df_train.loc[mask_tr3, FEATURES_ZIP3].values,
        df_test.loc[mask_te3, FEATURES_ZIP3].values,
        y_train[mask_tr3], y_test[mask_te3], 'With zip3'
    )

    print('\n=== AUC Comparison ===', flush=True)
    print(f'{"Model":<22} {"Baseline":>10} {"+ zip3":>10} {"Delta":>8}')
    print('-' * 54)
    for m in res_base:
        b = res_base[m]
        z = res_zip3.get(m, float('nan'))
        print(f'{m:<22} {b:>10.4f} {z:>10.4f} {z-b:>+8.4f}')

    output = {'baseline': res_base, 'with_zip3': res_zip3}
    with open(os.path.join(OUTPUT_DIR, 'results_zip3.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved → {OUTPUT_DIR}/results_zip3.json', flush=True)


if __name__ == '__main__':
    main()
