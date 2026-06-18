"""
prepare_sequences_censored2019.py
---------------------------------
Builds padded sequences with CALENDAR CENSORING at Dec 2019: every observation
whose monthly_reporting_period falls in 2020 or later is dropped, truncating each
loan's sequence at year-end 2019.

Why calendar censoring (not vintage filtering): a 2018-2019 ORIGINATION still has
33 months of history that run into 2020-21. Filtering by vintage alone leaves the
2020-21 refi boom inside the training sequences -> the model learns the boom and
"forecasting" it is not out-of-sample. Censoring by reporting month guarantees the
model sees ZERO 2020+ data, so forecasting 2020-21 is a genuine OOS test
(Gupta option b: "trained only on pre-2020 data").

Labels are also censored: a loan that prepaid in 2021 has no prepay event within
its <=2019 sequence, so it is correctly labeled non-prepaid as of Dec 2019.

Only 2018-2019 vintages contribute (2020+ originations have no pre-2020 months).

Output saved to: data/sequences_pre2020/   (overwrites the leaky vintage-only set)
Does NOT touch data/sequences/ (21-vintage full model).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import gc
import pickle

DATA_DIR   = '/scratch/at7095/mortgage_prepayment/data/raw'
PMMS_PATH  = '/scratch/at7095/mortgage_prepayment/data/pmms_monthly.csv'
ZHVI_PATH  = '/scratch/at7095/mortgage_prepayment/data/zhvi_zip3.csv'
SAVE_DIR   = '/scratch/at7095/mortgage_prepayment/data/sequences_pre2020'   # <-- separate dir

# ── KEY CHANGE: 2018-2019 only ────────────────────────────────────────────────
VINTAGES = [
    '2018Q1', '2018Q2', '2018Q3', '2018Q4',
    '2019Q1', '2019Q2', '2019Q3', '2019Q4',
]

MAX_SEQ_LEN  = 33
N_FEATURES   = 9
FEATURE_COLS = ['refi_incentive', 'borrower_credit_score', 'original_ltv',
                'current_ltv', 'original_upb', 'loan_age_months',
                'dti', 'loan_purpose_enc', 'property_type_enc']

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


def load_pmms():
    pmms = pd.read_csv(PMMS_PATH)
    pmms['reporting_period'] = pmms['reporting_period'].astype(int)
    return dict(zip(pmms['reporting_period'], pmms['rate_30yr']))


def load_zhvi():
    zhvi = pd.read_csv(ZHVI_PATH)
    zhvi['zip3'] = zhvi['zip3'].astype(int)
    zhvi['reporting_period'] = zhvi['reporting_period'].astype(int)
    return zhvi


def load_vintage(vintage, pmms_rates, zhvi_df, keep_ids=None, sample_frac=0.5):
    path = os.path.join(DATA_DIR, f'{vintage}.csv')
    print(f'  Loading {vintage}...', flush=True)

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
        all_cols.index('dti') + 1:                       'dti',
        all_cols.index('loan_purpose') + 1:              'loan_purpose',
        all_cols.index('property_type') + 1:             'property_type',
    }
    sorted_cols     = dict(sorted(col_map.items()))
    usecols_idx_raw = list(sorted_cols.keys())
    col_names_      = list(sorted_cols.values())

    chunks = []
    for chunk in pd.read_csv(
        path, sep='|', header=None,
        usecols=usecols_idx_raw, low_memory=False, chunksize=500_000
    ):
        chunk.columns = col_names_
        if keep_ids is not None:
            chunk = chunk[chunk['loan_id'].isin(keep_ids)]
        chunks.append(chunk)
        del chunk
        gc.collect()

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df = df.sort_values(['loan_id', 'monthly_reporting_period']).reset_index(drop=True)

    if keep_ids is None:
        all_loan_ids = df['loan_id'].unique()
        n_sample     = int(len(all_loan_ids) * sample_frac)
        sampled_ids  = np.random.default_rng(42).choice(all_loan_ids, size=n_sample, replace=False)
        df           = df[df['loan_id'].isin(set(sampled_ids))].copy()
        gc.collect()

    df['zip3']                     = pd.to_numeric(df['zip3'], errors='coerce')
    df['origination_date']         = pd.to_numeric(df['origination_date'], errors='coerce')
    df['monthly_reporting_period'] = pd.to_numeric(df['monthly_reporting_period'], errors='coerce')
    df['zero_balance_code_actual'] = pd.to_numeric(df['zero_balance_code_actual'], errors='coerce')

    # ── CALENDAR CENSORING: keep only observations through Dec 2019 ────────────
    # monthly_reporting_period is M(M)YYYY (e.g. 12018=Jan2018, 122019=Dec2019,
    # 12020=Jan2020). The YEAR is always the last 4 digits regardless of whether
    # the month is 1 or 2 digits. Keep rows with year <= 2019.
    rp_str = df['monthly_reporting_period'].astype('Int64').astype(str)
    rp_year = pd.to_numeric(rp_str.str[-4:], errors='coerce')
    df = df[rp_year <= 2019].copy()
    if df.empty:
        print(f'    -> {vintage}: no observations <= 2019, skipping', flush=True)
        return df

    df['market_rate']    = df['monthly_reporting_period'].map(pmms_rates)
    df['refi_incentive'] = df['original_interest_rate'] - df['market_rate']

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
    df['loan_age_months']     = df['loan_age'].astype(float)

    df['dti'] = pd.to_numeric(df['dti'], errors='coerce')
    df['loan_purpose_enc']  = df['loan_purpose'].map({'N': 0, 'Y': 1}).fillna(0).astype(float)
    df['property_type_enc'] = df['property_type'].map({'P': 0, 'R': 1, 'C': 2}).fillna(0).astype(float)

    prepaid_set   = set(df.loc[df['zero_balance_code_actual'] == 1.0, 'loan_id'].unique())
    df['prepaid'] = df['loan_id'].isin(prepaid_set).astype(int)

    keep_cols = ['loan_id', 'monthly_reporting_period', 'prepaid',
                 'zero_balance_code_actual'] + FEATURE_COLS
    df = df[keep_cols].dropna(subset=FEATURE_COLS)

    n_loans   = df['loan_id'].nunique()
    prepay_rt = df.groupby('loan_id')['prepaid'].first().mean() * 100
    print(f'    -> {n_loans:,} loans | prepay {prepay_rt:.2f}% | rows: {len(df):,}', flush=True)
    return df


def build_sequences(df, scaler):
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    loan_ids_unique = df['loan_id'].unique()
    loan_id_to_idx  = {lid: i for i, lid in enumerate(loan_ids_unique)}
    df['loan_idx']  = df['loan_id'].map(loan_id_to_idx)
    df['timestep']  = df.groupby('loan_id').cumcount()
    df              = df[df['timestep'] < MAX_SEQ_LEN].copy()

    n_loans   = len(loan_ids_unique)
    sequences = np.zeros((n_loans, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
    masks     = np.zeros((n_loans, MAX_SEQ_LEN), dtype=bool)
    labels    = np.zeros(n_loans, dtype=np.float32)
    prepay_t  = np.full(n_loans, -1, dtype=np.int32)

    loan_idx  = df['loan_idx'].values
    timesteps = df['timestep'].values
    sequences[loan_idx, timesteps, :] = df[FEATURE_COLS].values.astype(np.float32)
    masks[loan_idx, timesteps]        = True

    label_df = df.groupby('loan_idx')['prepaid'].first()
    labels[label_df.index.values] = label_df.values.astype(np.float32)

    prepaid_rows = df[df['zero_balance_code_actual'] == 1.0]
    if not prepaid_rows.empty:
        first_prepay = prepaid_rows.groupby('loan_idx')['timestep'].min()
        prepay_t[first_prepay.index.values] = first_prepay.values.astype(np.int32)

    return sequences, masks, labels, prepay_t, loan_ids_unique


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'Saving to: {SAVE_DIR}', flush=True)
    print(f'Vintages: {VINTAGES}', flush=True)

    print('Loading PMMS...', flush=True)
    pmms_rates = load_pmms()
    print('Loading ZHVI...', flush=True)
    zhvi_df = load_zhvi()

    # ── Pass 1: determine train/test split ────────────────────────────────────
    print('\nPass 1: determining train/test split...', flush=True)
    loan_info_chunks = []
    for v in VINTAGES:
        df = load_vintage(v, pmms_rates, zhvi_df, keep_ids=None, sample_frac=0.5)
        info = df.groupby('loan_id')['prepaid'].first().reset_index()
        loan_info_chunks.append(info)
        del df
        gc.collect()

    loan_info = pd.concat(loan_info_chunks, ignore_index=True)
    del loan_info_chunks
    gc.collect()

    loan_info      = loan_info.groupby('loan_id')['prepaid'].max().reset_index()
    loan_ids       = loan_info['loan_id'].values
    aligned_labels = loan_info['prepaid'].values

    total_loans    = len(loan_ids)
    overall_prepay = aligned_labels.mean()
    print(f'\nTotal loans: {total_loans:,} | Prepay rate: {overall_prepay*100:.2f}%', flush=True)

    train_ids, test_ids = train_test_split(
        loan_ids, test_size=0.2, random_state=42, stratify=aligned_labels
    )
    train_id_set = set(train_ids.tolist())
    test_id_set  = set(test_ids.tolist())
    print(f'Train: {len(train_ids):,} | Test: {len(test_ids):,}', flush=True)

    del loan_info
    gc.collect()

    # ── Pass 2: fit scaler on train data ─────────────────────────────────────
    print('\nPass 2: fitting scaler on train data...', flush=True)
    scaler = StandardScaler()
    for v in VINTAGES:
        df = load_vintage(v, pmms_rates, zhvi_df, keep_ids=train_id_set)
        scaler.partial_fit(df[FEATURE_COLS].dropna())
        del df
        gc.collect()

    with open(os.path.join(SAVE_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print('Scaler saved.', flush=True)

    # ── Pass 3: build train sequences ─────────────────────────────────────────
    print('\nPass 3: building train sequences...', flush=True)
    train_seq_list, train_mask_list = [], []
    train_labels_list, train_prepay_list, train_ids_list = [], [], []

    for v in VINTAGES:
        df = load_vintage(v, pmms_rates, zhvi_df, keep_ids=train_id_set)
        if df.empty:
            continue
        seq, mask, lbl, pt, lids = build_sequences(df, scaler)
        train_seq_list.append(seq);    train_mask_list.append(mask)
        train_labels_list.append(lbl); train_prepay_list.append(pt)
        train_ids_list.append(lids)
        del df, seq, mask, lbl, pt, lids
        gc.collect()

    train_seq      = np.concatenate(train_seq_list,    axis=0)
    train_mask     = np.concatenate(train_mask_list,   axis=0)
    train_labels   = np.concatenate(train_labels_list, axis=0)
    train_prepay_t = np.concatenate(train_prepay_list, axis=0)
    train_loan_ids = np.concatenate(train_ids_list,    axis=0)
    del train_seq_list, train_mask_list, train_labels_list, train_prepay_list, train_ids_list
    gc.collect()

    np.save(os.path.join(SAVE_DIR, 'train_seq.npy'),             train_seq)
    np.save(os.path.join(SAVE_DIR, 'train_mask.npy'),            train_mask)
    np.save(os.path.join(SAVE_DIR, 'train_labels.npy'),          train_labels)
    np.save(os.path.join(SAVE_DIR, 'train_loan_ids.npy'),        train_loan_ids)
    np.save(os.path.join(SAVE_DIR, 'train_prepay_timestep.npy'), train_prepay_t)
    print(f'  Train shape: {train_seq.shape} — saved.', flush=True)

    del train_seq, train_mask, train_labels, train_prepay_t, train_loan_ids
    gc.collect()

    # ── Pass 4: build test sequences ──────────────────────────────────────────
    print('\nPass 4: building test sequences...', flush=True)
    test_seq_list, test_mask_list = [], []
    test_labels_list, test_prepay_list, test_ids_list = [], [], []

    for v in VINTAGES:
        df = load_vintage(v, pmms_rates, zhvi_df, keep_ids=test_id_set)
        if df.empty:
            continue
        seq, mask, lbl, pt, lids = build_sequences(df, scaler)
        test_seq_list.append(seq);    test_mask_list.append(mask)
        test_labels_list.append(lbl); test_prepay_list.append(pt)
        test_ids_list.append(lids)
        del df, seq, mask, lbl, pt, lids
        gc.collect()

    test_seq      = np.concatenate(test_seq_list,    axis=0)
    test_mask     = np.concatenate(test_mask_list,   axis=0)
    test_labels   = np.concatenate(test_labels_list, axis=0)
    test_prepay_t = np.concatenate(test_prepay_list, axis=0)
    test_loan_ids = np.concatenate(test_ids_list,    axis=0)
    del test_seq_list, test_mask_list, test_labels_list, test_prepay_list, test_ids_list
    gc.collect()

    np.save(os.path.join(SAVE_DIR, 'test_seq.npy'),             test_seq)
    np.save(os.path.join(SAVE_DIR, 'test_mask.npy'),            test_mask)
    np.save(os.path.join(SAVE_DIR, 'test_labels.npy'),          test_labels)
    np.save(os.path.join(SAVE_DIR, 'test_loan_ids.npy'),        test_loan_ids)
    np.save(os.path.join(SAVE_DIR, 'test_prepay_timestep.npy'), test_prepay_t)
    print(f'  Test shape: {test_seq.shape} — saved.', flush=True)

    print('\nDone. Pre-2020 sequences saved to', SAVE_DIR, flush=True)


if __name__ == '__main__':
    main()
