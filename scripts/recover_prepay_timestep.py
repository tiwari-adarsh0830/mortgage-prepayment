"""
Recover prepayment timestep for each loan from raw vintage CSVs.
Only reads loan_id, monthly_reporting_period, zero_balance_code_actual.
"""

import numpy as np
import pandas as pd
import os
import gc

BASE     = "/scratch/at7095/mortgage_prepayment"
DATA_DIR = os.path.join(BASE, "data/raw")
SAVE_DIR = os.path.join(BASE, "data/sequences")

VINTAGES = [
    '2020Q1', '2020Q2', '2020Q3', '2020Q4',
    '2021Q1', '2021Q2', '2021Q3', '2021Q4',
    '2023Q1'
]

MAX_SEQ_LEN = 33

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
    all_cols.index('loan_id') + 1:                   'loan_id',
    all_cols.index('monthly_reporting_period') + 1:  'monthly_reporting_period',
    all_cols.index('extra_13') + 1:                  'zero_balance_code_actual',
}
sorted_cols     = dict(sorted(col_map.items()))
usecols_idx_raw = list(sorted_cols.keys())
col_names       = list(sorted_cols.values())


def load_vintage_minimal(vintage, keep_ids):
    path = os.path.join(DATA_DIR, f'{vintage}.csv')
    print(f"  Loading {vintage}...", flush=True)
    chunks = []
    for chunk in pd.read_csv(
        path, sep='|', header=None,
        usecols=usecols_idx_raw, low_memory=False, chunksize=500_000
    ):
        chunk.columns = col_names
        chunk = chunk[chunk['loan_id'].isin(keep_ids)].copy()
        if not chunk.empty:
            chunks.append(chunk)
        del chunk
        gc.collect()
    if not chunks:
        return pd.DataFrame(columns=col_names)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    df['monthly_reporting_period'] = pd.to_numeric(df['monthly_reporting_period'], errors='coerce')
    df['zero_balance_code_actual'] = pd.to_numeric(df['zero_balance_code_actual'], errors='coerce')
    return df


def compute_prepay_timesteps(df):
    """
    Given df with loan_id, monthly_reporting_period, zero_balance_code_actual,
    compute the 0-indexed timestep of prepayment for each loan.
    Returns: dict {loan_id -> timestep} where timestep=-1 means no prepayment.
    """
    df = df.sort_values(['loan_id', 'monthly_reporting_period'])
    # Assign timestep within each loan (0-indexed, capped at MAX_SEQ_LEN)
    df['timestep'] = df.groupby('loan_id').cumcount()
    df = df[df['timestep'] < MAX_SEQ_LEN].copy()
    # Find first prepayment row per loan
    prepaid_rows = df[df['zero_balance_code_actual'] == 1.0]
    first_prepay = prepaid_rows.groupby('loan_id')['timestep'].min()
    return first_prepay.to_dict()  # {loan_id -> timestep}


def main():
    test_loan_ids = np.load(os.path.join(SAVE_DIR, 'test_loan_ids.npy'), allow_pickle=True)
    test_labels   = np.load(os.path.join(SAVE_DIR, 'test_labels.npy'))
    train_labels  = np.load(os.path.join(SAVE_DIR, 'train_labels.npy'))

    print(f"train_labels: {train_labels.shape[0]:,} loans")
    print(f"test_labels:  {test_labels.shape[0]:,} loans")
    print(f"test_loan_ids: {test_loan_ids.shape[0]:,}")

    # ── Step 1: Get all loan IDs from raw data ─────────────────────────────────
    print("\nCollecting all loan IDs from raw vintages...", flush=True)
    lid_col = all_cols.index('loan_id') + 1
    all_id_chunks = []
    for vintage in VINTAGES:
        path = os.path.join(DATA_DIR, f'{vintage}.csv')
        for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=[lid_col], low_memory=False, chunksize=500_000
        ):
            chunk.columns = ['loan_id']
            all_id_chunks.append(chunk['loan_id'].values)
            del chunk
            gc.collect()
    all_ids = np.unique(np.concatenate(all_id_chunks))
    del all_id_chunks
    gc.collect()
    print(f"Total unique loan IDs in raw data: {len(all_ids):,}")

    # ── Step 2: Reconstruct train_loan_ids ────────────────────────────────────
    test_id_set    = set(test_loan_ids.tolist())
    train_loan_ids = np.array([lid for lid in all_ids if lid not in test_id_set], dtype=np.int64)
    np.save(os.path.join(SAVE_DIR, 'train_loan_ids.npy'), train_loan_ids)
    print(f"Reconstructed train_loan_ids: {len(train_loan_ids):,} (saved)")
    print(f"  train_seq has {train_labels.shape[0]:,} rows — difference is due to subsampling in pipeline")

    # ── Step 3: Load vintages and compute prepay timesteps ────────────────────
    all_keep_ids = set(all_ids.tolist())  # all IDs — we need both train and test
    print(f"\nRecovering prepay timesteps for {len(all_keep_ids):,} loans...", flush=True)

    all_prepay_t = {}  # loan_id -> timestep

    for vintage in VINTAGES:
        df = load_vintage_minimal(vintage, all_keep_ids)
        if df.empty:
            continue
        batch_prepay = compute_prepay_timesteps(df)
        # Merge: keep earliest timestep if loan appears in multiple vintages
        for lid, t in batch_prepay.items():
            if lid not in all_prepay_t or t < all_prepay_t[lid]:
                all_prepay_t[lid] = t
        del df
        gc.collect()
        print(f"  Prepaid loans found so far: {len(all_prepay_t):,}", flush=True)

    # ── Step 4: Build aligned arrays ──────────────────────────────────────────
    train_prepay_t = np.array(
        [all_prepay_t.get(int(lid), -1) for lid in train_loan_ids],
        dtype=np.int32
    )
    test_prepay_t = np.array(
        [all_prepay_t.get(int(lid), -1) for lid in test_loan_ids],
        dtype=np.int32
    )

    np.save(os.path.join(SAVE_DIR, 'train_prepay_timestep.npy'), train_prepay_t)
    np.save(os.path.join(SAVE_DIR, 'test_prepay_timestep.npy'),  test_prepay_t)
    print(f"\nSaved train_prepay_timestep.npy shape={train_prepay_t.shape}")
    print(f"Saved test_prepay_timestep.npy  shape={test_prepay_t.shape}")

    # ── Step 5: Sanity checks ─────────────────────────────────────────────────
    n = min(len(test_labels), len(test_loan_ids))
    test_prepaid_mask = test_labels[:n] == 1
    test_prepay_trunc = test_prepay_t[:n]

    print(f"\n── Test sanity check (first {n:,} aligned rows) ──")
    print(f"  Prepaid loans:       {test_prepaid_mask.sum():,}")
    print(f"  With timestep >= 0:  {(test_prepay_trunc[test_prepaid_mask] >= 0).sum():,}")
    print(f"  With timestep == -1: {(test_prepay_trunc[test_prepaid_mask] == -1).sum():,}")
    print(f"  Timestep distribution (prepaid):")
    vals, counts = np.unique(test_prepay_trunc[test_prepaid_mask], return_counts=True)
    for v, c in zip(vals[:15], counts[:15]):
        print(f"    t={v:>2}: {c:,}")


if __name__ == "__main__":
    main()
