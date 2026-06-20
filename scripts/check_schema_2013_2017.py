import os
import pandas as pd
import numpy as np

DATA_DIR = '/scratch/at7095/mortgage_prepayment/data/raw'
vintages = [f'{y}Q{q}' for y in range(2013, 2018) for q in range(1, 5)]

# Columns we actually use (by position, 1-indexed)
# From prepare_sequences.py: loan_id=2, monthly_reporting_period=3,
# original_interest_rate=8, borrower_credit_score=24, original_ltv=20,
# original_upb=10, loan_age=16, origination_date=14, zip=33,
# extra_13=col 96 (13th extra), dti=23, loan_purpose=27, property_type=28
TARGET_COLS = {
    2:  'loan_id',
    3:  'monthly_reporting_period',
    8:  'original_interest_rate',
    10: 'original_upb',
    14: 'origination_date',
    16: 'loan_age',
    20: 'original_ltv',
    23: 'dti',
    24: 'borrower_credit_score',
    27: 'loan_purpose',
    28: 'property_type',
    33: 'zip',
}

print(f"{'Vintage':<10} {'NCol':<6} {'Rows':<12} {'LoanID_null%':<14} {'Rate_null%':<12} {'FICO_null%':<12} {'DTI_null%':<11} {'LTV_null%':<11} {'Purpose_vals':<20} {'PropType_vals'}")
print("-" * 140)

for v in vintages:
    path = os.path.join(DATA_DIR, f'{v}.csv')
    if not os.path.exists(path):
        print(f"{v:<10} MISSING")
        continue

    # Check column count from first row
    with open(path) as f:
        first_line = f.readline()
    ncols = len(first_line.split('|'))

    # Load only target columns, first 50k rows for speed
    usecols = list(TARGET_COLS.keys())
    try:
        df = pd.read_csv(path, sep='|', header=None, usecols=usecols,
                         nrows=50000, low_memory=False)
        df.columns = [TARGET_COLS[c] for c in usecols]
    except Exception as e:
        print(f"{v:<10} ERROR: {e}")
        continue

    n_rows = len(df)
    loan_null   = df['loan_id'].isna().mean() * 100
    rate_null   = pd.to_numeric(df['original_interest_rate'], errors='coerce').isna().mean() * 100
    fico_null   = pd.to_numeric(df['borrower_credit_score'], errors='coerce').isna().mean() * 100
    dti_null    = pd.to_numeric(df['dti'], errors='coerce').isna().mean() * 100
    ltv_null    = pd.to_numeric(df['original_ltv'], errors='coerce').isna().mean() * 100
    purpose_vals = ','.join(sorted(df['loan_purpose'].dropna().astype(str).unique()))
    proptype_vals = ','.join(sorted(df['property_type'].dropna().astype(str).unique()))

    print(f"{v:<10} {ncols:<6} {n_rows:<12} {loan_null:<14.1f} {rate_null:<12.1f} {fico_null:<12.1f} {dti_null:<11.1f} {ltv_null:<11.1f} {purpose_vals:<20} {proptype_vals}")

print("\nDone.")
