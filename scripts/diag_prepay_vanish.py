"""
diag_prepay_vanish.py — Trace where zbc==1 rows disappear in the rolling prep.

Mirrors load_vintage_filtered() step by step on 2014Q1 with cutoff Dec 2016,
printing the count of zbc==1 rows (and distinct prepaid loans) after each stage.
Whichever stage drops the count to 0 is the bug.
"""
import os, gc
import numpy as np
import pandas as pd

BASE      = '/scratch/at7095/mortgage_prepayment'
DATA_DIR  = os.path.join(BASE, 'data/raw')
PMMS_PATH = os.path.join(BASE, 'data/pmms_monthly.csv')
ZHVI_PATH = os.path.join(BASE, 'data/zhvi_zip3.csv')

VINTAGE   = '2014Q1'
CUTOFF    = 201612

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
_COL_MAP = dict(sorted({
    _ALL_COLS.index('loan_id')+1:'loan_id',
    _ALL_COLS.index('monthly_reporting_period')+1:'monthly_reporting_period',
    _ALL_COLS.index('original_interest_rate')+1:'original_interest_rate',
    _ALL_COLS.index('borrower_credit_score')+1:'borrower_credit_score',
    _ALL_COLS.index('original_ltv')+1:'original_ltv',
    _ALL_COLS.index('original_upb')+1:'original_upb',
    _ALL_COLS.index('loan_age')+1:'loan_age',
    _ALL_COLS.index('origination_date')+1:'origination_date',
    _ALL_COLS.index('zip')+1:'zip3',
    _ALL_COLS.index('extra_13')+1:'zero_balance_code_actual',
    _ALL_COLS.index('dti')+1:'dti',
    _ALL_COLS.index('loan_purpose')+1:'loan_purpose',
    _ALL_COLS.index('property_type')+1:'property_type',
}.items()))
_USECOLS  = list(_COL_MAP.keys())
_COLNAMES = list(_COL_MAP.values())

def mmyyyy_to_yyyymm(v):
    s = str(int(v))
    if len(s)==5: return int(s[1:])*100+int(s[0])
    return int(s[2:])*100+int(s[:2])

def zc(df, tag):
    if 'zero_balance_code_actual' not in df.columns:
        print(f'  [{tag}] zbc column MISSING'); return
    n1 = (pd.to_numeric(df['zero_balance_code_actual'], errors='coerce')==1.0).sum()
    loans = df.loc[pd.to_numeric(df['zero_balance_code_actual'],errors='coerce')==1.0,'loan_id'].nunique()
    print(f'  [{tag}] rows={len(df):,}  zbc==1 rows={n1:,}  prepaid loans={loans:,}')

print(f'Tracing {VINTAGE}, cutoff={CUTOFF}\n')

# Stage 1: raw read with usecols
chunks=[]
for ch in pd.read_csv(os.path.join(DATA_DIR,f'{VINTAGE}.csv'), sep='|', header=None,
                      usecols=_USECOLS, low_memory=False, chunksize=500_000):
    ch.columns=_COLNAMES; chunks.append(ch)
df=pd.concat(chunks, ignore_index=True); del chunks; gc.collect()
zc(df,'1 after read (usecols+rename)')
print(f'      zbc dtype={df["zero_balance_code_actual"].dtype}, '
      f'sample non-null: {df["zero_balance_code_actual"].dropna().unique()[:6].tolist()}')

# Stage 2: monthly_reporting_period numeric + notna + yyyymm
df['monthly_reporting_period']=pd.to_numeric(df['monthly_reporting_period'],errors='coerce')
df=df[df['monthly_reporting_period'].notna()].copy()
df['yyyymm']=df['monthly_reporting_period'].astype(int).apply(mmyyyy_to_yyyymm)
zc(df,'2 after mrp-numeric+yyyymm')
print(f'      yyyymm range: {df["yyyymm"].min()}–{df["yyyymm"].max()}')

# Stage 3: cutoff filter
df=df[df['yyyymm']<=CUTOFF].copy()
zc(df,'3 after cutoff<=201612')

# Stage 4: zbc to_numeric cast
df['zero_balance_code_actual']=pd.to_numeric(df['zero_balance_code_actual'],errors='coerce')
zc(df,'4 after zbc to_numeric')

# Stage 5: prepaid_set
prepaid_set=set(df.loc[df['zero_balance_code_actual']==1.0,'loan_id'].unique())
df['prepaid']=df['loan_id'].isin(prepaid_set).astype(int)
print(f'  [5] prepaid_set size={len(prepaid_set):,}  '
      f'prepay_rt(pre-dropna)={df.groupby("loan_id")["prepaid"].first().mean()*100:.3f}%')
