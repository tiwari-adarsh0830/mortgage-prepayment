"""
prepare_sequences_rolling.py — Calendar-time truncated sequence builder.

Builds train/test sequences for all loans with activity through Dec CUTOFF_YEAR.
Key differences from prepare_sequences.py:

  1. MMYYYY sort fix: Fannie's MMYYYY integer is NOT monotone across year
     boundaries (122018 = 122,018 > 12019 = 12,019, i.e. Dec-2018 sorts
     AFTER Jan-2019 in ascending integer order). We convert to YYYYMM for
     all filtering and sorting, keeping the raw MMYYYY integer only for
     PMMS/ZHVI dict lookups (which use it as a hash key, not an order key).

  2. Prepay label derived ONLY from rows within the cutoff window.
     A loan that prepays post-cutoff is labeled prepaid=0 — no lookahead.

  3. Fixed categorical encodings:
     loan_purpose:  R=0 (purchase), C=1 (refi), P=2 (cash-out)
     property_type: SF=0, PU=1, CO=2, MH=3
     The production pipeline had the wrong map (N/Y and P/R/C) → all-zero.

  4. Per-refit scaler: fit on training loans from the cutoff window only.
     Using the global production scaler would leak post-cutoff statistics
     (e.g., the 2020-21 refi-boom distribution into a 2016-cutoff model).

  5. Saves to data/sequences_rolling/cutoff_{CUTOFF_YEAR}/.

Usage:
    python prepare_sequences_rolling.py --cutoff_year 2018
    python prepare_sequences_rolling.py --cutoff_year 2018 --sample_frac 0.5
"""

import argparse
import os
import gc
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '/scratch/at7095/mortgage_prepayment'
DATA_DIR  = os.path.join(BASE, 'data/raw')
PMMS_PATH = os.path.join(BASE, 'data/pmms_monthly.csv')
ZHVI_PATH = os.path.join(BASE, 'data/zhvi_zip3.csv')

# All potentially available vintages — loader skips missing files silently.
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

MAX_SEQ_LEN  = 33
N_FEATURES   = 9
FEATURE_COLS = [
    'refi_incentive',       # [0] original_rate - PMMS (time-varying)
    'borrower_credit_score',# [1]
    'original_ltv',         # [2]
    'current_ltv',          # [3] ZHVI-adjusted, time-varying — key for equity interaction
    'original_upb',         # [4]
    'loan_age_months',      # [5]
    'dti',                  # [6]
    'loan_purpose_enc',     # [7] FIXED: R=0, C=1, P=2
    'property_type_enc',    # [8] FIXED: SF=0, PU=1, CO=2, MH=3
]

# ── Fannie Mae column schema (same as prepare_sequences.py) ───────────────────
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

# Build col_map once — sorted by file position index for correct CSV usecols.
_COL_MAP = dict(sorted({
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
_USECOLS = list(_COL_MAP.keys())
_COLNAMES = list(_COL_MAP.values())


# ── Date helpers ──────────────────────────────────────────────────────────────

def mmyyyy_to_yyyymm(v: int) -> int:
    """Convert Fannie MMYYYY integer to YYYYMM integer for monotone comparison.

    Fannie stores dates as MMYYYY without zero-padding the month, so:
        Jan 2018 = 12018   (5 digits)
        Oct 2018 = 102018  (6 digits)
        Dec 2018 = 122018  (6 digits)

    As raw integers, 122018 > 12019, making Dec-2018 sort AFTER Jan-2019.
    YYYYMM eliminates this: Dec-2018 = 201812 < Jan-2019 = 201901. ✓

    Examples:
        mmyyyy_to_yyyymm(12018)  → 201801
        mmyyyy_to_yyyymm(102018) → 201810
        mmyyyy_to_yyyymm(122018) → 201812
        mmyyyy_to_yyyymm(12019)  → 201901
    """
    s = str(int(v))
    if len(s) == 5:      # single-digit month: M|YYYY
        mm, yyyy = int(s[0]), int(s[1:])
    elif len(s) == 6:    # two-digit month: MM|YYYY
        mm, yyyy = int(s[:2]), int(s[2:])
    else:
        raise ValueError(f'Unexpected MMYYYY length for value {v!r}: {s!r}')
    return yyyy * 100 + mm


def dec_yyyymm(year: int) -> int:
    """YYYYMM for December of given year."""
    return year * 100 + 12


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_pmms() -> dict:
    pmms = pd.read_csv(PMMS_PATH)
    pmms['reporting_period'] = pmms['reporting_period'].astype(int)
    return dict(zip(pmms['reporting_period'], pmms['rate_30yr']))


def load_zhvi() -> pd.DataFrame:
    zhvi = pd.read_csv(ZHVI_PATH)
    zhvi['zip3']             = zhvi['zip3'].astype(int)
    zhvi['reporting_period'] = zhvi['reporting_period'].astype(int)
    return zhvi


def load_vintage_filtered(
    vintage: str,
    pmms_rates: dict,
    zhvi_df: pd.DataFrame,
    cutoff_yyyymm: int,
    keep_ids=None,
    sample_frac: float = 1.0,
) -> pd.DataFrame | None:
    """Load one vintage file, apply calendar cutoff, compute all features.

    CRITICAL — prepay label correctness:
        prepaid=1 only if zero_balance_code_actual==1 appears in a row that
        survives the cutoff filter. A loan that prepays AFTER the cutoff is
        labeled prepaid=0 in the training data — no lookahead leakage.
    """
    path = os.path.join(DATA_DIR, f'{vintage}.csv')
    if not os.path.exists(path):
        return None

    print(f'  Loading {vintage}...', flush=True)

    chunks = []
    for chunk in pd.read_csv(
        path, sep='|', header=None,
        usecols=_USECOLS, low_memory=False, chunksize=500_000,
    ):
        chunk.columns = _COLNAMES
        if keep_ids is not None:
            chunk = chunk[chunk['loan_id'].isin(keep_ids)]
        chunks.append(chunk)
        del chunk
        gc.collect()

    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # ── Calendar cutoff filter — must use YYYYMM, not raw MMYYYY integer ──────
    df['monthly_reporting_period'] = pd.to_numeric(
        df['monthly_reporting_period'], errors='coerce'
    )
    df = df[df['monthly_reporting_period'].notna()].copy()
    df['yyyymm'] = df['monthly_reporting_period'].astype(int).apply(mmyyyy_to_yyyymm)
    df = df[df['yyyymm'] <= cutoff_yyyymm].copy()
    if df.empty:
        return None

    # ── Sort chronologically using YYYYMM (not raw MMYYYY) ───────────────────
    # This fixes the sort bug in prepare_sequences.py where cross-year ordering
    # was wrong (Dec-2018 sorted before Jan-2019 as integers 122018 > 12019).
    df = df.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)

    # Optional loan subsampling (pass 1 discovery only)
    if keep_ids is None and sample_frac < 1.0:
        uids    = df['loan_id'].unique()
        n       = int(len(uids) * sample_frac)
        sampled = np.random.default_rng(42).choice(uids, size=n, replace=False)
        df      = df[df['loan_id'].isin(set(sampled))].copy()
        gc.collect()

    # ── Type casts ────────────────────────────────────────────────────────────
    df['zip3']                     = pd.to_numeric(df['zip3'],                     errors='coerce')
    df['origination_date']         = pd.to_numeric(df['origination_date'],         errors='coerce')
    df['zero_balance_code_actual'] = pd.to_numeric(df['zero_balance_code_actual'], errors='coerce')

    # ── PMMS refi incentive ───────────────────────────────────────────────────
    # market_rate uses the raw MMYYYY integer as dict key (same format as PMMS CSV)
    df['market_rate']    = df['monthly_reporting_period'].map(pmms_rates)
    df['refi_incentive'] = df['original_interest_rate'] - df['market_rate']

    # ── ZHVI current LTV (time-varying) ───────────────────────────────────────
    # Use original_upb (not current_actual_upb) — the latter is 0 for prepaid loans
    # and would leak the prepayment outcome into the LTV feature.
    df = df.merge(
        zhvi_df.rename(columns={
            'reporting_period': 'origination_date',
            'zhvi': 'zhvi_orig',
        }),
        on=['zip3', 'origination_date'], how='left',
    )
    df = df.merge(
        zhvi_df.rename(columns={
            'reporting_period': 'monthly_reporting_period',
            'zhvi': 'zhvi_now',
        }),
        on=['zip3', 'monthly_reporting_period'], how='left',
    )
    df['original_home_value'] = df['original_upb'] / (
        (df['original_ltv'] / 100).replace(0, np.nan)
    )
    df['price_appreciation'] = df['zhvi_now'] / df['zhvi_orig'].replace(0, np.nan)
    df['current_ltv'] = (
        df['original_upb'] /
        (df['original_home_value'] * df['price_appreciation']).replace(0, np.nan)
    ) * 100

    df['loan_age_months'] = df['loan_age'].astype(float)
    df['dti']             = pd.to_numeric(df['dti'], errors='coerce')

    # ── FIXED categorical encodings ───────────────────────────────────────────
    # Production pipeline used wrong maps (N/Y and P/R/C → all zeros).
    # Correct Fannie Mae codes:
    #   loan_purpose:  R=Purchase, C=Refinance, P=Cash-out Refinance
    #   property_type: SF=Single-family, PU=Planned unit dev, CO=Condo, MH=Manufactured
    df['loan_purpose_enc'] = df['loan_purpose'].map(
        {'R': 0, 'C': 1, 'P': 2}
    ).fillna(0).astype(float)

    df['property_type_enc'] = df['property_type'].map(
        {'SF': 0, 'PU': 1, 'CO': 2, 'MH': 3}
    ).fillna(0).astype(float)

    # ── Prepay label — CRITICAL: only from rows within cutoff window ──────────
    # Any row with zbc==1 at monthly_reporting_period <= cutoff is a prepay event.
    # Loans prepaying after the cutoff are labeled 0 (genuinely unknown at t=cutoff).
    prepaid_set   = set(df.loc[df['zero_balance_code_actual'] == 1.0, 'loan_id'].unique())
    df['prepaid'] = df['loan_id'].isin(prepaid_set).astype(int)

    keep = ['loan_id', 'yyyymm', 'monthly_reporting_period', 'prepaid',
            'zero_balance_code_actual'] + FEATURE_COLS
    df = df[keep].dropna(subset=FEATURE_COLS)

    n_loans   = df['loan_id'].nunique()
    prepay_rt = df.groupby('loan_id')['prepaid'].first().mean() * 100
    print(f'    -> {n_loans:,} loans | prepay {prepay_rt:.2f}% | rows: {len(df):,}', flush=True)
    return df


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, scaler: StandardScaler):
    """Build padded (N, MAX_SEQ_LEN, N_FEATURES) arrays from a loan panel.

    Takes the FIRST MAX_SEQ_LEN months per loan chronologically.
    df must be pre-sorted by (loan_id, yyyymm) before calling.
    """
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    loan_ids_unique = df['loan_id'].unique()
    id_to_idx       = {lid: i for i, lid in enumerate(loan_ids_unique)}
    df['loan_idx']  = df['loan_id'].map(id_to_idx)
    df['timestep']  = df.groupby('loan_id').cumcount()
    df              = df[df['timestep'] < MAX_SEQ_LEN].copy()

    n = len(loan_ids_unique)
    sequences = np.zeros((n, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
    masks     = np.zeros((n, MAX_SEQ_LEN), dtype=bool)
    labels    = np.zeros(n, dtype=np.float32)
    prepay_t  = np.full(n, -1, dtype=np.int32)

    li = df['loan_idx'].values
    ts = df['timestep'].values
    sequences[li, ts, :] = df[FEATURE_COLS].values.astype(np.float32)
    masks[li, ts]        = True

    lbl_df = df.groupby('loan_idx')['prepaid'].first()
    labels[lbl_df.index.values] = lbl_df.values.astype(np.float32)

    prepaid_rows = df[df['zero_balance_code_actual'] == 1.0]
    if not prepaid_rows.empty:
        fp = prepaid_rows.groupby('loan_idx')['timestep'].min()
        prepay_t[fp.index.values] = fp.values.astype(np.int32)

    return sequences, masks, labels, prepay_t, loan_ids_unique


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year', type=int, required=True,
                        help='Train through December of this year (e.g. 2018)')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                        help='Loan subsampling fraction for pass 1 (default=1.0)')
    args = parser.parse_args()

    cutoff_ym = dec_yyyymm(args.cutoff_year)     # e.g. 201812
    SAVE_DIR  = os.path.join(BASE, f'data/sequences_rolling/cutoff_{args.cutoff_year}')
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f'Rolling builder | cutoff = Dec {args.cutoff_year} (YYYYMM={cutoff_ym})', flush=True)
    print(f'Output dir: {SAVE_DIR}', flush=True)

    pmms_rates = load_pmms()
    zhvi_df    = load_zhvi()

    # ── Pass 1: loan ID discovery → train/test split ──────────────────────────
    # RESUME GUARD: if the splits already exist on disk (a prior run completed
    # Pass 1 before timing out in Pass 3), load them instead of re-scanning all
    # vintages. This makes the job restartable and avoids redoing the ~1.5hr scan.
    train_split_path = os.path.join(SAVE_DIR, 'train_loan_ids_split.npy')
    test_split_path  = os.path.join(SAVE_DIR, 'test_loan_ids_split.npy')

    if os.path.exists(train_split_path) and os.path.exists(test_split_path):
        train_ids = np.load(train_split_path, allow_pickle=True)
        test_ids  = np.load(test_split_path,  allow_pickle=True)
        train_id_set = set(train_ids.tolist())
        test_id_set  = set(test_ids.tolist())
        print(f'\nPass 1: SKIPPED — loaded existing splits '
              f'(train={len(train_ids):,}, test={len(test_ids):,})', flush=True)
    else:
        print('\nPass 1: loan ID discovery...', flush=True)
        info_chunks = []
        for v in ALL_VINTAGES:
            df = load_vintage_filtered(v, pmms_rates, zhvi_df, cutoff_ym,
                                       keep_ids=None, sample_frac=args.sample_frac)
            if df is None or df.empty:
                continue
            info_chunks.append(df.groupby('loan_id')['prepaid'].first().reset_index())
            del df; gc.collect()

        if not info_chunks:
            raise RuntimeError('No data loaded — verify vintage paths and cutoff_year.')

        loan_info = (pd.concat(info_chunks, ignore_index=True)
                       .groupby('loan_id')['prepaid'].max()
                       .reset_index())
        del info_chunks; gc.collect()

        loan_ids  = loan_info['loan_id'].values
        labels_1p = loan_info['prepaid'].values
        print(f'\nTotal loans: {len(loan_ids):,} | Prepay rate: {labels_1p.mean()*100:.2f}%',
              flush=True)

        train_ids, test_ids = train_test_split(
            loan_ids, test_size=0.2, random_state=42, stratify=labels_1p
        )
        train_id_set = set(train_ids.tolist())
        test_id_set  = set(test_ids.tolist())
        print(f'Train: {len(train_ids):,} | Test: {len(test_ids):,}', flush=True)

        np.save(train_split_path, train_ids)
        np.save(test_split_path,  test_ids)
        del loan_info; gc.collect()

    # ── Pass 2: fit scaler on a SAMPLE of train loans ────────────────────────
    # Full re-read of all vintages is extremely slow (billions of rows).
    # StandardScaler statistics are stable at 5-10% sample size for 10M+ loans.
    # We read each vintage with a hard cap of SCALER_ROWS_PER_VINTAGE rows from
    # train IDs, then stop. This cuts Pass 2 from ~2hrs to ~5min.
    # RESUME GUARD: skip if scaler.pkl already exists from a prior run.
    scaler_path = os.path.join(SAVE_DIR, 'scaler.pkl')
    _skip_pass2 = os.path.exists(scaler_path)
    SCALER_ROWS_PER_VINTAGE = 50_000

    if _skip_pass2:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print('\nPass 2: SKIPPED — loaded existing scaler.pkl', flush=True)
    else:
        print('\nPass 2: fitting scaler (sampled, fast)...', flush=True)
        scaler = StandardScaler()
        n_scaler_rows = 0
        for v in ALL_VINTAGES:
            path = os.path.join(DATA_DIR, f'{v}.csv')
            if not os.path.exists(path):
                continue
            rows = []
            for chunk in pd.read_csv(
                path, sep='|', header=None,
                usecols=_USECOLS, low_memory=False, chunksize=500_000,
            ):
                chunk.columns = _COLNAMES
                chunk = chunk[chunk['loan_id'].isin(train_id_set)]
                if chunk.empty:
                    continue
                rows.append(chunk)
                if sum(len(r) for r in rows) >= SCALER_ROWS_PER_VINTAGE:
                    break
            if not rows:
                continue
            sample = pd.concat(rows, ignore_index=True).head(SCALER_ROWS_PER_VINTAGE)
            del rows; gc.collect()

            # Minimal feature engineering for scaler fit
            sample['monthly_reporting_period'] = pd.to_numeric(
                sample['monthly_reporting_period'], errors='coerce')
            sample = sample[sample['monthly_reporting_period'].notna()].copy()
            sample['yyyymm'] = sample['monthly_reporting_period'].astype(int).apply(
                mmyyyy_to_yyyymm)
            sample = sample[sample['yyyymm'] <= cutoff_ym]
            if sample.empty:
                continue

            sample['market_rate']    = sample['monthly_reporting_period'].map(pmms_rates)
            sample['refi_incentive'] = sample['original_interest_rate'] - sample['market_rate']
            sample['zip3']           = pd.to_numeric(sample['zip3'], errors='coerce')
            sample['origination_date'] = pd.to_numeric(sample['origination_date'], errors='coerce')

            sample = sample.merge(
                zhvi_df.rename(columns={'reporting_period': 'origination_date', 'zhvi': 'zhvi_orig'}),
                on=['zip3', 'origination_date'], how='left')
            sample = sample.merge(
                zhvi_df.rename(columns={'reporting_period': 'monthly_reporting_period', 'zhvi': 'zhvi_now'}),
                on=['zip3', 'monthly_reporting_period'], how='left')
            sample['original_home_value'] = sample['original_upb'] / (
                (sample['original_ltv'] / 100).replace(0, np.nan))
            sample['price_appreciation'] = sample['zhvi_now'] / sample['zhvi_orig'].replace(0, np.nan)
            sample['current_ltv'] = (
                sample['original_upb'] /
                (sample['original_home_value'] * sample['price_appreciation']).replace(0, np.nan)
            ) * 100
            sample['loan_age_months'] = sample['loan_age'].astype(float)
            sample['dti']             = pd.to_numeric(sample['dti'], errors='coerce')
            sample['loan_purpose_enc']  = sample['loan_purpose'].map(
                {'R': 0, 'C': 1, 'P': 2}).fillna(0).astype(float)
            sample['property_type_enc'] = sample['property_type'].map(
                {'SF': 0, 'PU': 1, 'CO': 2, 'MH': 3}).fillna(0).astype(float)

            valid = sample[FEATURE_COLS].dropna()
            if len(valid) > 0:
                scaler.partial_fit(valid)
                n_scaler_rows += len(valid)
                print(f'  {v}: +{len(valid):,} rows  (total={n_scaler_rows:,})', flush=True)
            del sample, valid; gc.collect()

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'  Scaler saved. Total rows used: {n_scaler_rows:,}', flush=True)

    # ── Pass 3: build train + test sequences in a SINGLE read of each vintage ─
    # The old design read every vintage file TWICE (once for train, once for
    # test) — ~30 multi-GB files read twice = the dominant cost (~6h, timed out).
    # Here we read each vintage ONCE, split its rows into train/test by loan ID
    # in memory, and append per-vintage shards to disk. Per-vintage shard files
    # double as a resume checkpoint: if the job is killed, completed vintages are
    # skipped on rerun, so a restart only processes what's left.
    shard_dir = os.path.join(SAVE_DIR, '_shards')
    os.makedirs(shard_dir, exist_ok=True)

    # Final-output resume guard: if both seq arrays already exist, nothing to do.
    if (os.path.exists(os.path.join(SAVE_DIR, 'train_seq.npy')) and
            os.path.exists(os.path.join(SAVE_DIR, 'test_seq.npy'))):
        print('\nPass 3/4: SKIPPED — train_seq.npy and test_seq.npy already exist',
              flush=True)
        print(f'\nDone. cutoff={args.cutoff_year} | dir={SAVE_DIR}', flush=True)
        return

    print('\nPass 3: building train+test sequences (single read per vintage)...',
          flush=True)

    def _shard_done(v):
        return os.path.exists(os.path.join(shard_dir, f'{v}_train_seq.npy')) or \
               os.path.exists(os.path.join(shard_dir, f'{v}_empty.flag'))

    for v in ALL_VINTAGES:
        if _shard_done(v):
            print(f'  {v}: shard exists — skip', flush=True)
            continue

        df = load_vintage_filtered(v, pmms_rates, zhvi_df, cutoff_ym, keep_ids=None)
        if df is None or df.empty:
            # mark empty so rerun doesn't retry a vintage with no in-window rows
            open(os.path.join(shard_dir, f'{v}_empty.flag'), 'w').close()
            continue

        for split_name, id_set in (('train', train_id_set), ('test', test_id_set)):
            sub = df[df['loan_id'].isin(id_set)]
            if sub.empty:
                continue
            seq, mask, lbl, pt, lids = build_sequences(sub, scaler)
            sp = os.path.join(shard_dir, f'{v}_{split_name}')
            np.save(f'{sp}_seq.npy',  seq)
            np.save(f'{sp}_mask.npy', mask)
            np.save(f'{sp}_lbl.npy',  lbl)
            np.save(f'{sp}_pt.npy',   pt)
            np.save(f'{sp}_ids.npy',  lids)
            del sub, seq, mask, lbl, pt, lids; gc.collect()
        # mark vintage complete even if only one split had rows
        if not _shard_done(v):
            open(os.path.join(shard_dir, f'{v}_empty.flag'), 'w').close()
        del df; gc.collect()
        print(f'  {v}: shard written', flush=True)

    # ── Pass 4: concatenate per-vintage shards into final arrays ──────────────
    print('\nPass 4: concatenating shards...', flush=True)

    def _concat_split(split_name):
        comp = {'seq': [], 'mask': [], 'lbl': [], 'pt': [], 'ids': []}
        for v in ALL_VINTAGES:
            sp = os.path.join(shard_dir, f'{v}_{split_name}')
            if not os.path.exists(f'{sp}_seq.npy'):
                continue
            comp['seq'].append(np.load(f'{sp}_seq.npy'))
            comp['mask'].append(np.load(f'{sp}_mask.npy'))
            comp['lbl'].append(np.load(f'{sp}_lbl.npy'))
            comp['pt'].append(np.load(f'{sp}_pt.npy'))
            comp['ids'].append(np.load(f'{sp}_ids.npy', allow_pickle=True))
        if not comp['seq']:
            raise RuntimeError(f'No shards found for {split_name}')

        out_seq  = np.concatenate(comp['seq'],  axis=0)
        out_mask = np.concatenate(comp['mask'], axis=0)
        out_lbl  = np.concatenate(comp['lbl'],  axis=0)
        out_pt   = np.concatenate(comp['pt'],   axis=0)
        out_ids  = np.concatenate(comp['ids'],  axis=0)

        p = os.path.join(SAVE_DIR, split_name)
        # write to .tmp then rename → atomic, so a kill mid-write never leaves
        # a truncated file the resume guard would wrongly trust.
        np.save(f'{p}_seq.tmp.npy',             out_seq)
        np.save(f'{p}_mask.npy',                out_mask)
        np.save(f'{p}_labels.npy',              out_lbl)
        np.save(f'{p}_prepay_timestep.npy',     out_pt)
        np.save(f'{p}_loan_ids.npy',            out_ids)
        os.replace(f'{p}_seq.tmp.npy', f'{p}_seq.npy')  # atomic last step
        print(f'  {split_name}: shape={out_seq.shape}  prepay={out_lbl.mean()*100:.2f}%',
              flush=True)
        del out_seq, out_mask, out_lbl, out_pt, out_ids, comp; gc.collect()

    _concat_split('train')
    _concat_split('test')

    # cleanup shards once final arrays are written
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)

    print(f'\nDone. cutoff={args.cutoff_year} | dir={SAVE_DIR}', flush=True)


if __name__ == '__main__':
    main()
