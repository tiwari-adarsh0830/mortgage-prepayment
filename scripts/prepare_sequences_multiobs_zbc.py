"""
prepare_sequences_multiobs_zbc.py — Multiple-observations-per-loan sequence
builder. Copied from prepare_sequences_trailing_zbc.py; DO NOT re-merge the
trailing builder's build_sequences() here, the windowing semantics differ.

MOTIVATION: the trailing builder emits ONE observation per loan (last
MAX_SEQ_LEN months to the cutoff, loan-level label "did this loan ever
prepay in-window"). That cannot separate "high refi incentive" from "high
incentive and already declined it repeatedly" (burnout/survivor selection),
because every loan is observed exactly once, at its final age. This builder
samples each loan at SEVERAL (loan_id, ref_month) ages, so the same loan can
appear in the data both before and after burnout sets in.

An observation is (loan_id, ref_month): features are the last MAX_SEQ_LEN
rows with yyyymm <= ref_month (RIGHT-aligned — the last timestep of the
window is always ref_month itself, left-padded with zeros/mask=False for
loans younger than MAX_SEQ_LEN). Label is 1 iff a zero_balance_code_actual
== 1 row falls in (ref_month, ref_month + H], H = --label_horizon.

ELIGIBILITY for within-loan row index t (0-based, chronological):
    t >= min_hist - 1
    t <= L - 1 - H
    t <  term_t
  term_t = index of the FIRST zbc==01 row for prepaid loans, taken from the
  event index explicitly, else L-1 (last available row) for censored loans.
  It is NOT assumed to be the last row — the trailing builder's "Fannie
  stops reporting after zero-balance" comment was verified on a 2015Q1
  slice only. The t < term_t bound is LOAD-BEARING: without it, a reference
  month can land ON the payoff row, which becomes the last timestep of the
  feature window while the forward label reads 0 (nothing left to look
  ahead to) — the event enters X and vanishes from y at once.

SAMPLING per loan, fixed k = --k_draws (length-bias fix: a 120-month loan
and an 8-month loan contribute the same observation count, not a rate):
  - One MANDATORY draw at t = term_t - H (never at term_t itself — that is
    the degenerate/leaking case above). Skipped if term_t - H < min_hist-1.
  - The remaining k-1 slots are filled by BOTTOM-k selection on a
    deterministic hash of (loan_id, ref_month) — not rng.choice. This is
    reproducible across processes/machines and monotone-growable: raising k
    only ADDS observations, it never re-rolls the existing ones. Python's
    builtin hash() is salted for str and cannot be used for this. We take
    one hashlib.blake2b digest per UNIQUE loan_id (O(n_loans)), then mix it
    with the month via cheap integer arithmetic per candidate row
    (O(n_rows) but no further hashlib calls) — see _loan_base_hash/_mix_hash.
  - --draw_scheme incentive bins eligible non-mandatory months by UNSCALED
    refi_incentive at the reference month (--incentive_edges) and splits the
    k-1 budget round-robin across a loan's non-empty bins, with per-stratum
    inclusion probabilities.

KNOWN LIMITATION -- row_idx is a POSITION, not a calendar index. Upstream
dropna(subset=FEATURE_COLS) in load_vintage_filtered can remove an interior
row (e.g. one month with a NaN DTI), after which row_idx i and i+1 for that
loan are no longer 1 calendar month apart even though they're still
adjacent positions. select_observations drops two kinds of candidates for
this reason and prints both counts per call (so it is never silent in the
prep job log):
  - feature window spans a calendar gap: the drawn (s, t] window would
    silently splice two non-adjacent calendar months together.
  - label window row/calendar distance mismatch: for a real prepay, the
    calendar distance from ref_month to the event month must equal
    term_t - row_idx, or H means a different number of calendar months for
    this loan than for every other one.
  Reference point (2015Q1, cutoff_year=2020, k=5, H=1, uniform): out of
  2,166,647 candidate observations from 435,443 loans, 18 (10 loans) hit
  the window-gap filter and 4 (of 435,443 terminal draws) hit the
  label-mismatch filter -- both filters are rare but real; do not remove
  them assuming the gap can't occur.

WEIGHTS ARE EMITTED, NOT APPLIED. Per observation we save incl_prob (1.0 for
the mandatory/terminal draw, k_s/n_s otherwise), is_terminal, n_eligible,
k_actual, ref_month, and UNSCALED age_at_ref / incentive_at_ref (for
verification only). IPW and the King-Zeng intercept correction for
terminal-month oversampling are TRAINING-time decisions and must NOT be
baked into these arrays.

prepay_timestep IS STILL WRITTEN for file-shape compatibility with the
trailing builder, but it is MEANINGLESS here: by construction the event (if
any) is strictly outside the feature window (t < term_t), so no timestep in
X reflects it. It is filled with -1 for every observation. For the same
reason, train_hazard_rolling.py's HazardSampler (lines 78-95) must NEVER be
pointed at this output — it truncates each sequence at a random timestep
per loan per epoch, which destroys the window/label alignment this builder
constructs, and its own 50/50 prepaid oversampling would stack on top of
the mandatory-draw oversampling already baked into which rows exist here.

Output dir: data/sequences_rolling/cutoff_{CUTOFF_YEAR}_zbc_multiobs_k{k}_h{H}
            [_{scheme}]   (scheme suffix only for --draw_scheme != uniform)
so that train_hazard_rolling.py --label_suffix can reach it.

Usage:
    python prepare_sequences_multiobs_zbc.py --cutoff_year 2020
    python prepare_sequences_multiobs_zbc.py --cutoff_year 2020 --k_draws 5 \\
        --label_horizon 1 --reuse_from data/sequences_rolling/cutoff_2020_zbc_trail
"""

import argparse
import os
import gc
import pickle
import hashlib
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '/scratch/at7095/mortgage_prepayment'
DATA_DIR  = os.path.join(BASE, 'data/raw')            # 2013Q1-2023Q1 (modern era)
PRE2013_DATA_DIR = os.path.join(BASE, 'data_pre2013_raw')   # 2000Q1-2012Q4 (historical era)
PRE2013_CELL_SAMPLE_PATH = os.path.join(BASE, 'outputs', 'pre2013_cell_sample_loans.csv')
PMMS_PATH = os.path.join(BASE, 'data/pmms_monthly.csv')
ZHVI_PATH = os.path.join(BASE, 'data/zhvi_zip3.csv')

# All potentially available MODERN-era vintages — loader skips missing files
# silently. Unchanged from before the historical-era extension.
_MODERN_VINTAGES = [
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

# HISTORICAL-era acquisition-quarter files actually on disk under
# data_pre2013_raw/ (2000Q1.csv .. 2012Q4.csv, 52 files -- confirmed against
# `ls data_pre2013_raw/*.csv`). NOTE: this is NOT the same axis as the
# cell-grid sample's `vintage_quarter` label, which is origination-date-
# derived and runs back to 1999Q1 -- those 1999-origination loans have no
# file of their own, they ride inside the 2000Q1.csv+ ACQUISITION files
# (see count_prepay_events_pre2013.py's docstring) and are picked up
# automatically once that file is opened and cell-grid-gated below.
PRE2013_VINTAGES = [f'{y}Q{q}' for y in range(2000, 2013) for q in range(1, 5)]
_PRE2013_VINTAGE_SET = frozenset(PRE2013_VINTAGES)

# Module-level default: modern-only. Anything that reads ALL_VINTAGES without
# going through main() (census_panel_baseline.py, fixed_fraction_coupon_
# preview.py, etc.) is therefore completely unaffected by the historical-era
# extension. main() rebinds this global to PRE2013_VINTAGES + _MODERN_VINTAGES
# ONLY when --include_pre2013 is passed (same rebind technique as MAX_SEQ_LEN
# below) -- every existing invocation without that flag sees the identical
# list it always did, so e.g. a cutoff_2020 build's train/test split (which
# depends on the FULL discovered loan_id population, not per-vintage logic)
# is untouched.
ALL_VINTAGES = _MODERN_VINTAGES

MAX_SEQ_LEN  = 33   # default; override per-run with --max_seq_len
_DEFAULT_SEQ_LEN = 33   # frozen reference — MAX_SEQ_LEN is rebound at runtime
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
    _ALL_COLS.index('current_actual_upb') + 1:        'current_actual_upb',
    _ALL_COLS.index('loan_age') + 1:                  'loan_age',
    _ALL_COLS.index('origination_date') + 1:          'origination_date',
    _ALL_COLS.index('zip') + 1:                       'zip3',
    # original_loan_term is index 11 in _BASE_COLS (+1 = usecols 12 = awk
    # $13), well before the field-42 drift zone, so the name lookup is safe
    # -- verified: name-based result is usecols 12 on both eras, matching
    # the awk ground truth ($13 == 360 on first rows of 2000Q1 and 2013Q1).
    _ALL_COLS.index('original_loan_term') + 1:        'original_loan_term',
    # MODIFICATION FLAG -- hardcoded, do NOT switch to a name lookup, same
    # reason as zero_balance_code_actual below: _ALL_COLS.index(
    # 'modification_flag')+1 gives usecols 71 (drifted), but the real
    # modification_flag is usecols 41 (awk field 42) -- verified against
    # both eras (e.g. 2013Q1/2018Q1 row values 'N'/blank at this position).
    41:                                                'modification_flag',
    # LABEL COLUMN -- hardcoded, do NOT switch to a name lookup.
    # _ALL_COLS has 109 names for 113 fields, so it drifts:
    # _ALL_COLS.index('zero_balance_code')+1 gives usecols 42, but the
    # real zero-balance code is usecols 43 (awk field 44), verified
    # against data: set for 246,148 of 246,862 loans in 2000Q1, once per
    # loan, 241,392 coded 01. extra_13 (usecols 106) is what the original
    # script used; at a Dec-2018 cutoff on 2013Q1 it labels 0 loans
    # prepaid against 236,823 (34.8%) for this column.
    43:                                               'zero_balance_code_actual',
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


def _vintage_quarter_start_yyyymm(vintage: str) -> int:
    """Nominal acquisition-quarter start month as YYYYMM, e.g. '2002Q3' -> 200207.
    A vintage FILE cannot contain any row with monthly_reporting_period before
    this month -- loans begin reporting at acquisition, never earlier -- so
    RELEVANT_VINTAGES in main() uses this to skip opening files that provably
    hold nothing at or before a given cutoff.

    Margin verified empirically, not assumed (awk single-pass min-scan over
    raw files, 2026-09-19): actual min(monthly_reporting_period) equals this
    exact nominal value (delta 0) for every boundary file checked --
    2000Q1, 2002Q3, 2002Q4, 2003Q1, 2003Q2, 2011Q2, 2011Q3, 2012Q4 (pre-2013
    era), 2013Q1 (pre/post-2013 boundary), 2019Q4, 2020Q1 (modern era). No
    safety margin needed on top of the nominal quarter start."""
    yyyy, q = int(vintage[:4]), int(vintage[5])
    return yyyy * 100 + (q - 1) * 3 + 1


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


def _data_dir_for_vintage(vintage: str) -> str:
    """data_pre2013_raw/ for a PRE2013_VINTAGES acquisition-quarter file,
    else data/raw/ (modern era) -- so callers no longer hardcode DATA_DIR."""
    return PRE2013_DATA_DIR if vintage in _PRE2013_VINTAGE_SET else DATA_DIR


_pre2013_cell_sample_ids_cache: dict[str, frozenset] = {}


def _load_pre2013_cell_sample_ids(path: str = PRE2013_CELL_SAMPLE_PATH) -> frozenset:
    """loan_ids selected by the cell-grid sampler
    (scripts/build_cell_grid_sample_pre2013.py) at `path`. Cached per-path
    module-wide after the first call -- this is a multi-hundred-thousand-row
    CSV and load_vintage_filtered calls this once per PRE2013_VINTAGES file
    (52x for a full historical build)."""
    if path not in _pre2013_cell_sample_ids_cache:
        ids = pd.read_csv(path, usecols=['loan_id'])['loan_id']
        _pre2013_cell_sample_ids_cache[path] = frozenset(ids.tolist())
        print(f'  Loaded pre-2013 cell-grid sample from {path}: '
              f'{len(_pre2013_cell_sample_ids_cache[path]):,} loan_ids', flush=True)
    return _pre2013_cell_sample_ids_cache[path]


def _encode_categorical(series: pd.Series, mapping: dict, label: str) -> pd.Series:
    """Map a raw code column through `mapping`. property_type=='CP' (Co-op)
    and loan_purpose=='U' (Unknown) are both real, documented Fannie Mae
    codes present in BOTH eras, more densely in the earliest pre-2013
    vintages, which is what originally motivated adding them to the maps
    below (property_type CP=4, loan_purpose U=3) instead of falling back to
    0. Both are CONFIRMED in the modern era, not assumed, from two
    independent pieces of evidence:
      (1) job 17974041 (2026-09-18, cutoff_2020 rebuild, ran BEFORE this
          mapping fix existed -- CP/U still fell back to 0 that run, and the
          then-new unmapped-code warning caught it): raw loan-month ROWS with
          an unmapped code, one warning line per vintage per code. CP fired
          in EVERY one of the 32 modern (2013Q1-2020Q4) vintages, from 5,300
          rows (2020Q4) up to 318,574 rows (2013Q2). U fired in three of
          them: 96 rows (2013Q1), 82 rows (2014Q1), 24 rows (2014Q4) -- sparse
          and not present in every vintage, but real.
      (2) job 18022825 (2026-09-19, cutoff_2020 regression check, ran AFTER
          this fix): of the SAMPLED training population (fixed_fraction
          multiobs observations, a small draw off the full panel above), CP
          shows up as 55,028 train + 14,652 test observations (0.49-0.52%).
          U shows up as zero sampled observations in that same population --
          consistent with (1)'s much sparser raw U count getting missed by
          the sampler's draw, not with U being absent from the modern era.
    Anything STILL absent from `mapping`
    (missing values, or a genuinely new/unexpected code) still falls back to
    0, UNCHANGED encoding for known codes -- but is now visible: unmapped
    codes are counted and their actual values printed instead of silently
    landing on 0 indistinguishably from a genuine Purchase/Single-family
    row."""
    encoded = series.map(mapping)
    unknown = encoded.isna()
    n_unknown = int(unknown.sum())
    if n_unknown:
        seen = series[unknown].value_counts(dropna=False).to_dict()
        print(f'  WARNING: {label} - {n_unknown:,} unmapped code(s) (mapping={mapping}), '
              f'values seen: {seen}', flush=True)
    return encoded.fillna(0).astype(float)


def load_vintage_filtered(
    vintage: str,
    pmms_rates: dict,
    zhvi_df: pd.DataFrame,
    cutoff_yyyymm: int,
    keep_ids=None,
    sample_frac: float = 1.0,
    loan_purpose_map: dict | None = None,
    property_type_map: dict | None = None,
    extra_keep_cols: list[str] | None = None,
    term_filter: int | None = 360,
    cell_sample_path: str = PRE2013_CELL_SAMPLE_PATH,
) -> pd.DataFrame | None:
    """Load one vintage file, apply calendar cutoff, compute all features.

    CRITICAL — prepay label correctness:
        prepaid=1 only if zero_balance_code_actual==1 appears in a row that
        survives the cutoff filter. A loan that prepays AFTER the cutoff is
        labeled prepaid=0 in the training data — no lookahead leakage.

    loan_purpose_map / property_type_map: override the category-code maps
    passed to _encode_categorical(). Default (None) is the current fixed
    maps (CP/U mapped, see _encode_categorical docstring) -- pass the
    pre-fix maps ({'R':0,'C':1,'P':2} / {'SF':0,'PU':1,'CO':2,'MH':3}, no
    CP/U key) to reproduce features for a checkpoint trained before the
    CP/U fix landed (commit 2a5b283), so its scoring population matches
    what it actually trained on.

    extra_keep_cols: additional raw columns (already read via _USECOLS,
    e.g. 'current_actual_upb') to retain in the returned frame alongside
    the FEATURE_COLS-driven default. See `keep` below.

    term_filter: drop a loan ENTIRELY (all its rows) if ANY observed
    original_loan_term across its own rows != term_filter, including
    blank/unparseable (NaN) term values (default 360, i.e. 30-year loans
    only -- advisor's Sep 29 decision, applies throughout label/features/
    sampler/census/both realized series). This is a LOAN-level filter, not
    a row-level one: a loan is never partially kept. (Found necessary
    2026-10-01 -- 2 loans in 2002Q1 and 1 in 2018Q1 carry more than one
    distinct nonblank original_loan_term across their own rows; a row-level
    filter would keep only their 360-term rows instead of dropping them.)
    Pass None to disable (diagnostics only). Applied BEFORE the calendar
    cutoff filter, sort, loan_age derivation, or any window/sequence
    construction, so a dropped loan never reaches those stages.

    cell_sample_path: loan_id CSV (as produced by
    build_cell_grid_sample_pre2013.py) used to gate the historical-era
    (PRE2013_VINTAGES) population -- see _load_pre2013_cell_sample_ids().
    Default is PRE2013_CELL_SAMPLE_PATH (the original, pre-30y-filter
    sample) for backward compatibility; pass e.g.
    outputs/pre2013_cell_sample_30y_loans.csv to use a fresher sample.
    Ignored for modern-era vintages.
    """
    path = os.path.join(_data_dir_for_vintage(vintage), f'{vintage}.csv')
    if not os.path.exists(path):
        return None

    # HISTORICAL-ERA GATE: restrict loan selection to the cell-grid sample at
    # cell_sample_path, instead of processing the full pre-2013 corpus. Only
    # ever fires for PRE2013_VINTAGES -- modern-era `keep_ids` (None in Pass
    # 1/3's initial load, or the train/test id_set passed elsewhere) passes
    # through completely untouched.
    if vintage in _PRE2013_VINTAGE_SET:
        cell_ids = _load_pre2013_cell_sample_ids(cell_sample_path)
        keep_ids = cell_ids if keep_ids is None else (set(keep_ids) & cell_ids)

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

    # ── 30-year filter — must run before cutoff/sort/loan_age/windows ────────
    # Advisor's Sep 29 decision: 30-year loans only, throughout. Applied here,
    # immediately after the raw rows are materialized, so every downstream
    # computation (loan_age, L, term_t, windows) only ever sees 360-term rows.
    # LOAN-LEVEL, not row-level: a loan with ANY row whose original_loan_term
    # != term_filter (including blank/NaN -- NaN != x is True) is dropped
    # entirely, never partially kept. A row-level filter would instead keep
    # just the matching rows of a mixed-term loan, which 3 real loans in this
    # dataset would otherwise trigger (see docstring).
    if term_filter is not None:
        df['original_loan_term'] = pd.to_numeric(df['original_loan_term'], errors='coerce')
        n_loans_before = df['loan_id'].nunique()
        bad_term_loans = set(df.loc[df['original_loan_term'] != term_filter, 'loan_id'].unique())
        df = df[~df['loan_id'].isin(bad_term_loans)].copy()
        n_loans_after = df['loan_id'].nunique()
        print(f'    term filter (=={term_filter}, loan-level): kept {n_loans_after:,} of '
              f'{n_loans_before:,} loans ({n_loans_before - n_loans_after:,} dropped)',
              flush=True)
        if df.empty:
            return None

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

    # ── Post-modification drop — must run before loan_age/window/sequence ───
    # Advisor's Sep 29 decision: drop every loan-month from the first
    # modification_flag=='Y' month onward, per loan, PERMANENTLY -- this is
    # a DESIGN CHOICE ("drop all post-modification data"), not a claim that
    # modification_flag itself never reverts. The flag WAS found non-monotone
    # in 2002Q1 (142 loans show a later N after an earlier Y -- see
    # scripts/diag/verify_term_mod_filters.py's decisive decode, 2026-10-01);
    # the sticky "first Y onward" rule below still drops such a loan from its
    # first Y forward regardless of any later reversion (see the reversion
    # diagnostic below, which reports but does not block on this).
    # CONSEQUENCE: a loan that is modified and later terminates has its
    # termination row dropped by this filter, so it becomes CENSORED at its
    # last pre-mod month (is_prepaid=False, term_t=L-1 in _prepare_panel
    # below) -- it is NOT excluded from the panel entirely, UNLESS its very
    # first kept row is already a Y month (modified before or at the first
    # row surviving the term/cutoff filters above), in which case every row
    # is dropped and the loan vanishes from this vintage's panel for this
    # cutoff.
    mod_y = df['modification_flag'] == 'Y'
    first_y_month = (df.loc[mod_y].groupby('loan_id')['yyyymm'].min())
    df['_first_y_month'] = df['loan_id'].map(first_y_month)
    on_or_after_first_y = df['_first_y_month'].notna() & (df['yyyymm'] >= df['_first_y_month'])
    # Reversion diagnostic -- NOT an assertion. modification_flag is treated
    # as sticky by design (drop from first Y onward, permanently); a later
    # N after that first Y does not change the drop, it is just reported here
    # so the reversion rate is visible per vintage.
    violation = on_or_after_first_y & (df['modification_flag'] == 'N')
    n_violations = int(violation.sum())
    if n_violations > 0:
        n_viol_loans = int(df.loc[violation, 'loan_id'].nunique())
        first_reversion_month = df.loc[violation].groupby('loan_id')['yyyymm'].min()
        hist = first_reversion_month.value_counts().sort_index()
        print(f'    modification_flag reversion diagnostic: {n_viol_loans:,} loans '
              f'({n_violations:,} loan-months) show an N after a prior Y -- '
              f'dropped anyway under the sticky rule. First-reversion-month '
              f'histogram:', flush=True)
        for ym, cnt in hist.items():
            print(f'      {int(ym)}: {int(cnt):,} loans', flush=True)
    n_loans_dropped_any = int(df.loc[on_or_after_first_y, 'loan_id'].nunique())
    n_rows_dropped = int(on_or_after_first_y.sum())
    df = df[~on_or_after_first_y].drop(columns=['_first_y_month']).copy()
    print(f'    post-mod drop: {n_rows_dropped:,} loan-months dropped across '
          f'{n_loans_dropped_any:,} ever-modified loans', flush=True)
    if df.empty:
        return None

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
    df['current_actual_upb']       = pd.to_numeric(df['current_actual_upb'],       errors='coerce')

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

    # LOAN AGE -- derived from origination, NOT read from the file's loan_age.
    # Fannie leaves loan_age BLANK on the payoff row (verified: 71,559 of
    # 71,559 zbc==1 rows in 2015Q1 have it null). Since loan_age_months is in
    # FEATURE_COLS, the dropna below then deleted 100% of prepayment rows,
    # leaving prepay_timestep all -1 while the loan-level label (computed
    # before the dropna) survived. Same root cause as the age-keyed realized
    # CPR bug.
    # Offset: the file's loan_age runs one month behind months-elapsed-since
    # origination (382,207 of ~400k non-null rows at derived-minus-field == 1).
    # A ~4.4% tail sits at 0/2/6/9, most likely first-payment-date variation
    # (Fannie counts from when interest begins accruing, not the note date) --
    # not investigated further; immaterial at sequence granularity.
    _orig = pd.to_numeric(df['origination_date'], errors='coerce').map(
        mmyyyy_to_yyyymm, na_action='ignore')
    df['loan_age_months'] = (
        (df['yyyymm'] // 100 - _orig // 100) * 12
        + (df['yyyymm'] % 100 - _orig % 100)
        - 1
    ).clip(lower=0).astype(float)  # clip: reporting month == origination gives -1
    df['dti']             = pd.to_numeric(df['dti'], errors='coerce')

    # ── FIXED categorical encodings ───────────────────────────────────────────
    # Production pipeline used wrong maps (N/Y and P/R/C → all zeros).
    # Correct Fannie Mae codes:
    #   loan_purpose:  R=Purchase, C=Refinance, P=Cash-out Refinance, U=Unknown
    #   property_type: SF=Single-family, PU=Planned unit dev, CO=Condo,
    #                  MH=Manufactured, CP=Co-op
    # U and CP are legitimate documented codes present in BOTH eras (denser
    # pre-2013, e.g. 2000Q1 has 357 distinct CP loans, 0.8% of that vintage's
    # loans -- but also confirmed in every modern-era vintage, see
    # _encode_categorical's docstring for the two independent checks) --
    # previously both silently fell back to 0 (indistinguishable from a
    # genuine Purchase/Single-family row), now each gets its own code.
    _lp_map = loan_purpose_map  if loan_purpose_map  is not None else {'R': 0, 'C': 1, 'P': 2, 'U': 3}
    _pt_map = property_type_map if property_type_map is not None else {'SF': 0, 'PU': 1, 'CO': 2, 'MH': 3, 'CP': 4}
    df['loan_purpose_enc'] = _encode_categorical(df['loan_purpose'], _lp_map, 'loan_purpose')
    df['property_type_enc'] = _encode_categorical(df['property_type'], _pt_map, 'property_type')

    # ── Prepay label — CRITICAL: only from rows within cutoff window ──────────
    # Any row with zbc==1 at monthly_reporting_period <= cutoff is a prepay event.
    # Loans prepaying after the cutoff are labeled 0 (genuinely unknown at t=cutoff).
    prepaid_set   = set(df.loc[df['zero_balance_code_actual'] == 1.0, 'loan_id'].unique())
    df['prepaid'] = df['loan_id'].isin(prepaid_set).astype(int)

    # original_interest_rate rides along unused by build_sequences_multiobs
    # (which selects columns from FEATURE_COLS by name) -- kept so
    # census_panel_baseline.py can coupon-bucket the panel without a second
    # raw-file pass. extra_keep_cols: same idea, for callers that need one
    # more raw column (e.g. current_actual_upb for UPB-weighting) without a
    # second pass over the same multi-GB file -- default None/[] is a no-op
    # for every existing caller.
    keep = (['loan_id', 'yyyymm', 'monthly_reporting_period', 'prepaid',
             'zero_balance_code_actual', 'original_interest_rate'] + FEATURE_COLS
            + list(extra_keep_cols or []))
    df = df[keep].dropna(subset=FEATURE_COLS)

    n_loans   = df['loan_id'].nunique()
    prepay_rt = df.groupby('loan_id')['prepaid'].first().mean() * 100
    print(f'    -> {n_loans:,} loans | prepay {prepay_rt:.2f}% | rows: {len(df):,}', flush=True)
    return df


# ── Deterministic hashing ───────────────────────────────────────────────────
# One blake2b digest per UNIQUE loan_id -- O(n_loans), the only place we pay
# for cryptographic hashing. Every candidate (loan_id, ref_month) then gets a
# cheap integer mix of that digest with the month (O(n_rows), no more hashlib
# calls). Reproducible across processes/machines (unlike Python's salted
# str hash()) and monotone-growable: bottom-k selection on this fixed order
# means raising k only adds observations, never re-rolls existing ones.

_HASH_GOLDEN = 0x9E3779B97F4A7C15
_HASH_MUL1   = 0xBF58476D1CE4E5B9
_HASH_MUL2   = 0x94D049BB133111EB


def _loan_base_hash(loan_id) -> int:
    digest = hashlib.blake2b(str(loan_id).encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big')


def _mix_hash(base_hash: np.ndarray, ref_month: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64-style finalizer mixing a per-loan base hash with
    ref_month (yyyymm). base_hash/ref_month must already be np.uint64."""
    m = np.uint64(_HASH_GOLDEN)
    x = base_hash ^ (ref_month * m)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(_HASH_MUL1)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(_HASH_MUL2)
    x = x ^ (x >> np.uint64(31))
    return x


_OBS_COLS = ['loan_id', 't', 'ref_month', 'label', 'is_terminal', 'incl_prob',
             'n_eligible', 'k_actual', 'age_at_ref', 'incentive_at_ref']


def _empty_obs_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in _OBS_COLS})


# ── Panel prep ────────────────────────────────────────────────────────────────

def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a loan panel chronologically and attach row_idx / L / term_t /
    calendar-gap bookkeeping.

    row_idx : 0-based position within the loan, chronological.
    L       : number of rows the loan has (in this cutoff-filtered panel).
    term_t  : index of the FIRST zero_balance_code_actual==1 row for a
              prepaid loan (taken explicitly from the event index, never
              assumed to be the last row), else L-1 for a censored loan.
    monthidx: yyyymm collapsed to a single monotone integer (year*12+month),
              so gaps can be measured with plain subtraction.
    gap_flag: True at row i if row i-1 was NOT the immediately preceding
              calendar month -- i.e. dropna(subset=FEATURE_COLS) upstream
              removed an interior row for this loan, and row_idx now
              understates the true calendar distance across that point.
    cumgap  : running count of gap_flag up to and including this row, so
              "does the range (row_idx=a, row_idx=b] contain a gap" is a
              single subtraction: cumgap[b] - cumgap[a] > 0.
    """
    df = df.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
    df['row_idx'] = df.groupby('loan_id').cumcount()
    df['L']       = df.groupby('loan_id')['row_idx'].transform('max') + 1

    zbc_idx = (df.loc[df['zero_balance_code_actual'] == 1.0]
                 .groupby('loan_id')['row_idx'].min())
    df['is_prepaid'] = df['loan_id'].isin(zbc_idx.index)
    df['term_t']     = df['loan_id'].map(zbc_idx)
    df['term_t']     = df['term_t'].fillna(df['L'] - 1).astype(int)

    df['monthidx']      = (df['yyyymm'] // 100) * 12 + (df['yyyymm'] % 100)
    prev_monthidx        = df.groupby('loan_id')['monthidx'].shift(1)
    df['gap_flag']       = ((df['monthidx'] - prev_monthidx) > 1).fillna(False)
    df['cumgap']         = df.groupby('loan_id')['gap_flag'].cumsum()
    return df


# ── Per-loan fixed-k sampling of (loan_id, ref_month) observations ────────────

def _eligible_candidates(df: pd.DataFrame, H: int, min_hist: int) -> pd.DataFrame:
    """Panel prep + eligibility mask + both calendar filters -- the FULL
    (loan_id, ref_month) candidate population, before any sampling is
    applied. Factored out of select_observations() (which samples from this
    set) so scripts/diag/census_panel_baseline.py can enumerate the SAME set
    with no sampling at all, as the reference any sampler must reproduce
    under IPW. Do not reimplement this filtering a second time elsewhere.

    Returns the filtered candidate frame, carrying every _prepare_panel /
    df column (row_idx, L, term_t, is_prepaid, monthidx, cumgap, and all
    original df columns) -- one row per eligible (loan_id, ref_month).
    Empty frame (not None) if nothing survives; callers decide what an
    empty result means for them.
    """
    panel = _prepare_panel(df)

    eligible = (
        (panel['row_idx'] >= (min_hist - 1)) &
        (panel['row_idx'] <= (panel['L'] - 1 - H)) &
        (panel['row_idx'] < panel['term_t'])          # LOAD-BEARING, see module docstring
    )
    elig = panel[eligible].copy()
    if elig.empty:
        return elig

    # ── CALENDAR filters (row_idx is a post-dropna POSITION, not a calendar
    # index -- see _prepare_panel). Both are applied to `elig` BEFORE the
    # mandatory/pool split, so a dropped mandatory candidate correctly falls
    # back to "no mandatory draw for this loan" (full k budget from the
    # pool), the same as the existing min_hist skip. Reference point from
    # the 2015Q1/cutoff_2020 one-vintage diagnostic at k=5,H=1: 18 / 2,166,647
    # observations (10 / 435,443 loans) hit the window-gap filter, and
    # 4 / 435,443 terminal draws hit the label-window filter.
    row_idx_lookup  = panel.set_index(['loan_id', 'row_idx'])
    cumgap_at_row   = row_idx_lookup['cumgap']
    monthidx_at_row = row_idx_lookup['monthidx']

    # Filter 2 -- feature window spans an interior calendar gap: drop any
    # candidate whose drawn window (row_idx in (s, t]) contains a gap_flag,
    # since build_sequences_multiobs would then silently splice together
    # non-adjacent calendar months as if they were adjacent.
    elig['_win_start'] = np.maximum(0, elig['row_idx'] - MAX_SEQ_LEN + 1)
    cumgap_t = cumgap_at_row.reindex(list(zip(elig['loan_id'], elig['row_idx']))).to_numpy()
    cumgap_s = cumgap_at_row.reindex(list(zip(elig['loan_id'], elig['_win_start']))).to_numpy()
    drop_window_gap = (cumgap_t - cumgap_s) > 0

    # Filter 1 -- label window: for a real prepay, the calendar distance
    # from ref_month to the event month must equal term_t - row_idx (the
    # row-index distance H is defined in terms of). If a gap sits between
    # them, "H months ahead" means a different number of calendar months
    # for this loan than for every other loan, silently.
    is_prepaid = elig['is_prepaid'].to_numpy()
    event_monthidx = monthidx_at_row.reindex(
        list(zip(elig['loan_id'], elig['term_t']))).to_numpy()
    calendar_dist = event_monthidx - elig['monthidx'].to_numpy()
    row_dist      = (elig['term_t'] - elig['row_idx']).to_numpy()
    drop_label_mismatch = is_prepaid & (calendar_dist != row_dist)

    n_dropped_window_gap     = int(drop_window_gap.sum())
    n_dropped_label_mismatch = int(drop_label_mismatch.sum())
    print(f'  [multiobs calendar filters] dropped {n_dropped_window_gap:,} obs '
          f'(feature window spans a calendar gap), {n_dropped_label_mismatch:,} obs '
          f'(label-window row/calendar distance mismatch) '
          f'out of {len(elig):,} otherwise-eligible candidates', flush=True)

    elig = elig[~(drop_window_gap | drop_label_mismatch)].drop(columns=['_win_start'])
    return elig


def select_observations(
    df: pd.DataFrame, k_draws: int, H: int, min_hist: int,
    draw_scheme: str = 'uniform', incentive_edges: list[float] | None = None,
    sampling_mode: str = 'fixed_k', frac_draws: float | None = None,
) -> pd.DataFrame:
    """Select (loan_id, ref_month) observations per loan under one of two
    sampling_mode's:

    fixed_k (default, UNCHANGED from the original design): up to k_draws
        observations per loan -- k_draws-1 non-mandatory slots when a
        mandatory/terminal draw exists, else k_draws. incl_prob for a
        non-mandatory draw is budget/n_pool, PROPORTIONAL TO LOAN LENGTH
        (a longer loan's pool is larger, so each of its non-mandatory
        candidates is individually less likely to be drawn) -- this is the
        length bias fixed-fraction mode exists to remove.

    fixed_fraction: every loan's non-mandatory budget is
        ceil(frac_draws * n_pool) instead of k_draws-derived, where n_pool
        is the SAME quantity this function already emits as the per-
        observation `n_eligible` field (the non-mandatory pool size --
        see _OBS_COLS/out['n_eligible'] below; NOT the mandatory+pool total
        census_panel_baseline.py reports under the same name for a DIFFERENT
        loan-level statistic -- the two must not be confused). Because
        n_pool cancels in budget/n_pool = ceil(f*n_pool)/n_pool, the
        non-mandatory incl_prob is CONSTANT ≈ frac_draws regardless of loan
        length, unlike fixed_k. Two honest caveats, not swept under the rug:
          (a) ceil() rounds up, so incl_prob is always >= frac_draws, most
              visibly for small pools -- a pool of size 1 gets incl_prob
              exactly 1.0 for ANY frac_draws in (0, 1], since you cannot
              draw a fractional observation. The emitted incl_prob is
              always the REALIZED per-loan probability (never the nominal
              frac_draws), so IPW stays valid even where it deviates from
              frac_draws -- but "constant f" is an asymptotic property here,
              not exact for short-history loans.
          (b) The MANDATORY/TERMINAL draw's incl_prob is untouched by
              sampling_mode -- see below.

    TERMINAL-DRAW CONVENTION (a real design decision, not incidental):
        the mandatory draw at row_idx = term_t - H keeps incl_prob = 1.0
        (deterministic inclusion) under BOTH sampling_mode's, exactly as
        in the original fixed_k design. Under fixed_fraction this means the
        terminal observation's weight (1/incl_prob = 1) is NOT 1/frac_draws
        like every other (non-terminal) observation's weight -- it is a
        distinct stratum, oversampled by construction, same as it always
        was under fixed_k. This function does not attempt to correct that
        via incl_prob/weight; per the module docstring, "IPW and the
        King-Zeng intercept correction for terminal-month oversampling are
        TRAINING-time decisions" -- unchanged by this feature.
        Two alternatives considered and NOT implemented:
          - Assign the terminal draw incl_prob = frac_draws too (uniform
            weight 1/frac_draws everywhere). REJECTED: it is not actually
            included with probability frac_draws, it is included with
            probability 1 by construction -- assigning frac_draws would
            make the emitted incl_prob simply wrong for that row, breaking
            Horvitz-Thompson correctness for every prepaid loan's one
            guaranteed observation.
          - Drop the mandatory-draw guarantee entirely and let the terminal
            candidate compete for inclusion at probability frac_draws like
            any other pool row (a fully homogeneous incl_prob=frac_draws
            design, no special-casing). REJECTED for this step: it would
            let frac_draws-many prepaid loans (especially short-history
            ones) end up with ZERO sampled observations at all, starving
            training exposure to the event -- a materially different
            design than "currently included with probability 1", which the
            task specification treats as a given, not something to change.

    df needs loan_id, yyyymm, zero_balance_code_actual, and (for the
    incentive scheme / verification output) refi_incentive, loan_age_months
    — all UNSCALED. Returns one row per selected observation; see _OBS_COLS.
    """
    if sampling_mode == 'fixed_fraction':
        if frac_draws is None or not (0.0 < frac_draws <= 1.0):
            raise ValueError(f'sampling_mode=fixed_fraction requires frac_draws in (0, 1], '
                              f'got {frac_draws!r}')
    elif sampling_mode != 'fixed_k':
        raise ValueError(f'unknown sampling_mode {sampling_mode!r}')

    elig = _eligible_candidates(df, H, min_hist)
    if elig.empty:
        return _empty_obs_frame()

    # The mandatory draw is a member of elig (when eligible) because it
    # auto-satisfies the other two eligibility bounds whenever H >= 1:
    #   term_t-H <= L-1-H always (term_t <= L-1), and term_t-H < term_t always.
    # Only the min_hist bound can exclude it, matching "skip if term_t-H <
    # min_hist-1" exactly.
    elig['is_mandatory'] = elig['row_idx'] == (elig['term_t'] - H)

    base_hash_map = {lid: _loan_base_hash(lid) for lid in elig['loan_id'].unique()}
    base_hash_arr = np.array(elig['loan_id'].map(base_hash_map).tolist(), dtype=np.uint64)
    month_arr     = elig['yyyymm'].to_numpy().astype(np.uint64)
    elig['hash']  = _mix_hash(base_hash_arr, month_arr)

    mandatory = elig[elig['is_mandatory']].copy()
    pool      = elig[~elig['is_mandatory']].copy()

    has_mandatory = elig.groupby('loan_id')['is_mandatory'].any()
    n_pool        = pool.groupby('loan_id').size()

    if sampling_mode == 'fixed_k':
        budget = (k_draws - has_mandatory.astype(int))  # non-mandatory slots per loan
    else:  # fixed_fraction -- budget is a fraction of the POOL (n_pool), not
           # of n_pool+has_mandatory; see docstring for why. Ceil'd per loan,
           # so it's always an integer count downstream code already expects.
        budget = np.ceil(frac_draws * n_pool).astype(int)

    if pool.empty:
        selected_pool = pool.assign(incl_prob=pd.Series(dtype=float))
    elif draw_scheme == 'uniform':
        pool['rank']   = pool.groupby('loan_id')['hash'].rank(method='first').astype(int)
        pool['budget'] = pool['loan_id'].map(budget).astype(int)
        pool['n_pool'] = pool['loan_id'].map(n_pool).astype(int)
        selected_pool  = pool[pool['rank'] <= pool['budget']].copy()
        selected_pool['incl_prob'] = (
            np.minimum(selected_pool['budget'], selected_pool['n_pool'])
            / selected_pool['n_pool']
        )
    elif draw_scheme == 'incentive':
        selected_pool = _select_incentive_stratified(pool, budget, incentive_edges)
    else:
        raise ValueError(f'unknown draw_scheme {draw_scheme!r}')

    selected_pool['is_terminal'] = False
    mandatory['incl_prob']       = 1.0
    mandatory['is_terminal']     = True

    out = pd.concat([mandatory, selected_pool], ignore_index=True, sort=False)
    if out.empty:
        return _empty_obs_frame()

    out['n_eligible'] = out['loan_id'].map(n_pool).fillna(0).astype(int)
    k_actual          = out.groupby('loan_id').size()
    out['k_actual']   = out['loan_id'].map(k_actual).astype(int)

    # label: 1 iff the loan's terminal event is a real prepay AND falls in
    # (ref_month, ref_month+H]. Eligibility already guarantees row_idx <
    # term_t, so term_t - row_idx is always >= 1 here.
    out['label'] = (out['is_prepaid'] & ((out['term_t'] - out['row_idx']) <= H)).astype(np.float32)
    out['t']                = out['row_idx'].astype(int)
    out['ref_month']        = out['yyyymm'].astype(int)
    out['age_at_ref']       = out['loan_age_months'].astype(float)
    out['incentive_at_ref'] = out['refi_incentive'].astype(float)

    return out[_OBS_COLS].reset_index(drop=True)


def _select_incentive_stratified(
    pool: pd.DataFrame, budget: pd.Series, incentive_edges: list[float],
) -> pd.DataFrame:
    """Round-robin the k-1 budget across a loan's non-empty refi_incentive
    bins (UNSCALED), bottom-k by hash WITHIN each bin, per-stratum incl_prob.
    """
    edges  = [-np.inf] + list(incentive_edges) + [np.inf]
    n_bins = len(edges) - 1
    pool   = pool.copy()
    pool['bin'] = pd.cut(pool['refi_incentive'], bins=edges, labels=False,
                          include_lowest=True).astype(int)

    bin_sizes = (pool.groupby(['loan_id', 'bin']).size()
                     .unstack(fill_value=0)
                     .reindex(columns=range(n_bins), fill_value=0))
    loan_order = bin_sizes.index
    sizes_mat  = bin_sizes.to_numpy(dtype=int)
    budget_arr = budget.reindex(loan_order).fillna(0).to_numpy(dtype=int)

    allocated = np.zeros_like(sizes_mat)
    remaining = budget_arr.copy()
    # At most `budget` rounds are ever needed: each round hands out at most
    # one slot per active bin, so a single-bin loan drains its whole budget
    # in `budget` rounds and a multi-bin loan finishes sooner.
    for _ in range(int(budget_arr.max()) if len(budget_arr) else 0):
        if not remaining.any():
            break
        for b in range(n_bins):
            can_take = (remaining > 0) & (allocated[:, b] < sizes_mat[:, b])
            allocated[can_take, b] += 1
            remaining[can_take]    -= 1

    alloc_df   = pd.DataFrame(allocated, index=loan_order, columns=range(n_bins))
    alloc_long = alloc_df.stack().rename('alloc').reset_index()
    alloc_long.columns = ['loan_id', 'bin', 'alloc']

    pool['rank_in_bin'] = pool.groupby(['loan_id', 'bin'])['hash'].rank(method='first').astype(int)
    pool['bin_size']    = pool.groupby(['loan_id', 'bin'])['loan_id'].transform('size')
    pool = pool.merge(alloc_long, on=['loan_id', 'bin'], how='left')
    pool['alloc'] = pool['alloc'].fillna(0).astype(int)

    selected = pool[pool['rank_in_bin'] <= pool['alloc']].copy()
    selected['incl_prob'] = selected['alloc'] / selected['bin_size']
    return selected


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences_multiobs(
    df: pd.DataFrame, scaler: StandardScaler, k_draws: int, H: int, min_hist: int,
    draw_scheme: str = 'uniform', incentive_edges: list[float] | None = None,
    sampling_mode: str = 'fixed_k', frac_draws: float | None = None,
    obs: pd.DataFrame | None = None,
):
    """Build RIGHT-aligned (N, MAX_SEQ_LEN, N_FEATURES) arrays: one row per
    SELECTED (loan_id, ref_month) observation, not one row per loan.

    The last timestep (index MAX_SEQ_LEN-1) is always ref_month itself;
    shorter histories are left-padded with zeros/mask=False. This differs
    from the trailing builder's LEFT-aligned convention on purpose — here
    the reference age varies observation to observation, so "last timestep
    = now" must hold for every row for the positional embedding to mean
    anything consistent.

    obs: optional pre-built observation table, same schema as
    select_observations()'s return (_OBS_COLS: loan_id, t, ref_month, label,
    is_terminal, incl_prob, n_eligible, k_actual, age_at_ref,
    incentive_at_ref). When given, select_observations() is skipped and this
    table drives the gather directly -- lets a caller substitute exactly one
    right-aligned window per loan (e.g. one Dec-2002 ref_month per test
    loan) without duplicating the gather/scaling logic below.
    """
    if obs is None:
        obs = select_observations(df, k_draws, H, min_hist, draw_scheme, incentive_edges,
                                   sampling_mode, frac_draws)
    extras_dtype = {
        'incl_prob': np.float32, 'is_terminal': bool, 'n_eligible': np.int32,
        'k_actual': np.int32, 'ref_month': np.int64,
        'age_at_ref': np.float32, 'incentive_at_ref': np.float32,
    }
    if obs.empty:
        empty_extras = {k: np.zeros(0, dtype=v) for k, v in extras_dtype.items()}
        return (np.zeros((0, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32),
                np.zeros((0, MAX_SEQ_LEN), dtype=bool),
                np.zeros(0, dtype=np.float32),
                np.full(0, -1, dtype=np.int32),
                np.array([], dtype=object),
                empty_extras)

    panel = df.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
    panel['row_idx'] = panel.groupby('loan_id').cumcount()
    feat_scaled = scaler.transform(panel[FEATURE_COLS]).astype(np.float32)

    # First-occurrence position of each loan_id in `panel` -- panel is sorted
    # by loan_id so every loan is one contiguous block; row_idx t of a loan
    # therefore sits at absolute position loan_start[loan_id] + t.
    first_mask = panel['loan_id'].ne(panel['loan_id'].shift(1))
    loan_start = pd.Series(np.flatnonzero(first_mask.to_numpy()),
                            index=panel.loc[first_mask, 'loan_id'].to_numpy())

    end_pos = obs['loan_id'].map(loan_start).to_numpy(dtype=np.int64) + obs['t'].to_numpy(dtype=np.int64)
    length  = np.minimum(MAX_SEQ_LEN, obs['t'].to_numpy(dtype=np.int64) + 1)

    n_obs = len(obs)
    rel      = (MAX_SEQ_LEN - 1) - np.arange(MAX_SEQ_LEN)[None, :]          # (1, W): W-1..0
    src_idx  = end_pos[:, None] - rel                                       # (n_obs, W)
    valid    = np.arange(MAX_SEQ_LEN)[None, :] >= (MAX_SEQ_LEN - length[:, None])
    src_idx  = np.clip(src_idx, 0, feat_scaled.shape[0] - 1)

    gathered  = feat_scaled[src_idx]                     # (n_obs, W, N_FEATURES)
    sequences = np.where(valid[..., None], gathered, 0.0).astype(np.float32)
    masks     = valid

    labels = obs['label'].to_numpy(dtype=np.float32)
    # MEANINGLESS here by construction (see module docstring) -- the event,
    # if any, is strictly outside the feature window. Kept only for
    # file-shape compatibility with the trailing builder's output.
    prepay_t = np.full(n_obs, -1, dtype=np.int32)
    loan_ids_out = obs['loan_id'].to_numpy()

    extras = {
        'incl_prob':        obs['incl_prob'].to_numpy(dtype=np.float32),
        'is_terminal':      obs['is_terminal'].to_numpy(dtype=bool),
        'n_eligible':       obs['n_eligible'].to_numpy(dtype=np.int32),
        'k_actual':         obs['k_actual'].to_numpy(dtype=np.int32),
        'ref_month':        obs['ref_month'].to_numpy(dtype=np.int64),
        'age_at_ref':       obs['age_at_ref'].to_numpy(dtype=np.float32),
        'incentive_at_ref': obs['incentive_at_ref'].to_numpy(dtype=np.float32),
    }

    return sequences, masks, labels, prepay_t, loan_ids_out, extras


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year', type=int, required=True,
                        help='Train through December of this year (e.g. 2018)')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                        help='Loan subsampling fraction for pass 1 (default=1.0)')
    parser.add_argument('--max_seq_len', type=int, default=_DEFAULT_SEQ_LEN,
                        help='Months of history per loan (default=33). '
                             'Non-default values append _L{n} to the output dir.')
    parser.add_argument('--k_draws', type=int, default=5,
                        help='Fixed number of (loan_id, ref_month) observations '
                             'drawn per loan (default=5). Used only when '
                             '--sampling_mode=fixed_k; ignored under fixed_fraction.')
    parser.add_argument('--sampling_mode', choices=['fixed_k', 'fixed_fraction'], default='fixed_k',
                        help='fixed_k (default, unchanged): k_draws observations per loan. '
                             'fixed_fraction: ceil(frac_draws * n_pool) non-mandatory '
                             'observations per loan, giving every non-terminal eligible '
                             'month the SAME inclusion probability regardless of loan '
                             'length -- see select_observations() docstring.')
    parser.add_argument('--frac_draws', type=float, default=None,
                        help='Fraction in (0, 1] of each loan\'s non-mandatory eligible '
                             'pool to draw. Required when --sampling_mode=fixed_fraction; '
                             'must not be set otherwise.')
    parser.add_argument('--label_horizon', type=int, default=1,
                        help='H: label window is (ref_month, ref_month+H] months (default=1).')
    parser.add_argument('--min_hist', type=int, default=1,
                        help='Minimum months of history required at the reference '
                             'row (t >= min_hist-1, default=1).')
    parser.add_argument('--draw_scheme', choices=['uniform', 'incentive'], default='uniform',
                        help='How the k-1 non-mandatory draws are sampled (default=uniform).')
    parser.add_argument('--incentive_edges', type=str, default='-0.25,0.25',
                        help='Comma-separated UNSCALED refi_incentive bin edges for '
                             '--draw_scheme incentive (default="-0.25,0.25").')
    parser.add_argument('--reuse_from', type=str, default=None,
                        help='Dir to copy train/test loan-id splits and scaler.pkl from, '
                             'so this build is directly comparable to that one. Copied '
                             'BEFORE the resume guards run; errors hard if anything is missing.')
    parser.add_argument('--include_pre2013', action='store_true',
                        help='Include the historical era (2000Q1-2012Q4 acquisition '
                             'files, PRE2013_VINTAGES), gated to the cell-grid sample at '
                             '--cell_sample instead of the full 29.1M-loan pre-2013 '
                             'corpus. Default off -- every modern-era (2013+) invocation '
                             'is byte-for-byte unaffected: ALL_VINTAGES only gains the '
                             'historical files when this flag is passed.')
    parser.add_argument('--cell_sample', type=str, default=PRE2013_CELL_SAMPLE_PATH,
                        help='loan_id CSV (build_cell_grid_sample_pre2013.py output) '
                             'gating the historical-era population when --include_pre2013 '
                             'is passed; default is the original pre-30y-filter sample '
                             '(outputs/pre2013_cell_sample_loans.csv) for backward '
                             'compatibility. Ignored without --include_pre2013.')
    args = parser.parse_args()

    if args.sampling_mode == 'fixed_fraction':
        if args.frac_draws is None or not (0.0 < args.frac_draws <= 1.0):
            parser.error('--sampling_mode=fixed_fraction requires --frac_draws in (0, 1]')
    elif args.frac_draws is not None:
        parser.error('--frac_draws is only used with --sampling_mode=fixed_fraction '
                      '(sampling_mode is fixed_k, --k_draws applies instead)')

    global MAX_SEQ_LEN, ALL_VINTAGES
    MAX_SEQ_LEN = args.max_seq_len
    # Rebind ALL_VINTAGES here (same technique as MAX_SEQ_LEN above) so every
    # loop that reads the global -- Pass 1/2/3 directly in main(), and Pass
    # 4's _concat_split closure -- sees the same list. Without --include_pre2013
    # this is a no-op: ALL_VINTAGES stays exactly _MODERN_VINTAGES, the same
    # object every prior invocation of this script saw.
    ALL_VINTAGES = (PRE2013_VINTAGES + _MODERN_VINTAGES) if args.include_pre2013 else _MODERN_VINTAGES
    incentive_edges = [float(x) for x in args.incentive_edges.split(',')] if args.incentive_edges else []

    cutoff_ym = dec_yyyymm(args.cutoff_year)     # e.g. 201812

    # RELEVANT_VINTAGES: files whose acquisition quarter starts after the
    # cutoff can contain no row with monthly_reporting_period <= cutoff_ym
    # (see _vintage_quarter_start_yyyymm's docstring for the empirical check
    # this relies on -- no safety margin needed). Computed once here and used
    # at all four ALL_VINTAGES loop sites (Pass 1/2/3 below, and Pass 4's
    # _concat_split closure) so e.g. a cutoff_2002 build with --include_pre2013
    # never opens the 2013Q1-2023Q1 files it used to read and immediately
    # discard as empty.
    RELEVANT_VINTAGES = [v for v in ALL_VINTAGES
                         if _vintage_quarter_start_yyyymm(v) <= cutoff_ym]
    print(f'Relevant vintages: {len(RELEVANT_VINTAGES)}/{len(ALL_VINTAGES)} '
          f'(cutoff_ym={cutoff_ym}, files after cutoff skipped entirely)', flush=True)

    _scheme_suffix = '' if args.draw_scheme == 'uniform' else f'_{args.draw_scheme}'
    _cap = '' if args.max_seq_len == _DEFAULT_SEQ_LEN else f'_L{args.max_seq_len}'
    _budget_tag = f'k{args.k_draws}' if args.sampling_mode == 'fixed_k' else f'f{args.frac_draws}'
    _hist_suffix = '_hist' if args.include_pre2013 else ''
    SAVE_DIR = os.path.join(
        BASE, f'data/sequences_rolling/cutoff_{args.cutoff_year}_zbc_multiobs'
              f'_{_budget_tag}_h{args.label_horizon}{_scheme_suffix}{_cap}{_hist_suffix}')
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f'Multiobs builder | cutoff = Dec {args.cutoff_year} (YYYYMM={cutoff_ym}) | '
          f'sampling_mode={args.sampling_mode} {_budget_tag} H={args.label_horizon} '
          f'min_hist={args.min_hist} scheme={args.draw_scheme}', flush=True)
    print(f'Output dir: {SAVE_DIR}', flush=True)

    # --reuse_from MUST run before the resume guards below, so the copied
    # split/scaler are what the guards find and load -- a re-derived split
    # looks fine on its own but silently breaks comparability to the run
    # this one is meant to match.
    if args.reuse_from:
        _required = ['train_loan_ids_split.npy', 'test_loan_ids_split.npy', 'scaler.pkl']
        _missing  = [f for f in _required
                     if not os.path.exists(os.path.join(args.reuse_from, f))]
        if _missing:
            raise FileNotFoundError(
                f'--reuse_from {args.reuse_from} is missing required file(s): {_missing}')
        for f in _required:
            shutil.copy2(os.path.join(args.reuse_from, f), os.path.join(SAVE_DIR, f))
        print(f'Reused split/scaler from {args.reuse_from}', flush=True)

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
        # PER-VINTAGE CHECKPOINTING: pickle (next_idx, info_chunks) after each
        # vintage and resume from it if present, same discipline as
        # count_prepay_events_pre2013.py/build_cell_grid_sample_pre2013.py --
        # a long Pass 1 scan lost to a SLURM timeout previously had to restart
        # from vintage 0.
        pass1_ckpt_path = os.path.join(SAVE_DIR, 'pass1_ckpt.pkl')
        if os.path.exists(pass1_ckpt_path):
            with open(pass1_ckpt_path, 'rb') as f:
                _start_idx, info_chunks = pickle.load(f)
            print(f'  RESUME Pass 1 from vintage {_start_idx}/{len(RELEVANT_VINTAGES)}',
                  flush=True)
        else:
            _start_idx = 0
            info_chunks = []

        for vi in range(_start_idx, len(RELEVANT_VINTAGES)):
            v = RELEVANT_VINTAGES[vi]
            df = load_vintage_filtered(v, pmms_rates, zhvi_df, cutoff_ym,
                                       keep_ids=None, sample_frac=args.sample_frac,
                                       cell_sample_path=args.cell_sample)
            if df is not None and not df.empty:
                info_chunks.append(df.groupby('loan_id')['prepaid'].first().reset_index())
                del df; gc.collect()
            with open(pass1_ckpt_path, 'wb') as f:
                pickle.dump((vi + 1, info_chunks), f)

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
        # PER-VINTAGE CHECKPOINTING: pickle (next_idx, scaler, n_scaler_rows)
        # after each vintage and resume from it if present -- same discipline
        # as Pass 1 above and count_prepay_events_pre2013.py/
        # build_cell_grid_sample_pre2013.py.
        pass2_ckpt_path = os.path.join(SAVE_DIR, 'pass2_ckpt.pkl')
        if os.path.exists(pass2_ckpt_path):
            with open(pass2_ckpt_path, 'rb') as f:
                _start_idx, scaler, n_scaler_rows = pickle.load(f)
            print(f'  RESUME Pass 2 from vintage {_start_idx}/{len(RELEVANT_VINTAGES)}',
                  flush=True)
        else:
            _start_idx = 0
            scaler = StandardScaler()
            n_scaler_rows = 0

        for vi in range(_start_idx, len(RELEVANT_VINTAGES)):
            v = RELEVANT_VINTAGES[vi]
            try:
                path = os.path.join(_data_dir_for_vintage(v), f'{v}.csv')
                if not os.path.exists(path):
                    continue
                # No separate cell-grid filter needed here: this loop only keeps
                # rows with loan_id in train_id_set (below), and for pre-2013
                # vintages train_id_set is ALREADY cell-grid-gated -- it was
                # built from Pass 1's discovered loan population, which goes
                # through load_vintage_filtered's gate. Duplicating the filter
                # here would be redundant, not more correct.
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
                sample['loan_purpose_enc']  = _encode_categorical(
                    sample['loan_purpose'], {'R': 0, 'C': 1, 'P': 2, 'U': 3}, 'loan_purpose')
                sample['property_type_enc'] = _encode_categorical(
                    sample['property_type'], {'SF': 0, 'PU': 1, 'CO': 2, 'MH': 3, 'CP': 4}, 'property_type')

                valid = sample[FEATURE_COLS].dropna()
                if len(valid) > 0:
                    scaler.partial_fit(valid)
                    n_scaler_rows += len(valid)
                    print(f'  {v}: +{len(valid):,} rows  (total={n_scaler_rows:,})', flush=True)
                del sample, valid; gc.collect()
            finally:
                # Checkpoint fires on every path through the loop body above
                # (continue included), so resume always advances past a
                # vintage that was fully handled, not just ones that fit data.
                with open(pass2_ckpt_path, 'wb') as f:
                    pickle.dump((vi + 1, scaler, n_scaler_rows), f)

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

    for v in RELEVANT_VINTAGES:
        if _shard_done(v):
            print(f'  {v}: shard exists — skip', flush=True)
            continue

        df = load_vintage_filtered(v, pmms_rates, zhvi_df, cutoff_ym, keep_ids=None,
                                   cell_sample_path=args.cell_sample)
        if df is None or df.empty:
            # mark empty so rerun doesn't retry a vintage with no in-window rows
            open(os.path.join(shard_dir, f'{v}_empty.flag'), 'w').close()
            continue

        for split_name, id_set in (('train', train_id_set), ('test', test_id_set)):
            sub = df[df['loan_id'].isin(id_set)]
            if sub.empty:
                continue
            seq, mask, lbl, pt, lids, extra = build_sequences_multiobs(
                sub, scaler, args.k_draws, args.label_horizon, args.min_hist,
                args.draw_scheme, incentive_edges,
                args.sampling_mode, args.frac_draws)
            sp = os.path.join(shard_dir, f'{v}_{split_name}')
            np.save(f'{sp}_seq.npy',  seq)
            np.save(f'{sp}_mask.npy', mask)
            np.save(f'{sp}_lbl.npy',  lbl)
            np.save(f'{sp}_pt.npy',   pt)
            np.save(f'{sp}_ids.npy',  lids)
            for field, arr in extra.items():
                np.save(f'{sp}_{field}.npy', arr)
            del sub, seq, mask, lbl, pt, lids, extra; gc.collect()
        # mark vintage complete even if only one split had rows
        if not _shard_done(v):
            open(os.path.join(shard_dir, f'{v}_empty.flag'), 'w').close()
        del df; gc.collect()
        print(f'  {v}: shard written', flush=True)

    # ── Pass 4: stream per-vintage shards into final arrays ───────────────────
    # NOTE: at scale (e.g. 1,496,727 train loans x k=5 -> up to 7.5M
    # observations x 33 x 9 float32 -> ~8.9 GB) np.concatenate on the seq
    # shards would hold every shard PLUS a full concatenated copy resident at
    # once (~18 GB) and OOM. Instead we count total rows across shards from
    # their headers only (no full load), then stream each shard's seq array
    # directly into a pre-sized np.lib.format.open_memmap on a .tmp path, so
    # peak resident memory is one shard at a time, not the whole dataset.
    # Smaller per-observation arrays (mask/labels/weights/etc.) are fine to
    # concatenate normally.
    print('\nPass 4: streaming shards into final arrays...', flush=True)

    _EXTRA_FIELDS = ['incl_prob', 'is_terminal', 'n_eligible', 'k_actual',
                      'ref_month', 'age_at_ref', 'incentive_at_ref']

    def _concat_split(split_name):
        shard_paths = [os.path.join(shard_dir, f'{v}_{split_name}')
                       for v in RELEVANT_VINTAGES
                       if os.path.exists(os.path.join(shard_dir, f'{v}_{split_name}_seq.npy'))]
        if not shard_paths:
            raise RuntimeError(f'No shards found for {split_name}')

        total_n = 0
        tail_shape = None
        for sp in shard_paths:
            arr = np.load(f'{sp}_seq.npy', mmap_mode='r')
            total_n += arr.shape[0]
            if tail_shape is None:
                tail_shape = arr.shape[1:]
            del arr
        projected_gb = total_n * int(np.prod(tail_shape)) * 4 / 1e9   # float32
        print(f'  {split_name}: projected {total_n:,} observations, '
              f'~{projected_gb:.2f} GB for the seq array', flush=True)

        p = os.path.join(SAVE_DIR, split_name)
        tmp_seq_path = f'{p}_seq.tmp.npy'
        seq_mm = np.lib.format.open_memmap(
            tmp_seq_path, mode='w+', dtype=np.float32, shape=(total_n,) + tail_shape)
        offset = 0
        for sp in shard_paths:
            shard_seq = np.load(f'{sp}_seq.npy')
            n = shard_seq.shape[0]
            seq_mm[offset:offset + n] = shard_seq
            offset += n
            del shard_seq; gc.collect()
        seq_mm.flush()
        del seq_mm
        os.replace(tmp_seq_path, f'{p}_seq.npy')  # atomic last step

        def _concat_field(field, allow_pickle=False):
            pieces = [np.load(f'{sp}_{field}.npy', allow_pickle=allow_pickle) for sp in shard_paths]
            out = np.concatenate(pieces, axis=0)
            del pieces; gc.collect()
            return out

        out_mask = _concat_field('mask')
        out_lbl  = _concat_field('lbl')
        out_pt   = _concat_field('pt')
        out_ids  = _concat_field('ids', allow_pickle=True)
        np.save(f'{p}_mask.npy',            out_mask)
        np.save(f'{p}_labels.npy',          out_lbl)
        np.save(f'{p}_prepay_timestep.npy', out_pt)   # MEANINGLESS -- see module docstring
        np.save(f'{p}_loan_ids.npy',        out_ids)
        del out_mask, out_pt, out_ids; gc.collect()

        for field in _EXTRA_FIELDS:
            out_field = _concat_field(field)
            np.save(f'{p}_{field}.npy', out_field)
            del out_field; gc.collect()

        print(f'  {split_name}: shape=({total_n},)+{tail_shape}  '
              f'prepay={out_lbl.mean()*100:.2f}%', flush=True)
        del out_lbl; gc.collect()

    _concat_split('train')
    _concat_split('test')

    # cleanup shards once final arrays are written
    shutil.rmtree(shard_dir, ignore_errors=True)

    print(f'\nDone. cutoff={args.cutoff_year} | dir={SAVE_DIR}', flush=True)


if __name__ == '__main__':
    main()
