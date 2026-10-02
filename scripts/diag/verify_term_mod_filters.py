"""
Verification for the 30-year filter + post-mod drop patch (step 1 of the
rebuild, Sep 29 advisor decisions). Run under srun (see sbatch wrapper).

Covers D.1-D.5 of the task spec on ONE pre-2013 vintage file (2002Q1) and
ONE 2018+ vintage file (2018Q1):
  D.1  raw term distribution (ground truth, direct raw read, no cell gate)
  D.2  loan-months before/after term filter, before/after post-mod drop
  D.3  ever-modified loan count, dropped loan-months, monotonicity check
  D.4  of dropped post-mod loan-months, how many carry a termination / code 01
  D.5  (fixed 2026-10-01, picked from LOADER OUTPUT, not raw) loan_age +
       raw-vs-kept sequence verification for one never-modified (>=24 kept
       rows) and one modified (raw has a Y month, >=12 kept rows) loan

Also verifies (2026-10-01 fix): term_filter is now LOAN-level everywhere --
a loan with ANY row (including blank/NaN) whose original_loan_term !=
term_filter is dropped entirely, not row-filtered. The D.1 mixed-term-loan
count above should now have zero effect on the loader's kept/dropped totals
(every mixed-term loan is fully dropped, consistently with every other
reader) -- check the "term filter (==360, loan-level)" print line below.

NOTE: 2002Q1 is a PRE2013 vintage, so prepare_sequences_multiobs_zbc.py's
load_vintage_filtered() auto-intersects loan selection with the 1.68M-id
cell-grid sample BEFORE the term filter runs (the "HISTORICAL-ERA GATE").
That sample predates this patch. So D.1's raw counts come from a direct
pandas read of the raw file (no cell gate, no keep_ids) -- the loader's
own before/after counts are reported SEPARATELY and labeled cell-gated.
2018Q1 is modern-era: keep_ids=None passes through load_vintage_filtered
untouched, so its loader counts are NOT cell-gated and should track the
raw counts closely (modulo the pre-existing dropna(subset=FEATURE_COLS)).
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import prepare_sequences_multiobs_zbc as m

BASE = '/scratch/at7095/mortgage_prepayment'

CASES = [
    ('2002Q1', os.path.join(BASE, 'data_pre2013_raw', '2002Q1.csv'), True),
    ('2018Q1', os.path.join(BASE, 'data', 'raw', '2018Q1.csv'), False),
]

COL_LOAN, COL_TERM, COL_ORIG, COL_MOD, COL_ZBC = 1, 12, 13, 41, 43


def raw_level_check(vintage, path):
    print(f'\n===== {vintage}: RAW-LEVEL (direct read, no cell gate) =====', flush=True)
    chunks = []
    for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=[COL_LOAN, COL_TERM, COL_ORIG, COL_MOD, COL_ZBC],
            names=['loan_id', 'term', 'orig', 'mod', 'zbc'],
            dtype=str, chunksize=1_000_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df['term'] = pd.to_numeric(df['term'], errors='coerce')

    n_loanmonths_total = len(df)
    n_loans_total = df['loan_id'].nunique()
    print(f'D.1  loan-months total: {n_loanmonths_total:,} | loans total: {n_loans_total:,}')

    term_counts = df['term'].value_counts(dropna=False).sort_index()
    print('D.1  distinct original_loan_term values (loan-MONTH counts, raw, not per-loan):')
    for t, n in term_counts.items():
        label = 'blank/unparseable' if pd.isna(t) else t
        print(f'       term={label!r}: {n:,} loan-months')

    term_nunique = df.groupby('loan_id')['term'].nunique(dropna=True)
    n_multi_term_loans = int((term_nunique > 1).sum())
    print(f'D.1  loans with more than one distinct nonblank original_loan_term value: '
          f'{n_multi_term_loans:,} (expected 0 -- a loan-level attribute should not vary '
          f'across its own rows)')
    if n_multi_term_loans > 0:
        example_ids = term_nunique[term_nunique > 1].index[:5]
        print('       NOTE (fixed 2026-10-01): load_vintage_filtered() and '
              'count_prepay_events_pre2013.py both now filter at the LOAN '
              'level -- a loan with ANY row whose term != 360 (including '
              'blank/NaN) is dropped entirely, in every reader. Previously '
              'the loader filtered ROW-BY-ROW (kept only the 360-term rows '
              'of a mixed-term loan) while count_prepay_events_pre2013.py '
              'used only the first-observed term per loan; both could '
              'disagree with each other and with this raw-level count. '
              'That inconsistency is resolved now; this block is kept as a '
              'diagnostic to surface any future mixed-term loans, which '
              'will be fully dropped by every reader.')
        for lid in example_ids:
            vals = df.loc[df['loan_id'] == lid, 'term'].dropna().unique()
            print(f'       example loan {lid}: distinct terms seen = {sorted(vals)}')

    is_360 = df['term'] == 360
    n_lm_360 = int(is_360.sum())
    n_lm_non360 = int((~is_360).sum())
    loans_360 = df.loc[is_360, 'loan_id'].unique()
    loans_non360_only = set(df.loc[~is_360, 'loan_id'].unique()) - set(loans_360)
    print(f'D.1  loan-months with term==360: {n_lm_360:,} | term!=360: {n_lm_non360:,}')
    print(f'D.1  loans with ANY term==360 row: {len(set(loans_360)):,} | '
          f'loans with ONLY non-360 rows: {len(loans_non360_only):,}')

    # D.2: term filter effect on loan-months (raw level)
    df_360 = df[is_360].copy()
    print(f'D.2  loan-months before term filter: {n_loanmonths_total:,} | '
          f'after term filter (==360): {len(df_360):,}')

    # D.3: post-mod on the term-filtered population
    mod_y = df_360['mod'] == 'Y'
    n_loans_ever_y = df_360.loc[mod_y, 'loan_id'].nunique()
    first_y = df_360.loc[mod_y].groupby('loan_id').cumcount()  # placeholder, real logic below
    # Need chronological order for first-Y-onward; this reader has no
    # monthly_reporting_period column (not needed for raw ground truth
    # here), so the monotonicity check and "first Y onward" count use
    # row order within the file as a proxy for chronological order (same
    # caveat as count_prepay_events_pre2013.py's scan_file). The patched
    # production readers (which DO sort by yyyymm first) are the
    # authoritative monotonicity check -- see their own printed output
    # when load_vintage_filtered() is called below.
    df_360['_rownum'] = np.arange(len(df_360))
    y_first_rownum = df_360.loc[mod_y].groupby('loan_id')['_rownum'].min()
    df_360['_first_y_rownum'] = df_360['loan_id'].map(y_first_rownum)
    on_or_after = df_360['_first_y_rownum'].notna() & (df_360['_rownum'] >= df_360['_first_y_rownum'])
    violation = on_or_after & (df_360['mod'] == 'N')
    n_rows_dropped = int(on_or_after.sum())
    print(f'D.3  (term==360 population) loans with any Y: {n_loans_ever_y:,} | '
          f'loan-months that would be dropped by post-mod (row-order proxy): {n_rows_dropped:,} | '
          f'monotonicity violations (row-order proxy): {int(violation.sum())}')

    # D.4 denominators: total terminations in the whole 360 population, so the
    # dropped counts below can be read as shares of this file's total.
    zbc_all = df_360['zbc'].astype(str).str.strip()
    has_zbc_all = df_360['zbc'].notna() & (zbc_all != '')
    is_01_all = zbc_all == '01'
    n_zbc_all = int(has_zbc_all.sum())
    n_01_all = int(is_01_all.sum())
    print(f'D.4  360-population denominators: {n_zbc_all:,} total nonblank-zbc '
          f'terminations, {n_01_all:,} total code-01 terminations (out of '
          f'{len(df_360):,} term==360 loan-months)')

    # D.4: of the dropped loan-months, how many carry a termination / code 01
    dropped = df_360[on_or_after]
    has_zbc = dropped['zbc'].notna() & (dropped['zbc'].astype(str).str.strip() != '')
    is_01 = dropped['zbc'].astype(str).str.strip() == '01'
    n_zbc_dropped = int(has_zbc.sum())
    n_01_dropped = int(is_01.sum())
    pct_zbc = 100 * n_zbc_dropped / n_zbc_all if n_zbc_all else float('nan')
    pct_01 = 100 * n_01_dropped / n_01_all if n_01_all else float('nan')
    print(f'D.4  of {len(dropped):,} dropped post-mod loan-months: '
          f'{n_zbc_dropped:,} carry ANY nonblank zero_balance_code '
          f'({pct_zbc:.2f}% of the {n_zbc_all:,} total), '
          f'{n_01_dropped:,} carry code 01 specifically '
          f'({pct_01:.2f}% of the {n_01_all:,} total)')

    return df_360


def investigate_monotonicity_violations(vintage, path, n_examples=5):
    """Real chronological-order check (sorts by yyyymm, not file row order)
    -- same logic the patched production reader uses -- to find and print
    actual example loan sequences where 'N' follows a prior 'Y'."""
    print(f'\n===== {vintage}: MONOTONICITY VIOLATION DETAIL (chronological) =====', flush=True)
    chunks = []
    for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=[1, 2, 12, 41],
            names=['loan_id', 'month', 'term', 'mod'],
            dtype=str, chunksize=1_000_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df['term'] = pd.to_numeric(df['term'], errors='coerce')
    df = df[df['term'] == 360].copy()
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    df = df[df['month'].notna()].copy()
    df['yyyymm'] = df['month'].astype(int).apply(m.mmyyyy_to_yyyymm)
    df = df.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)

    mod_y = df['mod'] == 'Y'
    first_y_month = df.loc[mod_y].groupby('loan_id')['yyyymm'].min()
    df['_first_y_month'] = df['loan_id'].map(first_y_month)
    on_or_after = df['_first_y_month'].notna() & (df['yyyymm'] >= df['_first_y_month'])
    violation = on_or_after & (df['mod'] == 'N')
    n_viol_rows = int(violation.sum())
    n_viol_loans = df.loc[violation, 'loan_id'].nunique()
    print(f'  chronological check: {n_viol_rows:,} violating loan-months across '
          f'{n_viol_loans:,} loans (of {df["loan_id"].nunique():,} term==360 loans total)')

    example_loans = df.loc[violation, 'loan_id'].unique()[:n_examples]
    for lid in example_loans:
        seq = df[df['loan_id'] == lid][['yyyymm', 'mod']]
        print(f'  loan {lid}: {list(zip(seq["yyyymm"], seq["mod"]))}')


def loader_level_check(vintage, cutoff_ym, pmms_rates, zhvi_df):
    print(f'\n===== {vintage}: LOADER-LEVEL (load_vintage_filtered, patched) =====', flush=True)
    try:
        df = m.load_vintage_filtered(vintage, pmms_rates, zhvi_df, cutoff_ym, keep_ids=None)
    except AssertionError as e:
        print(f'  MONOTONICITY ASSERTION FAILED (reported, not silenced): {e}', flush=True)
        return None
    if df is None or df.empty:
        print('  loader returned empty/None')
        return None
    n_loans = df['loan_id'].nunique()
    print(f'  loader output: {len(df):,} loan-months, {n_loans:,} loans '
          f'(cell-gated for pre-2013 vintages, else = raw-level post-filter)')
    return df


def pick_and_verify_example_loans(vintage, path, loaded):
    """D.5, picked from the LOADER OUTPUT (not the raw frame), per the
    2026-10-01 fix: one never-modified 360-term loan with >=24 kept rows;
    one loan whose RAW rows contain a Y month and that still has >=12 kept
    rows after the post-mod drop. Prints the raw (yyyymm, mod, zbc)
    sequence and the loader's kept yyyymm + loan_age, then verifies:
      1. first kept loan_age == months-since-origination - 1, clipped at 0
      2. loan_age on every kept row matches that same formula (unchanged
         by the drop -- the drop removes rows, it does not alter loan_age)
      3. no kept row has yyyymm >= the raw sequence's first Y month
    """
    print(f'\n===== {vintage}: D.5 (loader-output example picker) =====', flush=True)
    chunks = []
    for chunk in pd.read_csv(
            path, sep='|', header=None,
            usecols=[COL_LOAN, 2, COL_TERM, COL_ORIG, COL_MOD, COL_ZBC],
            names=['loan_id', 'month', 'term', 'orig', 'mod', 'zbc'],
            dtype=str, chunksize=1_000_000, low_memory=False):
        chunks.append(chunk)
    raw = pd.concat(chunks, ignore_index=True)
    # loan_id must match the production loader's dtype (plain pd.read_csv
    # infers int64 for this column since every value parses cleanly) -- this
    # read used dtype=str for every column, so loan_id came back as strings.
    # isin() against loaded_ids (int64) would silently match nothing without
    # this cast (found 2026-10-01: both vintages returned zero candidates).
    raw['loan_id'] = pd.to_numeric(raw['loan_id'], errors='coerce')
    raw = raw[raw['loan_id'].notna()].copy()
    raw['loan_id'] = raw['loan_id'].astype(np.int64)
    raw['term'] = pd.to_numeric(raw['term'], errors='coerce')
    raw = raw[raw['term'] == 360].copy()
    raw['month_i'] = pd.to_numeric(raw['month'], errors='coerce')
    raw = raw[raw['month_i'].notna()].copy()
    raw['yyyymm'] = raw['month_i'].astype(int).apply(m.mmyyyy_to_yyyymm)
    raw = raw.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)

    if loaded is None or loaded.empty:
        print('  loader output empty/None -- cannot pick examples', flush=True)
        return

    loaded_ids = set(loaded['loan_id'].unique())
    raw_in_loaded = raw[raw['loan_id'].isin(loaded_ids)]
    ever_y = raw_in_loaded.assign(_y=(raw_in_loaded['mod'] == 'Y')).groupby('loan_id')['_y'].any()
    kept_counts = loaded.groupby('loan_id').size()

    never_mod_ids = [lid for lid in ever_y[~ever_y].index if kept_counts.get(lid, 0) >= 24]
    mod_ids = [lid for lid in ever_y[ever_y].index if kept_counts.get(lid, 0) >= 12]
    never_mod_loan = never_mod_ids[0] if never_mod_ids else None
    mod_loan = mod_ids[0] if mod_ids else None

    print(f'D.5  {vintage}: picked from LOADER OUTPUT -- never-mod (no Y, >=24 kept '
          f'rows): {never_mod_loan!r} | modified (raw has Y, >=12 kept rows): '
          f'{mod_loan!r}', flush=True)

    for label, lid, min_rows in [('never-mod', never_mod_loan, 24), ('modified', mod_loan, 12)]:
        if lid is None:
            print(f'  {label}: no candidate found meeting criteria (>= {min_rows} kept rows)',
                  flush=True)
            continue

        raw_seq = raw[raw['loan_id'] == lid][['yyyymm', 'mod', 'zbc']]
        print(f'\n  {label} loan {lid}:', flush=True)
        print(f'    RAW (yyyymm, mod, zbc) sequence, {len(raw_seq)} rows:', flush=True)
        print('      ' + ', '.join(
            f"({int(r.yyyymm)},{r.mod!r},{r.zbc!r})" for r in raw_seq.itertuples()), flush=True)

        sub = loaded[loaded['loan_id'] == lid].sort_values('yyyymm')
        kept_yyyymm = sub['yyyymm'].tolist()
        kept_loan_age = sub['loan_age_months'].tolist()
        print(f'    LOADER kept yyyymm ({len(kept_yyyymm)} rows): {kept_yyyymm}', flush=True)
        print(f'    LOADER loan_age_months: {kept_loan_age}', flush=True)

        orig_mmyyyy = pd.to_numeric(raw.loc[raw['loan_id'] == lid, 'orig'], errors='coerce').iloc[0]
        orig_yyyymm = m.mmyyyy_to_yyyymm(int(orig_mmyyyy))

        def _expected_age(ym):
            return max((ym // 100 - orig_yyyymm // 100) * 12
                       + (ym % 100 - orig_yyyymm % 100) - 1, 0)

        first_ym = kept_yyyymm[0]
        expected_first_age = _expected_age(first_ym)
        match = (kept_loan_age[0] == expected_first_age)
        print(f'    CHECK 1 (first kept loan_age): origination={orig_yyyymm}, first kept '
              f'yyyymm={first_ym}, expected={expected_first_age}, actual={kept_loan_age[0]} '
              f'-> {"MATCH" if match else "MISMATCH"}', flush=True)

        recomputed = [_expected_age(ym) for ym in kept_yyyymm]
        all_match = all(a == b for a, b in zip(recomputed, kept_loan_age))
        print(f'    CHECK 2 (loan_age unchanged by drop, all kept rows): '
              f'{"MATCH" if all_match else "MISMATCH"}', flush=True)
        if not all_match:
            print(f'      recomputed: {recomputed}', flush=True)

        y_rows = raw_seq.loc[raw_seq['mod'] == 'Y', 'yyyymm']
        if not y_rows.empty:
            first_y = int(y_rows.min())
            survivors = [ym for ym in kept_yyyymm if ym >= first_y]
            print(f'    CHECK 3 (no kept row >= first Y month {first_y}): '
                  f'{"PASS" if not survivors else f"FAIL -- survivors: {survivors}"}',
                  flush=True)
        else:
            print('    CHECK 3: no Y month in raw sequence -- never-mod candidate, N/A',
                  flush=True)


def main():
    pmms_rates = m.load_pmms()
    zhvi_df = m.load_zhvi()

    for vintage, path, is_pre2013 in CASES:
        raw_level_check(vintage, path)
        investigate_monotonicity_violations(vintage, path)
        cutoff_ym = 203012  # unbounded-ish, keep everything through this vintage's data
        loaded = loader_level_check(vintage, cutoff_ym, pmms_rates, zhvi_df)

        pick_and_verify_example_loans(vintage, path, loaded)


if __name__ == '__main__':
    main()
