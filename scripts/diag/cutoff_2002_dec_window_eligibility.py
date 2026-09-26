"""
cutoff_2002_dec_window_eligibility.py -- diagnostic (read-only, writes no
sequence files) answering: for the cutoff_2002 test-split loans, restricted
to those active in CY2003, does the Dec-2002 row survive select_observations()'s
own eligibility rules (_eligible_candidates in prepare_sequences_multiobs_zbc.py),
when the panel is truncated at yyyymm<=200212 exactly as the actual cutoff_2002
training build truncated it (load_vintage_filtered(cutoff_yyyymm=200212))?

Reports, per exclusion rule, how many test loans it removes -- and separately
whether the loan has a raw Dec-2002 row at all and whether it has ANY CY2003
row (read from the SAME 12 historical vintage files, unbounded by the 200212
cutoff, restricted to the test loan ids).

Usage: python scripts/diag/cutoff_2002_dec_window_eligibility.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_sequences_multiobs_zbc import (
    load_vintage_filtered, load_pmms, load_zhvi, _prepare_panel,
    _eligible_candidates, PRE2013_VINTAGES, _vintage_quarter_start_yyyymm,
    dec_yyyymm, BASE,
)

CUTOFF_YEAR = 2002
CUTOFF_YM   = dec_yyyymm(CUTOFF_YEAR)          # 200212
H           = 1                                 # matches h1 in the frozen build's dir name
MIN_HIST    = 1                                  # default, not overridden by the training sbatch
SEQ_DIR     = os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist')
UNBOUNDED_YM = 209912                            # effectively no truncation, for the CY2003-activity check


def main():
    test_ids = np.load(os.path.join(SEQ_DIR, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())
    print(f'Test split loans: {len(test_ids_set):,}', flush=True)

    relevant = [v for v in PRE2013_VINTAGES if _vintage_quarter_start_yyyymm(v) <= CUTOFF_YM]
    print(f'Relevant historical vintages (acquisition <= Dec 2002): {relevant}', flush=True)

    pmms_rates = load_pmms()
    zhvi_df = load_zhvi()

    frames = []
    for v in relevant:
        df = load_vintage_filtered(v, pmms_rates, zhvi_df, UNBOUNDED_YM, keep_ids=test_ids_set)
        if df is not None and not df.empty:
            frames.append(df)
        print(f'  {v}: {"empty" if df is None or df.empty else len(df)} rows', flush=True)
    full_df = pd.concat(frames, ignore_index=True)
    del frames
    print(f'Full (unbounded) panel for test loans: {len(full_df):,} rows, '
          f'{full_df["loan_id"].nunique():,} distinct loan_ids', flush=True)

    found_ids = set(full_df['loan_id'].unique().tolist())
    missing_from_panel = test_ids_set - found_ids
    print(f'Test loans with ZERO rows in the unbounded panel (dropna/cell-sample/keep_ids '
          f'mismatch): {len(missing_from_panel):,}', flush=True)

    # ── CY2003 activity: any row with 200301 <= yyyymm <= 200312 in the UNBOUNDED panel ──
    cy2003_ids = set(full_df.loc[(full_df['yyyymm'] >= 200301) & (full_df['yyyymm'] <= 200312),
                                  'loan_id'].unique().tolist())
    print(f'Test loans with >=1 row in CY2003 ("active in CY2003"): {len(cy2003_ids):,}', flush=True)

    # ── Dec-2002 row presence (raw, from the unbounded panel) ──────────────────
    dec2002_ids = set(full_df.loc[full_df['yyyymm'] == CUTOFF_YM, 'loan_id'].unique().tolist())
    print(f'Test loans with a raw Dec-2002 (yyyymm=={CUTOFF_YM}) row: {len(dec2002_ids):,}', flush=True)

    target = cy2003_ids & dec2002_ids
    no_dec_row_but_cy2003 = cy2003_ids - dec2002_ids
    print(f'Active-in-CY2003 loans WITHOUT a Dec-2002 row (excluded per your instruction, '
          f'not given an earlier window): {len(no_dec_row_but_cy2003):,}', flush=True)
    print(f'Target population (active in CY2003 AND has a Dec-2002 row): {len(target):,}', flush=True)

    # ── Truncated-at-200212 panel, exactly matching the training build's own
    # load_vintage_filtered(cutoff_yyyymm=200212) truncation ──────────────────
    trunc_df = full_df[full_df['yyyymm'] <= CUTOFF_YM].copy()
    panel = _prepare_panel(trunc_df)

    # Every target loan's Dec-2002 row is, by construction of the truncation,
    # the row with max yyyymm for that loan (== the last row_idx == L-1),
    # UNLESS dropna(subset=FEATURE_COLS) inside load_vintage_filtered already
    # removed that specific row for missing features -- check that directly.
    dec_rows = panel[(panel['loan_id'].isin(target)) & (panel['yyyymm'] == CUTOFF_YM)]
    ids_with_dec_row_post_dropna = set(dec_rows['loan_id'].unique().tolist())
    dropped_by_feature_dropna = target - ids_with_dec_row_post_dropna
    print(f'\nOf the {len(target):,}-loan target population, '
          f'{len(dropped_by_feature_dropna):,} lose their Dec-2002 row to '
          f'dropna(subset=FEATURE_COLS) inside load_vintage_filtered (missing a feature '
          f'value that month, e.g. DTI) -- excluded, not given an earlier window.', flush=True)

    dec_rows = dec_rows.copy()
    is_last_row = dec_rows['row_idx'] == (dec_rows['L'] - 1)
    print(f'Of the loans that keep their Dec-2002 row, {int(is_last_row.sum()):,}/{len(dec_rows):,} '
          f'have it as row_idx == L-1 (the last row of the truncated panel), confirming the '
          f'structural point below.', flush=True)

    # ── Apply _eligible_candidates()'s exact rule set to this truncated panel,
    # and check membership of the Dec-2002 row in the surviving frame ─────────
    elig = _eligible_candidates(trunc_df, H, MIN_HIST)
    elig_key = set(zip(elig['loan_id'], elig['row_idx']))

    dec_row_keys = list(zip(dec_rows['loan_id'], dec_rows['row_idx']))
    survives_all_rules = sum(1 for k in dec_row_keys if k in elig_key)
    print(f'\nOf {len(dec_row_keys):,} loans with a (post-dropna) Dec-2002 row, '
          f'{survives_all_rules:,} survive _eligible_candidates() applying ALL of: '
          f'row_idx>=min_hist-1, row_idx<=L-1-H, row_idx<term_t, calendar-gap filter, '
          f'label-window filter.', flush=True)

    # ── Break out each rule individually on the Dec-2002 rows to show WHERE the
    # exclusions come from (min_hist bound is vacuous here since row_idx=L-1>=0
    # for any loan with min_hist<=1; the informative ones are L-1-H and term_t) ──
    rule_min_hist = dec_rows['row_idx'] >= (MIN_HIST - 1)
    rule_L1H      = dec_rows['row_idx'] <= (dec_rows['L'] - 1 - H)
    rule_term_t   = dec_rows['row_idx'] <  dec_rows['term_t']
    n = len(dec_rows)
    print(f'\nPer-rule pass/fail on the {n:,} post-dropna Dec-2002 rows:')
    print(f'  row_idx >= min_hist-1  : passes {int(rule_min_hist.sum()):,} / fails {int((~rule_min_hist).sum()):,}')
    print(f'  row_idx <= L-1-H       : passes {int(rule_L1H.sum()):,} / fails {int((~rule_L1H).sum()):,}')
    print(f'  row_idx <  term_t      : passes {int(rule_term_t.sum()):,} / fails {int((~rule_term_t).sum()):,}')
    print('\n(The calendar-gap and label-window filters are computed inside '
          '_eligible_candidates on the full frame, not decomposed per-row above; '
          'their aggregate effect is captured by the survives_all_rules count.)')


if __name__ == '__main__':
    main()
