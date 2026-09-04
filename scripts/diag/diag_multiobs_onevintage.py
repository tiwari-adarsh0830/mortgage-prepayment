"""diag_multiobs_onevintage.py — ONE-vintage sanity check on the multiobs
sampler before any full run.

Loads a single vintage (2015Q1) through prepare_sequences_multiobs_zbc's own
load_vintage_filtered() at cutoff_year=2020, runs select_observations() at
k=5, H=1, uniform, and reports raw diagnostics -- not just summary stats:

  - total observations, unique loans, observations-per-loan distribution
  - sampled label rate, fraction of observations that are terminal draws
  - age_at_ref min/median/max split by terminal vs non-terminal
  - CALENDAR CONTIGUITY: row_idx is a post-dropna position index, not a
    calendar-month index. If load_vintage_filtered's dropna(subset=
    FEATURE_COLS) removes an interior row (e.g. one month with a NaN DTI),
    row_idx i and i+1 for that loan are no longer 1 calendar month apart.
    A drawn window [s, t] is "affected" if any such gap falls strictly
    inside it. Reports affected-loan / affected-observation counts and
    prints two concrete yyyymm sequences showing the divergence.
  - whether term_t - H lands exactly H calendar months before the event
    month (it must, for the terminal draw's label window to be correct);
    reports the count where it does not, driven by the same gap mechanism.

This reads real cluster data (data/raw/2015Q1.csv) via the production
loader, which is why it runs as a batch job rather than on the login node.
It does NOT call build_sequences_multiobs, does NOT fit/save a scaler, and
does NOT write anything under data/sequences_rolling/ -- diagnostic only.
"""
import sys
sys.path.insert(0, 'scripts')

import numpy as np
import pandas as pd

import prepare_sequences_multiobs_zbc as m

VINTAGE     = '2015Q1'
CUTOFF_YEAR = 2020
K_DRAWS, H, MIN_HIST = 5, 1, 1

pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 20)


def monthidx(yyyymm) -> pd.Series:
    yyyymm = pd.Series(yyyymm)
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def main():
    cutoff_ym = m.dec_yyyymm(CUTOFF_YEAR)
    print(f'Loading vintage {VINTAGE} at cutoff_year={CUTOFF_YEAR} (YYYYMM={cutoff_ym})', flush=True)

    pmms_rates = m.load_pmms()
    zhvi_df    = m.load_zhvi()
    df = m.load_vintage_filtered(VINTAGE, pmms_rates, zhvi_df, cutoff_ym, keep_ids=None)
    if df is None or df.empty:
        print('No rows loaded for this vintage/cutoff -- nothing to diagnose.')
        return

    print(f'\nLoaded {len(df):,} rows, {df["loan_id"].nunique():,} loans '
          f'(post-dropna on FEATURE_COLS)', flush=True)

    # ── Panel with row_idx / calendar-gap bookkeeping ──────────────────────
    panel = df.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
    panel['row_idx']       = panel.groupby('loan_id').cumcount()
    panel['monthidx']      = monthidx(panel['yyyymm'])
    panel['prev_monthidx'] = panel.groupby('loan_id')['monthidx'].shift(1)
    panel['gap_flag']      = ((panel['monthidx'] - panel['prev_monthidx']) > 1).fillna(False)
    panel['cumgap']        = panel.groupby('loan_id')['gap_flag'].cumsum()

    n_gap_loans_total = panel.groupby('loan_id')['gap_flag'].any().sum()
    print(f'Loans with >=1 interior calendar gap ANYWHERE in their panel '
          f'(post-dropna): {n_gap_loans_total:,} / {panel["loan_id"].nunique():,}', flush=True)

    # ── Sample observations ─────────────────────────────────────────────────
    obs = m.select_observations(df, K_DRAWS, H, MIN_HIST, draw_scheme='uniform')
    print(f'\n=== select_observations(k={K_DRAWS}, H={H}, min_hist={MIN_HIST}, uniform) ===')
    print(f'Total observations: {len(obs):,}')
    print(f'Unique loans represented: {obs["loan_id"].nunique():,}')

    print('\n-- Observations-per-loan distribution --')
    per_loan_counts = obs.groupby('loan_id').size()
    print(per_loan_counts.value_counts().sort_index().to_string())

    print(f'\nSampled label rate: {obs["label"].mean():.6f}  '
          f'({int(obs["label"].sum()):,} positive / {len(obs):,})')
    print(f'Fraction terminal draws: {obs["is_terminal"].mean():.6f}  '
          f'({int(obs["is_terminal"].sum()):,} / {len(obs):,})')

    print('\n-- age_at_ref by terminal vs non-terminal --')
    print(obs.groupby('is_terminal')['age_at_ref'].agg(['count', 'min', 'median', 'max']).to_string())

    # ── Calendar contiguity WITHIN each drawn window ────────────────────────
    cumgap_lookup = panel.set_index(['loan_id', 'row_idx'])['cumgap']

    obs = obs.copy()
    obs['s'] = np.maximum(0, obs['t'] - m.MAX_SEQ_LEN + 1)
    obs['cumgap_t'] = cumgap_lookup.reindex(list(zip(obs['loan_id'], obs['t']))).to_numpy()
    obs['cumgap_s'] = cumgap_lookup.reindex(list(zip(obs['loan_id'], obs['s']))).to_numpy()
    obs['n_gaps_in_window'] = obs['cumgap_t'] - obs['cumgap_s']
    obs['window_has_gap']   = obs['n_gaps_in_window'] > 0

    n_affected_obs   = int(obs['window_has_gap'].sum())
    n_affected_loans = obs.loc[obs['window_has_gap'], 'loan_id'].nunique()
    print(f'\n=== Calendar contiguity of DRAWN windows ===')
    print(f'Observations whose feature window spans a calendar gap: '
          f'{n_affected_obs:,} / {len(obs):,}')
    print(f'Distinct loans contributing at least one such observation: '
          f'{n_affected_loans:,} / {obs["loan_id"].nunique():,}')

    print('\n-- Two concrete examples (full yyyymm sequence for the loan) --')
    example_loans = obs.loc[obs['window_has_gap'], 'loan_id'].unique()[:2]
    if len(example_loans) == 0:
        print('(none found in this vintage/cutoff)')
    for lid in example_loans:
        loan_rows = panel[panel['loan_id'] == lid].sort_values('row_idx')
        loan_obs  = obs[(obs['loan_id'] == lid) & (obs['window_has_gap'])]
        print(f'\nloan_id={lid}')
        print('  row_idx -> yyyymm:', list(zip(loan_rows['row_idx'], loan_rows['yyyymm'])))
        print('  gap_flag rows (row_idx where a gap precedes it):',
              loan_rows.loc[loan_rows['gap_flag'], 'row_idx'].tolist())
        print('  affected observation(s) [t, s, ref_month, n_gaps_in_window]:')
        print(loan_obs[['t', 's', 'ref_month', 'n_gaps_in_window']].to_string(index=False))

    # ── term_t - H exactly H calendar months before the event month? ───────
    monthidx_lookup = panel.set_index(['loan_id', 'row_idx'])['monthidx']
    term_rows = obs[obs['is_terminal']].copy()
    term_rows['event_row_idx']  = term_rows['t'] + H
    term_rows['event_monthidx'] = monthidx_lookup.reindex(
        list(zip(term_rows['loan_id'], term_rows['event_row_idx']))).to_numpy()
    term_rows['ref_monthidx']   = monthidx(term_rows['ref_month'])
    term_rows['month_gap']      = term_rows['event_monthidx'] - term_rows['ref_monthidx']

    mismatch = term_rows[term_rows['month_gap'] != H]
    print(f'\n=== term_t - H vs event month (terminal draws only) ===')
    print(f'Terminal draws checked: {len(term_rows):,}')
    print(f'Terminal draws where (event_month - ref_month) != H={H}: '
          f'{len(mismatch):,} / {len(term_rows):,}')
    if len(mismatch) > 0:
        print('\n-- Up to 5 mismatch examples --')
        print(mismatch[['loan_id', 't', 'ref_month', 'event_row_idx',
                         'month_gap']].head(5).to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
