"""
score_rolling_one_step.py -- one-step-ahead rolling forecast test (advisor's
Aug 30 design). For each reference month t from Dec(cutoff_year) through
Nov(forecast_year) (12 months), one right-aligned window per eligible loan
ending at t, predicting t+1. Predicted events = sum(h) over loan-months;
realized events = loans with zero_balance_code_actual==01 at t+1.

REUSES THE EXISTING CACHED COMBINED PASS -- no new raw scan. The cache
(built by score_multiobs_dec_window.py for the same cutoff_year/map_era/
population) is already truncated at Dec(forecast_year), one month beyond
the latest t used here (Nov(forecast_year)), so every t's eligibility check
already has a genuine future row to test against -- see
score_multiobs_dec_window.py's module docstring for the full argument
(term_t/L-1-H hold for the same structural reason at every t, not just the
single Dec-cutoff reference month that design was built for).

ELIGIBILITY, PER MONTH t: _eligible_candidates() is computed ONCE on the
whole combined pass (not per month -- it's t-independent by construction:
row_idx/L/term_t/monthidx/cumgap don't depend on which t we're asking
about), then filtered to yyyymm==t per iteration. This guarantees identical
eligibility logic (min_hist, calendar-gap, term_t) at every t, and confirms
by construction that features at t never read rows after t (build_sequences_
multiobs's gather is backward-only from t's row_idx regardless of what the
passed-in df contains after that position).

FORWARD-ADJACENCY FILTER (addition B): a loan-month at t only counts (in
BOTH predicted and realized) if the loan's immediately-following panel row
is EXACTLY calendar month t+1 -- computed via next_monthidx = monthidx+1,
using a groupby('loan_id').shift(-1) on the (already sorted) panel, not
assumed. Excluded count is reported per month, not silently absorbed.

CONSISTENCY CHECK (addition A): the t=Dec(cutoff_year) month must reproduce
the frozen single-window run's h values (dec_window_scores_seed*.csv)
exactly (max abs diff < 1e-6) for loans in both populations -- the two
populations need not be IDENTICAL (the frozen run additionally required
"active anywhere in the forecast year"; this run additionally requires
forward-adjacency at t+1), but shared loans must score identically, since
both paths do the same right-aligned gather off the same cached df.

Usage:
    python scripts/score_rolling_one_step.py \\
        --cutoff_year 2002 --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist \\
        --ckpt_path outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist/hazard_best.pt \\
        --map_era fixed --include_pre2013 --seed_label seed42 \\
        --frozen_scores outputs/rolling/dec_window_cutoff_2002_seed42/dec_window_scores_seed42.csv \\
        --out_dir outputs/rolling/rolling_onestep_cutoff_2002_seed42
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_multiobs_dec_window import (
    build_combined_pass, load_checkpoint, verify_checkpoint, verify_windows,
    score, BASE,
)
from prepare_sequences_multiobs_zbc import (
    _prepare_panel, _eligible_candidates, build_sequences_multiobs, dec_yyyymm,
    FEATURE_COLS, PRE2013_CELL_SAMPLE_PATH,
)

H = 1
MIN_HIST = 1
INCENTIVE_EDGES = [-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
MIN_N = 1000   # loan-months per bucket, for dispersion/incentive-slope qualification
CONSISTENCY_TOL = 1e-6


def reference_months(cutoff_year: int) -> list[int]:
    fy = cutoff_year + 1
    return [dec_yyyymm(cutoff_year)] + [fy * 100 + m for m in range(1, 12)]


def build_rolling_eligible(full_df: pd.DataFrame):
    """ONE eligibility + forward-adjacency computation over the whole
    combined pass, independent of which t is being asked about."""
    elig = _eligible_candidates(full_df, H, MIN_HIST)
    panel = _prepare_panel(full_df).sort_values(['loan_id', 'row_idx']).reset_index(drop=True)
    g = panel.groupby('loan_id')
    panel['next_monthidx'] = g['monthidx'].shift(-1)
    panel['next_zbc'] = g['zero_balance_code_actual'].shift(-1)
    elig = elig.merge(panel[['loan_id', 'row_idx', 'next_monthidx', 'next_zbc']],
                       on=['loan_id', 'row_idx'], how='left')
    n_no_next_row = int(elig['next_monthidx'].isna().sum())
    assert n_no_next_row == 0, (
        f'{n_no_next_row} eligible rows have no next panel row at all -- this should be '
        f'structurally impossible given row_idx<term_t<=L-1 guarantees row_idx+1<=L-1. '
        f'STOP, investigate before trusting the forward-adjacency filter.')
    elig['forward_adjacent'] = elig['next_monthidx'] == (elig['monthidx'] + 1)
    return elig


def month_result(full_df, elig, scaler, ref_ym, ckpt_model, batch_size, out_dir):
    month_path = os.path.join(out_dir, f'rolling_month_{ref_ym}.csv')
    if os.path.exists(month_path):
        print(f'  {ref_ym}: checkpoint hit ({month_path})', flush=True)
        return pd.read_csv(month_path)

    month_rows_all = elig[elig['yyyymm'] == ref_ym]
    n_before = len(month_rows_all)
    month_rows = month_rows_all[month_rows_all['forward_adjacent']].copy()
    n_excluded_gap = n_before - len(month_rows)

    obs = pd.DataFrame({
        'loan_id':          month_rows['loan_id'].to_numpy(),
        't':                month_rows['row_idx'].astype(int).to_numpy(),
        'ref_month':        month_rows['yyyymm'].astype(int).to_numpy(),
        'label':            np.zeros(len(month_rows), dtype=np.float32),
        'is_terminal':      np.zeros(len(month_rows), dtype=bool),
        'incl_prob':        np.ones(len(month_rows), dtype=np.float32),
        'n_eligible':       np.ones(len(month_rows), dtype=np.int32),
        'k_actual':         np.ones(len(month_rows), dtype=np.int32),
        'age_at_ref':       month_rows['loan_age_months'].astype(float).to_numpy(),
        'incentive_at_ref': month_rows['refi_incentive'].astype(float).to_numpy(),
    })

    sequences, masks, _l, _p, loan_ids_out, _extras = build_sequences_multiobs(
        full_df, scaler, k_draws=1, H=H, min_hist=MIN_HIST, obs=obs)
    verify_windows(sequences, masks, obs, month_rows, scaler)
    h_vals = score(ckpt_model, sequences, masks, batch_size)

    mr = month_rows.set_index('loan_id')
    realized_event = (mr.loc[loan_ids_out, 'next_zbc'] == 1.0).to_numpy()
    non_prepay_term = (mr.loc[loan_ids_out, 'next_zbc'].notna() &
                        (mr.loc[loan_ids_out, 'next_zbc'] != 1.0)).to_numpy()
    out = pd.DataFrame({
        'loan_id':            loan_ids_out,
        'ref_month':          ref_ym,
        'h':                  h_vals,
        'note_rate':          mr.loc[loan_ids_out, 'original_interest_rate'].to_numpy(),
        'incentive_at_ref':   mr.loc[loan_ids_out, 'refi_incentive'].to_numpy(),
        'current_actual_upb': mr.loc[loan_ids_out, 'current_actual_upb'].to_numpy(),
        'realized_event':     realized_event.astype(int),
        'non_prepay_term':    non_prepay_term.astype(int),
        'n_excluded_by_forward_gap': n_excluded_gap,   # constant per row -- survives the CSV round-trip on resume
    })
    out.to_csv(month_path, index=False)
    print(f'  {ref_ym}: n_eligible_before_gap_filter={n_before:,}  excluded_by_forward_gap={n_excluded_gap:,}  '
          f'scored={len(out):,}', flush=True)
    return out


def consistency_check(rolling_dec_month: pd.DataFrame, frozen_scores_path: str, ref_ym: int):
    frozen = pd.read_csv(frozen_scores_path)
    merged = rolling_dec_month.merge(frozen[['loan_id', 'h_t']], on='loan_id', how='inner')
    n_rolling, n_frozen, n_common = len(rolling_dec_month), len(frozen), len(merged)
    max_abs_diff = float((merged['h'] - merged['h_t']).abs().max()) if n_common else float('nan')
    print(f'\nCONSISTENCY CHECK (t={ref_ym} vs. {frozen_scores_path}):', flush=True)
    print(f'  rolling t={ref_ym} population: {n_rolling:,}  frozen population: {n_frozen:,}  '
          f'shared: {n_common:,}', flush=True)
    print(f'  max abs diff on shared loans: {max_abs_diff:.3e}', flush=True)
    assert n_common > 0, 'Zero loans in common between the rolling Dec-month and the frozen run -- STOP.'
    assert max_abs_diff < CONSISTENCY_TOL, (
        f'Consistency check FAILED: max abs diff {max_abs_diff:.3e} >= {CONSISTENCY_TOL:.0e} on '
        f'{n_common:,} shared loans -- the rolling and frozen paths do not agree. STOP, do not '
        f'trust any rolling-run number until this is resolved.')
    print(f'  PASSED (< {CONSISTENCY_TOL:.0e}).', flush=True)


def rate_table(df: pd.DataFrame, group_col: str, weight_col: str | None, min_n: int) -> pd.DataFrame:
    d = df.copy()
    if weight_col is not None:
        d = d[d[weight_col].notna() & (d[weight_col] != 0)]
        w = d[weight_col]
    else:
        w = pd.Series(1.0, index=d.index)
    rows = []
    for key, g in d.groupby(group_col, observed=True):
        wg = w.loc[g.index]
        n = wg.sum()
        pred_rate = float((g['h'] * wg).sum() / n)
        real_rate = float((g['realized_event'] * wg).sum() / n)
        rows.append({
            group_col: key, 'n_loan_months': len(g), 'weighted_n': n,
            'predicted_rate_monthly': pred_rate, 'realized_rate_monthly': real_rate,
            'predicted_rate_annualized': 1 - (1 - pred_rate) ** 12,
            'realized_rate_annualized': 1 - (1 - real_rate) ** 12,
        })
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def dispersion_stats(table: pd.DataFrame, rate_col_pred: str, rate_col_real: str, min_n_col: str, min_n: int) -> dict:
    f = table[table[min_n_col] >= min_n]
    pred_disp = f[rate_col_pred].max() / f[rate_col_pred].min()
    real_disp = f[rate_col_real].max() / f[rate_col_real].min()
    f = f.copy()
    f['ratio'] = f[rate_col_pred] / f[rate_col_real]
    return {
        'PRIMARY_predicted_dispersion_max_over_min': pred_disp,
        'PRIMARY_realized_dispersion_max_over_min': real_disp,
        'PRIMARY_dispersion_ratio': pred_disp / real_disp,
        'SECONDARY_ratio_max': float(f['ratio'].max()),
        'SECONDARY_ratio_min': float(f['ratio'].min()),
        'n_groups': len(f),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff_year', type=int, required=True)
    ap.add_argument('--seq_dir', type=str, required=True)
    ap.add_argument('--ckpt_path', type=str, required=True)
    ap.add_argument('--map_era', choices=['fixed', 'prefix'], required=True)
    ap.add_argument('--include_pre2013', action='store_true')
    ap.add_argument('--seed_label', type=str, required=True)
    ap.add_argument('--frozen_scores', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default=None)
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--cell_sample', type=str, default=PRE2013_CELL_SAMPLE_PATH,
                     help='loan_id CSV gating the historical-era (PRE2013_VINTAGES) population, '
                          'threaded to build_combined_pass the same way '
                          'prepare_sequences_multiobs_zbc.py --cell_sample is. Default is the '
                          'original pre-30y-filter sample, for backward compatibility; pass e.g. '
                          'outputs/pre2013_cell_sample_30y_loans.csv when --seq_dir is a _30y build, '
                          'so test_ids_set (drawn from that build) is not silently intersected '
                          'against the wrong cell sample.')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(BASE, 'outputs/rolling/_dec_window_raw_cache')

    with open(os.path.join(args.seq_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    test_ids = np.load(os.path.join(args.seq_dir, 'test_loan_ids_split.npy'), allow_pickle=True)
    test_ids_set = set(test_ids.tolist())

    full_df = build_combined_pass(args.cutoff_year, args.include_pre2013, args.map_era,
                                   test_ids_set, cache_dir, cell_sample_path=args.cell_sample)
    elig = build_rolling_eligible(full_df)

    model = load_checkpoint(args.ckpt_path)
    results_json_path = os.path.join(os.path.dirname(args.ckpt_path), 'results.json')
    ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
    verify_checkpoint(model, args.seq_dir, results_json_path, ckpt_dict, args.batch_size)

    ref_months = reference_months(args.cutoff_year)
    dec_ym = dec_yyyymm(args.cutoff_year)
    monthly_rows = []
    all_months = []
    for ref_ym in ref_months:
        m = month_result(full_df, elig, scaler, ref_ym, model, args.batch_size, args.out_dir)
        all_months.append(m)
        monthly_rows.append({
            'ref_month': ref_ym, 'n_active': len(m),
            'predicted_events': m['h'].sum(), 'realized_events': m['realized_event'].sum(),
            'predicted_rate_monthly': m['h'].mean(), 'realized_rate_monthly': m['realized_event'].mean(),
            'non_prepay_term_count': m['non_prepay_term'].sum(),
            'n_excluded_by_forward_gap': int(m['n_excluded_by_forward_gap'].iloc[0]),
        })
        if ref_ym == dec_ym:
            dec_month_df = m

    consistency_check(dec_month_df, args.frozen_scores, dec_ym)

    full = pd.concat(all_months, ignore_index=True)
    full.to_csv(os.path.join(args.out_dir, 'rolling_all_months.csv'), index=False)
    full['coupon'] = ((full['note_rate'] - 0.5) * 2).round() / 2

    print(f'\n{"=" * 100}\n(a) Predicted vs realized, by coupon, summed over the year -- COUNT-weighted\n{"=" * 100}')
    a_count = rate_table(full, 'coupon', None, MIN_N)
    print(a_count.to_string(index=False))
    n_upb_bad = int((full['current_actual_upb'].isna() | (full['current_actual_upb'] == 0)).sum())
    print(f'\n(a) UPB-weighted  [UPB excluded: {n_upb_bad:,}/{len(full):,} '
          f'({100 * n_upb_bad / len(full):.2f}%) blank/zero, excluded not imputed]')
    a_upb = rate_table(full, 'coupon', 'current_actual_upb', MIN_N)
    print(a_upb.to_string(index=False))

    print(f'\n{"=" * 100}\n(b) Dispersion (computed on MONTHLY rates; annualized shown in (a) for readability) '
          f'-- COUNT-weighted\n{"=" * 100}')
    print(dispersion_stats(a_count, 'predicted_rate_monthly', 'realized_rate_monthly', 'n_loan_months', MIN_N))
    print('\n(b) UPB-weighted')
    print(dispersion_stats(a_upb, 'predicted_rate_monthly', 'realized_rate_monthly', 'n_loan_months', MIN_N))

    print(f'\n{"=" * 100}\n(c) Incentive-bin slope, contemporaneous incentive_at_ref per loan-month\n{"=" * 100}')
    full['incentive_bin'] = pd.cut(full['incentive_at_ref'], bins=INCENTIVE_EDGES)
    n_outside = int(full['incentive_bin'].isna().sum())
    c_tab = rate_table(full.dropna(subset=['incentive_bin']), 'incentive_bin', None, MIN_N)
    print(c_tab.to_string(index=False))
    print(f'  values outside [-6,6]: {n_outside:,} (excluded, counted not silent)')
    qualifying = c_tab[c_tab['n_loan_months'] >= MIN_N]
    if len(qualifying) >= 2:
        lo, hi = qualifying.iloc[0], qualifying.iloc[-1]
        print(f'  slope: predicted {hi["predicted_rate_monthly"] / lo["predicted_rate_monthly"]:.4f}  '
              f'realized {hi["realized_rate_monthly"] / lo["realized_rate_monthly"]:.4f}')

    print(f'\n{"=" * 100}\n(d) Pooled predicted/realized -- COUNT-weighted and UPB-weighted\n{"=" * 100}')
    for label, wcol in [('count', None), ('upb', 'current_actual_upb')]:
        d = full if wcol is None else full[full[wcol].notna() & (full[wcol] != 0)]
        w = pd.Series(1.0, index=d.index) if wcol is None else d[wcol]
        pred = float((d['h'] * w).sum() / w.sum())
        real = float((d['realized_event'] * w).sum() / w.sum())
        print(f'  {label}: predicted_monthly={pred:.5f}  realized_monthly={real:.5f}  '
              f'predicted_annualized={1 - (1 - pred) ** 12:.4f}  realized_annualized={1 - (1 - real) ** 12:.4f}')
    print(f'  upb_n_excluded_blank_or_zero: {n_upb_bad:,}')

    print(f'\n{"=" * 100}\n(e) Month-by-month pooled predicted vs realized (the rate path)\n{"=" * 100}')
    e_tab = pd.DataFrame(monthly_rows)
    e_tab['predicted_rate_annualized'] = 1 - (1 - e_tab['predicted_rate_monthly']) ** 12
    e_tab['realized_rate_annualized'] = 1 - (1 - e_tab['realized_rate_monthly']) ** 12
    print(e_tab.to_string(index=False))
    e_tab.to_csv(os.path.join(args.out_dir, 'monthly_pooled.csv'), index=False)

    print('\nDone.')


if __name__ == '__main__':
    main()
