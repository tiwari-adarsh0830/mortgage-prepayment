"""census_panel_baseline.py -- census-panel baseline for a given cutoff.

Enumerates the FULL eligible (loan_id, ref_month) candidate population --
NO sampling -- using the exact SAME eligibility definition
select_observations() draws its k (or, later, fixed-fraction) sample from.
That definition lives in prepare_sequences_multiobs_zbc.py's
_eligible_candidates() (panel prep + the row_idx bounds + both calendar
filters) and is imported here, not reimplemented, so this baseline cannot
silently drift from what the sampler actually does.

This is the reference any sampler must reproduce under inverse-probability
weighting. Per the advisor's instruction: "in terms of validation, let's
always compare the validation to the census panel you are sampling from."

Emits to outputs/, overall and broken out by coupon bucket
(((original_interest_rate - 0.5) * 2).round() / 2 -- the SAME convention as
mean_h_adj_by_coupon() in forecast_matched_population_cpr.py:208, reused
here for consistency with every other coupon-level table in this repo):

  - n_loans                    : loans in the cutoff-filtered panel (may
                                  exceed loans with >=1 eligible month --
                                  e.g. a 1-row loan is never eligible).
  - n_eligible_loan_months     : total eligible (loan_id, ref_month) rows.
  - n_prepay_events            : eligible loan-months where the
                                  select_observations() label formula
                                  (is_prepaid & term_t - row_idx <= H)
                                  would evaluate to 1.
  - raw_prepay_rate            : n_prepay_events / n_eligible_loan_months.
  - n_eligible per loan        : mean/median/p10/p90/max, computed over
                                  ALL loans in the panel (including loans
                                  with 0 eligible months) so it's honest
                                  about what fraction of loans contribute
                                  nothing. NOTE: this differs from
                                  select_observations()'s emitted
                                  per-observation `n_eligible` field, which
                                  is the non-mandatory POOL size only (one
                                  less than this script's count, when a
                                  mandatory draw exists for that loan). Do
                                  not compare the two without adjusting.
  - terminal-month classification: prepay is gated on `is_prepaid` (zbc==1
                                  ANYWHERE in the loan's panel, via term_t --
                                  the SAME definition _eligible_candidates()
                                  / the label formula use), NOT on whether
                                  the panel-LAST row (row_idx == L-1)
                                  happens to be the payoff row.
                                  _prepare_panel's docstring explicitly does
                                  not assume payoff is the last row, and an
                                  earlier version of this script did make
                                  that assumption -- a prepaid loan with any
                                  trailing row after payoff would then count
                                  as a positive in n_prepay_events but
                                  "censored_survivor" in this breakdown,
                                  silently disagreeing with itself. Fixed to
                                  gate on is_prepaid; a per-vintage
                                  diagnostic (below) counts how often the
                                  payoff row is NOT literally the last row,
                                  so the assumption is verified, not asserted.
                                  non_prepay_termination / censored_survivor
                                  are still read off the panel-last row's
                                  zero_balance_code_actual (in
                                  {2,3,6,9,15,16}, else censored) -- a
                                  non-prepaid loan has no earlier
                                  termination event to prefer instead.

Checkpointed per vintage (resume guard): each vintage's aggregates are
pickled to --checkpoint_dir immediately on completion. A rerun skips any
vintage whose checkpoint already exists, so a timeout mid-scan only costs
the in-flight vintage, not the whole job. Only small per-loan summaries
(not full per-loan-month frames) are ever held across vintages, to keep
memory bounded regardless of panel size.

Usage:
    python scripts/diag/census_panel_baseline.py --cutoff_year 2020
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare_sequences_multiobs_zbc as m

BASE = '/scratch/at7095/mortgage_prepayment'
NON_PREPAY_ZBC = {2, 3, 6, 9, 15, 16}
H, MIN_HIST = 1, 1   # matches the production cutoff_2020 multiobs run (k5_h1, min_hist default=1)


def coupon_bucket(rate: pd.Series) -> pd.Series:
    """SAME convention as mean_h_adj_by_coupon() in
    forecast_matched_population_cpr.py:208 -- reused verbatim, not
    reimplemented, so coupon buckets here match every other coupon-level
    table in this repo."""
    return ((rate - 0.5) * 2).round() / 2


def classify_terminal(is_prepaid: np.ndarray, last_zbc: pd.Series) -> np.ndarray:
    """Vectorized. Prepay is gated on is_prepaid (zbc==1 anywhere, via
    term_t -- the SAME definition the eligibility/label engine uses), NOT
    on the panel-last row's own code. See module docstring."""
    non_prepay = last_zbc.isin(NON_PREPAY_ZBC).to_numpy()
    return np.where(is_prepaid, 'prepay',
                     np.where(non_prepay, 'non_prepay_termination', 'censored_survivor'))


def process_vintage(vintage, cutoff_ym, pmms_rates, zhvi_df, ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, f'{vintage}.pkl')
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            result = pickle.load(f)
        print(f'  {vintage}: loaded from checkpoint', flush=True)
        return result

    df = m.load_vintage_filtered(vintage, pmms_rates, zhvi_df, cutoff_ym, keep_ids=None)
    if df is None or df.empty:
        result = None
    else:
        panel = m._prepare_panel(df)
        panel['coupon'] = coupon_bucket(panel['original_interest_rate'])

        # ── terminal-month classification: ALL loans ──────────────────────
        # prepay gated on is_prepaid (constant per loan); non-prepay/censored
        # read off the panel-LAST row's own code (see module docstring).
        term_rows = panel[panel['row_idx'] == panel['L'] - 1][
            ['loan_id', 'coupon', 'zero_balance_code_actual', 'is_prepaid']].copy()
        term_rows['terminal_class'] = classify_terminal(
            term_rows['is_prepaid'].to_numpy(), term_rows['zero_balance_code_actual'])

        # Diagnostic, not an assumption: how often is a prepaid loan's
        # panel-last row NOT literally the payoff row? If this is ever > 0,
        # it means data continues being reported after payoff for that loan.
        divergent = term_rows['is_prepaid'] & (term_rows['zero_balance_code_actual'] != 1.0)
        n_prepaid_loans = int(term_rows['is_prepaid'].sum())
        n_divergent     = int(divergent.sum())
        print(f'  {vintage}: prepaid loans whose panel-last row is NOT the payoff row: '
              f'{n_divergent:,} / {n_prepaid_loans:,} prepaid loans', flush=True)

        per_loan_terminal = term_rows[['loan_id', 'coupon', 'terminal_class']]

        # ── eligibility: SAME function select_observations() samples from ─
        elig = m._eligible_candidates(df, H, MIN_HIST)
        if elig.empty:
            eligible_months_by_coupon = pd.Series(dtype=np.int64)
            prepay_events_by_coupon   = pd.Series(dtype=np.int64)
            per_loan_n_eligible = (
                per_loan_terminal[['loan_id', 'coupon']]
                .assign(n_eligible=0)
            )
        else:
            elig = elig.copy()
            elig['coupon'] = coupon_bucket(elig['original_interest_rate'])
            elig['label']  = (elig['is_prepaid'] & ((elig['term_t'] - elig['row_idx']) <= H)).astype(int)

            eligible_months_by_coupon = elig.groupby('coupon').size()
            prepay_events_by_coupon   = elig.groupby('coupon')['label'].sum()

            n_elig_per_loan = elig.groupby('loan_id').size().rename('n_eligible')
            # left-join onto ALL loans (per_loan_terminal has one row per
            # loan in the panel) so loans with 0 eligible months are 0, not
            # missing -- the n_eligible distribution must be honest about them.
            per_loan_n_eligible = per_loan_terminal[['loan_id', 'coupon']].merge(
                n_elig_per_loan, on='loan_id', how='left')
            per_loan_n_eligible['n_eligible'] = per_loan_n_eligible['n_eligible'].fillna(0).astype(int)

        result = {
            'vintage': vintage,
            'n_loans': int(panel['loan_id'].nunique()),
            'n_prepaid_loans': n_prepaid_loans,
            'n_divergent_terminal': n_divergent,
            'eligible_months_by_coupon': eligible_months_by_coupon,
            'prepay_events_by_coupon':   prepay_events_by_coupon,
            'per_loan_n_eligible':       per_loan_n_eligible,
            'per_loan_terminal':         per_loan_terminal,
        }

    with open(ckpt_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'  {vintage}: checkpoint written', flush=True)
    return result


def summarize(coupon_key, n_eligible_series, terminal_df, months, events):
    n_loans = len(terminal_df)
    stats = {
        'n_loans': int(n_loans),
        'n_eligible_loan_months': int(months),
        'n_prepay_events': int(events),
        'raw_prepay_rate': (events / months) if months > 0 else None,
        'n_eligible_per_loan': {
            'mean':   float(n_eligible_series.mean())   if n_loans else None,
            'median': float(n_eligible_series.median()) if n_loans else None,
            'p10':    float(n_eligible_series.quantile(0.10)) if n_loans else None,
            'p90':    float(n_eligible_series.quantile(0.90)) if n_loans else None,
            'max':    float(n_eligible_series.max())    if n_loans else None,
        },
        'terminal_month': {},
    }
    vc = terminal_df['terminal_class'].value_counts()
    for cls in ('prepay', 'non_prepay_termination', 'censored_survivor'):
        c = int(vc.get(cls, 0))
        stats['terminal_month'][cls] = {
            'count': c,
            'share': (c / n_loans) if n_loans else None,
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year', type=int, required=True)
    args = parser.parse_args()

    cutoff_ym = m.dec_yyyymm(args.cutoff_year)
    ckpt_dir = os.path.join(BASE, 'outputs', 'census_panel_baseline_checkpoints',
                             f'cutoff_{args.cutoff_year}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Census panel baseline | cutoff = Dec {args.cutoff_year} (YYYYMM={cutoff_ym}) | '
          f'H={H} min_hist={MIN_HIST} (matches production k5_h1 run)', flush=True)
    print(f'Checkpoint dir: {ckpt_dir}', flush=True)

    pmms_rates = m.load_pmms()
    zhvi_df    = m.load_zhvi()

    per_loan_n_eligible_parts = []
    per_loan_terminal_parts   = []
    eligible_months_total = pd.Series(dtype=np.int64)
    prepay_events_total   = pd.Series(dtype=np.int64)
    n_prepaid_loans_total   = 0
    n_divergent_total       = 0

    for v in m.ALL_VINTAGES:
        result = process_vintage(v, cutoff_ym, pmms_rates, zhvi_df, ckpt_dir)
        if result is None:
            continue
        per_loan_n_eligible_parts.append(result['per_loan_n_eligible'])
        per_loan_terminal_parts.append(result['per_loan_terminal'])
        eligible_months_total = eligible_months_total.add(result['eligible_months_by_coupon'], fill_value=0)
        prepay_events_total   = prepay_events_total.add(result['prepay_events_by_coupon'], fill_value=0)
        n_prepaid_loans_total += result['n_prepaid_loans']
        n_divergent_total     += result['n_divergent_terminal']

    per_loan_n_eligible = pd.concat(per_loan_n_eligible_parts, ignore_index=True)
    per_loan_terminal   = pd.concat(per_loan_terminal_parts, ignore_index=True)

    print(f'\nAll vintages processed. Total loans: {len(per_loan_terminal):,}', flush=True)
    print(f'Prepaid loans whose panel-last row is NOT the payoff row (all vintages): '
          f'{n_divergent_total:,} / {n_prepaid_loans_total:,} prepaid loans', flush=True)

    out = {
        'cutoff_year': args.cutoff_year,
        'H': H,
        'min_hist': MIN_HIST,
        'terminal_classification_diagnostic': {
            'n_prepaid_loans': n_prepaid_loans_total,
            'n_where_payoff_row_is_not_last_row': n_divergent_total,
            'note': ('prepay in terminal_month is gated on is_prepaid (zbc==1 anywhere), '
                     'not on the panel-last row -- this counts how often those two would '
                     'have disagreed, i.e. how load-bearing that choice actually is.'),
        },
        'overall': summarize('ALL', per_loan_n_eligible['n_eligible'], per_loan_terminal,
                              eligible_months_total.sum(), prepay_events_total.sum()),
        'by_coupon': {},
    }

    coupons = sorted(set(eligible_months_total.index) | set(per_loan_terminal['coupon'].unique()))
    for c in coupons:
        sub_n_elig = per_loan_n_eligible.loc[per_loan_n_eligible['coupon'] == c, 'n_eligible']
        sub_term   = per_loan_terminal.loc[per_loan_terminal['coupon'] == c]
        months = eligible_months_total.get(c, 0)
        events = prepay_events_total.get(c, 0)
        out['by_coupon'][str(c)] = summarize(c, sub_n_elig, sub_term, months, events)

    out_dir = os.path.join(BASE, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'census_panel_baseline_cutoff_{args.cutoff_year}.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'Wrote {json_path}', flush=True)

    rows = []
    for key, stats in [('ALL', out['overall'])] + [(c, out['by_coupon'][c]) for c in out['by_coupon']]:
        rows.append({
            'coupon': key,
            'n_loans': stats['n_loans'],
            'n_eligible_loan_months': stats['n_eligible_loan_months'],
            'n_prepay_events': stats['n_prepay_events'],
            'raw_prepay_rate': stats['raw_prepay_rate'],
            'n_eligible_mean':   stats['n_eligible_per_loan']['mean'],
            'n_eligible_median': stats['n_eligible_per_loan']['median'],
            'n_eligible_p10':    stats['n_eligible_per_loan']['p10'],
            'n_eligible_p90':    stats['n_eligible_per_loan']['p90'],
            'n_eligible_max':    stats['n_eligible_per_loan']['max'],
            'terminal_prepay_count':       stats['terminal_month']['prepay']['count'],
            'terminal_prepay_share':       stats['terminal_month']['prepay']['share'],
            'terminal_nonprepay_count':    stats['terminal_month']['non_prepay_termination']['count'],
            'terminal_nonprepay_share':    stats['terminal_month']['non_prepay_termination']['share'],
            'terminal_censored_count':     stats['terminal_month']['censored_survivor']['count'],
            'terminal_censored_share':     stats['terminal_month']['censored_survivor']['share'],
        })
    csv_path = os.path.join(out_dir, f'census_panel_baseline_cutoff_{args.cutoff_year}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}', flush=True)

    print('\n=== OVERALL ===')
    print(json.dumps(out['overall'], indent=2, default=str))


if __name__ == '__main__':
    main()
