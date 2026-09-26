"""test_multiobs_sampler_fixed_fraction.py -- synthetic-panel unit tests for
sampling_mode='fixed_fraction' in prepare_sequences_multiobs_zbc.py's
select_observations(). Companion to test_multiobs_sampler.py (which covers
the original fixed_k path); this file covers only what's NEW.

NO cluster data is touched -- login-node-safe, same discipline as
test_multiobs_sampler.py. Verifies:
  - every non-terminal draw's incl_prob is the SAME within a loan's pool
    (budget/n_pool), and converges to frac_draws as n_pool grows
  - the terminal draw keeps incl_prob == 1.0 regardless of frac_draws --
    the terminal-draw convention documented in select_observations()
  - the documented ceiling artifact: a pool of size 1 gets incl_prob == 1.0
    for any frac_draws in (0, 1], since ceil() cannot draw a fraction of
    one observation
  - the fixed_k path (default sampling_mode) is byte-for-byte unaffected by
    the new parameters existing
  - frac_draws validation guards (required + range) raise ValueError

Run:
    cd /scratch/at7095/mortgage_prepayment
    python scripts/diag/test_multiobs_sampler_fixed_fraction.py
"""
import sys
sys.path.insert(0, 'scripts')

import numpy as np
import pandas as pd

import prepare_sequences_multiobs_zbc as m

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail and not cond else ''))
    if not cond:
        FAILURES.append(name)


def make_loan(loan_id, n_rows, rate=3.0, start_yyyymm=202001, prepay_at=None):
    rows = []
    ym = start_yyyymm
    for t in range(n_rows):
        y, mo = divmod(ym, 100)
        zbc = 1.0 if (prepay_at is not None and t == prepay_at) else np.nan
        rows.append({
            'loan_id': loan_id, 'yyyymm': ym, 'monthly_reporting_period': ym,
            'prepaid': 1 if prepay_at is not None else 0,
            'zero_balance_code_actual': zbc,
            'original_interest_rate': rate,
            'refi_incentive': 0.0, 'borrower_credit_score': 750.0,
            'original_ltv': 80.0, 'current_ltv': 75.0, 'original_upb': 300000.0,
            'loan_age_months': float(t), 'dti': 30.0,
            'loan_purpose_enc': 0.0, 'property_type_enc': 0.0,
        })
        mo += 1
        if mo > 12:
            mo = 1
            y += 1
        ym = y * 100 + mo
    return rows


H, MIN_HIST = 1, 1

# ── constant incl_prob property, varying loan lengths ──────────────────────
rows = []
rows += make_loan('SHORT', 6,  prepay_at=None)     # small pool
rows += make_loan('LONG',  40, prepay_at=None)     # large pool
df = pd.DataFrame(rows)

FRAC = 0.3
obs = m.select_observations(df, k_draws=999, H=H, min_hist=MIN_HIST,
                             sampling_mode='fixed_fraction', frac_draws=FRAC)
print(obs[['loan_id', 't', 'incl_prob', 'is_terminal', 'n_eligible']].to_string())

for lid in ['SHORT', 'LONG']:
    sub = obs[(obs['loan_id'] == lid) & (~obs['is_terminal'])]
    incl = sub['incl_prob'].unique()
    check(f'{lid}: all non-terminal incl_prob identical within loan',
          len(incl) == 1, detail=f'{incl}')

short_incl  = obs.loc[(obs['loan_id'] == 'SHORT') & (~obs['is_terminal']), 'incl_prob'].iloc[0]
long_incl   = obs.loc[(obs['loan_id'] == 'LONG')  & (~obs['is_terminal']), 'incl_prob'].iloc[0]
n_pool_long = obs.loc[obs['loan_id'] == 'LONG', 'n_eligible'].iloc[0]
check('LONG incl_prob is close to frac_draws (large pool, ceil() rounding small)',
      abs(long_incl - FRAC) < 1.0 / n_pool_long + 1e-9,
      detail=f'{long_incl} vs {FRAC}, n_pool={n_pool_long}')
check('SHORT incl_prob >= frac_draws (ceil() only rounds UP)', short_incl >= FRAC,
      detail=f'{short_incl} vs {FRAC}')

# ── terminal draw always incl_prob == 1.0, any frac_draws ─────────────────
rows2 = make_loan('EVT', 20, prepay_at=15)
df2 = pd.DataFrame(rows2)
for f in (0.01, 0.5, 1.0):
    obs2 = m.select_observations(df2, k_draws=999, H=H, min_hist=MIN_HIST,
                                  sampling_mode='fixed_fraction', frac_draws=f)
    term = obs2[obs2['is_terminal']]
    check(f'terminal draw incl_prob == 1.0 at frac_draws={f}',
          len(term) == 1 and term['incl_prob'].iloc[0] == 1.0,
          detail=f'{term["incl_prob"].tolist()}')
    non_term_incl = obs2.loc[~obs2['is_terminal'], 'incl_prob'].unique()
    # At f=1.0 EVERY candidate (terminal and non-terminal alike) is
    # legitimately included with certainty -- incl_prob coinciding at 1.0
    # there is correct, not a sign the terminal-draw special-case broke.
    if f < 1.0:
        check(f'non-terminal weight (1/incl_prob) != terminal weight (1.0) at frac_draws={f}',
              len(non_term_incl) == 0 or non_term_incl[0] != 1.0,
              detail=f'{non_term_incl}')

# ── known ceiling artifact -- pool of exactly size 1 always incl_prob=1.0 ─
# L=2, H=1 -> eligible row_idx <= L-1-H=0, row_idx < term_t=1 -> row_idx=0
# only, and it IS the mandatory draw (term_t-H=0) -> pool is empty here, so
# use L=3 with NO mandatory draw (censored) to get a clean pool of size 1:
# eligible row_idx <= L-1-H=1, row_idx<term_t=L-1=2 -> row_idx in {0,1},
# neither equals term_t-H=1... row_idx=1 DOES equal term_t-H=1 -> mandatory.
# So row_idx=0 is the only non-mandatory (pool) candidate -> n_pool=1.
rows3 = make_loan('TINY1', 3, prepay_at=None)
df3 = pd.DataFrame(rows3)
for f in (0.01, 0.1, 0.99):
    obs3 = m.select_observations(df3, k_draws=999, H=H, min_hist=MIN_HIST,
                                  sampling_mode='fixed_fraction', frac_draws=f)
    non_term = obs3[~obs3['is_terminal']]
    check(f'known artifact: pool-of-1 gets incl_prob==1.0 at frac_draws={f}',
          len(non_term) == 1 and (non_term['incl_prob'] == 1.0).all(),
          detail=f'{non_term["incl_prob"].tolist()}, n_pool={non_term["n_eligible"].tolist()}')

# ── fixed_k default path completely unaffected by the new parameters ──────
obs_k_old = m.select_observations(df, k_draws=3, H=H, min_hist=MIN_HIST)
obs_k_new = m.select_observations(df, k_draws=3, H=H, min_hist=MIN_HIST,
                                   sampling_mode='fixed_k', frac_draws=None)
check('fixed_k results identical with/without explicit sampling_mode=fixed_k',
      obs_k_old.equals(obs_k_new))

# ── validation guards ───────────────────────────────────────────────────────
try:
    m.select_observations(df, k_draws=5, H=H, min_hist=MIN_HIST,
                          sampling_mode='fixed_fraction', frac_draws=None)
    check('fixed_fraction with frac_draws=None raises', False)
except ValueError:
    check('fixed_fraction with frac_draws=None raises', True)

try:
    m.select_observations(df, k_draws=5, H=H, min_hist=MIN_HIST,
                          sampling_mode='fixed_fraction', frac_draws=1.5)
    check('fixed_fraction with frac_draws=1.5 (out of range) raises', False)
except ValueError:
    check('fixed_fraction with frac_draws=1.5 (out of range) raises', True)

print(f'\n{"ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILURES: " + str(FAILURES)}')
sys.exit(1 if FAILURES else 0)
