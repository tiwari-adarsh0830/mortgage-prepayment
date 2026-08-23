
"""
Does the spread channel predict an empirical duration ABOVE or BELOW the
cashflow duration, and by how much?

His Aug 23 note: 'TBA empirical durations exceed cashflow durations (for
example, because spreads move with rates)'. Read literally that is not a
claim something is broken -- it says the excess is real. The Phase 28 spread
test asked whether a spread REGRESSOR absorbs the gap. This asks the prior
question: what does the channel predict?

Arithmetic. Price return is driven by the discount rate the TBA is valued at,
which is the Treasury level plus the mortgage spread:

    dP/P = -D_cash * (d_level + d_spread)

If d_spread = beta * d_level + noise, then regressing dP/P on d_level alone
recovers

    D_emp = D_cash * (1 + beta)

So beta > 0 (spread WIDENS as rates rise) implies empirical ABOVE cashflow,
which is his hypothesis. beta < 0 (spread tightens) implies empirical BELOW,
the conventional finding. The observed ratio is ~1.35, needing beta ~ +0.35.
"""
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

pan = pd.read_csv(PANEL)
pan['info_date'] = pd.to_datetime(pan['info_date'])
u = (pan.drop_duplicates('ret_month')[['ret_month','info_date','pmms','d_level']]
        .sort_values('info_date').reset_index(drop=True))

dly = pd.read_csv(os.path.join(DATA,'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()
t10 = dly['10yr'].reset_index(); t10.columns = ['Date','y10']
u = pd.merge_asof(u, t10.sort_values('Date'),
                  left_on='info_date', right_on='Date', direction='backward')

# window: forward from info_date to the NEXT info_date (Phase 28 trap)
u['sp'] = u['pmms'] - u['y10']
u['d_spread'] = u['sp'].shift(-1) - u['sp']
m = u.dropna(subset=['d_spread','d_level'])
# d_level on row i already spans forward, so it pairs with d_spread on row i
X = np.column_stack([np.ones(len(m)), m.d_level.values])
co, *_ = np.linalg.lstsq(X, m.d_spread.values, rcond=None)
r  = m.d_spread.values - X @ co
s2 = (r @ r) / (len(m) - 2)
se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
beta, tb = co[1], co[1]/se[1]

print('months: %d' % len(m))
print('beta (d_spread on d_level) = %+.4f   t = %+.2f' % (beta, tb))
print('corr(d_level, d_spread)    = %+.4f'
      % np.corrcoef(m.d_level, m.d_spread)[0,1])
print()
print('implied D_emp / D_cash = 1 + beta = %.4f' % (1.0 + beta))
print('observed ratio                    = 1.35')
print('beta required for 1.35            = +0.35')
print()
if beta < 0:
    print('SIGN: spread TIGHTENS as rates rise, so the channel predicts empirical')
    print('duration BELOW cashflow duration -- the opposite of what we observe.')
else:
    print('SIGN: spread WIDENS as rates rise; channel predicts empirical ABOVE.')
print()
print('subsample check (channel stability):')
h = len(m)//2
for lbl, s in [('first half', m.iloc[:h]), ('second half', m.iloc[h:])]:
    Xs = np.column_stack([np.ones(len(s)), s.d_level.values])
    cs, *_ = np.linalg.lstsq(Xs, s.d_spread.values, rcond=None)
    print('  %-12s beta %+.4f  -> 1+beta %.3f  (n=%d)'
          % (lbl, cs[1], 1+cs[1], len(s)))
