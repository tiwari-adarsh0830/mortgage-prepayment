"""
SUPERSEDED — kept as a documented dead end. Do not use this result.

Intended as an independent re-derivation of the spread effect. It differences
the spread on info_date in place, which spans the PREVIOUS month, while the
panel's d_level spans FORWARD from info_date to the next info_date. The
off-by-one produces corr(d_level, d_spread) = -0.0099 and a ratio delta of
0.000 — a plausible-looking null that briefly overturned the correct result.

Correct alignment is in verify_spread_effect_v2.py, which asserts the window
against the panel rather than inferring it. diag_spread_alignment.py is what
diagnosed the difference. See README Phase 28, "Trap: the panel's d_level
spans FORWARD from info_date".
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

# --- 10yr on info_date, own read, own merge ---------------------------------
dly = pd.read_csv(os.path.join(DATA,'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()
t10 = dly['10yr'].reset_index()
t10.columns = ['Date','y10']

dates = pan[['ret_month','info_date']].drop_duplicates().sort_values('info_date')
dates = pd.merge_asof(dates, t10.sort_values('Date'),
                      left_on='info_date', right_on='Date', direction='backward')

# --- PMMS keyed to the month of info_date (panel convention, verified) ------
pm = pd.read_csv(os.path.join(DATA,'pmms_monthly.csv'))
def parse(x):
    s = str(int(x))
    if len(s)==5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]),  day=1)
    if len(s)==6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm['k'] = pm['reporting_period'].apply(parse)
pm = pm.dropna(subset=['k'])
pmap = dict(zip(pm['k'], pm['rate_30yr']))
dates['pmms_mine'] = [pmap.get(pd.Timestamp(d.year,d.month,1), np.nan)
                      for d in dates['info_date']]

# GUARD A: my PMMS must equal the panel column exactly
pchk = dates.merge(pan.drop_duplicates('ret_month')[['ret_month','pmms']],
                   on='ret_month')
gap = (pchk['pmms_mine'] - pchk['pmms']).abs().max()
print('GUARD A  max |pmms_mine - panel pmms| = %.3e' % gap)
if gap > 1e-9:
    raise SystemExit('ABORT: PMMS keying differs from panel')

# spread differenced over the return month, same convention as d_level
dates = dates.sort_values('info_date').reset_index(drop=True)
dates['spread'] = dates['pmms_mine'] - dates['y10']
dates['d_spread'] = dates['spread'].diff()
sp = dates[['ret_month','d_spread']].dropna()

m = pan.merge(sp, on='ret_month', how='inner')
print('GUARD B  months after merge: %d' % m['ret_month'].nunique())
print('         corr(d_level,d_spread) = %.4f'
      % np.corrcoef(m.d_level, m.d_spread)[0,1])

def fit(y, X):
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    return co

print()
print('cpn  ratio3   ratio_spr   delta')
r3s, r4s = [], []
for cpn, g in m.groupby('coupon'):
    y  = (g['tba_total_return'] - g['income']).values
    X3 = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve])
    X4 = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve, g.d_spread])
    dm = g['D_level'].mean()
    a  = -100.0*fit(y,X3)[1]/dm
    b  = -100.0*fit(y,X4)[1]/dm
    r3s.append(a); r4s.append(b)
    print('%.1f %7.3f %10.3f %8.3f' % (cpn, a, b, b-a))

m3, m4 = float(np.median(r3s)), float(np.median(r4s))
print()
print('median ratio3   : %.3f   (first run 1.352)' % m3)
print('median ratio_spr: %.3f   (first run 1.381)' % m4)
print('change          : %+.3f   (first run +0.029)' % (m4-m3))
print()
ok = abs(m3-1.352) < 0.005 and abs(m4-1.381) < 0.005
print('REPRODUCES first run' if ok else
      'DOES NOT reproduce -- reconcile before reporting')
