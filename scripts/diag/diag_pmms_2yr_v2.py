#!/usr/bin/env python3
"""diag_pmms_2yr_v2.py -- rebuild the PMMS/2yr sensitivity regression using
diag_pmms_spread.py's own verified loading pattern, instead of my earlier
'ME'-resample + to_period('M') approach, to check whether the lag0/lag1
discrepancy found in diag_pmms_2yr_regime.py was a genuine reporting lag or a
resample-convention mismatch between the two.

diag_pmms_spread.py uses resample('MS').last() (month-START labeled bucket)
against pm['date'] = reporting_period parsed to day=1, joined with NO lag term.
My earlier regressions used resample('ME').last() (month-END labeled bucket)
joined via to_period('M'). These are not guaranteed to select the same
underlying daily observations for a given calendar month, so lag0 in each
script may not mean the same thing.
"""
import numpy as np
import pandas as pd

d = pd.read_csv('data/treasury_yields.csv', index_col=0, parse_dates=True).sort_index()
pm = pd.read_csv('data/pmms_monthly.csv')

def parse(x):
    s = str(int(x))
    if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm['date'] = pm['reporting_period'].apply(parse)
pms = pm.dropna(subset=['date']).set_index('date')['rate_30yr']

d2  = d['2yr'].resample('MS').last()
d5  = d['5yr'].resample('MS').last()
d10 = d['10yr'].resample('MS').last()

j = pd.DataFrame({'pmms': pms, 'y2': d2, 'y5': d5, 'y10': d10}).dropna()
j['dpmms'] = j.pmms.diff()
j['dy2'] = j.y2.diff()
j['dy5'] = j.y5.diff()
j['dy10'] = j.y10.diff()
j = j.dropna()
j2 = j[j.index >= '2018-01-01']

def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    se = np.sqrt(np.diag(float(r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(XtX)))
    return co, se

print("using diag_pmms_spread.py's MS-resample convention, post-2018:")
X = np.column_stack([np.ones(len(j2)), j2.dy2.values, j2.dy5.values, j2.dy10.values])
co, se = ols(j2.dpmms.values, X)
for i, c in enumerate(['const','dy2','dy5','dy10']):
    print("  %-6s coef=%.4f  t=%6.2f" % (c, co[i], co[i]/se[i] if i else float('nan')))

print("\nlag0 vs lag1, univariate, post-2018 (MS convention):")
for lab, s in [('lag0', j2.dy2), ('lag1', j2.dy2.shift(1))]:
    m = j2.assign(x=s).dropna(subset=['x'])
    Xu = np.column_stack([np.ones(len(m)), m.x.values])
    cu, seu = ols(m.dpmms.values, Xu)
    print("  %-4s beta=%.3f t=%.2f n=%d" % (lab, cu[1], cu[1]/seu[1], len(m)))

print("\ncompare against my earlier ME-convention regression on the same window:")
print("  ME lag0: beta=0.452 t=4.06   ME lag1: beta=0.617 t=6.54")
print("  If MS lag0 above is now close to ME lag1, the discrepancy WAS a")
print("  resample-convention artifact, not a real reporting lag.")
