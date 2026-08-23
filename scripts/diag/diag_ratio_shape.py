
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

def ols(y, X):
    XtX = X.T @ X
    co  = np.linalg.solve(XtX, X.T @ y)
    r   = y - X @ co
    s2  = (r @ r) / (len(y) - X.shape[1])
    V   = s2 * np.linalg.inv(XtX)
    se  = np.sqrt(np.diag(V))
    return co, se, co / se

pan = pd.read_csv(PANEL)
pan['info_date'] = pd.to_datetime(pan['info_date'])
pan = pan.sort_values(['coupon','info_date']).reset_index(drop=True)

# GUARD: tents3 identity check -- d_level + d_slope/2 is NOT dy10 here
dly = pd.read_csv(os.path.join(DATA,'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()
u = pan.drop_duplicates('ret_month')[['ret_month','info_date','d_level','d_slope']]
r10 = dly[['10yr']].reset_index(); r10.columns=['Date','y10']
u = pd.merge_asof(u.sort_values('info_date'), r10.sort_values('Date'),
                  left_on='info_date', right_on='Date', direction='backward')
u['dy10_true'] = u['y10'].diff()
u['dy10_ident'] = u['d_level'] + u['d_slope']/2.0
gap = (u['dy10_true'] - u['dy10_ident']).abs().mean()
print('mean |dy10_true - (d_level+d_slope/2)| = %.4f' % gap)
print('  -> confirms the Phase 22 two-tent identity does NOT hold on tents3'
      if gap > 1e-6 else '  -> identity holds (unexpected)')

print()
print('=== per-coupon ratio, full-sample, with SE ===')
print('cpn   D_model   D_fit    se     ratio   se_ratio')
rows = []
for cpn, g in pan.groupby('coupon'):
    y = (g['tba_total_return'] - g['income']).values
    X = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve])
    co, se, _ = ols(y, X)
    dm  = g['D_level'].mean()
    fit = -100.0*co[1]
    sef = 100.0*se[1]
    rows.append({'coupon':cpn,'D_model':dm,'D_fit':fit,'se':sef,
                 'ratio':fit/dm,'se_ratio':sef/dm})
    print('%.1f %8.3f %8.3f %6.3f %7.3f %8.3f'
          % (cpn, dm, fit, sef, fit/dm, sef/dm))
d = pd.DataFrame(rows)

print()
print('=== shape tests on the ratio ===')
c  = d['coupon'].values
w  = 1.0/d['se_ratio'].values**2
for lbl, Xs in [('constant only', np.ones((9,1))),
                ('linear',  np.column_stack([np.ones(9), c-c.mean()])),
                ('quadratic', np.column_stack([np.ones(9), c-c.mean(), (c-c.mean())**2]))]:
    Xw = Xs * np.sqrt(w)[:,None]
    yw = d['ratio'].values * np.sqrt(w)
    co, se, tt = ols(yw, Xw)
    res = d['ratio'].values - Xs @ co
    chi = float((res**2 * w).sum())
    print('%-14s chi2=%7.2f  dof=%d  coefs=%s'
          % (lbl, chi, 9-Xs.shape[1],
             ' '.join('%.4f(t=%.2f)' % (a,b) for a,b in zip(co,tt))))
print()
print('quadratic t on coupon^2 is the test: |t|>2 means the gap has shape,')
print('not a single scalar. chi2 for constant-only vs dof=8 says whether a')
print('flat ratio is even consistent with the estimation error.')

print()
print('=== expanding-window arm (36mo burn-in) ===')
print('cpn   ratio_full  ratio_EW   n_EW')
ew = []
for cpn, g in pan.groupby('coupon'):
    g = g.sort_values('info_date').reset_index(drop=True)
    fits = []
    for t in range(36, len(g)):
        h = g.iloc[:t]
        y = (h['tba_total_return'] - h['income']).values
        X = np.column_stack([np.ones(len(h)), h.d_level, h.d_slope, h.d_curve])
        try:
            co, _, _ = ols(y, X)
            fits.append(-100.0*co[1] / h['D_level'].mean())
        except np.linalg.LinAlgError:
            pass
    rf = float(d.loc[d.coupon==cpn,'ratio'].iloc[0])
    ew.append(np.mean(fits))
    print('%.1f %10.3f %10.3f %6d' % (cpn, rf, np.mean(fits), len(fits)))
print()
print('median full %.3f | median EW %.3f'
      % (d['ratio'].median(), float(np.median(ew))))
