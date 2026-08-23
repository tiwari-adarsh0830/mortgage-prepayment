
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

pan = pd.read_csv(PANEL).sort_values(['coupon','info_date'])

def fit_ratio(h):
    y = (h['tba_total_return'] - h['income']).values
    X = np.column_stack([np.ones(len(h)), h.d_level, h.d_slope, h.d_curve])
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    return -100.0*co[1] / h['D_level'].mean()

print('cpn  full   EWmean  EWmed  EWlast24  EW_final  min    max')
tab = []
for cpn, g in pan.groupby('coupon'):
    g = g.reset_index(drop=True)
    fits = [fit_ratio(g.iloc[:t]) for t in range(36, len(g)+1)]
    tab.append({'c':cpn,'full':fit_ratio(g),'mean':np.mean(fits),
                'med':np.median(fits),'last24':np.mean(fits[-24:]),
                'final':fits[-1],'min':min(fits),'max':max(fits)})
    r = tab[-1]
    print('%.1f %6.3f %7.3f %6.3f %9.3f %9.3f %6.3f %6.3f'
          % (cpn,r['full'],r['mean'],r['med'],r['last24'],r['final'],r['min'],r['max']))

d = pd.DataFrame(tab)
print()
for k in ['full','mean','med','last24','final']:
    print('median across coupons, %-7s: %.3f' % (k, d[k].median()))
print()
print('EW_final uses all 99 months, so it MUST equal full. Check:')
print('  max |EW_final - full| = %.2e' % (d['final']-d['full']).abs().max())
print()
print('if EWmed ~ full but EWmean < full, the drop is early-window noise;')
print('if EWmed and last24 are both below full, the scalar genuinely drifts.')
