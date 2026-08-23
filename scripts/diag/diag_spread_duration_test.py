"""
First run of the spread-control test. Result is correct (median ratio 1.352 ->
1.381) and reproduced by verify_spread_effect_v2.py, but this script infers
the alignment from a correlation rather than asserting it. Prefer v2.
"""

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
    se  = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    ss  = ((y - y.mean()) ** 2).sum()
    return co, co / se, 1.0 - (r @ r) / ss

# ---- yields, as-of aligned to the clean month-end dates --------------------
dly = pd.read_csv(os.path.join(DATA, 'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()
cln = pd.read_excel(os.path.join(DATA, 'treasury_yields_clean.xlsx'),
                    sheet_name='Treasury_Yields', header=1)
cln.columns = [str(c).strip() for c in cln.columns]
cln['Date'] = pd.to_datetime(cln['Date'])
cln = cln.sort_values('Date').reset_index(drop=True)

right = dly[['2yr','5yr','10yr']].reset_index()
right.columns = ['Date','y2','y5','y10']
al = pd.merge_asof(pd.DataFrame({'Date': cln['Date']}).sort_values('Date'),
                   right.sort_values('Date'), on='Date', direction='backward')

# ---- PMMS on the SAME dates, own construction -----------------------------
pm = pd.read_csv(os.path.join(DATA, 'pmms_monthly.csv'))
def parse(p):
    s = str(int(p))
    if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]),  day=1)
    if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm['mstart'] = pm['reporting_period'].apply(parse)
pm = pm.dropna(subset=['mstart'])
pmap = dict(zip(pm['mstart'], pm['rate_30yr']))
al['pmms'] = [pmap.get(pd.Timestamp(d.year, d.month, 1), np.nan) for d in al['Date']]

al['spread']  = al['pmms'] - al['y10']
al['d_spread'] = al['spread'].diff()
al['d_level']  = ((al.y2 + al.y5 + al.y10) / 3.0).diff()
al['d_slope']  = (al.y10 - al.y2).diff()
al['d_curve']  = (2.0 * al.y5 - al.y2 - al.y10).diff()
al['ret_month'] = al['Date'].shift(-1).dt.strftime('%Y-%m')

shk = pd.DataFrame({'ret_month': al['ret_month'],
                    'd_level': al['d_level'].shift(-1),
                    'd_slope': al['d_slope'].shift(-1),
                    'd_curve': al['d_curve'].shift(-1),
                    'd_spread': al['d_spread'].shift(-1),
                    'pmms_end': al['pmms'].shift(-1),
                    'pmms_start': al['pmms']}).dropna()

pan = pd.read_csv(PANEL)

# ---- GUARD 1: shocks must reproduce the panel ------------------------------
ck = pan.drop_duplicates('ret_month')[['ret_month','d_level','d_slope','d_curve','pmms']]
ck = shk.merge(ck, on='ret_month', suffixes=('_mine','_pan'))
for leg in ['d_level','d_slope','d_curve']:
    d = (ck[leg+'_mine'] - ck[leg+'_pan']).abs().max()
    print('shock check %-8s max abs diff %.3e' % (leg, d))
    if d > 1e-9:
        raise SystemExit('ABORT: shocks do not reproduce panel')

# ---- GUARD 2: confirm the panel pmms keying (Phase 22 trap) ----------------
c_start = np.corrcoef(ck['pmms_start'], ck['pmms'])[0,1]
c_end   = np.corrcoef(ck['pmms_end'],   ck['pmms'])[0,1]
print()
print('corr(panel pmms, my start-of-return-month pmms) = %.4f' % c_start)
print('corr(panel pmms, my end-of-return-month pmms)   = %.4f' % c_end)
if not (c_start > c_end):
    raise SystemExit('ABORT: panel pmms is not info-date keyed as assumed')
print('-> panel pmms is info-date (start-of-return-month) keyed, as expected;')
print('   d_spread built end-minus-start over the return month, matching d_level.')

print()
print('corr(d_level, d_spread) = %.4f   (Phase 22 reported -0.601)'
      % np.corrcoef(shk.d_level, shk.d_spread)[0,1])

# ---- the test -------------------------------------------------------------
print()
print('cpn   D_model   D_fit(3)  ratio3   D_fit(+spr) ratio_s   t_spr   dR2')
rows = []
for cpn, g in pan.groupby('coupon'):
    g = g.merge(shk[['ret_month','d_spread']], on='ret_month', how='inner')
    y = (g['tba_total_return'] - g['income']).values
    X3 = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve])
    X4 = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve, g.d_spread])
    c3, t3, r3 = ols(y, X3)
    c4, t4, r4 = ols(y, X4)
    dm = g['D_level'].mean()
    f3, f4 = -100.0*c3[1], -100.0*c4[1]
    rows.append((cpn, dm, f3, f3/dm, f4, f4/dm, t4[4], r4-r3))
    print('%.1f %8.3f %9.3f %7.3f %11.3f %8.3f %7.2f %6.3f'
          % rows[-1])

# ---- GUARD 3: control must reproduce the established ratio -----------------
med3 = np.median([r[3] for r in rows])
med4 = np.median([r[5] for r in rows])
print()
print('median ratio, 3 shocks      : %.3f  (established 1.31-1.36)' % med3)
print('median ratio, + spread      : %.3f' % med4)
if not (1.28 < med3 < 1.40):
    raise SystemExit('ABORT: control does not reproduce; discard run')
print('change in median ratio      : %+.3f' % (med4 - med3))
print()
print('if ratio_s moves toward 1.00, spread exposure explains part of the gap;')
print('if it is unchanged, the spread control is not the explanation.')
