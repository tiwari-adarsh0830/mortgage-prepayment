
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

dly = pd.read_csv(os.path.join(DATA, 'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()
cln = pd.read_excel(os.path.join(DATA, 'treasury_yields_clean.xlsx'),
                    sheet_name='Treasury_Yields', header=1)
cln.columns = [str(c).strip() for c in cln.columns]
cln['Date'] = pd.to_datetime(cln['Date'])
cln = cln.sort_values('Date').reset_index(drop=True)

# as-of backward join (merge_asof, not the loop the pricer uses)
left  = pd.DataFrame({'Date': cln['Date']})
right = dly[['2yr','5yr','10yr']].reset_index()
right.columns = ['Date','y2','y5','y10']
al = pd.merge_asof(left.sort_values('Date'), right.sort_values('Date'),
                   on='Date', direction='backward')

al['d_level'] = ((al.y2 + al.y5 + al.y10) / 3.0).diff()
al['d_slope'] = (al.y10 - al.y2).diff()
al['d_curve'] = (2.0 * al.y5 - al.y2 - al.y10).diff()
al['ret_month'] = al['Date'].shift(-1).dt.strftime('%Y-%m')

def price_loop(y_pct, cpn_pct, times):
    y = y_pct / 100.0 / 2.0
    c = cpn_pct / 2.0
    p = 0.0
    for i, t in enumerate(times):
        cf = c + (100.0 if i == len(times) - 1 else 0.0)
        p += cf / (1.0 + y) ** (2.0 * t)
    return p

def mod_dur_closed(y_pct, cpn_pct, times):
    y = y_pct / 100.0 / 2.0
    c = cpn_pct / 2.0
    p = wsum = 0.0
    for i, t in enumerate(times):
        cf = c + (100.0 if i == len(times) - 1 else 0.0)
        pv = cf / (1.0 + y) ** (2.0 * t)
        p += pv; wsum += t * pv
    return (wsum / p) / (1.0 + y)

T     = [0.5 * k for k in range(1, 11)]
Taged = [t - 1.0 / 12.0 for t in T]

recs = []
for i in range(len(al) - 1):
    rm = al.ret_month.iloc[i]
    y0, y1 = al.y5.iloc[i], al.y5.iloc[i + 1]
    if not isinstance(rm, str) or pd.isna(y0) or pd.isna(y1):
        continue
    y0, y1 = float(y0), float(y1)
    p1 = price_loop(y1, y0, Taged)
    recs.append({'ret_month': rm,
                 'excess': (p1 + y0/12.0 - 100.0)/100.0 - y0/12.0/100.0,
                 'D_closed': mod_dur_closed(y0, y0, T),
                 'd_level': al.d_level.iloc[i+1],
                 'd_slope': al.d_slope.iloc[i+1],
                 'd_curve': al.d_curve.iloc[i+1]})
m = pd.DataFrame(recs).dropna()
print('months:', len(m), m.ret_month.min(), '..', m.ret_month.max())

pan = pd.read_csv(PANEL).drop_duplicates('ret_month')[
    ['ret_month','d_level','d_slope','d_curve']]
chk = m.merge(pan, on='ret_month', suffixes=('_mine','_panel'))
print('cross-check rows:', len(chk))
ok = True
for leg in ['d_level','d_slope','d_curve']:
    d = (chk[leg+'_mine'] - chk[leg+'_panel']).abs().max()
    print('  max abs diff %-8s: %.3e' % (leg, d))
    if d > 1e-9:
        ok = False
if not ok:
    print('  -> shocks DIFFER from panel; alignment convention not reproduced')
else:
    print('  -> shocks reproduce panel exactly from raw daily file')

X = np.column_stack([np.ones(len(m)), m.d_level, m.d_slope, m.d_curve])
co, _, _, _ = np.linalg.lstsq(X, m.excess.values, rcond=None)
dfit, dana = -100.0 * co[1], m.D_closed.mean()
print()
print('D_fit  (independent) : %.3f' % dfit)
print('D_closed-form        : %.3f' % dana)
print('ratio                : %.4f' % (dfit / dana))
print('first script         : D_fit 4.531  D_analytic 4.657  ratio 0.9729')
print('difference in D_fit  : %+.4f' % (dfit - 4.531))
print('reproduces within 0.02y' if abs(dfit - 4.531) < 0.02
      else 'DOES NOT reproduce -- reconcile before reporting either number')
