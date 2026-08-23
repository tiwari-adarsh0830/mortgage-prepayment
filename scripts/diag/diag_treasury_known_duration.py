
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')
TRE   = os.path.join(DATA, 'treasury_yields_clean.xlsx')

def ols(y, X):
    XtX = X.T @ X
    co  = np.linalg.solve(XtX, X.T @ y)
    r   = y - X @ co
    dof = len(y) - X.shape[1]
    s2  = (r @ r) / dof
    se  = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    return co, co / se

def bond_price(y_pct, c_pct, times):
    yd = y_pct / 100.0
    cd = c_pct / 100.0
    df = (1.0 + yd / 2.0) ** (-2.0 * times)
    return (cd * 100.0 / 2.0) * df.sum() + 100.0 * df[-1]

def mod_duration(y_pct, c_pct, times, h=0.01):
    up = bond_price(y_pct + h, c_pct, times)
    dn = bond_price(y_pct - h, c_pct, times)
    p0 = bond_price(y_pct, c_pct, times)
    return -(up - dn) / (2.0 * p0 * (h / 100.0))

pan = pd.read_csv(PANEL)
pan['info_date'] = pd.to_datetime(pan['info_date'])
shock = pan[['ret_month','info_date','d_level','d_slope','d_curve','income']].drop_duplicates('ret_month')
shock = shock.sort_values('info_date').reset_index(drop=True)

tr = pd.read_excel(TRE, sheet_name='Treasury_Yields', header=1)
tr.columns = [str(c).strip() for c in tr.columns]
tr['Date'] = pd.to_datetime(tr['Date'])
tr = tr.sort_values('Date').reset_index(drop=True)
col5 = [c for c in tr.columns if '5yr' in c.lower() and '1' not in c.lower().replace('5yr','')][0]
print('using 5yr column:', col5)
tr = tr[['Date', col5]].dropna().rename(columns={col5: 'y5'})

T = np.arange(1, 11) * 0.5

rows = []
for i in range(1, len(tr)):
    d0, d1 = tr['Date'].iloc[i-1], tr['Date'].iloc[i]
    y0, y1 = float(tr['y5'].iloc[i-1]), float(tr['y5'].iloc[i])
    p1 = bond_price(y1, y0, T - 1.0/12.0)
    tot = (p1 + y0/12.0 - 100.0) / 100.0
    rows.append({'info_date': d0,
                 'ret_month': d1.strftime('%Y-%m'),
                 'y0': y0,
                 'ust5_total_return': tot,
                 'ust5_carry': y0/12.0/100.0,
                 'D_analytic': mod_duration(y0, y0, T)})
ust = pd.DataFrame(rows)

m = shock.merge(ust[['ret_month','ust5_total_return','ust5_carry','D_analytic','y0']],
                on='ret_month', how='inner')
print('merged months:', len(m), m['ret_month'].min(), '..', m['ret_month'].max())
if len(m) < 60:
    raise SystemExit('ABORT: merge too small, check ret_month alignment')

X = np.column_stack([np.ones(len(m)), m['d_level'].values,
                     m['d_slope'].values, m['d_curve'].values])

print()
print('=== CONTROL: TBA coupons through this harness ===')
print('cpn   D_fit    D_model   ratio')
ratios = []
for cpn, g in pan.groupby('coupon'):
    g = g.sort_values('info_date')
    gm = g[g['ret_month'].isin(m['ret_month'])]
    Xc = np.column_stack([np.ones(len(gm)), gm['d_level'].values,
                          gm['d_slope'].values, gm['d_curve'].values])
    yc = (gm['tba_total_return'] - gm['income']).values
    co, tt = ols(yc, Xc)
    dfit = -100.0 * co[1]
    dmod = gm['D_level'].mean()
    ratios.append(dfit / dmod)
    print('%.1f  %7.3f  %7.3f  %6.3f' % (cpn, dfit, dmod, dfit/dmod))
print('median control ratio: %.3f' % np.median(ratios))
print('(established value is ~1.33-1.36; if this is far off, discard the run)')

print()
print('=== TREASURY 5yr, same shocks, same regression ===')
yv = (m['ust5_total_return'] - m['ust5_carry']).values
co, tt = ols(yv, X)
dfit = -100.0 * co[1]
dana = m['D_analytic'].mean()
print('D_fit (regression)      : %.3f' % dfit)
print('D_analytic (closed form): %.3f' % dana)
print('ratio D_fit / D_analytic: %.4f' % (dfit / dana))
print('t(d_level)=%.2f  t(d_slope)=%.2f  t(d_curve)=%.2f' % (tt[1], tt[2], tt[3]))
print('mean 5yr yield %.3f, D_analytic range %.3f..%.3f'
      % (m['y0'].mean(), m['D_analytic'].min(), m['D_analytic'].max()))

print()
print('d_level distinct values (rounding check):', sorted(set(m['d_level'].values))[:8])
pd.DataFrame({'ret_month': m['ret_month'], 'ust5_ret': m['ust5_total_return'],
              'D_analytic': m['D_analytic']}).to_csv(
    os.path.join(OUT, 'diag_treasury_known_duration.csv'), index=False)
