
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

right = dly[['2yr','5yr','10yr']].reset_index()
right.columns = ['Date','y2','y5','y10']
al = pd.merge_asof(pd.DataFrame({'Date': cln['Date']}).sort_values('Date'),
                   right.sort_values('Date'), on='Date', direction='backward')
al['d_level'] = ((al.y2 + al.y5 + al.y10)/3.0).diff()
al['d_slope'] = (al.y10 - al.y2).diff()
al['d_curve'] = (2.0*al.y5 - al.y2 - al.y10).diff()
al['ret_month'] = al['Date'].shift(-1).dt.strftime('%Y-%m')

def price_loop(y_pct, cpn_pct, times):
    y = y_pct/100.0/2.0; c = cpn_pct/2.0; p = 0.0
    for i, t in enumerate(times):
        cf = c + (100.0 if i == len(times)-1 else 0.0)
        p += cf/(1.0+y)**(2.0*t)
    return p

def mod_dur_closed(y_pct, cpn_pct, times):
    y = y_pct/100.0/2.0; c = cpn_pct/2.0; p = w = 0.0
    for i, t in enumerate(times):
        cf = c + (100.0 if i == len(times)-1 else 0.0)
        pv = cf/(1.0+y)**(2.0*t); p += pv; w += t*pv
    return (w/p)/(1.0+y)

pan_months = set(pd.read_csv(PANEL)['ret_month'].unique())

print('tenor  n   D_fit   D_closed  ratio   t_lvl    t_slp    t_crv')
res = {}
for label, ycol, yrs in [('2yr','y2',2), ('5yr','y5',5), ('10yr','y10',10)]:
    T     = [0.5*k for k in range(1, 2*yrs+1)]
    Taged = [t - 1.0/12.0 for t in T]
    recs = []
    for i in range(len(al)-1):
        rm = al.ret_month.iloc[i]
        y0, y1 = al[ycol].iloc[i], al[ycol].iloc[i+1]
        if not isinstance(rm, str) or rm not in pan_months or pd.isna(y0) or pd.isna(y1):
            continue
        y0, y1 = float(y0), float(y1)
        p1 = price_loop(y1, y0, Taged)
        recs.append({'excess': (p1 + y0/12.0 - 100.0)/100.0 - y0/12.0/100.0,
                     'D_closed': mod_dur_closed(y0, y0, T),
                     'D_aged': mod_dur_closed(y0, y0, Taged),
                     'd_level': al.d_level.iloc[i+1],
                     'd_slope': al.d_slope.iloc[i+1],
                     'd_curve': al.d_curve.iloc[i+1]})
    m = pd.DataFrame(recs).dropna()
    X = np.column_stack([np.ones(len(m)), m.d_level, m.d_slope, m.d_curve])
    XtX = X.T @ X
    co  = np.linalg.solve(XtX, X.T @ m.excess.values)
    r   = m.excess.values - X @ co
    s2  = (r @ r)/(len(m)-X.shape[1])
    se  = np.sqrt(np.diag(s2*np.linalg.inv(XtX)))
    tt  = co/se
    dfit, dana = -100.0*co[1], m.D_closed.mean()
    dagd = m.D_aged.mean()
    res[label] = dfit/dana
    print('        aged-maturity D %.3f -> ratio %.4f' % (dagd, dfit/dagd))
    print('%-5s %3d %7.3f %8.3f %7.4f %8.2f %8.2f %8.2f'
          % (label, len(m), dfit, dana, dfit/dana, tt[1], tt[2], tt[3]))

print()
print('5yr from prior verified run: D_fit 4.520 ratio 0.9710 (should match above)')
sp = max(res.values()) - min(res.values())
print('ratio spread across tenors: %.4f' % sp)
print('flat across maturity -> harness property, quotable as such'
      if sp < 0.03 else
      'maturity-dependent -> shortfall is a convention effect, needs naming')
