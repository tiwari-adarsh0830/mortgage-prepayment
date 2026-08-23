
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

pan = pd.read_csv(PANEL)
pan['info_date'] = pd.to_datetime(pan['info_date'])
u = pan.drop_duplicates('ret_month')[['ret_month','info_date','pmms',
                                      'd_level','d_slope','d_curve']]
u = u.sort_values('info_date').reset_index(drop=True)

dly = pd.read_csv(os.path.join(DATA,'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()

# Explicit start/end dates per return month, no shift() anywhere.
u['d_start'] = u['info_date']
u['d_end']   = u['info_date'].shift(-1)
u = u.dropna(subset=['d_end']).reset_index(drop=True)

def asof(dt, col):
    ix = dly.index[dly.index <= dt]
    return float(dly.loc[ix[-1], col]) if len(ix) else np.nan

for c in ['2yr','5yr','10yr']:
    u[c+'_s'] = [asof(d,c) for d in u['d_start']]
    u[c+'_e'] = [asof(d,c) for d in u['d_end']]

# ASSERTION 1: rebuild d_level from the same two dates. If the window is the
# one the panel used, this must match the panel column exactly.
u['dlvl_check'] = ((u['2yr_e']+u['5yr_e']+u['10yr_e'])/3.0
                 - (u['2yr_s']+u['5yr_s']+u['10yr_s'])/3.0)
u['dlvl_panel'] = u['d_level']
v = u.dropna(subset=['dlvl_panel','dlvl_check'])
gap = (v['dlvl_check'] - v['dlvl_panel']).abs().max()
print('ASSERT 1  max |rebuilt d_level - panel d_level| = %.3e' % gap)
if gap > 1e-9:
    raise SystemExit('ABORT: window does not match panel; spread would be misaligned')
print('          -> spread will be measured over the SAME window as d_level')

# PMMS at start and end of the SAME window
pm = pd.read_csv(os.path.join(DATA,'pmms_monthly.csv'))
def parse(x):
    s = str(int(x))
    if len(s)==5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]),  day=1)
    if len(s)==6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm['k'] = pm['reporting_period'].apply(parse)
pmap = dict(zip(pm.dropna(subset=['k'])['k'], pm['rate_30yr']))
u['pm_s'] = [pmap.get(pd.Timestamp(d.year,d.month,1), np.nan) for d in u['d_start']]
u['pm_e'] = [pmap.get(pd.Timestamp(d.year,d.month,1), np.nan) for d in u['d_end']]

# ASSERTION 2: start-of-window PMMS is the panel's own column
g2 = (u['pm_s'] - u['pmms']).abs().max()
print('ASSERT 2  max |pm_start - panel pmms| = %.3e' % g2)
if g2 > 1e-9:
    raise SystemExit('ABORT: PMMS keying differs from panel')

u['d_spread'] = (u['pm_e'] - u['10yr_e']) - (u['pm_s'] - u['10yr_s'])
sp = u[['ret_month','d_spread']].dropna()

m = pan.merge(sp, on='ret_month', how='inner')
print()
print('months: %d   corr(d_level,d_spread) = %+.4f'
      % (m['ret_month'].nunique(), np.corrcoef(m.d_level, m.d_spread)[0,1]))

print()
print('cpn   ratio3   ratio_spr   delta    t_spr')
a3, a4 = [], []
for cpn, g in m.groupby('coupon'):
    y  = (g['tba_total_return'] - g['income']).values
    dm = g['D_level'].mean()
    X3 = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve])
    X4 = np.column_stack([X3, g.d_spread.values])
    c3,*_ = np.linalg.lstsq(X3, y, rcond=None)
    c4,*_ = np.linalg.lstsq(X4, y, rcond=None)
    r  = y - X4 @ c4
    s2 = (r@r)/(len(y)-X4.shape[1])
    se = np.sqrt(np.diag(s2*np.linalg.pinv(X4.T@X4)))
    a, b = -100.0*c3[1]/dm, -100.0*c4[1]/dm
    a3.append(a); a4.append(b)
    print('%.1f %8.3f %10.3f %8.3f %8.2f' % (cpn,a,b,b-a,c4[4]/se[4]))

m3, m4 = float(np.median(a3)), float(np.median(a4))
print()
print('median ratio3    : %.3f' % m3)
print('median ratio_spr : %.3f' % m4)
print('change           : %+.3f   (alignment run gave +0.031)' % (m4-m3))
print('CONFIRMS: spread control widens the gap' if m4 > m3 + 0.01
      else 'DOES NOT confirm -- do not report')
