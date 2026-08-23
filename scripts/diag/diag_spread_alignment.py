
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

pan = pd.read_csv(PANEL)
pan['info_date'] = pd.to_datetime(pan['info_date'])
u = pan.drop_duplicates('ret_month')[['ret_month','info_date','pmms','d_level']]
u = u.sort_values('info_date').reset_index(drop=True)

dly = pd.read_csv(os.path.join(DATA,'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()
t10 = dly['10yr'].reset_index(); t10.columns=['Date','y10']
u = pd.merge_asof(u, t10.sort_values('Date'),
                  left_on='info_date', right_on='Date', direction='backward')

# PMMS is info-date keyed (start of return month). The 10yr on info_date is
# likewise start-of-return-month. d_level is the change ACROSS the return
# month, i.e. end minus start.
u['sp_info']  = u['pmms'] - u['y10']            # spread at start of ret month
u['A_diff_at_info'] = u['sp_info'].diff()       # change INTO the ret month
u['B_diff_fwd']     = u['sp_info'].shift(-1) - u['sp_info']  # change ACROSS
u['C_lagged']       = u['sp_info'].diff().shift(-1)

for lbl in ['A_diff_at_info','B_diff_fwd','C_lagged']:
    v = u[[lbl,'d_level']].dropna()
    print('%-16s corr(d_level, .) = %+.4f   n=%d'
          % (lbl, np.corrcoef(v['d_level'], v[lbl])[0,1], len(v)))

print()
print('B_diff_fwd is the one that matches d_level timing (both span the return')
print('month). A_diff_at_info spans the PREVIOUS month -> misaligned by one.')
print()
print('cpn   ratio3  +A      +B      +C')
for cpn, g0 in pan.groupby('coupon'):
    g = g0.merge(u[['ret_month','A_diff_at_info','B_diff_fwd','C_lagged']],
                 on='ret_month', how='inner').dropna(
                 subset=['A_diff_at_info','B_diff_fwd','C_lagged'])
    y  = (g['tba_total_return'] - g['income']).values
    dm = g['D_level'].mean()
    base = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve])
    out = []
    for col in [None,'A_diff_at_info','B_diff_fwd','C_lagged']:
        X = base if col is None else np.column_stack([base, g[col].values])
        co,*_ = np.linalg.lstsq(X, y, rcond=None)
        out.append(-100.0*co[1]/dm)
    print('%.1f %7.3f %7.3f %7.3f %7.3f' % (cpn,*out))
