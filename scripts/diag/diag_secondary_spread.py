"""
Advisor's Aug 27 diagnosis: Phase 28's spread test used PMMS, a PRIMARY
market rate. The TBA is discounted off the SECONDARY market spread, and the
two move differently -- that is why beta came out -0.4734 instead of large
and positive.

This is diag_spread_channel_sign.py with ONE thing changed: the spread
series. Panel, dedup, treasury source, merge direction, forward-window
convention and regression are all byte-identical to that script so the sign
comparison is clean and attributable to the spread definition alone.

Secondary spread here = current-coupon rate - 10yr Treasury, where the
current-coupon rate is the FNCL coupon that prices at par, interpolated
each month between the two coupons that BRACKET par.

Note on the interpolation: FNCL prices are NOT monotonic in coupon at the
premium end (2018-01: 6.0 = 111.33 but 6.5 = 109.61, payup compression).
Sorting by price would scramble coupon order. We therefore bracket par
locally and interpolate only between the adjacent pair straddling 100,
which over this sample sits at 3.0/3.5 early and 5.0/5.5 late -- inside the
monotonic region. Months where par is not bracketed are FLAGGED and dropped,
not extrapolated.
"""
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

# ---------- build current-coupon (secondary market) rate ----------
raw = pd.read_excel(os.path.join(DATA, 'fncl_tba_prices_clean.xlsx'), skiprows=1)
raw.columns = ['Date'] + [c for c in raw.columns[1:]]
hdr = pd.read_excel(os.path.join(DATA, 'fncl_tba_prices_clean.xlsx'), nrows=1).iloc[0].tolist()
coupons = np.array([float(str(h).replace('FNCL', '').strip()) for h in hdr[1:]])
print('coupons parsed from header row: %s' % coupons)

raw['Date'] = pd.to_datetime(raw['Date'])
px = raw.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
print('price matrix: %s   dates %s .. %s'
      % (px.shape, raw['Date'].min().date(), raw['Date'].max().date()))

cc, flag = [], []
for i in range(len(raw)):
    p = px[i]
    ok = ~np.isnan(p)
    c_i, p_i = coupons[ok], p[ok]
    # adjacent pairs straddling par, in COUPON order (never sorted by price)
    hit = None
    for j in range(len(c_i) - 1):
        if (p_i[j] - 100.0) * (p_i[j+1] - 100.0) <= 0:
            w = (100.0 - p_i[j]) / (p_i[j+1] - p_i[j])
            hit = c_i[j] + w * (c_i[j+1] - c_i[j])
            break
    cc.append(hit if hit is not None else np.nan)
    flag.append('OK' if hit is not None else 'PAR_NOT_BRACKETED')

ccdf = pd.DataFrame({'Date': raw['Date'], 'cc': cc, 'flag': flag})
nbad = (ccdf['flag'] != 'OK').sum()
print('current-coupon built: %d months, %d dropped (par not bracketed)'
      % (len(ccdf), nbad))
if nbad:
    print(ccdf[ccdf['flag'] != 'OK'].to_string())
print('cc range: %.3f .. %.3f' % (ccdf['cc'].min(), ccdf['cc'].max()))

# ---------- identical to diag_spread_channel_sign.py from here ----------
pan = pd.read_csv(PANEL)
pan['info_date'] = pd.to_datetime(pan['info_date'])
u = (pan.drop_duplicates('ret_month')[['ret_month','info_date','pmms','d_level']]
        .sort_values('info_date').reset_index(drop=True))

dly = pd.read_csv(os.path.join(DATA,'treasury_yields.csv'),
                  index_col=0, parse_dates=True).sort_index()
t10 = dly['10yr'].reset_index(); t10.columns = ['Date','y10']
u = pd.merge_asof(u, t10.sort_values('Date'),
                  left_on='info_date', right_on='Date', direction='backward')

u = pd.merge_asof(u.sort_values('info_date'),
                  ccdf[['Date','cc']].sort_values('Date'),
                  left_on='info_date', right_on='Date',
                  direction='backward', suffixes=('','_cc'))

# window: forward from info_date to the NEXT info_date (Phase 28 trap)
u['sp_sec'] = u['cc'] - u['y10']
u['sp_pri'] = u['pmms'] - u['y10']
u['d_spread']     = u['sp_sec'].shift(-1) - u['sp_sec']
u['d_spread_pri'] = u['sp_pri'].shift(-1) - u['sp_pri']

m = u.dropna(subset=['d_spread','d_level'])
X = np.column_stack([np.ones(len(m)), m.d_level.values])
co, *_ = np.linalg.lstsq(X, m.d_spread.values, rcond=None)
r  = m.d_spread.values - X @ co
s2 = (r @ r) / (len(m) - 2)
se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
beta, tb = co[1], co[1]/se[1]

print()
print('=== SECONDARY spread (current coupon - 10yr) ===')
print('months: %d' % len(m))
print('mean level of secondary spread = %+.4f' % m.sp_sec.mean())
print('mean level of primary  spread  = %+.4f' % m.sp_pri.mean())
print('beta (d_spread on d_level) = %+.4f   t = %+.2f' % (beta, tb))
print('corr(d_level, d_spread)    = %+.4f'
      % np.corrcoef(m.d_level, m.d_spread)[0,1])
print('corr(d_sec, d_pri)         = %+.4f'
      % np.corrcoef(m.d_spread, m.d_spread_pri)[0,1])
print()
print('implied D_emp / D_cash = 1 + beta = %.4f' % (1.0 + beta))
print('observed ratio                    = 1.35')
print('beta required for 1.35            = +0.35')
print('Phase 28 PMMS result              = -0.4734')
print()
if beta < 0:
    print('SIGN: spread TIGHTENS as rates rise -> channel predicts empirical')
    print('duration BELOW cashflow. Same direction as the PMMS result.')
else:
    print('SIGN: spread WIDENS as rates rise -> channel predicts empirical ABOVE,')
    print('which is the advisor hypothesis. NOT confirmed until robustness below.')
print()
print('subsample check (channel stability):')
h = len(m)//2
for lbl, s in [('first half', m.iloc[:h]), ('second half', m.iloc[h:])]:
    Xs = np.column_stack([np.ones(len(s)), s.d_level.values])
    cs, *_ = np.linalg.lstsq(Xs, s.d_spread.values, rcond=None)
    print('  %-12s beta %+.4f  -> 1+beta %.3f  (n=%d)'
          % (lbl, cs[1], 1+cs[1], len(s)))

m.to_csv(os.path.join(OUT, 'diag_secondary_spread_sample.csv'), index=False)
print()
print('wrote outputs/diag_secondary_spread_sample.csv')
