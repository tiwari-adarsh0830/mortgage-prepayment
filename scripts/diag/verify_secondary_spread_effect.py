"""
Step (3) of the advisor's Aug 27 task: swap the SECONDARY market spread in as
the control, in place of PMMS-10yr, and see whether the 1.33-1.36 ratio moves
toward 1.0.

This is verify_spread_effect_v2.py with ONE substantive change: d_spread is
built from (current coupon - 10yr) instead of (PMMS - 10yr). Panel, window,
regression, ratio definition and the median-across-coupons summary are
unchanged, so any movement is attributable to the spread definition.

GUARDS -- note the asymmetry with v2, do not overstate it:
  ASSERT 1 is KEPT verbatim: d_level is rebuilt from the two window endpoints
    and must match the panel to 1e-9. Same guarantee as v2.
  ASSERT 2 CANNOT be carried over. v2 checked start-of-window PMMS against the
    panel's own pmms column. The secondary spread has no panel counterpart, so
    there is nothing to assert against. It is REPLACED by a coverage guard:
    the current coupon must resolve at BOTH endpoints of every retained month,
    and the number of months lost is printed. This is a weaker guarantee than
    v2 had and should be described as such.

The current coupon is interpolated between the two FNCL coupons that BRACKET
par. It is not defined in most of 2020-2021 (every quoted coupon above par,
and the 2.5/3.0 slope is flat-to-inverted there, so extrapolation is not
viable -- it produces implied coupons from -8.09 to +22.06). Those months are
DROPPED, not extrapolated. The sample is therefore smaller than v2's and
excludes the QE window; the PMMS arm is re-run on the SAME reduced sample so
the comparison is like-for-like.
"""
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

u['d_start'] = u['info_date']
u['d_end']   = u['info_date'].shift(-1)
u = u.dropna(subset=['d_end']).reset_index(drop=True)

def asof(dt, col):
    ix = dly.index[dly.index <= dt]
    return float(dly.loc[ix[-1], col]) if len(ix) else np.nan

for c in ['2yr','5yr','10yr']:
    u[c+'_s'] = [asof(d,c) for d in u['d_start']]
    u[c+'_e'] = [asof(d,c) for d in u['d_end']]

# ---- ASSERT 1: kept verbatim from v2 ---------------------------------------
u['dlvl_check'] = ((u['2yr_e']+u['5yr_e']+u['10yr_e'])/3.0
                 - (u['2yr_s']+u['5yr_s']+u['10yr_s'])/3.0)
v = u.dropna(subset=['d_level','dlvl_check'])
gap = (v['dlvl_check'] - v['d_level']).abs().max()
print('ASSERT 1  max |rebuilt d_level - panel d_level| = %.3e' % gap)
if gap > 1e-9:
    raise SystemExit('ABORT: window does not match panel; spread would be misaligned')
print('          -> spread measured over the SAME window as d_level')

# ---- current coupon, bracketing par ----------------------------------------
raw = pd.read_excel(os.path.join(DATA,'fncl_tba_prices_clean.xlsx'), skiprows=1)
raw.columns = ['Date'] + list(raw.columns[1:])
hdr = pd.read_excel(os.path.join(DATA,'fncl_tba_prices_clean.xlsx'), nrows=1).iloc[0].tolist()
coupons = np.array([float(str(h).replace('FNCL','').strip()) for h in hdr[1:]])
raw['Date'] = pd.to_datetime(raw['Date'])
px = raw.iloc[:,1:].apply(pd.to_numeric, errors='coerce').values

cc = []
for i in range(len(raw)):
    p = px[i]; ok = ~np.isnan(p)
    c_i, p_i = coupons[ok], p[ok]
    hit = np.nan
    for j in range(len(c_i)-1):
        if (p_i[j]-100.0)*(p_i[j+1]-100.0) <= 0:
            w = (100.0-p_i[j])/(p_i[j+1]-p_i[j])
            hit = c_i[j] + w*(c_i[j+1]-c_i[j]); break
    cc.append(hit)
ccs = pd.Series(cc, index=raw['Date']).dropna().sort_index()
print('current coupon defined in %d of %d months' % (len(ccs), len(raw)))

def cc_asof(dt):
    ix = ccs.index[ccs.index <= dt]
    return float(ccs.loc[ix[-1]]) if len(ix) else np.nan

# strict: require an EXACT month-end match at both endpoints, no stale carry
ccset = {pd.Timestamp(d).normalize(): float(x) for d, x in ccs.items()}
u['cc_s'] = [ccset.get(pd.Timestamp(d).normalize(), np.nan) for d in u['d_start']]
u['cc_e'] = [ccset.get(pd.Timestamp(d).normalize(), np.nan) for d in u['d_end']]

# ---- REPLACEMENT GUARD (weaker than v2's ASSERT 2) -------------------------
n_all = len(u)
u_ok = u.dropna(subset=['cc_s','cc_e']).copy()
print('GUARD (coverage)  months with current coupon at BOTH endpoints: %d of %d'
      % (len(u_ok), n_all))
print('                  NOTE: this is a coverage check, NOT an equality check')
print('                  against a panel column. Weaker than v2 ASSERT 2.')
if len(u_ok) < 40:
    raise SystemExit('ABORT: too few months to fit 9 coupons reliably')

# ---- PMMS on the SAME reduced sample, for like-for-like ---------------------
pm = pd.read_csv(os.path.join(DATA,'pmms_monthly.csv'))
def parse(x):
    s = str(int(x))
    if len(s)==5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]),  day=1)
    if len(s)==6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm['k'] = pm['reporting_period'].apply(parse)
pmap = dict(zip(pm.dropna(subset=['k'])['k'], pm['rate_30yr']))
u_ok['pm_s'] = [pmap.get(pd.Timestamp(d.year,d.month,1), np.nan) for d in u_ok['d_start']]
u_ok['pm_e'] = [pmap.get(pd.Timestamp(d.year,d.month,1), np.nan) for d in u_ok['d_end']]

g2 = (u_ok['pm_s'] - u_ok['pmms']).abs().max()
print('ASSERT 2b max |pm_start - panel pmms| = %.3e  (PMMS arm only)' % g2)
if g2 > 1e-9:
    raise SystemExit('ABORT: PMMS keying differs from panel')

u_ok['d_spread_sec'] = (u_ok['cc_e'] - u_ok['10yr_e']) - (u_ok['cc_s'] - u_ok['10yr_s'])
u_ok['d_spread_pri'] = (u_ok['pm_e'] - u_ok['10yr_e']) - (u_ok['pm_s'] - u_ok['10yr_s'])

sp = u_ok[['ret_month','d_spread_sec','d_spread_pri']].dropna()
m = pan.merge(sp, on='ret_month', how='inner')
print()
print('months: %d' % m['ret_month'].nunique())
print('corr(d_level, d_spread_sec) = %+.4f' % np.corrcoef(m.d_level, m.d_spread_sec)[0,1])
print('corr(d_level, d_spread_pri) = %+.4f' % np.corrcoef(m.d_level, m.d_spread_pri)[0,1])

def run(col, label):
    print()
    print('=== control: %s ===' % label)
    print('cpn   ratio3   ratio_spr   delta    t_spr')
    a3, a4 = [], []
    for cpn, g in m.groupby('coupon'):
        y  = (g['tba_total_return'] - g['income']).values
        dm = g['D_level'].mean()
        X3 = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve])
        X4 = np.column_stack([X3, g[col].values])
        c3,*_ = np.linalg.lstsq(X3, y, rcond=None)
        c4,*_ = np.linalg.lstsq(X4, y, rcond=None)
        r  = y - X4 @ c4
        s2 = (r@r)/(len(y)-X4.shape[1])
        se = np.sqrt(np.diag(s2*np.linalg.pinv(X4.T@X4)))
        a, b = -100.0*c3[1]/dm, -100.0*c4[1]/dm
        a3.append(a); a4.append(b)
        print('%.1f %8.3f %10.3f %8.3f %8.2f' % (cpn,a,b,b-a,c4[4]/se[4]))
    m3, m4 = float(np.median(a3)), float(np.median(a4))
    print('median ratio3    : %.3f' % m3)
    print('median ratio_spr : %.3f' % m4)
    print('change           : %+.3f' % (m4-m3))
    return m3, m4

p3, p4 = run('d_spread_pri', 'PMMS - 10yr (Phase 28 baseline, reduced sample)')
s3, s4 = run('d_spread_sec', 'current coupon - 10yr (SECONDARY)')

print()
print('=== summary ===')
print('uncontrolled ratio (reduced sample) : %.3f' % s3)
print('with PMMS control                   : %.3f  (%+.3f)' % (p4, p4-p3))
print('with SECONDARY control              : %.3f  (%+.3f)' % (s4, s4-s3))
print()
if s4 < s3 - 0.01:
    print('Secondary control NARROWS the gap. Direction advisor predicted.')
    print('NOT a confirmed explanation -- see caveats in docstring (sample')
    print('excludes QE; channel is post-2022 only; effect is close to the')
    print('arithmetic already implied by beta=+0.30).')
elif s4 > s3 + 0.01:
    print('Secondary control WIDENS the gap -- same direction as PMMS. Report as is.')
else:
    print('Secondary control leaves the ratio essentially unchanged.')

m.to_csv(os.path.join(OUT,'verify_secondary_spread_effect_sample.csv'), index=False)
print()
print('wrote outputs/verify_secondary_spread_effect_sample.csv')
