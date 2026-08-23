
"""
Is the panel's TBA total return right?

The known-duration test covered the shocks and the regression but built the
Treasury return from yields, so it never touched the FNCL price series or the
return formula. The pricer uses (P_curr + c/12 - P_prev)/P_prev on month-end
TBA quotes. Three things could be wrong with that and none has been checked:

  1. It may disagree with the workbook's own Raw_MoM_Returns sheet.
  2. Adding c/12 to a forward price is a carry convention, not obviously the
     right one for a TBA.
  3. If the error correlates with the rate shock it inflates fitted duration,
     which is the direction of the observed gap.

This compares the two, and regresses any difference on the level shock. A
difference that is orthogonal to rates cannot produce a duration error; one
that loads on d_level can, and this quantifies how much.
"""
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed_srcdaily.csv')
XL    = os.path.join(DATA, 'fncl_tba_prices_clean.xlsx')

pan = pd.read_csv(PANEL)

bb = pd.read_excel(XL, sheet_name='Raw_MoM_Returns', header=1)
bb.columns = [str(c).strip() for c in bb.columns]
bb['Date'] = pd.to_datetime(bb['Date'])
bb = bb.sort_values('Date')
bb['ret_month'] = bb['Date'].dt.strftime('%Y-%m')

px = pd.read_excel(XL, sheet_name='Last_Price_Decimal', header=1)
px.columns = [str(c).strip() for c in px.columns]
px['Date'] = pd.to_datetime(px['Date'])
px = px.sort_values('Date').reset_index(drop=True)

rows = []
for c in sorted(pan.coupon.unique()):
    col = 'FNCL %.1f' % c
    if col not in bb.columns: col = 'FNCL %g' % c
    if col not in bb.columns:
        print('WARNING: %s not in Raw_MoM_Returns' % col); continue
    b = bb[['ret_month', col]].rename(columns={col: 'bb_ret'})
    p = px[['Date', col]].copy()
    p['prev'] = p[col].shift(1)
    p['ret_month'] = p['Date'].dt.strftime('%Y-%m')
    p['px_only'] = (p[col] - p['prev']) / p['prev']          # no coupon
    p['with_cpn'] = (p[col] + c/12.0 - p['prev']) / p['prev'] # pricer formula
    g = (pan[pan.coupon == c][['ret_month','tba_total_return','d_level','income']]
         .merge(b, on='ret_month').merge(
             p[['ret_month','px_only','with_cpn']], on='ret_month').dropna())
    rows.append((c, g))

print('cpn   n   max|panel-formula|  corr(panel,bb)  mean(panel-bb)bp  t(diff~d_level)  implied_D_err')
for c, g in rows:
    d_form = (g.tba_total_return - g.with_cpn).abs().max()
    cr     = np.corrcoef(g.tba_total_return, g.bb_ret)[0,1]
    diff   = g.tba_total_return - g.bb_ret
    X = np.column_stack([np.ones(len(g)), g.d_level.values])
    co, *_ = np.linalg.lstsq(X, diff.values, rcond=None)
    r  = diff.values - X @ co
    s2 = (r @ r) / (len(g) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    print('%.1f %4d %18.2e %15.4f %17.2f %16.2f %14.3f'
          % (c, len(g), d_form, cr, diff.mean()*1e4, co[1]/se[1], -100.0*co[1]))

print()
print('Reading: max|panel-formula| ~0 confirms the panel uses the stated formula.')
print('corr(panel,bb) near 1 with a small mean gap means the workbook return is')
print('the same object. implied_D_err is the duration, in years, that the')
print('difference between the two would contribute if it loaded on the level')
print('shock -- compare against the ~1.9y gap at coupon 2.5.')
