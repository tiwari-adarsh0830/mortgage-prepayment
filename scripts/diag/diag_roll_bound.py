
"""
Can the omitted TBA roll/drop account for the duration under-sizing?

The panel builds TBA return as (P_curr + c/12 - P_prev)/P_prev. A TBA quote is
a forward price for a settlement month, so consecutive month-end quotes are
different contracts; the drop is the market price of that month of carry
(coupon minus financing). The formula adds c/12 and omits the financing leg,
so the residual carry term is missing.

That term is rate-sensitive -- cheaper financing relative to coupon widens the
drop -- so an omission of it can masquerade as duration, and its size scales
with coupon. This asks how large the drop's rate-sensitivity would have to be
to supply the missing duration, expressed as a multiple of the drop's own
level. A multiple far above 1 means the channel cannot physically do it.

LIMITATION, stated up front: tba_roll_snapshot.xlsx is a single point in time
(June 2026, nine coupons). This treats one month's drop level as
representative of the 99-month sample. It bounds plausibility; it is not a
measurement of the realised roll series. A real test needs a drop or repo
series over the panel window.
"""
import os
import numpy as np
import pandas as pd

BASE  = '/scratch/at7095/mortgage_prepayment'
DATA  = os.path.join(BASE, 'data')
OUT   = os.path.join(BASE, 'outputs')
PANEL = os.path.join(OUT, 'model_hedge_panel_10_tents3_pinnedfixed.csv')

roll = pd.read_excel(os.path.join(DATA, 'tba_roll_snapshot.xlsx'),
                     sheet_name='TBA_Roll_Snapshot', header=2)
roll.columns = [str(c).strip() for c in roll.columns]
ccol = [c for c in roll.columns if c.lower().startswith('coupon')][0]
fcol = [c for c in roll.columns if 'front' in c.lower()][0]
dcol = [c for c in roll.columns if 'drop'  in c.lower()][0]
roll = roll[[ccol, fcol, dcol]].dropna()
roll.columns = ['coupon', 'front_px', 'drop_pts']
roll['coupon'] = roll['coupon'].astype(float)
print('roll snapshot rows: %d, coupons %s'
      % (len(roll), sorted(roll.coupon.tolist())))

pan = pd.read_csv(PANEL)

def fit_ratio(g):
    y = (g['tba_total_return'] - g['income']).values
    X = np.column_stack([np.ones(len(g)), g.d_level, g.d_slope, g.d_curve])
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    return -100.0 * co[1] / g['D_level'].mean(), g['D_level'].mean()

rows = []
for c, g in pan.groupby('coupon'):
    r, dm = fit_ratio(g)
    rows.append({'coupon': c, 'ratio': r, 'D_model': dm})
fit = pd.DataFrame(rows)

m = fit.merge(roll, on='coupon', how='inner')
if len(m) != 9:
    raise SystemExit('ABORT: expected 9 coupons after merge, got %d' % len(m))

m['drop_pct_px'] = m.drop_pts.abs() / m.front_px * 100.0
m['gap_D']       = m.D_model * (m.ratio - 1.0)
m['ret_err_25bp']= m.gap_D * 0.25          # percent of price per 25bp
m['swing_mult']  = m.ret_err_25bp / m.drop_pct_px

print()
print('cpn  D_model  ratio  gap_D   drop%px  err@25bp  swing_needed')
for _, r in m.iterrows():
    print('%.1f %8.3f %6.3f %6.3f %9.4f %9.4f %12.1f'
          % (r.coupon, r.D_model, r.ratio, r.gap_D,
             r.drop_pct_px, r.ret_err_25bp, r.swing_mult))

lo = m[m.coupon <= 4.5]
hi = m[m.coupon >= 5.5]
print()
print('discount 2.5-4.5 : swing multiple %.1f - %.1f  -> implausible'
      % (lo.swing_mult.min(), lo.swing_mult.max()))
print('premium  5.5-6.5 : swing multiple %.1f - %.1f  -> within reach'
      % (hi.swing_mult.min(), hi.swing_mult.max()))
print()
print('Reading: a multiple of ~1 means the drop would have to vary by about its')
print('own level per 25bp, which drops do over a cycle. A multiple near 10 means')
print('it cannot. The roll is therefore ruled out where the gap is LARGEST')
print('(discounts) and remains a live partial candidate at premiums.')

m.to_csv(os.path.join(OUT, 'diag_roll_bound.csv'), index=False)
print()
print('wrote outputs/diag_roll_bound.csv')
