"""D_new/D_old at FROZEN CPR: only the curve construction changes.

The old bootstrap fails par repricing by up to 3.34pt at 20yr on real curves
(zero-rate interpolation plus flat extrapolation into the unsolved gap, and a
short-end convention error). bootstrap_v3 solves each node by root-find and
reprices par to 2.4e-13. Question here: how much does that move DURATIONS?

If the ratio is ~1.33 and flat across coupons, the bootstrap defect is the
source of the under-sizing and the correction is DERIVED, not fitted. If ~1.00,
the bug is real but is not the 1.33.

Frozen CPR throughout -- no hazard model, no re-forecast, no returns. Cannot be
circular with the regression that produced 1.33.
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import model_hedge_krd as M
from bootstrap_v3 import bootstrap_zeros_v3

LAB, YRS = M.MAT_LABELS, M.MAT_YEARS
h = M.BUMP_BP / 100.0
COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

def dur(coupon, par, cpr360, w, bs):
    p0 = M.price_path(coupon, cpr360, bs(par))
    px = {}
    for sgn in (+1, -1):
        bp = {l: float(par[l]) + sgn * h * wi for l, wi in zip(LAB, w)}
        px[sgn] = M.price_path(coupon, cpr360, bs(bp))
    return (px[-1] - px[+1]) / (2.0 * p0 * (h / 100.0))

d = pd.read_csv(os.path.join("data", "treasury_yields.csv")).dropna(subset=LAB)
d['DATE'] = pd.to_datetime(d['DATE'])
me = d.groupby(d['DATE'].dt.to_period('M')).last()
me = me[me.index >= '2018-01']
print("month-end curves: %d  (%s .. %s)" % (len(me), me.index[0], me.index[-1]))

old = lambda p: M.bootstrap_zeros(p)
new = lambda p: bootstrap_zeros_v3(p)
w_par = np.ones(len(LAB))
w10 = M.key_rate_weights3('10yr')

for cpr_lvl in (0.06, 0.12, 0.20):
    print("\n" + "=" * 68)
    print("FROZEN CPR = %.2f" % cpr_lvl)
    print("=" * 68)
    print("%5s%11s%11s%9s | %10s%10s%9s" % (
        "cpn", "Dlvl_old", "Dlvl_new", "ratio", "K10_old", "K10_new", "ratio"))
    ratios = []
    for c in COUPONS:
        z = np.full(M.N_MONTHS, cpr_lvl)
        ro = rn = ko = kn = 0.0
        n = 0
        for _, r in me.iterrows():
            par = {l: float(r[l]) for l in LAB}
            ro += dur(c, par, z, w_par, old)
            rn += dur(c, par, z, w_par, new)
            ko += dur(c, par, z, w10, old)
            kn += dur(c, par, z, w10, new)
            n += 1
        ro, rn, ko, kn = ro/n, rn/n, ko/n, kn/n
        ratios.append(rn/ro)
        print("%5.1f%11.4f%11.4f%9.4f | %10.4f%10.4f%9.4f" % (
            c, ro, rn, rn/ro, ko, kn, kn/ko if abs(ko) > 1e-9 else np.nan))
    print("  D_level ratio: mean %.4f  min %.4f  max %.4f  spread %.4f" % (
        np.mean(ratios), min(ratios), max(ratios), max(ratios)-min(ratios)))

print("\nratio ~1.33 and flat -> bootstrap defect IS the under-sizing")
print("ratio ~1.00          -> bootstrap bug real but not the 1.33")
