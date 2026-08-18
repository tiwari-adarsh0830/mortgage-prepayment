"""Par->zero bootstrap solved as a root-find per node.

Why v2 failed: when solving node T, the coupon dates in (T_prev, T) need
discount factors that depend on node T itself. Both the original (flat
extrapolation of the last solved zero) and v2 (forward extrapolation) guess
those, so the solved node absorbs the guess error. Between 10 and 20yr that is
19 guessed coupon PVs.

Fix: treat the instantaneous forward over (T_prev, T] as the unknown and solve
for the value that prices the par bond to exactly 100. Coupons inside the gap
are then priced with a forward consistent with the node being solved -- self
-consistent by construction, so par repricing is exact to solver tolerance.

Units match bootstrap_zeros: par dict in percent keyed by MAT_LABELS; returns
360 monthly continuously-compounded zeros in percent.
"""
import numpy as np, sys, os
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

LAB, YRS = M.MAT_LABELS, M.MAT_YEARS
N = M.N_MONTHS

def bootstrap_zeros_v3(par, n_months=N, short_convert=True):
    nodes = {0.0: 0.0}                      # T -> -ln(DF(T))
    for T, lab in zip(YRS, LAB):
        if T <= 0.5:
            # genuine zero-coupon quotes (1mo, 3mo, 6mo bills): money-market
            # simple yield -> continuous
            r = float(par[lab]) / 100.0
            nodes[T] = np.log(1.0 + r * T) if short_convert else r * T

    def nld(T, kt, extra=None):
        """-ln(DF(T)) with piecewise-linear interpolation on the solved nodes.
        `extra` = (T_new, val_new) provisionally appended, so coupons inside the
        gap see the node being solved."""
        ks = list(kt)
        vals = dict(nodes)
        if extra is not None:
            ks = ks + [extra[0]]
            vals[extra[0]] = extra[1]
            ks = sorted(ks)
        if T <= ks[0]:
            return 0.0
        if T >= ks[-1]:
            k1, k0 = ks[-1], ks[-2]
            f = (vals[k1] - vals[k0]) / (k1 - k0)
            return vals[k1] + f * (T - k1)
        for i in range(1, len(ks)):
            if T <= ks[i]:
                k0, k1 = ks[i - 1], ks[i]
                w = (T - k0) / (k1 - k0)
                return vals[k0] + w * (vals[k1] - vals[k0])

    for T, lab in zip(YRS, LAB):
        if T <= 0.5:
            continue
        c = float(par[lab]) / 100.0
        ts = np.arange(0.5, T + 1e-9, 0.5)
        kt = sorted(nodes)

        def resid(x):
            pv = 0.0
            for t in ts:
                d = np.exp(-nld(t, kt, extra=(T, x)))
                pv += (c / 2) * 100 * d
            pv += 100.0 * np.exp(-x)
            return pv - 100.0

        # resid is monotonically DECREASING in x (= -ln DF(T)).
        # x may be negative: a downward-bumped curve in a near-zero-rate month
        # legitimately implies negative zero rates. Expand the bracket until
        # the sign change is captured rather than assuming a fixed range.
        lo, hi = -2.0 * max(T, 1.0), 5.0 * max(T, 1.0)
        f_lo, f_hi = resid(lo), resid(hi)
        it = 0
        while f_lo * f_hi > 0 and it < 60:
            lo *= 1.5
            hi *= 1.5
            f_lo, f_hi = resid(lo), resid(hi)
            it += 1
        if f_lo * f_hi > 0:
            raise ValueError(
                "no sign change for T=%s after %d expansions "
                "(resid(%.3f)=%.3e, resid(%.3f)=%.3e)" % (T, it, lo, f_lo, hi, f_hi))
        nodes[T] = brentq(resid, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=300)

    kt = sorted(nodes)
    return np.array([nld(m / 12.0, kt) / (m / 12.0) * 100.0
                     for m in range(1, n_months + 1)])

def price_par_bond(T, cpn_pct, zgrid):
    tt = np.arange(1, len(zgrid) + 1) / 12.0
    z = lambda t: float(np.interp(t, tt, zgrid))
    ts = np.arange(0.5, T + 1e-9, 0.5)
    pv = sum((cpn_pct / 2) * np.exp(-z(t) / 100.0 * t) for t in ts)
    return pv + 100.0 * np.exp(-z(T) / 100.0 * T)

if __name__ == "__main__":
    import pandas as pd
    d = pd.read_csv(os.path.join("data", "treasury_yields.csv")).dropna(subset=LAB)
    d['DATE'] = pd.to_datetime(d['DATE'])
    me = d.groupby(d['DATE'].dt.to_period('M')).last()
    print("month-end curves: %d" % len(me))
    wo = wn = 0.0; won = wnn = None
    for i in range(len(me)):
        r = me.iloc[i]
        par = {l: float(r[l]) for l in LAB}
        zo = M.bootstrap_zeros(par)
        zn = bootstrap_zeros_v3(par)
        for T, lab in zip(YRS, LAB):
            if T < 1.0:
                continue
            eo = abs(price_par_bond(T, par[lab], zo) - 100.0)
            en = abs(price_par_bond(T, par[lab], zn) - 100.0)
            if eo > wo: wo, won = eo, (lab, str(me.index[i]))
            if en > wn: wn, wnn = en, (lab, str(me.index[i]))
    print("\nPAR REPRICING, %d month-end curves x 8 nodes" % len(me))
    print("  OLD  max err = %.4e   at %s" % (wo, won))
    print("  NEW  max err = %.4e   at %s" % (wn, wnn))
    ok = wn < 1e-6
    print("\n  VERDICT: %s" % ("PASS - bootstrap self-consistent"
                               if ok else "FAIL - do not proceed"))
    sys.exit(0 if ok else 1)
