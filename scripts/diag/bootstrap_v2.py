"""Corrected par->zero bootstrap.

The existing bootstrap_zeros interpolates the ZERO RATE linearly between solved
nodes when valuing intermediate coupon dates. Real curves are convex between
10/20/30yr, so those coupon PVs are wrong and the solved long zeros absorb the
error -- 1.4-1.8pt par-repricing failure at 20yr on every real curve, sign
flipping at 30yr. Flat synthetic curves hide it entirely (linear interp of a
constant is exact), which is why it survived.

Fix: interpolate on LOG DISCOUNT FACTORS, i.e. piecewise-constant instantaneous
forwards between nodes. Standard, arbitrage-free, and exact for the recursion
because each coupon PV then uses forwards consistent with the bracketing nodes.

Same signature and units as bootstrap_zeros: takes a par dict keyed by
MAT_LABELS in percent, returns 360 monthly continuously-compounded zeros in
percent. Short end converted annual->continuous (the original assigns par
directly, a ~12bp error at 5%).
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

LAB, YRS = M.MAT_LABELS, M.MAT_YEARS
N = M.N_MONTHS

def bootstrap_zeros_v2(par, n_months=N, short_convert=True):
    nodes = {0.0: 0.0}          # T -> -ln(DF), i.e. z*T ("integrated zero")
    for T, lab in zip(YRS, LAB):
        if T <= 1.0:
            r = float(par[lab]) / 100.0
            z = np.log(1.0 + r) if short_convert else r
            nodes[T] = z * T

    def neglogdf(T):
        """Piecewise-constant-forward interpolation on -ln(DF)."""
        kт = sorted(nodes)
        if T <= kт[0]:
            return 0.0
        if T >= kт[-1]:
            k1, k0 = kт[-1], kт[-2]
            f = (nodes[k1] - nodes[k0]) / (k1 - k0)
            return nodes[k1] + f * (T - k1)
        for i in range(1, len(kт)):
            if T <= kт[i]:
                k0, k1 = kт[i - 1], kт[i]
                w = (T - k0) / (k1 - k0)
                return nodes[k0] + w * (nodes[k1] - nodes[k0])

    def df(T):
        return np.exp(-neglogdf(T))

    for T, lab in zip(YRS, LAB):
        if T <= 1.0:
            continue
        c = float(par[lab]) / 100.0
        ts = np.arange(0.5, T - 1e-9, 0.5)
        pv = sum((c / 2) * 100 * df(t) for t in ts)
        dfT = (100 - pv) / (c / 2 * 100 + 100)
        if dfT <= 0:
            raise ValueError("bad DF at T=%s" % T)
        nodes[T] = -np.log(dfT)

    return np.array([neglogdf(m / 12.0) / (m / 12.0) * 100.0
                     for m in range(1, n_months + 1)])

def price_par_bond(T, cpn_pct, zgrid):
    """From-scratch semiannual pricer off a 360-month zero grid."""
    tt = np.arange(1, len(zgrid) + 1) / 12.0
    def z(t):
        return float(np.interp(t, tt, zgrid))
    ts = np.arange(0.5, T + 1e-9, 0.5)
    pv = sum((cpn_pct / 2) * np.exp(-z(t) / 100.0 * t) for t in ts)
    return pv + 100.0 * np.exp(-z(T) / 100.0 * T)

if __name__ == "__main__":
    import pandas as pd
    d = pd.read_csv(os.path.join("data", "treasury_yields.csv")).dropna(subset=LAB)
    print("curves: %d" % len(d))
    worst_old = worst_new = 0.0
    wo_node = wn_node = None
    for i in range(len(d)):
        r = d.iloc[i]
        par = {l: float(r[l]) for l in LAB}
        zo = M.bootstrap_zeros(par)
        zn = bootstrap_zeros_v2(par)
        for T, lab in zip(YRS, LAB):
            if T < 1.0:
                continue
            eo = abs(price_par_bond(T, par[lab], zo) - 100.0)
            en = abs(price_par_bond(T, par[lab], zn) - 100.0)
            if eo > worst_old:
                worst_old, wo_node = eo, (lab, str(r['DATE']))
            if en > worst_new:
                worst_new, wn_node = en, (lab, str(r['DATE']))
        if i % 1000 == 0:
            print("  ...%d" % i, flush=True)
    print("\nPAR REPRICING, all %d real curves x 8 nodes" % len(d))
    print("  OLD  max err = %.4e   at %s" % (worst_old, wo_node))
    print("  NEW  max err = %.4e   at %s" % (worst_new, wn_node))
    print("\n  NEW < 1e-9 -> corrected bootstrap is exact")
