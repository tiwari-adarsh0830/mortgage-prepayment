"""Which convention should the 1mo/3mo/6mo FRED quotes be read under?

Cannot be settled by "reproducing the quote" -- the quote is the input, so every
convention reproduces it trivially. Two indirect tests instead:

T1 CONTINUITY AT 1YR. The 1yr node is solved by root-find under a semiannual
   bond convention (that path reprices par to 1e-13, so it is trusted). The
   bills are ASSIGNED under whatever convention we choose. A wrong choice shows
   up as a kink in the forward rate across the 6mo->1yr boundary. Different code
   paths on either side, so this is a real test, not a tautology.

T2 DURATION IMPACT. Whether the choice moves pass-through durations at all.

Candidates:
  simple    z*T = ln(1 + r*T)          money-market / simple interest
  disc360   z*T = -ln(1 - r*T*360/365) true discount-basis bill
  bey       z*T = 2*ln(1 + r/2)*T      bond-equivalent, semiannual
  cont      z*T = r*T                  the original's implicit choice
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import model_hedge_krd as M
from scipy.optimize import brentq

LAB, YRS = M.MAT_LABELS, M.MAT_YEARS
N = M.N_MONTHS

CONV = {
    'simple':  lambda r, T: np.log(1.0 + r * T),
    'disc360': lambda r, T: -np.log(1.0 - r * T * 360.0 / 365.0),
    'bey':     lambda r, T: 2.0 * np.log(1.0 + r / 2.0) * T,
    'cont':    lambda r, T: r * T,
}

def boot(par, conv, n_months=N):
    nodes = {0.0: 0.0}
    for T, lab in zip(YRS, LAB):
        if T <= 0.5:
            nodes[T] = CONV[conv](float(par[lab]) / 100.0, T)

    def nld(T, kt, extra=None):
        ks, vals = list(kt), dict(nodes)
        if extra is not None:
            ks = sorted(ks + [extra[0]]); vals[extra[0]] = extra[1]
        if T <= ks[0]: return 0.0
        if T >= ks[-1]:
            k1, k0 = ks[-1], ks[-2]
            return vals[k1] + (vals[k1]-vals[k0])/(k1-k0)*(T-k1)
        for i in range(1, len(ks)):
            if T <= ks[i]:
                k0, k1 = ks[i-1], ks[i]
                w = (T-k0)/(k1-k0)
                return vals[k0] + w*(vals[k1]-vals[k0])

    for T, lab in zip(YRS, LAB):
        if T <= 0.5: continue
        c = float(par[lab]) / 100.0
        ts = np.arange(0.5, T + 1e-9, 0.5)
        kt = sorted(nodes)
        def resid(x):
            pv = sum((c/2)*100*np.exp(-nld(t, kt, extra=(T, x))) for t in ts)
            return pv + 100.0*np.exp(-x) - 100.0
        lo, hi = -2.0*max(T,1.0), 5.0*max(T,1.0)
        f0, f1 = resid(lo), resid(hi); it = 0
        while f0*f1 > 0 and it < 60:
            lo *= 1.5; hi *= 1.5; f0, f1 = resid(lo), resid(hi); it += 1
        nodes[T] = brentq(resid, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=300)

    kt = sorted(nodes)
    nl = np.array([nld(m/12.0, kt) for m in range(1, n_months+1)])
    return nl / (np.arange(1, n_months+1)/12.0) * 100.0, nl

def fwd_kink(nl):
    """Instantaneous fwd is d(-lnDF)/dt. Kink = |f(just after 6mo) - f(just
    before 6mo)| in bp, i.e. the discontinuity at the bill/bond boundary."""
    f = np.diff(nl) * 12.0 * 100.0
    return abs(f[6] - f[4])

d = pd.read_csv(os.path.join("data","treasury_yields.csv")).dropna(subset=LAB)
d['DATE'] = pd.to_datetime(d['DATE'])
me = d.groupby(d['DATE'].dt.to_period('M')).last()
me = me[me.index >= '2018-01']

print("="*70)
print("T1  FORWARD-RATE KINK AT THE 6mo/1yr BOUNDARY  (bp, lower = better)")
print("="*70)
print("%10s" % "curve" + "".join("%11s" % c for c in CONV))
agg = {c: [] for c in CONV}
for i in (0, len(me)//4, len(me)//2, 3*len(me)//4, len(me)-1):
    r = me.iloc[i]; par = {l: float(r[l]) for l in LAB}
    row = []
    for c in CONV:
        _, nl = boot(par, c); k = fwd_kink(nl); agg[c].append(k); row.append(k)
    print("%10s" % str(me.index[i]) + "".join("%11.2f" % v for v in row))
print("-"*70)
print("%10s" % "MEAN" + "".join("%11.2f" % np.mean(agg[c]) for c in CONV))

print("\n" + "="*70)
print("T2  DURATION IMPACT  (frozen CPR=0.12, parallel bump, mean over curves)")
print("="*70)
h = M.BUMP_BP/100.0
w = np.ones(len(LAB))
z12 = np.full(N, 0.12)
print("%6s" % "cpn" + "".join("%11s" % c for c in CONV))
for cpn in (2.5, 4.0, 6.5):
    row = []
    for c in CONV:
        ds = []
        for i in (0, len(me)//2, len(me)-1):
            r = me.iloc[i]; par = {l: float(r[l]) for l in LAB}
            p0 = M.price_path(cpn, z12, boot(par, c)[0])
            px = {}
            for sgn in (+1,-1):
                bp = {l: par[l] + sgn*h*wi for l, wi in zip(LAB, w)}
                px[sgn] = M.price_path(cpn, z12, boot(bp, c)[0])
            ds.append((px[-1]-px[+1])/(2.0*p0*(h/100.0)))
        row.append(np.mean(ds))
    print("%6.1f" % cpn + "".join("%11.4f" % v for v in row))
print("\nspread across conventions vs the 1.33 target is the thing to read")
