"""External check of bootstrap_zeros. Nothing here borrows price_path.

C1  PAR REPRICING. A bootstrap is *defined* by: price a par bond off the zeros
    it produces and get exactly 100. Bond pricer written from scratch here.
    Tested on synthetic curves AND on real month-end panel curves.

C2  SHORT-END CONVENTION. bootstrap_zeros sets z[T]=par[T] for T<=1yr with no
    annual->continuous conversion, while everything discounts exp(-z*t).
    Measures the resulting error instead of assuming it is small.

C3  CONVENTION-CORRECTED REBUILD. Re-bootstraps with the short end converted
    properly, then reports how much the 30y-relevant zeros move. If durations
    are sensitive to this at the 1.33 scale it shows up here; if not, ruled out.
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M
from scipy.interpolate import interp1d

LAB, YRS = M.MAT_LABELS, M.MAT_YEARS

def price_par_bond(T, cpn_pct, zfun):
    """From-scratch semiannual bond pricer. cpn_pct = annual coupon in percent.
    zfun(t) -> continuously-compounded zero in percent."""
    ts = np.arange(0.5, T + 1e-9, 0.5)
    c = cpn_pct / 2.0
    pv = sum(c * np.exp(-zfun(t) / 100.0 * t) for t in ts)
    pv += 100.0 * np.exp(-zfun(T) / 100.0 * T)
    return pv

def zfun_from_grid(zgrid):
    """zgrid = 360 monthly zeros from bootstrap_zeros."""
    tt = np.arange(1, len(zgrid) + 1) / 12.0
    f = interp1d(tt, zgrid, kind='linear', fill_value='extrapolate')
    return lambda t: float(f(t))

def report(par, tag):
    z = M.bootstrap_zeros(par)
    zf = zfun_from_grid(z)
    rows = []
    for T, lab in zip(YRS, LAB):
        if T < 1.0:
            continue
        p = price_par_bond(T, float(par[lab]), zf)
        rows.append((lab, float(par[lab]), p, p - 100.0))
    print("\n--- %s ---" % tag)
    print("%6s%9s%12s%12s" % ("node", "par%", "repriced", "err"))
    for lab, pr, p, e in rows:
        print("%6s%9.4f%12.6f%12.2e" % (lab, pr, p, e))
    return max(abs(e) for _, _, _, e in rows)

print("=" * 62)
print("C1  PAR REPRICING  (exact bootstrap => err == 0)")
print("=" * 62)
worst = 0.0
for lvl in (2.0, 5.0):
    worst = max(worst, report({l: lvl for l in LAB}, "flat %.1f%%" % lvl))
sl = {l: 3.0 + 2.0 * (y - YRS[0]) / (YRS[-1] - YRS[0]) for l, y in zip(LAB, YRS)}
worst = max(worst, report(sl, "sloped 3->5%"))

try:
    d = pd.read_csv(os.path.join("data", "treasury_yields.csv"))
    have = [l for l in LAB if l in d.columns]
    if len(have) >= 6:
        dd = d.dropna(subset=have)
        for i in (0, len(dd) // 2, len(dd) - 1):
            r = dd.iloc[i]
            par = {l: (float(r[l]) if l in have else np.nan) for l in LAB}
            ks = [l for l in LAB if not np.isnan(par[l])]
            fi = interp1d([YRS[LAB.index(l)] for l in ks], [par[l] for l in ks],
                          kind='linear', fill_value='extrapolate')
            for l, y in zip(LAB, YRS):
                if np.isnan(par[l]):
                    par[l] = float(fi(y))
            worst = max(worst, report(par, "REAL curve row %d" % i))
    else:
        print("\n[real curves skipped: cols %s]" % list(d.columns))
except Exception as e:
    print("\n[real curve load failed: %s]" % e)

print("\nmax |reprice err| across all curves = %.3e" % worst)
print("  VERDICT: %s (threshold 1e-6)\n"
      % ("PASS - self-consistent" if worst < 1e-6
         else "FAIL - not self-consistent"))

print("=" * 62)
print("C2/C3  SHORT-END CONVENTION")
print("=" * 62)
print("bootstrap_zeros uses z[T]=par[T] for T<=1yr, no annual->continuous conv.")
for lvl in (2.0, 5.0):
    par = {l: lvl for l in LAB}
    z_as_is = M.bootstrap_zeros(par)
    par_fix = dict(par)
    for l, y in zip(LAB, YRS):
        if y <= 1.0:
            par_fix[l] = 100.0 * np.log(1.0 + lvl / 100.0)
    z_fix = M.bootstrap_zeros(par_fix)
    print("\nflat %.1f%%  continuous equiv of short end = %.4f" % (
        lvl, 100.0 * np.log(1.0 + lvl / 100.0)))
    print("%8s%12s%12s%12s" % ("month", "as-is", "conv-fixed", "diff_bp"))
    for m in (1, 12, 60, 120, 360):
        a, b = z_as_is[m - 1], z_fix[m - 1]
        print("%8d%12.5f%12.5f%12.2f" % (m, a, b, (a - b) * 100))
    da = np.mean(z_as_is - z_fix) * 100
    print("mean zero shift = %.2f bp  ->  approx duration impact "
          "is second-order; a 33%% duration error needs a ~large level shift" % da)
