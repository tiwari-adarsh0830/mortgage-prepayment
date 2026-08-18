"""Isolate the 1.33 duration gap from the hazard model entirely.

CPR is FROZEN (no model call, no re-forecast under bump), so every number here
is pure pricing / discounting / bump geometry.

  T1  bumped duration vs CLOSED-FORM duration at CPR=0, parallel bump.
      Discounting is continuous, so Macaulay == modified exactly.
      A mismatch here is the discounting term correction.

  T2  sum of three tent KRDs vs ONE parallel par bump, at the PRICE level.
      README verifies partition-of-unity on WEIGHTS; this is the stronger claim.

  T3  same at nonzero CPR, to check the identity survives front-loading.

Imports nothing that produced the 1.36/1.33 figures.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

h = M.BUMP_BP / 100.0
N = M.N_MONTHS
YRS = np.asarray(M.MAT_YEARS, float)

def flat_par(level):
    return {lab: float(level) for lab in M.MAT_LABELS}

def sloped_par(short, long_):
    f = (YRS - YRS.min()) / (YRS.max() - YRS.min())
    return {lab: float(short + (long_ - short) * v)
            for lab, v in zip(M.MAT_LABELS, f)}

def dur_bumped(coupon, par, cpr360, w):
    p0 = M.price_path(coupon, cpr360, M.bootstrap_zeros(par))
    px = {}
    for sgn in (+1, -1):
        bp = {lab: float(par[lab]) + sgn * h * wi
              for lab, wi in zip(M.MAT_LABELS, w)}
        px[sgn] = M.price_path(coupon, cpr360, M.bootstrap_zeros(bp))
    return p0, (px[-1] - px[+1]) / (2.0 * p0 * (h / 100.0))

def dur_closed_form(coupon, par, cpr360):
    note_m = (coupon + M.GFEE) / 100.0 / 12.0
    inv_m = coupon / 100.0 / 12.0
    smm = 1.0 - (1.0 - np.clip(cpr360, 0.0, 0.99)) ** (1.0 / 12.0)
    zeros = M.bootstrap_zeros(par)
    t_yrs = np.arange(1, N + 1) / 12.0
    disc = np.exp(-zeros / 100.0 * t_yrs)
    bal = 100.0
    pmt = bal * note_m / (1.0 - (1.0 + note_m) ** (-N))
    pv = 0.0
    wt = 0.0
    for t in range(N):
        if bal <= 1e-12:
            break
        sp = max(min(pmt - bal * note_m, bal), 0.0)
        pp = (bal - sp) * smm[t]
        cf = bal * inv_m + sp + pp
        pv += cf * disc[t]
        wt += t_yrs[t] * cf * disc[t]
        bal -= (sp + pp)
    return pv, wt / pv

CURVES = [("flat 4%", flat_par(4.0)),
          ("flat 6%", flat_par(6.0)),
          ("sloped 3->5%", sloped_par(3.0, 5.0)),
          ("inverted 5->4%", sloped_par(5.0, 4.0))]
COUPONS = [2.5, 4.0, 6.5]
w_par = np.ones_like(YRS)

print("=" * 74)
print("T1  bumped duration vs closed form,  CPR = 0,  parallel par bump")
print("=" * 74)
print("%-15s%5s%11s%11s%10s" % ("curve", "cpn", "D_bump", "D_closed", "ratio"))
worst = 0.0
for name, par in CURVES:
    for c in COUPONS:
        z = np.zeros(N)
        _, db = dur_bumped(c, par, z, w_par)
        _, dc = dur_closed_form(c, par, z)
        r = db / dc
        worst = max(worst, abs(r - 1.0))
        print("%-15s%5.1f%11.4f%11.4f%10.5f" % (name, c, db, dc, r))
print("\nmax |ratio-1| = %.3e" % worst)
print("  < 1e-3  -> discounting/duration core CORRECT; 1.33 is not here")
print("  ~ 0.25  -> FOUND IT: the term correction is the cause\n")

print("=" * 74)
print("T2/T3  sum of three tents vs one parallel bump (PRICE level)")
print("=" * 74)
print("%-15s%5s%6s%9s%9s%9s%10s%10s%9s" % (
    "curve", "cpn", "CPR", "K2", "K5", "K10", "sum", "parallel", "ratio"))
for cpr_lvl in (0.0, 0.12):
    for name, par in CURVES:
        for c in COUPONS:
            z = np.full(N, cpr_lvl)
            ks = [dur_bumped(c, par, z, M.key_rate_weights3(t))[1]
                  for t in ('2yr', '5yr', '10yr')]
            s = sum(ks)
            _, dp = dur_bumped(c, par, z, w_par)
            print("%-15s%5.1f%6.2f%9.4f%9.4f%9.4f%10.4f%10.4f%9.5f" % (
                name, c, cpr_lvl, ks[0], ks[1], ks[2], s, dp, s / dp))
print("\nsum/parallel ~ 1.0 -> tents span correctly at the price level")
print("Read the ratio column above -- this legend prints regardless of outcome.")
