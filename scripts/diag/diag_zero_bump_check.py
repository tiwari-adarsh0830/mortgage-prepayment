"""Corrected T1. Bumps the ZERO curve directly, so Macaulay == modified holds
exactly and any deviation is a genuine discounting/duration defect.

The earlier par-bump version was invalid: a parallel par shift is not a parallel
zero shift, so its deviation tracked curve slope and proved nothing.

Also reports the compounding convention actually in force, since price_path
discounts as exp(-z*t) and that only matches bootstrap_zeros if it returns
continuously-compounded zeros.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

N = M.N_MONTHS
t_yrs = np.arange(1, N + 1) / 12.0

def price_from_zeros(coupon, cpr360, zeros):
    return M.price_path(coupon, cpr360, zeros)

def dur_zero_bumped(coupon, cpr360, zeros, dz_bp=25.0):
    p0 = price_from_zeros(coupon, cpr360, zeros)
    dz = dz_bp / 100.0
    pm = price_from_zeros(coupon, cpr360, zeros - dz)
    pp = price_from_zeros(coupon, cpr360, zeros + dz)
    return p0, (pm - pp) / (2.0 * p0 * (dz / 100.0))

def dur_closed(coupon, cpr360, zeros):
    note_m = (coupon + M.GFEE) / 100.0 / 12.0
    inv_m = coupon / 100.0 / 12.0
    smm = 1.0 - (1.0 - np.clip(cpr360, 0.0, 0.99)) ** (1.0 / 12.0)
    disc = np.exp(-zeros / 100.0 * t_yrs)
    bal = 100.0
    pmt = bal * note_m / (1.0 - (1.0 + note_m) ** (-N))
    pv = wt = 0.0
    for t in range(N):
        if bal <= 1e-12:
            break
        sp = max(min(pmt - bal * note_m, bal), 0.0)
        pp_ = (bal - sp) * smm[t]
        cf = bal * inv_m + sp + pp_
        pv += cf * disc[t]
        wt += t_yrs[t] * cf * disc[t]
        bal -= (sp + pp_)
    return pv, wt / pv

print("=" * 70)
print("CORRECTED T1: parallel ZERO bump vs closed form")
print("=" * 70)
print("%6s%6s%11s%11s%10s" % ("zlvl", "cpn", "D_bump", "D_closed", "ratio"))
worst = 0.0
for zlvl in (3.0, 4.0, 6.0):
    for c in (2.5, 4.0, 6.5):
        for cpr in (0.0, 0.12):
            z = np.full(N, zlvl)
            cp = np.full(N, cpr)
            _, db = dur_zero_bumped(c, cp, z)
            _, dc = dur_closed(c, cp, z)
            worst = max(worst, abs(db / dc - 1.0))
            print("%6.1f%6.1f%11.4f%11.4f%10.5f  cpr=%.2f" % (
                zlvl, c, db, dc, db / dc, cpr))
print("\nmax |ratio-1| = %.3e" % worst)
print("  <1e-3 -> discounting core CORRECT, term correction ruled out")
print("  else  -> the defect is real and this is its size\n")

print("=" * 70)
print("COMPOUNDING CONVENTION CHECK")
print("=" * 70)
par_flat = {lab: 5.0 for lab in M.MAT_LABELS}
z = M.bootstrap_zeros(par_flat)
print("flat 5%% par -> zeros[0]=%.6f  zeros[-1]=%.6f  spread=%.4f"
      % (z[0], z[-1], z[-1] - z[0]))
print("continuous equiv of 5%% annual = %.6f" % (100 * np.log(1.05)))
print("If zeros come back ~5.00 on a flat 5%% par curve but price_path")
print("discounts exp(-z*t), the convention is ANNUAL fed into a CONTINUOUS")
print("discounter -- a systematic term error. Expect ~4.879 if continuous.")
