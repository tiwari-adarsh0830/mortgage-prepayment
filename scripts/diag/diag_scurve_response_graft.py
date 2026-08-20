"""Advisor test 2026-08-20: keep the transformer's BASELINE CPR path but take
the RESPONSE TO BUMP from the realized S-curve.

For months 1-33, production uses cpr_path(bumped_inc). Here we instead use
    cpr_path(unbumped_inc) + [terminal_cpr(bumped_inc) - terminal_cpr(unbumped_inc)]
so the level is the model's and the derivative is realized-calibrated.
Months 34-360 keep production behaviour (terminal at the bumped incentive).

Reports the duration under each scheme and the ratio. If the ratio is ~1.33 the
model's incentive response is the source of the under-sizing.

Parallel bump with PMMS moving, applied identically to both schemes, so the
ratio is like-for-like even though it is not the production KRD convention.
No returns, no regressions.
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

LAB = M.MAT_LABELS
h = M.BUMP_BP / 100.0
MAXS = M.MAX_SEQ
COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]


def cpr360(coupon, pmms_base, pmms_bump, model, scaler, a, b, asof, graft):
    note = coupon + M.GFEE
    i0, i1 = note - pmms_base, note - pmms_bump
    if graft:
        d = M.terminal_cpr(i1, asof=asof) - M.terminal_cpr(i0, asof=asof)
        p33 = M.cpr_path(i0, model, scaler, a, b) + d
        p33 = np.clip(p33, 0.0, 0.99)
    else:
        p33 = M.cpr_path(i1, model, scaler, a, b)
    out = np.empty(M.N_MONTHS)
    out[:MAXS] = p33
    out[MAXS:] = M.terminal_cpr(i1, asof=asof)
    return out


def dur(coupon, par, pmms, model, scaler, a, b, asof, graft):
    z0 = M.bootstrap_zeros(par)
    p0 = M.price_path(coupon, cpr360(coupon, pmms, pmms, model, scaler,
                                     a, b, asof, graft), z0)
    px = {}
    for sgn in (+1, -1):
        bp = {l: float(par[l]) + sgn * h for l in LAB}
        pmb = pmms + sgn * h
        px[sgn] = M.price_path(coupon,
                               cpr360(coupon, pmms, pmb, model, scaler,
                                      a, b, asof, graft),
                               M.bootstrap_zeros(bp))
    return (px[-1] - px[+1]) / (2.0 * p0 * (h / 100.0))


print("loading hazard model...", flush=True)
model, scaler, a, b = M.load_hazard()

d = pd.read_csv(os.path.join(M.DATA, "treasury_yields.csv"))
d['DATE'] = pd.to_datetime(d['DATE'])
me = d.dropna(subset=LAB).groupby(d['DATE'].dt.to_period('M')).last()

pm = pd.read_csv(os.path.join(M.DATA, "pmms_monthly.csv"))
def _p(x):
    t = str(int(x))
    if len(t) == 5: return pd.Timestamp(year=int(t[1:]), month=int(t[0]), day=1)
    if len(t) == 6: return pd.Timestamp(year=int(t[2:]), month=int(t[:2]), day=1)
    return pd.NaT
pm['date'] = pm['reporting_period'].apply(_p)
pm = pm.dropna(subset=['date'])
pms = pm.set_index(pm['date'].dt.to_period('M'))['rate_30yr']

months = [p for p in me.index if p >= pd.Period('2018-01') and p in pms.index]
print("months: %d (%s .. %s)\n" % (len(months), months[0], months[-1]), flush=True)

print("=" * 62)
print("MODEL RESPONSE vs GRAFTED REALIZED S-CURVE RESPONSE")
print("=" * 62)
print("%6s%12s%12s%10s" % ("cpn", "D_model", "D_graft", "ratio"))
rs = []
for c in COUPONS:
    dm, dg = [], []
    for p in months:
        r = me.loc[p]
        par = {l: float(r[l]) for l in LAB}
        pv = float(pms.loc[p]); asof = str(p)
        dm.append(dur(c, par, pv, model, scaler, a, b, asof, False))
        dg.append(dur(c, par, pv, model, scaler, a, b, asof, True))
    mm, mg = float(np.mean(dm)), float(np.mean(dg))
    rs.append(mg / mm)
    print("%6.1f%12.4f%12.4f%10.4f" % (c, mm, mg, mg / mm))
print("-" * 62)
print("%6s%12s%12s%10.4f   (min %.4f  max %.4f)" % (
    "MEAN", "", "", float(np.mean(rs)), min(rs), max(rs)))
print("\n~1.33 flat -> the model's incentive response IS the under-sizing")
print("~1.00      -> response is not the lever")
print("<1.00      -> realized response is STEEPER; graft shortens duration")
