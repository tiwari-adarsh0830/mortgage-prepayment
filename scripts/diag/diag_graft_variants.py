"""S-curve response graft, all variants, so the answer does not depend on how
"apply the change to the model path" is read.

GRAFT FORMS (months 1-33; months 34-360 keep production behaviour, i.e. the
terminal evaluated at the bumped incentive):
  add   p33(i0) + [S(i1) - S(i0)]        additive level change
  mult  p33(i0) * [S(i1) / S(i0)]        proportional change
  base  p33(i1)                          production, model responds itself

BUMP CONVENTIONS:
  par   parallel bump on every node, PMMS moves with it (diagnostic; makes the
        CPR response as visible as possible)
  krd   production tents3 convention: PMMS moves only on the 10yr leg, so the
        2yr leg has no CPR response at all. D_level = k5 + k10 as in the pricer.

The krd/D_level column is the one comparable to the panel's D_level, which is
the duration the 1.33 scalar multiplies.
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

LAB = M.MAT_LABELS
h = M.BUMP_BP / 100.0
MAXS = M.MAX_SEQ
COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
EPS = 1e-9


def cpr360(coupon, pmms0, pmms1, model, scaler, a, b, asof, form):
    """pmms0 = unbumped (baseline), pmms1 = what the response sees."""
    note = coupon + M.GFEE
    i0, i1 = note - pmms0, note - pmms1
    if form == "base":
        p33 = M.cpr_path(i1, model, scaler, a, b)
    else:
        s0 = M.terminal_cpr(i0, asof=asof)
        s1 = M.terminal_cpr(i1, asof=asof)
        p0 = M.cpr_path(i0, model, scaler, a, b)
        p33 = p0 + (s1 - s0) if form == "add" else p0 * (s1 / max(s0, EPS))
    p33 = np.clip(p33, 0.0, 0.99)
    out = np.empty(M.N_MONTHS)
    out[:MAXS] = p33
    out[MAXS:] = M.terminal_cpr(i1, asof=asof)
    return out


def _price(coupon, par, pmms0, pmms1, model, scaler, a, b, asof, form):
    return M.price_path(coupon,
                        cpr360(coupon, pmms0, pmms1, model, scaler, a, b, asof, form),
                        M.bootstrap_zeros(par))


def dur_par(coupon, par, pmms, model, scaler, a, b, asof, form):
    p0 = _price(coupon, par, pmms, pmms, model, scaler, a, b, asof, form)
    px = {}
    for sgn in (+1, -1):
        bp = {l: float(par[l]) + sgn * h for l in LAB}
        px[sgn] = _price(coupon, bp, pmms, pmms + sgn * h,
                         model, scaler, a, b, asof, form)
    return (px[-1] - px[+1]) / (2.0 * p0 * (h / 100.0))


def dlevel_krd(coupon, par, pmms, model, scaler, a, b, asof, form):
    """Production tents3: D_level = k5 + k10, PMMS moves only on the 10yr leg."""
    p0 = _price(coupon, par, pmms, pmms, model, scaler, a, b, asof, form)
    tot = 0.0
    for ten in ('5yr', '10yr'):
        w = M.key_rate_weights3(ten)
        dp = h if ten == '10yr' else 0.0
        px = {}
        for sgn in (+1, -1):
            bp = {l: float(par[l]) + sgn * h * wi for l, wi in zip(LAB, w)}
            px[sgn] = _price(coupon, bp, pmms, pmms + sgn * dp,
                             model, scaler, a, b, asof, form)
        tot += (px[-1] - px[+1]) / (2.0 * p0 * (h / 100.0))
    return tot


# match the panel run: --floor-mode pinned-fixed. FLOOR_MODE defaults to
# "fitted" at module level and is only overridden inside main(), which we
# bypass by importing, so it must be set explicitly here.
M.FLOOR_MODE = "pinned-fixed"
print("floor mode: %s" % M.FLOOR_MODE, flush=True)
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
pm['date'] = pm['reporting_period'].apply(_p); pm = pm.dropna(subset=['date'])
pms = pm.set_index(pm['date'].dt.to_period('M'))['rate_30yr']
months = [p for p in me.index if p >= pd.Period('2018-01') and p in pms.index]
print("months: %d (%s .. %s)\n" % (len(months), months[0], months[-1]), flush=True)

res = {}
for form in ("base", "add", "mult"):
    for conv, fn in (("par", dur_par), ("krd", dlevel_krd)):
        for c in COUPONS:
            v = []
            for p in months:
                r = me.loc[p]
                par = {l: float(r[l]) for l in LAB}
                v.append(fn(c, par, float(pms.loc[p]), model, scaler, a, b, str(p), form))
            res[(form, conv, c)] = float(np.mean(v))

for conv, lab in (("krd", "PRODUCTION tents3 D_level (k5+k10, PMMS on 10yr only)"),
                  ("par", "parallel bump, PMMS moves fully (diagnostic)")):
    print("=" * 72)
    print(lab)
    print("=" * 72)
    print("%6s%11s%11s%11s%10s%10s" % (
        "cpn", "base", "add", "mult", "add/base", "mult/base"))
    ra, rm = [], []
    for c in COUPONS:
        b0 = res[("base", conv, c)]
        aa = res[("add", conv, c)]
        mm = res[("mult", conv, c)]
        ra.append(aa/b0); rm.append(mm/b0)
        print("%6.1f%11.4f%11.4f%11.4f%10.4f%10.4f" % (c, b0, aa, mm, aa/b0, mm/b0))
    print("-" * 72)
    print("%6s%33s%10.4f%10.4f   (add %.3f-%.3f, mult %.3f-%.3f)" % (
        "MEAN", "", float(np.mean(ra)), float(np.mean(rm)),
        min(ra), max(ra), min(rm), max(rm)))
    print()
print("target for the scalar to go to 1: ratio ~1.33, flat across coupons")
