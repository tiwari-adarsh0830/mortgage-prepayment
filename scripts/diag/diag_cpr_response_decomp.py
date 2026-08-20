"""Where does duration come from: discounting, the model segment, or the terminal?

Every mechanical test so far froze CPR, so none of them could see the prepayment
feedback channel. This partitions total duration four ways by controlling which
CPR segment is allowed to see the BUMPED incentive:

  NONE   both segments held at the unbumped path -> pure discounting duration
  M      months 1-33 (hazard model) respond, terminal frozen
  T      months 34-360 (terminal S-curve) respond, model frozen
  BOTH   production behaviour

Reads: if NONE is already close to BOTH, prepayment feedback contributes little
and no fix to the response reaches 1.33x. If M+T falls well short of BOTH the
channels interact. Whichever of M/T carries the shortfall is where to look.

Restricted to coupons 2.5-4.0: terminal saturation is exactly 0% there (checked
on the panel), so the flat-extrapolation channel is absent and cannot confound.
Those are also the four worst-hedged coupons (t_lvl -5.15 to -6.35).

No returns, no regressions -- a pricing decomposition only, so it cannot be
circular with the fit that produced 1.33.
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

LAB, YRS = M.MAT_LABELS, M.MAT_YEARS
h = M.BUMP_BP / 100.0
COUPONS = [2.5, 3.0, 3.5, 4.0]
MAXS = M.MAX_SEQ


def build_cpr(coupon, pmms_for_model, pmms_for_term, model, scaler, a, b, asof):
    """360-month CPR. The two segments can be driven by DIFFERENT pmms values,
    which is how a segment is frozen: pass the unbumped pmms to freeze it."""
    note = coupon + M.GFEE
    p33 = M.cpr_path(note - pmms_for_model, model, scaler, a, b)
    out = np.empty(M.N_MONTHS)
    out[:MAXS] = p33
    out[MAXS:] = M.terminal_cpr(note - pmms_for_term, asof=asof)
    return out


def dur(coupon, par, pmms, model, scaler, a, b, asof, w, resp_model, resp_term):
    """Two-sided parallel-bump duration. resp_* select whether that segment
    sees the bumped PMMS. Curve is bumped in all cases -- only the CPR
    RESPONSE is switched, never the discounting."""
    z0 = M.bootstrap_zeros(par)
    c0 = build_cpr(coupon, pmms, pmms, model, scaler, a, b, asof)
    p0 = M.price_path(coupon, c0, z0)
    px = {}
    for sgn in (+1, -1):
        bp = {l: float(par[l]) + sgn * h * wi for l, wi in zip(LAB, w)}
        pm = pmms + sgn * h                      # PMMS moves with the curve
        cpr = build_cpr(coupon,
                        pm if resp_model else pmms,
                        pm if resp_term else pmms,
                        model, scaler, a, b, asof)
        px[sgn] = M.price_path(coupon, cpr, M.bootstrap_zeros(bp))
    return (px[-1] - px[+1]) / (2.0 * p0 * (h / 100.0))


print("loading hazard model...", flush=True)
model, scaler, a, b = M.load_hazard()

daily = pd.read_csv(os.path.join(M.DATA, "treasury_yields.csv"))
daily['DATE'] = pd.to_datetime(daily['DATE'])
daily = daily.dropna(subset=LAB)
me = daily.groupby(daily['DATE'].dt.to_period('M')).last()

pm = pd.read_csv(os.path.join(M.DATA, "pmms_monthly.csv"))
def _parse(x):
    t = str(int(x))
    if len(t) == 5: return pd.Timestamp(year=int(t[1:]), month=int(t[0]), day=1)
    if len(t) == 6: return pd.Timestamp(year=int(t[2:]), month=int(t[:2]), day=1)
    return pd.NaT
pm['date'] = pm['reporting_period'].apply(_parse)
pms = pm.dropna(subset=['date']).set_index(pm.dropna(subset=['date'])['date']
                                           .dt.to_period('M'))['rate_30yr']

months = [p for p in me.index if p >= pd.Period('2018-01') and p in pms.index]
print("months: %d  (%s .. %s)" % (len(months), months[0], months[-1]), flush=True)

w_par = np.ones(len(LAB))
CASES = [("NONE", False, False), ("M", True, False),
         ("T", False, True), ("BOTH", True, True)]

print("\n" + "=" * 78)
print("DURATION BY RESPONDING SEGMENT   (parallel bump, mean over months)")
print("=" * 78)
print("%6s%10s%10s%10s%10s | %9s%9s" % (
    "cpn", "NONE", "M", "T", "BOTH", "M-NONE", "T-NONE"))
rows = []
for c in COUPONS:
    acc = {k: [] for k, _, _ in CASES}
    for p in months:
        r = me.loc[p]
        par = {l: float(r[l]) for l in LAB}
        pmv = float(pms.loc[p])
        asof = str(p)
        for name, rm, rt in CASES:
            acc[name].append(dur(c, par, pmv, model, scaler, a, b,
                                 asof, w_par, rm, rt))
    mu = {k: float(np.mean(v)) for k, v in acc.items()}
    rows.append((c, mu))
    print("%6.1f%10.4f%10.4f%10.4f%10.4f | %9.4f%9.4f" % (
        c, mu["NONE"], mu["M"], mu["T"], mu["BOTH"],
        mu["M"] - mu["NONE"], mu["T"] - mu["NONE"]))

print("\n" + "=" * 78)
print("ADDITIVITY AND HEADROOM")
print("=" * 78)
print("%6s%12s%12s%12s%12s" % (
    "cpn", "sum-parts", "BOTH", "interact", "BOTH/NONE"))
for c, mu in rows:
    s = mu["NONE"] + (mu["M"] - mu["NONE"]) + (mu["T"] - mu["NONE"])
    print("%6.1f%12.4f%12.4f%12.4f%12.4f" % (
        c, s, mu["BOTH"], mu["BOTH"] - s, mu["BOTH"] / mu["NONE"]))

print("\nNONE ~ BOTH        -> feedback contributes little; response is not the lever")
print("BOTH << NONE       -> feedback SHORTENS duration; question is whether too much")
print("large |interact|   -> segments are not separable; treat parts with care")
