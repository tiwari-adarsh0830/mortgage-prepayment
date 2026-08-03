"""
Model-based level/slope hedge -- built to advisor spec (2026-07-24).

SPEC (implemented literally):
  For each coupon, each month:
    - move Treasury curve +/-25bp in the 5y region only, then separately 10y
    - pass the bump through to the mortgage rate via the PMMS-Treasury spread
    - recompute refi incentive, re-run hazard model -> NEW CPR PATH
    - long-run CPR beyond the 33m forecast horizon also shifts with the bump
      (terminal CPR taken from the BUMPED path, held flat to month 360)
    - reprice under each bump, two-sided difference -> KRD5, KRD10
    - convert: D_level = KRD5 + KRD10 ; D_slope = (KRD10 - KRD5)/2
  Hedged return:
    hedged = tba_total_return - income + (D_level*d_level + D_slope*d_slope)/100
    where d_level = (dy5+dy10)/2, d_slope = dy10 - dy5.
    DERIVATION: dP/P = -KRD5*dy5 - KRD10*dy10, and with dy5 = level - slope/2,
    dy10 = level + slope/2, this equals -D_level*level - D_slope*slope. So a
    position of equal parts 5y/10y sized to D_level plus a long-10y/short-5y
    position sized to D_slope reproduces the exposure exactly, as specified.
  Ratios use data through the PRIOR month-end, locked at month start.
  Verification: regress hedged returns on [level, slope] per coupon; all
    coefficients should be statistically zero with no pattern across coupons.

TWO IMPLEMENTATION NOTES (flagged, not silently chosen):
  1. BUMP LOCALIZATION. Standard key-rate taper for the 5y key rate is 0 at 3y,
     1 at 5y, 0 at 7y. The curve's par nodes are exactly [...3, 5, 7...], so the
     taper touches only the 5y node. Same for 10y (0 at 7y, 1 at 10y, 0 at 20y).
     "5y region only" therefore == single-node bump on this grid. Consequence:
     D_level + D_slope does NOT equal total effective duration -- localized key
     rates leave curve exposure between/beyond the knots. That is inherent to
     localized key rates, not an error.
  2. PMMS KEYING (--pmms-key). The 189bp spread is PMMS minus the 10yr
     (risk_neutral_rates.py computes it that way). Under that keying a 5y-only
     bump leaves PMMS unchanged, so the 5y KRD is pure discounting while the 10y
     KRD carries the prepayment response. Default '10yr'. '--pmms-key any' moves
     PMMS 1:1 with whichever tenor is bumped. Both are run and reported.

CALIBRATION: uses config/hazard_calibration_cpr_forecast.json (a=0.4559,
  b=-3.1376) -- the cohort-CPR forecast Platt pair. NOT the OAS loan-level pair.
"""
import os, json, pickle, argparse
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")
SEQ  = os.path.join(BASE, "data/sequences")

MAT_LABELS = ['1mo','3mo','6mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
MAT_YEARS  = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
N_MONTHS   = 360
MAX_SEQ    = 33
N_FEATURES = 9
DEAD_COLS  = [7, 8]
GFEE       = 0.50
BUMP_BP    = 25.0
COUPONS    = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
REP = dict(credit_score=740.0, orig_ltv=75.0, current_ltv=70.0,
           orig_upb=250000.0, dti=35.0, loan_purpose_enc=0.0, property_type_enc=0.0)


# ── curve ────────────────────────────────────────────────────────────────────
def bootstrap_zeros(par, n_months=N_MONTHS):
    z = {}
    for T, lab in zip(MAT_YEARS, MAT_LABELS):
        if T <= 1.0:
            z[T] = float(par[lab])
    def get_zero(T):
        kT = sorted(z); kz = [z[t] for t in kT]
        if T <= kT[0]:  return kz[0]
        if T >= kT[-1]: return kz[-1]
        return float(interp1d(kT, kz, kind='linear')(T))
    def dfac(T, zp): return np.exp(-zp/100.0*T)
    for T, lab in zip(MAT_YEARS, MAT_LABELS):
        if T <= 1.0: continue
        c  = float(par[lab])/100.0
        pv = sum((c/2)*100*dfac(t, get_zero(t)) for t in np.arange(0.5, T, 0.5))
        dfT = (100 - pv) / (c/2*100 + 100)
        if dfT <= 0: raise ValueError(f"bad DF at T={T}")
        z[T] = -np.log(dfT)/T*100
    kT = sorted(z); kz = [z[t] for t in kT]
    f = interp1d(kT, kz, kind='linear', fill_value='extrapolate')
    return np.array([float(f(m/12.0)) for m in range(1, n_months+1)])


def key_rate_weights(tenor, spanning=False):
    """Bump shape. spanning=False: standard localized key-rate taper (advisor's
    'region only' -- on this node set it selects a single node, and the two bumps
    together capture only ~1/3 of effective duration). spanning=True: partition of
    unity, w5=1 for T<=5 tapering to 0 at 10y, w10=1-w5, so D_level equals total
    effective duration and the pair spans the curve."""
    T = np.asarray(MAT_YEARS, float)
    if spanning:
        w5 = np.clip((10.0 - T)/5.0, 0.0, 1.0)
        return w5 if tenor == '5yr' else 1.0 - w5
    if tenor == '5yr':   lo, pk, hi = 3.0, 5.0, 7.0
    elif tenor == '10yr': lo, pk, hi = 7.0, 10.0, 20.0
    else: raise ValueError(tenor)
    w = np.zeros_like(T)
    left  = (T > lo) & (T <= pk)
    right = (T > pk) & (T < hi)
    w[left]  = (T[left]-lo)/(pk-lo)
    w[right] = (hi-T[right])/(hi-pk)
    return w


# ── hazard CPR path ──────────────────────────────────────────────────────────
def load_hazard():
    import importlib, sys
    sys.path.insert(0, os.path.join(BASE, "scripts"))
    m = importlib.import_module("stage2_forecast_cpr_gfee050")
    model  = m.load_model()
    scaler = pickle.load(open(os.path.join(SEQ, "scaler.pkl"), "rb"))
    cal    = json.load(open(os.path.join(BASE, "config",
                       "hazard_calibration_cpr_forecast.json")))
    a = float(cal.get("a", cal.get("platt_a")))
    b = float(cal.get("b", cal.get("platt_b")))
    print(f"  Platt (cpr_forecast): a={a:.4f} b={b:.4f}")
    return model, scaler, a, b


_CPR_CACHE = {}
def cpr_path(incentive, model, scaler, a, b):
    """(33,) CPR path for a constant refi incentive. All synthetic rows are
    identical (constant incentive + fixed REP loan), so n_paths=1 is exact."""
    k = round(float(incentive), 6)
    if k in _CPR_CACHE: return _CPR_CACHE[k]
    s = np.zeros((1, MAX_SEQ, N_FEATURES), dtype=np.float32)
    s[:, :, 0] = incentive
    s[:, :, 1] = REP["credit_score"]; s[:, :, 2] = REP["orig_ltv"]
    s[:, :, 3] = REP["current_ltv"];  s[:, :, 4] = REP["orig_upb"]
    s[:, :, 5] = np.arange(1, MAX_SEQ+1)[None, :]
    s[:, :, 6] = REP["dti"]; s[:, :, 7] = REP["loan_purpose_enc"]
    s[:, :, 8] = REP["property_type_enc"]
    flat = scaler.transform(s.reshape(-1, N_FEATURES)).reshape(1, MAX_SEQ, N_FEATURES)
    for c in DEAD_COLS: flat[:, :, c] = 0.0
    x = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(1, MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        logit = model(x, mask=mask, return_per_timestep=True).numpy()
    smm = 1.0/(1.0+np.exp(-(a*logit + b)))
    cpr = (1.0 - (1.0-smm)**12)[0]
    _CPR_CACHE[k] = cpr
    return cpr


_SCURVE_EXP = {}
FLOOR_MODE = "fitted"          # fitted | seasoned-fit | pinned-seasoned
_SEASONED = None
DEEP_INC = -2.5                # depth at which the pinned floor is estimated
FIXED_FLOOR = 0.0459           # advisor 2026-08-03: realized all-loan CPR at inc <= -2.5
                               # FULL-SAMPLE value -- introduces look-ahead by construction


def _load_seasoned():
    """Realized CPR restricted to loans aged > 60 months, rebuilt from the
    age-keyed panel. UPB is summed across seasoning levels 60 and 120 and CPR
    recomputed -- averaging the per-level cpr_upb would weight a thin 120+
    cell equally with a large 60-119 cell."""
    global _SEASONED
    if _SEASONED is not None:
        return _SEASONED
    import pandas as _pd
    b = _pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v6_upb_byage.csv"))
    b = b[b["age_group"] >= 60]
    g = (b.groupby(["coupon_bucket", "implied_mbs_coupon", "yyyymm"], as_index=False)
           [["upb_atrisk", "upb_prepay"]].sum())
    g = g[g["upb_atrisk"] > 0].copy()
    g["smm"] = g["upb_prepay"] / g["upb_atrisk"]
    g["cpr_upb"] = 1.0 - (1.0 - g["smm"]) ** 12
    g["date"] = _pd.to_datetime(g["yyyymm"].astype(str), format="%Y%m")
    _pm = _pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))

    def _parse(x):
        t = str(int(x))
        if len(t) == 5: return _pd.Timestamp(year=int(t[1:]), month=int(t[0]), day=1)
        if len(t) == 6: return _pd.Timestamp(year=int(t[2:]), month=int(t[:2]), day=1)
        return _pd.NaT

    _pm["date"] = _pm["reporting_period"].apply(_parse)
    _ps = _pm.dropna(subset=["date"]).set_index("date")["rate_30yr"]
    g["pmms"] = g["date"].map(_ps)
    g["inc"] = (g["implied_mbs_coupon"] + GFEE) - g["pmms"]
    _SEASONED = g.dropna(subset=["inc", "cpr_upb"])
    return _SEASONED


def scurve_params_asof(cutoff):
    """Terminal S-curve fitted ONLY on realized CPR strictly before `cutoff`
    (a YYYY-MM string), so the ratios use data through the prior month-end as
    specified. Cached per cutoff. Expanding window: ~460 obs at the panel start
    (realized file begins 2013-07), converging to the full sample by 2026."""
    _key = (cutoff, FLOOR_MODE)
    if _key in _SCURVE_EXP:
        return _SCURVE_EXP[_key]
    import pandas as _pd
    from scipy.optimize import curve_fit as _cf
    global _REALIZED
    try:
        _REALIZED
    except NameError:
        _r = _pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v6_upb.csv"))
        _r["date"] = _pd.to_datetime(_r["date"])
        _pm = _pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))
        def _parse(x):
            t = str(int(x))
            if len(t) == 5: return _pd.Timestamp(year=int(t[1:]), month=int(t[0]), day=1)
            if len(t) == 6: return _pd.Timestamp(year=int(t[2:]), month=int(t[:2]), day=1)
            return _pd.NaT
        _pm["date"] = _pm["reporting_period"].apply(_parse)
        _ps = _pm.dropna(subset=["date"]).set_index("date")["rate_30yr"]
        _r["pmms"] = _r["date"].map(_ps)
        _r["inc"] = (_r["implied_mbs_coupon"] + GFEE) - _r["pmms"]
        _REALIZED = _r.dropna(subset=["inc", "cpr_upb"])
    # Restrict to the coupon range actually being hedged. The realized file spans
    # coupons 1.0-8.0 and dates from 2013-07; coupons outside 2.5-6.5 are 25.5% of
    # the unrestricted sample and are a different population from the TBA stack.
    # Restricting raises R2 from 0.43 to 0.52 (more homogeneous) and shifts the
    # curve up (floor 0.0517->0.0546, sat 0.2204->0.2492). A further 2018+ cut fits
    # better still (R2 0.567) but leaves ~9 observations at the panel start, so it
    # is incompatible with the expanding window.
    h = _REALIZED[(_REALIZED["date"] < _pd.Timestamp(cutoff + "-01"))
                  & (_REALIZED["inc"] >= -4.0) & (_REALIZED["inc"] <= 2.0)
                  & (_REALIZED["implied_mbs_coupon"] >= 2.5)
                  & (_REALIZED["implied_mbs_coupon"] <= 6.5)]
    if len(h) < 40:
        raise ValueError("insufficient history before %s (n=%d)" % (cutoff, len(h)))
    def _sc(x, f, sa, k, x0):
        return f + (sa - f) / (1.0 + np.exp(-k * (x - x0)))
    if FLOOR_MODE == "seasoned-fit":
        _s = _load_seasoned()
        h = _s[(_s["date"] < _pd.Timestamp(cutoff + "-01"))
               & (_s["inc"] >= -4.0) & (_s["inc"] <= 2.0)
               & (_s["implied_mbs_coupon"] >= 2.5)
               & (_s["implied_mbs_coupon"] <= 6.5)]
        if len(h) < 40:
            raise ValueError("insufficient seasoned history before %s (n=%d)"
                             % (cutoff, len(h)))

    if FLOOR_MODE == "pinned-fixed":
        _fl = FIXED_FLOOR

        def _sc_fix(x, sa, k, x0):
            return _fl + (sa - _fl) / (1.0 + np.exp(-k * (x - x0)))

        pof, _ = _cf(_sc_fix, h["inc"].values, h["cpr_upb"].values,
                     p0=[0.22, 3.0, 0.4],
                     bounds=([0.05, 0.2, -2.0], [0.60, 10.0, 3.0]), maxfev=40000)
        q = dict(floor=_fl, sat=float(pof[0]), k=float(pof[1]), x0=float(pof[2]),
                 n=int(len(h)))
    elif FLOOR_MODE == "pinned-seasoned":
        _s = _load_seasoned()
        _d = _s[(_s["date"] < _pd.Timestamp(cutoff + "-01"))
                & (_s["inc"] <= DEEP_INC)
                & (_s["implied_mbs_coupon"] >= 2.5)
                & (_s["implied_mbs_coupon"] <= 6.5)]
        if len(_d) < 10:
            raise ValueError("insufficient deep-discount seasoned history before "
                             "%s (n=%d)" % (cutoff, len(_d)))
        _fl = float(_d["cpr_upb"].mean())

        def _sc_pin(x, sa, k, x0):
            return _fl + (sa - _fl) / (1.0 + np.exp(-k * (x - x0)))

        po3, _ = _cf(_sc_pin, h["inc"].values, h["cpr_upb"].values,
                     p0=[0.22, 3.0, 0.4],
                     bounds=([0.05, 0.2, -2.0], [0.60, 10.0, 3.0]), maxfev=40000)
        q = dict(floor=_fl, sat=float(po3[0]), k=float(po3[1]), x0=float(po3[2]),
                 n=int(len(h)), n_deep=int(len(_d)))
    else:
        po, _ = _cf(_sc, h["inc"].values, h["cpr_upb"].values, p0=[0.04, 0.22, 3.0, 0.4],
                    bounds=([0.005, 0.05, 0.2, -2.0], [0.15, 0.60, 10.0, 3.0]),
                    maxfev=40000)
        q = dict(floor=float(po[0]), sat=float(po[1]), k=float(po[2]), x0=float(po[3]),
                 n=int(len(h)))
    _SCURVE_EXP[_key] = q
    return q


_SCURVE = None
def _scurve_params():
    """Terminal-incentive S-curve fitted to REALIZED seasoned CPR.
    Model month-33 output is ~4x the realized floor at deep discounts (0.140 vs
    0.035) with its steepest response at incentive 0.00 rather than ~+0.5, so the
    terminal is anchored to realized data instead of extracted from the model."""
    global _SCURVE
    if _SCURVE is None:
        import json as _j
        _SCURVE = _j.load(open(os.path.join(BASE, "config", "terminal_scurve.json")))
    return _SCURVE


def terminal_cpr(incentive, asof=None):
    """S-curve terminal hazard. Argument is the (bumped) refi incentive, so the
    terminal shifts with the bump and preserves the CPR/rate relationship."""
    q = scurve_params_asof(asof) if asof else _scurve_params()
    return q["floor"] + (q["sat"] - q["floor"]) / (
        1.0 + np.exp(-q["k"] * (float(incentive) - q["x0"])))


def extend_cpr(path33, n=N_MONTHS, incentive=None, asof=None):
    """Months 1-33 from the hazard model. Months 34-360 from the terminal S-curve
    evaluated at the (bumped) incentive. If incentive is None, falls back to
    holding path33[-1] flat (the previous behaviour) for comparison runs."""
    out = np.empty(n)
    out[:MAX_SEQ] = path33
    out[MAX_SEQ:] = (path33[-1] if incentive is None
                     else terminal_cpr(incentive, asof=asof))
    return out


# ── pricing with a CPR path ──────────────────────────────────────────────────
def price_path(coupon, cpr360, zeros, gfee=GFEE, n=N_MONTHS):
    note_m = (coupon+gfee)/100.0/12.0
    inv_m  = coupon/100.0/12.0
    smm    = 1.0 - (1.0 - np.clip(cpr360, 0.0, 0.99))**(1.0/12.0)
    bal = 100.0
    pmt = bal*note_m/(1.0-(1.0+note_m)**(-n))
    disc = np.exp(-zeros/100.0*(np.arange(1, n+1)/12.0))
    pv = 0.0
    for t in range(n):
        if bal <= 1e-12: break
        sp = max(min(pmt - bal*note_m, bal), 0.0)
        pp = (bal - sp)*smm[t]
        pv += (bal*inv_m + sp + pp)*disc[t]
        bal -= (sp + pp)
    return pv


def krd_pair(coupon, par, pmms, model, scaler, a, b, pmms_key='10yr', spanning=False, asof=None):
    """Returns (P0, KRD5, KRD10) with CPR re-forecast under each bump."""
    note = coupon + GFEE
    z0 = bootstrap_zeros(par)
    p0 = price_path(coupon, extend_cpr(cpr_path(note-pmms, model, scaler, a, b),
                                       incentive=note-pmms, asof=asof), z0)
    h = BUMP_BP/100.0
    out = {}
    for ten in ('5yr', '10yr'):
        w = key_rate_weights(ten, spanning)
        # PMMS moves only if this tenor is the one PMMS is keyed to
        dp = h if (pmms_key == 'any' or pmms_key == ten) else 0.0
        prices = {}
        for sgn in (+1, -1):
            bp = {lab: float(par[lab]) + sgn*h*wi for lab, wi in zip(MAT_LABELS, w)}
            inc = note - (pmms + sgn*dp)
            prices[sgn] = price_path(coupon,
                            extend_cpr(cpr_path(inc, model, scaler, a, b),
                                       incentive=inc, asof=asof),
                            bootstrap_zeros(bp))
        out[ten] = (prices[-1] - prices[+1]) / (2.0*p0*(h/100.0))
    return p0, out['5yr'], out['10yr']


def ols(y, X):
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X@co; n, k = X.shape
    se = np.sqrt(np.diag(float(r@r)/(n-k)*np.linalg.pinv(X.T@X)))
    ss = float(((y-y.mean())**2).sum())
    return co, se, (1-float(r@r)/ss if ss > 1e-12 else np.nan)


def main(pmms_key, spanning=False):
    print(f"PMMS keying: {pmms_key} | bump shape: {'spanning' if spanning else 'localized'}")
    model, scaler, a, b = load_hazard()

    daily = pd.read_csv(os.path.join(DATA, "treasury_yields.csv"),
                        index_col=0, parse_dates=True).sort_index()
    clean = pd.read_excel(os.path.join(DATA, "treasury_yields_clean.xlsx"),
                          sheet_name="Treasury_Yields", header=1)
    clean.columns = [str(c).strip() for c in clean.columns]
    clean["Date"] = pd.to_datetime(clean["Date"]); clean = clean.sort_values("Date").reset_index(drop=True)
    y5c  = [c for c in clean.columns if "5yr" in c.lower() and "avg" not in c.lower()][0]
    y10c = [c for c in clean.columns if "10yr" in c.lower()][0]
    clean["d_level"] = ((clean[y5c]+clean[y10c])/2).diff()
    clean["d_slope"] = (clean[y10c]-clean[y5c]).diff()
    clean["income"]  = ((clean[y5c]+clean[y10c])/2)/12.0/100.0

    fncl = pd.read_excel(os.path.join(DATA, "fncl_tba_prices_clean.xlsx"),
                         sheet_name="Last_Price_Decimal", header=1)
    fncl.columns = [str(c).strip() for c in fncl.columns]
    fncl["Date"] = pd.to_datetime(fncl["Date"]); fncl = fncl.sort_values("Date").reset_index(drop=True)

    pm = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))
    def parse(p):
        s = str(int(p))
        if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
        if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
        return pd.NaT
    pm["date"] = pm["reporting_period"].apply(parse)
    pmms_s = pm.dropna(subset=["date"]).set_index("date")["rate_30yr"]

    # sanity: realized PMMS - 10yr spread (advisor cited 189bp)
    j = clean.set_index("Date")[[y10c]].copy()
    j["ym"] = j.index.to_period("M").to_timestamp()
    j["pmms"] = j["ym"].map(pmms_s)
    sp = (j["pmms"] - j[y10c]).dropna()
    print(f"  PMMS - 10yr spread: mean {sp.mean()*100:.0f}bp  (advisor cited 189bp)")

    rows = []
    for i in range(1, len(clean)):
        prev, curr = clean["Date"].iloc[i-1], clean["Date"].iloc[i]
        idx = daily.index[daily.index <= prev]
        if not len(idx): continue
        par = daily.loc[idx[-1]].to_dict()
        pmk = pd.Timestamp(prev.year, prev.month, 1)
        if pmk not in pmms_s.index: continue
        pmms = float(pmms_s[pmk])
        for c in COUPONS:
            col = f"FNCL {c}"
            if col not in fncl.columns: continue
            pc, pp = fncl.loc[fncl["Date"] == curr, col], fncl.loc[fncl["Date"] == prev, col]
            if pc.empty or pp.empty or pd.isna(pc.iloc[0]) or pd.isna(pp.iloc[0]): continue
            tba = (float(pc.iloc[0]) + c/12.0 - float(pp.iloc[0])) / float(pp.iloc[0])
            p0, k5, k10 = krd_pair(c, par, pmms, model, scaler, a, b, pmms_key,
                                   spanning, asof=str(curr.to_period("M")))
            rows.append(dict(ret_month=str(curr.to_period("M")), info_date=str(prev.date()),
                             coupon=c, pmms=pmms, price=p0, krd5=k5, krd10=k10,
                             D_level=k5+k10, D_slope=(k10-k5)/2.0,
                             tba_total_return=tba,
                             income=float(clean["income"].iloc[i]),
                             d_level=float(clean["d_level"].iloc[i]),
                             d_slope=float(clean["d_slope"].iloc[i])))

    p = pd.DataFrame(rows).dropna(subset=["d_level", "d_slope"])
    p["hedged"] = (p["tba_total_return"] - p["income"]
                   + (p["D_level"]*p["d_level"] + p["D_slope"]*p["d_slope"])/100.0)
    tag = pmms_key.replace("yr", "") + ("_span" if spanning else "_local")
    if FLOOR_MODE != "fitted":
        tag = tag + "_" + FLOOR_MODE.replace("-", "")
    p.to_csv(os.path.join(OUT, f"model_hedge_panel_{tag}.csv"), index=False)

    print(f"\npanel: {len(p)} coupon-months, {p['ret_month'].nunique()} months "
          f"({p['ret_month'].min()} -> {p['ret_month'].max()})")
    print("\nmean model durations by coupon:")
    print(p.groupby("coupon")[["krd5","krd10","D_level","D_slope"]].mean().round(3).to_string())

    print("\n=== VERIFICATION: hedged returns on [level, slope] (want all t ~ 0) ===")
    print(f"{'cpn':>4} {'n':>4} {'b_lvl':>9} {'t_lvl':>7} {'b_slp':>9} {'t_slp':>7} {'r2':>6}")
    for c, g in p.groupby("coupon"):
        g = g.dropna(subset=["hedged"])
        co, se, r2 = ols(g["hedged"].values,
                         np.column_stack([np.ones(len(g)), g["d_level"], g["d_slope"]]))
        print(f"{c:>4} {len(g):>4} {co[1]:>9.5f} {co[1]/se[1]:>7.2f} "
              f"{co[2]:>9.5f} {co[2]/se[2]:>7.2f} {r2:>6.3f}")
    print(f"\nSaved: outputs/model_hedge_panel_{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmms-key", default="10yr", choices=["10yr", "any"],
                    help="'10yr': PMMS moves only on the 10y bump (189bp spread is "
                         "PMMS-10yr). 'any': PMMS moves 1:1 with whichever tenor is bumped.")
    ap.add_argument("--spanning", action="store_true",
                    help="partition-of-unity bumps instead of localized key rates")
    ap.add_argument("--floor-mode", default="fitted",
                    choices=["fitted", "seasoned-fit", "pinned-seasoned",
                             "pinned-fixed"],
                    help="terminal S-curve floor specification")
    args = ap.parse_args()
    FLOOR_MODE = args.floor_mode
    print("floor mode: %s" % FLOOR_MODE)
    main(args.pmms_key, args.spanning)
