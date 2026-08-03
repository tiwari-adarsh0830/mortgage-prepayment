"""
Patch model_hedge_krd.py to support three terminal-floor specifications.

Run from /scratch/at7095/mortgage_prepayment. Every replacement asserts exactly
one match; a .bak is written first, so a failed assertion leaves the file
untouched.

v2: the previous attempt used regex to find the argparse block and mangled a
multi-line add_argument call. This version anchors on exact strings only.
It also puts the floor mode into the output filename tag -- without that, all
three runs would overwrite the same panel CSV and the comparison would be
meaningless.

SPECS
  fitted            Current default, behaviour unchanged. Full-range logistic
                    on all-ages realized CPR, coupons 2.5-6.5, expanding
                    window. floor ~0.0546.

  seasoned-fit      The advisor's literal request: same fit, on realized CPR
                    restricted to loans aged > 60 months. Produces floor
                    ~0.0700, which the floor-check diagnostic showed is a
                    fitting artifact -- the seasoned sample is IDENTICAL to
                    all-ages below inc -1.5 (28/40/41/44/53 obs), so the
                    restriction removes nothing where the floor is identified;
                    it only thins the mid-range and distorts curvature.
                    Included so the claim is measured, not asserted.

  pinned-seasoned   Two-stage. The floor is a horizontal asymptote and should
                    be identified by deep-discount data, so estimate it
                    directly as mean realized seasoned CPR at inc <= -2.5
                    within the expanding window, then fit (sat, k, x0) on the
                    full range with the floor held fixed.
                    Seasoned is the right anchor because the terminal covers
                    months 34-360, i.e. ages 34+. Realized seasoned
                    deep-discount CPR is 0.0516 vs the production fit's 0.0546.

All three preserve the expanding window (data strictly before the cutoff), so
no look-ahead is introduced in any spec.
"""
import os
import shutil

P = "scripts/model_hedge_krd.py"
assert os.path.exists(P), "run from /scratch/at7095/mortgage_prepayment"

src = open(P).read()


def rep(s, old, new, label):
    n = s.count(old)
    assert n == 1, "%s: expected 1 match, got %d" % (label, n)
    return s.replace(old, new)


# ---------------------------------------------------------------- 1. globals
old1 = "_SCURVE_EXP = {}\ndef scurve_params_asof(cutoff):"
new1 = '''_SCURVE_EXP = {}
FLOOR_MODE = "fitted"          # fitted | seasoned-fit | pinned-seasoned
_SEASONED = None
DEEP_INC = -2.5                # depth at which the pinned floor is estimated


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


def scurve_params_asof(cutoff):'''
src = rep(src, old1, new1, "globals")

# ------------------------------------------------------- 2. cache key by mode
src = rep(src,
          "    if cutoff in _SCURVE_EXP:\n        return _SCURVE_EXP[cutoff]",
          "    _key = (cutoff, FLOOR_MODE)\n    if _key in _SCURVE_EXP:\n        return _SCURVE_EXP[_key]",
          "cache-get")

src = rep(src,
          "    _SCURVE_EXP[cutoff] = q\n    return q",
          "    _SCURVE_EXP[_key] = q\n    return q",
          "cache-set")

# ------------------------------------------------------------- 3. the fitting
old4 = '''    po, _ = _cf(_sc, h["inc"].values, h["cpr_upb"].values, p0=[0.04, 0.22, 3.0, 0.4],
                bounds=([0.005, 0.05, 0.2, -2.0], [0.15, 0.60, 10.0, 3.0]), maxfev=40000)
    q = dict(floor=float(po[0]), sat=float(po[1]), k=float(po[2]), x0=float(po[3]),
             n=int(len(h)))'''
new4 = '''    if FLOOR_MODE == "seasoned-fit":
        _s = _load_seasoned()
        h = _s[(_s["date"] < _pd.Timestamp(cutoff + "-01"))
               & (_s["inc"] >= -4.0) & (_s["inc"] <= 2.0)
               & (_s["implied_mbs_coupon"] >= 2.5)
               & (_s["implied_mbs_coupon"] <= 6.5)]
        if len(h) < 40:
            raise ValueError("insufficient seasoned history before %s (n=%d)"
                             % (cutoff, len(h)))

    if FLOOR_MODE == "pinned-seasoned":
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
                 n=int(len(h)))'''
src = rep(src, old4, new4, "fit")

# --------------------------------------------- 4. output tag carries the mode
old5 = '    tag = pmms_key.replace("yr", "") + ("_span" if spanning else "_local")'
assert src.count(old5) == 1, "tag: expected 1 match, got %d" % src.count(old5)
new5 = ('    tag = pmms_key.replace("yr", "") + ("_span" if spanning else "_local")\n'
        '    if FLOOR_MODE != "fitted":\n'
        '        tag = tag + "_" + FLOOR_MODE.replace("-", "")')
src = src.replace(old5, new5)

# ------------------------------------------------------- 5. CLI arg + dispatch
old6 = "    args = ap.parse_args()\n    main(args.pmms_key, args.spanning)"
new6 = ('    ap.add_argument("--floor-mode", default="fitted",\n'
        '                    choices=["fitted", "seasoned-fit", "pinned-seasoned"],\n'
        '                    help="terminal S-curve floor specification")\n'
        "    args = ap.parse_args()\n"
        "    FLOOR_MODE = args.floor_mode\n"
        '    print("floor mode: %s" % FLOOR_MODE)\n'
        "    main(args.pmms_key, args.spanning)")
src = rep(src, old6, new6, "cli")

shutil.copy(P, P + ".bak")
open(P, "w").write(src)
print("patched %s (backup at %s.bak)" % (P, P))
print("modes: fitted | seasoned-fit | pinned-seasoned")
print("output tags: _span (fitted) | _span_seasonedfit | _span_pinnedseasoned")
