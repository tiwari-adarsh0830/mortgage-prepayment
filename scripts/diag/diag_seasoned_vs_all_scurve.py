"""
Seasoned-only vs all-ages terminal S-curve: go/no-go diagnostic.

Advisor asked whether the terminal curve should be fitted to seasoned loans
only. The age-keyed aggregation (realized_cpr_by_coupon_v6_upb_byage.csv) makes
that possible. Seasoned share is 14.05% of at-risk UPB across coupons 2.5-6.5,
and the deep-discount incentive buckets that set the FLOOR were already thin in
the full sample (n=44-54 below incentive -2.25). So the question this answers is
not just "what is the seasoned floor" but "is it estimable at all".

Fits the same logistic used in the pipeline

    CPR(inc) = floor + (sat - floor) / (1 + exp(-k*(inc - x0)))

on the same scope as the current production fit (coupons 2.5-6.5, incentive
-4..+2) for three samples: all ages, seasoned (age > 60mo), and unseasoned.

METHOD NOTE: seasoned CPR is rebuilt by summing upb_prepay and upb_atrisk over
the seasoned age levels and recomputing SMM/CPR. Averaging the per-level cpr_upb
would weight a thin 120+ cell equally with a large <60 cell.

Reports floor / sat / x0 / k / R2 / n for each, per-bucket counts in the
discount region, and a bootstrap SE on the floor -- the floor is the parameter
the whole exercise turns on, and a point estimate without a SE cannot answer
whether the seasoned fit is usable.

Baseline to compare against (README, all-ages, coupons 2.5-6.5):
    floor=0.0546  sat=0.2492  x0=0.493  R2=0.515  n=1037
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

GFEE = 0.50
BYAGE = 'outputs/realized_cpr_by_coupon_v6_upb_byage.csv'
INC_LO, INC_HI = -4.0, 2.0
CPN_LO, CPN_HI = 2.5, 6.5
SEED = 0


def scurve(x, floor, sat, k, x0):
    return floor + (sat - floor) / (1.0 + np.exp(-k * (x - x0)))


def load_pmms():
    pm = pd.read_csv('data/pmms_monthly.csv')

    def parse(x):
        s = str(int(x))
        if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
        if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
        return pd.NaT

    pm['date'] = pm['reporting_period'].apply(parse)
    return pm.dropna(subset=['date']).set_index('date')['rate_30yr']


def build(df, label):
    """Collapse age levels, recompute CPR from summed UPB, attach incentive."""
    g = (df.groupby(['coupon_bucket', 'implied_mbs_coupon', 'yyyymm'], as_index=False)
           [['upb_atrisk', 'upb_prepay', 'n_atrisk', 'n_prepay']].sum())
    g = g[g['upb_atrisk'] > 0].copy()
    g['smm'] = g['upb_prepay'] / g['upb_atrisk']
    g['cpr'] = 1.0 - (1.0 - g['smm']) ** 12
    g['date'] = pd.to_datetime(g['yyyymm'].astype(str), format='%Y%m')
    g['pmms'] = g['date'].map(load_pmms())
    g['inc'] = (g['implied_mbs_coupon'] + GFEE) - g['pmms']
    g = g.dropna(subset=['inc', 'cpr'])
    g = g[(g['inc'] >= INC_LO) & (g['inc'] <= INC_HI)]
    g = g[(g['implied_mbs_coupon'] >= CPN_LO) & (g['implied_mbs_coupon'] <= CPN_HI)]
    g['sample'] = label
    return g


def fit(g, seed=SEED, n_boot=200):
    """Returns (params, r2, floor_se, error_string). Never raises."""
    x, y = np.asarray(g['inc'].values, float), np.asarray(g['cpr'].values, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 20:
        return None, np.nan, np.nan, "only %d usable rows" % len(y)
    den = float(((y - y.mean()) ** 2).sum())
    if den == 0.0:
        return None, np.nan, np.nan, "cpr has zero variance across %d rows" % len(y)

    p0 = [0.05, 0.25, 3.0, 0.45]
    bounds = ([0.0, 0.0, 0.1, -3.0], [0.5, 1.0, 20.0, 3.0])
    try:
        p, _ = curve_fit(scurve, x, y, p0=p0, bounds=bounds, maxfev=40000)
    except Exception as e:
        return None, np.nan, np.nan, "curve_fit: %s" % e
    r = y - scurve(x, *p)
    r2 = 1 - float(r @ r) / den

    rng = np.random.default_rng(seed)
    floors = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        yb = y[idx]
        if float(((yb - yb.mean()) ** 2).sum()) == 0.0:
            continue
        try:
            pb, _ = curve_fit(scurve, x[idx], yb, p0=p0, bounds=bounds, maxfev=40000)
            floors.append(pb[0])
        except Exception:
            pass
    fse = float(np.std(floors)) if len(floors) > 20 else np.nan
    return p, r2, fse, None


d = pd.read_csv(BYAGE)
d = d[d['age_group'] >= 0]          # drop missing-age level

samples = {
    'all ages':   build(d, 'all ages'),
    'seasoned':   build(d[d['age_group'] >= 60], 'seasoned'),
    'unseasoned': build(d[d['age_group'] == 0], 'unseasoned'),
}

print("=" * 82)
print("SAMPLE DIAGNOSTICS (before fitting)")
print("=" * 82)
print("%-12s %8s %12s %12s %10s %10s" % ("sample","n","var(cpr)","var(inc)","cpr_min","cpr_max"))
for _nm, _g in samples.items():
    _y = np.asarray(_g['cpr'].values, float); _xx = np.asarray(_g['inc'].values, float)
    _v = float(((_y-_y.mean())**2).sum()) if len(_y) else float('nan')
    _vx = float(((_xx-_xx.mean())**2).sum()) if len(_xx) else float('nan')
    print("%-12s %8d %12.6g %12.6g %10.4f %10.4f" %
          (_nm, len(_g), _v, _vx,
           _y.min() if len(_y) else float('nan'),
           _y.max() if len(_y) else float('nan')))
print()
print("=" * 82)
print("TERMINAL S-CURVE FIT: seasoned vs all ages")
print("scope: coupons %.1f-%.1f, incentive %.1f..%.1f" % (CPN_LO, CPN_HI, INC_LO, INC_HI))
print("=" * 82)
print("%-12s %8s %8s %8s %8s %8s %7s %10s" %
      ("sample", "n", "floor", "floorSE", "sat", "x0", "k", "R2"))

fits = {}
for name, g in samples.items():
    p, r2, fse, err = fit(g)
    fits[name] = (p, r2, fse)
    if p is None:
        print("%-12s %8d   FIT FAILED: %s" % (name, len(g), err))
        continue
    print("%-12s %8d %8.4f %8.4f %8.4f %8.3f %7.2f %10.3f" %
          (name, len(g), p[0], fse, p[1], p[3], p[2], r2))

print("\nREADME baseline (all ages, 2.5-6.5): floor=0.0546 sat=0.2492 x0=0.493 R2=0.515 n=1037")

print("\n" + "=" * 82)
print("DISCOUNT-REGION SUPPORT (where the floor is identified)")
print("=" * 82)
bins = np.arange(INC_LO, 0.5, 0.5)
rows = []
for name, g in samples.items():
    c = pd.cut(g['inc'], bins=bins)
    rows.append(g.groupby(c, observed=True).size().rename(name))
sup = pd.concat(rows, axis=1).fillna(0).astype(int)
print(sup.to_string())

print("\n" + "=" * 82)
print("FITTED vs REALIZED IN THE DISCOUNT REGION")
print("=" * 82)
print("%8s %10s %10s %10s %10s" %
      ("inc", "real_all", "fit_all", "real_seas", "fit_seas"))
ga, gs = samples['all ages'], samples['seasoned']
pa, ps = fits['all ages'][0], fits['seasoned'][0]
for lo in np.arange(INC_LO, 0.5, 0.5):
    hi = lo + 0.5
    ma = ga[(ga['inc'] >= lo) & (ga['inc'] < hi)]
    ms = gs[(gs['inc'] >= lo) & (gs['inc'] < hi)]
    if len(ma) == 0 and len(ms) == 0:
        continue
    mid = lo + 0.25
    print("%8.2f %10s %10.4f %10s %10.4f" % (
        mid,
        ("%.4f" % ma['cpr'].mean()) if len(ma) else "    --",
        scurve(mid, *pa) if pa is not None else np.nan,
        ("%.4f" % ms['cpr'].mean()) if len(ms) else "    --",
        scurve(mid, *ps) if ps is not None else np.nan))

if fits['seasoned'][0] is not None and fits['all ages'][0] is not None:
    fa, fs = fits['all ages'][0][0], fits['seasoned'][0][0]
    sea, sfse = fits['seasoned'][0][0], fits['seasoned'][2]
    print("\nfloor: all ages %.4f -> seasoned %.4f (%+.1f%%)" %
          (fa, fs, 100 * (fs - fa) / fa))
    if np.isfinite(sfse):
        print("seasoned floor %.4f +/- %.4f  =>  %s" %
              (sea, sfse,
               "distinguishable from all-ages floor" if abs(sea - fa) > 2 * sfse
               else "NOT distinguishable from all-ages floor at 2 SE"))
