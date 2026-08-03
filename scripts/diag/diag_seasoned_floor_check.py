"""
Is the seasoned floor of 0.0700 real, or a fitting artifact?

The full-range seasoned fit returns floor=0.0700 (+28.2% vs all-ages 0.0546,
2.05 SE apart). But the fitted-vs-realized table shows the seasoned curve
OVERSHOOTS actual seasoned CPR at depth:

    inc      realized_seasoned    fitted_seasoned
    -3.75          0.0501              0.0700
    -3.25          0.0496              0.0700
    -2.75          0.0543              0.0700

Realized seasoned CPR below -2.75 (0.0496-0.0543) is closer to the ALL-AGES
floor (0.0546) than to the fitted seasoned floor (0.0700). The floor is a
horizontal asymptote, so it should be pinned by the deep-discount data -- but
seasoned counts below -1.5 are IDENTICAL to all-ages (28/40/41/44/53), meaning
the seasoned restriction removes nothing at depth. The floor moved because the
fit changed shape in the -1.5..0 region, where seasoned counts do drop
(86->69, 122->86, 137->100).

So the parameter may be absorbing curvature from the mid-range rather than
describing deep-discount seasoned prepayment.

This script estimates the floor DIRECTLY from the deep-discount region and
compares:
  1. direct mean of realized CPR below a depth threshold (unweighted and
     UPB-weighted -- cells differ hugely in balance, and the production fit
     is unweighted, so this also tests whether weighting matters here)
  2. the full-range fitted floor
  3. a fit restricted to inc <= -1.0, where the logistic is near its asymptote
Bootstrap SEs on each so the comparison is against sampling noise, not eyeball.

If the direct estimate sits near the all-ages floor while the full-range fit
says 0.0700, the headline is "the seasoned fit overshoots at depth", NOT
"seasoned prepayment is 28% higher".
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

GFEE = 0.50
BYAGE = 'outputs/realized_cpr_by_coupon_v6_upb_byage.csv'
CPN_LO, CPN_HI = 2.5, 6.5
INC_LO, INC_HI = -4.0, 2.0
DEPTHS = [-2.5, -2.0, -1.5]
NBOOT = 500
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


def build(df):
    g = (df.groupby(['coupon_bucket', 'implied_mbs_coupon', 'yyyymm'], as_index=False)
           [['upb_atrisk', 'upb_prepay']].sum())
    g = g[g['upb_atrisk'] > 0].copy()
    g['smm'] = g['upb_prepay'] / g['upb_atrisk']
    g['cpr'] = 1.0 - (1.0 - g['smm']) ** 12
    g['date'] = pd.to_datetime(g['yyyymm'].astype(str), format='%Y%m')
    g['pmms'] = g['date'].map(load_pmms())
    g['inc'] = (g['implied_mbs_coupon'] + GFEE) - g['pmms']
    g = g.dropna(subset=['inc', 'cpr'])
    g = g[(g['inc'] >= INC_LO) & (g['inc'] <= INC_HI)]
    g = g[(g['implied_mbs_coupon'] >= CPN_LO) & (g['implied_mbs_coupon'] <= CPN_HI)]
    return g


def fit_floor(x, y, seed=SEED, nboot=NBOOT):
    p0 = [0.05, 0.25, 3.0, 0.45]
    bd = ([0.0, 0.0, 0.1, -3.0], [0.5, 1.0, 20.0, 3.0])
    if len(y) < 20 or float(((y - y.mean()) ** 2).sum()) == 0:
        return np.nan, np.nan, np.nan
    try:
        p, _ = curve_fit(scurve, x, y, p0=p0, bounds=bd, maxfev=40000)
    except Exception:
        return np.nan, np.nan, np.nan
    r = y - scurve(x, *p)
    r2 = 1 - float(r @ r) / float(((y - y.mean()) ** 2).sum())
    rng = np.random.default_rng(seed)
    fs = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x))
        yb = y[i]
        if float(((yb - yb.mean()) ** 2).sum()) == 0:
            continue
        try:
            pb, _ = curve_fit(scurve, x[i], yb, p0=p0, bounds=bd, maxfev=40000)
            fs.append(pb[0])
        except Exception:
            pass
    return p[0], (float(np.std(fs)) if len(fs) > 20 else np.nan), r2


def boot_mean(v, w=None, seed=SEED, nboot=NBOOT):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    if w is None:
        m = v.mean()
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(nboot)]
    else:
        w = np.asarray(w, float)
        m = float((v * w).sum() / w.sum())
        bs = []
        for _ in range(nboot):
            i = rng.integers(0, len(v), len(v))
            bs.append(float((v[i] * w[i]).sum() / w[i].sum()))
    return m, float(np.std(bs))


d = pd.read_csv(BYAGE)
d = d[d['age_group'] >= 0]
samples = {'all ages': build(d), 'seasoned': build(d[d['age_group'] >= 60])}

print("=" * 84)
print("DIRECT FLOOR ESTIMATE FROM THE DEEP-DISCOUNT REGION")
print("=" * 84)
print("%-10s %8s %6s %12s %10s %12s %10s" %
      ("sample", "depth", "n", "mean_cpr", "SE", "upbw_cpr", "SE"))
direct = {}
for name, g in samples.items():
    for dep in DEPTHS:
        s = g[g['inc'] <= dep]
        if len(s) < 10:
            print("%-10s %8.1f %6d   too few rows" % (name, dep, len(s)))
            continue
        m, se = boot_mean(s['cpr'].values)
        mw, sew = boot_mean(s['cpr'].values, s['upb_atrisk'].values)
        direct[(name, dep)] = (m, se, mw, sew)
        print("%-10s %8.1f %6d %12.4f %10.4f %12.4f %10.4f" %
              (name, dep, len(s), m, se, mw, sew))

print("\n" + "=" * 84)
print("FITTED FLOOR: full range vs restricted to inc <= -1.0")
print("=" * 84)
print("%-10s %-22s %6s %10s %10s %8s" %
      ("sample", "spec", "n", "floor", "SE", "R2"))
fitted = {}
for name, g in samples.items():
    for tag, sub in [("full range -4..+2", g),
                     ("restricted inc<=-1.0", g[g['inc'] <= -1.0])]:
        f, se, r2 = fit_floor(sub['inc'].values, sub['cpr'].values)
        fitted[(name, tag)] = (f, se)
        print("%-10s %-22s %6d %10.4f %10.4f %8.3f" %
              (name, tag, len(sub), f, se, r2))

print("\n" + "=" * 84)
print("VERDICT")
print("=" * 84)
fa = fitted[('all ages', 'full range -4..+2')][0]
fs, fsse = fitted[('seasoned', 'full range -4..+2')]
print("all-ages fitted floor (full range)      : %.4f" % fa)
print("seasoned fitted floor (full range)      : %.4f +/- %.4f" % (fs, fsse))
key = ('seasoned', -2.5)
if key in direct:
    m, se, mw, sew = direct[key]
    print("seasoned REALIZED mean, inc <= -2.5     : %.4f +/- %.4f" % (m, se))
    print("seasoned REALIZED mean, UPB-weighted    : %.4f +/- %.4f" % (mw, sew))
    gap = fs - m
    print("\nfitted seasoned floor minus realized    : %+.4f (%.1f SE of realized)" %
          (gap, gap / se if se > 0 else np.nan))
    if abs(fs - m) > 2 * se:
        print("=> the full-range fit does NOT match realized seasoned CPR at depth.")
        print("   The 0.0700 floor is a fitting artifact, not a deep-discount fact.")
    else:
        print("=> the full-range fit is consistent with realized seasoned CPR at depth.")
    print("\nrealized seasoned (%.4f) vs all-ages fitted floor (%.4f): diff %+.4f"
          % (m, fa, m - fa))
