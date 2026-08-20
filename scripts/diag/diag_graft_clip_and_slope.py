"""Two checks on the S-curve graft result.

1. Is the [0, 0.99] clip binding? If it is, the graft is truncated and the
   duration ratio is partly an artifact of the clip rather than the response.
2. How much does each source actually move per 25bp? This is the direct answer
   to "how much the CPR moves when the curve is bumped" -- the model's mean
   response over months 1-33 vs the realized S-curve's.
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_hedge_krd as M

# terminal_cpr depends on FLOOR_MODE via scurve_params_asof. It defaults to
# "fitted" at module level and is only set inside main(), which importing
# bypasses, so match the panel run explicitly.
M.FLOOR_MODE = "pinned-fixed"
print("floor mode: %s" % M.FLOOR_MODE, flush=True)
model, scaler, a, b = M.load_hazard()
LAB = M.MAT_LABELS
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
h = M.BUMP_BP / 100.0

nclip = ntot = 0; dmin, dmax = 9.0, -9.0
rows = []
for c in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]:
    ms, ss = [], []
    for p in months:
        pv = float(pms.loc[p]); asof = str(p); note = c + M.GFEE
        i0 = note - pv
        for sgn in (+1, -1):
            i1 = note - (pv + sgn * h)
            dlt = M.terminal_cpr(i1, asof=asof) - M.terminal_cpr(i0, asof=asof)
            raw = M.cpr_path(i0, model, scaler, a, b) + dlt
            nclip += int(((raw < 0) | (raw > 0.99)).sum()); ntot += raw.size
            dmin = min(dmin, float(raw.min())); dmax = max(dmax, float(raw.max()))
        i1 = note - (pv - h)                       # -25bp: rates down
        ss.append(M.terminal_cpr(i1, asof=asof) - M.terminal_cpr(i0, asof=asof))
        ms.append(float((M.cpr_path(i1, model, scaler, a, b)
                         - M.cpr_path(i0, model, scaler, a, b)).mean()))
    rows.append((c, float(np.mean(ms)), float(np.mean(ss))))

print("clip binding: %d / %d elements (%.4f%%)" % (nclip, ntot, 100.0*nclip/ntot))
print("raw grafted CPR range: %.4f .. %.4f\n" % (dmin, dmax))
print("=" * 56)
print("CPR RESPONSE PER -25bp  (rates down, incentive up)")
print("=" * 56)
print("%6s%14s%14s%10s" % ("cpn", "model 1-33", "S-curve", "ratio"))
for c, m, s in rows:
    print("%6.1f%14.5f%14.5f%10.3f" % (c, m, s, s/m if abs(m) > 1e-12 else float('nan')))
am = float(np.mean([m for _, m, _ in rows])); asv = float(np.mean([s for _, _, s in rows]))
print("-" * 56)
print("%6s%14.5f%14.5f%10.3f" % ("MEAN", am, asv, asv/am))
