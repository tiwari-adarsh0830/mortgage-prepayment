#!/usr/bin/env python3
"""diag_pmms_2yr_sensitivity.py -- empirical PMMS sensitivity to 2yr, 5yr, 10yr.

Found: krd2 has ~0 correlation with realized dy2 because the 2yr bump was
designed to never move PMMS (my assumption -- PMMS is a 30yr survey rate
tracking the long end). That cuts off the only channel (CPR response via
incentive = note - pmms) through which krd2 could reflect prepayment risk at
all, leaving it a small, mechanically real but economically disconnected
discounting-only Greek (9% of total duration at coupon 2.5, rising to 88% at
6.5 -- but that rise tracks krd5+krd10 collapsing toward zero at high coupons,
not genuine 2yr sensitivity).

Rather than re-guess the PMMS/2yr relationship, this regresses monthly PMMS
changes directly on the three curve legs, same spirit as the existing
PMMS-10yr spread work (Phase 22). If PMMS empirically moves materially with
the 2yr leg, that argues for a fractional pass-through (dp = beta_2yr * h)
rather than the current dp=0. If it doesn't, dp=0 was the right call and the
krd2 finding stands as a genuine (if small) discounting-only factor.
"""
import pandas as pd
import numpy as np

BASE = "/scratch/at7095/mortgage_prepayment"

pm = pd.read_csv(f"{BASE}/data/pmms_monthly.csv")
pm["date"] = pd.to_datetime(pm.year.astype(str) + "-" + pm.month.astype(str).str.zfill(2))
pm = pm.set_index("date")[["rate_30yr"]].sort_index()

t = pd.read_csv(f"{BASE}/data/treasury_yields.csv", parse_dates=["DATE"])
me = t.set_index("DATE")[["2yr", "5yr", "10yr"]].sort_index().resample("ME").last()
me.index = me.index.to_period("M").to_timestamp()

m = pm.join(me, how="inner").dropna()
m["dpmms"] = m.rate_30yr.diff()
m["dy2"] = m["2yr"].diff()
m["dy5"] = m["5yr"].diff()
m["dy10"] = m["10yr"].diff()
m = m.dropna()

print("sample: %s .. %s, n=%d" % (m.index.min().date(), m.index.max().date(), len(m)))

def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    se = np.sqrt(np.diag(float(r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(XtX)))
    r2 = 1 - float(r @ r) / float(((y - y.mean()) ** 2).sum())
    return co, se, r2

print("\n=== univariate: dpmms on each leg alone ===")
for c in ["dy2", "dy5", "dy10"]:
    X = np.column_stack([np.ones(len(m)), m[c].values])
    co, se, r2 = ols(m.dpmms.values, X)
    print("  %-4s  beta=%.3f  t=%.2f  r2=%.3f" % (c, co[1], co[1]/se[1], r2))

print("\n=== multivariate: dpmms on all three jointly ===")
X = np.column_stack([np.ones(len(m)), m.dy2.values, m.dy5.values, m.dy10.values])
co, se, r2 = ols(m.dpmms.values, X)
for i, c in enumerate(["const", "dy2", "dy5", "dy10"]):
    print("  %-6s  coef=%.4f  t=%6.2f" % (c, co[i], co[i]/se[i] if i else float("nan")))
print("  r2=%.3f" % r2)

print("\n=== same regression, post-2018 only (matches the hedge panel's own window) ===")
m2 = m[m.index >= "2018-01-01"]
X2 = np.column_stack([np.ones(len(m2)), m2.dy2.values, m2.dy5.values, m2.dy10.values])
co2, se2, r22 = ols(m2.dpmms.values, X2)
for i, c in enumerate(["const", "dy2", "dy5", "dy10"]):
    print("  %-6s  coef=%.4f  t=%6.2f" % (c, co2[i], co2[i]/se2[i] if i else float("nan")))
print("  r2=%.3f  n=%d" % (r22, len(m2)))
