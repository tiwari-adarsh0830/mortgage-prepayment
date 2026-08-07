#!/usr/bin/env python3
"""diag_pmms_2yr_regime.py -- is the dy2/PMMS break real, gradual, sharp, or an
alignment/collinearity artifact?

Split-sample found dy2 coef 0.05 (t=0.20) pre-2022, 1.09 (t=2.90) post-2022 --
a sharp apparent regime change at an arbitrary chronological midpoint. Three
checks before trusting that story enough to put in an email:

  1. ROLLING WINDOW. A single midpoint split can manufacture the appearance of
     a break from what is really a gradual drift, or place the break at the
     wrong date. 24-month rolling regression of the dy2 coefficient shows
     whether it's a step change (and around when) or continuous drift.

  2. COLLINEARITY IN THE SECOND HALF. dy5=-2.196 is a large, sign-flipped
     coefficient. If dy2/dy5/dy10 are highly collinear in 2022+ (plausible --
     parallel curve moves during a hiking cycle), OLS coefficients become
     unstable and large in magnitude without genuine economic content. Check
     pairwise correlations and a condition-number proxy in each half.

  3. ALIGNMENT. Phase 22 found the hedge panel's own pmms column is
     INFO-DATE keyed (corr with ret_month pmms lagged one month = 1.0000
     exactly) -- a misaligned spec looked BETTER there, which is the trap to
     rule out here specifically. This script's pmms comes from a different
     source (pmms_monthly.csv keyed by year/month directly, not the panel's
     own pmms column), so it may not have the same issue, but that must be
     checked, not assumed. Test dpmms against dy2 at lag 0 vs lag 1.
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
m = m[m.index >= "2018-01-01"].reset_index().rename(columns={"index": "date"})

def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    se = np.sqrt(np.diag(float(r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(XtX)))
    return co, se

print("=== 1. rolling 24-month dy2 coefficient ===")
print("%10s %8s %6s %10s" % ("window_end", "dy2_coef", "t", "cond_no"))
for i in range(24, len(m)+1, 3):
    w = m.iloc[i-24:i]
    X = np.column_stack([np.ones(len(w)), w.dy2.values, w.dy5.values, w.dy10.values])
    co, se = ols(w.dpmms.values, X)
    cond = np.linalg.cond(X[:, 1:])
    print("%10s %8.3f %6.2f %10.1f" % (w.date.iloc[-1].strftime("%Y-%m"), co[1], co[1]/se[1], cond))

print("\n=== 2. pairwise correlation of dy2/dy5/dy10, each half ===")
half = len(m)//2
for lab, sl in [("first half", slice(0, half)), ("second half", slice(half, len(m)))]:
    c = m.iloc[sl][["dy2","dy5","dy10"]].corr()
    print("  %s (%s..%s):" % (lab, m.date.iloc[sl].iloc[0].strftime("%Y-%m"),
                              m.date.iloc[sl].iloc[-1].strftime("%Y-%m")))
    print(c.round(3).to_string().replace("\n", "\n    "))
    X = m.iloc[sl][["dy2","dy5","dy10"]].values
    print("    condition number: %.1f  (>30 is usually considered concerning)" % np.linalg.cond(X))

print("\n=== 3. alignment: dpmms vs dy2 at lag 0 vs lag 1 (post-2022 only) ===")
m2 = m[m.date >= "2022-02-01"].copy()
m2["dy2_lag1"] = m2.dy2.shift(1)
m2 = m2.dropna(subset=["dy2_lag1"])
for lab, col in [("lag0 (contemporaneous)", "dy2"), ("lag1", "dy2_lag1")]:
    X = np.column_stack([np.ones(len(m2)), m2[col].values])
    co, se = ols(m2.dpmms.values, X)
    print("  %-24s beta=%.3f  t=%.2f" % (lab, co[1], co[1]/se[1]))
print("  if lag1 fits notably better than lag0, the same info-date trap from")
print("  Phase 22 may be present here and the contemporaneous 0.73/1.09 figures")
print("  would need revisiting.")

print("\n=== 4. pmms_monthly.csv date convention check ===")
print(pm.tail(8).to_string())
raw = pd.read_csv(f"{BASE}/data/pmms_monthly.csv")
print("\nreporting_period tail:")
print(raw[["year","month","reporting_period","rate_30yr"]].tail(8).to_string())
