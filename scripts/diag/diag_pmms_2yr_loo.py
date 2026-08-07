#!/usr/bin/env python3
"""diag_pmms_2yr_loo.py -- leave-one-out stability of the post-2018 PMMS/2yr fit.

Post-2018 multivariate dpmms ~ dy2+dy5+dy10 gave dy2 coef 0.732 (t=2.96), dy5
coef -1.158 (t=-2.40) -- sign-flipped relative to the full-sample fit, on
n=99 with three correlated regressors. Before proposing 0.73 as the pass-through
into the pricer, check whether it's stable or driven by a handful of extreme
months (2020 cut-to-zero, 2022-23 hiking are the obvious suspects).

Two checks: (1) leave-one-out refit, report the range and which single month
moves the coefficient most; (2) split the 99 months in half chronologically and
refit each half separately -- if the two halves disagree sharply, this isn't a
stable relationship to hang a pricer constant on.
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

y = m.dpmms.values
X = np.column_stack([np.ones(len(m)), m.dy2.values, m.dy5.values, m.dy10.values])
co_full, se_full = ols(y, X)
print("full post-2018 fit: dy2=%.3f (t=%.2f)  dy5=%.3f (t=%.2f)  dy10=%.3f (t=%.2f)  n=%d"
      % (co_full[1], co_full[1]/se_full[1], co_full[2], co_full[2]/se_full[2],
         co_full[3], co_full[3]/se_full[3], len(m)))

print("\n=== leave-one-out: dy2 coefficient range ===")
loo = []
for i in range(len(m)):
    yi = np.delete(y, i)
    Xi = np.delete(X, i, axis=0)
    co, _ = ols(yi, Xi)
    loo.append(co[1])
loo = np.array(loo)
print("  min %.3f  max %.3f  mean %.3f  std %.3f" % (loo.min(), loo.max(), loo.mean(), loo.std()))
worst = np.argmax(np.abs(loo - co_full[1]))
print("  most influential single month: %s (removing it moves dy2 coef from %.3f to %.3f)"
      % (m.date.iloc[worst].strftime("%Y-%m"), co_full[1], loo[worst]))
print("  dy2, dy5, dy10 that month: %.3f  %.3f  %.3f"
      % (m.dy2.iloc[worst], m.dy5.iloc[worst], m.dy10.iloc[worst]))

print("\n=== chronological split-sample ===")
half = len(m) // 2
for lab, sl in [("first half  (%s..%s)" % (m.date.iloc[0].strftime("%Y-%m"), m.date.iloc[half-1].strftime("%Y-%m")), slice(0, half)),
                ("second half (%s..%s)" % (m.date.iloc[half].strftime("%Y-%m"), m.date.iloc[-1].strftime("%Y-%m")), slice(half, len(m)))]:
    ys, Xs = y[sl], X[sl]
    co, se = ols(ys, Xs)
    print("  %-28s dy2=%6.3f (t=%5.2f)  dy5=%6.3f (t=%5.2f)  dy10=%6.3f (t=%5.2f)  n=%d"
          % (lab, co[1], co[1]/se[1], co[2], co[2]/se[2], co[3], co[3]/se[3], sl.stop-sl.start))
