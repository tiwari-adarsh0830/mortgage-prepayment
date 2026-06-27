"""
DER (JF 2021) Equation 22 — baseline price-of-prepayment-risk regression.
No statsmodels dependency — uses numpy OLS + manual time-clustered SEs.

Verified specification (JF 2021, Eq. 22):
    R^e_{i,t} = a + kappa*(r - c_i) + delta*(r - c_i)*(%disc_t - 50%) + eps

Regressors:
  rel         = pmms - coupon          (negative relative coupon, r - c_i)
  interaction = rel * disc_centered    (disc_centered = price_disc_share - 0.5)
  intercept   = 1

Standard errors: plain HC0 + time-clustered (DER's preferred).
Time-clustered SE: sum squared within-month score contributions.

One departure from DER: their %RPB_disc uses eMBS balance weights; we use
equal-weighted price<100 count across 9 coupons. Sign/significance directly
comparable; magnitude indicative.

DER Table VIII benchmark: delta = +0.11..0.16%/mo, t(time-clustered) = 2.18..2.42
"""
import numpy as np
import pandas as pd

PANEL = "outputs/stage3_excess_returns.csv"

def ols(X, y):
    XtX = X.T @ X
    XtXinv = np.linalg.inv(XtX)
    coef = XtXinv @ X.T @ y
    resid = y - X @ coef
    return coef, resid, XtXinv

def se_plain(X, resid, XtXinv):
    meat = X.T @ (resid[:, None]**2 * X)
    V = XtXinv @ meat @ XtXinv
    return np.sqrt(np.diag(V))

def se_time_cluster(X, resid, XtXinv, groups):
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        idx = groups == g
        score_g = (X[idx] * resid[idx, None]).sum(0)
        meat += np.outer(score_g, score_g)
    V = XtXinv @ meat @ XtXinv
    return np.sqrt(np.diag(V))

def run(sub, label):
    sub = sub.copy()
    sub["rel"] = sub["pmms"] - sub["coupon"]
    sub["is_disc"] = (sub["price"] < 100).astype(float)
    share = sub.groupby("Date")["is_disc"].transform("mean")
    sub["disc_centered"] = share - 0.5
    sub["interaction"] = sub["rel"] * sub["disc_centered"]
    sub = sub.dropna(subset=["excess_return","rel","interaction"])

    y = sub["excess_return"].values
    X = np.column_stack([np.ones(len(sub)), sub["rel"].values, sub["interaction"].values])
    groups = sub["Date"].values

    coef, resid, XtXinv = ols(X, y)
    se_p  = se_plain(X, resid, XtXinv)
    se_tc = se_time_cluster(X, resid, XtXinv, groups)
    rsq   = 1 - np.var(resid) / np.var(y)

    names = ["intercept", "kappa (r-c)", "delta (interaction)"]
    print(f"\n=== DER Eq.22 — {label} ===")
    print(f"  n={len(sub)} obs, {sub['Date'].nunique()} months, R²={rsq:.3f}")
    print(f"  {'':22s} {'coef (%/mo)':>12} {'t(plain)':>10} {'t(time-clust)':>14}")
    for i, nm in enumerate(names):
        c_pct = coef[i] * 100
        print(f"  {nm:22s} {c_pct:+12.4f}% {coef[i]/se_p[i]:+10.2f} {coef[i]/se_tc[i]:+14.2f}")

    delta_pct = coef[2] * 100
    t_tc = coef[2] / se_tc[2]
    print(f"\n  *** OUR delta = {delta_pct:+.4f}%/mo  t(time-clust) = {t_tc:+.2f} ***")
    print(f"  DER Table VIII:  delta = +0.11..0.16%/mo  t(time-clust) = 2.18..2.42")
    print(f"  DER prediction:  delta > 0  ->  {'CORRECT SIGN ✓' if delta_pct > 0 else 'WRONG SIGN ✗'}")

def main():
    d = pd.read_csv(PANEL)
    d["Date"] = pd.to_datetime(d["Date"])
    run(d, "FULL SAMPLE 2018-2026")
    run(d[d["Date"] >= "2020-01-01"], "2020-2026 (factor-shock window)")
    print("\nNOTE: discount share is price-based (price<100), equal-weighted")
    print("across 9 coupons — proxy for DER's eMBS RPB-weighted share.")
    print("Sign/significance directly comparable; magnitude indicative.")

if __name__ == "__main__":
    main()
