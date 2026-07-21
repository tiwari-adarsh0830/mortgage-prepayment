"""
Duration-hedge diagnostic.

stage3_der_factor_shocks.load_excess_returns() hedges every coupon with a single
constant D_MOD_AVG = 6.5 (blended 5y/10y modified duration). If that misses the
true per-coupon duration, residual rate exposure survives in excess_return and
can be misattributed to the prepayment factor.

Test: per coupon, regress excess_return on [const, dy5, dy10]. A clean hedge
implies both slopes ~ 0.

Also back out implied effective duration from a univariate regression on
dy_avg: excess ~ -(D_c - D_MOD_AVG)*dy/100, so D_c = D_MOD_AVG - 100*coef.
Expect D_c to fall with coupon if the fixed hedge is the problem.

Output: outputs/hedge_diagnostic.json
"""
import os, json
import numpy as np
import pandas as pd

import stage3_der_factor_shocks as base

OUT, DATA = base.OUT, base.DATA


def ols(y, X):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    if n <= k:
        return coef, np.full(k, np.nan), np.nan
    s2 = float(resid @ resid) / (n - k)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 1e-12 else np.nan
    return coef, se, r2


def load_rate_changes():
    treas = pd.read_excel(os.path.join(DATA, "treasury_yields_clean.xlsx"),
                          sheet_name="Treasury_Yields", header=1)
    treas.columns = [str(c).strip() for c in treas.columns]
    treas["Date"] = pd.to_datetime(treas["Date"])
    treas = treas.sort_values("Date").reset_index(drop=True)
    y5 = [c for c in treas.columns if "5yr" in c.lower()][0]
    y10 = [c for c in treas.columns if "10yr" in c.lower()][0]
    print(f"  yield columns selected: 5yr -> '{y5}' | 10yr -> '{y10}'")
    if "15" in y5:
        raise SystemExit(f"'{y5}' matched the 5yr search but looks like a 15yr column.")
    treas["dy5"] = treas[y5].diff()
    treas["dy10"] = treas[y10].diff()
    treas["dy_avg"] = ((treas[y5] + treas[y10]) / 2).diff()
    treas["date"] = treas["Date"].apply(base.ym)
    return treas[["date", "dy5", "dy10", "dy_avg"]].dropna()


def run(returns, rates, label):
    df = returns.merge(rates, on="date", how="inner")
    print(f"\n=== {label} ===  n_months={df['date'].nunique()}")
    out = []
    for c, g in df.groupby("coupon"):
        g = g.dropna(subset=["excess_return", "dy5", "dy10"])
        if len(g) < 8:
            continue
        y = g["excess_return"].values
        X = np.column_stack([np.ones(len(g)), g["dy5"].values, g["dy10"].values])
        coef, se, r2 = ols(y, X)
        t5, t10 = coef[1] / se[1], coef[2] / se[2]

        Xa = np.column_stack([np.ones(len(g)), g["dy_avg"].values])
        ca, sea, _ = ols(y, Xa)
        implied_dur = base.D_MOD_AVG - 100.0 * ca[1]

        out.append(dict(coupon=float(c), n=len(g),
                        b_dy5=float(coef[1]), t_dy5=float(t5),
                        b_dy10=float(coef[2]), t_dy10=float(t10),
                        r2=float(r2),
                        b_dy_avg=float(ca[1]), t_dy_avg=float(ca[1] / sea[1]),
                        implied_duration=float(implied_dur)))
    res = pd.DataFrame(out)
    if not res.empty:
        print(res.to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))
        sig = int(((res["t_dy5"].abs() > 2) | (res["t_dy10"].abs() > 2)).sum())
        print(f"  coupons with |t| > 2 on either rate change: {sig} of {len(res)}")
        print(f"  implied duration range: {res['implied_duration'].min():.2f} "
              f"to {res['implied_duration'].max():.2f} (hedge assumes "
              f"{base.D_MOD_AVG:.2f} for all)")
        rho = res["coupon"].corr(res["implied_duration"], method="spearman")
        print(f"  Spearman(coupon, implied_duration) = {rho:.3f}")
    return res.to_dict(orient="records")


if __name__ == "__main__":
    returns = base.load_excess_returns()
    rates = load_rate_changes()

    results = {}
    results["full_available"] = run(returns, rates, "FULL AVAILABLE SAMPLE")

    w = returns[(returns["date"] >= pd.Timestamp("2022-01-01")) &
                (returns["date"] <= pd.Timestamp("2024-12-01"))]
    results["ex_cutoff_2020_window"] = run(w, rates, "EX-CUTOFF_2020 WINDOW (2022-01..2024-12)")

    json.dump(results, open(os.path.join(OUT, "hedge_diagnostic.json"), "w"),
              indent=2, default=str)
    print("\nSaved: outputs/hedge_diagnostic.json")
