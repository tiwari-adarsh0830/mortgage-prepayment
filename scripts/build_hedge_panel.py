"""
Per-coupon level/slope rate hedge — builder + validation.

WHY
  load_excess_returns() in stage3_der_factor_shocks.py hedges every coupon with
  a single constant D_MOD_AVG = 6.5 years on the blended 5y/10y change. That
  leaves residual rate exposure (per hedge_diagnostic.json). Advisor's basis:
  regress on level = (dy5+dy10)/2 and slope = (dy10-dy5) instead of the two raw
  tenors, which are ~0.98 correlated. In that basis level and slope are ~-0.47
  correlated and both identify cleanly per coupon.

  NOTE: level == dy_avg exactly (difference of an average = average of the
  differences), so the level durations here are the same univariate durations
  already reported (8.05 -> 1.97).

WHAT THIS DOES
  Builds excess returns under TWO per-coupon hedges, in the advisor's basis:
    (A) EMPIRICAL: per-coupon OLS of raw excess on [const, level, slope] over the
        window; fitted (b_level, b_slope) define the hedge. Well-identified in
        this basis, but fit in-sample on the same return series.
    (B) MODEL: (D_level, D_slope) from krd_pricer, beginning-of-month, no
        in-sample fitting. D_level = krd5+krd10, D_slope = (krd10-krd5)/2.
        Compressed (see krd_panel_diag) and near-zero slope.

  Both use the SAME income leg as the current pipeline (mean-yield carry), so the
  ONLY thing that differs from the existing excess_return is the price hedge.
  This isolates the hedge change.

RETURN / INCOME CONSTRUCTION
  Reused verbatim from stage3_der_factor_shocks.load_excess_returns():
    tba_total_return = (P + coupon/12 - P_prev) / P_prev
    income           = mean(y5, y10) / 12 / 100
  Price hedge leg (the part that changes):
    (A) hedge_ret = b_level * d_level + b_slope * d_slope
        excess_A  = tba_total_return - (hedge_ret + income) ... but b_* are FIT on
        excess, so operationally excess_A = OLS residual + intercept-free refit;
        see build. We report the residual rate exposure, which is the test.
    (B) price_ret = -(D_level*d_level + D_slope*d_slope)/100  [durations in years,
        d_* in pp]; excess_B = tba_total_return - (price_ret_hedge + income)

VALIDATION (the point)
  For each coupon, after hedging, regress the hedged excess on the 2yr change
  (dy2) -- which is OUTSIDE the level/slope span -- and on raw [dy5,dy10]. A hedge
  that actually neutralizes rate exposure leaves ~0 exposure to dy2. (Testing the
  residual against dy5/dy10 for hedge (A) is circular by construction; dy2 is not,
  so dy2 is the honest out-of-basis check.)

OUTPUTS (all gitignored under outputs/)
  outputs/hedge_panel_levelslope.csv   per coupon-month: date, coupon,
      tba_total_return, income, d_level, d_slope, excess_A, excess_B,
      b_level_A, b_slope_A, D_level_B, D_slope_B
  outputs/hedge_panel_validation.csv   per coupon: residual t-stats vs dy2 (and
      vs dy5/dy10 for reference) under no-hedge / hedge_A / hedge_B
"""
import os, json
import numpy as np
import pandas as pd

import stage3_der_factor_shocks as base

OUT, DATA = base.OUT, base.DATA
COUPONS   = base.COUPONS
GFEE      = base.GFEE
W0, W1    = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-01")


def ols(y, X):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    if n <= k:
        return coef, np.full(k, np.nan), np.nan, resid
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss if ss > 1e-12 else np.nan
    return coef, se, r2, resid


def load_rate_basis():
    """Advisor's basis + a 2yr control, all beginning-of-month-aligned."""
    t = pd.read_excel(os.path.join(DATA, "treasury_yields_clean.xlsx"),
                      sheet_name="Treasury_Yields", header=1)
    t.columns = [str(c).strip() for c in t.columns]
    t["Date"] = pd.to_datetime(t["Date"])
    t = t.sort_values("Date").reset_index(drop=True)
    y5  = [c for c in t.columns if "5yr"  in c.lower() and "avg" not in c.lower()][0]
    y10 = [c for c in t.columns if "10yr" in c.lower()][0]
    t["d_level"] = ((t[y5] + t[y10]) / 2).diff()
    t["d_slope"] = (t[y10] - t[y5]).diff()
    t["date"]    = t["Date"].apply(base.ym)
    out = t[["date", "d_level", "d_slope"]].dropna()

    # 2yr control from the daily H.15 file (clean.xlsx has no 2yr); month-end aligned
    d = pd.read_csv(os.path.join(DATA, "treasury_yields.csv"),
                    index_col=0, parse_dates=True).sort_index()
    # month-end value = last obs on/before each clean.xlsx date
    dy2 = []
    for dt in t["Date"]:
        idx = d.index[d.index <= dt]
        dy2.append(d.loc[idx[-1], "2yr"] if len(idx) else np.nan)
    t["_2yr"] = dy2
    t["dy2"]  = t["_2yr"].diff()
    ctrl = t[["date", "dy2"]].dropna()
    return out.merge(ctrl, on="date", how="left")


def load_model_durations():
    """(D_level, D_slope) per coupon-month from krd_panel_diag.csv."""
    p = pd.read_csv(os.path.join(OUT, "krd_panel_diag.csv"))
    p["date"] = pd.to_datetime(p["ret_month"].astype(str) + "-01")
    p["D_level_B"] = p["krd5"] + p["krd10"]
    p["D_slope_B"] = (p["krd10"] - p["krd5"]) / 2.0
    return p[["date", "coupon", "D_level_B", "D_slope_B"]]


def main():
    print("Loading raw TBA returns + income (reusing pipeline construction)...", flush=True)
    # load_excess_returns gives excess already; we need the components, so rebuild
    # the raw pieces here exactly as it does, then apply our own hedges.
    fncl = pd.read_excel(os.path.join(DATA, "fncl_tba_prices_clean.xlsx"),
                         sheet_name="Last_Price_Decimal", header=1)
    fncl.columns = [str(c).strip() for c in fncl.columns]
    fncl["Date"] = pd.to_datetime(fncl["Date"])
    fncl = fncl.sort_values("Date").reset_index(drop=True)

    treas = pd.read_excel(os.path.join(DATA, "treasury_yields_clean.xlsx"),
                          sheet_name="Treasury_Yields", header=1)
    treas.columns = [str(c).strip() for c in treas.columns]
    treas["Date"] = pd.to_datetime(treas["Date"])
    treas = treas.sort_values("Date").reset_index(drop=True)
    y5  = [c for c in treas.columns if "5yr"  in c.lower() and "avg" not in c.lower()][0]
    y10 = [c for c in treas.columns if "10yr" in c.lower()][0]
    treas["income"] = ((treas[y5] + treas[y10]) / 2) / 12.0 / 100.0
    treas["date"]   = treas["Date"].apply(base.ym)
    income = treas[["date", "income"]]

    rows = []
    for c in COUPONS:
        col = f"FNCL {c}"
        if col not in fncl.columns:
            print(f"  WARNING {col} missing"); continue
        p = fncl[["Date", col]].dropna(subset=[col]).sort_values("Date").copy()
        p["P_prev"] = p[col].shift(1)
        p["tba_total_return"] = (p[col] + c / 12.0 - p["P_prev"]) / p["P_prev"]
        p = p.dropna(subset=["tba_total_return"])
        p["coupon"] = c
        p["date"] = p["Date"].apply(base.ym)
        rows.append(p[["date", "coupon", "tba_total_return"]])
    tba = pd.concat(rows, ignore_index=True)

    rates = load_rate_basis()
    dur_B = load_model_durations()

    panel = (tba.merge(income, on="date", how="inner")
                 .merge(rates, on="date", how="inner")
                 .merge(dur_B, on=["date", "coupon"], how="left"))
    win = panel[(panel["date"] >= W0) & (panel["date"] <= W1)].copy()
    print(f"  window {W0.date()}..{W1.date()}: {win['date'].nunique()} months, "
          f"{len(win)} coupon-months", flush=True)

    # ---- Hedge (A): empirical per-coupon betas on [level, slope] ----
    # ---- Hedge (B): model durations ----
    outrows, valrows = [], []
    for c, g in win.groupby("coupon"):
        g = g.dropna(subset=["tba_total_return", "d_level", "d_slope"]).sort_values("date")
        y = g["tba_total_return"].values - g["income"].values   # hedge the price part
        X = np.column_stack([np.ones(len(g)), g["d_level"].values, g["d_slope"].values])
        coefA, seA, r2A, residA = ols(y, X)
        bL, bS = coefA[1], coefA[2]
        excess_A = residA + coefA[0]        # keep intercept (the alpha), remove rate part

        # model hedge: price_ret = -(D_level*d_level + D_slope*d_slope)/100  (years * pp)
        if g["D_level_B"].notna().all():
            price_hedge = (g["D_level_B"].values * g["d_level"].values
                           + g["D_slope_B"].values * g["d_slope"].values) / 100.0
            excess_B = (g["tba_total_return"].values - g["income"].values) + price_hedge
        else:
            excess_B = np.full(len(g), np.nan)

        for i, (_, r) in enumerate(g.iterrows()):
            outrows.append(dict(date=str(r["date"].date()), coupon=c,
                                tba_total_return=r["tba_total_return"], income=r["income"],
                                d_level=r["d_level"], d_slope=r["d_slope"],
                                excess_A=excess_A[i], excess_B=excess_B[i],
                                b_level_A=bL, b_slope_A=bS,
                                D_level_B=r.get("D_level_B", np.nan),
                                D_slope_B=r.get("D_slope_B", np.nan)))

        # ---- validation: residual exposure to dy2 (out-of-basis) ----
        gg = g.dropna(subset=["dy2"])
        def tstat_on(series_vals, controls):
            yv = series_vals[gg.index.isin(g.index)] if False else series_vals
            Xc = np.column_stack([np.ones(len(gg))] + [gg[cn].values for cn in controls])
            co, se, _, _ = ols(yv, Xc)
            return {cn: float(co[j+1] / se[j+1]) for j, cn in enumerate(controls)}

        # align excess arrays to gg rows
        mask = g["dy2"].notna().values
        raw_excess = (g["tba_total_return"].values - g["income"].values)
        valrows.append(dict(coupon=c,
            n=int(mask.sum()),
            t_dy2_nohedge=tstat_on(raw_excess[mask], ["dy2"])["dy2"],
            t_dy2_hedgeA =tstat_on(excess_A[mask],  ["dy2"])["dy2"],
            t_dy2_hedgeB =(tstat_on(excess_B[mask], ["dy2"])["dy2"]
                           if not np.isnan(excess_B).any() else np.nan),
            b_level_A=float(bL), b_slope_A=float(bS), r2_A=float(r2A)))

    out = pd.DataFrame(outrows)
    val = pd.DataFrame(valrows).sort_values("coupon")
    out.to_csv(os.path.join(OUT, "hedge_panel_levelslope.csv"), index=False)
    val.to_csv(os.path.join(OUT, "hedge_panel_validation.csv"), index=False)

    print("\n=== validation: residual exposure to 2yr change (out-of-basis) ===")
    print(f"{'cpn':>4} {'n':>3} {'t_dy2_none':>11} {'t_dy2_A':>9} {'t_dy2_B':>9} "
          f"{'bL_A':>7} {'bS_A':>7} {'r2_A':>6}")
    for _, r in val.iterrows():
        b = f"{r['t_dy2_hedgeB']:>9.2f}" if not np.isnan(r['t_dy2_hedgeB']) else f"{'NA':>9}"
        print(f"{r['coupon']:>4} {r['n']:>3} {r['t_dy2_nohedge']:>11.2f} "
              f"{r['t_dy2_hedgeA']:>9.2f} {b} {r['b_level_A']:>7.2f} "
              f"{r['b_slope_A']:>7.2f} {r['r2_A']:>6.3f}")
    print("\nRead: |t_dy2| near 0 after hedging = rate exposure neutralized.")
    print("Saved: hedge_panel_levelslope.csv, hedge_panel_validation.csv")


if __name__ == "__main__":
    main()
