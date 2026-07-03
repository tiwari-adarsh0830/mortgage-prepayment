"""
Stage 3 (factor-shock variant): DER cross-sectional regression using EMPIRICAL
prepayment-surprise factors instead of analytical price-formula betas.

WHAT'S DIFFERENT FROM stage3_der_regression_v2.py
  v2 computes beta_x/beta_y per coupon-month from the DER price formula
  (analytical derivatives d(phi)/dx=1, d(phi)/dy=incentive). This script instead:
    1. Forms the prepayment surprise   shock[c,t] = realized_CPR[c,t] - forecast_CPR[c,t]
    2. Decomposes the cross-section of surprises each month into two factor series:
         f_level[t] = cross-sectional MEAN of shock over coupons   (parallel factor x)
         f_slope[t] = cross-sectional OLS slope of shock on borrower moneyness
                      (m_c - PMMS_t)                                (rate-sensitivity y)
    3. Estimates EMPIRICAL betas by time-series regression of each coupon's excess
       return on (f_level, f_slope):  R^e_{c,.} = b_x^c f_level + b_y^c f_slope + e
    4. Fama-MacBeth: each month, cross-sectional regression of excess returns on the
       empirical (b_x^c, b_y^c) -> lambda_x[t], lambda_y[t]; average over months.

  >>> METHODOLOGICAL FLAG <<<
  The factor DEFINITION in step 2 (level = mean, slope = moneyness-regression) is the
  standard non-traded-factor construction and is economically faithful to DER's x/y
  interpretation, but it should be confirmed against DER (NBER w22851) Section on
  factor construction. It is isolated in build_factors() so it can be swapped without
  touching the rest of the pipeline.

GFEE CONSISTENCY (critical)
  realized_cpr_by_coupon_v6.csv buckets coupons at GFEE=0.50
  (implied_mbs_coupon = round(note*2)/2 - 0.50). The forecast fed here MUST use the
  same convention. Pass a GFEE=0.50 forecast via --forecast. The default 0.75 forecast
  (forecast_cpr_timeseries.csv) would offset the merge by 0.25pp and corrupt the shock.
  The script PREFLIGHTS this and refuses to run on a mismatched convention.

OUTPUTS
  outputs/factor_shock_panel.csv     shock[c,t] and moneyness
  outputs/factor_shock_factors.csv   f_level[t], f_slope[t]
  outputs/factor_shock_betas.csv     empirical b_x, b_y per coupon
  outputs/factor_shock_lambda_ts.csv lambda_x[t], lambda_y[t] (Fama-MacBeth)
  outputs/factor_shock_results.json  summary + comparison to analytical lambdas
"""

import os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")

COUPONS    = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
GFEE       = 0.50          # MUST match realized_cpr_v5 bucketing
D_MOD_AVG  = 6.5           # blended 5y/10y modified duration for UST hedge
WAC_PROXY  = 3.5           # PM/DM split threshold (same as v2)

# Restrict to Jan 2020+ : pre-2020 realized CPR has known artifacts (Dec-2018
# early-month UPB-reporting lag produces spurious ~80-100% coupon CPR spikes;
# 8 such coupon-months exist pre-2020). This matches the 2020-2025 realized-CPR
# window standardized elsewhere in the pipeline.
START_DATE = pd.Timestamp("2020-01-01")


# ── Month key helper ──────────────────────────────────────────────────────────
def ym(dt):
    dt = pd.Timestamp(dt)
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)


# ── Excess returns (verbatim construction from stage3_der_regression_v2) ───────
def load_excess_returns():
    """TBA total return - Treasury-duration-matched return, per coupon-month.
    Returns DataFrame [date, coupon, excess_return, pmms]."""
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
    y5  = [c for c in treas.columns if "5yr"  in c.lower()][0]
    y10 = [c for c in treas.columns if "10yr" in c.lower()][0]
    treas["dy_avg"]           = ((treas[y5] + treas[y10]) / 2).diff()
    treas["ust_price_ret"]    = -D_MOD_AVG * treas["dy_avg"] / 100.0
    treas["ust_income_ret"]   = ((treas[y5] + treas[y10]) / 2) / 12.0 / 100.0
    treas["ust_total_return"] = treas["ust_price_ret"] + treas["ust_income_ret"]
    treas = treas.dropna(subset=["ust_total_return"])

    pmms_df = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))
    def parse(p):
        s = str(int(p))
        if len(s) == 5:  return pd.Timestamp(year=int(s[1:]), month=int(s[0]),  day=1)
        if len(s) == 6:  return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
        return pd.NaT
    pmms_df["date"] = pmms_df["reporting_period"].apply(parse)
    pmms_series = pmms_df.dropna(subset=["date"]).set_index("date")["rate_30yr"]
    def get_pmms(dt):
        k = ym(dt)
        if k in pmms_series.index: return float(pmms_series[k])
        i = np.argmin(np.abs((pmms_series.index - k).days))
        return float(pmms_series.iloc[i])

    rows = []
    for coupon in COUPONS:
        col = f"FNCL {coupon}"
        if col not in fncl.columns:
            print(f"WARNING: {col} not found"); continue
        p = fncl[["Date", col]].dropna(subset=[col]).sort_values("Date").copy()
        p["P_prev"] = p[col].shift(1)
        p["tba_total_return"] = (p[col] + coupon/12.0 - p["P_prev"]) / p["P_prev"]
        p = p.dropna(subset=["tba_total_return"])
        p["coupon"] = coupon
        rows.append(p[["Date", "coupon", "tba_total_return"]])
    tba = pd.concat(rows, ignore_index=True)
    panel = tba.merge(treas[["Date", "ust_total_return"]], on="Date", how="inner")
    panel["excess_return"] = panel["tba_total_return"] - panel["ust_total_return"]
    panel["date"] = panel["Date"].apply(ym)
    panel["pmms"] = panel["Date"].apply(get_pmms)
    return panel[["date", "coupon", "excess_return", "pmms"]]


# ── Forecast / realized loaders ───────────────────────────────────────────────
def load_forecast(path):
    df = pd.read_csv(path)
    df["date"] = df["date"].apply(ym)
    # Preflight: forecast must use GFEE=0.50 (note_rate = coupon + 0.50)
    if "note_rate" in df.columns:
        off = (df["note_rate"] - df["coupon"]).round(2)
        bad = sorted(set(off.unique()) - {GFEE})
        if bad:
            raise SystemExit(
                f"GFEE MISMATCH: forecast note_rate-coupon = {sorted(set(off.unique()))} "
                f"but realized_cpr_v5 requires {GFEE}. Regenerate the forecast at "
                f"GFEE={GFEE} (see header) before running the factor-shock pipeline.")
    return df[["date", "coupon", "forecast_cpr"]]


def load_realized():
    df = pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v6.csv"))
    df["date"] = pd.to_datetime(df["date"]).apply(ym)
    df = df.rename(columns={"implied_mbs_coupon": "coupon", "cpr": "realized_cpr"})
    return df[["date", "coupon", "realized_cpr"]]


# ── Factor construction — DER (w22851) Eqs 15-18 ──────────────────────────────
def _fit_level_slope(cpr, incentive):
    """OLS of CPR on [1, max(0, m_i - PMMS)] -> (intercept=level, slope=rate-sens).
    incentive is already the truncated borrower moneyness max(0, m_i - PMMS)."""
    X = np.column_stack([np.ones_like(incentive), incentive])
    # If the truncated incentive has no spread (all discount -> all zeros), the
    # slope is unidentified; return intercept = mean CPR, slope = 0 (DER: beta_y=0
    # for discounts, so y-factor is identified off premium months).
    if np.ptp(incentive) < 1e-12:
        return float(np.mean(cpr)), 0.0
    coef, *_ = np.linalg.lstsq(X, cpr, rcond=None)
    return float(coef[0]), float(coef[1])


def build_factors(panel):
    """DER Eqs 15-18. panel: [date, coupon, forecast_cpr, realized_cpr, incentive]
       where incentive = max(0, note_rate - PMMS_t)  (truncated borrower moneyness).

    Each month, regress forecast and realized CPR separately on [1, incentive]:
        forecast: intercept x_fcst, slope y_fcst
        realized: intercept x_real, slope y_real
    Factor innovations (the priced shocks):
        f_level[t] = x_real - x_fcst      (level / turnover surprise, DER x_t)
        f_slope[t] = y_real - y_fcst      (rate-sensitivity surprise, DER y_t)
    """
    out = []
    for dt, g in panel.groupby("date"):
        g = g.dropna(subset=["forecast_cpr", "realized_cpr", "incentive"])
        if len(g) < 3:
            continue
        inc = g["incentive"].values
        x_f, y_f = _fit_level_slope(g["forecast_cpr"].values, inc)
        x_r, y_r = _fit_level_slope(g["realized_cpr"].values, inc)
        out.append(dict(date=dt, f_level=x_r - x_f, f_slope=y_r - y_f,
                        x_fcst=x_f, x_real=x_r, y_fcst=y_f, y_real=y_r))
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


# ── Empirical betas: time-series regression of returns on factors ─────────────
def empirical_betas(returns, factor_ts):
    """For each coupon, regress excess_return_t on [f_level_t, f_slope_t] (with const).
    Returns DataFrame [coupon, b_x, b_y, n_obs, r2]."""
    merged = returns.merge(factor_ts, on="date", how="inner")
    rows = []
    for c, g in merged.groupby("coupon"):
        g = g.dropna(subset=["excess_return", "f_level", "f_slope"])
        if len(g) < 6:                      # need enough months to identify loadings
            continue
        X = np.column_stack([np.ones(len(g)), g["f_level"].values, g["f_slope"].values])
        y = g["excess_return"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        rows.append(dict(coupon=c, b_x=float(coef[1]), b_y=float(coef[2]),
                         n_obs=len(g), r2=round(r2, 4)))
    return pd.DataFrame(rows).sort_values("coupon").reset_index(drop=True)


# ── Fama-MacBeth on empirical betas ───────────────────────────────────────────
def fama_macbeth(returns, betas, rho_max=0.90, single_factor_if_collinear=True):
    """Each month: cross-sectional OLS of excess_return on (b_x, b_y), no intercept
    (per DER). DER drop months where corr(b_x,b_y) across coupons > rho_max.

    Because DER use FIXED full-sample loadings, that cross-sectional correlation is
    (near) constant across months. If it exceeds rho_max in (nearly) every month the
    two-factor model is unidentified in this sample -- which is itself a result. In
    that case, if single_factor_if_collinear, we fall back to a one-factor model on
    b_x alone (lambda_y set to None) so lambda_x is still estimable, and we report
    the collinearity explicitly.

    Returns (lambda_df, diag) where diag describes the collinearity / mode used.
    """
    bmap = betas.set_index("coupon")[["b_x", "b_y"]]

    # Cross-sectional corr of the (fixed) loadings, computed once on the beta table
    bx, by = betas["b_x"].values, betas["b_y"].values
    rho_full = (float(np.corrcoef(bx, by)[0, 1])
                if betas["b_x"].std() > 1e-12 and betas["b_y"].std() > 1e-12 else np.nan)
    collinear = (not np.isnan(rho_full)) and abs(rho_full) > rho_max
    mode = "single_factor(b_x)" if (collinear and single_factor_if_collinear) else "two_factor"

    rows, dropped = [], 0
    for dt, g in returns.groupby("date"):
        g = g[g["coupon"].isin(bmap.index)].copy()
        if len(g) < 4:
            continue
        g = g.join(bmap, on="coupon").dropna(subset=["b_x", "b_y", "excess_return"])
        if len(g) < 4:
            continue

        if mode == "two_factor":
            # per-month guard (kept for completeness; with fixed betas == rho_full)
            if g["b_x"].std() > 1e-12 and g["b_y"].std() > 1e-12:
                rho = float(np.corrcoef(g["b_x"].values, g["b_y"].values)[0, 1])
                if abs(rho) > rho_max:
                    dropped += 1
                    continue
            X = g[["b_x", "b_y"]].values
            y = g["excess_return"].values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            lx, ly = float(coef[0]), float(coef[1])
            yhat = X @ coef
        else:
            # single-factor: regress on b_x only
            X = g[["b_x"]].values
            y = g["excess_return"].values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            lx, ly = float(coef[0]), None
            yhat = X @ coef

        ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        rows.append(dict(date=str(pd.Timestamp(dt).date()),
                         lambda_x=lx, lambda_y=ly,
                         n=len(g), r2=(round(r2, 4) if not np.isnan(r2) else None)))

    lam = pd.DataFrame(rows)
    if lam.empty:
        lam = pd.DataFrame(columns=["date", "lambda_x", "lambda_y", "n", "r2"])
    diag = dict(rho_loadings_full=rho_full, collinear=bool(collinear),
                rho_max=rho_max, mode=mode, months_dropped=dropped,
                months_used=len(lam))
    return lam, diag

def main(forecast_path):
    print("Loading excess returns...", flush=True)
    returns = load_excess_returns()
    returns = returns[returns["date"] >= START_DATE].copy()
    print(f"  returns panel: {len(returns)} obs, {returns['date'].nunique()} months "
          f"(from {START_DATE.date()})", flush=True)

    print("Loading forecast + realized, building shock...", flush=True)
    fc   = load_forecast(forecast_path)
    real = load_realized()
    shock = fc.merge(real, on=["date", "coupon"], how="inner")
    shock["shock"] = shock["realized_cpr"] - shock["forecast_cpr"]

    # Drop pre-2020 artifact months (see START_DATE note)
    n_before = len(shock)
    shock = shock[shock["date"] >= START_DATE].copy()
    print(f"  dropped {n_before - len(shock)} pre-{START_DATE.year} coupon-months "
          f"(realized-CPR artifacts)", flush=True)

    # Truncated borrower moneyness (DER): incentive = max(0, note_rate - PMMS_t)
    pmms_by_date = returns.drop_duplicates("date").set_index("date")["pmms"]
    shock["pmms"] = shock["date"].map(pmms_by_date)
    shock = shock.dropna(subset=["pmms"])
    shock["moneyness"] = (shock["coupon"] + GFEE) - shock["pmms"]
    shock["incentive"] = np.maximum(0.0, shock["moneyness"])   # DER ReLU incentive
    print(f"  shock panel: {len(shock)} obs, {shock['date'].nunique()} months", flush=True)

    factor_ts = build_factors(
        shock[["date", "coupon", "forecast_cpr", "realized_cpr", "incentive"]])
    print(f"  factors: {len(factor_ts)} months | "
          f"f_level mean={factor_ts['f_level'].mean():.5f}  "
          f"f_slope mean={factor_ts['f_slope'].mean():.5f}", flush=True)

    betas = empirical_betas(returns, factor_ts)
    print("\nEmpirical betas (return loadings on prepayment-surprise factors):")
    print(betas.to_string(index=False))

    # FM cross-section restricted to months where the rolling shock/factor
    # actually exists -- otherwise fixed betas get priced against out-of-window
    # returns and n silently reverts to the full-sample count, defeating the
    # point of a genuine rolling OOS test.
    returns_fm = returns[returns["date"].isin(factor_ts["date"])].copy()
    print(f"\nFM restricted to factor-coverage months: {returns_fm['date'].nunique()} "
          f"(full returns panel had {returns['date'].nunique()})")

    lam, diag = fama_macbeth(returns_fm, betas)
    print(f"\nLoading collinearity: corr(b_x,b_y)={diag['rho_loadings_full']:.3f} "
          f"(threshold {diag['rho_max']}) -> mode={diag['mode']}")
    if diag["collinear"]:
        print("  NOTE: two factors are near-collinear in this (discount-heavy) "
              "sample -> reporting single-factor lambda_x. This matches DER's "
              "multicollinearity caveat (Section 5.3).")
    print(f"Fama-MacBeth: {diag['months_used']} monthly regressions "
          f"({diag['months_dropped']} dropped)")

    if lam.empty:
        print("No estimable months -> cannot report lambda. "
              "Inspect betas/returns alignment.")
        lx = pd.Series(dtype=float); ly = pd.Series(dtype=float)
    else:
        lx = lam["lambda_x"].dropna()
        ly = lam["lambda_y"].dropna() if "lambda_y" in lam.columns else pd.Series(dtype=float)
    out = {}
    if len(lx):
        t, p = stats.ttest_1samp(lx, 0)
        out["lambda_x"] = dict(mean=float(lx.mean()), std=float(lx.std()),
                               t=float(t), p=float(p), n=int(len(lx)))
        print(f"lambda_x: mean={lx.mean():.6f}  t={t:.3f}  p={p:.4f}  n={len(lx)}")
    if len(ly):
        t, p = stats.ttest_1samp(ly, 0)
        out["lambda_y"] = dict(mean=float(ly.mean()), std=float(ly.std()),
                               t=float(t), p=float(p), n=int(len(ly)))
        print(f"lambda_y: mean={ly.mean():.6f}  t={t:.3f}  p={p:.4f}  n={len(ly)}")

    # Compare to analytical-beta lambdas if present
    apath = os.path.join(OUT, "stage3_lambda_ts.csv")
    if os.path.exists(apath):
        a = pd.read_csv(apath)
        out["analytical_comparison"] = dict(
            analytical_lambda_x_mean=float(a["lambda_x"].dropna().mean()),
            factor_shock_lambda_x_mean=(float(lx.mean()) if len(lx) else None))
        print(f"\nAnalytical lambda_x mean: {a['lambda_x'].dropna().mean():.6f}")
        print(f"Factor-shock lambda_x mean: {lx.mean():.6f}" if len(lx) else "")

    shock.to_csv(os.path.join(OUT, "factor_shock_panel.csv"), index=False)
    factor_ts.to_csv(os.path.join(OUT, "factor_shock_factors.csv"), index=False)
    betas.to_csv(os.path.join(OUT, "factor_shock_betas.csv"), index=False)
    lam.to_csv(os.path.join(OUT, "factor_shock_lambda_ts.csv"), index=False)
    out["fm_diagnostics"] = diag
    json.dump(out, open(os.path.join(OUT, "factor_shock_results.json"), "w"), indent=2)
    print("\nSaved: factor_shock_panel.csv, factor_shock_factors.csv, "
          "factor_shock_betas.csv, factor_shock_lambda_ts.csv, factor_shock_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", default=os.path.join(OUT, "forecast_cpr_timeseries_gfee050.csv"),
                    help="GFEE=0.50 forecast CSV (must match realized_cpr_v5 bucketing)")
    args = ap.parse_args()
    main(args.forecast)
