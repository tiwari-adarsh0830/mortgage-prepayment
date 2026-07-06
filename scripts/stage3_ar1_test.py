"""
AR(1)/persistence test for the factor-shock series — extended to the rolling
series per advisor's 7/5 request.

Procedure:
  1. Build f_level[t], f_slope[t] exactly as in stage3_der_factor_shocks.py
  2. Fit AR(1): f[t] = alpha + rho*f[t-1] + innovation[t]
  3. Replace f_level/f_slope with the AR(1) *residual* (the innovation-only series)
  4. Recompute empirical betas + Fama-MacBeth on the innovation factors
  5. Report rho_x, rho_y, and lambda_x/t-stat RAW vs AR(1)-residualized, side by side

Works for both full-sample and rolling by pointing --forecast at either file,
and for count/UPB weighting via --realized-col.
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

import stage3_der_factor_shocks as base

OUT = base.OUT


def ar1_residualize(series: pd.Series):
    """Fit f[t] = alpha + rho*f[t-1] + eps[t]. Returns (rho, residual_series)."""
    s = series.reset_index(drop=True)
    lag = s.shift(1)
    valid = lag.notna()
    X = np.column_stack([np.ones(valid.sum()), lag[valid].values])
    y = s[valid].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, rho = float(coef[0]), float(coef[1])
    resid = s - (alpha + rho * lag)
    return rho, resid


def run(forecast_path, realized_path, realized_col, label, oos_only=False):
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    returns = base.load_excess_returns()
    returns = returns[returns["date"] >= base.START_DATE].copy()

    if oos_only:
        import pandas as pd
        raw = pd.read_csv(forecast_path)
        if "is_oos" not in raw.columns:
            raise SystemExit(f"oos_only=True but 'is_oos' column missing from {forecast_path}")
        n_before = len(raw)
        raw = raw[raw["is_oos"] == True].copy()
        print(f"  OOS filter: kept {len(raw)}/{n_before} rows (dropped in-sample production-model months)")
        tmp_path = forecast_path.replace(".csv", "_oos_only_tmp.csv")
        raw.to_csv(tmp_path, index=False)
        forecast_path = tmp_path

    fc   = base.load_forecast(forecast_path)
    real = base.load_realized(realized_path, realized_col)
    shock = fc.merge(real, on=["date", "coupon"], how="inner")
    shock = shock[shock["date"] >= base.START_DATE].copy()

    pmms_by_date = returns.drop_duplicates("date").set_index("date")["pmms"]
    shock["pmms"] = shock["date"].map(pmms_by_date)
    shock = shock.dropna(subset=["pmms"])
    shock["moneyness"] = (shock["coupon"] + base.GFEE) - shock["pmms"]
    shock["incentive"] = np.maximum(0.0, shock["moneyness"])

    factor_ts = base.build_factors(
        shock[["date", "coupon", "forecast_cpr", "realized_cpr", "incentive"]]
    ).sort_values("date").reset_index(drop=True)

    if len(factor_ts) < 6:
        print(f"  Only {len(factor_ts)} months — too few for AR(1) test. Skipping.")
        return None

    # ---- RAW result (as currently reported) ----
    betas_raw  = base.empirical_betas(returns, factor_ts)
    returns_fm = returns[returns["date"].isin(factor_ts["date"])].copy()
    lam_raw, _ = base.fama_macbeth(returns_fm, betas_raw)
    lx_raw = lam_raw["lambda_x"].dropna()
    t_raw, _ = stats.ttest_1samp(lx_raw, 0) if len(lx_raw) else (np.nan, np.nan)

    # ---- AR(1)-residualized (innovation only) ----
    rho_x, resid_level = ar1_residualize(factor_ts["f_level"])
    rho_y, resid_slope = ar1_residualize(factor_ts["f_slope"])

    factor_innov = factor_ts.copy()
    factor_innov["f_level"] = resid_level
    factor_innov["f_slope"] = resid_slope
    factor_innov = factor_innov.dropna(subset=["f_level", "f_slope"]).reset_index(drop=True)

    betas_innov = base.empirical_betas(returns, factor_innov)
    returns_fm_i = returns[returns["date"].isin(factor_innov["date"])].copy()
    lam_innov, _ = base.fama_macbeth(returns_fm_i, betas_innov)
    lx_innov = lam_innov["lambda_x"].dropna()
    t_innov, _ = stats.ttest_1samp(lx_innov, 0) if len(lx_innov) else (np.nan, np.nan)

    print(f"  rho(f_level) = {rho_x:.3f}   rho(f_slope) = {rho_y:.3f}")
    print(f"  RAW:         lambda_x mean={lx_raw.mean():.6f}  t={t_raw:.3f}  n={len(lx_raw)}")
    print(f"  AR(1)-resid: lambda_x mean={lx_innov.mean():.6f}  t={t_innov:.3f}  n={len(lx_innov)}")

    return dict(
        label=label, rho_f_level=rho_x, rho_f_slope=rho_y,
        raw=dict(mean=float(lx_raw.mean()) if len(lx_raw) else None,
                  t=float(t_raw) if len(lx_raw) else None, n=int(len(lx_raw))),
        ar1_residualized=dict(mean=float(lx_innov.mean()) if len(lx_innov) else None,
                              t=float(t_innov) if len(lx_innov) else None,
                              n=int(len(lx_innov))),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realized-col", default="cpr_upb",
                    help="'cpr' (count) or 'cpr_upb' (UPB) — defaults to UPB per 7/5 ask")
    ap.add_argument("--realized-path", default=None,
                    help="Override realized-CPR CSV path if UPB lives in a separate file")
    args = ap.parse_args()

    results = {}
    results["full_sample"] = run(
        forecast_path=os.path.join(OUT, "forecast_cpr_timeseries_gfee050.csv"),
        realized_path=args.realized_path,
        realized_col=args.realized_col,
        label=f"FULL-SAMPLE ({args.realized_col})",
    )
    results["rolling"] = run(
        forecast_path=os.path.join(OUT, "rolling_forecast_cpr_timeseries.csv"),
        realized_path=args.realized_path,
        realized_col=args.realized_col,
        label=f"ROLLING OOS-ONLY ({args.realized_col})",
        oos_only=True,
    )

    json.dump(results, open(os.path.join(OUT, "ar1_persistence_test_results.json"), "w"),
              indent=2, default=str)
    print("\nSaved: ar1_persistence_test_results.json")
