"""
AR(1)/persistence test for the factor-shock series — extended to the rolling
series.

Procedure:
  1. Build f_level[t], f_slope[t] exactly as in stage3_der_factor_shocks.py
  2. Fit AR(1): f[t] = alpha + rho*f[t-1] + innovation[t]
  3. Replace f_level/f_slope with the AR(1) *residual* (the innovation-only series)
  4. Recompute empirical betas + Fama-MacBeth on the innovation factors
  5. Report rho_x, rho_y, and lambda_x/t-stat RAW vs AR(1)-residualized, side by side

Works for both full-sample and rolling by pointing --forecast at either file,
and for count/UPB weighting via --realized-col.
"""
import os, re, json, argparse
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


def lag1_autocorr(series: pd.Series):
    """Lag-1 autocorrelation of a series (e.g. the AR(1) residual itself), NOT
    to be confused with rho from ar1_residualize (that's the AR(1) coefficient
    fit to the RAW series). This checks whether the residual/innovation series
    left over after removing that AR(1) structure is itself still persistent.
    OLS orthogonality (resid _|_ lag of the fitted regressor) does not by
    itself guarantee resid[t] _|_ resid[t-1] -- e.g. if the true process has
    higher-order structure an AR(1) fit won't capture, so this needs to be
    checked on the actual residual series, not assumed (7/16 advisor ask).
    Returns np.nan if fewer than 3 non-NaN observations."""
    s = series.dropna().reset_index(drop=True)
    if len(s) < 3:
        return float("nan")
    a, b = s[1:].reset_index(drop=True), s[:-1].reset_index(drop=True)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def standardize_factors(factor_ts: pd.DataFrame, cols=("f_level", "f_slope")):
    """Rescale each surprise series to unit variance WITHIN ITS OWN WINDOW
    (7/7 advisor request) -- i.e. each specification (full-sample / rolling /
    rolling-ex-cutoff_2020, RAW or AR(1)-residualized) is z-scored using its
    own mean/std, not a global one. This makes b_x, b_y loadings-per-1-SD
    move, and lambda the premium per one-within-window-SD exposure, in
    every specification.

    NOTE (analytical, not a result to re-derive by hand each time): because
    empirical_betas() and fama_macbeth() are both linear in the factor
    columns, rescaling f_level/f_slope by constants s_x, s_y within a fixed
    window rescales b_x -> b_x/s_x and b_y -> b_y/s_y for every coupon
    uniformly, which rescales lambda_x[t] -> lambda_x[t]*s_x and
    lambda_y[t] -> lambda_y[t]*s_y for every month uniformly. Mean AND
    cross-month std of the lambda series scale by the same factor, so the
    t-stat (and the collinearity diagnostic rho_full, and the single-vs-two
    -factor mode) is UNCHANGED by this transform -- only the reported
    lambda magnitude changes. That's the point: it isolates "was 2020-21
    carrying magnitude" from "was 2020-21 carrying significance" cleanly.

    Returns (standardized_copy, {col: std_used})."""
    out = factor_ts.copy()
    stds = {}
    for col in cols:
        s = out[col]
        std = float(s.std(ddof=1))
        stds[col] = std
        if std < 1e-12:
            raise SystemExit(f"standardize_factors: '{col}' has ~zero variance "
                              f"in this window -- cannot standardize.")
        out[col] = (s - s.mean()) / std
    return out, stds


def run(forecast_path, realized_path, realized_col, label, oos_only=False, exclude_cutoffs=None):
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
        suffix = "_oos_only_tmp"
        if exclude_cutoffs:
            n_before_ex = len(raw)
            raw = raw[~raw["model_used"].isin(exclude_cutoffs)].copy()
            print(f"  Excluding {exclude_cutoffs}: kept {len(raw)}/{n_before_ex} rows")
            suffix += "_ex_" + "_".join(c.replace("cutoff_", "") for c in exclude_cutoffs)
        tmp_path = forecast_path.replace(".csv", f"{suffix}.csv")
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
    ly_raw = lam_raw["lambda_y"].dropna()
    ty_raw, _ = stats.ttest_1samp(ly_raw, 0) if len(ly_raw) else (np.nan, np.nan)

    # ---- RAW, standardized (7/7 request): unit-variance rescale within this
    # window before estimating betas -> lambda = premium per 1-SD exposure ----
    factor_ts_std, std_raw = standardize_factors(factor_ts)
    betas_raw_std  = base.empirical_betas(returns, factor_ts_std)
    lam_raw_std, _ = base.fama_macbeth(returns_fm, betas_raw_std)
    lx_raw_std = lam_raw_std["lambda_x"].dropna()
    t_raw_std, _ = stats.ttest_1samp(lx_raw_std, 0) if len(lx_raw_std) else (np.nan, np.nan)
    ly_raw_std = lam_raw_std["lambda_y"].dropna()
    ty_raw_std, _ = stats.ttest_1samp(ly_raw_std, 0) if len(ly_raw_std) else (np.nan, np.nan)

    # ---- AR(1)-residualized (innovation only) ----
    rho_x, resid_level = ar1_residualize(factor_ts["f_level"])
    rho_y, resid_slope = ar1_residualize(factor_ts["f_slope"])

    # Post-residualization check (7/16 ask #1): is the leftover innovation
    # series itself close to white noise, or does persistence remain?
    resid_rho_x = lag1_autocorr(resid_level)
    resid_rho_y = lag1_autocorr(resid_slope)

    factor_innov = factor_ts.copy()
    factor_innov["f_level"] = resid_level
    factor_innov["f_slope"] = resid_slope
    factor_innov = factor_innov.dropna(subset=["f_level", "f_slope"]).reset_index(drop=True)

    betas_innov = base.empirical_betas(returns, factor_innov)
    returns_fm_i = returns[returns["date"].isin(factor_innov["date"])].copy()
    lam_innov, _ = base.fama_macbeth(returns_fm_i, betas_innov)
    lx_innov = lam_innov["lambda_x"].dropna()
    t_innov, _ = stats.ttest_1samp(lx_innov, 0) if len(lx_innov) else (np.nan, np.nan)
    ly_innov = lam_innov["lambda_y"].dropna()
    ty_innov, _ = stats.ttest_1samp(ly_innov, 0) if len(ly_innov) else (np.nan, np.nan)

    # ---- AR(1)-residualized, standardized (7/7 request) ----
    factor_innov_std, std_innov = standardize_factors(factor_innov)
    betas_innov_std  = base.empirical_betas(returns, factor_innov_std)
    lam_innov_std, _ = base.fama_macbeth(returns_fm_i, betas_innov_std)
    lx_innov_std = lam_innov_std["lambda_x"].dropna()
    t_innov_std, _ = stats.ttest_1samp(lx_innov_std, 0) if len(lx_innov_std) else (np.nan, np.nan)
    ly_innov_std = lam_innov_std["lambda_y"].dropna()
    ty_innov_std, _ = stats.ttest_1samp(ly_innov_std, 0) if len(ly_innov_std) else (np.nan, np.nan)

    # Save the per-month standardized AR(1)-resid lambda_x series so leave-one-out
    # robustness can be checked on the actual magnitude comparison without
    # re-running the full pipeline (7/7 follow-up: verify before sending).
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    lam_innov_std[["date", "lambda_x"]].to_csv(
        os.path.join(OUT, f"ar1_std_lambda_x_{slug}.csv"), index=False)

    print(f"  rho(f_level) [RAW series, AR(1) coef]        = {rho_x:.3f}   "
          f"rho(f_slope) [RAW series, AR(1) coef]        = {rho_y:.3f}")
    print(f"  lag-1 autocorr(resid_level) [post-resid check] = {resid_rho_x:.3f}   "
          f"lag-1 autocorr(resid_slope) [post-resid check] = {resid_rho_y:.3f}")
    print(f"  RAW:              lambda_x mean={lx_raw.mean():.6f}  t={t_raw:.3f}  n={len(lx_raw)}")
    print(f"                    lambda_y mean={ly_raw.mean():.6f}  t={ty_raw:.3f}  n={len(ly_raw)}")
    print(f"  RAW (std'zd):     lambda_x mean={lx_raw_std.mean():.6f}  t={t_raw_std:.3f}  n={len(lx_raw_std)}  "
          f"[std(f_level)={std_raw['f_level']:.6f}, std(f_slope)={std_raw['f_slope']:.6f}]")
    print(f"                    lambda_y mean={ly_raw_std.mean():.6f}  t={ty_raw_std:.3f}  n={len(ly_raw_std)}")
    print(f"  AR(1)-resid:      lambda_x mean={lx_innov.mean():.6f}  t={t_innov:.3f}  n={len(lx_innov)}")
    print(f"                    lambda_y mean={ly_innov.mean():.6f}  t={ty_innov:.3f}  n={len(ly_innov)}")
    print(f"  AR(1) (std'zd):   lambda_x mean={lx_innov_std.mean():.6f}  t={t_innov_std:.3f}  n={len(lx_innov_std)}  "
          f"[std(f_level)={std_innov['f_level']:.6f}, std(f_slope)={std_innov['f_slope']:.6f}]")
    print(f"                    lambda_y mean={ly_innov_std.mean():.6f}  t={ty_innov_std:.3f}  n={len(ly_innov_std)}")

    return dict(
        label=label, rho_f_level=rho_x, rho_f_slope=rho_y,
        resid_autocorr_f_level=resid_rho_x, resid_autocorr_f_slope=resid_rho_y,
        raw=dict(mean=float(lx_raw.mean()) if len(lx_raw) else None,
                  t=float(t_raw) if len(lx_raw) else None, n=int(len(lx_raw)),
                  lambda_y_mean=float(ly_raw.mean()) if len(ly_raw) else None,
                  lambda_y_t=float(ty_raw) if len(ly_raw) else None),
        raw_standardized=dict(mean=float(lx_raw_std.mean()) if len(lx_raw_std) else None,
                  t=float(t_raw_std) if len(lx_raw_std) else None, n=int(len(lx_raw_std)),
                  lambda_y_mean=float(ly_raw_std.mean()) if len(ly_raw_std) else None,
                  lambda_y_t=float(ty_raw_std) if len(ly_raw_std) else None,
                  std_f_level=std_raw["f_level"], std_f_slope=std_raw["f_slope"]),
        ar1_residualized=dict(mean=float(lx_innov.mean()) if len(lx_innov) else None,
                              t=float(t_innov) if len(lx_innov) else None,
                              n=int(len(lx_innov)),
                              lambda_y_mean=float(ly_innov.mean()) if len(ly_innov) else None,
                              lambda_y_t=float(ty_innov) if len(ly_innov) else None),
        ar1_residualized_standardized=dict(
                              mean=float(lx_innov_std.mean()) if len(lx_innov_std) else None,
                              t=float(t_innov_std) if len(lx_innov_std) else None,
                              n=int(len(lx_innov_std)),
                              lambda_y_mean=float(ly_innov_std.mean()) if len(ly_innov_std) else None,
                              lambda_y_t=float(ty_innov_std) if len(ly_innov_std) else None,
                              std_f_level=std_innov["f_level"], std_f_slope=std_innov["f_slope"]),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realized-col", default="cpr_upb",
                    help="'cpr' (count) or 'cpr_upb' (UPB) — defaults to UPB per 7/5 ask")
    ap.add_argument("--realized-path",
                    default=os.path.join(OUT, "realized_cpr_by_coupon_v6_upb.csv"),
                    help="Realized-CPR CSV (default: realized_cpr_by_coupon_v6_upb.csv, "
                         "UPB version -- must match --realized-col=cpr_upb default)")
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
    results["rolling_ex_cutoff_2020"] = run(
        forecast_path=os.path.join(OUT, "rolling_forecast_cpr_timeseries.csv"),
        realized_path=args.realized_path,
        realized_col=args.realized_col,
        label=f"ROLLING OOS-ONLY, EX-CUTOFF_2020 ({args.realized_col})",
        oos_only=True,
        exclude_cutoffs=["cutoff_2020"],
    )

    json.dump(results, open(os.path.join(OUT, "ar1_persistence_test_results.json"), "w"),
              indent=2, default=str)
    print("\nSaved: ar1_persistence_test_results.json")
