"""
Economic-magnitude follow-up to Phase 20 standardization (7/16 advisor ask #2):

  1. Cross-sectional SPREAD of the standardized b_x loadings across the 9
     coupons (max - min). Because f_level/f_slope were rescaled to unit
     variance within-window (Phase 20), b_x is already in "excess-return-
     per-1-SD-surprise" units, so this spread is directly comparable across
     specifications.
  2. Multiply that spread by lambda_x (premium per 1-SD exposure, monthly)
     to get the expected monthly excess-return GAP between the most- and
     least-exposed coupon. Annualize (x12, x10000) -> bps/yr.
  3. Factor portfolio: long the highest-b_x coupon, short the lowest-b_x
     coupon, each month, using realized excess returns. Report mean, std,
     and annualized Sharpe (x sqrt(12)) of that zero-cost portfolio's
     return series -- this is OUR estimate to sit next to DER's reported
     Sharpe (pull the exact DER figure from their empirical results section
     when writing this up; not fabricated here).

Runs on the SAME three specifications as stage3_ar1_test.py (full-sample,
rolling OOS-only, rolling OOS-only ex-cutoff_2020) so magnitudes line up
with the numbers already reported on 7/8 and 7/16.

Outputs: outputs/beta_spread_sharpe_results.json
"""
import os, json, argparse
import numpy as np
import pandas as pd

import stage3_der_factor_shocks as base
from stage3_ar1_test import ar1_residualize, standardize_factors

OUT = base.OUT
MONTHS_PER_YEAR = 12
ANNUALIZE_VOL = np.sqrt(12)


def load_factor_ts(forecast_path, realized_path, realized_col, oos_only=False, exclude_cutoffs=None):
    """Rebuild factor_ts exactly as stage3_ar1_test.run() does, up through the
    point of AR(1) residualization + standardization (mirrors that script so
    results are directly comparable -- do not duplicate logic beyond this)."""
    returns = base.load_excess_returns()
    returns = returns[returns["date"] >= base.START_DATE].copy()

    if oos_only:
        raw = pd.read_csv(forecast_path)
        if "is_oos" not in raw.columns:
            raise SystemExit(f"oos_only=True but 'is_oos' column missing from {forecast_path}")
        raw = raw[raw["is_oos"] == True].copy()
        suffix = "_oos_only_tmp"
        if exclude_cutoffs:
            raw = raw[~raw["model_used"].isin(exclude_cutoffs)].copy()
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

    return returns, factor_ts


def beta_spread_and_sharpe(returns, factor_ts, label):
    if len(factor_ts) < 6:
        print(f"  [{label}] only {len(factor_ts)} months -- too few, skipping.")
        return None

    # AR(1)-residualize, then standardize (mirrors the 7/16-verified spec:
    # the headline lambda_x numbers reported so far are AR(1)-resid + standardized).
    _, resid_level = ar1_residualize(factor_ts["f_level"])
    _, resid_slope = ar1_residualize(factor_ts["f_slope"])
    factor_innov = factor_ts.copy()
    factor_innov["f_level"] = resid_level
    factor_innov["f_slope"] = resid_slope
    factor_innov = factor_innov.dropna(subset=["f_level", "f_slope"]).reset_index(drop=True)
    factor_std, stds = standardize_factors(factor_innov)

    betas = base.empirical_betas(returns, factor_std)
    if betas.empty or len(betas) < 2:
        print(f"  [{label}] fewer than 2 coupons with identifiable betas -- skipping.")
        return None

    returns_fm = returns[returns["date"].isin(factor_std["date"])].copy()
    lam, diag = base.fama_macbeth(returns_fm, betas)
    lx = lam["lambda_x"].dropna()
    if lx.empty:
        print(f"  [{label}] lambda_x not estimable -- skipping.")
        return None
    lambda_x_mean = float(lx.mean())

    # ---- (1) Cross-sectional spread of standardized b_x ----
    b_hi_row = betas.loc[betas["b_x"].idxmax()]
    b_lo_row = betas.loc[betas["b_x"].idxmin()]
    spread_bx = float(b_hi_row["b_x"] - b_lo_row["b_x"])

    # ---- (2) Implied premium gap, annualized, in bps/yr ----
    monthly_gap = lambda_x_mean * spread_bx
    bps_per_year = monthly_gap * MONTHS_PER_YEAR * 10000.0

    # ---- (3) Factor portfolio: long highest-b_x coupon, short lowest-b_x coupon ----
    hi_c, lo_c = float(b_hi_row["coupon"]), float(b_lo_row["coupon"])
    hi_rows = returns_fm[returns_fm["coupon"] == hi_c]
    lo_rows = returns_fm[returns_fm["coupon"] == lo_c]
    # Guard: pandas Series subtraction with a DUPLICATED date index silently
    # produces a Cartesian-product-like expansion instead of erroring, which
    # would corrupt the Sharpe number without any warning. Fail loud instead.
    for name, rows in [(hi_c, hi_rows), (lo_c, lo_rows)]:
        if rows.empty:
            raise SystemExit(f"[{label}] no return rows found for coupon {name} -- "
                              f"check coupon float equality / upstream filter.")
        if rows["date"].duplicated().any():
            raise SystemExit(f"[{label}] duplicate (date, coupon={name}) rows in "
                              f"returns_fm -- fix upstream before trusting this Sharpe.")
    hi_ret = hi_rows.set_index("date")["excess_return"]
    lo_ret = lo_rows.set_index("date")["excess_return"]
    port = (hi_ret - lo_ret).dropna()
    if len(port) < 6 or port.std() < 1e-12:
        sharpe = None
        port_mean = float(port.mean()) if len(port) else None
        port_std = float(port.std()) if len(port) else None
    else:
        port_mean, port_std = float(port.mean()), float(port.std())
        sharpe = (port_mean / port_std) * ANNUALIZE_VOL

    result = dict(
        label=label,
        n_coupons_with_betas=len(betas),
        b_x_high_coupon=hi_c, b_x_high=float(b_hi_row["b_x"]),
        b_x_low_coupon=lo_c, b_x_low=float(b_lo_row["b_x"]),
        b_x_spread=spread_bx,
        lambda_x_mean=lambda_x_mean,
        implied_monthly_gap=monthly_gap,
        implied_bps_per_year=bps_per_year,
        factor_portfolio_n_months=int(len(port)),
        factor_portfolio_mean_monthly=port_mean,
        factor_portfolio_std_monthly=port_std,
        factor_portfolio_sharpe_annualized=sharpe,
        std_f_level_used=stds["f_level"], std_f_slope_used=stds["f_slope"],
        fm_mode=diag["mode"], rho_loadings_full=diag["rho_loadings_full"],
        full_betas=betas.to_dict(orient="records"),
    )
    print(f"  [{label}]")
    print("    full standardized beta table (all coupons, sorted by coupon):")
    print(betas.to_string(index=False))
    print(f"    b_x spread: {spread_bx:.4f} (coupon {hi_c} = {b_hi_row['b_x']:.4f} high, "
          f"coupon {lo_c} = {b_lo_row['b_x']:.4f} low)")
    print(f"    lambda_x (per 1-SD, monthly): {lambda_x_mean:.6f}")
    print(f"    implied gap: {monthly_gap:.6f}/mo -> {bps_per_year:.1f} bps/yr")
    if sharpe is not None:
        print(f"    long-{hi_c}/short-{lo_c} factor portfolio: mean={port_mean:.5f} "
              f"std={port_std:.5f} Sharpe(ann.)={sharpe:.3f}  n={len(port)}")
    else:
        print("    factor portfolio Sharpe: not estimable (insufficient variance/obs)")

    # Save the full beta table separately too, per spec, so it can be inspected
    # without re-parsing the combined JSON (monotonicity / illiquid-coupon check).
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    betas.to_csv(os.path.join(OUT, f"beta_spread_full_betas_{slug}.csv"), index=False)

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realized-col", default="cpr_upb")
    ap.add_argument("--realized-path", default=os.path.join(OUT, "realized_cpr_by_coupon_v6_upb.csv"))
    args = ap.parse_args()

    specs = [
        dict(forecast_path=os.path.join(OUT, "forecast_cpr_timeseries_gfee050.csv"),
             label=f"FULL-SAMPLE ({args.realized_col})", oos_only=False, exclude_cutoffs=None),
        dict(forecast_path=os.path.join(OUT, "rolling_forecast_cpr_timeseries.csv"),
             label=f"ROLLING OOS-ONLY ({args.realized_col})", oos_only=True, exclude_cutoffs=None),
        dict(forecast_path=os.path.join(OUT, "rolling_forecast_cpr_timeseries.csv"),
             label=f"ROLLING OOS-ONLY, EX-CUTOFF_2020 ({args.realized_col})",
             oos_only=True, exclude_cutoffs=["cutoff_2020"]),
    ]

    results = {}
    for spec in specs:
        label = spec.pop("label")
        returns, factor_ts = load_factor_ts(
            spec["forecast_path"], args.realized_path, args.realized_col,
            oos_only=spec["oos_only"], exclude_cutoffs=spec["exclude_cutoffs"])
        results[label] = beta_spread_and_sharpe(returns, factor_ts, label)

    json.dump(results, open(os.path.join(OUT, "beta_spread_sharpe_results.json"), "w"),
              indent=2, default=str)
    print("\nSaved: beta_spread_sharpe_results.json")
