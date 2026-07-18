"""
Leave-one-out robustness check for the ROLLING EX-CUTOFF_2020 headline numbers
(769.7 bps/yr, Sharpe 0.918) before treating them as reportable -- same practice
as the Phase 20 lambda_x leave-one-out (drop one month, recompute everything
downstream of it, check the range).

For each of the ~35 months in this spec: drop that month, re-run AR(1)-resid +
standardize + empirical_betas + fama_macbeth + spread/bps/Sharpe on the
remaining months, record the result. Reports the min/max range across all
drops and flags whether any single month's exclusion flips the sign of the
spread, bps/yr, or Sharpe (that would mean the headline is being driven by
one month rather than the whole window).

Only runs the ex-cutoff_2020 spec since it's the only one of the three with a
statistically real (monotonic) beta profile -- no point stress-testing a
number built on a profile we already know isn't reliable.

Output: outputs/beta_spread_loo_ex_cutoff_2020.json
"""
import os, json
import numpy as np
import pandas as pd

import stage3_der_factor_shocks as base
from stage3_ar1_test import ar1_residualize, standardize_factors
from stage3_beta_spread_sharpe import load_factor_ts, OUT, MONTHS_PER_YEAR, ANNUALIZE_VOL


def compute_headline(returns, factor_ts):
    """Same computation as beta_spread_and_sharpe(), stripped of printing,
    returning None on any failure mode rather than raising -- a leave-one-out
    drop can legitimately push a fold below the viable-data threshold."""
    if len(factor_ts) < 6:
        return None
    _, resid_level = ar1_residualize(factor_ts["f_level"])
    _, resid_slope = ar1_residualize(factor_ts["f_slope"])
    factor_innov = factor_ts.copy()
    factor_innov["f_level"] = resid_level
    factor_innov["f_slope"] = resid_slope
    factor_innov = factor_innov.dropna(subset=["f_level", "f_slope"]).reset_index(drop=True)
    if len(factor_innov) < 5:
        return None
    factor_std, _ = standardize_factors(factor_innov)

    betas = base.empirical_betas(returns, factor_std)
    if betas.empty or len(betas) < 2:
        return None

    returns_fm = returns[returns["date"].isin(factor_std["date"])].copy()
    lam, _ = base.fama_macbeth(returns_fm, betas)
    lx = lam["lambda_x"].dropna()
    if lx.empty:
        return None
    lambda_x_mean = float(lx.mean())

    b_hi_row = betas.loc[betas["b_x"].idxmax()]
    b_lo_row = betas.loc[betas["b_x"].idxmin()]
    spread_bx = float(b_hi_row["b_x"] - b_lo_row["b_x"])
    monthly_gap = lambda_x_mean * spread_bx
    bps_per_year = monthly_gap * MONTHS_PER_YEAR * 10000.0

    hi_c, lo_c = float(b_hi_row["coupon"]), float(b_lo_row["coupon"])
    hi_rows = returns_fm[returns_fm["coupon"] == hi_c]
    lo_rows = returns_fm[returns_fm["coupon"] == lo_c]
    for rows in (hi_rows, lo_rows):
        if rows.empty or rows["date"].duplicated().any():
            return None
    hi_ret = hi_rows.set_index("date")["excess_return"]
    lo_ret = lo_rows.set_index("date")["excess_return"]
    port = (hi_ret - lo_ret).dropna()
    if len(port) < 5 or port.std() < 1e-12:
        return None
    sharpe = (port.mean() / port.std()) * ANNUALIZE_VOL

    return dict(b_x_spread=spread_bx, bps_per_year=bps_per_year, sharpe=float(sharpe),
                hi_coupon=hi_c, lo_coupon=lo_c)


if __name__ == "__main__":
    returns, factor_ts = load_factor_ts(
        os.path.join(OUT, "rolling_forecast_cpr_timeseries.csv"),
        os.path.join(OUT, "realized_cpr_by_coupon_v6_upb.csv"),
        "cpr_upb", oos_only=True, exclude_cutoffs=["cutoff_2020"])

    full = compute_headline(returns, factor_ts)
    assert full is not None, "full-window computation failed -- check inputs before trusting any LOO result"
    print(f"Full window (n={len(factor_ts)}): spread={full['b_x_spread']:.4f} "
          f"bps/yr={full['bps_per_year']:.1f} sharpe={full['sharpe']:.3f} "
          f"(high={full['hi_coupon']}, low={full['lo_coupon']})")

    results = []
    for i, dropped_date in enumerate(factor_ts["date"]):
        sub_factor_ts = factor_ts.drop(factor_ts.index[i]).reset_index(drop=True)
        r = compute_headline(returns, sub_factor_ts)
        if r is not None:
            r["dropped_month"] = str(dropped_date)
            results.append(r)
        else:
            print(f"  [drop {dropped_date}] fold failed (insufficient data after drop) -- excluded from range")

    n_ok = len(results)
    bps_vals = [r["bps_per_year"] for r in results]
    sharpe_vals = [r["sharpe"] for r in results]
    spread_vals = [r["b_x_spread"] for r in results]

    print(f"\nLeave-one-out over {n_ok}/{len(factor_ts)} folds:")
    print(f"  bps/yr:  min={min(bps_vals):.1f}  max={max(bps_vals):.1f}  "
          f"full-window={full['bps_per_year']:.1f}")
    print(f"  Sharpe:  min={min(sharpe_vals):.3f}  max={max(sharpe_vals):.3f}  "
          f"full-window={full['sharpe']:.3f}")
    print(f"  b_x spread: min={min(spread_vals):.4f}  max={max(spread_vals):.4f}")

    sign_flips_bps = sum(1 for v in bps_vals if (v > 0) != (full["bps_per_year"] > 0))
    sign_flips_sharpe = sum(1 for v in sharpe_vals if (v > 0) != (full["sharpe"] > 0))
    print(f"  sign flips vs full-window: bps/yr={sign_flips_bps}/{n_ok}  "
          f"sharpe={sign_flips_sharpe}/{n_ok}")

    hi_coupons = set(r["hi_coupon"] for r in results)
    lo_coupons = set(r["lo_coupon"] for r in results)
    print(f"  high-exposure coupon stays at {full['hi_coupon']} in "
          f"{sum(1 for r in results if r['hi_coupon'] == full['hi_coupon'])}/{n_ok} folds "
          f"(other values seen: {hi_coupons - {full['hi_coupon']}})")
    print(f"  low-exposure coupon stays at {full['lo_coupon']} in "
          f"{sum(1 for r in results if r['lo_coupon'] == full['lo_coupon'])}/{n_ok} folds "
          f"(other values seen: {lo_coupons - {full['lo_coupon']}})")

    json.dump(dict(full=full, folds=results), open(
        os.path.join(OUT, "beta_spread_loo_ex_cutoff_2020.json"), "w"), indent=2, default=str)
    print("\nSaved: beta_spread_loo_ex_cutoff_2020.json")
