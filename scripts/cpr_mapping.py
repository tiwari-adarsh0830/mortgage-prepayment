#!/usr/bin/env python3
"""
cpr_mapping.py -- expanding-window calibration from model cohort CPR to realized
CPR, for use inside the pricer.

Advisor spec (2026-08-04): every CPR path, baseline and bumped, passes through a
mapping fitted on history through t-1 before it is priced.

SPEC CHOSEN, AND THE DEVIATION FROM THE LITERAL REQUEST
  The advisor proposed "log realized refi against log model with coefficients
  which vary by the incentive". Two departures, both measured (diag_cpr_mapping_v2,
  UPB-weighted, min-window 36, lag 1, 524 scored cells):

  1. LOGIT rather than LOG. Log-space fits are steep under UPB weighting
     (log-model slope ~1.80) and map the high-incentive corner past CPR = 1.0 --
     in-range failed at 100% of cutoffs, peaking at a mapped CPR of 1.585 at
     incentive +4.12. price_path computes 1-(1-clip(cpr,0,0.99))**(1/12), so an
     out-of-range CPR is silently clamped and priced wrongly with no error. The
     logit link is bounded in (0,1) by construction and passes in-range at 100%
     of cutoffs, at a cost of 0.3461 against 0.3346 OOS log RMSE.

  2. NO INCENTIVE-VARYING COEFFICIENTS. In the logit family incentive terms score
     slightly worse (logit 0.3461, logit_inc 0.3548, logit_interact 0.3535). The
     literal bucketed form is worse than that: it is monotone at only 25.4% of
     cutoffs (0.0% for the logit variant), meaning at least one incentive bucket
     carries a NEGATIVE model->realized slope at every cutoff. Under a rate bump
     that inverts the sign of the KRD. It also splits scalar and pointwise path
     application by 403% at incentive -4.0, against ~7% for plain logit.

  Both departures are reportable findings, not silent implementation choices.

WHAT IT IS WORTH
  Identity (no mapping) OOS log RMSE 0.4761; logit 0.3461, a 27.3% reduction.
  Level MAE 0.0367 -> 0.0211. Deep-discount bucket mean log error -0.358 -> -0.117.

WHAT IT DOES NOT FIX
  2022 gets worse under every spec (identity 0.6546, logit 0.7738). The debiased
  residual time share falls 0.572 -> 0.242 but the coupon share RISES 0.116 ->
  0.400: the mapping trades time structure for cross-sectional structure. For a
  cross-sectional asset pricing application that trade needs justifying before
  this feeds the factor-shock chain.

APPLICATION MODE
  cpr_path() returns 33 monthly values; the mapping is fitted on their mean
  (stage2_forecast_cpr_gfee050.forecast_cpr returns cpr.mean()). Scalar mode
  rescales the whole path by mapped/model evaluated at the path mean, preserving
  the model's age shape -- the mapping was never fitted at individual ages.
  Pointwise applies it to each of the 33 values. They differ by ~7% at deep
  discounts under plain logit. Scalar is the default as the more defensible.

  The mapping applies to months 1-33 ONLY. The terminal segment (months 34-360)
  comes from scurve_params_asof(), already fitted to realized cpr_upb, so passing
  it through a realized-calibrated mapping would double-correct.
"""
import os
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
OUT = os.path.join(BASE, "outputs")

FCST_FILE = "forecast_cpr_timeseries_gfee050.csv"
REAL_FILE = "realized_cpr_by_coupon_v6_upb.csv"

MIN_WINDOW_MONTHS = 36
LAG_MONTHS = 1
CPR_LO, CPR_HI = 1e-6, 1.0 - 1e-6

_PANEL = None
_FIT_CACHE = {}


def _logit(c):
    c = np.clip(np.asarray(c, dtype=float), CPR_LO, CPR_HI)
    return np.log(c / (1.0 - c))


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))


def _panel():
    """Coupon-month cells: model cohort CPR against UPB-weighted realized CPR.

    Model side is forecast_cpr_timeseries_gfee050.csv, produced by the same module
    model_hedge_krd.py imports, so it is the same construction as cpr_path and the
    same GFEE. Realized side is cpr_upb, matching scurve_params_asof so the mapped
    months 1-33 and the terminal months 34-360 share a weighting convention.
    forecast_vs_realized_cpr_gfee050.csv is deliberately NOT used: its realized
    column is count-weighted."""
    global _PANEL
    if _PANEL is not None:
        return _PANEL
    f = pd.read_csv(os.path.join(OUT, FCST_FILE), parse_dates=["date"])
    r = pd.read_csv(os.path.join(OUT, REAL_FILE), parse_dates=["date"])
    r = r[["date", "implied_mbs_coupon", "cpr_upb", "upb_atrisk"]].rename(
        columns={"implied_mbs_coupon": "coupon"})
    m = f.merge(r, on=["date", "coupon"], how="inner")
    m = m[(m.forecast_cpr > 0) & (m.cpr_upb > 0)].copy()
    m["x"] = _logit(m.forecast_cpr.values)
    m["y"] = _logit(m.cpr_upb.values)
    _PANEL = m.sort_values(["date", "coupon"]).reset_index(drop=True)
    return _PANEL


def fit_asof(asof):
    """UPB-weighted logit-logit fit on cells strictly before `asof`.

    `asof` is the pricer's information date (month end). The forecast panel is
    information-date keyed -- date=2018-01-01 corresponds to info_date=2018-01-31
    and ret_month=2018-02 -- so the cutoff is the first of the asof month, and the
    window ends LAG_MONTHS earlier again to allow for reporting lag.

    Returns (intercept, slope) or None when the window is too short, in which case
    the caller must fall back to the unmapped path."""
    key = pd.Timestamp(asof).strftime("%Y-%m")
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]

    cutoff = pd.Timestamp(asof).replace(day=1) - pd.DateOffset(months=LAG_MONTHS)
    tr = _panel()
    tr = tr[tr.date < cutoff]
    if tr.date.nunique() < MIN_WINDOW_MONTHS:
        _FIT_CACHE[key] = None
        return None

    w = tr.upb_atrisk.values.astype(float)
    s = np.sqrt(w / w.mean())
    X = np.column_stack([np.ones(len(tr)), tr.x.values]) * s[:, None]
    beta, *_ = np.linalg.lstsq(X, tr.y.values * s, rcond=None)

    if beta[1] <= 0:
        raise RuntimeError(
            "mapping slope %.4f <= 0 at asof %s -- a decreasing mapping inverts "
            "the KRD sign under a bump and must not be priced" % (beta[1], key))

    _FIT_CACHE[key] = (float(beta[0]), float(beta[1]))
    return _FIT_CACHE[key]


def scale_factor(path33, asof):
    """Multiplicative level correction implied by the mapping at this path.

    Computed from the path mean, matching the object the mapping was fitted on
    (stage2_forecast_cpr_gfee050.forecast_cpr returns cpr.mean()). Returns None
    when the expanding window is too short, so the caller falls back to unmapped.
    """
    fit = fit_asof(asof)
    if fit is None:
        return None
    a, b = fit
    p = np.asarray(path33, dtype=float)
    mean_model = float(np.clip(p.mean(), CPR_LO, CPR_HI))
    mapped_mean = float(_sigmoid(a + b * _logit(mean_model)))
    return mapped_mean / mean_model


def apply_factor(path33, factor):
    """Apply a precomputed level correction. Used by frozen mode so that a rate
    bump moves the model path only, not the correction applied to it."""
    p = np.asarray(path33, dtype=float)
    if factor is None:
        return p
    return np.clip(p * factor, CPR_LO, CPR_HI)


def apply_mapping(path33, asof, mode="scalar"):
    """Map a 33-month model CPR path. mode: off | scalar | pointwise.

    Returns (mapped_path, applied) so the caller can record how many months were
    actually mapped rather than assuming."""
    p = np.asarray(path33, dtype=float)
    if mode == "off":
        return p, False
    fit = fit_asof(asof)
    if fit is None:
        return p, False
    a, b = fit

    if mode == "pointwise":
        return np.clip(_sigmoid(a + b * _logit(p)), CPR_LO, CPR_HI), True

    if mode != "scalar":
        raise ValueError("map-mode must be off, scalar or pointwise")

    mean_model = float(np.clip(p.mean(), CPR_LO, CPR_HI))
    mapped_mean = float(_sigmoid(a + b * _logit(mean_model)))
    out = np.clip(p * (mapped_mean / mean_model), CPR_LO, CPR_HI)
    return out, True


def describe(asof):
    fit = fit_asof(asof)
    if fit is None:
        return "asof %s: window too short, mapping off" % pd.Timestamp(asof).date()
    a, b = fit
    return "asof %s: logit mapping a=%.4f b=%.4f" % (pd.Timestamp(asof).date(), a, b)


if __name__ == "__main__":
    p = _panel()
    print("panel cells %d, %s .. %s" % (len(p), p.date.min().date(), p.date.max().date()))
    for d in ["2021-06-30", "2023-01-31", "2025-06-30"]:
        print(" ", describe(d))
        path = np.linspace(0.05, 0.09, 33)
        for mode in ["off", "scalar", "pointwise"]:
            out, applied = apply_mapping(path, d, mode)
            print("     %-9s mean %.4f -> %.4f (applied=%s)"
                  % (mode, path.mean(), out.mean(), applied))
