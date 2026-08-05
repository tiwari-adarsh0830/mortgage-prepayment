#!/usr/bin/env python3
"""
diag_cpr_mapping.py (v2) -- expanding-window calibration mapping from model
cohort CPR to realized CPR.

SPEC (advisor, 2026-08-04):
  "For each month t, we take the history through t-1. For every coupon-month cell,
   we have two things: what the model predicts (aggregated to the cohort CPR) and
   what happened. With those two numbers, we want to fit a curve: realized CPR as
   a function of (model CPR, incentive). Concretely, this could be something as
   simple as a regression of log realized refi against log model with coefficients
   which vary by the incentive. Then ... every CPR path (the baseline and each
   bumped path) goes through this mapping before it gets priced. This has to be
   done through an expanding-window way, so that we can't let the look ahead bias
   seep in."

CHANGES FROM v1
  1. LOGIT-SPACE SPECS. Under UPB weighting the log-space fits are steep (log-model
     slope ~1.80) and map high-incentive cells past CPR = 1.0 -- in-range failed at
     100% of cutoffs on the empirical support. That is not a grid artifact: CPR is
     a rate on [0,1] and a log-log fit does not respect the ceiling. price_path
     computes 1-(1-clip(cpr,0,0.99))**(1/12), so an out-of-range CPR would be
     silently clamped and priced wrongly with no error raised. The logit family
     (logit realized ~ logit model, with the same incentive terms) is bounded in
     (0,1) by construction. This DEVIATES from the advisor's literal "log realized
     against log model" and the deviation is deliberate and reportable, not silent.
  2. bucketed specs no longer crash path_application (v1 looked bucketed up in
     SMOOTH_SPECS, where it does not live; it only became best-scoring once UPB
     weighting was switched on).
  3. Range diagnostics report WHERE a mapping leaves (0,1), not just that it did.

WEIGHTING (v1 patch 02, retained)
  UPB-weighted least squares throughout. The 33 realized-zero cells hold 0.0001%
  of at-risk UPB (median 2.91e6 against 1.41e11, median 48 loans against 754,961);
  under unweighted OLS they carried the same weight as cohorts holding a hundred
  billion dollars, which is what made the headline flip between --zero-mode drop
  and --zero-mode floor. Under weighting, drop and floor agree to four decimals
  and the --min-upb sweep is flat from 0 to 1e10, so the size filter is redundant
  and retained only as a control.

DATA SOURCES (deliberate)
  Model : outputs/forecast_cpr_timeseries_gfee050.csv -- built by
          stage2_forecast_cpr_gfee050.build_batch_constant_refi, the same module
          model_hedge_krd.py imports at line 114, so forecast_cpr and cpr_path are
          the same construction. GFEE=0.50 matches the pricer. forecast_cpr is the
          MEAN over ages 1..33, not a single-age query.
  Realized: outputs/realized_cpr_by_coupon_v6_upb.csv, column cpr_upb. UPB-weighted,
          matching scurve_params_asof() which fits the terminal segment on cpr_upb.
          outputs/forecast_vs_realized_cpr_gfee050.csv is NOT used: its realized_cpr
          is count-weighted (matches cpr_count to 1.1e-4, cpr_upb only to 0.524).

TIMING
  The forecast file's `date` is information-date keyed: date=2018-01-01 <->
  info_date=2018-01-31 <-> ret_month=2018-02 in model_hedge_panel_*.csv. Realized
  CPR at date d is activity during month d, so the window is strictly date < cutoff,
  with --lag-months to widen the gap for reporting lag.

USAGE
  python3 scripts/diag/diag_cpr_mapping.py --min-window 36 --lag-months 1 --tag v2
  python3 scripts/diag/diag_cpr_mapping.py --unweighted --tag v2_unw
  python3 scripts/diag/diag_cpr_mapping.py --sweep-min-upb
"""
import os
import io
import json
import argparse
import contextlib
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
OUT = os.path.join(BASE, "outputs")

FCST_FILE = "forecast_cpr_timeseries_gfee050.csv"
REAL_FILE = "realized_cpr_by_coupon_v6_upb.csv"

COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

INC_EDGES = [-99.0, -2.5, -1.5, -0.5, 0.5, 1.5, 99.0]
INC_LABELS = ["<=-2.5", "-2.5..-1.5", "-1.5..-0.5", "-0.5..0.5", "0.5..1.5", ">1.5"]
MIN_BUCKET_N = 20

CPR_LO, CPR_HI = 1e-6, 1.0 - 1e-6

MIN_UPB = 0.0
UNWEIGHTED = False


# ----------------------------------------------------------------- transforms
def to_log(c):
    return np.log(np.clip(c, CPR_LO, CPR_HI))


def from_log(v):
    return np.exp(v)


def to_logit(c):
    c = np.clip(c, CPR_LO, CPR_HI)
    return np.log(c / (1.0 - c))


def from_logit(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))


FAMILY = {
    "log": (to_log, from_log),
    "logit": (to_logit, from_logit),
}


# --------------------------------------------------------------- design matrices
def _X_lin(t, inc):
    return np.column_stack([np.ones_like(t), t])


def _X_inc(t, inc):
    return np.column_stack([np.ones_like(t), t, inc])


def _X_interact(t, inc):
    return np.column_stack([np.ones_like(t), t, inc, inc * t])


def _X_quad(t, inc):
    return np.column_stack([np.ones_like(t), t, inc, inc ** 2, inc * t])


# spec -> (family, design builder) ; bucketed specs handled separately
SMOOTH_SPECS = {
    "loglog":            ("log", _X_lin),
    "loglog_inc":        ("log", _X_inc),
    "loglog_interact":   ("log", _X_interact),
    "loglog_quad":       ("log", _X_quad),
    "logit":             ("logit", _X_lin),
    "logit_inc":         ("logit", _X_inc),
    "logit_interact":    ("logit", _X_interact),
    "logit_quad":        ("logit", _X_quad),
}
BUCKET_SPECS = {"bucketed": "log", "bucketed_logit": "logit"}
ALL_SPECS = ["identity"] + list(SMOOTH_SPECS.keys()) + list(BUCKET_SPECS.keys())


# ------------------------------------------------------------------- data load
def load_panel(zero_mode, eps):
    f = pd.read_csv(os.path.join(OUT, FCST_FILE), parse_dates=["date"])
    r = pd.read_csv(os.path.join(OUT, REAL_FILE), parse_dates=["date"])
    keep = [c for c in ["date", "implied_mbs_coupon", "cpr_upb", "cpr_count",
                        "upb_atrisk", "n_atrisk"] if c in r.columns]
    r = r[keep].rename(columns={"implied_mbs_coupon": "coupon"})

    m = f.merge(r, on=["date", "coupon"], how="inner")
    m = m[m.coupon.isin(COUPONS)].copy()
    n_raw = len(m)
    m = m[m.forecast_cpr > 0].copy()

    if "upb_atrisk" not in m.columns:
        raise RuntimeError("upb_atrisk missing -- needed for weighting")
    n_before = len(m)
    n_small_zero = int(((m.upb_atrisk < MIN_UPB) & (m.cpr_upb <= 0)).sum())
    m = m[m.upb_atrisk >= MIN_UPB].copy()

    n_zero = int((m.cpr_upb <= 0).sum())
    if zero_mode == "drop":
        m = m[m.cpr_upb > 0].copy()
    else:
        m["cpr_upb"] = m.cpr_upb.clip(lower=eps)

    m["log_model"] = to_log(m.forecast_cpr.values)
    m["log_real"] = to_log(m.cpr_upb.values)
    m["logit_model"] = to_logit(m.forecast_cpr.values)
    m["logit_real"] = to_logit(m.cpr_upb.values)
    m["inc"] = m.refi_incentive.values
    m["inc_bucket"] = pd.cut(m.inc, INC_EDGES, labels=INC_LABELS)
    m = m.sort_values(["date", "coupon"]).reset_index(drop=True)

    print("=== panel ===")
    print("  merged cells            : %d" % n_raw)
    print("  min-upb filter (%.3g)   : dropped %d (%d of them zero-CPR)"
          % (MIN_UPB, n_before - len(m) - (n_zero if zero_mode == "drop" else 0),
             n_small_zero))
    print("  realized cpr <= 0       : %d (zero-mode=%s)" % (n_zero, zero_mode))
    print("  final cells             : %d" % len(m))
    print("  months                  : %d  (%s .. %s)"
          % (m.date.nunique(), m.date.min().date(), m.date.max().date()))
    print("  weighting               : %s"
          % ("UNWEIGHTED (OLS)" if UNWEIGHTED else "UPB-weighted (WLS)"))
    return m


# ------------------------------------------------------------------- fitting
def _wls(X, y, w):
    if w is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    s = np.sqrt(w / w.mean())
    beta, *_ = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)
    return beta


def spec_inputs(spec, df):
    """(transformed model, transformed realized) for this spec's family."""
    fam = SMOOTH_SPECS[spec][0] if spec in SMOOTH_SPECS else BUCKET_SPECS[spec]
    if fam == "log":
        return df.log_model.values, df.log_real.values, fam
    return df.logit_model.values, df.logit_real.values, fam


def fit_spec(spec, df, w):
    t, y, fam = spec_inputs(spec, df)
    inc = df.inc.values
    if spec in SMOOTH_SPECS:
        X = SMOOTH_SPECS[spec][1](t, inc)
        return {"kind": "smooth", "fam": fam, "beta": _wls(X, y, w)}

    pooled = _wls(_X_lin(t, inc), y, w)
    coefs, fellback = {}, []
    bucket = df.inc_bucket.values
    for lab in INC_LABELS:
        sel = (bucket == lab)
        n = int(sel.sum())
        if n >= MIN_BUCKET_N:
            coefs[lab] = _wls(np.column_stack([np.ones(n), t[sel]]), y[sel],
                              None if w is None else w[sel])
        else:
            coefs[lab] = pooled
            fellback.append(lab)
    return {"kind": "bucket", "fam": fam, "pooled": pooled,
            "by_bucket": coefs, "fellback": fellback}


def predict_cpr(spec, fit, model_cpr, inc, bucket=None):
    """Mapped CPR (not log, not logit) for raw model CPR inputs."""
    fam = fit["fam"]
    t = to_log(model_cpr) if fam == "log" else to_logit(model_cpr)
    if fit["kind"] == "smooth":
        v = SMOOTH_SPECS[spec][1](t, inc) @ fit["beta"]
    else:
        v = np.empty_like(t)
        barr = np.asarray(bucket, dtype=object)
        for i in range(len(t)):
            b = fit["by_bucket"].get(barr[i], fit["pooled"])
            v[i] = b[0] + b[1] * t[i]
    return FAMILY[fam][1](v)


# ---------------------------------------------------------------- safety checks
def build_support(panel, inc_step=0.25, margin=0.35, n_cpr=40):
    lo = np.floor(panel.inc.min() / inc_step) * inc_step
    hi = np.ceil(panel.inc.max() / inc_step) * inc_step
    edges = np.arange(lo, hi + inc_step, inc_step)
    support = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = panel[(panel.inc >= a) & (panel.inc < b)]
        if len(sel) < 3:
            continue
        l, h = np.log(sel.forecast_cpr.min()), np.log(sel.forecast_cpr.max())
        support.append((0.5 * (a + b),
                        np.exp(np.linspace(l - margin, h + margin, n_cpr))))
    print("  safety support          : %d incentive slices, inc %.2f..%.2f"
          % (len(support), edges[0], edges[-1]))
    return support


def safety_check(spec, fit, support):
    """Monotone increasing in model CPR, and mapped CPR inside (0,1). A
    non-monotone mapping inverts the KRD sign under a bump; an out-of-range one
    is silently clamped by price_path and priced wrongly."""
    mono, in_range = True, True
    worst_inc, worst_val = None, 0.0
    for inc, cprs in support:
        mv = predict_cpr(spec, fit, cprs, np.full_like(cprs, inc),
                         bucket=[bucket_of(inc)] * len(cprs))
        if np.any(np.diff(mv) <= 0):
            mono = False
        if np.any(mv <= 0.0) or np.any(mv >= 1.0):
            in_range = False
            if mv.max() > worst_val:
                worst_val, worst_inc = float(mv.max()), float(inc)
    return bool(mono), bool(in_range), worst_inc, worst_val


def bucket_of(inc):
    for lab, lo, hi in zip(INC_LABELS, INC_EDGES[:-1], INC_EDGES[1:]):
        if lo < inc <= hi:
            return lab
    return INC_LABELS[-1]


# ------------------------------------------------------------ expanding window
def expanding_window(panel, min_window, lag_months, support):
    dates = np.sort(panel.date.unique())
    rows, coef_rows = [], []
    n_skipped = 0

    for cutoff in dates:
        train_end = pd.Timestamp(cutoff) - pd.DateOffset(months=lag_months)
        tr = panel[panel.date < train_end]
        te = panel[panel.date == cutoff]
        if te.empty:
            continue
        if tr.date.nunique() < min_window:
            n_skipped += len(te)
            continue

        w = None if UNWEIGHTED else tr.upb_atrisk.values.astype(float)
        crow = {"cutoff": pd.Timestamp(cutoff), "n_train": len(tr),
                "n_train_months": tr.date.nunique(), "n_test": len(te)}
        mapped = {"identity": te.forecast_cpr.values}

        for spec in list(SMOOTH_SPECS) + list(BUCKET_SPECS):
            fit = fit_spec(spec, tr, w)
            mapped[spec] = predict_cpr(spec, fit, te.forecast_cpr.values,
                                       te.inc.values, te.inc_bucket.values)
            if fit["kind"] == "smooth":
                for j, bv in enumerate(fit["beta"]):
                    crow["%s_b%d" % (spec, j)] = float(bv)
            else:
                crow["%s_n_fellback" % spec] = len(fit["fellback"])
                for lab in INC_LABELS:
                    crow["%s_slope_%s" % (spec, lab)] = float(fit["by_bucket"][lab][1])
            mono, rng, winc, wval = safety_check(spec, fit, support)
            crow["%s_monotone" % spec] = mono
            crow["%s_in_range" % spec] = rng
            crow["%s_worst_inc" % spec] = winc
            crow["%s_worst_val" % spec] = wval

        coef_rows.append(crow)
        for spec in ALL_SPECS:
            mc = np.clip(mapped[spec], CPR_LO, CPR_HI)
            for i, (_, cell) in enumerate(te.reset_index().iterrows()):
                rows.append({
                    "date": cell.date, "coupon": cell.coupon, "spec": spec,
                    "inc": cell.inc, "inc_bucket": cell.inc_bucket,
                    "model_cpr": cell.forecast_cpr, "real_cpr": cell.cpr_upb,
                    "mapped_cpr": float(mc[i]),
                    "log_err": float(np.log(cell.cpr_upb) - np.log(mc[i])),
                    "upb_atrisk": float(cell.upb_atrisk),
                    "clamped": bool(mapped[spec][i] >= 1.0 or mapped[spec][i] <= 0.0),
                })

    print("  cells skipped (<%d mo)   : %d" % (min_window, n_skipped))
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


# ------------------------------------------------------------------- scoring
def variance_shares(df, col):
    v = df[col].var()
    if v <= 0 or len(df) < 10:
        return np.nan, np.nan
    out = []
    for g in ["date", "coupon"]:
        gm = df.groupby(g)[col].mean()
        raw = gm.var() / v
        k = df.groupby(g).size()
        exp_null = float(np.mean(1.0 / k)) * (1.0 - 1.0 / len(k))
        out.append(max(raw - exp_null, 0.0))
    return tuple(out)


def score(oos):
    print("\n=== out-of-sample scores by spec (%s) ==="
          % ("unweighted" if UNWEIGHTED else "UPB-weighted"))
    print("%-18s %6s %9s %9s %9s %9s %9s %7s"
          % ("spec", "n", "logRMSE", "logMAE", "lvlMAE", "time_adj", "cpn_adj", "clamp%"))
    summ = []
    for spec in ALL_SPECS:
        d = oos[oos.spec == spec]
        if d.empty:
            continue
        if UNWEIGHTED:
            w = np.ones(len(d)) / len(d)
        else:
            w = d.upb_atrisk.values.astype(float)
            w = w / w.sum()
        lr = float(np.sqrt(np.sum(w * d.log_err.values ** 2)))
        lm_ = float(np.sum(w * np.abs(d.log_err.values)))
        lv = float(np.sum(w * np.abs(d.mapped_cpr.values - d.real_cpr.values)))
        t_adj, c_adj = variance_shares(d, "log_err")
        clamp = 100.0 * d.clamped.mean()
        print("%-18s %6d %9.4f %9.4f %9.5f %9.3f %9.3f %6.1f%%"
              % (spec, len(d), lr, lm_, lv, t_adj, c_adj, clamp))
        summ.append({"spec": spec, "n": len(d), "log_rmse": lr, "log_mae": lm_,
                     "level_mae": lv, "time_share_adj": t_adj,
                     "coupon_share_adj": c_adj, "clamped_pct": clamp})
    return pd.DataFrame(summ)


def score_by_bucket(oos):
    print("\n=== OOS mean log error by incentive bucket (negative = still high) ===")
    piv = (oos.groupby(["spec", "inc_bucket"], observed=True).log_err
           .agg(["size", "mean"]).round(3).reset_index())
    wide = piv.pivot(index="inc_bucket", columns="spec", values="mean")
    print(wide.to_string())
    return piv


def score_by_year(oos):
    print("\n=== OOS log RMSE by year ===")
    t = oos.copy()
    t["year"] = t.date.dt.year
    w = (t.groupby(["spec", "year"]).log_err
         .apply(lambda s: float(np.sqrt(np.mean(s ** 2)))).unstack(0).round(4))
    print(w.to_string())
    return w


def stability(coefs):
    print("\n=== log/logit-model slope across cutoffs ===")
    for spec in SMOOTH_SPECS:
        c = "%s_b1" % spec
        if c not in coefs.columns:
            continue
        s = coefs[c]
        note = ""
        if SMOOTH_SPECS[spec][0] == "log":
            if s.min() > 1.0:
                note = "  slope>1 always: AMPLIFIES bump response, shortens duration"
            elif s.max() < 1.0:
                note = "  slope<1 always: COMPRESSES bump response, lengthens duration"
            else:
                note = "  crosses 1: duration effect sign not stable"
        print("  %-18s mean %6.3f  sd %5.3f  min %6.3f  max %6.3f%s"
              % (spec, s.mean(), s.std(), s.min(), s.max(), note))

    print("\n=== safety (share of cutoffs passing) ===")
    for spec in list(SMOOTH_SPECS) + list(BUCKET_SPECS):
        mc, rc = "%s_monotone" % spec, "%s_in_range" % spec
        if mc not in coefs.columns:
            continue
        line = ("  %-18s monotone %5.1f%%   in-range %5.1f%%"
                % (spec, 100 * coefs[mc].mean(), 100 * coefs[rc].mean()))
        if coefs[rc].mean() < 1.0:
            wv = coefs["%s_worst_val" % spec].max()
            wi = coefs.loc[coefs["%s_worst_val" % spec].idxmax(), "%s_worst_inc" % spec]
            line += "   (max mapped CPR %.3f at inc %+.2f)" % (wv, wi)
        print(line)


def path_application(coefs, panel, best_spec, w_last):
    """cpr_path returns 33 monthly values; the mapping is fit on their mean.
    Scalar mode rescales the path by mapped/model at the path mean, preserving
    the model's age shape. Pointwise applies the mapping to each of the 33
    values, at ages the mapping was never fit on."""
    print("\n=== path application: scalar vs pointwise (%s) ===" % best_spec)
    if best_spec == "identity":
        print("  identity -- nothing to apply.")
        return
    fit = fit_spec(best_spec, panel, w_last)
    shape = np.linspace(0.55, 1.45, 33)
    print("  %8s %12s %12s %12s %10s"
          % ("inc", "mean model", "scalar", "pointwise", "rel diff"))
    for inc in [-4.0, -2.5, -1.0, 0.0, 1.0, 2.0]:
        sub = panel[np.abs(panel.inc - inc) < 0.5]
        mean_model = float(sub.forecast_cpr.mean()) if len(sub) else 0.1
        path = mean_model * shape
        b = [bucket_of(inc)]
        mm = float(predict_cpr(best_spec, fit, np.array([mean_model]),
                               np.array([inc]), b)[0])
        scalar = path * (mm / mean_model)
        pw = predict_cpr(best_spec, fit, path, np.full(33, inc), b * 33)
        rel = (pw.mean() - scalar.mean()) / scalar.mean()
        print("  %8.2f %12.4f %12.4f %12.4f %9.1f%%"
              % (inc, mean_model, scalar.mean(), pw.mean(), 100 * rel))


def score_quiet(oos):
    with contextlib.redirect_stdout(io.StringIO()):
        return score(oos)


def sweep(args):
    global MIN_UPB
    print("\n=== min-upb sweep (weighted=%s) ===" % (not UNWEIGHTED))
    print("%10s %7s %10s %10s %10s" % ("min_upb", "cells", "identity", "best", "red"))
    for thr in [0.0, 1e6, 1e7, 1e8, 5e8, 1e9, 5e9, 1e10]:
        MIN_UPB = thr
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                panel = load_panel(args.zero_mode, args.eps)
                support = build_support(panel)
                oos, _ = expanding_window(panel, args.min_window,
                                          args.lag_months, support)
            if oos.empty:
                continue
            s = score_quiet(oos)
            ident = float(s[s.spec == "identity"].log_rmse.iloc[0])
            nid = s[s.spec != "identity"]
            bn = nid.sort_values("log_rmse").iloc[0]
            print("%10.3g %7d %10.4f %10.4f %9.1f%%  (%s)"
                  % (thr, len(panel), ident, float(bn.log_rmse),
                     100 * (ident - float(bn.log_rmse)) / ident, bn.spec))
        except Exception as e:
            print("%10.3g   failed: %s" % (thr, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-window", type=int, default=36)
    ap.add_argument("--lag-months", type=int, default=1)
    ap.add_argument("--zero-mode", default="drop", choices=["floor", "drop"])
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--min-upb", type=float, default=0.0)
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--sweep-min-upb", action="store_true")
    ap.add_argument("--safe-only", action="store_true",
                    help="pick the best spec only among those passing both "
                         "safety checks at every cutoff")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    global MIN_UPB, UNWEIGHTED
    MIN_UPB, UNWEIGHTED = args.min_upb, args.unweighted

    print("diag_cpr_mapping v2  min_window=%d lag=%d zero_mode=%s min_upb=%.3g"
          % (args.min_window, args.lag_months, args.zero_mode, args.min_upb))
    if args.sweep_min_upb:
        sweep(args)
        return

    panel = load_panel(args.zero_mode, args.eps)
    support = build_support(panel)
    oos, coefs = expanding_window(panel, args.min_window, args.lag_months, support)
    if oos.empty:
        print("\nNo scored cells -- min-window too long.")
        return

    summ = score(oos)
    score_by_bucket(oos)
    score_by_year(oos)
    stability(coefs)

    cand = summ[summ.spec != "identity"].copy()
    if args.safe_only:
        ok = [s for s in cand.spec
              if coefs.get("%s_monotone" % s, pd.Series([False])).all()
              and coefs.get("%s_in_range" % s, pd.Series([False])).all()]
        cand = cand[cand.spec.isin(ok)]
        print("\n  --safe-only: %d of %d specs pass both checks at every cutoff"
              % (len(cand), len(summ) - 1))
    best = cand.sort_values("log_rmse").iloc[0].spec if len(cand) else "identity"

    ident = float(summ[summ.spec == "identity"].log_rmse.iloc[0])
    bl = float(summ[summ.spec == best].log_rmse.iloc[0]) if best != "identity" else ident
    print("\n=== headline ===")
    print("  identity (no mapping)   : %.4f" % ident)
    print("  best %-18s: %.4f  (%.1f%% reduction)"
          % (best, bl, 100 * (ident - bl) / ident))
    print("  log family best  : %s"
          % summ[summ.spec.str.startswith("loglog")].sort_values("log_rmse")
            .iloc[0][["spec", "log_rmse"]].to_dict())
    print("  logit family best: %s"
          % summ[summ.spec.str.startswith("logit")].sort_values("log_rmse")
            .iloc[0][["spec", "log_rmse"]].to_dict())

    w_last = None if UNWEIGHTED else panel.upb_atrisk.values.astype(float)
    path_application(coefs, panel, best, w_last)

    o1 = os.path.join(OUT, "cpr_mapping_oos_%s.csv" % args.tag)
    o2 = os.path.join(OUT, "cpr_mapping_coefs_%s.csv" % args.tag)
    o3 = os.path.join(OUT, "cpr_mapping_summary_%s.json" % args.tag)
    oos.to_csv(o1, index=False)
    coefs.to_csv(o2, index=False)
    json.dump({"args": vars(args), "summary": summ.to_dict(orient="records"),
               "best_spec": best}, open(o3, "w"), indent=2, default=str)
    print("\nwrote %s\n      %s\n      %s" % (o1, o2, o3))


if __name__ == "__main__":
    main()
