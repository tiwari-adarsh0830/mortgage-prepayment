#!/usr/bin/env python3
"""
diag_cpr_mapping.py -- expanding-window calibration mapping from model cohort CPR
to realized CPR.

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

WHAT THIS SCRIPT DOES / DOES NOT DO
  Does: fits the mapping, scores it OUT OF SAMPLE on an expanding window, and runs
  the checks that determine whether it is safe to put inside a pricer.
  Does NOT: modify model_hedge_krd.py or reprice anything. Read-only apart from
  its own CSV/JSON outputs.

DATA SOURCES (deliberate)
  Model : outputs/forecast_cpr_timeseries_gfee050.csv
          date, coupon, note_rate, pmms, refi_incentive, forecast_cpr
          Built by stage2_forecast_cpr_gfee050.build_batch_constant_refi, the same
          module model_hedge_krd.py imports at line 114, so forecast_cpr and
          cpr_path are the same construction. GFEE=0.50 matches the pricer.
          NOTE: forecast_cpr is the MEAN over ages 1..33 (forecast_cpr() returns
          cpr.mean()), not a single-age query. The 0.140 figure in the Aug 3 email
          is model CPR at age 33 specifically -- a different object. Both are
          in-range; this one is what the pricer consumes.
  Realized: outputs/realized_cpr_by_coupon_v6_upb.csv, column cpr_upb.
          UPB-weighted, matching scurve_params_asof() which fits the terminal
          segment on cpr_upb. outputs/forecast_vs_realized_cpr_gfee050.csv is NOT
          used: its realized_cpr column is count-weighted (matches cpr_count to
          1.1e-4, cpr_upb only to 0.524), predating the 2026-07-06 UPB rebuild.
          Fitting the mapping on that file would calibrate months 1-33 to
          count-weighted realized and months 34-360 to UPB-weighted realized.

TIMING
  The forecast file's `date` is information-date keyed: date=2018-01-01 corresponds
  to info_date=2018-01-31 and ret_month=2018-02 in model_hedge_panel_*.csv, and its
  pmms matches the panel's pmms for that info_date. Realized CPR at date d is
  activity during month d, not observable when pricing at the start of d. The
  expanding window is therefore STRICTLY date < cutoff, with --lag-months to widen
  the gap for reporting lag.

USAGE
  python3 scripts/diag/diag_cpr_mapping.py
  python3 scripts/diag/diag_cpr_mapping.py --lag-months 1 --min-window 30
  python3 scripts/diag/diag_cpr_mapping.py --zero-mode drop --tag droptest
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
OUT = os.path.join(BASE, "outputs")

FCST_FILE = "forecast_cpr_timeseries_gfee050.csv"
REAL_FILE = "realized_cpr_by_coupon_v6_upb.csv"

COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

# set from argv in main(); module-level so the fit/score helpers can see them
MIN_UPB = 0.0
UNWEIGHTED = False

# incentive buckets for the advisor's "coefficients which vary by the incentive"
INC_EDGES = [-99.0, -2.5, -1.5, -0.5, 0.5, 1.5, 99.0]
INC_LABELS = ["<=-2.5", "-2.5..-1.5", "-1.5..-0.5", "-0.5..0.5", "0.5..1.5", ">1.5"]

# a bucket needs at least this many training cells to get its own coefficients;
# otherwise it falls back to the pooled log-log fit for that cutoff
MIN_BUCKET_N = 20

# grid used for the monotonicity / range safety checks
GRID_CPR = np.linspace(0.005, 0.60, 120)
GRID_INC = np.linspace(-5.0, 3.0, 33)


# ---------------------------------------------------------------- data loading
def load_panel(zero_mode, eps):
    """Join model forecast to UPB-weighted realized CPR on (date, coupon)."""
    f = pd.read_csv(os.path.join(OUT, FCST_FILE), parse_dates=["date"])
    r = pd.read_csv(os.path.join(OUT, REAL_FILE), parse_dates=["date"])

    keep = ["date", "implied_mbs_coupon", "cpr_upb", "cpr_count",
            "upb_atrisk", "n_atrisk"]
    keep = [c for c in keep if c in r.columns]
    r = r[keep].rename(columns={"implied_mbs_coupon": "coupon"})

    m = f.merge(r, on=["date", "coupon"], how="inner")
    m = m[m.coupon.isin(COUPONS)].copy()

    n_raw = len(m)
    m = m[m.forecast_cpr > 0].copy()
    n_pos_model = len(m)

    if "upb_atrisk" not in m.columns:
        raise RuntimeError("upb_atrisk missing from the realized panel -- the "
                           "cohort-size filter and UPB weighting both need it")
    n_before = len(m)
    n_small_zero = int(((m.upb_atrisk < MIN_UPB) & (m.cpr_upb <= 0)).sum())
    m = m[m.upb_atrisk >= MIN_UPB].copy()
    print("  min-upb filter (%.3g)         : dropped %d cells (%d of them zero-CPR)"
          % (MIN_UPB, n_before - len(m), n_small_zero))

    n_zero = int((m.cpr_upb <= 0).sum())
    if zero_mode == "drop":
        m = m[m.cpr_upb > 0].copy()
    elif zero_mode == "floor":
        m["cpr_upb"] = m.cpr_upb.clip(lower=eps)
    else:
        raise ValueError("zero-mode must be drop or floor")

    m["log_model"] = np.log(m.forecast_cpr.values)
    m["log_real"] = np.log(m.cpr_upb.values)
    m["inc"] = m.refi_incentive.values
    m["inc_bucket"] = pd.cut(m.inc, INC_EDGES, labels=INC_LABELS)
    m = m.sort_values(["date", "coupon"]).reset_index(drop=True)

    print("=== panel ===")
    print("  merged cells                : %d" % n_raw)
    print("  dropped (model cpr <= 0)    : %d" % (n_raw - n_pos_model))
    print("  realized cpr <= 0           : %d (zero-mode=%s)" % (n_zero, zero_mode))
    print("  final cells                 : %d" % len(m))
    print("  months                      : %d  (%s .. %s)"
          % (m.date.nunique(), m.date.min().date(), m.date.max().date()))
    print("  coupons                     : %s" % sorted(m.coupon.unique()))
    if n_zero:
        z = pd.read_csv(os.path.join(OUT, FCST_FILE), parse_dates=["date"]).merge(
            r, on=["date", "coupon"], how="inner")
        z = z[z.cpr_upb <= 0]
        print("  zero cells by coupon        : %s"
              % z.groupby("coupon").size().to_dict())
    return m


# ------------------------------------------------------------------- the specs
# Each spec builds a design matrix from (log_model, inc). Prediction is in log
# space; the mapped CPR is exp(prediction).

def _X_identity(lm, inc):
    return None  # handled specially: mapped = model


def _X_loglog(lm, inc):
    return np.column_stack([np.ones_like(lm), lm])


def _X_loglog_inc(lm, inc):
    return np.column_stack([np.ones_like(lm), lm, inc])


def _X_loglog_interact(lm, inc):
    return np.column_stack([np.ones_like(lm), lm, inc, inc * lm])


def _X_loglog_quad(lm, inc):
    return np.column_stack([np.ones_like(lm), lm, inc, inc ** 2, inc * lm])


SMOOTH_SPECS = {
    "loglog": _X_loglog,
    "loglog_inc": _X_loglog_inc,
    "loglog_interact": _X_loglog_interact,
    "loglog_quad": _X_loglog_quad,
}


def _wls(X, y, w):
    """Weighted least squares via sqrt-weight scaling. w=None -> OLS."""
    if w is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    s = np.sqrt(w / w.mean())
    beta, *_ = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)
    return beta


def fit_smooth(spec, lm, inc, y, w=None):
    X = SMOOTH_SPECS[spec](lm, inc)
    return _wls(X, y, w)


def predict_smooth(spec, beta, lm, inc):
    X = SMOOTH_SPECS[spec](lm, inc)
    return X @ beta


def fit_bucketed(lm, inc, y, bucket, w=None):
    """Advisor's literal form: separate (intercept, log-model slope) per incentive
    bucket. Buckets with fewer than MIN_BUCKET_N training cells fall back to the
    pooled log-log coefficients, recorded so the fallback rate is visible."""
    pooled = fit_smooth("loglog", lm, inc, y, w)
    coefs, fellback = {}, []
    for lab in INC_LABELS:
        sel = (bucket == lab).values if hasattr(bucket, "values") else (bucket == lab)
        n = int(sel.sum())
        if n >= MIN_BUCKET_N:
            Xb = np.column_stack([np.ones(n), lm[sel]])
            b = _wls(Xb, y[sel], None if w is None else w[sel])
            coefs[lab] = b
        else:
            coefs[lab] = pooled
            fellback.append(lab)
    return {"pooled": pooled, "by_bucket": coefs, "fellback": fellback}


def predict_bucketed(fit, lm, inc, bucket):
    out = np.empty_like(lm)
    barr = np.asarray(bucket, dtype=object)
    for i in range(len(lm)):
        b = fit["by_bucket"].get(barr[i], fit["pooled"])
        out[i] = b[0] + b[1] * lm[i]
    return out


ALL_SPECS = ["identity"] + list(SMOOTH_SPECS.keys()) + ["bucketed"]


# ------------------------------------------------------- expanding-window score
def expanding_window(panel, min_window, lag_months, support):
    """For each cutoff month, fit on strictly-earlier months and predict that
    month's cells. Returns per-cell OOS predictions and the coefficient path."""
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

        lm_tr = tr.log_model.values
        inc_tr = tr.inc.values
        y_tr = tr.log_real.values
        w_tr = None if UNWEIGHTED else tr.upb_atrisk.values.astype(float)
        lm_te = te.log_model.values
        inc_te = te.inc.values

        preds = {"identity": np.log(te.forecast_cpr.values)}
        crow = {"cutoff": pd.Timestamp(cutoff), "n_train": len(tr),
                "n_train_months": tr.date.nunique(), "n_test": len(te)}

        for spec in SMOOTH_SPECS:
            beta = fit_smooth(spec, lm_tr, inc_tr, y_tr, w_tr)
            preds[spec] = predict_smooth(spec, beta, lm_te, inc_te)
            for j, bv in enumerate(beta):
                crow["%s_b%d" % (spec, j)] = float(bv)

        bfit = fit_bucketed(lm_tr, inc_tr, y_tr, tr.inc_bucket, w_tr)
        preds["bucketed"] = predict_bucketed(bfit, lm_te, inc_te, te.inc_bucket)
        crow["bucketed_n_fellback"] = len(bfit["fellback"])
        for lab in INC_LABELS:
            crow["bucketed_slope_%s" % lab] = float(bfit["by_bucket"][lab][1])

        # safety checks on this cutoff's fitted mapping
        for spec in SMOOTH_SPECS:
            beta = np.array([crow["%s_b%d" % (spec, j)]
                             for j in range(len(SMOOTH_SPECS[spec](
                                 np.array([0.0]), np.array([0.0]))[0]))])
            mono, rng = safety_check(spec, beta, support)
            crow["%s_monotone" % spec] = mono
            crow["%s_in_range" % spec] = rng

        coef_rows.append(crow)

        for spec in ALL_SPECS:
            for i, (_, cell) in enumerate(te.iterrows()):
                rows.append({
                    "date": cell.date, "coupon": cell.coupon, "spec": spec,
                    "inc": cell.inc, "inc_bucket": cell.inc_bucket,
                    "model_cpr": cell.forecast_cpr, "real_cpr": cell.cpr_upb,
                    "mapped_cpr": float(np.exp(preds[spec][i])),
                    "log_err": float(cell.log_real - preds[spec][i]),
                    "upb_atrisk": float(cell.upb_atrisk),
                })

    print("\n  cells skipped (window < %d months): %d" % (min_window, n_skipped))
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def build_support(panel, inc_step=0.25, margin=0.35, n_cpr=40):
    """Observed (incentive, model-CPR) region, widened by `margin` in log-CPR to
    cover what a +/-25bp bump reaches. Checking a full cross product instead
    tests combinations the pricer never queries -- model CPR 0.60 at incentive
    -5.0 does not occur, since deep-discount cells sit near 0.064."""
    lo = np.floor(panel.inc.min() / inc_step) * inc_step
    hi = np.ceil(panel.inc.max() / inc_step) * inc_step
    edges = np.arange(lo, hi + inc_step, inc_step)
    support = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = panel[(panel.inc >= a) & (panel.inc < b)]
        if len(sel) < 3:
            continue
        lcl, lch = np.log(sel.forecast_cpr.min()), np.log(sel.forecast_cpr.max())
        support.append((0.5 * (a + b),
                        np.exp(np.linspace(lcl - margin, lch + margin, n_cpr))))
    print("  safety support: %d incentive slices, inc %.2f..%.2f"
          % (len(support), edges[0], edges[-1]))
    return support


def safety_check(spec, beta, support):
    """Monotone increasing in model CPR, and mapped CPR inside (0,1), over the
    empirical support. A non-monotone mapping inverts the sign of the KRD under
    a bump, so a spec failing this cannot be priced regardless of its score."""
    mono, in_range = True, True
    for inc, cprs in support:
        lm = np.log(cprs)
        v = predict_smooth(spec, beta, lm, np.full_like(lm, inc))
        if np.any(np.diff(v) <= 0):
            mono = False
        mv = np.exp(v)
        if np.any(mv <= 0) or np.any(mv >= 1.0):
            in_range = False
    return bool(mono), bool(in_range)


# -------------------------------------------------------------------- scoring
def variance_shares(df, col):
    """Share of residual variance attributable to time and to coupon. Group means
    of a pure-noise residual have nonzero variance, so the naive share is biased
    up when a grouping has many levels; the debiased version subtracts the
    expected contribution under no group effect."""
    v = df[col].var()
    if v <= 0 or len(df) < 10:
        return np.nan, np.nan, np.nan, np.nan
    out = []
    for g in ["date", "coupon"]:
        gm = df.groupby(g)[col].mean()
        raw = gm.var() / v
        k = df.groupby(g).size()
        exp_null = float(np.mean(1.0 / k)) * (1.0 - 1.0 / len(k))
        out += [raw, max(raw - exp_null, 0.0)]
    return tuple(out)


def score(oos):
    print("\n=== out-of-sample scores by spec ===")
    print("%-18s %6s %9s %9s %9s %9s %9s %9s"
          % ("spec", "n", "logRMSE", "logMAE", "lvlMAE", "time_sh", "time_adj", "cpn_adj"))
    summ = []
    for spec in ALL_SPECS:
        d = oos[oos.spec == spec]
        if d.empty:
            continue
        if UNWEIGHTED or "upb_atrisk" not in d.columns:
            lr = np.sqrt(np.mean(d.log_err ** 2))
            lm_ = np.mean(np.abs(d.log_err))
            lv = np.mean(np.abs(d.mapped_cpr - d.real_cpr))
        else:
            w = d.upb_atrisk.values.astype(float)
            w = w / w.sum()
            lr = float(np.sqrt(np.sum(w * d.log_err.values ** 2)))
            lm_ = float(np.sum(w * np.abs(d.log_err.values)))
            lv = float(np.sum(w * np.abs(d.mapped_cpr.values - d.real_cpr.values)))
        t_raw, t_adj, c_raw, c_adj = variance_shares(d, "log_err")
        print("%-18s %6d %9.4f %9.4f %9.5f %9.3f %9.3f %9.3f"
              % (spec, len(d), lr, lm_, lv, t_raw, t_adj, c_adj))
        summ.append({"spec": spec, "n": len(d), "log_rmse": lr, "log_mae": lm_,
                     "level_mae": lv, "time_share_raw": t_raw,
                     "time_share_adj": t_adj, "coupon_share_adj": c_adj})
    return pd.DataFrame(summ)


def score_by_bucket(oos):
    print("\n=== out-of-sample mean log error by incentive bucket ===")
    print("   (negative = mapped still above realized)")
    piv = (oos.groupby(["spec", "inc_bucket"], observed=True).log_err
           .agg(["size", "mean"]).round(3).reset_index())
    wide = piv.pivot(index="inc_bucket", columns="spec", values="mean")
    n = piv.pivot(index="inc_bucket", columns="spec", values="size").iloc[:, 0]
    wide.insert(0, "n", n)
    print(wide.to_string())
    return piv


def score_by_year(oos):
    print("\n=== out-of-sample log RMSE by year ===")
    t = oos.copy()
    t["year"] = t.date.dt.year
    w = (t.groupby(["spec", "year"]).log_err
         .apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
         .unstack(0).round(4))
    print(w.to_string())
    return w


def coefficient_stability(coefs):
    print("\n=== coefficient stability across cutoffs ===")
    for spec in SMOOTH_SPECS:
        c = "%s_b1" % spec
        if c not in coefs.columns:
            continue
        s = coefs[c]
        print("  %-18s log-model slope: mean %.3f  sd %.3f  min %.3f  max %.3f"
              % (spec, s.mean(), s.std(), s.min(), s.max()))
        if s.min() > 1.0:
            print("      slope > 1 at every cutoff: the mapping AMPLIFIES the CPR "
                  "response to a rate bump, shortening model duration.")
        elif s.max() < 1.0:
            print("      slope < 1 at every cutoff: the mapping COMPRESSES the CPR "
                  "response to a rate bump, lengthening model duration.")
        else:
            print("      slope crosses 1 across cutoffs -- sign of the duration "
                  "effect is not stable.")
    print("\n=== safety checks (share of cutoffs passing) ===")
    for spec in SMOOTH_SPECS:
        mc, rc = "%s_monotone" % spec, "%s_in_range" % spec
        if mc in coefs.columns:
            print("  %-18s monotone %5.1f%%   in-range %5.1f%%"
                  % (spec, 100 * coefs[mc].mean(), 100 * coefs[rc].mean()))
    if "bucketed_n_fellback" in coefs.columns:
        print("\n  bucketed: mean buckets falling back to pooled per cutoff: %.2f of %d"
              % (coefs.bucketed_n_fellback.mean(), len(INC_LABELS)))


def path_application(oos, coefs, best_spec):
    """cpr_path returns 33 monthly values; the mapping was fit on their mean.
    Scalar mode rescales the whole path by mapped/model computed at the path mean,
    preserving the model's age shape. Pointwise mode applies the mapping to each
    of the 33 values, at ages the mapping was never fit on. Quantify the gap."""
    print("\n=== path application: scalar vs pointwise ===")
    if best_spec == "identity" or coefs.empty:
        print("  best spec is identity -- nothing to apply.")
        return
    last = coefs.iloc[-1]
    nb = len(SMOOTH_SPECS[best_spec](np.array([0.0]), np.array([0.0]))[0])
    beta = np.array([last["%s_b%d" % (best_spec, j)] for j in range(nb)])

    # a stylised within-window age profile: model CPR rises with age over 1..33.
    # The RATIO of endpoint to mean is what matters, so the profile shape is the
    # only input; replace with a real cpr_path() draw when wiring in.
    shape = np.linspace(0.55, 1.45, 33)
    print("  %8s %12s %12s %12s %10s"
          % ("inc", "mean model", "scalar", "pointwise", "rel diff"))
    for inc in [-4.0, -2.5, -1.0, 0.0, 1.0, 2.0]:
        mean_model = 0.065 if inc <= -2.5 else (0.12 if inc < 0 else 0.30)
        path = mean_model * shape
        mapped_mean = float(np.exp(predict_smooth(
            best_spec, beta, np.array([np.log(mean_model)]), np.array([inc]))[0]))
        scalar = path * (mapped_mean / mean_model)
        pointwise = np.exp(predict_smooth(
            best_spec, beta, np.log(path), np.full(33, inc)))
        rel = (pointwise.mean() - scalar.mean()) / scalar.mean()
        print("  %8.2f %12.4f %12.4f %12.4f %9.1f%%"
              % (inc, mean_model, scalar.mean(), pointwise.mean(), 100 * rel))
    print("  A large gap means the choice of application mode is itself a modelling")
    print("  decision, not an implementation detail.")


# ------------------------------------------------------------------------ main
def sweep(args):
    """Headline across cohort-size thresholds. If the sign of the result moves
    with the threshold, the threshold is doing the work and the result is not
    reportable."""
    global MIN_UPB
    print("\n=== min-upb sweep (weighted=%s) ===" % (not UNWEIGHTED))
    print("%10s %7s %10s %10s %10s"
          % ("min_upb", "cells", "identity", "best", "reduction"))
    for thr in [0.0, 1e6, 1e7, 1e8, 5e8, 1e9, 5e9, 1e10]:
        MIN_UPB = thr
        try:
            panel = load_panel(args.zero_mode, args.eps)
            support = build_support(panel)
            oos, _ = expanding_window(panel, args.min_window, args.lag_months,
                                      support)
            if oos.empty:
                continue
            s = score_quiet(oos)
            ident = float(s[s.spec == "identity"].log_rmse.iloc[0])
            nid = s[s.spec != "identity"]
            best = float(nid.log_rmse.min())
            bname = nid.sort_values("log_rmse").iloc[0].spec
            print("%10.3g %7d %10.4f %10.4f %9.1f%%  (%s)"
                  % (thr, len(panel), ident, best,
                     100 * (ident - best) / ident, bname))
        except Exception as e:
            print("%10.3g   failed: %s" % (thr, e))


def score_quiet(oos):
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return score(oos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-window", type=int, default=24,
                    help="minimum training months before the mapping is applied")
    ap.add_argument("--lag-months", type=int, default=0,
                    help="extra gap between training data and the priced month")
    ap.add_argument("--zero-mode", default="drop", choices=["floor", "drop"],
                    help="handling of realized cpr <= 0 cells")
    ap.add_argument("--eps", type=float, default=0.002,
                    help="floor value for zero realized cells; must be a "
                         "plausible CPR, not a numerical epsilon")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--min-upb", type=float, default=1e9,
                    help="drop cells with at-risk UPB below this, whatever their "
                         "realized CPR -- an outcome-independent filter")
    ap.add_argument("--unweighted", action="store_true",
                    help="control: OLS instead of UPB-weighted least squares")
    ap.add_argument("--sweep-min-upb", action="store_true",
                    help="report the headline across thresholds and exit")
    args = ap.parse_args()

    global MIN_UPB, UNWEIGHTED
    MIN_UPB = args.min_upb
    UNWEIGHTED = args.unweighted

    if args.sweep_min_upb:
        sweep(args)
        return

    print("diag_cpr_mapping.py  min_window=%d lag=%d zero_mode=%s eps=%g"
          % (args.min_window, args.lag_months, args.zero_mode, args.eps))
    if args.zero_mode == "floor" and args.eps < 1e-3:
        print("  WARNING: eps=%g floors zero cells at log(eps)=%.2f, far outside"
              % (args.eps, np.log(args.eps)))
        print("           the realized log-CPR range. Scores will be dominated by"
              " those cells rather than by the mapping.")

    panel = load_panel(args.zero_mode, args.eps)
    support = build_support(panel)
    oos, coefs = expanding_window(panel, args.min_window, args.lag_months, support)
    if oos.empty:
        print("\nNo scored cells -- min-window too long for this sample.")
        return

    summ = score(oos)
    score_by_bucket(oos)
    score_by_year(oos)
    coefficient_stability(coefs)

    non_id = summ[summ.spec != "identity"]
    best = non_id.sort_values("log_rmse").iloc[0].spec if not non_id.empty else "identity"
    base = float(summ[summ.spec == "identity"].log_rmse.iloc[0])
    bl = float(summ[summ.spec == best].log_rmse.iloc[0])
    print("\n=== headline ===")
    print("  identity (no mapping) OOS log RMSE : %.4f" % base)
    print("  best spec (%s) : %.4f  (%.1f%% reduction)"
          % (best, bl, 100 * (base - bl) / base))
    print("  incremental value of incentive terms over plain log-log:")
    if "loglog" in set(summ.spec):
        p = float(summ[summ.spec == "loglog"].log_rmse.iloc[0])
        for s in ["loglog_inc", "loglog_interact", "loglog_quad", "bucketed"]:
            if s in set(summ.spec):
                q = float(summ[summ.spec == s].log_rmse.iloc[0])
                print("      %-18s %.4f  (%+.1f%% vs loglog)"
                      % (s, q, 100 * (q - p) / p))

    path_application(oos, coefs, best)

    o1 = os.path.join(OUT, "cpr_mapping_oos_%s.csv" % args.tag)
    o2 = os.path.join(OUT, "cpr_mapping_coefs_%s.csv" % args.tag)
    o3 = os.path.join(OUT, "cpr_mapping_summary_%s.json" % args.tag)
    oos.to_csv(o1, index=False)
    coefs.to_csv(o2, index=False)
    json.dump({"args": vars(args), "summary": summ.to_dict(orient="records"),
               "best_spec": best},
              open(o3, "w"), indent=2, default=str)
    print("\nwrote %s\n      %s\n      %s" % (o1, o2, o3))


if __name__ == "__main__":
    main()
