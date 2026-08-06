#!/usr/bin/env python3
"""
verify_email_numbers_v2.py -- independent recomputation of every figure in the
draft email that has not already been confirmed through a second code path.

WHY THIS EXISTS
  The four hedge panels were verified by verify_map_modes.py, which reruns the
  regressions from the saved panels using normal equations rather than lstsq, and
  reproduced the Aug 3 control to 0.0044. Those numbers are safe.

  Everything from diag_cpr_mapping_v2.py is NOT: the 27% reduction, the 0.476 and
  0.346 log RMSEs, the safety percentages, the fitted slope, the deep-discount
  ratio. Those come from one script, written in one sitting, and nothing has
  reproduced them. In this session alone the headline flipped three times on an
  implementation choice rather than on the data -- a 1e-4 epsilon, a safety grid
  spanning combinations that never occur, and unweighted OLS giving a 48-loan
  cohort the same weight as a $141bn one -- and a fourth error (merging Treasury
  changes on info_date rather than ret_month, correlation 0.025 against the
  panel's own d_level) produced a full table of regressions on noise before a
  guard caught it. So none of these figures should go in an email unreproduced.

WHAT IT DOES NOT DO
  It does not import diag_cpr_mapping_v2. Every quantity is recomputed here from
  the source CSVs with independently written code, so a bug in that script cannot
  propagate into its own verification. Where this file and that one disagree, the
  disagreement is the finding.

FIGURES CHECKED (each tagged with the claim it supports)
  A. OOS log RMSE, identity vs mapping, and the percentage reduction   -> "about 27%"
  B. Deep-discount ratio, model vs realized below -2.5 incentive       -> "about 0.7 times"
  C. Fitted logit slope across cutoffs                                 -> "about 1.9"
  D. Peak mapped CPR under the log-log spec                            -> "peak 1.59"
  E. Bucketed spec: cutoffs with a negative slope in some bucket       -> "at every cutoff"
  F. Duration ratio under spanning, and under localized                -> "1.35x", "3.5x"
  G. Coupons where expanding-window fitted durations leave |t_dy2| > 2 -> "seven of nine"
  H. Capture under each map mode                                       -> "74 / 83 / 94"

USAGE
  python3 scripts/diag/verify_email_numbers_v2.py
"""
import os
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
os.chdir(BASE)
OUT = "outputs"
DATA = "data"

MIN_WINDOW, LAG = 36, 1
CPR_LO, CPR_HI = 1e-6, 1.0 - 1e-6
INC_EDGES = [-99.0, -2.5, -1.5, -0.5, 0.5, 1.5, 99.0]
INC_LABELS = ["<=-2.5", "-2.5..-1.5", "-1.5..-0.5", "-0.5..0.5", "0.5..1.5", ">1.5"]
MIN_BUCKET_N = 20

PANELS = {
    "off":       "outputs/model_hedge_panel_10_span_pinnedfixed.MAPOFF.csv",
    "frozen":    "outputs/model_hedge_panel_10_span_pinnedfixed_mapfrozen.csv",
    "scalar":    "outputs/model_hedge_panel_10_span_pinnedfixed_mapscalar.csv",
    "pointwise": "outputs/model_hedge_panel_10_span_pinnedfixed_mappointwise.csv",
    "localized": "outputs/model_hedge_panel_10_local_pinnedfixed.csv",
}


def logit(c):
    c = np.clip(np.asarray(c, float), CPR_LO, CPR_HI)
    return np.log(c / (1 - c))


def sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))


def wls(X, y, w):
    s = np.sqrt(w / w.mean())
    b, *_ = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)
    return b


def panel():
    f = pd.read_csv(f"{OUT}/forecast_cpr_timeseries_gfee050.csv", parse_dates=["date"])
    r = pd.read_csv(f"{OUT}/realized_cpr_by_coupon_v6_upb.csv", parse_dates=["date"])
    r = r[["date", "implied_mbs_coupon", "cpr_upb", "upb_atrisk"]].rename(
        columns={"implied_mbs_coupon": "coupon"})
    m = f.merge(r, on=["date", "coupon"], how="inner")
    m = m[(m.forecast_cpr > 0) & (m.cpr_upb > 0)].copy()
    m["bucket"] = pd.cut(m.refi_incentive, INC_EDGES, labels=INC_LABELS)
    return m.sort_values(["date", "coupon"]).reset_index(drop=True)


def expanding(m, spec):
    """spec: 'logit' | 'loglog' | 'bucketed'. Returns per-cell OOS predictions."""
    rows, slopes, neg_bucket_cutoffs, ncut = [], [], 0, 0
    for cutoff in np.sort(m.date.unique()):
        end = pd.Timestamp(cutoff) - pd.DateOffset(months=LAG)
        tr, te = m[m.date < end], m[m.date == cutoff]
        if te.empty or tr.date.nunique() < MIN_WINDOW:
            continue
        ncut += 1
        w = tr.upb_atrisk.values.astype(float)
        if spec == "logit":
            x, y = logit(tr.forecast_cpr), logit(tr.cpr_upb)
            b = wls(np.column_stack([np.ones(len(tr)), x]), y, w)
            slopes.append(b[1])
            pred = sig(b[0] + b[1] * logit(te.forecast_cpr.values))
        elif spec == "loglog":
            x, y = np.log(tr.forecast_cpr), np.log(tr.cpr_upb)
            b = wls(np.column_stack([np.ones(len(tr)), x]), y, w)
            slopes.append(b[1])
            pred = np.exp(b[0] + b[1] * np.log(te.forecast_cpr.values))
        else:
            x, y = np.log(tr.forecast_cpr.values), np.log(tr.cpr_upb.values)
            pooled = wls(np.column_stack([np.ones(len(tr)), x]), y, w)
            bb, any_neg = {}, False
            for lab in INC_LABELS:
                s = (tr.bucket.values == lab)
                if s.sum() >= MIN_BUCKET_N:
                    bb[lab] = wls(np.column_stack([np.ones(int(s.sum())), x[s]]),
                                  y[s], w[s])
                else:
                    bb[lab] = pooled
                if bb[lab][1] <= 0:
                    any_neg = True
            neg_bucket_cutoffs += int(any_neg)
            pred = np.array([np.exp(bb.get(bk, pooled)[0]
                                    + bb.get(bk, pooled)[1] * np.log(fc))
                             for bk, fc in zip(te.bucket.values,
                                               te.forecast_cpr.values)])
        for i, (_, c) in enumerate(te.reset_index().iterrows()):
            rows.append(dict(date=c.date, coupon=c.coupon, bucket=c.bucket,
                             real=c.cpr_upb, model=c.forecast_cpr,
                             pred=float(pred[i]), upb=c.upb_atrisk))
    return pd.DataFrame(rows), np.array(slopes), neg_bucket_cutoffs, ncut


def wrmse(d, col):
    w = d.upb.values / d.upb.sum()
    e = np.log(d.real.values) - np.log(np.clip(d[col].values, CPR_LO, CPR_HI))
    return float(np.sqrt(np.sum(w * e ** 2)))


def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    se = np.sqrt(np.diag(float(r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(XtX)))
    return co, se


def main():
    m = panel()
    print("panel: %d cells, %d months, %s .. %s"
          % (len(m), m.date.nunique(), m.date.min().date(), m.date.max().date()))

    # ---- A: OOS RMSE and reduction ---------------------------------------
    d, sl_logit, _, ncut = expanding(m, "logit")
    ident = wrmse(d, "model")
    mapped = wrmse(d, "pred")
    print("\n[A] cutoffs scored %d, cells %d" % (ncut, len(d)))
    print("    identity log RMSE  %.4f" % ident)
    print("    logit    log RMSE  %.4f" % mapped)
    print("    reduction          %.1f%%   <- draft says 'about 27%%'"
          % (100 * (ident - mapped) / ident))

    # ---- B: deep-discount ratio ------------------------------------------
    dd = m[m.refi_incentive <= -2.5]
    print("\n[B] cells below -2.5 incentive: %d" % len(dd))
    print("    mean model %.4f, mean realized %.4f, ratio %.3f"
          % (dd.forecast_cpr.mean(), dd.cpr_upb.mean(),
             dd.cpr_upb.mean() / dd.forecast_cpr.mean()))
    dm = d[d.bucket == "<=-2.5"]
    if len(dm):
        print("    after mapping, OOS ratio %.3f   <- draft says 'about 0.7', closing"
              % (dm.real.mean() / dm.pred.mean()))

    # ---- C: fitted slope --------------------------------------------------
    print("\n[C] logit slope: mean %.3f  sd %.3f  min %.3f  max %.3f"
          % (sl_logit.mean(), sl_logit.std(), sl_logit.min(), sl_logit.max()))
    print("    draft says 'about 1.9'; above 1 at every cutoff: %s"
          % bool((sl_logit > 1).all()))

    # ---- D: log-log peak --------------------------------------------------
    dl, sl_log, _, _ = expanding(m, "loglog")
    print("\n[D] log-log peak mapped CPR on scored cells %.3f" % dl.pred.max())
    grid_max = 0.0
    for cutoff_slope, cutoff_int in [(sl_log.max(), None)]:
        pass
    # peak over the empirical support, per cutoff, as the safety check did
    for cutoff in np.sort(m.date.unique()):
        end = pd.Timestamp(cutoff) - pd.DateOffset(months=LAG)
        tr = m[m.date < end]
        if tr.date.nunique() < MIN_WINDOW:
            continue
        w = tr.upb_atrisk.values.astype(float)
        b = wls(np.column_stack([np.ones(len(tr)), np.log(tr.forecast_cpr)]),
                np.log(tr.cpr_upb), w)
        grid_max = max(grid_max, float(np.exp(b[0] + b[1] * np.log(m.forecast_cpr.max()))))
    print("    peak over observed model-CPR range %.3f   <- draft says 'peak 1.59'"
          % grid_max)
    print("    exceeds 1.0: %s" % (grid_max > 1.0))

    # ---- E: bucketed negative slopes -------------------------------------
    _, _, neg_cut, ncut_b = expanding(m, "bucketed")
    print("\n[E] bucketed: %d of %d cutoffs have a negative slope in some bucket"
          % (neg_cut, ncut_b))
    print("    draft says 'at every cutoff': %s" % (neg_cut == ncut_b))

    # ---- F/H: duration ratios and capture --------------------------------
    t = pd.read_csv(f"{DATA}/treasury_yields.csv", parse_dates=["DATE"])
    me = t.set_index("DATE")[["2yr", "5yr", "10yr"]].sort_index().resample("ME").last()
    me["dy2"] = me["2yr"].diff()
    me = me.reset_index()
    me["key"] = me.DATE.dt.to_period("M")

    print("\n[F/H] per panel: duration ratio, capture, |t_dy2|>2 count")
    for lab, path in PANELS.items():
        if not os.path.exists(path):
            print("    %-10s MISSING" % lab)
            continue
        p = pd.read_csv(path)
        p["key"] = pd.PeriodIndex(p.ret_month, freq="M")
        p = p.merge(me[["key", "dy2"]], on="key", how="left")
        ratios, mdl, imp, nsig = [], [], [], 0
        for c, g in p.groupby("coupon"):
            g = g.dropna(subset=["hedged", "d_level", "d_slope", "dy2"]).sort_values("ret_month")
            y = (g.tba_total_return - g.income).values
            X = np.column_stack([np.ones(len(g)), g.d_level.values, g.d_slope.values])
            co, _ = ols(y, X)
            DL = -100 * co[1]
            ratios.append(DL / g.D_level.mean() if g.D_level.mean() else np.nan)
            mdl.append(g.D_level.mean()); imp.append(DL)
            # expanding-window fitted hedge, then t on dy2
            hed = np.full(len(g), np.nan)
            for i in range(MIN_WINDOW, len(g)):
                cc, _ = ols(y[:i], X[:i])
                hed[i] = y[i] + ((-100 * cc[1]) * g.d_level.values[i]
                                 + (-100 * cc[2]) * g.d_slope.values[i]) / 100.0
            ok = ~np.isnan(hed)
            if ok.sum() > 10:
                cc, ss = ols(hed[ok], np.column_stack([np.ones(ok.sum()), g.dy2.values[ok]]))
                if abs(cc[1] / ss[1]) > 2:
                    nsig += 1
        mdl, imp = np.array(mdl), np.array(imp)
        cap = 100 * (mdl.max() - mdl.min()) / (imp.max() - imp.min())
        print("    %-10s median ratio %6.2f   capture %5.1f%%   |t_dy2|>2 at %d of 9"
              % (lab, np.nanmedian(ratios), cap, nsig))

    print("\n  draft claims to check against the above:")
    print("    spanning ratio ~1.35, localized ~3.5")
    print("    capture 74 / 83 / 94 for off / frozen / scalar")
    print("    'seven of nine' coupons significant on dy2 under fitted durations")


if __name__ == "__main__":
    main()
