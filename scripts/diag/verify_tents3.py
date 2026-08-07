#!/usr/bin/env python3
"""verify_tents3.py -- does the third (curvature) factor absorb the dy2 residual?

diag_duration_gap.py found: fitting level/slope durations by regression (sized
as well as the data allows) still leaves |t(dy2)| > 2 at seven of nine coupons.
dy2 is out-of-basis for a two-factor level/slope hedge by construction, so that
was the honest test. Advisor's read: a separate 2yr component, fixed via a third
tent -> level/slope/curvature. This script tests it the same way, non-circularly:
dy2 was never used to fit the model durations, so if D_curve genuinely captures
what dy2 was picking up, adding it as a THIRD regressor (not fitting a coefficient
on dy2 itself) should still reduce residual dy2 exposure when tested via the
model-implied durations from the tents3 panel.

TWO CHECKS, NOT ONE
  (1) Model durations (D_level, D_slope, D_curve) applied directly (i.e. exactly
      what model_hedge_krd.py priced) -- the same "off-the-shelf" test as
      t_dy2_mdl in diag_duration_gap.py, extended to three factors.
  (2) Fitted durations (regression on realized returns, expanding-window,
      36-month burn-in) using [level, slope, curve] as regressors -- the same
      "as well as the data allows" test as t_dy2_fit there, extended to three
      factors. This is the one that actually tests whether curvature is real
      structure or just noise the regression can chase.

  Neither check regresses against dy2 to size anything -- dy2 is only ever the
  thing being tested against afterward, so this is not circular the way
  hedge_panel_validation.csv's in-sample t_dy2=-0.12 was.
"""
import os
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
os.chdir(BASE)
DATA, OUT = "data", "outputs"
MIN_WINDOW = 36

PANEL = f"{OUT}/model_hedge_panel_10_tents3_pinnedfixed.csv"
BASELINE = f"{OUT}/model_hedge_panel_10_span_pinnedfixed.MAPOFF.csv"


def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    se = np.sqrt(np.diag(float(r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(XtX)))
    return co, se


def load_dy2(panel):
    t = pd.read_csv(f"{DATA}/treasury_yields.csv", parse_dates=["DATE"])
    me = t.set_index("DATE")[["2yr", "5yr", "10yr"]].sort_index().resample("ME").last()
    me["dy2"] = me["2yr"].diff()
    me["dy5"] = me["5yr"].diff()
    me["dy10"] = me["10yr"].diff()
    me = me.reset_index()
    me["key"] = me.DATE.dt.to_period("M")
    p = panel.copy()
    p["key"] = pd.PeriodIndex(p.ret_month, freq="M")
    out = p.merge(me[["key", "dy2", "dy5", "dy10"]], on="key", how="left")

    chk = out.dropna(subset=["dy5", "dy10", "d_level"])
    recon = (chk.dy5 + chk.dy10) / 2.0
    corr = float(np.corrcoef(recon, chk.d_level)[0, 1])
    print("  d_level reconstruction corr: %.4f (must be > 0.95)" % corr)
    assert corr > 0.95, "merge key wrong -- stop, do not trust anything below"

    u = out.copy()
    u["unh"] = u.tba_total_return - u.income
    g0 = u[u.coupon == u.coupon.min()].dropna(subset=["unh", "dy2"])
    co, se = ols(g0.unh.values, np.column_stack([np.ones(len(g0)), g0.dy2.values]))
    t0 = co[1] / se[1]
    print("  unhedged t(dy2) sanity, coupon %.1f: %.2f (must be strongly negative)"
          % (g0.coupon.iloc[0], t0))
    assert abs(t0) > 2.0, "dy2 alignment still wrong -- stop"
    return out


def t_on(vals, dy2, mask):
    m = mask & ~np.isnan(vals) & ~np.isnan(dy2)
    if m.sum() < 10:
        return np.nan
    co, se = ols(vals[m], np.column_stack([np.ones(m.sum()), dy2[m]]))
    return co[1] / se[1]


def fitted_hedged(g, factors, expanding=True):
    y = (g.tba_total_return - g.income).values
    X = np.column_stack([np.ones(len(g))] + [g[f].values for f in factors])
    if not expanding:
        co, _ = ols(y, X)
        durs = -100 * co[1:]
        hedged = y + (X[:, 1:] @ durs) / 100.0
        return hedged, np.ones(len(g), bool), durs
    hedged = np.full(len(g), np.nan)
    ok = np.zeros(len(g), bool)
    for i in range(MIN_WINDOW, len(g)):
        co, _ = ols(y[:i], X[:i])
        durs = -100 * co[1:]
        hedged[i] = y[i] + (X[i, 1:] @ durs) / 100.0
        ok[i] = True
    return hedged, ok, None


def main():
    print("=== loading tents3 panel ===")
    p3 = pd.read_csv(PANEL)
    print("  %d rows, %d months, %d coupons" % (len(p3), p3.ret_month.nunique(), p3.coupon.nunique()))
    p3 = load_dy2(p3)

    print("\n=== loading baseline (2-factor) panel for comparison ===")
    p2 = pd.read_csv(BASELINE)
    p2 = load_dy2(p2)

    print("\n%5s | %25s | %25s" % ("cpn", "2-FACTOR (level,slope)", "3-FACTOR (level,slope,curve)"))
    print("%5s | %8s %8s %8s | %8s %8s %8s %8s"
          % ("", "t_dy2_mdl", "t_dy2_fit", "", "t_dy2_mdl", "t_dy2_fit", "D_curve", ""))

    rows = []
    for c in sorted(p3.coupon.unique()):
        g2 = p2[p2.coupon == c].dropna(subset=["hedged", "d_level", "d_slope", "dy2"]).sort_values("ret_month")
        g3 = p3[p3.coupon == c].dropna(subset=["hedged", "d_level", "d_slope", "d_curve", "dy2"]).sort_values("ret_month")

        # 2-factor: model-implied (off the shelf) and expanding-fitted
        t2_mdl = t_on(g2.hedged.values, g2.dy2.values, np.ones(len(g2), bool))
        h2f, ok2, _ = fitted_hedged(g2, ["d_level", "d_slope"])
        t2_fit = t_on(h2f, g2.dy2.values, ok2)

        # 3-factor: model-implied (curvature priced directly by the pricer)
        t3_mdl = t_on(g3.hedged.values, g3.dy2.values, np.ones(len(g3), bool))
        h3f, ok3, _ = fitted_hedged(g3, ["d_level", "d_slope", "d_curve"])
        t3_fit = t_on(h3f, g3.dy2.values, ok3)

        print("%5.1f | %8.2f %8.2f          | %8.2f %8.2f %8.3f"
              % (c, t2_mdl, t2_fit, t3_mdl, t3_fit, g3.D_curve.mean()))
        rows.append(dict(coupon=c, t2_mdl=t2_mdl, t2_fit=t2_fit,
                         t3_mdl=t3_mdl, t3_fit=t3_fit, D_curve_mean=g3.D_curve.mean()))

    r = pd.DataFrame(rows)
    n_sig_2 = int((r.t2_fit.abs() > 2).sum())
    n_sig_3 = int((r.t3_fit.abs() > 2).sum())
    print("\n=== reading ===")
    print("  fitted (expanding-window) coupons with |t_dy2| > 2:")
    print("    2-factor: %d of 9   (diag_duration_gap.py found 7 of 9)" % n_sig_2)
    print("    3-factor: %d of 9" % n_sig_3)
    print("  If 3-factor is materially lower, curvature absorbs real structure.")
    print("  If unchanged, the third tent is not fixing what dy2 was flagging.")

    r.to_csv(f"{OUT}/verify_tents3.csv", index=False)
    print("\nwrote %s/verify_tents3.csv" % OUT)


if __name__ == "__main__":
    main()
