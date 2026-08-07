#!/usr/bin/env python3
"""diag_curvature_failure.py -- why did adding D_curve not reduce t(dy2)?

Three candidate explanations, tested in order of cheapness:

  1. COLLINEARITY. If d_curve is highly correlated with d_level or d_slope in
     this sample, a joint OLS regression will show curvature as redundant even
     when the underlying factor is real -- the regression can't tell which
     regressor to credit. Check corr(d_curve, d_level), corr(d_curve, d_slope).

  2. WRONG SIGN / SCALE ON D_curve (model side). If the pricer's D_curve moves
     the wrong way relative to actual dy2 risk, it would ADD noise rather than
     absorb it. Check corr(D_curve, dy2) and corr(d_curve, dy2) directly --
     the model duration should predict something about realized curvature risk,
     and the realized curve factor should correlate with dy2 (it should, since
     dy2 is one of its three inputs by construction: d_curve = 2*d5-d2-d10).

  3. REGIME DEPENDENCE. If curvature risk is concentrated in specific stretches
     (eg the 2022-23 hiking cycle) rather than uniform across 99 months, a single
     pooled coefficient will underfit it. Split t(dy2) by year under the fitted
     hedge and see whether 2022-23 looks different from the rest.

Also checks the advisor's literal curvature definition against what was built:
he wrote "the middle moving against the two ends" -- C = dy5 - (dy2+dy10)/2.
This script used C = 2*dy5 - dy2 - dy10 = 2*(dy5 - (dy2+dy10)/2), a factor of 2
different in scale only. Confirmed algebraically here rather than assumed.
"""
import os
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
os.chdir(BASE)
DATA, OUT = "data", "outputs"
MIN_WINDOW = 36
PANEL = f"{OUT}/model_hedge_panel_10_tents3_pinnedfixed.csv"


def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    se = np.sqrt(np.diag(float(r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(XtX)))
    return co, se


def load(panel):
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
    corr = float(np.corrcoef((chk.dy5 + chk.dy10) / 2.0, chk.d_level)[0, 1])
    assert corr > 0.95, "merge key wrong, corr=%.3f" % corr
    return out


def main():
    p = pd.read_csv(PANEL)
    p = load(p)
    g = p[p.coupon == p.coupon.min()].dropna(
        subset=["d_level", "d_slope", "d_curve", "D_curve", "dy2", "dy5", "dy10"]).sort_values("ret_month")

    print("=== curvature definition check ===")
    c_theirs = g.dy5 - (g.dy2 + g.dy10) / 2.0
    c_mine = g.d_curve
    ratio = (c_mine / c_theirs).replace([np.inf, -np.inf], np.nan).dropna()
    print("  built definition / advisor's literal definition, median ratio: %.4f"
        % ratio.median())
    print("  (should be 2.0 -- built used 2*dy5-dy2-dy10, he wrote dy5-(dy2+dy10)/2)")

    print("\n=== 1. collinearity among rate factors (all 9 coupons pooled by date) ===")
    dd = p.drop_duplicates("ret_month")[["d_level", "d_slope", "d_curve"]].dropna()
    corr_mat = dd.corr()
    print(corr_mat.round(3).to_string())
    print("  corr(d_curve, d_level) = %.3f" % corr_mat.loc["d_curve", "d_level"])
    print("  corr(d_curve, d_slope) = %.3f" % corr_mat.loc["d_curve", "d_slope"])
    print("  (if either is large, the joint regression can't separate them cleanly)")

    print("\n=== 2. does D_curve (model) and d_curve (realized) actually track dy2 ===")
    for coup in sorted(p.coupon.unique())[::2]:
        gc = p[p.coupon == coup].dropna(subset=["D_curve", "d_curve", "dy2"])
        print("  coupon %.1f:  corr(D_curve, dy2) = %6.3f   corr(d_curve, dy2) = %6.3f"
              % (coup, gc.D_curve.corr(gc.dy2), gc.d_curve.corr(gc.dy2)))
    print("  d_curve should correlate with dy2 by construction (dy2 is one of its")
    print("  three inputs); if D_curve (model side) does NOT correlate with dy2,")
    print("  the pricer's curvature duration isn't actually sensitive to 2yr risk")
    print("  regardless of the tent shape being geometrically correct.")

    print("\n=== 3. regime dependence: t(dy2) by year, 2-factor vs 3-factor fitted ===")
    for c in sorted(p.coupon.unique()):
        gg = p[p.coupon == c].dropna(
            subset=["d_level", "d_slope", "d_curve", "dy2",
                    "tba_total_return", "income"]).sort_values("ret_month").reset_index(drop=True)
        y = (gg.tba_total_return - gg.income).values
        X2 = np.column_stack([np.ones(len(gg)), gg.d_level.values, gg.d_slope.values])
        X3 = np.column_stack([np.ones(len(gg)), gg.d_level.values, gg.d_slope.values, gg.d_curve.values])
        h2 = np.full(len(gg), np.nan)
        h3 = np.full(len(gg), np.nan)
        for i in range(MIN_WINDOW, len(gg)):
            co2, _ = ols(y[:i], X2[:i]); h2[i] = y[i] + (X2[i, 1:] @ (-100 * co2[1:])) / 100.0
            co3, _ = ols(y[:i], X3[:i]); h3[i] = y[i] + (X3[i, 1:] @ (-100 * co3[1:])) / 100.0
        gg["h2"] = h2; gg["h3"] = h3
        gg["year"] = pd.PeriodIndex(gg.ret_month, freq="M").year
        if c not in (2.5, 5.0, 6.5):
            continue
        print("  coupon %.1f:" % c)
        for yr, ygrp in gg.dropna(subset=["h2", "h3"]).groupby("year"):
            if len(ygrp) < 6:
                continue
            def tstat(vals):
                m = ygrp.dy2.notna()
                co, se = ols(vals[m].values, np.column_stack(
                    [np.ones(m.sum()), ygrp.dy2[m].values]))
                return co[1] / se[1]
            print("    %d (n=%2d)  t2=%6.2f  t3=%6.2f" % (yr, len(ygrp), tstat(ygrp.h2), tstat(ygrp.h3)))


if __name__ == "__main__":
    main()
