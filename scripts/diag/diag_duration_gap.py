#!/usr/bin/env python3
"""
diag_duration_gap.py -- is the residual level exposure an omitted factor, or are
the pricer's durations simply the wrong size?

WHY THIS, AND WHY NOT ANOTHER CPR FIX

Four consecutive interventions aimed at the CPR forecast have left the level
t-statistics essentially unchanged: seasoned-only S-curve (a wash), spread
control (made level exposure larger), terminal floor refit (capture 72.2 ->
74.1%), and the expanding-window CPR mapping (27.3% better forecast, hedge
t-statistics uniformly worse). When that many targeted corrections all fail to
move a number, the usual cause is that the diagnosis is wrong.

outputs/hedge_panel_validation.csv already contains the evidence. It regresses
hedged returns on the 2yr change, a tenor deliberately OUTSIDE the level/slope
span, so it is not circular the way regressing on dy5/dy10 would be:

    coupon   t_dy2 unhedged   t_dy2 hedge A (fitted)   t_dy2 hedge B (model D)
      2.5         -8.00               -0.12                    -3.72
      3.0         -8.07               -0.05                    -3.75

Hedge A neutralises out-of-basis exposure at the two coupons where hedge B fails
worst. Same returns, same factors, opposite verdict. If a missing risk factor
were driving discount-coupon returns, regression coefficients on level and slope
could not neutralise it either -- an omitted orthogonal factor is by definition
not spanned by the regressors. So the residual IS spanned by level and slope,
and the difference between the two hedges is the SIZE of the durations, not the
factor set.

That reframes the problem: not "fix the CPR forecast" but "find out why the
pricer's durations are too short at discounts."

THREE CAVEATS ON THAT VALIDATION FILE, WHICH THIS SCRIPT EXISTS TO REMOVE
  1. n=36 there against 99 months in the KRD panel -- different samples, so the
     comparison is not yet like-for-like.
  2. r2_A ~ 0.96 because hedge A's coefficients are fitted on the same returns
     they are then evaluated against. Its dy2 pass is in-sample and therefore
     flattered. An honest version needs expanding-window or split-sample fitting.
  3. The file is 10 lines and predates the pinned-fixed floor and spanning bumps.

WHAT THIS SCRIPT DOES
  On the SAME 99-month panel the KRD hedge uses, for each coupon:
    (a) fit level/slope durations by regression (hedge A style), in-sample;
    (b) refit them expanding-window, so the comparison is not in-sample-flattered;
    (c) place both against the pricer's model durations;
    (d) test residual exposure to dy2 under each, the out-of-basis check;
    (e) report the ratio fitted/model per coupon, which is the claim to be tested:
        Phase 21 found implied duration ~8.05y at coupon 2.5 against a model 5.46,
        with Spearman(coupon, implied) = -1.000.

  If fitted durations exceed model durations systematically at discounts, the
  pricer understates duration there and that gap is the whole story. If they
  agree, this hypothesis is dead and the residual is something else.

DELIBERATELY NOT A FACTOR SEARCH
  With 99 months and 9 coupons, throwing candidate factors at the residual will
  find something significant by chance. dy2 is here because it is out-of-basis
  and was already the project's chosen honest check, not because it was selected
  from a menu.

USAGE
  python3 scripts/diag/diag_duration_gap.py
  python3 scripts/diag/diag_duration_gap.py --panel outputs/model_hedge_panel_10_span_pinnedfixed_mapfrozen.csv
"""
import os
import argparse
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")

DEFAULT_PANEL = os.path.join(OUT, "model_hedge_panel_10_span_pinnedfixed.MAPOFF.csv")


def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    s2 = float(r @ r) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    ss = float(((y - y.mean()) ** 2).sum())
    return co, se, (1 - float(r @ r) / ss if ss > 1e-12 else np.nan), r


def load_dy2(panel):
    """2yr Treasury monthly change, aligned to the panel's info_date convention.

    The panel's d_level/d_slope are built from month-end to month-end changes, so
    dy2 must be the same differencing on the same dates or the out-of-basis test
    compares misaligned series."""
    t = pd.read_csv(os.path.join(DATA, "treasury_yields.csv"), parse_dates=["DATE"])
    t = t.set_index("DATE")[["2yr", "5yr", "10yr"]].sort_index()
    me = t.resample("ME").last()
    me["dy2"] = me["2yr"].diff()
    me["dy5"] = me["5yr"].diff()
    me["dy10"] = me["10yr"].diff()

    p = panel.copy()
    # d_level is the change OVER the return month, so ret_month is the key.
    # Keying on info_date gives corr 0.025 against the panel's own d_level;
    # keying on ret_month gives 0.994. Verified, not assumed.
    p["key"] = pd.PeriodIndex(p["ret_month"], freq="M")
    me["key"] = me.index.to_period("M")
    mm = me.reset_index()[["key", "dy2", "dy5", "dy10"]]
    out = p.merge(mm, on="key", how="left")

    # sanity: the panel's own d_level should reconstruct from dy5/dy10
    chk = out.dropna(subset=["dy5", "dy10", "d_level"])
    if not len(chk):
        raise RuntimeError("no rows survived the Treasury merge -- key mismatch")
    recon = (chk.dy5 + chk.dy10) / 2.0
    err = (recon - chk.d_level).abs()
    corr = float(np.corrcoef(recon, chk.d_level)[0, 1])
    worst = chk.loc[err.idxmax(), "ret_month"]
    print("  d_level reconstruction: corr %.4f, max abs err %.4f (worst %s), "
          "d_level sd %.4f" % (corr, err.max(), worst, chk.d_level.std()))
    if corr < 0.95:
        raise RuntimeError(
            "dy series misaligned with the panel (corr %.4f). Every t_dy2 below "
            "would be a regression on noise. Check the merge key before reading "
            "any result." % corr)
    if err.max() > 0.5 * chk.d_level.std():
        raise RuntimeError(
            "reconstruction error %.4f is large against d_level sd %.4f -- more "
            "than a month-end convention difference." % (err.max(), chk.d_level.std()))
    return out


def fitted_durations(g, expanding=False, min_months=36):
    """Level/slope durations implied by regressing UNHEDGED excess on [level, slope].

    excess = tba_total_return - income; dP/P = -(D_level*level + D_slope*slope)/100,
    so D_level = -100 * coefficient. Same convention as recompute()'s resid_dur.

    expanding=True refits at each month on strictly earlier data and applies the
    prior fit, so the resulting hedge is not evaluated on the returns that set its
    coefficients."""
    g = g.sort_values("ret_month")
    y = (g.tba_total_return - g.income).values
    X = np.column_stack([np.ones(len(g)), g.d_level.values, g.d_slope.values])

    if not expanding:
        co, se, r2, _ = ols(y, X)
        hedged = y + ((-100 * co[1]) * g.d_level.values
                      + (-100 * co[2]) * g.d_slope.values) / 100.0
        return -100 * co[1], -100 * co[2], r2, hedged, np.ones(len(g), bool)

    DL = np.full(len(g), np.nan)
    DS = np.full(len(g), np.nan)
    for i in range(len(g)):
        if i < min_months:
            continue
        co, _, _, _ = ols(y[:i], X[:i])
        DL[i], DS[i] = -100 * co[1], -100 * co[2]
    ok = ~np.isnan(DL)
    hedged = np.full(len(g), np.nan)
    hedged[ok] = y[ok] + (DL[ok] * g.d_level.values[ok]
                          + DS[ok] * g.d_slope.values[ok]) / 100.0
    return np.nanmean(DL), np.nanmean(DS), np.nan, hedged, ok


def t_on_dy2(vals, dy2, mask):
    m = mask & ~np.isnan(vals) & ~np.isnan(dy2)
    if m.sum() < 10:
        return np.nan
    X = np.column_stack([np.ones(m.sum()), dy2[m]])
    co, se, _, _ = ols(vals[m], X)
    return co[1] / se[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--min-months", type=int, default=36)
    args = ap.parse_args()

    p = pd.read_csv(args.panel)
    print("panel: %s" % os.path.basename(args.panel))
    print("  %d rows, %d months, %d coupons"
          % (len(p), p.ret_month.nunique(), p.coupon.nunique()))
    p = load_dy2(p)

    # An unhedged MBS return must show large rate exposure. If it does not,
    # the dy2 series is wrong regardless of what the reconstruction check said.
    _u = p.dropna(subset=["dy2"]).copy()
    _u["unh"] = _u.tba_total_return - _u.income
    _g = _u[_u.coupon == _u.coupon.min()]
    _X = np.column_stack([np.ones(len(_g)), _g.dy2.values])
    _co, _se, _, _ = ols(_g.unh.values, _X)
    print("  unhedged t(dy2) at coupon %.1f: %.2f (expect strongly negative)"
          % (_g.coupon.iloc[0], _co[1] / _se[1]))
    if abs(_co[1] / _se[1]) < 2.0:
        raise RuntimeError(
            "unhedged exposure to dy2 is insignificant (t=%.2f). That cannot be "
            "right for an MBS return; the dy2 alignment is still wrong."
            % (_co[1] / _se[1]))

    print("\n=== fitted vs model durations, same 99-month sample ===")
    print("%5s %9s %9s %8s %9s %9s %9s %9s"
          % ("cpn", "D_model", "D_fit_IS", "ratio", "D_fit_EW",
             "t_dy2_non", "t_dy2_mdl", "t_dy2_fit"))
    rows = []
    for c, g in p.groupby("coupon"):
        g = g.dropna(subset=["hedged", "d_level", "d_slope"]).sort_values("ret_month")
        dy2 = g.dy2.values
        unh = (g.tba_total_return - g.income).values

        dl_is, ds_is, r2_is, hedged_is, m_is = fitted_durations(g, False)
        dl_ew, ds_ew, _, hedged_ew, m_ew = fitted_durations(g, True, args.min_months)

        d_model = g.D_level.mean()
        t_non = t_on_dy2(unh, dy2, np.ones(len(g), bool))
        t_mdl = t_on_dy2(g.hedged.values, dy2, np.ones(len(g), bool))
        t_fit = t_on_dy2(hedged_ew, dy2, m_ew)

        print("%5.1f %9.3f %9.3f %8.2f %9.3f %9.2f %9.2f %9.2f"
              % (c, d_model, dl_is, dl_is / d_model if d_model else np.nan,
                 dl_ew, t_non, t_mdl, t_fit))
        rows.append(dict(coupon=c, D_model=d_model, D_fit_IS=dl_is,
                         D_fit_EW=dl_ew, ratio=dl_is / d_model if d_model else np.nan,
                         t_dy2_none=t_non, t_dy2_model=t_mdl, t_dy2_fit=t_fit))

    r = pd.DataFrame(rows)
    print("\n=== reading ===")
    print("  If D_fit exceeds D_model at discounts and the gap narrows with coupon,")
    print("  the pricer understates duration there and that is the whole story.")
    print("  If t_dy2 is near zero under fitted durations but not under model")
    print("  durations, the residual is spanned by level/slope and no factor is")
    print("  missing -- the durations are simply the wrong size.")
    try:
        from scipy.stats import spearmanr
        rho, pv = spearmanr(r.coupon, r.ratio)
        print("\n  Spearman(coupon, D_fit/D_model) = %.3f (p=%.4f)" % (rho, pv))
        print("  Phase 21 found Spearman(coupon, implied_duration) = -1.000;")
        print("  a strongly negative rho here is the same finding from a second angle.")
    except Exception as e:
        print("  spearman unavailable: %s" % e)

    print("\n  IS = in-sample (coefficients fitted on the returns they hedge --")
    print("  flattered, shown for comparability with hedge_panel_validation.csv).")
    print("  EW = expanding-window, %d-month burn-in, the honest version."
          % args.min_months)

    o = os.path.join(OUT, "duration_gap_%s.csv"
                     % os.path.basename(args.panel).replace(".csv", ""))
    r.to_csv(o, index=False)
    print("\nwrote %s" % o)


if __name__ == "__main__":
    main()
