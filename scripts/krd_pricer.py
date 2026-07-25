"""
Model-based key-rate durations (5y / 10y) for TBA coupons.

WHY THIS EXISTS
  The hedge in stage3_der_factor_shocks.py applies a single constant
  D_MOD_AVG = 6.5 years to every coupon. Empirical implied durations run
  8.05y (coupon 2.5) to 1.97y (coupon 6.5), so the hedge leaves large
  residual rate exposure that shows up as alpha.

  The Monte Carlo OAS engine cannot supply the replacement: its zero curve
  is built on a 33-month grid (MAX_SEQ), so bumping the 5y or 10y par node
  moves the discount curve by exactly zero (verified: 0/33 months move on a
  +1bp bump at either tenor; a 2y control moves 21/33). This module is the
  deterministic alternative -- model-implied, not empirical, but priced off
  a full 360-month curve so the tenor bumps actually bite.

METHOD
  Price coupon c as a 30y amortizing pass-through:
    - borrower note rate  = c + GFEE          (GFEE=0.50, matches the
                                               realized/forecast bucketing)
    - investor coupon     = c                 (servicing strip = GFEE)
    - prepayment          = flat lifetime CPR from the hazard-model forecast
                            for that coupon-month, converted to SMM
    - discounting         = bootstrapped zero curve, continuous comp,
                            360 monthly nodes
  KRD_tenor = (P_down - P_up) / (2 * P0 * h), central difference on a
  single-node par-yield bump.

LIMITATION (deliberate, flagged)
  CPR is held fixed under the bump, so these are STATIC-CASHFLOW key rate
  durations: no prepayment response, therefore no negative convexity. That
  is precisely the MBS-specific effect. Treated as a first pass -- compare
  krd5+krd10 against the empirical implied durations in hedge_diagnostic.json
  to size the gap before deciding whether a rate-responsive CPR is needed.
"""

import numpy as np
from scipy.interpolate import interp1d

MAT_LABELS = ['1mo','3mo','6mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
MAT_YEARS  = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

N_MONTHS = 360
GFEE     = 0.50          # MUST match stage3_der_factor_shocks.GFEE


def bootstrap_zeros(par_yields, n_months=N_MONTHS):
    """par_yields: mapping label -> par yield in %. Returns (n_months,) zero rates in %.
    Same recursion as risk_neutral_rates.bootstrap_zero_curve, but on a 360m grid."""
    z = {}
    for T, lab in zip(MAT_YEARS, MAT_LABELS):
        if T <= 1.0:
            z[T] = float(par_yields[lab])

    def get_zero(T):
        kT = sorted(z); kz = [z[t] for t in kT]
        if T <= kT[0]:  return kz[0]
        if T >= kT[-1]: return kz[-1]
        return float(interp1d(kT, kz, kind='linear')(T))

    def dfac(T, zpct):
        return np.exp(-zpct / 100.0 * T)

    for T, lab in zip(MAT_YEARS, MAT_LABELS):
        if T <= 1.0:
            continue
        c = float(par_yields[lab]) / 100.0
        pv = sum((c / 2) * 100 * dfac(t, get_zero(t)) for t in np.arange(0.5, T, 0.5))
        final = c / 2 * 100 + 100
        df_T = (100 - pv) / final
        if df_T <= 0:
            raise ValueError(f"Non-positive discount factor at T={T}")
        z[T] = -np.log(df_T) / T * 100

    kT = sorted(z); kz = [z[t] for t in kT]
    f = interp1d(kT, kz, kind='linear', fill_value='extrapolate')
    return np.array([float(f(m / 12.0)) for m in range(1, n_months + 1)])


def price_mbs(coupon, cpr_annual, zeros, gfee=GFEE, n_months=N_MONTHS):
    """Price a 30y pass-through at flat lifetime CPR. Returns price as % of par.
    coupon: investor coupon in % (e.g. 3.0). cpr_annual: decimal (e.g. 0.08).
    zeros: (n_months,) annualized zero rates in %."""
    note_m = (coupon + gfee) / 100.0 / 12.0
    inv_m  = coupon / 100.0 / 12.0
    cpr    = float(np.clip(cpr_annual, 0.0, 0.99))
    smm    = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)

    bal = 100.0
    pmt = bal * note_m / (1.0 - (1.0 + note_m) ** (-n_months))

    t_years = np.arange(1, n_months + 1) / 12.0
    disc    = np.exp(-zeros / 100.0 * t_years)

    pv = 0.0
    for t in range(n_months):
        if bal <= 1e-12:
            break
        sched_prin = min(pmt - bal * note_m, bal)
        sched_prin = max(sched_prin, 0.0)
        prepay     = (bal - sched_prin) * smm
        cf         = bal * inv_m + sched_prin + prepay
        pv        += cf * disc[t]
        bal        = bal - sched_prin - prepay
    return pv


def _bump_weights(tenor, mat_years=MAT_YEARS):
    """Partition-of-unity weights over the par nodes for a two-factor split.

    Single-node bumps do NOT span the curve: at a 25bp bump on the 5y and 10y
    nodes alone, KRD5+KRD10 recovers only ~58% of effective duration on a flat
    8% CPR, and just 34% at coupon 6.5 (fast prepay -> exposure concentrated at
    1-3y nodes that neither bump reaches). Hedging on those would leave most of
    the duration unhedged.

    Weights: w5(T) = 1 for T<=5, linear taper to 0 at T=10, 0 beyond.
             w10(T) = 1 - w5(T).
    Since w5 + w10 = 1 at every node, KRD5 + KRD10 = effective duration to
    first order, while still separating short-end from long-end exposure.

    Interpretation: KRD5 is the loading on a short-end factor proxied by the
    5y yield, KRD10 on a long-end factor proxied by the 10y. With only two
    hedge instruments this is the natural two-factor decomposition; it assumes
    curve moves project onto these two shapes.
    """
    T = np.asarray(mat_years, dtype=float)
    w5 = np.clip((10.0 - T) / 5.0, 0.0, 1.0)
    if tenor == '5yr':
        return w5
    if tenor == '10yr':
        return 1.0 - w5
    raise ValueError(f"unsupported tenor for spanning weights: {tenor}")


def key_rate_durations(coupon, cpr_annual, par_yields, bump_bp=25.0,
                       tenors=('5yr', '10yr'), gfee=GFEE, spanning=True):
    """Central-difference KRDs (years) from single-node par-yield bumps.
    Returns (price, {tenor: krd})."""
    base_z = bootstrap_zeros(par_yields)
    p0 = price_mbs(coupon, cpr_annual, base_z, gfee=gfee)

    h = bump_bp / 100.0          # bump in percentage points
    out = {}
    for ten in tenors:
        up, dn = dict(par_yields), dict(par_yields)
        if spanning:
            w = _bump_weights(ten)
            for lab, wi in zip(MAT_LABELS, w):
                up[lab] = float(par_yields[lab]) + h * wi
                dn[lab] = float(par_yields[lab]) - h * wi
        else:
            up[ten] = float(par_yields[ten]) + h
            dn[ten] = float(par_yields[ten]) - h
        p_up = price_mbs(coupon, cpr_annual, bootstrap_zeros(up), gfee=gfee)
        p_dn = price_mbs(coupon, cpr_annual, bootstrap_zeros(dn), gfee=gfee)
        # h is in pp; convert to decimal yield for duration in years
        out[ten] = (p_dn - p_up) / (2.0 * p0 * (h / 100.0))
    return p0, out
