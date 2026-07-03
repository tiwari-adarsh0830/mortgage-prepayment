
2026-07-03 | 4h | Rolling forecast-leg factor shocks + FM restriction bugfix
  - Fixed calibration fallback: cutoff_2020/2021 had no own Platt file, silently
    used OAS Platt (b=-4.840) instead of cohort-CPR Platt for forecast leg.
    Forced cohort-CPR onto all four rolling cutoffs.
  - Found and fixed FM sample-leakage bug in stage3_der_factor_shocks.py:
    fama_macbeth() was using the unrestricted 77-month returns panel instead
    of the factor-coverage window. Affected BOTH the rolling result and the
    original full-sample result already sent to advisor on Jun 27.
  - Corrected full-sample: lambda_x=0.057 t=2.35 n=72 (sent value was t=2.52 n=77)
  - Rolling (theta_t-, genuine OOS): lambda_x=0.149 t=3.04 n=48, corr(bx,by)=0.390
    -- survives OOS test, stronger than full-sample, decorrelation holds
  - AR(1) robustness (DER's own test, Sec IV.B.1): rho_x=0.91 rho_y=0.57 on
    full-sample factors. Unlike DER ("nearly identical"), our result is NOT
    robust: lambda_x t drops 2.35->1.08. Real finding, not a bug -- full-sample
    forecast leg carries more persistent/forecastable structure than a dealer
    survey does, likely because it's one fixed hazard model rather than a
    panel of independently-updating dealers.
  - Attempted per-cutoff-model debias (level-additive, log-space, log-space
    ex-cutoff_2020): all three broke the cross-section (corr(bx,by) moved
    0.39 -> 0.51 -> -0.58). Diagnosed root cause: rolling shock is 53%
    time-driven / 8% coupon-driven, with trend direction flipping sign across
    cutoffs (cutoff_2021 decays -1.4 log-pts, cutoff_2023 rises +0.35). A
    scalar bias-per-cutoff cannot represent this. Correctly abandoned rather
    than shipped broken -- not a coding error, a specification mismatch with
    DER's stationary-AR(1) precondition, which doesn't hold on 12-month
    rolling segments.
  - Net: Task A complete and reportable. Task B open on the rolling series;
    AR(1) result on full-sample is itself a reportable finding in Task B's
    place.
