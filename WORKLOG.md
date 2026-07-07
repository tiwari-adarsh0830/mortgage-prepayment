
2026-07-03 | 4h | Rolling forecast-leg factor shocks + FM restriction bugfix
  - Fixed calibration fallback: cutoff_2020/2021 had no own Platt file, silently
    used OAS Platt (b=-4.840) instead of cohort-CPR Platt for forecast leg.
    Forced cohort-CPR onto all four rolling cutoffs.
  - Found and fixed FM sample-leakage bug in stage3_der_factor_shocks.py:
    fama_macbeth() was using the unrestricted 77-month returns panel instead
    of the factor-coverage window. Affected BOTH the rolling result and the
    original full-sample result already sent on Jun 27.
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

2026-07-04 | 3h | UPB balance-weighting + pipeline integration
  - Built realized_cpr_v6_upb.py: extends v6's Pass 0 to track each loan's
    top-2 (YYYYMM, UPB) rows, since the payoff month's own UPB=0 by
    construction (that's the prepay-detection signal) -- correct weight for
    the payoff month is the PRIOR month's balance, not the payoff row itself.
  - First attempt (nohup on login node) died silently ~file 11/41, likely
    session disconnect or OOM -- no per-file checkpoint existed yet to
    diagnose or recover from.
  - Rewrote with per-5-file Pass 0 checkpointing + resume support before
    resubmitting via SLURM (--time=20:00:00 --mem=96G). First SLURM attempt
    (8h) hit TIMEOUT mid-Pass-1 (Pass 0 had completed and checkpointed).
    Resubmission completed Pass 1 in 8:27:27 using the cached Pass 0 result.
  - Verified: 2/13.77M prepaid loans excluded (no prior-month row) --
    negligible. cpr_count column matches existing v6.csv's cpr to within
    1.1e-4 max diff across 1,748 coupon-months -- confirms the new script's
    counting logic is consistent with the already-trusted v6 pipeline.
  - One exact-zero-CPR month (2013-07, coupon 2.5) traced to a panel-boundary
    artifact (loans already near payoff when the 2013 extension window
    starts) -- consistent with the same class of early-window artifact
    already documented for the 2018 boundary.
  - Parameterized stage3_der_factor_shocks.py's load_realized() to accept
    --realized/--realized-col, defaults preserve original behavior exactly.
  - Ran both forecast legs (full-sample, rolling) against UPB-weighted
    realized CPR. lambda_x positive and significant (p<0.05) in all four
    leg x weighting combinations; UPB raises the coefficient ~20-25% on
    both legs; corr(bx,by) stays well under DER's 0.90 threshold throughout.
2026-07-05 | 3h | UPB-weighting made pipeline default + AR(1) persistence test on rolling series
    - Made cpr_upb the default --realized-col and its file the default --realized in stage3_der_factor_shocks.py (was previously opt-in flag only)
    - Extended AR(1)/persistence residualization test to rolling factor-shock series (new script: stage3_ar1_test.py)
    - Fixed OOS-only filtering bug in rolling AR(1) test (was including in-sample 2020 production-model months, n=60 -> corrected n=48)
    - Verified two-factor mode stays intact after AR(1)-residualizing in all four cases (no silent single-factor fallback)
    - Investigated and diagnosed rho(b_x,b_y) sign flip in rolling AR(1) case (0.39->-0.53) - traced to broad R^2 decline across coupons at n=47, not a single outlier
    - Sent follow-up email 7/5 with results
2026-07-06 | 2h | AR(1)-residualized rolling FM test, ex-cutoff_2020 robustness (advisor request)
    - Patched stage3_ar1_test.py: added exclude_cutoffs filter param + lambda_y
      reporting (additive only; verified via diff against pre-patch backup)
    - Initial run failed: default realized-CPR file lacks cpr_upb column.
      Correct source is realized_cpr_by_coupon_v6_upb.csv (separate file,
      previously uncommitted -- added this session)
    - Rolling OOS-only, ex-cutoff_2020 (n=36, UPB-weighted):
      RAW lambda_x=0.0486 t=2.586 | AR(1)-resid lambda_x=0.0318 t=2.310 (n=35)
    - lambda_y unidentified in this window: rho(b_x,b_y)=0.986 trips the
      existing single-factor fallback (rho_max=0.90) in fama_macbeth().
      Confirmed via standalone diagnostic on empirical_betas() output --
      not a bug, same collinearity mechanism as DER's own result. Ruled out
      all-discount-market explanation (31/36 months have a premium coupon)
    - Robustness check on RAW result: 25/36 months positive sign,
      leave-one-out t-stat range 2.32-3.14 (no single month drives result)
    - Sent results + lambda_y caveat to advisor
2026-07-06 | 2h | AR(1)-residualized rolling FM test, ex-cutoff_2020 robustness (advisor request)
    - Patched stage3_ar1_test.py: added exclude_cutoffs filter param + lambda_y
      reporting (additive only; verified via diff against pre-patch backup)
    - Initial run failed: default realized-CPR file lacks cpr_upb column.
      Correct source is realized_cpr_by_coupon_v6_upb.csv (separate file,
      previously uncommitted -- added this session)
    - Rolling OOS-only, ex-cutoff_2020 (n=36, UPB-weighted):
      RAW lambda_x=0.0486 t=2.586 | AR(1)-resid lambda_x=0.0318 t=2.310 (n=35)
    - lambda_y unidentified in this window: rho(b_x,b_y)=0.986 trips the
      existing single-factor fallback (rho_max=0.90) in fama_macbeth().
      Confirmed via standalone diagnostic on empirical_betas() output --
      not a bug, same collinearity mechanism as DER's own result. Ruled out
      all-discount-market explanation (31/36 months have a premium coupon)
    - Robustness check on RAW result: 25/36 months positive sign,
      leave-one-out t-stat range 2.32-3.14 (no single month drives result)
    - Sent results + lambda_y caveat to advisor
