"""
Risk-Neutral Rate Path Generation for OAS
==========================================
Steps:
1. Bootstrap zero-coupon Treasury curve from today's par yields
2. Compute implied forward rates from zero curve
3. Apply drift correction to DDPM paths: shift each month's mean to match forward rate
4. Validate: average discounted ZCB prices match today's Treasury curve
5. Compute PMMS paths = Treasury paths + historical spread (for refi incentive)
6. Save corrected paths
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import json

BASE    = "/scratch/at7095/mortgage_prepayment"
OUTPUTS = os.path.join(BASE, "outputs")
DATA    = os.path.join(BASE, "data")

MAX_SEQ = 33


# ── Step 1: Bootstrap zero-coupon curve ──────────────────────────────────────
def bootstrap_zero_curve(treasury_csv):
    """
    Bootstrap continuous zero rates from Treasury par yields.
    Returns: monthly_zeros (33,) array of annualized zero rates in %
    """
    df = pd.read_csv(treasury_csv, index_col=0, parse_dates=True)
    latest = df.iloc[-1]
    print(f"Using Treasury curve dated: {df.index[-1].date()}")

    mat_labels = ['1mo', '3mo', '6mo', '1yr', '2yr', '3yr', '5yr', '7yr', '10yr', '20yr', '30yr']
    mat_years  = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    par_yields = [latest[m] for m in mat_labels]  # in %

    print("Par yields:")
    for m, y in zip(mat_labels, par_yields):
        print(f"  {m:>4}: {y:.4f}%")

    zero_rates = {}

    # Short end (T <= 1yr): par yield = zero rate
    for T, y in zip(mat_years, par_yields):
        if T <= 1.0:
            zero_rates[T] = y

    def get_zero(T):
        known_T = sorted(zero_rates.keys())
        known_z = [zero_rates[t] for t in known_T]
        if T <= known_T[0]:  return known_z[0]
        if T >= known_T[-1]: return known_z[-1]
        return float(interp1d(known_T, known_z, kind='linear')(T))

    def df(T, z_pct):
        return np.exp(-z_pct / 100 * T)

    # Long end: bootstrap semi-annual coupon bonds
    for T, par_y in zip(mat_years, par_yields):
        if T <= 1.0:
            continue
        c = par_y / 100
        coupon_times = np.arange(0.5, T, 0.5)
        pv_coupons = sum((c/2) * 100 * df(t, get_zero(t)) for t in coupon_times)
        final_cf = (c/2 * 100 + 100)
        df_T = (100 - pv_coupons) / final_cf
        assert df_T > 0, f"Negative discount factor at T={T}"
        zero_rates[T] = -np.log(df_T) / T * 100

    # Interpolate to monthly grid
    known_T = sorted(zero_rates.keys())
    known_z = [zero_rates[t] for t in known_T]
    f_interp = interp1d(known_T, known_z, kind='linear', fill_value='extrapolate')
    monthly_zeros = np.array([float(f_interp(m/12)) for m in range(1, MAX_SEQ+1)])

    print(f"\nBootstrapped zero rates (sample):")
    for m in [1, 6, 12, 24, 33]:
        print(f"  Month {m:2d}: {monthly_zeros[m-1]:.4f}%  "
              f"(ZCB={np.exp(-monthly_zeros[m-1]/100 * m/12)*100:.3f})")

    return monthly_zeros


# ── Step 2: Compute forward rates ─────────────────────────────────────────────
def compute_forward_rates(monthly_zeros):
    """
    Compute monthly forward rates from zero curve.
    f(m) = [z_m*(m/12) - z_{m-1}*((m-1)/12)] * 12  [annualized %]
    """
    forward_rates = np.zeros(MAX_SEQ)
    forward_rates[0] = monthly_zeros[0]
    for m in range(1, MAX_SEQ):
        z_m  = monthly_zeros[m]   * (m+1)/12
        z_m1 = monthly_zeros[m-1] *  m/12
        forward_rates[m] = (z_m - z_m1) * 12
    return forward_rates


# ── Step 3: Drift correction ──────────────────────────────────────────────────
def apply_drift_correction(paths, forward_rates, floor_rate=0.01):
    """
    Shift each month's rate mean to match the implied forward rate.
    Preserves volatility structure exactly.
    paths:         (N, T) DDPM rate paths in %
    forward_rates: (T,)   target forward rates in %
    floor_rate:    minimum rate after correction (% annualized)
    Returns: corrected_paths (N, T), shifts (T,)
    """
    N, T = paths.shape
    current_means = paths.mean(axis=0)           # (T,)
    shifts        = forward_rates[:T] - current_means  # (T,)
    corrected     = paths + shifts[np.newaxis, :]
    corrected     = np.maximum(corrected, floor_rate)
    return corrected, shifts


# ── Step 4: Validate ZCB repricing ───────────────────────────────────────────
def validate_zcb_repricing(paths, monthly_zeros, tol_bp=1.0):
    """
    Check E[exp(-sum(r_t/100/12, t=1..T))] ≈ exp(-z_T/100 * T/12)
    Prints errors in basis points.
    """
    print("\nZCB repricing validation:")
    print(f"{'Month':>5} {'Model_DF':>10} {'Target_DF':>10} {'Error_bp':>10} {'Status':>8}")
    max_err = 0.0
    for m in [1, 3, 6, 12, 24, 33]:
        model_df  = np.exp(-(paths[:, :m] / 100.0 / 12.0).sum(axis=1)).mean()
        target_df = np.exp(-monthly_zeros[m-1] / 100.0 * m / 12.0)
        err_bp    = abs(model_df - target_df) * 10000
        max_err   = max(max_err, err_bp)
        status    = "OK" if err_bp < tol_bp else "FAIL"
        print(f"{m:>5} {model_df:>10.6f} {target_df:>10.6f} {err_bp:>10.3f} {status:>8}")
    print(f"Max error: {max_err:.3f}bp  ({'PASS' if max_err < tol_bp else 'FAIL'})")
    return max_err < tol_bp


# ── Step 5: Compute PMMS paths from Treasury paths ───────────────────────────
def compute_pmms_paths(treasury_paths, pmms_csv, treasury_csv):
    """
    PMMS = Treasury short rate + historical spread.
    Computes mean historical (PMMS - 10yr Treasury) spread and adds to paths.
    Returns: pmms_paths (N, T) in %, spread (float) in %
    """
    pmms_df     = pd.read_csv(pmms_csv)
    treasury_df = pd.read_csv(treasury_csv, index_col=0, parse_dates=True)

    # Parse PMMS dates from year/month columns
    pmms_df['DATE'] = pd.to_datetime(
        pmms_df['year'].astype(str) + '-' + pmms_df['month'].astype(str).str.zfill(2))
    pmms_df = pmms_df.set_index('DATE')[['rate_30yr']]

    merged = pmms_df.join(treasury_df[['10yr']], how='inner').dropna()
    spread = (merged['rate_30yr'] - merged['10yr']).mean()

    print(f"\nHistorical PMMS - 10yr Treasury spread: {spread:.4f}%")
    print(f"  (Based on {len(merged)} months of overlapping data)")
    pmms_paths = treasury_paths + spread
    return pmms_paths, spread


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Risk-Neutral Rate Path Generation")
    print("=" * 60)

    # Load DDPM paths
    ddpm_paths = np.load(os.path.join(OUTPUTS, 'ddpm_rate_paths.npy'))
    ddpm_paths = ddpm_paths[:, :MAX_SEQ]
    print(f"\nDDPM paths: {ddpm_paths.shape}, mean={ddpm_paths.mean():.3f}%")

    treasury_csv = os.path.join(DATA, 'treasury_yields.csv')
    pmms_csv     = os.path.join(DATA, 'pmms_monthly.csv')

    # Step 1: Bootstrap zero curve
    print("\n--- Step 1: Bootstrap Treasury Zero Curve ---")
    monthly_zeros = bootstrap_zero_curve(treasury_csv)

    # Step 2: Forward rates
    print("\n--- Step 2: Forward Rates ---")
    forward_rates = compute_forward_rates(monthly_zeros)
    print(f"Forward rates (months 1,6,12,24,33): "
          f"{forward_rates[[0,5,11,23,32]].round(4)}")

    # Step 3: Drift correction
    print("\n--- Step 3: Drift Correction ---")
    treasury_paths, shifts = apply_drift_correction(ddpm_paths, forward_rates)
    print(f"Shifts (months 1,6,12,24,33): {shifts[[0,5,11,23,32]].round(4)}")
    print(f"Corrected paths: mean={treasury_paths.mean():.3f}%, "
          f"std={treasury_paths.std():.3f}%, min={treasury_paths.min():.3f}%")

    # Step 4: Validate
    print("\n--- Step 4: ZCB Validation ---")
    passed = validate_zcb_repricing(treasury_paths, monthly_zeros)

    # Step 5: PMMS paths
    print("\n--- Step 5: PMMS Paths (for refi incentive) ---")
    pmms_paths, spread = compute_pmms_paths(treasury_paths, pmms_csv, treasury_csv)
    print(f"PMMS paths: mean={pmms_paths.mean():.3f}%, std={pmms_paths.std():.3f}%")

    # Save
    np.save(os.path.join(OUTPUTS, 'treasury_rate_paths.npy'), treasury_paths)
    np.save(os.path.join(OUTPUTS, 'pmms_rate_paths_rn.npy'),  pmms_paths)
    np.save(os.path.join(OUTPUTS, 'monthly_zero_rates.npy'),  monthly_zeros)

    results = {
        'monthly_zeros':             monthly_zeros.tolist(),
        'forward_rates':             forward_rates.tolist(),
        'pmms_treasury_spread_pct':  float(spread),
        'zcb_validation_passed':     bool(passed),
        'treasury_paths_mean':       float(treasury_paths.mean()),
        'treasury_paths_std':        float(treasury_paths.std()),
        'note': ('Treasury paths for discounting. '
                 'PMMS paths for refi incentive. '
                 'Drift corrected to reprice today\'s Treasury curve.')
    }
    with open(os.path.join(OUTPUTS, 'risk_neutral_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved:")
    print(f"  treasury_rate_paths.npy  — risk-free paths for discounting")
    print(f"  pmms_rate_paths_rn.npy   — PMMS paths for refi incentive")
    print(f"  monthly_zero_rates.npy   — bootstrapped zero curve")
    print(f"  risk_neutral_results.json")
    print(f"\nNext: update oas_engine.py to use these paths + add terminal value.")


if __name__ == "__main__":
    main()
