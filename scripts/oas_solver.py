"""
OAS Spread Solver
=================
Given market prices (from Bloomberg/ICE), solves for OAS spread s such that:
  E_Q[ sum_t CF_t / prod(1 + (r_t + s)/100/12) ] = market_price

Requires: oas_cashflows.npy saved by oas_engine.py (see note below)
Input:    market prices as % of par (per loan or pool level)
Output:   OAS in basis points per loan, pool-level OAS

NOTE: oas_engine.py must be updated to save cashflow matrix.
Run update_oas_engine.py first if oas_cashflows.npy doesn't exist.
"""

import numpy as np
from scipy.optimize import brentq
import os
import json

BASE    = "/scratch/at7095/mortgage_prepayment"
OUTPUTS = os.path.join(BASE, "outputs")

SPREAD_LB = -500.0   # bp lower bound for OAS search
SPREAD_UB = 2000.0   # bp upper bound for OAS search
SPREAD_TOL = 0.1     # bp tolerance for convergence


def price_at_spread(cashflows_loan, treasury_paths, upb, spread_bps):
    """
    Compute model price for one loan at given OAS spread.
    cashflows_loan: (N_paths, T) monthly cashflows in dollars
    treasury_paths: (N_paths, T) risk-free rates in %
    upb:            float, original UPB
    spread_bps:     float, OAS spread in basis points
    Returns: float, price as % of par
    """
    spread_pct = spread_bps / 10000.0
    disc_rates = (treasury_paths / 100.0 + spread_pct) / 12.0  # (N_paths, T)
    cum_disc   = np.cumprod(1 + disc_rates, axis=1)              # (N_paths, T)
    pv_per_path = (cashflows_loan / cum_disc).sum(axis=1)        # (N_paths,)
    return pv_per_path.mean() / upb * 100.0


def solve_oas_loan(cashflows_loan, treasury_paths, upb, market_price_pct):
    """
    Solve for OAS of a single loan.
    Returns: (oas_bps, converged) tuple
    """
    # Check bounds bracket the solution
    p_lb = price_at_spread(cashflows_loan, treasury_paths, upb, SPREAD_LB)
    p_ub = price_at_spread(cashflows_loan, treasury_paths, upb, SPREAD_UB)

    if market_price_pct > p_lb:
        # Market price too high — OAS below lower bound (very cheap financing)
        return SPREAD_LB, False
    if market_price_pct < p_ub:
        # Market price too low — OAS above upper bound (distressed)
        return SPREAD_UB, False

    try:
        oas = brentq(
            lambda s: price_at_spread(cashflows_loan, treasury_paths, upb, s) - market_price_pct,
            SPREAD_LB, SPREAD_UB, xtol=SPREAD_TOL
        )
        return oas, True
    except ValueError:
        return np.nan, False


def solve_pool_oas(cashflows, treasury_paths, orig_upbs, pool_market_price_pct):
    """
    Solve for pool-level OAS.
    cashflows:    (N_loans, N_paths, T)
    treasury_paths: (N_paths, T)
    orig_upbs:    (N_loans,)
    pool_market_price_pct: float, pool price as % of par (WAC-weighted or simple average)
    Returns: pool OAS in bps
    """
    N_loans = len(orig_upbs)

    def pool_price(s):
        prices = np.array([
            price_at_spread(cashflows[i], treasury_paths, orig_upbs[i], s)
            for i in range(N_loans)
        ])
        # UPB-weighted average (more accurate for pool pricing)
        weights = orig_upbs / orig_upbs.sum()
        return (prices * weights).sum()

    p_lb = pool_price(SPREAD_LB)
    p_ub = pool_price(SPREAD_UB)

    if pool_market_price_pct > p_lb or pool_market_price_pct < p_ub:
        print(f"  WARNING: pool market price {pool_market_price_pct:.2f}% outside "
              f"model range [{p_ub:.2f}%, {p_lb:.2f}%]")
        return np.nan

    pool_oas = brentq(
        lambda s: pool_price(s) - pool_market_price_pct,
        SPREAD_LB, SPREAD_UB, xtol=SPREAD_TOL
    )
    return pool_oas


def main(pool_market_price=None, loan_market_prices=None):
    """
    pool_market_price:  float, pool price as % of par from Bloomberg/ICE
                        If None, uses model price as placeholder (OAS=0 expected)
    loan_market_prices: (N_loans,) array, per-loan prices if available
                        If None, uses pool_market_price for all loans
    """
    print("=" * 60)
    print("OAS Spread Solver")
    print("=" * 60)

    # Load cashflows and treasury paths
    cashflow_path = os.path.join(OUTPUTS, 'oas_cashflows.npy')
    if not os.path.exists(cashflow_path):
        print(f"ERROR: {cashflow_path} not found.")
        print("Please run oas_engine.py with save_cashflows=True first.")
        return

    print("Loading cashflows and rate paths...")
    cashflows      = np.load(cashflow_path)                           # (N_loans, N_paths, T)
    treasury_paths = np.load(os.path.join(OUTPUTS, 'treasury_rate_paths.npy'))  # (N_paths, T)
    orig_upbs      = np.load(os.path.join(OUTPUTS, 'oas_orig_upbs.npy'))        # (N_loans,)
    model_prices   = np.load(os.path.join(OUTPUTS, 'oas_loan_prices.npy')).mean(axis=1)  # (N_loans,)

    N_loans, N_paths, T = cashflows.shape
    print(f"Loans: {N_loans}, Paths: {N_paths}, Months: {T}")
    print(f"Model prices: mean={model_prices.mean():.2f}%, median={np.median(model_prices):.2f}%")

    # If no market price provided, use model price (sanity check: OAS should be ~0)
    if pool_market_price is None:
        print("\nNo market price provided — using model price as placeholder.")
        print("Expected OAS ≈ 0bp (sanity check).")
        pool_market_price = float(model_prices.mean())

    if loan_market_prices is None:
        loan_market_prices = np.full(N_loans, pool_market_price)

    # Solve per-loan OAS
    print(f"\nSolving OAS for {N_loans} loans (pool market price: {pool_market_price:.2f}%)...")
    loan_oas    = np.zeros(N_loans)
    converged   = np.zeros(N_loans, dtype=bool)

    for i in range(N_loans):
        if i % 100 == 0:
            print(f"  Loan {i}/{N_loans}...", flush=True)
        oas, conv = solve_oas_loan(
            cashflows[i], treasury_paths[:N_paths, :T],
            orig_upbs[i], loan_market_prices[i]
        )
        loan_oas[i]  = oas
        converged[i] = conv

    n_conv = converged.sum()
    print(f"\nConverged: {n_conv}/{N_loans} loans ({n_conv/N_loans*100:.1f}%)")
    print(f"\nPer-loan OAS (bp):")
    print(f"  Mean:   {loan_oas[converged].mean():.2f}")
    print(f"  Median: {np.median(loan_oas[converged]):.2f}")
    print(f"  Std:    {loan_oas[converged].std():.2f}")
    print(f"  Min:    {loan_oas[converged].min():.2f}")
    print(f"  Max:    {loan_oas[converged].max():.2f}")

    # Pool-level OAS
    print(f"\nSolving pool-level OAS...")
    pool_oas = solve_pool_oas(
        cashflows, treasury_paths[:N_paths, :T], orig_upbs, pool_market_price
    )
    print(f"Pool OAS: {pool_oas:.2f}bp")

    # Save results
    results = {
        'pool_market_price_pct': pool_market_price,
        'pool_oas_bps':          float(pool_oas) if not np.isnan(pool_oas) else None,
        'loan_oas_mean_bps':     float(loan_oas[converged].mean()),
        'loan_oas_median_bps':   float(np.median(loan_oas[converged])),
        'loan_oas_std_bps':      float(loan_oas[converged].std()),
        'n_loans':               N_loans,
        'n_converged':           int(n_conv),
        'note': 'OAS in basis points. Positive = MBS cheaper than model (higher yield).'
    }
    np.save(os.path.join(OUTPUTS, 'oas_spreads.npy'), loan_oas)
    with open(os.path.join(OUTPUTS, 'oas_spread_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: oas_spreads.npy, oas_spread_results.json")


if __name__ == "__main__":
    import sys
    # Usage: python oas_solver.py [pool_market_price]
    # Example: python oas_solver.py 96.5
    if len(sys.argv) > 1:
        market_price = float(sys.argv[1])
        print(f"Using market price: {market_price:.2f}%")
        main(pool_market_price=market_price)
    else:
        main()  # sanity check mode
