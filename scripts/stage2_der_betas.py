"""
Stage 2 (DER-correct): Compute beta_x and beta_y per coupon bucket
following Diep, Eisfeldt, Richardson (2021) Equations (5) and (6).

beta_x^i = (r - c^i) / [(r + phi^i)(phi^i + c^i)]   * d(phi^i)/dx
beta_y^i = (r - c^i) / [(r + phi^i)(phi^i + c^i)]   * d(phi^i)/dy

where:
  r     = par coupon rate = current PMMS (opportunity cost for MBS investor)
  c^i   = MBS coupon for security i
  phi^i = mean CPR for coupon bucket i (from hazard model, stage2_coupon_cpr.json)
  d(phi^i)/dx = 1 for all securities (level shock affects all equally)
  d(phi^i)/dy = max(0, m^i - m_t) = refi incentive = (note_rate - PMMS) for premium,
                0 for discount securities

Note: sigma_x and sigma_y scaling factors are absorbed into lambda in the
cross-sectional regression, so we set them = 1 here (standard normalization).

Outputs:
  outputs/der_betas.json  — beta_x, beta_y per coupon + classification
  outputs/der_betas.csv   — same in CSV
"""

import json, csv, os
import numpy as np

BASE  = "/scratch/at7095/mortgage_prepayment"
OUT   = os.path.join(BASE, "outputs")
DATA  = os.path.join(BASE, "data")

# ── Load stage2 CPR results ───────────────────────────────────────────────────
stage2 = json.load(open(os.path.join(OUT, "stage2_coupon_cpr.json")))

# ── Load current PMMS (par coupon rate = r) ───────────────────────────────────
# Use most recent PMMS from pmms_monthly.csv as the par rate
import pandas as pd
pmms_df = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))
# reporting_period format: MYYYYMM e.g. 41971=Apr 1971, 122026=Dec 2026
pmms_df = pmms_df.sort_values("reporting_period")
r_pmms = float(pmms_df.iloc[-1]["rate_30yr"])   # most recent PMMS in %
print(f"Par coupon rate r (current PMMS) = {r_pmms:.4f}%")

# Also load the PMMS time series for historical par rates (used in Stage 3)

# ── Compute DER betas per coupon ──────────────────────────────────────────────
# sigma_x = sigma_y = 1 (absorbed into lambda in regression)
sigma_x = 1.0
sigma_y = 1.0

rows = []
print(f"\n{'coupon':>8} {'c-r':>8} {'phi':>8} {'beta_x':>10} {'beta_y':>10} {'type':>10}")
print("-" * 60)

for rec in stage2:
    c     = rec["coupon"]          # MBS coupon (%)
    m_i   = rec["note_rate"]       # WAC / note rate (%)
    phi_i = rec["mean_cpr"]        # mean CPR from hazard model (annual, decimal)

    # Convert CPR to monthly rate for consistency with DER continuous-time formula
    # DER uses continuous-time phi, so annualized CPR is appropriate here
    # (the formula is scale-invariant with sigma absorbed into lambda)

    # Investor moneyness: c^i - r (determines sign of betas)
    moneyness_investor = c - r_pmms   # negative = discount, positive = premium

    # Borrower moneyness: m^i - m_t (used for beta_y)
    moneyness_borrower = max(0.0, m_i - r_pmms)   # = 0 for discount securities

    # DER Equation (5): beta_x
    # beta_x = sigma_x * (r - c^i) / [(r + phi^i)(phi^i + c^i)] * d(phi)/dx
    # d(phi)/dx = 1 for all securities
    denom = (r_pmms/100 + phi_i) * (phi_i + c/100)   # use decimal units
    if abs(denom) < 1e-10:
        beta_x = 0.0
    else:
        beta_x = sigma_x * (r_pmms/100 - c/100) / denom * 1.0

    # DER Equation (6): beta_y
    # beta_y = sigma_y * (r - c^i) / [(r + phi^i)(phi^i + c^i)] * d(phi)/dy
    # d(phi)/dy = max(0, m^i - m_t) for premium, 0 for discount
    dphi_dy = moneyness_borrower / 100.0   # convert % to decimal
    if abs(denom) < 1e-10:
        beta_y = 0.0
    else:
        beta_y = sigma_y * (r_pmms/100 - c/100) / denom * dphi_dy

    security_type = "premium" if moneyness_investor > 0 else "discount"

    rows.append(dict(
        coupon            = c,
        note_rate         = m_i,
        mean_cpr          = phi_i,
        par_rate_pmms     = round(r_pmms, 4),
        moneyness_investor= round(moneyness_investor, 4),   # c - r
        moneyness_borrower= round(moneyness_borrower, 4),   # m - m_t
        beta_x            = round(beta_x, 6),
        beta_y            = round(beta_y, 6),
        security_type     = security_type,
    ))

    print(f"{c:>8.1f} {moneyness_investor:>+8.2f} {phi_i:>8.4f} "
          f"{beta_x:>10.6f} {beta_y:>10.6f} {security_type:>10}")

# ── Sanity checks (DER Lemmas 1 and 2) ───────────────────────────────────────
print("\n=== DER Lemma checks ===")
discounts = [r for r in rows if r["security_type"] == "discount"]
premiums  = [r for r in rows if r["security_type"] == "premium"]

if discounts:
    bx_disc = [r["beta_x"] for r in discounts]
    print(f"Discount securities: beta_x range = [{min(bx_disc):.6f}, {max(bx_disc):.6f}]")
    print(f"  Lemma 1: beta_x > 0 for discounts? {all(b > 0 for b in bx_disc)}")
    print(f"  Lemma 2(ii): beta_x decreasing in coupon (larger discount = larger beta_x)?")
    for r in discounts:
        print(f"    coupon={r['coupon']:.1f}  beta_x={r['beta_x']:.6f}")

if premiums:
    bx_prem = [r["beta_x"] for r in premiums]
    by_prem = [r["beta_y"] for r in premiums]
    print(f"\nPremium securities: beta_x range = [{min(bx_prem):.6f}, {max(bx_prem):.6f}]")
    print(f"  Lemma 1: beta_x < 0 for premiums? {all(b < 0 for b in bx_prem)}")
    print(f"  Lemma 1: beta_y < 0 for premiums? {all(b < 0 for b in by_prem)}")

# ── Save outputs ──────────────────────────────────────────────────────────────
json.dump(rows, open(os.path.join(OUT, "der_betas.json"), "w"), indent=2)
with open(os.path.join(OUT, "der_betas.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"\nSaved: outputs/der_betas.{{json,csv}}")
print(f"Coupons processed: {len(rows)}")
print(f"Discount: {len(discounts)}  Premium: {len(premiums)}")
