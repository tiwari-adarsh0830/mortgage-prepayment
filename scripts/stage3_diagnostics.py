"""
stage3_diagnostics.py
---------------------
Audits the DER regression BEFORE making changes. Checks three suspects for the
small/insignificant lambda_y and the 4x magnitude gap vs DER:

  1. Excess-return scale  — are hedged excess returns a sensible magnitude
                            (MBS excess returns are typically ~10-40 bp/month)?
  2. Beta collinearity    — corr(beta_x, beta_y) per month. DER drops months
                            where this exceeds 0.90; we never did. If beta_y is
                            nearly collinear with beta_x, lambda_y is unidentified.
  3. Duration hedge       — flat D_mod=6.5 for ALL coupons. MBS duration varies
                            hugely by coupon (premium ~2yr, discount ~6yr). A flat
                            hedge leaves coupon-correlated residual rate exposure
                            that contaminates the cross-sectional lambda_x.

Reads existing outputs/data. Writes nothing. Pure diagnostic.
"""
import os, json
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")

COUPONS   = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
GFEE      = 0.75
WAC_PROXY = 3.5
D_MOD_AVG = 6.5

stage2 = {r["coupon"]: r["mean_cpr"]
          for r in json.load(open(os.path.join(OUT, "stage2_coupon_cpr.json")))}

# ── Reload the panel we already built (excess returns) ────────────────────────
panel = pd.read_csv(os.path.join(OUT, "stage3_excess_returns.csv"))
panel["Date"] = pd.to_datetime(panel["Date"])

print("="*70)
print("DIAGNOSTIC 1 — EXCESS RETURN SCALE")
print("="*70)
print("\nPer-coupon excess return summary (in basis points/month):")
print(f"{'Coupon':>8s} {'mean_bp':>10s} {'std_bp':>10s} {'min_bp':>10s} {'max_bp':>10s} {'n':>5s}")
for c in COUPONS:
    sub = panel[panel["coupon"] == c]["excess_return"].dropna()
    if len(sub) == 0:
        continue
    print(f"{c:>8.1f} {sub.mean()*1e4:>10.2f} {sub.std()*1e4:>10.2f} "
          f"{sub.min()*1e4:>10.2f} {sub.max()*1e4:>10.2f} {len(sub):>5d}")

overall = panel["excess_return"].dropna()
print(f"\nOverall excess return: mean={overall.mean()*1e4:.2f} bp/mo, "
      f"std={overall.std()*1e4:.2f} bp/mo")
print("Sanity check: MBS hedged excess returns in DER avg ~15-40 bp/mo.")
print("If ours are ~10x smaller or larger, return construction has a scale problem.")

# Also check raw TBA total returns and UST returns separately
print(f"\nRaw TBA total return:  mean={panel['tba_total_return'].mean()*1e4:.2f} bp/mo, "
      f"std={panel['tba_total_return'].std()*1e4:.2f} bp/mo")
print(f"UST total return:      mean={panel['ust_total_return'].mean()*1e4:.2f} bp/mo, "
      f"std={panel['ust_total_return'].std()*1e4:.2f} bp/mo")

# ── DIAGNOSTIC 2 — beta collinearity per month ────────────────────────────────
print("\n" + "="*70)
print("DIAGNOSTIC 2 — BETA COLLINEARITY (beta_x vs beta_y)")
print("="*70)

def compute_betas(coupon, r_t):
    c = coupon; m_i = coupon + GFEE
    phi = stage2.get(c, 0.01)
    r_dec = r_t/100.0; c_dec = c/100.0
    denom = (r_dec + phi)*(phi + c_dec)
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    base = (r_dec - c_dec)/denom
    moneyness = max(0.0, (m_i - r_t)/100.0)
    return base, base*moneyness

def get_pmms_for_date(dt):
    sub = panel[panel["Date"] == dt]
    return float(sub["pmms"].iloc[0]) if len(sub) else np.nan

dates = sorted(panel["Date"].unique())
corr_rows = []
for dt in dates:
    month = panel[panel["Date"] == dt].copy()
    if len(month) < 4:
        continue
    r_t = float(month["pmms"].iloc[0])
    month["beta_x"] = month["coupon"].apply(lambda c: compute_betas(c, r_t)[0])
    month["beta_y"] = month["coupon"].apply(lambda c: compute_betas(c, r_t)[1])
    # only meaningful if there's variation in beta_y (some premium coupons)
    if month["beta_y"].abs().sum() < 1e-12:
        corr = np.nan
    elif month["beta_y"].std() < 1e-12 or month["beta_x"].std() < 1e-12:
        corr = np.nan
    else:
        corr = np.corrcoef(month["beta_x"], month["beta_y"])[0, 1]
    mtype = month["market_type"].iloc[0]
    corr_rows.append(dict(date=dt, market_type=mtype, corr=corr,
                          n_premium=int((month["coupon"] + GFEE > r_t).sum())))

corr_df = pd.DataFrame(corr_rows)
pm_corr = corr_df[corr_df["market_type"] == "PM"]["corr"].dropna()
dm_corr = corr_df[corr_df["market_type"] == "DM"]["corr"].dropna()

print(f"\nPM months: corr(beta_x, beta_y)  mean={pm_corr.mean():.3f}, "
      f"median={pm_corr.median():.3f}, max={pm_corr.max():.3f}")
print(f"  months with |corr| > 0.90: {(pm_corr.abs() > 0.90).sum()} / {len(pm_corr)}")
print(f"DM months: corr(beta_x, beta_y)  "
      f"{'mean=%.3f' % dm_corr.mean() if len(dm_corr) else 'no beta_y variation (all discount)'}")
print("\nIf PM corr is near +/-1, beta_y is collinear with beta_x -> lambda_y unidentified.")
print("DER drops months where this exceeds 0.90. We currently keep all of them.")

# ── DIAGNOSTIC 3 — model-implied coupon-specific durations ────────────────────
print("\n" + "="*70)
print("DIAGNOSTIC 3 — DURATION HEDGE (flat 6.5 vs coupon-specific)")
print("="*70)
print("\nModel-implied effective duration  D_i = 1/(r+phi_i)  (from P=(c+phi)/(r+phi)):")
print("(absolute level runs long in continuous-time model; the CROSS-SECTIONAL")
print(" pattern — premium short, discount long — is what matters for the hedge)")
print(f"\n{'Coupon':>8s} {'phi':>8s} {'D_i (r=3%)':>12s} {'D_i (r=6%)':>12s}")
for c in COUPONS:
    phi = stage2.get(c, 0.01)
    d3 = 1.0/(0.03 + phi)
    d6 = 1.0/(0.06 + phi)
    print(f"{c:>8.1f} {phi:>8.4f} {d3:>12.2f} {d6:>12.2f}")
print(f"\nFlat hedge uses {D_MOD_AVG} for all. Spread in model durations above shows")
print("how much coupon-correlated residual rate exposure the flat hedge leaves.")
print("\nNOTE: model duration LEVELS are too long (continuous perpetuity). For a real")
print("hedge improvement, use Bloomberg OAD per coupon (pullable at Bobst) or rescale")
print("the model durations to a sensible range (e.g. 2-6yr) preserving the ordering.")
