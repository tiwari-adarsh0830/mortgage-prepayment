"""
stage3_robustness_orthog.py
---------------------------
Robustness check for lambda_y identification under beta_x/beta_y collinearity
(PM corr ~0.85). Standard fix for collinear regressors: orthogonalize beta_y
against beta_x, so lambda_y_orth prices the REFI-SPECIFIC component of risk that
is not already spanned by the level (turnover) factor.

Procedure each month:
  1. Regress beta_y on beta_x (cross-sectionally): beta_y = a + b*beta_x + resid
  2. beta_y_orth = resid  (the part of refi loading orthogonal to level loading)
  3. Run R_e = lambda_x * beta_x + lambda_y_orth * beta_y_orth
     (lambda_x is unchanged in interpretation; lambda_y_orth is now clean)

Also recomputes the baseline (non-orthogonal) lambdas so both sit side by side,
and writes a compact summary for the economic-magnitude comparison table.

Reads existing outputs. Login-node safe (no heavy compute).
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")

COUPONS   = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
GFEE      = 0.75
WAC_PROXY = 3.5

stage2 = {r["coupon"]: r["mean_cpr"]
          for r in json.load(open(os.path.join(OUT, "stage2_coupon_cpr.json")))}

panel = pd.read_csv(os.path.join(OUT, "stage3_excess_returns.csv"))
panel["Date"] = pd.to_datetime(panel["Date"])


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


rows = []
for dt in sorted(panel["Date"].unique()):
    month = panel[panel["Date"] == dt].copy()
    if len(month) < 4:
        continue
    r_t = float(month["pmms"].iloc[0])
    month["beta_x"] = month["coupon"].apply(lambda c: compute_betas(c, r_t)[0])
    month["beta_y"] = month["coupon"].apply(lambda c: compute_betas(c, r_t)[1])
    month = month.dropna(subset=["beta_x", "beta_y", "excess_return"])
    if len(month) < 4:
        continue

    bx = month["beta_x"].values
    by = month["beta_y"].values
    y  = month["excess_return"].values
    mtype = month["market_type"].iloc[0]

    # ── Baseline: R on [beta_x, beta_y] ──────────────────────────────────────
    if np.all(np.abs(by) < 1e-12):
        lx_base = float(np.linalg.lstsq(bx[:, None], y, rcond=None)[0][0])
        ly_base = np.nan
    else:
        X = np.column_stack([bx, by])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        lx_base, ly_base = float(coef[0]), float(coef[1])

    # ── Orthogonalized: beta_y_orth = resid of beta_y ~ beta_x ───────────────
    ly_orth = np.nan
    lx_orth = lx_base
    if not np.all(np.abs(by) < 1e-12) and bx.std() > 1e-12:
        # regress by on bx (with intercept)
        A = np.column_stack([np.ones_like(bx), bx])
        ab = np.linalg.lstsq(A, by, rcond=None)[0]
        by_orth = by - A @ ab
        if by_orth.std() > 1e-12:
            X2 = np.column_stack([bx, by_orth])
            coef2 = np.linalg.lstsq(X2, y, rcond=None)[0]
            lx_orth, ly_orth = float(coef2[0]), float(coef2[1])

    rows.append(dict(date=str(dt.date()), market_type=mtype, pmms=round(r_t, 4),
                     lx_base=lx_base, ly_base=ly_base,
                     lx_orth=lx_orth, ly_orth=ly_orth))

df = pd.DataFrame(rows)

def summarize(series):
    s = series.dropna()
    if len(s) < 2:
        return dict(n=len(s), mean=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(s, 0)
    return dict(n=len(s), mean=float(s.mean()), t=float(t), p=float(p))

print("="*72)
print("LAMBDA SUMMARY — BASELINE vs ORTHOGONALIZED beta_y")
print("="*72)

for mtype in ["DM", "PM"]:
    sub = df[df["market_type"] == mtype]
    print(f"\n── {mtype} ({len(sub)} months) ─────────────────────────────")
    for label, col in [("lambda_x (baseline)",   "lx_base"),
                       ("lambda_y (baseline)",   "ly_base"),
                       ("lambda_x (orthog)",     "lx_orth"),
                       ("lambda_y_orth (refi-specific)", "ly_orth")]:
        st = summarize(sub[col])
        if np.isnan(st["mean"]):
            print(f"  {label:32s}  n={st['n']:>2d}  (no variation)")
        else:
            sig = "***" if st["p"] < 0.01 else "**" if st["p"] < 0.05 else "*" if st["p"] < 0.10 else ""
            print(f"  {label:32s}  mean={st['mean']*1e4:+8.3f} bp  "
                  f"t={st['t']:+6.3f}  p={st['p']:.4f} {sig}")

# ── Save compact summary for documentation ────────────────────────────────────
summary = {}
for mtype in ["DM", "PM"]:
    sub = df[df["market_type"] == mtype]
    summary[mtype] = {
        col: summarize(sub[col])
        for col in ["lx_base", "ly_base", "lx_orth", "ly_orth"]
    }
json.dump(summary, open(os.path.join(OUT, "stage3_robustness_orthog.json"), "w"), indent=2)
df.to_csv(os.path.join(OUT, "stage3_robustness_orthog.csv"), index=False)
print("\nSaved: stage3_robustness_orthog.json, stage3_robustness_orthog.csv")
print("\nInterpretation:")
print("  - lambda_x should be ~unchanged between baseline and orthog (sanity).")
print("  - lambda_y_orth = price of refi risk NOT spanned by the level factor.")
print("  - If lambda_y_orth stays insignificant, refi risk is not separately")
print("    identified with analytical betas -> motivates the factor-shock approach.")
