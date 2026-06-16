"""
Stage 3: Treasury-hedged TBA excess returns + DER cross-sectional regression.

Step 1: Load FNCL monthly prices and UST yields from Bloomberg data.
Step 2: Compute raw monthly TBA price return per coupon.
Step 3: Compute Treasury hedge return (duration-matched UST).
Step 4: Excess return = TBA return - Treasury return.
Step 5: Fama-MacBeth cross-sectional regression each month:
          R^e_{c,t} = lambda_x_t * beta_x_c + lambda_y_t * beta_y_c + eps_{c,t}
Step 6: Classify each month as discount market (DM) or premium market (PM)
          using PMMS vs WAC=3.5% proxy (per Gupta/Float Dashboard).
Step 7: Report lambda_x and lambda_y means by market type.
          DER prediction: lambda_x > 0 in DM, lambda_x < 0 in PM.

Output: outputs/stage3_results.json, stage3_lambda_ts.csv, stage3_excess_returns.csv
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE  = "/scratch/at7095/mortgage_prepayment"
OUT   = os.path.join(BASE, "outputs")
DATA  = os.path.join(BASE, "data")

COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

# ── Step 1: Load data ─────────────────────────────────────────────────────────

# FNCL TBA prices (Bloomberg, cleaned)
fncl = pd.read_excel(
    os.path.join(DATA, "fncl_tba_prices_clean.xlsx"),
    sheet_name="Last_Price_Decimal",
    header=1       # row index 1 is the actual header after the title row
)
fncl.columns = [str(c).strip() for c in fncl.columns]
fncl["Date"] = pd.to_datetime(fncl["Date"])
fncl = fncl.sort_values("Date").reset_index(drop=True)
print(f"FNCL prices: {len(fncl)} months, cols: {list(fncl.columns)}")

# Treasury yields (Bloomberg, constant maturity)
treas = pd.read_excel(
    os.path.join(DATA, "treasury_yields_clean.xlsx"),
    sheet_name="Treasury_Yields",
    header=1
)
treas.columns = [str(c).strip() for c in treas.columns]
treas["Date"] = pd.to_datetime(treas["Date"])
treas = treas.sort_values("Date").reset_index(drop=True)
print(f"Treasury yields: {len(treas)} months, cols: {list(treas.columns)}")

# PMMS (for market type classification)
pmms_df = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))
# reporting_period: integer MYYYYMM, e.g. 12026 means Jan 2026? Check format
# From memory: format is MYYYYMM e.g. 12023 = Jan 2023
# Parse: take last 4 digits as MMYY? Actually stored as integer period
# Let's check what columns exist
print(f"PMMS cols: {list(pmms_df.columns)}")
print(pmms_df.tail(3))

# DER betas (from stage2_der_betas.py)
betas = pd.DataFrame(json.load(open(os.path.join(OUT, "der_betas.json"))))
print(f"\nDER betas:\n{betas[['coupon','beta_x','beta_y','security_type']].to_string()}")

# ── Step 2: Build TBA return series per coupon ────────────────────────────────
# Raw monthly price return: R_t = P_t / P_{t-1} - 1
# Plus coupon accrual: monthly coupon = annual_coupon / 12 / 100 (as fraction of par)
# Total return approx = price_return + coupon_accrual / P_{t-1}
# For precision: TBA total return = (P_t + c/12) / P_{t-1} - 1
# where c = coupon rate in decimal, P in decimal (100 = par)

returns_list = []

for coupon in COUPONS:
    col = f"FNCL {coupon}"
    if col not in fncl.columns:
        col = f"FNCL {coupon:.1f}"
    if col not in fncl.columns:
        print(f"WARNING: column {col} not found. Available: {[c for c in fncl.columns if 'FNCL' in c]}")
        continue

    prices = fncl[["Date", col]].copy()
    prices = prices.dropna(subset=[col])
    prices = prices.sort_values("Date").reset_index(drop=True)

    # Monthly coupon accrual as fraction of par (price is in % of par)
    monthly_coupon_pct = coupon / 12.0   # e.g. 3.0/12 = 0.25 per month in price pts

    # Total return (in decimal): (P_t + coupon_accrual - P_{t-1}) / P_{t-1}
    prices["P_prev"]  = prices[col].shift(1)
    prices["ret_raw"] = (prices[col] + monthly_coupon_pct - prices["P_prev"]) / prices["P_prev"]

    # Drop first row (no lagged price)
    prices = prices.dropna(subset=["ret_raw"])
    prices["coupon"] = coupon
    prices = prices.rename(columns={col: "price", "ret_raw": "tba_total_return"})
    returns_list.append(prices[["Date", "coupon", "price", "tba_total_return"]])

tba_ret = pd.concat(returns_list, ignore_index=True)
print(f"\nTBA returns panel: {len(tba_ret)} obs, {tba_ret['Date'].nunique()} months")

# ── Step 3: Compute Treasury hedge return ─────────────────────────────────────
# Treasury return from yield change using modified duration approximation:
# R_UST = -D_mod * delta_yield / (1 + y/2)  (price return)
# Plus yield income: y/12 per month
#
# For duration D_mod: use average of 5yr and 10yr duration as proxy
# D_mod(5yr UST) ≈ 4.5 years, D_mod(10yr UST) ≈ 8.5 years
# Use 7yr proxy (midpoint) as Gupta confirmed 5+10 hedge
# More precisely: compute from the yield data directly
# Duration ≈ (1 - 1/(1+y/2)^(2n)) / y  for annual coupon bond
# For simplicity use D_mod = 6.5 years (mid-point 5+10yr average)
# This is a reasonable approximation for a 5/10yr blended hedge

D_MOD_5YR  = 4.5    # approximate modified duration for 5yr UST
D_MOD_10YR = 8.5    # approximate modified duration for 10yr UST
D_MOD_AVG  = (D_MOD_5YR + D_MOD_10YR) / 2.0   # = 6.5

# Identify yield columns
y5_col  = [c for c in treas.columns if "5yr" in c.lower() or "5 yr" in c.lower()][0]
y10_col = [c for c in treas.columns if "10yr" in c.lower() or "10 yr" in c.lower()][0]
print(f"\nUsing Treasury columns: {y5_col}, {y10_col}")

treas["y5_prev"]  = treas[y5_col].shift(1)
treas["y10_prev"] = treas[y10_col].shift(1)
treas["dy5"]      = treas[y5_col]  - treas["y5_prev"]    # yield change in %
treas["dy10"]     = treas[y10_col] - treas["y10_prev"]
treas["dy_avg"]   = (treas["dy5"] + treas["dy10"]) / 2.0

# Treasury price return ≈ -D_mod * delta_y (in decimal)
# delta_y in % → divide by 100 first
treas["ust_price_ret"] = -D_MOD_AVG * treas["dy_avg"] / 100.0

# Treasury income return: avg yield / 12 / 100
treas["ust_income_ret"] = ((treas[y5_col] + treas[y10_col]) / 2.0) / 12.0 / 100.0

# Total UST return
treas["ust_total_return"] = treas["ust_price_ret"] + treas["ust_income_ret"]
treas = treas.dropna(subset=["ust_total_return"])

# ── Step 4: Excess returns ────────────────────────────────────────────────────
panel = tba_ret.merge(
    treas[["Date", "ust_total_return"]],
    on="Date", how="inner"
)
panel["excess_return"] = panel["tba_total_return"] - panel["ust_total_return"]
print(f"\nExcess return panel: {len(panel)} obs after merge")
print(panel.groupby("coupon")["excess_return"].describe().round(4))

# ── Step 5: Market type classification ───────────────────────────────────────
# DM/PM classification: PMMS > WAC → discount market, PMMS < WAC → premium market
# WAC proxy = 3.5% (balance-weighted from Fannie Mae Float Dashboard, June 2026)
WAC_PROXY = 3.5

# Parse PMMS dates
# reporting_period format from memory: integer like 12023 = Jan 2023, 122024?
# Let's parse more carefully
print(f"\nPMMS sample:\n{pmms_df.head(5)}")

# Try to parse reporting_period to date
# Format appears to be: MYYYYMM where M=month, YYYY=year? 
# e.g. 12023 could be Jan 2023 or the integer 12023
# From prior sessions: key is `reporting_period` e.g. `12023`
# Let's try: if 5 digits: M+YYYY (1-digit month + 4-digit year)
# if 6 digits: MM+YYYY
def parse_pmms_period(p):
    # Format: M YYYY e.g. 41971=Apr 1971, 121999=Dec 1999
    s = str(int(p))
    if len(s) == 5:    # M YYYY  e.g. 41971
        month = int(s[0])
        year  = int(s[1:])
    elif len(s) == 6:  # MM YYYY e.g. 121999
        month = int(s[:2])
        year  = int(s[2:])
    else:
        return pd.NaT
    return pd.Timestamp(year=year, month=month, day=1)

pmms_df["date"] = pmms_df["reporting_period"].apply(parse_pmms_period)
pmms_df = pmms_df.dropna(subset=["date"]).sort_values("date")
print(f"PMMS parsed: {pmms_df['date'].min()} → {pmms_df['date'].max()}")
print(pmms_df[["date","rate_30yr"]].tail(5))

# Merge PMMS into panel
panel["month"] = panel["Date"].dt.to_period("M").dt.to_timestamp()
pmms_monthly = pmms_df.set_index("date")["rate_30yr"].to_dict()

def get_pmms(dt):
    # try exact match, then nearest month
    key = dt.replace(day=1)
    if key in pmms_monthly:
        return pmms_monthly[key]
    # fallback: find nearest
    nearest = min(pmms_monthly.keys(), key=lambda d: abs((d - key).days))
    return pmms_monthly[nearest]

panel["pmms"] = panel["month"].apply(get_pmms)
panel["market_type"] = np.where(panel["pmms"] > WAC_PROXY, "DM", "PM")

print(f"\nMarket type distribution:")
print(panel[["Date","market_type"]].drop_duplicates("Date")["market_type"].value_counts())

# ── Step 6: Fama-MacBeth cross-sectional regression ──────────────────────────
# Each month t: regress excess_return_c on beta_x_c and beta_y_c (no intercept)
# R^e_{c,t} = lambda_x_t * beta_x_c + lambda_y_t * beta_y_c + eps

beta_map = betas.set_index("coupon")[["beta_x","beta_y"]].to_dict("index")

lambda_rows = []
dates = sorted(panel["Date"].unique())

for dt in dates:
    month_data = panel[panel["Date"] == dt].copy()
    if len(month_data) < 4:   # need at least 4 coupons for regression
        continue

    # Map betas
    month_data["beta_x"] = month_data["coupon"].map(lambda c: beta_map.get(c, {}).get("beta_x", np.nan))
    month_data["beta_y"] = month_data["coupon"].map(lambda c: beta_map.get(c, {}).get("beta_y", np.nan))
    month_data = month_data.dropna(subset=["beta_x","beta_y","excess_return"])

    if len(month_data) < 4:
        continue

    # OLS: R^e = lambda_x * beta_x + lambda_y * beta_y (no intercept, per DER)
    X = month_data[["beta_x","beta_y"]].values
    y = month_data["excess_return"].values

    # Only fit lambda_y if there are premium securities with nonzero beta_y
    if np.all(X[:,1] == 0):
        # All discount: only fit lambda_x
        res = np.linalg.lstsq(X[:,[0]], y, rcond=None)
        lx = float(res[0][0]); ly = np.nan
        r2 = float(1 - np.sum((y - X[:,0]*lx)**2) / np.sum((y - y.mean())**2)) if y.std() > 0 else np.nan
    else:
        res = np.linalg.lstsq(X, y, rcond=None)
        lx, ly = float(res[0][0]), float(res[0][1])
        y_hat = X @ res[0]
        r2 = float(1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)) if y.std() > 0 else np.nan

    mtype = month_data["market_type"].iloc[0]
    lambda_rows.append(dict(
        date=dt, market_type=mtype,
        lambda_x=round(lx,6), lambda_y=round(ly,6) if not np.isnan(ly) else None,
        n_coupons=len(month_data), r2=round(r2,4) if not np.isnan(r2) else None,
        pmms=float(month_data["pmms"].iloc[0])
    ))

lambda_df = pd.DataFrame(lambda_rows)
print(f"\nFama-MacBeth: {len(lambda_df)} monthly regressions")
print(lambda_df[["date","market_type","lambda_x","lambda_y","r2"]].tail(10).to_string())

# ── Step 7: Summary by market type ───────────────────────────────────────────
print("\n=== DER Hypothesis 1 Test ===")
print("Prediction: lambda_x > 0 in DM, lambda_x < 0 in PM\n")

for mtype in ["DM","PM"]:
    sub = lambda_df[lambda_df["market_type"] == mtype]["lambda_x"].dropna()
    if len(sub) == 0:
        print(f"{mtype}: no observations"); continue
    t_stat, p_val = stats.ttest_1samp(sub, 0)
    print(f"{mtype} ({len(sub)} months):")
    print(f"  lambda_x mean = {sub.mean():.6f}  std = {sub.std():.6f}")
    print(f"  t-stat = {t_stat:.3f}  p-value = {p_val:.4f}")
    print(f"  Sign correct? {'YES' if (mtype=='DM' and sub.mean()>0) or (mtype=='PM' and sub.mean()<0) else 'NO'}")

sub_ly = lambda_df["lambda_y"].dropna()
if len(sub_ly) > 0:
    print(f"\nlambda_y overall ({len(sub_ly)} months):")
    print(f"  mean = {sub_ly.mean():.6f}  std = {sub_ly.std():.6f}")
    print(f"  Prediction: lambda_y < 0 always → {'YES' if sub_ly.mean()<0 else 'NO'}")

# ── Save outputs ──────────────────────────────────────────────────────────────
lambda_df.to_csv(os.path.join(OUT, "stage3_lambda_ts.csv"), index=False)
panel.to_csv(os.path.join(OUT, "stage3_excess_returns.csv"), index=False)

summary = {
    "n_months": len(lambda_df),
    "market_type_counts": lambda_df["market_type"].value_counts().to_dict(),
    "lambda_x_by_market": {
        mtype: {
            "mean": round(float(lambda_df[lambda_df["market_type"]==mtype]["lambda_x"].mean()), 6),
            "std":  round(float(lambda_df[lambda_df["market_type"]==mtype]["lambda_x"].std()),  6),
            "n":    int((lambda_df["market_type"]==mtype).sum())
        }
        for mtype in ["DM","PM"]
    },
    "lambda_y_mean": round(float(lambda_df["lambda_y"].mean()), 6),
    "wac_proxy": WAC_PROXY,
    "D_mod_used": D_MOD_AVG,
}
json.dump(summary, open(os.path.join(OUT, "stage3_results.json"), "w"), indent=2)

print(f"\nSaved: stage3_lambda_ts.csv, stage3_excess_returns.csv, stage3_results.json")
