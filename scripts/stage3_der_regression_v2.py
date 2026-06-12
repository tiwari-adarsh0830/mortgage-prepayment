"""
Stage 3 v2: Treasury-hedged TBA excess returns + DER cross-sectional regression.

KEY FIX from v1: betas are time-varying — beta_x^i(t) and beta_y^i(t) are
recomputed each month using that month's PMMS as the par rate r_t.
This is critical because in 2020-21 (PM market), PMMS was ~2.8-3.5%,
making coupons 4.0-6.5 premium with negative beta_x — opposite of today.

DER Equations (5) and (6):
  beta_x^i(t) = (r_t - c^i) / [(r_t + phi^i)(phi^i + c^i)]
  beta_y^i(t) = (r_t - c^i) / [(r_t + phi^i)(phi^i + c^i)] * max(0, m^i - r_t)

where r_t = PMMS at month t, phi^i = mean CPR (fixed from hazard model).

Cross-sectional regression each month t (no intercept, per DER):
  R^e_{c,t} = lambda_x_t * beta_x^i(t) + lambda_y_t * beta_y^i(t) + eps

Output: stage3_results.json, stage3_lambda_ts.csv, stage3_excess_returns.csv
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")

COUPONS      = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
GFEE         = 0.75    # g-fee + servicing: note_rate = coupon + GFEE
WAC_PROXY    = 3.5     # balance-weighted WAC from Fannie Float Dashboard
D_MOD_AVG    = 6.5     # blended 5yr/10yr modified duration for UST hedge

# ── Load phi^i per coupon (fixed from hazard model) ───────────────────────────
stage2 = {r["coupon"]: r["mean_cpr"]
          for r in json.load(open(os.path.join(OUT, "stage2_coupon_cpr.json")))}
print("CPR per coupon:", {c: round(v,4) for c,v in stage2.items()})

# ── Load FNCL TBA prices ──────────────────────────────────────────────────────
fncl = pd.read_excel(
    os.path.join(DATA, "fncl_tba_prices_clean.xlsx"),
    sheet_name="Last_Price_Decimal", header=1)
fncl.columns = [str(c).strip() for c in fncl.columns]
fncl["Date"] = pd.to_datetime(fncl["Date"])
fncl = fncl.sort_values("Date").reset_index(drop=True)

# ── Load Treasury yields ──────────────────────────────────────────────────────
treas = pd.read_excel(
    os.path.join(DATA, "treasury_yields_clean.xlsx"),
    sheet_name="Treasury_Yields", header=1)
treas.columns = [str(c).strip() for c in treas.columns]
treas["Date"] = pd.to_datetime(treas["Date"])
treas = treas.sort_values("Date").reset_index(drop=True)
y5_col  = [c for c in treas.columns if "5yr" in c.lower()][0]
y10_col = [c for c in treas.columns if "10yr" in c.lower()][0]

# Compute monthly Treasury total return
treas["dy_avg"] = ((treas[y5_col] + treas[y10_col])/2).diff()
treas["ust_price_ret"]  = -D_MOD_AVG * treas["dy_avg"] / 100.0
treas["ust_income_ret"] = ((treas[y5_col] + treas[y10_col])/2) / 12.0 / 100.0
treas["ust_total_return"] = treas["ust_price_ret"] + treas["ust_income_ret"]
treas = treas.dropna(subset=["ust_total_return"])

# ── Load PMMS time series ─────────────────────────────────────────────────────
pmms_df = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))

def parse_pmms_period(p):
    s = str(int(p))
    if len(s) == 5:    # M YYYY
        return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    elif len(s) == 6:  # MM YYYY
        return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT

pmms_df["date"] = pmms_df["reporting_period"].apply(parse_pmms_period)
pmms_df = pmms_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
pmms_series = pmms_df.set_index("date")["rate_30yr"]

def get_pmms(dt):
    key = dt.replace(day=1)
    if key in pmms_series.index:
        return float(pmms_series[key])
    # nearest month fallback
    nearest = pmms_series.index[np.argmin(np.abs((pmms_series.index - key).days))]
    return float(pmms_series[nearest])

# ── Build TBA total return series ─────────────────────────────────────────────
returns_list = []
for coupon in COUPONS:
    col = f"FNCL {coupon}"
    if col not in fncl.columns:
        print(f"WARNING: {col} not found"); continue
    p = fncl[["Date", col]].dropna(subset=[col]).sort_values("Date").copy()
    monthly_coupon_pct = coupon / 12.0
    p["P_prev"]  = p[col].shift(1)
    p["tba_total_return"] = (p[col] + monthly_coupon_pct - p["P_prev"]) / p["P_prev"]
    p = p.dropna(subset=["tba_total_return"])
    p["coupon"] = coupon
    p = p.rename(columns={col: "price"})
    returns_list.append(p[["Date","coupon","price","tba_total_return"]])

tba_ret = pd.concat(returns_list, ignore_index=True)

# Merge Treasury return
panel = tba_ret.merge(treas[["Date","ust_total_return"]], on="Date", how="inner")
panel["excess_return"] = panel["tba_total_return"] - panel["ust_total_return"]

# Add PMMS per month
panel["pmms"] = panel["Date"].apply(get_pmms)
panel["market_type"] = np.where(panel["pmms"] > WAC_PROXY, "DM", "PM")

print(f"\nPanel: {len(panel)} obs, {panel['Date'].nunique()} months")
print(f"Market type: {panel.drop_duplicates('Date')['market_type'].value_counts().to_dict()}")

# ── Time-varying beta computation ─────────────────────────────────────────────
def compute_betas(coupon, r_t):
    """
    Compute beta_x and beta_y for coupon c at time t with par rate r_t.
    All rates in % (e.g. 6.19, not 0.0619).
    phi_i in decimal annual CPR (e.g. 0.0074).
    """
    c   = coupon
    m_i = coupon + GFEE          # note rate (WAC)
    phi = stage2.get(c, 0.01)    # mean CPR from hazard model

    # Convert to decimal for formula (r and c in %, phi in decimal)
    r_dec = r_t / 100.0
    c_dec = c   / 100.0

    denom = (r_dec + phi) * (phi + c_dec)
    if abs(denom) < 1e-12:
        return 0.0, 0.0

    base = (r_dec - c_dec) / denom

    # beta_x: d(phi)/dx = 1 for all securities
    beta_x = base * 1.0

    # beta_y: d(phi)/dy = max(0, m^i - r_t) for premium (borrower moneyness)
    #                   = 0 for discount
    borrower_moneyness = max(0.0, (m_i - r_t) / 100.0)
    beta_y = base * borrower_moneyness

    return beta_x, beta_y

# ── Fama-MacBeth cross-sectional regression ───────────────────────────────────
lambda_rows = []
dates = sorted(panel["Date"].unique())

for dt in dates:
    month = panel[panel["Date"] == dt].copy()
    if len(month) < 4:
        continue

    r_t = float(month["pmms"].iloc[0])

    # Compute time-varying betas for this month
    month["beta_x"] = month["coupon"].apply(lambda c: compute_betas(c, r_t)[0])
    month["beta_y"] = month["coupon"].apply(lambda c: compute_betas(c, r_t)[1])
    month = month.dropna(subset=["beta_x","beta_y","excess_return"])

    if len(month) < 4:
        continue

    X = month[["beta_x","beta_y"]].values
    y = month["excess_return"].values

    # If all beta_y are zero (all discount market): single-factor regression
    if np.all(np.abs(X[:,1]) < 1e-10):
        res = np.linalg.lstsq(X[:,[0]], y, rcond=None)
        lx  = float(res[0][0]); ly = np.nan
        yhat = X[:,0] * lx
    else:
        res = np.linalg.lstsq(X, y, rcond=None)
        lx, ly = float(res[0][0]), float(res[0][1])
        yhat = X @ res[0]

    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else np.nan

    mtype = month["market_type"].iloc[0]

    # Count premium/discount this month
    n_premium  = int((month["coupon"].apply(lambda c: c/100 > r_t/100)).sum())
    n_discount = len(month) - n_premium

    lambda_rows.append(dict(
        date=str(dt.date()), market_type=mtype, pmms=round(r_t,4),
        lambda_x=round(lx,8), lambda_y=round(float(ly),6) if not np.isnan(ly) else None,
        r2=round(r2,4) if not np.isnan(r2) else None,
        n_coupons=len(month), n_premium=n_premium, n_discount=n_discount
    ))

lambda_df = pd.DataFrame(lambda_rows)
print(f"\nFama-MacBeth: {len(lambda_df)} monthly regressions")

# ── DER Hypothesis 1 test ─────────────────────────────────────────────────────
print("\n=== DER Hypothesis 1 Test ===")
print("Prediction: lambda_x > 0 in DM (PMMS > WAC), lambda_x < 0 in PM\n")

for mtype in ["DM","PM"]:
    sub = lambda_df[lambda_df["market_type"]==mtype]["lambda_x"].dropna()
    if len(sub) == 0:
        print(f"{mtype}: no observations"); continue
    t_stat, p_val = stats.ttest_1samp(sub, 0)
    sign_ok = (mtype=="DM" and sub.mean()>0) or (mtype=="PM" and sub.mean()<0)
    print(f"{mtype} ({len(sub)} months):  mean={sub.mean():.6f}  std={sub.std():.6f}")
    print(f"  t-stat={t_stat:.3f}  p={p_val:.4f}  Sign correct? {'YES ✓' if sign_ok else 'NO ✗'}")

sub_ly = lambda_df["lambda_y"].dropna()
if len(sub_ly) > 0:
    print(f"\nlambda_y ({len(sub_ly)} months):  mean={sub_ly.mean():.4f}  std={sub_ly.std():.4f}")
    print(f"  Prediction: lambda_y < 0 always → {'YES ✓' if sub_ly.mean()<0 else 'NO ✗'}")

# Sample output
print("\n=== Sample lambda time series ===")
print(lambda_df[["date","market_type","pmms","lambda_x","n_premium","n_discount","r2"]].to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
lambda_df.to_csv(os.path.join(OUT, "stage3_lambda_ts.csv"), index=False)
panel.to_csv(os.path.join(OUT, "stage3_excess_returns.csv"), index=False)

summary = dict(
    n_months=len(lambda_df),
    market_type_counts=lambda_df["market_type"].value_counts().to_dict(),
    wac_proxy=WAC_PROXY,
    D_mod_used=D_MOD_AVG,
    lambda_x_by_market={
        mtype: dict(
            mean=round(float(lambda_df[lambda_df["market_type"]==mtype]["lambda_x"].mean()),6),
            std =round(float(lambda_df[lambda_df["market_type"]==mtype]["lambda_x"].std()), 6),
            n   =int((lambda_df["market_type"]==mtype).sum())
        ) for mtype in ["DM","PM"]
    },
    lambda_y_mean=round(float(lambda_df["lambda_y"].mean()),4),
    note="Betas are time-varying: recomputed each month using that month PMMS as par rate r_t"
)
json.dump(summary, open(os.path.join(OUT,"stage3_results.json"),"w"), indent=2)
print("\nSaved: stage3_lambda_ts.csv, stage3_excess_returns.csv, stage3_results.json")
