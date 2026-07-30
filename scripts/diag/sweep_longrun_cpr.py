"""Diagnostic: does the beyond-horizon CPR level drive the hedge failure?

Scales the terminal CPR (months 34-360) by k and re-runs the verification.
Multiplicative, so the long-run CPR still shifts with the bump as specified --
a fixed floor would remove that response.
"""
import sys; sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from model_hedge_krd import (load_hazard, cpr_path, price_path, bootstrap_zeros,
                             key_rate_weights, GFEE, BUMP_BP, MAX_SEQ, N_MONTHS,
                             COUPONS, MAT_LABELS, DATA, OUT)
import os

SCALES = [1.00, 0.75, 0.50, 0.35, 0.25, 0.15]


def extend_scaled(p33, k, n=N_MONTHS):
    out = np.empty(n)
    out[:MAX_SEQ] = p33
    out[MAX_SEQ:] = p33[-1] * k
    return out


def krd_scaled(c, par, pmms, model, scaler, a, b, k, spanning):
    note = c + GFEE
    z0 = bootstrap_zeros(par)
    p0 = price_path(c, extend_scaled(cpr_path(note - pmms, model, scaler, a, b), k), z0)
    h = BUMP_BP / 100.0
    out = {}
    for ten in ('5yr', '10yr'):
        w = key_rate_weights(ten, spanning)
        dp = h if ten == '10yr' else 0.0          # PMMS keyed to 10yr
        pr = {}
        for sgn in (+1, -1):
            bp = {lab: float(par[lab]) + sgn * h * wi for lab, wi in zip(MAT_LABELS, w)}
            inc = note - (pmms + sgn * dp)
            pr[sgn] = price_path(c, extend_scaled(cpr_path(inc, model, scaler, a, b), k),
                                 bootstrap_zeros(bp))
        out[ten] = (pr[-1] - pr[+1]) / (2.0 * p0 * (h / 100.0))
    return out['5yr'], out['10yr']


def ols_t(y, X):
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ co; n, kk = X.shape
    se = np.sqrt(np.diag(float(r @ r) / (n - kk) * np.linalg.pinv(X.T @ X)))
    return co, se


model, scaler, a, b = load_hazard()

daily = pd.read_csv(os.path.join(DATA, "treasury_yields.csv"), index_col=0,
                    parse_dates=True).sort_index()
clean = pd.read_excel(os.path.join(DATA, "treasury_yields_clean.xlsx"),
                      sheet_name="Treasury_Yields", header=1)
clean.columns = [str(x).strip() for x in clean.columns]
clean["Date"] = pd.to_datetime(clean["Date"]); clean = clean.sort_values("Date").reset_index(drop=True)
y5 = [x for x in clean.columns if "5yr" in x.lower() and "avg" not in x.lower()][0]
y10 = [x for x in clean.columns if "10yr" in x.lower()][0]
clean["d_level"] = ((clean[y5] + clean[y10]) / 2).diff()
clean["d_slope"] = (clean[y10] - clean[y5]).diff()
clean["income"] = ((clean[y5] + clean[y10]) / 2) / 12.0 / 100.0

fncl = pd.read_excel(os.path.join(DATA, "fncl_tba_prices_clean.xlsx"),
                     sheet_name="Last_Price_Decimal", header=1)
fncl.columns = [str(x).strip() for x in fncl.columns]
fncl["Date"] = pd.to_datetime(fncl["Date"]); fncl = fncl.sort_values("Date").reset_index(drop=True)

pm = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))
def parse(p):
    s = str(int(p))
    if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm["date"] = pm["reporting_period"].apply(parse)
pmms_s = pm.dropna(subset=["date"]).set_index("date")["rate_30yr"]

# pre-collect month rows once
months = []
for i in range(1, len(clean)):
    prev, curr = clean["Date"].iloc[i-1], clean["Date"].iloc[i]
    idx = daily.index[daily.index <= prev]
    if not len(idx): continue
    pmk = pd.Timestamp(prev.year, prev.month, 1)
    if pmk not in pmms_s.index: continue
    months.append(dict(prev=prev, curr=curr, par=daily.loc[idx[-1]].to_dict(),
                       pmms=float(pmms_s[pmk]), i=i))

for spanning in (True, False):
    label = "SPANNING" if spanning else "LOCALIZED (literal spec)"
    print("=" * 78)
    print(label)
    print("%6s %8s %8s | %s" % ("scale", "LR_2.5", "LR_6.5",
          " ".join("%6s" % c for c in COUPONS)))
    for k in SCALES:
        rows = []
        for m in months:
            for c in COUPONS:
                col = "FNCL %s" % c
                if col not in fncl.columns: continue
                pc = fncl.loc[fncl["Date"] == m["curr"], col]
                pp = fncl.loc[fncl["Date"] == m["prev"], col]
                if pc.empty or pp.empty or pd.isna(pc.iloc[0]) or pd.isna(pp.iloc[0]): continue
                tba = (float(pc.iloc[0]) + c/12.0 - float(pp.iloc[0])) / float(pp.iloc[0])
                k5, k10 = krd_scaled(c, m["par"], m["pmms"], model, scaler, a, b, k, spanning)
                rows.append(dict(coupon=c, D_level=k5+k10, D_slope=(k10-k5)/2.0,
                                 tba=tba, income=float(clean["income"].iloc[m["i"]]),
                                 d_level=float(clean["d_level"].iloc[m["i"]]),
                                 d_slope=float(clean["d_slope"].iloc[m["i"]]),
                                 pmms=m["pmms"]))
        p = pd.DataFrame(rows).dropna(subset=["d_level", "d_slope"])
        p["hedged"] = (p["tba"] - p["income"]
                       + (p["D_level"]*p["d_level"] + p["D_slope"]*p["d_slope"])/100.0)
        ts = []
        for c in COUPONS:
            g = p[p["coupon"] == c]
            X = np.column_stack([np.ones(len(g)), g["d_level"], g["d_slope"]])
            co, se = ols_t(g["hedged"].values, X)
            ts.append(co[1]/se[1])
        lr25 = np.mean([cpr_path((2.5+GFEE)-x, model, scaler, a, b)[32]*k for x in p[p.coupon==2.5]["pmms"]])
        lr65 = np.mean([cpr_path((6.5+GFEE)-x, model, scaler, a, b)[32]*k for x in p[p.coupon==6.5]["pmms"]])
        print("%6.2f %7.1f%% %7.1f%% | %s" % (k, lr25*100, lr65*100,
              " ".join("%6.1f" % t for t in ts)))
    print()
