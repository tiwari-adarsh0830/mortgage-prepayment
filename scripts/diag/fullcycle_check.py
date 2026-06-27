"""
FULL-CYCLE CHECK — recalibrated cohort-CPR Platt vs v6 realized, all months.

Purpose: before committing the recalibrated forecast Platt (a=0.4559, b=-3.1376)
to the pipeline, verify the forecast tracks v6 realized across the WHOLE cycle
(2020-2025), not just the two spot-checked months. We check:
  - per-regime mean forecast vs realized (boom / transition / trough)
  - cross-sectional monotonicity in coupon
  - time-series correlation per coupon
  - the 3.5% coupon monthly series (should show boom hump + trough collapse)

This uses the SAME representative-loan forecast as production
(stage2_forecast_cpr_gfee050) but swaps in the recalibrated (a,b). The POC
showed population vs representative barely changes aggregate CPR, so we keep
the cheap representative-loan forecast and just fix the calibration.

NOTE on calibration philosophy (from DER / Session 3 p.107 dealer survey):
  The forecast leg phi is a CONDITIONAL cohort CPR (what a dealer survey gives:
  e.g. FNCL at -100bps -> 30-90% CPR). It is a DIFFERENT object from the
  unconditional loan-level hazard (b=-4.84) used for OAS discounting. Two
  calibrations, two purposes -- documented, not hidden.

Writes nothing. Prints tables for inspection.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
SEQ  = os.path.join(BASE, "data/sequences")
sys.path.insert(0, os.path.join(BASE, "scripts"))

from stage2_forecast_cpr_gfee050 import (
    load_model, build_batch_constant_refi,
    MAX_SEQ, N_FEATURES, DEAD_COLS, GFEE,
)

# Recalibrated cohort-CPR Platt (fit on trough, validated on boom)
A_NEW, B_NEW = 0.4559, -3.1376
COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
LIQUID  = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
N_PATHS = 200

def annualize(smm): return 1.0 - (1.0 - smm) ** 12

def forecast_cpr(inc, model, scaler, a, b):
    seqs = build_batch_constant_refi(inc, N_PATHS)
    flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(N_PATHS, MAX_SEQ, N_FEATURES)
    for dc in DEAD_COLS: flat[:, :, dc] = 0.0
    x = torch.tensor(flat, dtype=torch.float32)
    mk = torch.ones(N_PATHS, MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        logit = model(x, mask=mk, return_per_timestep=True).numpy()
    smm = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    return float(annualize(smm.mean()))

def main():
    model  = load_model()
    scaler = pickle.load(open(os.path.join(SEQ, "scaler.pkl"), "rb"))
    print(f"Recalibrated Platt a={A_NEW} b={B_NEW}\n", flush=True)

    pmms = pd.read_csv(os.path.join(BASE, "data/pmms_monthly.csv"))
    def parse(p):
        s = str(int(p))
        if len(s)==5: return pd.Timestamp(year=int(s[1:]),  month=int(s[0]),  day=1)
        if len(s)==6: return pd.Timestamp(year=int(s[2:]),  month=int(s[:2]), day=1)
        return pd.NaT
    pmms["date"] = pmms["reporting_period"].apply(parse)
    pmms = pmms.dropna(subset=["date"]).sort_values("date")
    pmms = pmms[(pmms["date"]>="2020-01-01") & (pmms["date"]<="2025-12-01")]

    rows = []
    for _, r in pmms.iterrows():
        for c in COUPONS:
            inc = (c + GFEE) - r["rate_30yr"]
            rows.append(dict(date=r["date"], coupon=c,
                             forecast_cpr=forecast_cpr(inc, model, scaler, A_NEW, B_NEW)))
    fc = pd.DataFrame(rows)

    real = pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v6.csv"))
    real["date"] = pd.to_datetime(real["date"])
    real = real.rename(columns={"implied_mbs_coupon":"coupon","cpr":"realized_cpr"})
    real = real.groupby(["date","coupon"], as_index=False)["realized_cpr"].mean()

    m = fc.merge(real, on=["date","coupon"], how="inner")
    m = m[m["coupon"].isin(LIQUID)]
    print(f"Merged obs: {len(m)}\n", flush=True)

    regimes = {
        "BOOM 2020-06..2021-12":  ("2020-06-01","2021-12-01"),
        "TRANS 2022-01..2022-12": ("2022-01-01","2022-12-01"),
        "TROUGH 2023-01..2023-12":("2023-01-01","2023-12-01"),
        "RECENT 2024-01..2025-12":("2024-01-01","2025-12-01"),
    }
    for label,(lo,hi) in regimes.items():
        sub = m[(m["date"]>=lo)&(m["date"]<=hi)]
        if not len(sub): continue
        g = sub.groupby("coupon").agg(
            forecast=("forecast_cpr","mean"),
            realized=("realized_cpr","mean")).round(4)
        g["ratio"] = (g["forecast"]/g["realized"]).round(2)
        print(f"=== {label} ===")
        print(g.to_string()); print()

    print("=== Time-series corr(forecast, realized) per coupon (full 2020-2025) ===")
    for c in LIQUID:
        s = m[m["coupon"]==c]
        if len(s)>3:
            print(f"  {c}: corr={s['forecast_cpr'].corr(s['realized_cpr']):.3f}  n={len(s)}")
    print()

    print("=== 3.5% coupon monthly series (eyeball boom hump + trough) ===")
    s = m[m["coupon"]==3.5].sort_values("date")
    for _,r in s.iterrows():
        if r["date"].month in (1,4,7,10):
            print(f"  {r['date'].date()}  fcst={r['forecast_cpr']:.3f}  real={r['realized_cpr']:.3f}")

if __name__ == "__main__":
    main()
