"""
Recalibrate the forecast-leg Platt (a, b) to target realized CPR levels.

Problem: hazard_calibration.json was fit to the unconditional loan-month
prepay rate (calib_rate=0.00063), which is the wrong target for the forecast
leg. Applied to in-the-money boom scenarios it under-predicts CPR ~10x.

Approach (honest framing for Gupta):
  - Calibrate (a, b) on the TROUGH regime (2022-06..2023-12) only.
  - VALIDATE on the BOOM regime (2020-06..2021-12) held-out.
  If one (a, b) calibrated on the trough also fits the boom, that is a genuine
  cross-regime test, not a boom overfit.

Fits a, b by minimizing sum of squared (forecast_cpr - realized_cpr) over
liquid coupons (2.5..5.0) in the calibration window, where
  forecast_cpr(coupon, month) = annualize( mean_t sigmoid(a*logit_{c,m,t} + b) )

We precompute per-(coupon, month) the raw logits for the representative loan
under that month's incentive, so the optimization over (a,b) is cheap.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
SEQ  = os.path.join(BASE, "data/sequences")
sys.path.insert(0, os.path.join(BASE, "scripts"))

from stage2_forecast_cpr_gfee050 import (
    load_model, build_batch_constant_refi,
    MAX_SEQ, N_FEATURES, DEAD_COLS, GFEE, COUPONS,
)

LIQUID   = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
CAL_LO, CAL_HI   = "2022-06-01", "2023-12-01"   # calibrate on trough
VAL_LO, VAL_HI   = "2020-06-01", "2021-12-01"   # validate on boom
N_PATHS  = 200

def annualize(smm):
    return 1.0 - (1.0 - smm) ** 12

def main():
    print("Loading model / scaler / PMMS / realized v6...", flush=True)
    model  = load_model()
    scaler = pickle.load(open(os.path.join(SEQ, "scaler.pkl"), "rb"))

    pmms = pd.read_csv(os.path.join(BASE, "data/pmms_monthly.csv"))
    def parse_period(p):
        s = str(int(p))
        if len(s) == 5: return pd.Timestamp(year=int(s[1:]),  month=int(s[0]),  day=1)
        if len(s) == 6: return pd.Timestamp(year=int(s[2:]),  month=int(s[:2]), day=1)
        return pd.NaT
    pmms["date"] = pmms["reporting_period"].apply(parse_period)
    pmms = pmms.dropna(subset=["date"]).sort_values("date")
    pmms_map = dict(zip(pmms["date"], pmms["rate_30yr"]))

    real = pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v6.csv"))
    real["date"] = pd.to_datetime(real["date"])
    real = real.rename(columns={"implied_mbs_coupon": "coupon", "cpr": "realized_cpr"})

    # Precompute raw per-timestep logits for each (coupon, month) cell.
    # logit depends only on refi_incentive = (coupon+GFEE) - PMMS[month].
    print("Precomputing raw logits per (coupon, month)...", flush=True)
    months = sorted([d for d in pmms_map
                     if pd.Timestamp(CAL_LO) <= d <= pd.Timestamp(CAL_HI)
                     or pd.Timestamp(VAL_LO) <= d <= pd.Timestamp(VAL_HI)])

    cell_logits = {}   # (coupon, date) -> np.array of logits (n_paths*MAX_SEQ,)
    cell_real   = {}   # (coupon, date) -> realized cpr
    for d in months:
        pmms_t = pmms_map[d]
        for c in LIQUID:
            inc  = (c + GFEE) - pmms_t
            seqs = build_batch_constant_refi(inc, N_PATHS)
            flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(N_PATHS, MAX_SEQ, N_FEATURES)
            for dc in DEAD_COLS: flat[:, :, dc] = 0.0
            x    = torch.tensor(flat, dtype=torch.float32)
            mask = torch.ones(N_PATHS, MAX_SEQ, dtype=torch.bool)
            with torch.no_grad():
                lg = model(x, mask=mask, return_per_timestep=True).numpy().ravel()
            cell_logits[(c, d)] = lg
            rr = real[(real["coupon"] == c) & (real["date"] == d)]["realized_cpr"]
            cell_real[(c, d)] = float(rr.mean()) if len(rr) else np.nan

    cal_cells = [(c, d) for (c, d) in cell_logits
                 if pd.Timestamp(CAL_LO) <= d <= pd.Timestamp(CAL_HI)
                 and np.isfinite(cell_real[(c, d)])]
    val_cells = [(c, d) for (c, d) in cell_logits
                 if pd.Timestamp(VAL_LO) <= d <= pd.Timestamp(VAL_HI)
                 and np.isfinite(cell_real[(c, d)])]
    print(f"Calibration cells (trough): {len(cal_cells)}   "
          f"Validation cells (boom): {len(val_cells)}", flush=True)

    def forecast_cpr(ab, cell):
        a, b = ab
        smm = 1.0 / (1.0 + np.exp(-(a * cell_logits[cell] + b)))
        return annualize(smm.mean())

    def loss(ab):
        err = [forecast_cpr(ab, cell) - cell_real[cell] for cell in cal_cells]
        return float(np.sum(np.square(err)))

    print("Optimizing (a, b) on trough...", flush=True)
    res = minimize(loss, x0=np.array([1.0, -3.0]), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-8, maxiter=5000))
    a_new, b_new = res.x
    print(f"  fitted a={a_new:.4f} b={b_new:.4f}  cal_SSE={res.fun:.5f}", flush=True)

    def report(cells, label):
        rows = []
        for c in LIQUID:
            fs = [forecast_cpr([a_new, b_new], (c, d)) for (cc, d) in cells if cc == c]
            rs = [cell_real[(c, d)]                    for (cc, d) in cells if cc == c]
            if fs:
                rows.append((c, np.mean(fs), np.mean(rs),
                             np.mean(fs) / np.mean(rs) if np.mean(rs) else np.nan))
        df = pd.DataFrame(rows, columns=["coupon", "forecast", "realized", "ratio"]).round(4)
        print(f"\n=== {label} ===")
        print(df.to_string(index=False))
        return df

    report(cal_cells, "CALIBRATION (trough 2022-06..2023-12) — in-sample")
    report(val_cells, "VALIDATION (boom 2020-06..2021-12) — HELD OUT")

    print("\nOld (a,b) = (0.4934, -4.8404)  [unconditional loan-month rate]")
    print(f"New (a,b) = ({a_new:.4f}, {b_new:.4f})  [forecast CPR level]")
    print("\nNOT written to disk. Inspect the held-out boom fit before saving.")

if __name__ == "__main__":
    main()
