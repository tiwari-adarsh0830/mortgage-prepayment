"""
Stage 2b OOS: Out-of-sample forecast CPR using the CENSORED (<=Dec 2019) model.

Mirrors stage2_forecast_cpr_v2.py exactly, with three swaps:
  - checkpoint  -> hazard_pre2020_best.pt   (trained only on <=2019 data)
  - scaler      -> sequences_pre2020/scaler.pkl  (fitted on <=2019 data)
  - calibration -> hazard_pre2020_calibration.json (Platt on <=2019 holdout)

The model never saw a 2020+ observation. Forecasting 2020-21 CPR is therefore a
genuine out-of-sample test (advisor option b). Note PMMS fell to ~2.7% in 2020-21
vs 3.6-4.9% in the 2018-19 training window, so the model EXTRAPOLATES the
refi-incentive -> prepay relationship beyond its training range.

Compares forecast vs realized (v5) with emphasis on the 2020-21 OOS window.

Outputs: forecast_cpr_timeseries_oos.csv, forecast_vs_realized_cpr_oos.csv
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE   = "/scratch/at7095/mortgage_prepayment"
OUT    = os.path.join(BASE, "outputs")
DATA   = os.path.join(BASE, "data")
SEQ    = os.path.join(BASE, "data/sequences_pre2020")        # <-- censored scaler dir
CKPT   = os.path.join(OUT,  "hazard_pre2020_best.pt")        # <-- censored model
SCALER = os.path.join(SEQ,  "scaler.pkl")                    # <-- censored scaler
CALIB  = os.path.join(OUT,  "hazard_pre2020_calibration.json")  # <-- censored calib

MAX_SEQ    = 33
N_FEATURES = 9
GFEE       = 0.75
COUPONS    = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
DEAD_COLS  = [7, 8]
N_PATHS    = 100

REP = dict(credit_score=740.0, orig_ltv=75.0, current_ltv=70.0,
           orig_upb=250000.0, dti=35.0,
           loan_purpose_enc=0.0, property_type_enc=0.0)

class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, n_heads=4, n_layers=2,
                 dim_ff=256, max_seq=33, dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
              dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer  = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier   = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, mask=None, return_per_timestep=False):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out = self.input_proj(x) + self.pos_embedding(pos)
        pad = ~mask if mask is not None else None
        out = self.transformer(out, src_key_padding_mask=pad)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(1) / real.sum(1).clamp(min=1)
        else:
            out = out.mean(1)
        return self.classifier(out).squeeze(-1)

def load_model():
    ck  = torch.load(CKPT, map_location="cpu")
    cfg = ck.get("config", {})
    m   = PrepaymentTransformer(
        input_dim=N_FEATURES, d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.1))
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m

def forecast_all_coupons(pmms_t, model, scaler, a, b):
    total = len(COUPONS) * N_PATHS
    seqs  = np.zeros((total, MAX_SEQ, N_FEATURES), dtype=np.float32)
    for ci, coupon in enumerate(COUPONS):
        note_rate      = coupon + GFEE
        refi_incentive = note_rate - pmms_t
        start = ci * N_PATHS; end = start + N_PATHS
        seqs[start:end, :, 0] = refi_incentive
        seqs[start:end, :, 1] = REP["credit_score"]
        seqs[start:end, :, 2] = REP["orig_ltv"]
        seqs[start:end, :, 3] = REP["current_ltv"]
        seqs[start:end, :, 4] = REP["orig_upb"]
        seqs[start:end, :, 5] = np.arange(1, MAX_SEQ+1)[None, :]
        seqs[start:end, :, 6] = REP["dti"]
        seqs[start:end, :, 7] = REP["loan_purpose_enc"]
        seqs[start:end, :, 8] = REP["property_type_enc"]
    flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(total, MAX_SEQ, N_FEATURES)
    for c in DEAD_COLS:
        flat[:, :, c] = 0.0
    x    = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(total, MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        logit = model(x, mask=mask, return_per_timestep=True).numpy()
    smm = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    cpr = 1.0 - (1.0 - smm) ** 12
    result = {}
    for ci, coupon in enumerate(COUPONS):
        start = ci * N_PATHS; end = start + N_PATHS
        result[coupon] = float(cpr[start:end].mean())
    return result

def parse_period(p):
    s = str(int(p))
    if len(s) == 5:
        return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    elif len(s) == 6:
        return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT

def main():
    print("Loading CENSORED (<=2019) model...", flush=True)
    print(f"  checkpoint:  {CKPT}")
    print(f"  scaler:      {SCALER}")
    print(f"  calibration: {CALIB}")
    model  = load_model()
    scaler = pickle.load(open(SCALER, "rb"))
    cal    = json.load(open(CALIB))
    a, b   = cal["a"], cal["b"]
    print(f"Calibration: a={a:.4f} b={b:.4f}", flush=True)

    pmms_df = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))
    pmms_df["date"] = pmms_df["reporting_period"].apply(parse_period)
    pmms_df = pmms_df.dropna(subset=["date"]).sort_values("date")
    pmms_hist = pmms_df[
        (pmms_df["date"] >= pd.Timestamp("2018-01-01")) &
        (pmms_df["date"] <= pd.Timestamp("2026-05-01"))
    ].reset_index(drop=True)

    print(f"\nForecasting {len(pmms_hist)} months x {len(COUPONS)} coupons\n", flush=True)
    rows = []
    for i, (_, row) in enumerate(pmms_hist.iterrows()):
        date = row["date"]; pmms_t = row["rate_30yr"]
        cpr_dict = forecast_all_coupons(pmms_t, model, scaler, a, b)
        for coupon in COUPONS:
            rows.append(dict(
                date=date, coupon=coupon, note_rate=round(coupon+GFEE,3),
                pmms=round(pmms_t,4), refi_incentive=round(coupon+GFEE-pmms_t,4),
                forecast_cpr=round(cpr_dict[coupon],6)))
        if i % 12 == 0:
            print(f"  {date.strftime('%b %Y')}  PMMS={pmms_t:.2f}%  "
                  f"3.5c={cpr_dict[3.5]:.4f}  6.5c={cpr_dict[6.5]:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "forecast_cpr_timeseries_oos.csv"), index=False)
    print(f"\nSaved: forecast_cpr_timeseries_oos.csv ({len(df)} rows)", flush=True)

    # Compare vs realized v5
    real = pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v5.csv"))
    real["date"] = pd.to_datetime(real["date"])
    coupon_col = "implied_mbs_coupon" if "implied_mbs_coupon" in real.columns else "coupon"
    real_target = real[real[coupon_col].isin(COUPONS)]
    df["date"]  = pd.to_datetime(df["date"])
    merged = df.merge(
        real_target[["date", coupon_col, "cpr"]].rename(
            columns={coupon_col:"coupon","cpr":"realized_cpr"}),
        on=["date","coupon"], how="inner")
    print(f"\nMerged: {len(merged)} obs (date x coupon)")

    # ── OOS WINDOW: 2020-21 (the key test) ────────────────────────────────────
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE WINDOW 2020-2021 (model never saw this period)")
    print("="*60)
    oos = merged[merged["date"].between("2020-01-01","2021-12-31")]
    oos_summary = oos.groupby("coupon").agg(
        forecast=("forecast_cpr","mean"),
        realized=("realized_cpr","mean"),
        corr=("forecast_cpr", lambda x: x.corr(oos.loc[x.index,"realized_cpr"]))
    ).round(4)
    print(oos_summary)

    # Overall OOS error metrics
    oos_clean = oos.dropna(subset=["forecast_cpr","realized_cpr"])
    if len(oos_clean):
        err = oos_clean["forecast_cpr"] - oos_clean["realized_cpr"]
        mae  = float(err.abs().mean())
        rmse = float(np.sqrt((err**2).mean()))
        corr = float(oos_clean["forecast_cpr"].corr(oos_clean["realized_cpr"]))
        print(f"\nOOS 2020-21:  MAE={mae:.4f}  RMSE={rmse:.4f}  corr={corr:.3f}  n={len(oos_clean)}")

    # ── In-sample window 2018-19 (sanity: should fit well) ────────────────────
    print("\n" + "="*60)
    print("IN-SAMPLE WINDOW 2018-2019 (sanity check, should track)")
    print("="*60)
    ins = merged[merged["date"].between("2018-01-01","2019-12-31")]
    print(ins.groupby("coupon").agg(
        forecast=("forecast_cpr","mean"),
        realized=("realized_cpr","mean")).round(4))

    merged.to_csv(os.path.join(OUT, "forecast_vs_realized_cpr_oos.csv"), index=False)
    print("\nSaved: forecast_vs_realized_cpr_oos.csv")

if __name__ == "__main__":
    main()
