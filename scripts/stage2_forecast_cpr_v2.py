"""
Stage 2b v2: Time-varying forecast CPR — FAST VERSION.
Batches all 9 coupons together per month. Suppresses sklearn warnings.
Should complete in ~10 minutes vs 6+ hours for v1.
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

BASE  = "/scratch/at7095/mortgage_prepayment"
OUT   = os.path.join(BASE, "outputs")
DATA  = os.path.join(BASE, "data")
SEQ   = os.path.join(BASE, "data/sequences")
CKPT  = os.path.join(OUT,  "hazard_best.pt")
SCALER= os.path.join(SEQ,  "scaler.pkl")
CALIB = os.path.join(OUT,  "hazard_calibration.json")

MAX_SEQ    = 33
N_FEATURES = 9
GFEE       = 0.75
COUPONS    = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
DEAD_COLS  = [7, 8]
N_PATHS    = 100   # per coupon — enough for stable estimate

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
    """
    Batch forward pass for all 9 coupons at once.
    Returns dict: coupon -> forecast_cpr
    """
    # Build batch: N_PATHS sequences per coupon, all coupons stacked
    total = len(COUPONS) * N_PATHS
    seqs  = np.zeros((total, MAX_SEQ, N_FEATURES), dtype=np.float32)

    for ci, coupon in enumerate(COUPONS):
        note_rate      = coupon + GFEE
        refi_incentive = note_rate - pmms_t
        start = ci * N_PATHS
        end   = start + N_PATHS
        seqs[start:end, :, 0] = refi_incentive
        seqs[start:end, :, 1] = REP["credit_score"]
        seqs[start:end, :, 2] = REP["orig_ltv"]
        seqs[start:end, :, 3] = REP["current_ltv"]
        seqs[start:end, :, 4] = REP["orig_upb"]
        seqs[start:end, :, 5] = np.arange(1, MAX_SEQ+1)[None, :]
        seqs[start:end, :, 6] = REP["dti"]
        seqs[start:end, :, 7] = REP["loan_purpose_enc"]
        seqs[start:end, :, 8] = REP["property_type_enc"]

    # Scale
    flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(total, MAX_SEQ, N_FEATURES)
    for c in DEAD_COLS:
        flat[:, :, c] = 0.0

    x    = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(total, MAX_SEQ, dtype=torch.bool)

    with torch.no_grad():
        logit = model(x, mask=mask, return_per_timestep=True).numpy()

    smm = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    cpr = 1.0 - (1.0 - smm) ** 12   # (total, MAX_SEQ)

    # Average per coupon
    result = {}
    for ci, coupon in enumerate(COUPONS):
        start = ci * N_PATHS
        end   = start + N_PATHS
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
    print("Loading model...", flush=True)
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

    print(f"Running {len(pmms_hist)} months × {len(COUPONS)} coupons "
          f"(batch size {len(COUPONS)*N_PATHS} per month)\n", flush=True)

    rows = []
    for i, (_, row) in enumerate(pmms_hist.iterrows()):
        date   = row["date"]
        pmms_t = row["rate_30yr"]

        cpr_dict = forecast_all_coupons(pmms_t, model, scaler, a, b)

        for coupon in COUPONS:
            rows.append(dict(
                date           = date,
                coupon         = coupon,
                note_rate      = round(coupon + GFEE, 3),
                pmms           = round(pmms_t, 4),
                refi_incentive = round(coupon + GFEE - pmms_t, 4),
                forecast_cpr   = round(cpr_dict[coupon], 6),
            ))

        if i % 12 == 0:
            print(f"  {date.strftime('%b %Y')}  PMMS={pmms_t:.2f}%  "
                  f"3.5c CPR={cpr_dict[3.5]:.4f}  "
                  f"6.5c CPR={cpr_dict[6.5]:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "forecast_cpr_timeseries.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)", flush=True)

    # Compare vs realized
    real = pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v4.csv"))
    real["date"] = pd.to_datetime(real["date"])
    real_target  = real[real["implied_mbs_coupon"].isin(COUPONS)]
    df["date"]   = pd.to_datetime(df["date"])

    merged = df.merge(
        real_target[["date","implied_mbs_coupon","cpr"]].rename(
            columns={"implied_mbs_coupon":"coupon","cpr":"realized_cpr"}),
        on=["date","coupon"], how="inner")

    print(f"\nMerged: {len(merged)} obs (date × coupon)")

    print("\n=== Full period summary ===")
    print(merged.groupby("coupon").agg(
        mean_forecast = ("forecast_cpr","mean"),
        mean_realized = ("realized_cpr","mean"),
        corr          = ("forecast_cpr", lambda x:
                         x.corr(merged.loc[x.index,"realized_cpr"]))
    ).round(4))

    print("\n=== Peak 2020-21 ===")
    peak = merged[merged["date"].between("2020-06-01","2021-12-31")]
    print(peak.groupby("coupon").agg(
        forecast=("forecast_cpr","mean"),
        realized=("realized_cpr","mean"),
    ).round(4))

    print("\n=== Trough 2022-23 ===")
    trough = merged[merged["date"].between("2022-06-01","2023-12-31")]
    print(trough.groupby("coupon").agg(
        forecast=("forecast_cpr","mean"),
        realized=("realized_cpr","mean"),
    ).round(4))

    merged.to_csv(os.path.join(OUT, "forecast_vs_realized_cpr.csv"), index=False)
    print("\nSaved: forecast_vs_realized_cpr.csv")

if __name__ == "__main__":
    main()
