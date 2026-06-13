"""
Stage 2b: Time-varying forecast CPR per coupon per month.

For each historical month t (Jan 2020 → Mar 2026):
  1. Use that month's actual PMMS as the starting rate for conditional DDPM paths
  2. Run hazard model → expected CPR per coupon bucket
  3. Store forecast_CPR[coupon, month]

This gives the proper DER forecast leg — a time-varying CPR forecast that
tracks the actual rate environment each month, comparable to the dealer survey.

The contribution: our hazard model produces forecast_CPR[c,t] which we compare
to realized_CPR[c,t] from realized_cpr_by_coupon_v4.csv.
Factor shocks: shock[c,t] = realized_CPR[c,t] - forecast_CPR[c,t]

Key fix vs Stage 2: Stage 2 used current PMMS (6.19%) for all months.
Here we use the actual historical PMMS for each month.

Because we don't have DDPM paths anchored to every historical PMMS rate,
we use the hazard model directly with a deterministic refi incentive:
  refi_incentive[c,t] = note_rate[c] - PMMS[t]
and run the model with constant refi incentive across all 33 timesteps.
This is a reasonable approximation — equivalent to assuming rates stay flat
at PMMS[t] for the life of the loan, which isolates the current-period signal.
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import json
import pandas as pd

BASE  = "/scratch/at7095/mortgage_prepayment"
OUT   = os.path.join(BASE, "outputs")
DATA  = os.path.join(BASE, "data")
SEQ   = os.path.join(BASE, "data/sequences")
CKPT  = os.path.join(OUT,  "hazard_best.pt")
SCALER= os.path.join(SEQ,  "scaler.pkl")
CALIB = os.path.join(OUT,  "hazard_calibration.json")

MAX_SEQ     = 33
N_FEATURES  = 9
GFEE        = 0.75
COUPONS     = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
DEAD_COLS   = [7, 8]

# Representative loan (same as Stage 2)
REP = dict(credit_score=740.0, orig_ltv=75.0, current_ltv=70.0,
           orig_upb=250000.0, dti=35.0,
           loan_purpose_enc=0.0, property_type_enc=0.0)

# ── Model definition (must match training) ───────────────────────────────────
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
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out  = self.input_proj(x) + self.pos_embedding(pos)
        pad  = ~mask if mask is not None else None
        out  = self.transformer(out, src_key_padding_mask=pad)
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


def build_batch_constant_refi(refi_incentive, n_paths=500):
    """
    Build synthetic loan sequences with constant refi incentive
    (deterministic approximation: PMMS stays flat at current level).
    n_paths = number of "loans" to average over for stability.
    """
    s = np.zeros((n_paths, MAX_SEQ, N_FEATURES), dtype=np.float32)
    s[:, :, 0] = refi_incentive           # constant refi incentive
    s[:, :, 1] = REP["credit_score"]
    s[:, :, 2] = REP["orig_ltv"]
    s[:, :, 3] = REP["current_ltv"]
    s[:, :, 4] = REP["orig_upb"]
    s[:, :, 5] = np.arange(1, MAX_SEQ+1)[None, :]   # loan age
    s[:, :, 6] = REP["dti"]
    s[:, :, 7] = REP["loan_purpose_enc"]
    s[:, :, 8] = REP["property_type_enc"]
    return s


def forecast_cpr(refi_incentive, model, scaler, a, b, n_paths=500):
    """
    Forecast CPR for a given refi incentive (constant across timesteps).
    Returns scalar mean CPR.
    """
    seqs = build_batch_constant_refi(refi_incentive, n_paths)
    flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(n_paths, MAX_SEQ, N_FEATURES)
    for c in DEAD_COLS:
        flat[:, :, c] = 0.0
    x    = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(n_paths, MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        logit = model(x, mask=mask, return_per_timestep=True).numpy()
    smm = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    cpr = 1.0 - (1.0 - smm) ** 12
    return float(cpr.mean())


def main():
    print("Loading model and calibration...", flush=True)
    model  = load_model()
    scaler = pickle.load(open(SCALER, "rb"))
    cal    = json.load(open(CALIB))
    a, b   = cal["a"], cal["b"]
    print(f"Calibration: a={a:.4f} b={b:.4f}", flush=True)

    # Load PMMS time series
    pmms_df = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))

    def parse_period(p):
        s = str(int(p))
        if len(s) == 5:
            return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
        elif len(s) == 6:
            return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
        return pd.NaT

    pmms_df["date"] = pmms_df["reporting_period"].apply(parse_period)
    pmms_df = pmms_df.dropna(subset=["date"]).sort_values("date")

    # Filter to months we have Bloomberg TBA data for (Jan 2018 → May 2026)
    # and realized CPR for (Jan 2020 → Sep 2025)
    pmms_hist = pmms_df[
        (pmms_df["date"] >= pd.Timestamp("2018-01-01")) &
        (pmms_df["date"] <= pd.Timestamp("2026-05-01"))
    ].reset_index(drop=True)

    print(f"Running forecast for {len(pmms_hist)} months "
          f"({pmms_hist['date'].min().date()} → {pmms_hist['date'].max().date()})",
          flush=True)

    # For each month and coupon, compute forecast CPR
    rows = []
    for _, row in pmms_hist.iterrows():
        date   = row["date"]
        pmms_t = row["rate_30yr"]

        for coupon in COUPONS:
            note_rate      = coupon + GFEE
            refi_incentive = note_rate - pmms_t   # positive = in-the-money refi

            cpr = forecast_cpr(refi_incentive, model, scaler, a, b, n_paths=200)

            rows.append(dict(
                date           = date,
                coupon         = coupon,
                note_rate      = round(note_rate, 3),
                pmms           = round(pmms_t, 4),
                refi_incentive = round(refi_incentive, 4),
                forecast_cpr   = round(cpr, 6),
            ))

        if date.month == 1 or date.month == 7:
            print(f"  {date.strftime('%b %Y')}  PMMS={pmms_t:.2f}%  "
                  f"e.g. 3.5 coupon CPR={[r['forecast_cpr'] for r in rows if r['date']==date and r['coupon']==3.5][0]:.4f}",
                  flush=True)

    df = pd.DataFrame(rows)

    # Save
    out_path = os.path.join(OUT, "forecast_cpr_timeseries.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)", flush=True)

    # Comparison vs realized
    print("\n=== Forecast vs Realized CPR comparison ===")
    real = pd.read_csv(os.path.join(OUT, "realized_cpr_by_coupon_v4.csv"))
    real["date"] = pd.to_datetime(real["date"])
    real_target = real[real["implied_mbs_coupon"].isin(COUPONS)]

    # Merge on date and coupon
    df["date"] = pd.to_datetime(df["date"])
    merged = df.merge(
        real_target[["date","implied_mbs_coupon","cpr"]].rename(
            columns={"implied_mbs_coupon":"coupon","cpr":"realized_cpr"}),
        on=["date","coupon"], how="inner")

    print(f"Merged obs: {len(merged)} (date × coupon)")

    # Summary by coupon
    summary = merged.groupby("coupon").agg(
        mean_forecast = ("forecast_cpr", "mean"),
        mean_realized = ("realized_cpr", "mean"),
        max_forecast  = ("forecast_cpr", "max"),
        max_realized  = ("realized_cpr", "max"),
        corr          = ("forecast_cpr", lambda x: x.corr(
                          merged.loc[x.index, "realized_cpr"]))
    ).round(4)
    print(summary)

    # Peak period comparison (2020-21 refi boom)
    print("\n=== Peak period (Jun 2020 – Dec 2021) ===")
    peak = merged[merged["date"].between("2020-06-01","2021-12-31")]
    peak_sum = peak.groupby("coupon").agg(
        forecast = ("forecast_cpr","mean"),
        realized = ("realized_cpr","mean"),
        ratio    = ("realized_cpr", lambda x: x.mean() /
                     peak.loc[x.index,"forecast_cpr"].mean())
    ).round(4)
    print(peak_sum)

    print("\n=== Trough period (Jun 2022 – Dec 2023) ===")
    trough = merged[merged["date"].between("2022-06-01","2023-12-31")]
    if len(trough) > 0:
        trough_sum = trough.groupby("coupon").agg(
            forecast = ("forecast_cpr","mean"),
            realized = ("realized_cpr","mean"),
        ).round(4)
        print(trough_sum)

    # Save merged comparison
    merged.to_csv(os.path.join(OUT, "forecast_vs_realized_cpr.csv"), index=False)
    print(f"\nSaved: forecast_vs_realized_cpr.csv")


if __name__ == "__main__":
    main()
