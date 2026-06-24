"""
Stage 2b (Rolling): Time-varying forecast CPR per coupon per month,
using rolling model checkpoints for genuine t->t+1 OOS validation.

MODEL DISPATCH (which checkpoint forecasts each calendar month):
  Jan 2021 - Dec 2021  ->  cutoff_2020 model (trained through Dec 2020)  [OOS]
  Jan 2022 - Dec 2022  ->  cutoff_2021 model (trained through Dec 2021)  [OOS]
  Jan 2020 - Dec 2020  ->  production model  (in-sample baseline ref)

For 2021 and 2022 the model used to forecast month t was never exposed to
observations from month t or later -> a genuine rolling out-of-sample test.

-----------------------------------------------------------------------------
KNOWN LIMITATIONS / FIXES APPLIED (read before trusting absolute levels):

(A) GFEE = 0.50 to MATCH realized_cpr_by_coupon_v5.csv, which buckets loans as
    implied_mbs_coupon = round(note_rate*2)/2 - 0.50. The production forecast
    script used 0.75; using 0.75 here would misalign the merge by ~0.25pp.

(B) DEAD_COLS is per-model. The production model was trained with
    loan_purpose_enc / property_type_enc all-zero (dead) -> they must be zeroed.
    The rolling models were trained with FIXED categoricals (R/C/P, SF/PU/CO/MH)
    -> they are LIVE features and must NOT be zeroed. Zeroing them would feed
    the rolling models out-of-distribution input.

(C) Platt calibration (a, b) is model-specific. We load each model's own
    hazard_calibration.json if present; otherwise we fall back to the
    production (a, b) and print a LOUD warning. Reusing production calibration
    on a differently-scaled rolling logit distribution distorts CPR LEVELS
    (directional tracking and cross-sectional ordering remain valid). If levels
    look off, recompute Platt on each rolling test set.

(D) n_paths defaults to 1: build_batch_constant_refi creates identical rows and
    the model is deterministic at eval, so averaging over N paths returns the
    same number N times. n_paths=1 is identical and ~N x faster. CPU is fine.
-----------------------------------------------------------------------------

OUTPUT:
  outputs/rolling_forecast_cpr_timeseries.csv  - full coupon x month panel
  outputs/rolling_forecast_vs_realized.csv     - merged with realized CPR
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

# ── Model registry: ckpt + scaler + (optional) per-model calibration ──────────
PROD_CALIB = os.path.join(OUT, "hazard_calibration.json")  # production Platt

MODELS = {
    "production": {
        "ckpt":   os.path.join(OUT,  "hazard_best.pt"),
        "scaler": os.path.join(BASE, "data/sequences/scaler.pkl"),
        "calib":  PROD_CALIB,
        "dead_cols": [7, 8],          # categoricals were dead in production
    },
    "cutoff_2020": {
        "ckpt":   os.path.join(OUT,  "rolling/cutoff_2020/hazard_best.pt"),
        "scaler": os.path.join(BASE, "data/sequences_rolling/cutoff_2020/scaler.pkl"),
        "calib":  os.path.join(OUT,  "rolling/cutoff_2020/hazard_calibration.json"),
        "dead_cols": [],              # categoricals are LIVE in rolling models
    },
    "cutoff_2021": {
        "ckpt":   os.path.join(OUT,  "rolling/cutoff_2021/hazard_best.pt"),
        "scaler": os.path.join(BASE, "data/sequences_rolling/cutoff_2021/scaler.pkl"),
        "calib":  os.path.join(OUT,  "rolling/cutoff_2021/hazard_calibration.json"),
        "dead_cols": [],
    },
    "cutoff_2022": {
        "ckpt":   os.path.join(OUT,  "rolling/cutoff_2022/hazard_best.pt"),
        "scaler": os.path.join(BASE, "data/sequences_rolling/cutoff_2022/scaler.pkl"),
        "calib":  os.path.join(OUT,  "rolling/cutoff_2022/hazard_calibration.json"),
        "dead_cols": [],
    },
    "cutoff_2023": {
        "ckpt":   os.path.join(OUT,  "rolling/cutoff_2023/hazard_best.pt"),
        "scaler": os.path.join(BASE, "data/sequences_rolling/cutoff_2023/scaler.pkl"),
        "calib":  os.path.join(OUT,  "rolling/cutoff_2023/hazard_calibration.json"),
        "dead_cols": [],
    },
}

MAX_SEQ    = 33
N_FEATURES = 9
GFEE       = 0.50   # (A) MUST match realized_cpr_v5 bucketing
COUPONS    = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

# Representative loan. loan_purpose_enc=0 (first category, e.g. R),
# property_type_enc=0 (SF). For rolling models these pass through the scaler;
# for production they are zeroed post-scale via dead_cols.
REP = dict(credit_score=740.0, orig_ltv=75.0, current_ltv=70.0,
           orig_upb=250000.0, dti=35.0,
           loan_purpose_enc=0.0, property_type_enc=0.0)


def get_model_key(date: pd.Timestamp) -> str:
    if pd.Timestamp("2021-01-01") <= date <= pd.Timestamp("2021-12-01"):
        return "cutoff_2020"
    elif pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2022-12-01"):
        return "cutoff_2021"
    elif pd.Timestamp("2023-01-01") <= date <= pd.Timestamp("2023-12-01"):
        return "cutoff_2022"
    elif pd.Timestamp("2024-01-01") <= date <= pd.Timestamp("2024-12-01"):
        return "cutoff_2023"
    else:
        return "production"


# ── Model architecture (must match training) ──────────────────────────────────
class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, n_heads=4, n_layers=2,
                 dim_ff=256, max_seq=33, dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
              dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer   = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier    = nn.Sequential(
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


def load_model_from_ckpt(ckpt_path):
    ck  = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("config", {})
    m   = PrepaymentTransformer(
        input_dim=N_FEATURES, d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.1))
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def build_batch_constant_refi(refi_incentive, n_paths=1):
    s = np.zeros((n_paths, MAX_SEQ, N_FEATURES), dtype=np.float32)
    s[:, :, 0] = refi_incentive
    s[:, :, 1] = REP["credit_score"]
    s[:, :, 2] = REP["orig_ltv"]
    s[:, :, 3] = REP["current_ltv"]
    s[:, :, 4] = REP["orig_upb"]
    s[:, :, 5] = np.arange(1, MAX_SEQ + 1)[None, :]   # loan_age 1..33
    s[:, :, 6] = REP["dti"]
    s[:, :, 7] = REP["loan_purpose_enc"]
    s[:, :, 8] = REP["property_type_enc"]
    return s


def forecast_cpr(refi_incentive, model, scaler, a, b, dead_cols, n_paths=1):
    seqs = build_batch_constant_refi(refi_incentive, n_paths)
    flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(
               n_paths, MAX_SEQ, N_FEATURES)
    for c in dead_cols:                 # (B) only zero where genuinely dead
        flat[:, :, c] = 0.0
    x    = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(n_paths, MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        logit = model(x, mask=mask, return_per_timestep=True).numpy()
    smm = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    cpr = 1.0 - (1.0 - smm) ** 12
    return float(cpr.mean())


def main():
    # Load production calibration once (the fallback for any model lacking its own)
    if not os.path.exists(PROD_CALIB):
        raise RuntimeError(f"Production calibration not found: {PROD_CALIB}")
    prod_cal = json.load(open(PROD_CALIB))
    a_prod, b_prod = prod_cal["a"], prod_cal["b"]
    print(f"Production Platt: a={a_prod:.4f} b={b_prod:.4f}", flush=True)

    # Pre-load every available model with its scaler + calibration
    print("Loading model checkpoints...", flush=True)
    loaded = {}
    for key, p in MODELS.items():
        if not os.path.exists(p["ckpt"]):
            print(f"  WARNING: {key} ckpt missing ({p['ckpt']}) -> will fall back "
                  f"to production for its months.", flush=True)
            continue
        if not os.path.exists(p["scaler"]):
            print(f"  WARNING: {key} scaler missing ({p['scaler']}) -> will fall "
                  f"back to production for its months.", flush=True)
            continue
        # Per-model calibration, else production fallback (loud)
        if os.path.exists(p["calib"]):
            cal = json.load(open(p["calib"]))
            a_k, b_k = cal["a"], cal["b"]
            cal_src = "own"
        else:
            a_k, b_k = a_prod, b_prod
            cal_src = "PRODUCTION-FALLBACK"
            if key != "production":
                print(f"  *** WARNING: {key} has no own calibration -> using "
                      f"production a/b. CPR LEVELS for {key} may be biased; "
                      f"recompute Platt on the {key} test set for publication "
                      f"levels. Directional tracking still valid. ***", flush=True)
        loaded[key] = dict(model=load_model_from_ckpt(p["ckpt"]),
                           scaler=pickle.load(open(p["scaler"], "rb")),
                           a=a_k, b=b_k, dead_cols=p["dead_cols"])
        print(f"  Loaded {key:12s} calib={cal_src:20s} dead_cols={p['dead_cols']}",
              flush=True)

    if "production" not in loaded:
        raise RuntimeError("Production model not found - cannot proceed.")

    def resolve(date):
        key = get_model_key(date)
        if key not in loaded:
            key = "production"   # fallback if rolling ckpt absent
        return key, loaded[key]

    # PMMS time series
    pmms_df = pd.read_csv(os.path.join(DATA, "pmms_monthly.csv"))

    def parse_period(p):
        s = str(int(p))
        if len(s) == 5:
            return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
        elif len(s) == 6:
            return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
        return pd.NaT

    pmms_df["date"] = pmms_df["reporting_period"].apply(parse_period)
    pmms_df = pmms_df.dropna(subset=["date"]).sort_values("date")  # sort on Timestamp

    pmms_hist = pmms_df[
        (pmms_df["date"] >= pd.Timestamp("2020-01-01")) &
        (pmms_df["date"] <= pd.Timestamp("2024-12-01"))
    ].reset_index(drop=True)

    print(f"\nForecasting {len(pmms_hist)} months "
          f"({pmms_hist['date'].min().date()} -> {pmms_hist['date'].max().date()})",
          flush=True)
    print("OOS: 2021(cutoff_2020) 2022(cutoff_2021) 2023(cutoff_2022) 2024(cutoff_2023); 2020 = production baseline",
          flush=True)

    rows = []
    last_key = None
    for _, row in pmms_hist.iterrows():
        date   = row["date"]
        pmms_t = float(row["rate_30yr"])
        key, M = resolve(date)
        if key != last_key:
            print(f"\n  [{date.strftime('%b %Y')}] -> {key} model", flush=True)
            last_key = key

        for coupon in COUPONS:
            note_rate      = coupon + GFEE
            refi_incentive = note_rate - pmms_t
            cpr = forecast_cpr(refi_incentive, M["model"], M["scaler"],
                               M["a"], M["b"], M["dead_cols"], n_paths=1)
            rows.append(dict(
                date=date, coupon=coupon, model_used=key,
                note_rate=round(note_rate, 3), pmms=round(pmms_t, 4),
                refi_incentive=round(refi_incentive, 4),
                forecast_cpr=round(cpr, 6),
                is_oos=(key != "production"),
            ))

        if date.month in (1, 6, 12):
            ex = [r for r in rows if r["date"] == date and r["coupon"] == 3.5][0]
            print(f"  {date.strftime('%b %Y')}  PMMS={pmms_t:.2f}%  "
                  f"model={key}  FNCL3.5 CPR={ex['forecast_cpr']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "rolling_forecast_cpr_timeseries.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)", flush=True)

    # ── Compare vs realized v5 ────────────────────────────────────────────────
    real_path = os.path.join(OUT, "realized_cpr_by_coupon_v5.csv")
    if not os.path.exists(real_path):
        print(f"WARNING: {real_path} not found - skipping comparison.", flush=True)
        return

    real = pd.read_csv(real_path)
    real["date"] = pd.to_datetime(real["date"])
    df["date"]   = pd.to_datetime(df["date"])
    real_sub = real[real["implied_mbs_coupon"].isin(COUPONS)].rename(
        columns={"implied_mbs_coupon": "coupon", "cpr": "realized_cpr"})
    merged = df.merge(real_sub[["date", "coupon", "realized_cpr"]],
                      on=["date", "coupon"], how="inner")
    print(f"\nMerged obs: {len(merged)} (date x coupon)", flush=True)

    def block(label, lo, hi):
        sub = merged[merged["date"].between(lo, hi)]
        print(f"\n=== {label} ===")
        if not len(sub):
            print("  (no overlap with realized CPR)"); return
        g = sub.groupby("coupon").agg(
            forecast=("forecast_cpr", "mean"),
            realized=("realized_cpr", "mean")).round(4)
        g["ratio"] = (g["forecast"] / g["realized"].replace(0, np.nan)).round(3)
        print(g)

    block("2020 baseline (production, in-sample)", "2020-01-01", "2020-12-31")
    block("2021 OOS (cutoff_2020 model)",          "2021-01-01", "2021-12-31")
    block("2022 OOS (cutoff_2021 model)",          "2022-01-01", "2022-12-31")
    block("2023 OOS (cutoff_2022 model)",          "2023-01-01", "2023-12-31")
    block("2024 OOS (cutoff_2023 model)",          "2024-01-01", "2024-12-31")

    merged_path = os.path.join(OUT, "rolling_forecast_vs_realized.csv")
    merged.to_csv(merged_path, index=False)
    print(f"\nSaved: {merged_path}\nDone.", flush=True)


if __name__ == "__main__":
    main()
