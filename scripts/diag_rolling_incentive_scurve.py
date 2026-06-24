"""
Rolling-model incentive S-curve diagnostic.

PURPOSE
  Determine whether the 2022 cross-sectional CPR inversion (rolling forecast
  giving HIGHER prepay to out-of-money low coupons than in-money high coupons)
  reflects a genuinely inverted incentive response learned by the model, as
  opposed to (a) missing rolling Platt calibration or (b) burnout triggered by
  the constant-incentive forecast construction.

WHY THIS IS CALIBRATION-INDEPENDENT
  Platt calibration is sigmoid(a*logit + b) with a > 0 -> strictly monotonic.
  It can rescale hazard LEVELS but cannot reverse the ORDERING of hazard vs
  incentive. So the SHAPE verdict below uses the RAW model output (logit and
  sigmoid(logit)) with NO calibration. The absence of rolling Platt fits does
  not affect whether the curve slopes up (correct S-curve) or down (inverted).

WHY WE ALSO SHOW AGE x INCENTIVE
  Reading the per-timestep hazard across loan age (1..33) reveals whether any
  suppression is uniform (true inversion) or concentrated at late ages (burnout
  from feeding 33 months of constant incentive). A late-age-only dip is burnout,
  not a wrong incentive response.

OUTPUTS
  outputs/diag_rolling_incentive_scurve.png   (3-model S-curve overlay, LTV=70)
  outputs/diag_rolling_incentive_scurve.csv   (the sweep table)
  outputs/diag_rolling_age_incentive.png      (age x incentive heatmaps)
  outputs/diag_rolling_equity_incentive.png   (incentive x current-LTV heatmaps)

NOTE ON CATEGORICALS
  Production model: loan_purpose_enc / property_type_enc were dead (all-zero)
  -> zeroed post-scale (dead_cols=[7,8]).
  Rolling models: those categoricals are LIVE (R/C/P, SF/PU/CO/MH)
  -> NOT zeroed (dead_cols=[]).
"""

import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")

MAX_SEQ    = 33
N_FEATURES = 9

MODELS = {
    "production": dict(
        ckpt=os.path.join(OUT,  "hazard_best.pt"),
        scaler=os.path.join(BASE, "data/sequences/scaler.pkl"),
        dead_cols=[7, 8]),
    "cutoff_2020": dict(
        ckpt=os.path.join(OUT,  "rolling/cutoff_2020/hazard_best.pt"),
        scaler=os.path.join(BASE, "data/sequences_rolling/cutoff_2020/scaler.pkl"),
        dead_cols=[]),
    "cutoff_2021": dict(
        ckpt=os.path.join(OUT,  "rolling/cutoff_2021/hazard_best.pt"),
        scaler=os.path.join(BASE, "data/sequences_rolling/cutoff_2021/scaler.pkl"),
        dead_cols=[]),
    "cutoff_2022": dict(
        ckpt=os.path.join(OUT,  "rolling/cutoff_2022/hazard_best.pt"),
        scaler=os.path.join(BASE, "data/sequences_rolling/cutoff_2022/scaler.pkl"),
        dead_cols=[]),
    "cutoff_2023": dict(
        ckpt=os.path.join(OUT,  "rolling/cutoff_2023/hazard_best.pt"),
        scaler=os.path.join(BASE, "data/sequences_rolling/cutoff_2023/scaler.pkl"),
        dead_cols=[]),
}

# Representative loan (matches REP used in the forecast scripts).
REP = dict(credit_score=740.0, orig_ltv=75.0, current_ltv=70.0,
           orig_upb=250000.0, dti=35.0,
           loan_purpose_enc=0.0, property_type_enc=0.0)

INCENTIVE_GRID = np.round(np.arange(-2.0, 4.01, 0.25), 4)        # pp
LTV_GRID       = np.arange(30, 131, 10)                          # current LTV
INCENTIVE_COARSE = np.round(np.arange(-2.0, 4.01, 0.5), 4)       # for heatmaps


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


def load_model(ckpt_path):
    ck  = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("config", {})
    m   = PrepaymentTransformer(
        input_dim=N_FEATURES, d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.1))
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def build_seq(incentive, current_ltv):
    """One representative loan, constant incentive across all 33 months."""
    s = np.zeros((1, MAX_SEQ, N_FEATURES), dtype=np.float32)
    s[:, :, 0] = incentive
    s[:, :, 1] = REP["credit_score"]
    s[:, :, 2] = REP["orig_ltv"]
    s[:, :, 3] = current_ltv                 # equity variable
    s[:, :, 4] = REP["orig_upb"]
    s[:, :, 5] = np.arange(1, MAX_SEQ + 1)[None, :]
    s[:, :, 6] = REP["dti"]
    s[:, :, 7] = REP["loan_purpose_enc"]
    s[:, :, 8] = REP["property_type_enc"]
    return s


def per_timestep_logit(model, scaler, dead_cols, incentive, current_ltv):
    """Return raw per-timestep logit (shape [33]) - NO calibration."""
    seq  = build_seq(incentive, current_ltv)
    flat = scaler.transform(seq.reshape(-1, N_FEATURES)).reshape(1, MAX_SEQ, N_FEATURES)
    for c in dead_cols:
        flat[:, :, c] = 0.0
    x    = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(1, MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        logit = model(x, mask=mask, return_per_timestep=True).numpy()[0]  # (33,)
    return logit


def mean_hazard(model, scaler, dead_cols, incentive, current_ltv):
    """Mean raw hazard sigmoid(logit) over the 33 timesteps (drives forecast CPR)."""
    logit = per_timestep_logit(model, scaler, dead_cols, incentive, current_ltv)
    return float(np.mean(1.0 / (1.0 + np.exp(-logit))))


def main():
    loaded = {}
    for key, p in MODELS.items():
        if not (os.path.exists(p["ckpt"]) and os.path.exists(p["scaler"])):
            print(f"WARNING: {key} missing ckpt/scaler -> skipped", flush=True)
            continue
        loaded[key] = dict(model=load_model(p["ckpt"]),
                           scaler=pickle.load(open(p["scaler"], "rb")),
                           dead_cols=p["dead_cols"])
        print(f"Loaded {key} (dead_cols={p['dead_cols']})", flush=True)
    if not loaded:
        raise RuntimeError("No models loaded.")

    # ── (1) Incentive S-curve overlay, fixed current_ltv = REP value ──────────
    ltv0 = REP["current_ltv"]
    rows = []
    for key, M in loaded.items():
        for inc in INCENTIVE_GRID:
            h = mean_hazard(M["model"], M["scaler"], M["dead_cols"], inc, ltv0)
            rows.append(dict(model=key, incentive=inc, mean_hazard=h))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(OUT, "diag_rolling_incentive_scurve.csv"), index=False)

    # Monotonicity verdict per model (Spearman-free: sign of net change + #drops)
    print(f"\n=== Incentive response (mean raw hazard, current_ltv={ltv0}) ===")
    for key in loaded:
        s = sweep[sweep.model == key].sort_values("incentive")
        d = np.diff(s["mean_hazard"].values)
        net = s["mean_hazard"].values[-1] - s["mean_hazard"].values[0]
        frac_up = float(np.mean(d > 0))
        verdict = ("S-CURVE (rises with incentive)" if net > 0 and frac_up > 0.6
                   else "INVERTED (falls with incentive)" if net < 0 and frac_up < 0.4
                   else "FLAT / NON-MONOTONIC")
        print(f"  {key:12s} net Δ={net:+.4f}  frac-steps-up={frac_up:.2f}  -> {verdict}")

    plt.figure(figsize=(8, 5))
    for key in loaded:
        s = sweep[sweep.model == key].sort_values("incentive")
        plt.plot(s["incentive"], s["mean_hazard"], marker="o", ms=3, label=key)
    plt.axvline(0, color="grey", lw=0.8, ls="--")
    plt.xlabel("Rate incentive (note rate − PMMS, pp)")
    plt.ylabel("Mean monthly hazard (raw, uncalibrated)")
    plt.title(f"Incentive response by model  (current LTV={ltv0:.0f})\n"
              "raw σ(logit); Platt is monotonic and cannot change direction")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "diag_rolling_incentive_scurve.png"), dpi=140)
    plt.close()

    # ── (2) Age × incentive heatmap (per-timestep) to separate burnout ────────
    keys_for_age = [k for k in ("production","cutoff_2021","cutoff_2022","cutoff_2023") if k in loaded]
    fig, axes = plt.subplots(1, len(keys_for_age), figsize=(6*len(keys_for_age), 5),
                             squeeze=False)
    for ax, key in zip(axes[0], keys_for_age):
        M = loaded[key]
        Z = np.zeros((MAX_SEQ, len(INCENTIVE_COARSE)))
        for j, inc in enumerate(INCENTIVE_COARSE):
            lg = per_timestep_logit(M["model"], M["scaler"], M["dead_cols"], inc, ltv0)
            Z[:, j] = 1.0 / (1.0 + np.exp(-lg))
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis",
                       extent=[INCENTIVE_COARSE[0], INCENTIVE_COARSE[-1], 1, MAX_SEQ])
        ax.set_xlabel("Rate incentive (pp)"); ax.set_ylabel("Loan age (months)")
        ax.set_title(f"{key}: per-timestep hazard")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Age × incentive raw hazard (current LTV={ltv0:.0f})  "
                 "— uniform vs late-age dip distinguishes inversion from burnout")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diag_rolling_age_incentive.png"), dpi=140)
    plt.close(fig)

    # ── (3) Incentive × current-LTV heatmap (equity gate) ─────────────────────
    keys_for_eq = [k for k in ("production","cutoff_2021","cutoff_2022","cutoff_2023") if k in loaded]
    fig, axes = plt.subplots(1, len(keys_for_eq), figsize=(6*len(keys_for_eq), 5),
                             squeeze=False)
    for ax, key in zip(axes[0], keys_for_eq):
        M = loaded[key]
        Z = np.zeros((len(LTV_GRID), len(INCENTIVE_COARSE)))
        for i, ltv in enumerate(LTV_GRID):
            for j, inc in enumerate(INCENTIVE_COARSE):
                Z[i, j] = mean_hazard(M["model"], M["scaler"], M["dead_cols"], inc, ltv)
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="magma",
                       extent=[INCENTIVE_COARSE[0], INCENTIVE_COARSE[-1],
                               LTV_GRID[0], LTV_GRID[-1]])
        ax.axhline(100, color="cyan", lw=1, ls="--")  # underwater threshold
        ax.set_xlabel("Rate incentive (pp)"); ax.set_ylabel("Current LTV")
        ax.set_title(f"{key}: mean hazard")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Equity × incentive (mean raw hazard) — gate should suppress "
                 "refi above LTV≈100")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diag_rolling_equity_incentive.png"), dpi=140)
    plt.close(fig)

    print("\nSaved:")
    for f in ("diag_rolling_incentive_scurve.png", "diag_rolling_incentive_scurve.csv",
              "diag_rolling_age_incentive.png", "diag_rolling_equity_incentive.png"):
        print("  outputs/" + f)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
