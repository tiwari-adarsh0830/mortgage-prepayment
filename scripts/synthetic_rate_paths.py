"""
Synthetic Rate Path Analysis
Hull-White rate simulation → synthetic loan sequences → Transformer prepayment probabilities
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.optimize import minimize
import os

BASE        = "/scratch/at7095/mortgage_prepayment"
OUTPUTS     = os.path.join(BASE, "outputs")
CKPT_PATH   = os.path.join(OUTPUTS, "transformer_best.pt")
PMMS_PATH   = os.path.join(BASE, "data/pmms_monthly.csv")
SCALER_PATH = os.path.join(BASE, "data/sequences/scaler.pkl")

MAX_SEQ    = 33
N_FEATURES = 6
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PATHS    = 1000
SEED       = 42

FIXED_CREDIT_SCORE = 750.0
FIXED_ORIG_LTV     = 80.0
FIXED_CURRENT_LTV  = 80.0
FIXED_ORIG_UPB     = 300000.0

ORIG_RATES  = [3.0, 5.0, 7.0]
START_RATES = [3.0, 5.0, 7.0]


class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=64, n_heads=4, n_layers=2,
                 dim_ff=256, max_seq=33, dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, src_key_padding_mask=None):
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out = self.input_proj(x) + self.pos_embedding(positions)
        out = self.transformer(out, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            real = (~src_key_padding_mask).float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)


def load_model(ckpt_path):
    ckpt   = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt["config"]
    model  = PrepaymentTransformer(
        n_heads=config["n_heads"], n_layers=config["n_layers"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"Model loaded: {config}")
    return model


def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded: {type(scaler)}")
    # Print scaler stats so we can verify feature order
    if hasattr(scaler, 'mean_'):
        print(f"Scaler means:  {scaler.mean_}")
        print(f"Scaler stds:   {scaler.scale_}")
    return scaler


def calibrate_hull_white(rates):
    dt = 1/12
    r  = np.array(rates)

    def neg_log_likelihood(params):
        a, sigma = params
        if a <= 0 or sigma <= 0:
            return 1e10
        e     = np.exp(-a * dt)
        mu    = r[:-1] * e
        var   = sigma**2 * (1 - e**2) / (2 * a)
        resid = r[1:] - mu
        ll    = -0.5 * (np.log(2 * np.pi * var) + resid**2 / var)
        return -ll.sum()

    result = minimize(neg_log_likelihood, x0=[0.5, 0.5],
                      bounds=[(0.01, 5.0), (0.001, 2.0)],
                      method='L-BFGS-B')
    a, sigma = result.x
    print(f"Hull-White: a={a:.4f}, sigma={sigma:.4f}")
    return a, sigma


def simulate_hull_white(r0, a, sigma, theta, n_paths, n_steps, dt=1/12):
    np.random.seed(SEED)
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0
    for t in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        rates[:, t+1] = (rates[:, t]
                         + a * (theta - rates[:, t]) * dt
                         + sigma * dW)
        rates[:, t+1] = np.maximum(rates[:, t+1], 0.5)
    return rates  # (n_paths, n_steps+1)


def build_sequences(orig_rate, simulated_rates, scaler):
    """
    Build raw sequences then scale using the training scaler.
    simulated_rates: (n_paths, MAX_SEQ)
    Feature order: refi_incentive, borrower_credit_score, original_ltv,
                   current_ltv, original_upb, loan_age_months
    """
    n_paths = simulated_rates.shape[0]
    seqs    = np.zeros((n_paths, MAX_SEQ, N_FEATURES), dtype=np.float32)

    for t in range(MAX_SEQ):
        seqs[:, t, 0] = orig_rate - simulated_rates[:, t]  # refi_incentive
        seqs[:, t, 1] = FIXED_CREDIT_SCORE
        seqs[:, t, 2] = FIXED_ORIG_LTV
        seqs[:, t, 3] = FIXED_CURRENT_LTV
        seqs[:, t, 4] = FIXED_ORIG_UPB
        seqs[:, t, 5] = float(t + 1)  # loan_age_months

    # Apply scaler — reshape to 2D, scale, reshape back
    shape   = seqs.shape
    seqs_2d = seqs.reshape(-1, N_FEATURES)
    seqs_2d = scaler.transform(seqs_2d)
    seqs    = seqs_2d.reshape(shape).astype(np.float32)
    return seqs


def get_prepayment_probs_over_time(model, seqs):
    """
    For each prefix t=1..MAX_SEQ, run Transformer on sequence up to t.
    Returns: (n_paths, MAX_SEQ) probabilities
    """
    n_paths    = seqs.shape[0]
    probs      = np.zeros((n_paths, MAX_SEQ), dtype=np.float32)
    batch_size = 256

    for t in range(1, MAX_SEQ + 1):
        padded      = np.zeros((n_paths, MAX_SEQ, N_FEATURES), dtype=np.float32)
        padded[:, :t, :] = seqs[:, :t, :]
        mask        = np.ones((n_paths, MAX_SEQ), dtype=bool)
        mask[:, :t] = False

        all_p = []
        for i in range(0, n_paths, batch_size):
            x = torch.tensor(padded[i:i+batch_size], device=DEVICE)
            m = torch.tensor(mask[i:i+batch_size],   device=DEVICE)
            with torch.no_grad():
                logits = model(x, src_key_padding_mask=m)
                p      = torch.sigmoid(logits).cpu().numpy()
            all_p.append(p)

        probs[:, t-1] = np.concatenate(all_p)
        if t % 5 == 0:
            print(f"  timestep {t}/{MAX_SEQ} done")

    return probs


def main():
    print(f"Device: {DEVICE}")

    pmms  = pd.read_csv(PMMS_PATH)
    mask  = (pmms['year'] >= 2020) & (pmms['year'] <= 2023)
    rates = pmms[mask]['rate_30yr'].values
    print(f"Calibrating Hull-White on {len(rates)} observations (2020-2023)")
    a, sigma = calibrate_hull_white(rates)
    theta    = rates.mean()
    print(f"theta (long-run mean): {theta:.4f}%")

    scaler = load_scaler(SCALER_PATH)
    model  = load_model(CKPT_PATH)

    all_probs     = {}
    all_sim_rates = {}

    for orig_rate, start_rate in zip(ORIG_RATES, START_RATES):
        print(f"\n── orig_rate={orig_rate}%, start_rate={start_rate}% ──")

        sim_rates = simulate_hull_white(
            r0=start_rate, a=a, sigma=sigma, theta=theta,
            n_paths=N_PATHS, n_steps=MAX_SEQ-1
        )  # (1000, 34) — take first 33

        seqs  = build_sequences(orig_rate, sim_rates[:, :MAX_SEQ], scaler)
        print("Running Transformer over timesteps...")
        probs = get_prepayment_probs_over_time(model, seqs)

        all_probs[(orig_rate, start_rate)]     = probs
        all_sim_rates[(orig_rate, start_rate)] = sim_rates[:, :MAX_SEQ]

    months = np.arange(1, MAX_SEQ + 1)

    # ── Plot 1: Rate fan charts ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (orig_rate, start_rate) in zip(axes, zip(ORIG_RATES, START_RATES)):
        sr   = all_sim_rates[(orig_rate, start_rate)]
        p10  = np.percentile(sr, 10, axis=0)
        p25  = np.percentile(sr, 25, axis=0)
        p50  = np.percentile(sr, 50, axis=0)
        p75  = np.percentile(sr, 75, axis=0)
        p90  = np.percentile(sr, 90, axis=0)
        ax.fill_between(months, p10, p90, alpha=0.15, color='steelblue', label='10-90%')
        ax.fill_between(months, p25, p75, alpha=0.30, color='steelblue', label='25-75%')
        ax.plot(months, p50, color='steelblue', linewidth=2, label='Median')
        ax.axhline(orig_rate, color='red', linestyle='--', linewidth=1,
                   label=f'Orig rate {orig_rate}%')
        ax.set_title(f'Start rate {start_rate}%  (orig={orig_rate}%)', fontsize=11)
        ax.set_xlabel('Loan Age (months)')
        ax.set_ylabel('30yr Mortgage Rate (%)')
        ax.legend(fontsize=7)
        ax.set_ylim(0, 12)
    plt.suptitle('Hull-White Simulated Rate Paths (1000 paths each)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, 'synthetic_rate_paths.png'), dpi=150)
    plt.close()
    print("Saved synthetic_rate_paths.png")

    # ── Plot 2: Prepayment probability fan charts ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (orig_rate, start_rate) in zip(axes, zip(ORIG_RATES, START_RATES)):
        probs = all_probs[(orig_rate, start_rate)]
        p10   = np.percentile(probs, 10, axis=0)
        p25   = np.percentile(probs, 25, axis=0)
        p50   = np.percentile(probs, 50, axis=0)
        p75   = np.percentile(probs, 75, axis=0)
        p90   = np.percentile(probs, 90, axis=0)
        ax.fill_between(months, p10, p90, alpha=0.15, color='tomato', label='10-90%')
        ax.fill_between(months, p25, p75, alpha=0.30, color='tomato', label='25-75%')
        ax.plot(months, p50, color='tomato', linewidth=2, label='Median')
        ax.set_title(f'orig={orig_rate}%, start={start_rate}%', fontsize=11)
        ax.set_xlabel('Loan Age (months)')
        ax.set_ylabel('Prepayment Probability')
        ax.legend(fontsize=7)
    plt.suptitle('Transformer Prepayment Probability — Synthetic Rate Paths', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, 'synthetic_prepay_probs.png'), dpi=150)
    plt.close()
    print("Saved synthetic_prepay_probs.png")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── Median prepayment probability at month 33 ──")
    for orig_rate, start_rate in zip(ORIG_RATES, START_RATES):
        median = np.median(all_probs[(orig_rate, start_rate)][:, -1])
        print(f"  orig={orig_rate}%, start={start_rate}%  →  {median:.4f}")

    # ── Sanity check: verify scaler was applied correctly ─────────────────────
    print("\n── Scaler sanity check (first path, first timestep, unscaled refi) ──")
    for orig_rate, start_rate in zip(ORIG_RATES, START_RATES):
        raw_refi = orig_rate - start_rate
        print(f"  orig={orig_rate}%, start={start_rate}%  raw_refi={raw_refi:.2f}")


if __name__ == "__main__":
    main()
