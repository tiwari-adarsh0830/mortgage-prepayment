"""
DDPM for mortgage rate simulation
Trains on monthly PMMS rate changes using sliding windows
Generates synthetic rate paths for prepayment modeling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE      = "/scratch/at7095/mortgage_prepayment"
OUTPUTS   = os.path.join(BASE, "outputs")
PMMS_PATH = os.path.join(BASE, "data/pmms_monthly.csv")

SEQ_LEN    = 33
T_STEPS    = 200
N_EPOCHS   = 2000
BATCH_SIZE = 64
LR         = 1e-3
N_GENERATE = 1000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def prepare_data(pmms_path, seq_len):
    pmms         = pd.read_csv(pmms_path)
    rates        = pmms['rate_30yr'].values.astype(np.float32)
    changes      = np.diff(rates)
    mean         = changes.mean()
    std          = changes.std()
    changes_norm = (changes - mean) / std
    n_windows    = len(changes_norm) - seq_len + 1
    windows      = np.array([changes_norm[i:i+seq_len] for i in range(n_windows)])
    print(f"Training windows: {windows.shape}")
    print(f"Rate change mean={mean:.4f}, std={std:.4f}")
    return windows, mean, std, rates


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas      = torch.linspace(beta_start, beta_end, T)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t, alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    ab  = alpha_bars.to(t.device)[t].view(-1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -torch.arange(half, device=device) * np.log(10000) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DenoiseNet(nn.Module):
    """
    MLP-based denoiser for 1D sequences.
    Avoids UNet dimension mismatch issues with seq_len=33.
    Input: (batch, seq_len) noisy sequence + timestep embedding
    Output: (batch, seq_len) predicted noise
    """
    def __init__(self, seq_len=33, time_dim=64, hidden_dim=512):
        super().__init__()
        self.time_emb = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(seq_len + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, seq_len)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_emb(t))
        inp   = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


def train(model, windows, betas, alphas, alpha_bars, n_epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    dataset   = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    T         = len(betas)
    losses    = []

    for epoch in range(n_epochs):
        idx        = torch.randint(0, len(dataset), (batch_size,))
        x0         = dataset[idx]
        t          = torch.randint(0, T, (batch_size,), device=DEVICE)
        xt, noise  = q_sample(x0, t, alpha_bars)
        pred_noise = model(xt, t)
        loss       = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={np.mean(losses[-50:]):.5f}")

    return losses


@torch.no_grad()
def sample(model, n_samples, seq_len, betas, alphas, alpha_bars):
    model.eval()
    T  = len(betas)
    xt = torch.randn(n_samples, seq_len, device=DEVICE)

    for t_idx in reversed(range(T)):
        t_batch    = torch.full((n_samples,), t_idx, device=DEVICE, dtype=torch.long)
        pred_noise = model(xt, t_batch)
        beta_t     = betas[t_idx].to(DEVICE)
        alpha_t    = alphas[t_idx].to(DEVICE)
        ab_t       = alpha_bars[t_idx].to(DEVICE)
        mean       = (1 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1 - ab_t)) * pred_noise
        )
        if t_idx > 0:
            ab_prev = alpha_bars[t_idx - 1].to(DEVICE)
            var     = beta_t * (1 - ab_prev) / (1 - ab_t)
            xt      = mean + torch.sqrt(var) * torch.randn_like(xt)
        else:
            xt = mean

    return xt.cpu().numpy()


def changes_to_rates(change_samples, start_rate, mean, std):
    changes     = change_samples * std + mean
    rates       = np.zeros((len(changes), SEQ_LEN + 1))
    rates[:, 0] = start_rate
    for t in range(SEQ_LEN):
        rates[:, t+1] = rates[:, t] + changes[:, t]
    return np.maximum(rates, 0.5)


def main():
    print(f"Device: {DEVICE}")

    windows, mean, std, all_rates = prepare_data(PMMS_PATH, SEQ_LEN)
    start_rate = all_rates[-1]
    print(f"Starting rate: {start_rate:.4f}%")

    betas, alphas, alpha_bars = make_beta_schedule(T_STEPS)

    model    = DenoiseNet(seq_len=SEQ_LEN).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nTraining DDPM for {N_EPOCHS} epochs...")
    losses = train(model, windows, betas, alphas, alpha_bars,
                   N_EPOCHS, BATCH_SIZE, LR)

    torch.save({
        'model_state': model.state_dict(),
        'mean': mean, 'std': std,
        'config': {'seq_len': SEQ_LEN, 'T_steps': T_STEPS}
    }, os.path.join(OUTPUTS, 'ddpm_rate.pt'))
    print("Model saved.")

    # Loss plot — smoothed
    fig, ax = plt.subplots(figsize=(8, 4))
    window   = 20
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='steelblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (smoothed)')
    ax.set_title('DDPM Training Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, 'ddpm_training_loss.png'), dpi=150)
    plt.close()
    print("Training loss plot saved.")

    # Generate paths
    print(f"\nGenerating {N_GENERATE} rate paths...")
    change_samples = sample(model, N_GENERATE, SEQ_LEN, betas, alphas, alpha_bars)
    rate_paths     = changes_to_rates(change_samples, start_rate, mean, std)
    np.save(os.path.join(OUTPUTS, 'ddpm_rate_paths.npy'), rate_paths)
    print(f"Rate paths saved: {rate_paths.shape}")

    # Fan chart
    months = np.arange(SEQ_LEN + 1)
    p10    = np.percentile(rate_paths, 10, axis=0)
    p25    = np.percentile(rate_paths, 25, axis=0)
    p50    = np.percentile(rate_paths, 50, axis=0)
    p75    = np.percentile(rate_paths, 75, axis=0)
    p90    = np.percentile(rate_paths, 90, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(months, p10, p90, alpha=0.15, color='steelblue', label='10-90%')
    ax.fill_between(months, p25, p75, alpha=0.30, color='steelblue', label='25-75%')
    ax.plot(months, p50, color='steelblue', linewidth=2, label='Median')
    ax.axhline(start_rate, color='red', linestyle='--', linewidth=1,
               label=f'Start rate {start_rate:.2f}%')
    ax.set_xlabel('Months ahead')
    ax.set_ylabel('30yr Mortgage Rate (%)')
    ax.set_title('DDPM Generated Rate Paths (1000 paths)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, 'ddpm_rate_paths.png'), dpi=150)
    plt.close()
    print("Fan chart saved.")

    # Distribution comparison histogram
    real_changes = np.diff(all_rates)
    gen_changes  = (change_samples * std + mean).flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_changes, bins=50, alpha=0.5, color='steelblue',
            density=True, label='Real PMMS changes')
    ax.hist(gen_changes,  bins=50, alpha=0.5, color='tomato',
            density=True, label='DDPM generated changes')
    ax.set_xlabel('Monthly rate change (%)')
    ax.set_ylabel('Density')
    ax.set_title('Real vs Generated Rate Change Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, 'ddpm_distribution_comparison.png'), dpi=150)
    plt.close()
    print("Distribution comparison saved.")

    # Console summary
    print(f"\n── Generated path stats at month 33 ──")
    print(f"  median={p50[-1]:.3f}%  p10={p10[-1]:.3f}%  p90={p90[-1]:.3f}%")

    real_changes_arr = np.diff(all_rates)
    gen_changes_arr  = change_samples * std + mean
    print(f"\n── Rate change distribution comparison ──")
    print(f"  Real:      mean={real_changes_arr.mean():.4f}  std={real_changes_arr.std():.4f}  "
          f"min={real_changes_arr.min():.4f}  max={real_changes_arr.max():.4f}")
    print(f"  Generated: mean={gen_changes_arr.mean():.4f}  std={gen_changes_arr.std():.4f}  "
          f"min={gen_changes_arr.min():.4f}  max={gen_changes_arr.max():.4f}")


if __name__ == "__main__":
    main()
