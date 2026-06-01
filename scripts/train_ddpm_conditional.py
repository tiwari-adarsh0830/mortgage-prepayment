"""
Conditional DDPM for mortgage rate simulation
Conditions on starting rate level so generated paths are realistic
given today's rate environment.
Key change vs unconditional: DenoiseNet takes start_rate as extra input.
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

    n_windows  = len(changes_norm) - seq_len + 1
    windows    = np.array([changes_norm[i:i+seq_len] for i in range(n_windows)])

    # Starting rate for each window (rate before first change)
    # Normalize start rates for stable training
    start_rates     = rates[:n_windows]
    rate_mean       = start_rates.mean()
    rate_std        = start_rates.std()
    start_rates_norm = (start_rates - rate_mean) / rate_std

    print(f"Training windows: {windows.shape}")
    print(f"Rate change: mean={mean:.4f}, std={std:.4f}")
    print(f"Start rates: mean={rate_mean:.4f}%, std={rate_std:.4f}%")
    print(f"Today's rate: {rates[-1]:.4f}%")

    return windows, mean, std, rates, start_rates_norm, rate_mean, rate_std


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


class ConditionalDenoiseNet(nn.Module):
    """
    Conditional MLP denoiser.
    Conditions on start_rate (normalized scalar) in addition to diffusion timestep.
    Input: (batch, seq_len) noisy sequence
           (batch,) diffusion timestep
           (batch,) normalized starting rate
    Output: (batch, seq_len) predicted noise
    """
    def __init__(self, seq_len=33, time_dim=64, rate_dim=64, hidden_dim=512):
        super().__init__()

        # Diffusion timestep embedding (unchanged from unconditional)
        self.time_emb = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        # Starting rate conditioning embedding
        self.rate_mlp = nn.Sequential(
            nn.Linear(1, rate_dim),
            nn.SiLU(),
            nn.Linear(rate_dim, rate_dim)
        )

        # Main denoising network
        # Input: seq_len + time_dim + rate_dim
        self.net = nn.Sequential(
            nn.Linear(seq_len + time_dim + rate_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, seq_len)
        )

    def forward(self, x, t, start_rate):
        """
        x:          (B, seq_len) noisy sequence
        t:          (B,) diffusion timestep
        start_rate: (B,) normalized starting rate scalar
        """
        t_emb    = self.time_mlp(self.time_emb(t))               # (B, time_dim)
        r_emb    = self.rate_mlp(start_rate.unsqueeze(-1).float()) # (B, rate_dim)
        inp      = torch.cat([x, t_emb, r_emb], dim=-1)
        return self.net(inp)


def train(model, windows, start_rates_norm, betas, alphas, alpha_bars,
          n_epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    dataset      = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    start_rates  = torch.tensor(start_rates_norm, dtype=torch.float32).to(DEVICE)
    T            = len(betas)
    losses       = []

    for epoch in range(n_epochs):
        idx        = torch.randint(0, len(dataset), (batch_size,))
        x0         = dataset[idx]
        sr         = start_rates[idx]                              # (B,) conditioning
        t          = torch.randint(0, T, (batch_size,), device=DEVICE)
        xt, noise  = q_sample(x0, t, alpha_bars)
        pred_noise = model(xt, t, sr)
        loss       = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={np.mean(losses[-50:]):.5f}",
                  flush=True)

    return losses


@torch.no_grad()
def sample(model, n_samples, seq_len, betas, alphas, alpha_bars,
           start_rate_norm):
    """
    Generate rate change sequences conditioned on today's starting rate.
    start_rate_norm: normalized scalar (float)
    """
    model.eval()
    T  = len(betas)
    xt = torch.randn(n_samples, seq_len, device=DEVICE)

    # Broadcast conditioning to all samples
    sr = torch.full((n_samples,), start_rate_norm, device=DEVICE, dtype=torch.float32)

    for t_idx in reversed(range(T)):
        t_batch    = torch.full((n_samples,), t_idx, device=DEVICE, dtype=torch.long)
        pred_noise = model(xt, t_batch, sr)
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

    (windows, mean, std, all_rates,
     start_rates_norm, rate_mean, rate_std) = prepare_data(PMMS_PATH, SEQ_LEN)

    today_rate      = all_rates[-1]
    today_rate_norm = (today_rate - rate_mean) / rate_std
    print(f"\nToday's rate: {today_rate:.4f}% (normalized: {today_rate_norm:.4f})")

    betas, alphas, alpha_bars = make_beta_schedule(T_STEPS)

    model    = ConditionalDenoiseNet(seq_len=SEQ_LEN).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nTraining conditional DDPM for {N_EPOCHS} epochs...")
    losses = train(model, windows, start_rates_norm, betas, alphas, alpha_bars,
                   N_EPOCHS, BATCH_SIZE, LR)

    torch.save({
        'model_state':  model.state_dict(),
        'mean':         mean,
        'std':          std,
        'rate_mean':    rate_mean,
        'rate_std':     rate_std,
        'today_rate':   today_rate,
        'config': {'seq_len': SEQ_LEN, 'T_steps': T_STEPS, 'conditional': True}
    }, os.path.join(OUTPUTS, 'ddpm_conditional.pt'))
    print("Model saved.")

    # Loss plot
    fig, ax = plt.subplots(figsize=(8, 4))
    window   = 20
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='steelblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (smoothed)')
    ax.set_title('Conditional DDPM Training Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, 'ddpm_conditional_loss.png'), dpi=150)
    plt.close()

    # Generate paths conditioned on today's rate
    print(f"\nGenerating {N_GENERATE} paths conditioned on {today_rate:.2f}%...")
    change_samples = sample(model, N_GENERATE, SEQ_LEN, betas, alphas, alpha_bars,
                            today_rate_norm)
    rate_paths     = changes_to_rates(change_samples, today_rate, mean, std)
    np.save(os.path.join(OUTPUTS, 'ddpm_conditional_paths.npy'), rate_paths)
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
    ax.axhline(today_rate, color='red', linestyle='--', linewidth=1,
               label=f'Today\'s rate {today_rate:.2f}%')
    ax.set_xlabel('Months ahead')
    ax.set_ylabel('30yr Mortgage Rate (%)')
    ax.set_title(f'Conditional DDPM Rate Paths (start={today_rate:.2f}%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, 'ddpm_conditional_paths.png'), dpi=150)
    plt.close()
    print("Fan chart saved.")

    # Summary
    print(f"\n── Generated path stats ──")
    print(f"  Start:  {rate_paths[:, 0].mean():.3f}% (should be {today_rate:.3f}%)")
    print(f"  Month 12: median={np.percentile(rate_paths[:,12],50):.3f}%")
    print(f"  Month 33: median={p50[-1]:.3f}%  p10={p10[-1]:.3f}%  p90={p90[-1]:.3f}%")


if __name__ == "__main__":
    main()
