"""
OAS Engine
Monte Carlo cashflow discounting using DDPM rate paths + hazard model.
For each sampled loan x rate path:
  - Rebuild sequence with path-driven refi_incentive at each month
  - Run hazard model -> per-month prepayment probability h_t
  - Compute monthly cashflows (scheduled P+I adjusted for prepayments)
  - Discount at path rates -> price
Average across paths = model fair price per loan.
OAS = spread s such that discounting at (r_t + s) matches market price (pending from advisor).

Note: per-loan orig_rate recovered from refi_incentive at t=0 + PMMS mean 2020-2021.
Per-loan orig_rate recovered from refi_incentive at t=0 + PMMS mean.
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import json

BASE    = "/scratch/at7095/mortgage_prepayment"
SEQ_DIR = os.path.join(BASE, "data/sequences")
OUTPUTS = os.path.join(BASE, "outputs")

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ        = 33
N_FEATURES     = 9
N_LOANS_SAMPLE = 1000
N_PATHS        = 1000
# Per-loan orig_rate recovered from sequences (see main())
# PMMS mean 2020-2021 = 3.03%; orig_rate = refi_incentive_unscaled[t=0] + pmms_mean
PMMS_MEAN_2020_2021 = 3.03  # % — used to recover per-loan orig_rate


# ── Model ─────────────────────────────────────────────────────────────────────
class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, n_heads=4, n_layers=2,
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

    def forward(self, x, mask=None, return_per_timestep=False):
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out = self.input_proj(x) + self.pos_embedding(positions)
        padding_mask = ~mask if mask is not None else None
        out = self.transformer(out, src_key_padding_mask=padding_mask)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)


def load_hazard_model():
    ckpt  = torch.load(os.path.join(OUTPUTS, 'hazard_best.pt'), map_location=DEVICE)
    cfg   = ckpt['config']
    model = PrepaymentTransformer(
        n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
        d_model=cfg['d_model'], dropout=cfg['dropout']
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded hazard model (epoch {ckpt['epoch']}, AUC={ckpt['auc']:.4f})")
    return model


def get_hazard_probs(model, seqs, masks, batch_size=512):
    """
    seqs:  (N, T, 6) float32
    masks: (N, T)    bool True=real
    returns: (N, T)  float32 hazard probabilities per timestep
    """
    all_h = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            x = torch.tensor(seqs[i:i+batch_size],  device=DEVICE)
            m = torch.tensor(masks[i:i+batch_size], device=DEVICE)
            logits = model(x, mask=m, return_per_timestep=True)
            h = torch.sigmoid(logits).cpu().numpy()
            all_h.append(h)
    return np.concatenate(all_h, axis=0)  # (N, T)


def compute_prices_vectorized(h, seq_lens, orig_upbs, annual_rate, path_rates):
    """
    Vectorized cashflow computation across all loans for one rate path.
    h:          (N, T) hazard probabilities
    seq_lens:   (N,)   active months per loan
    orig_upbs:  (N,)   original UPB in dollars
    annual_rate: (N,) array of per-loan note rates (decimal)
    path_rates: (T,)   monthly rates in % from DDPM path
    returns:    (N,)   price as % of par
    """
    N, T = h.shape
    # annual_rate can be scalar or (N,) array
    annual_rate  = np.asarray(annual_rate)
    monthly_note = annual_rate / 12          # (N,) or scalar
    n_payments   = 360
    pmt = orig_upbs * monthly_note / (1 - (1 + monthly_note) ** (-n_payments))  # (N,)

    balance  = orig_upbs.copy()   # (N,)
    survival = np.ones(N)         # (N,)
    pv       = np.zeros(N)        # (N,) present value accumulator
    cum_disc = np.ones(N)         # (N,) cumulative discount factor

    monthly_disc = path_rates / 100.0 / 12.0  # (T,) monthly discount rates

    for t in range(T):
        active = t < seq_lens  # (N,) bool — loan still active at t

        h_t       = h[:, t]                                    # (N,)
        interest  = balance * monthly_note                     # (N,)
        principal = np.minimum(pmt - interest, balance)        # (N,)
        prepay    = balance * h_t                              # (N,)

        # Total cashflow: balance already tracks survival via depletion
        cf = interest + principal + prepay                     # (N,)

        # Accumulate discount
        cum_disc *= (1 + monthly_disc[t])

        # Add discounted cashflow for active loans
        pv += np.where(active, cf / cum_disc, 0.0)

        # Update state
        balance  = np.maximum(balance - principal - prepay, 0.0)
        survival = survival * (1 - h_t)  # kept for potential future use

    # Terminal value: remaining balance at end of sequence discounted back
    # This captures the value of cashflows beyond month 33
    # Use the last month's discount factor and remaining balance
    terminal_value = balance / cum_disc  # (N,) remaining balance discounted to t=0
    pv += terminal_value

    # Price as % of par
    prices = pv / orig_upbs * 100.0
    return prices


def main():
    print(f"Device: {DEVICE}")

    # Load data
    print("Loading sequences and rate paths...")
    test_seq   = np.load(os.path.join(SEQ_DIR, 'test_seq.npy'),  mmap_mode='r')
    test_mask  = np.load(os.path.join(SEQ_DIR, 'test_mask.npy'), mmap_mode='r')
    treasury_paths = np.load(os.path.join(OUTPUTS, 'treasury_rate_paths.npy'))  # (1000, 33)
    pmms_paths     = np.load(os.path.join(OUTPUTS, 'pmms_rate_paths_rn.npy'))    # (1000, 33)
    treasury_paths = treasury_paths[:N_PATHS, :MAX_SEQ]
    pmms_paths     = pmms_paths[:N_PATHS, :MAX_SEQ]

    with open(os.path.join(SEQ_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    refi_mean = scaler.mean_[0]
    refi_std  = scaler.scale_[0]
    upb_mean  = scaler.mean_[4]
    upb_std   = scaler.scale_[4]

    # Sample loans
    rng = np.random.default_rng(42)
    loan_idx     = rng.choice(len(test_seq), size=N_LOANS_SAMPLE, replace=False)
    sampled_seq  = np.array(test_seq[loan_idx])   # (N, 33, 6)
    sampled_mask = np.array(test_mask[loan_idx])  # (N, 33) True=real

    # Recover UPB per loan (unscale feature index 4 at t=0)
    orig_upb = sampled_seq[:, 0, 4] * upb_std + upb_mean  # (N,)
    seq_lens = sampled_mask.sum(axis=1).astype(np.int32)   # (N,)

    # Recover per-loan orig_rate (in decimal) from refi_incentive at t=0
    # refi_incentive = orig_rate(%) - pmms(%), so orig_rate(%) = refi_unscaled + pmms_mean
    refi_unscaled = sampled_seq[:, 0, 0] * refi_std + refi_mean  # (N,) in pp
    orig_rate     = (refi_unscaled + PMMS_MEAN_2020_2021) / 100.0  # (N,) decimal
    orig_rate     = np.clip(orig_rate, 0.01, 0.12)  # sanity clip
    print(f"Per-loan orig_rate: mean={orig_rate.mean()*100:.2f}% std={orig_rate.std()*100:.2f}%")

    # Load hazard model
    model = load_hazard_model()

    print(f"\nRunning OAS engine: {N_LOANS_SAMPLE} loans x {N_PATHS} paths...")
    loan_prices = np.zeros((N_LOANS_SAMPLE, N_PATHS), dtype=np.float32)

    for p in range(N_PATHS):
        if p % 100 == 0:
            print(f"  Path {p}/{N_PATHS}...", flush=True)

        pmms_path     = pmms_paths[p]      # (33,) PMMS rates in % — for refi incentive
        treasury_path = treasury_paths[p]  # (33,) Treasury rates in % — for discounting

        # Rebuild sequences with path-driven refi_incentive using PMMS path
        orig_rate_pct = orig_rate * 100  # (N,) convert decimal to %
        path_seq = sampled_seq.copy()
        for t in range(MAX_SEQ):
            pmms_t = pmms_path[t]                              # scalar
            refi_t = orig_rate_pct - pmms_t                    # (N,) per-loan
            path_seq[:, t, 0] = (refi_t - refi_mean) / refi_std

        # Get per-timestep hazard probs under this PMMS path
        h = get_hazard_probs(model, path_seq, sampled_mask)  # (N, 33)

        # Vectorized cashflow discounting using Treasury path
        prices = compute_prices_vectorized(
            h, seq_lens, orig_upb, orig_rate, treasury_path
        )
        loan_prices[:, p] = prices

    # Average across paths
    model_prices = loan_prices.mean(axis=1)
    price_std    = loan_prices.std(axis=1)

    print(f"\nModel prices (% of par):")
    print(f"  Mean:   {model_prices.mean():.2f}")
    print(f"  Median: {np.median(model_prices):.2f}")
    print(f"  Std:    {model_prices.std():.2f}")
    print(f"  Min:    {model_prices.min():.2f}")
    print(f"  Max:    {model_prices.max():.2f}")

    # Save
    results = {
        'loan_idx':     loan_idx.tolist(),
        'model_prices': model_prices.tolist(),
        'price_std':    price_std.tolist(),
        'n_loans':      N_LOANS_SAMPLE,
        'n_paths':      N_PATHS,
        'annual_coupon': 'per_loan_recovered',
        'note': 'Prices as % of par. OAS pending market price data from advisor.'
    }
    np.save(os.path.join(OUTPUTS, 'oas_loan_prices.npy'), loan_prices)
    with open(os.path.join(OUTPUTS, 'oas_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: oas_loan_prices.npy, oas_results.json")
    print("Next step: plug in market prices from advisor to compute OAS spread.")


if __name__ == "__main__":
    main()
