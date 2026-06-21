"""
diag_equity_incentive.py — 2D equity × rate-incentive diagnostic.

Answers Gupta's question: does the Transformer learn that low-LTV loans
(high equity) prepay much more aggressively when rate incentive is positive,
while high-LTV loans (low equity) don't refinance even with strong incentive?

Method:
  1. Load production model (outputs/hazard_best.pt) + scaler.
  2. Build a grid of synthetic 12-month sequences varying two features:
       X: refi_incentive = original_rate - PMMS in {-2.0, -1.75, ..., +4.0}
       Y: current_ltv    = {30, 35, ..., 130}
     All other features held at prime conforming medians.
  3. Run model in return_per_timestep=True mode; take h at position 11
     (last real timestep of a steady-state 12-month sequence) as the
     predicted monthly prepay hazard.
  4. Plot heatmap + save grid CSV.

Note on data coverage:
  Training vintages 2013Q1–2023Q1 have LTV concentrated in 30–100 range.
  Underwater loans (LTV>100) are ~0.1% of panel — model has limited exposure
  to the worst post-GFC underwater cohort (2009-2012 originations).
  The interaction is demonstrable in the 60–100 LTV range, which IS well-
  represented in training data.

Usage:
    python diag_equity_incentive.py
    python diag_equity_incentive.py --model_path outputs/rolling/cutoff_2020/hazard_best.pt
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE         = '/scratch/at7095/mortgage_prepayment'
MODEL_PATH   = os.path.join(BASE, 'outputs/hazard_best.pt')
SCALER_PATH  = os.path.join(BASE, 'data/sequences/scaler.pkl')
OUT_DIR      = os.path.join(BASE, 'outputs')

MAX_SEQ    = 33
N_FEATURES = 9
SEQ_LEN    = 12   # synthetic sequence length — enough for positional encoding to engage
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_COLS = [
    'refi_incentive',        # [0] — varied on X axis
    'borrower_credit_score', # [1]
    'original_ltv',          # [2]
    'current_ltv',           # [3] — varied on Y axis
    'original_upb',          # [4]
    'loan_age_months',       # [5]
    'dti',                   # [6]
    'loan_purpose_enc',      # [7]
    'property_type_enc',     # [8]
]

# Median feature values — prime conforming 30yr fixed circa 2020
MEDIANS = {
    'refi_incentive':        0.0,    # placeholder; overridden by grid
    'borrower_credit_score': 750.0,
    'original_ltv':          80.0,
    'current_ltv':           70.0,   # placeholder; overridden by grid
    'original_upb':          320_000.0,
    'loan_age_months':       18.0,
    'dti':                   35.0,
    'loan_purpose_enc':      0.0,    # Purchase
    'property_type_enc':     0.0,    # Single family
}

# Grid axes
REFI_GRID = np.arange(-2.0, 4.25, 0.25)   # 25 points
LTV_GRID  = np.arange(30, 135, 5)          # 21 points — covers 30–130


# ── Model ─────────────────────────────────────────────────────────────────────

class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=N_FEATURES, d_model=64, n_heads=4, n_layers=2,
                 dim_ff=256, max_seq=MAX_SEQ, dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
              dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, mask=None, return_per_timestep=False):
        B, T, _ = x.shape
        pos      = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out      = self.input_proj(x) + self.pos_embedding(pos)
        pad_mask = ~mask if mask is not None else None
        out      = self.transformer(out, src_key_padding_mask=pad_mask)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)


def load_model(path: str) -> PrepaymentTransformer:
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get('config', {})
    m = PrepaymentTransformer(
        input_dim=cfg.get('input_dim', N_FEATURES),
        d_model=cfg.get('d_model',    64),
        n_heads=cfg.get('n_heads',    4),
        n_layers=cfg.get('n_layers',  2),
        dim_ff=cfg.get('dim_ff',      256),
        dropout=cfg.get('dropout',    0.1),
    ).to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    print(f'Loaded model from {path}', flush=True)
    print(f'  epoch={ckpt.get("epoch","?")}  AUC={ckpt.get("auc","?"):.4f}', flush=True)
    return m


# ── Synthetic sequence builder ─────────────────────────────────────────────────

def build_synthetic_grid(scaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build batch of synthetic sequences for the REFI × LTV grid.

    Each sequence has SEQ_LEN timesteps where ALL timesteps have the same
    feature values (steady-state assumption). This isolates the model's
    response to the current-state features vs temporal dynamics.

    Returns:
        sequences: (N_GRID, MAX_SEQ, N_FEATURES) float32
        masks:     (N_GRID, MAX_SEQ) bool
        meta:      DataFrame with refi_incentive, current_ltv per grid point
    """
    refi_vals = REFI_GRID
    ltv_vals  = LTV_GRID
    n_total   = len(refi_vals) * len(ltv_vals)

    # Build unscaled feature matrix: (N_GRID, N_FEATURES)
    raw_grid = np.zeros((n_total, N_FEATURES), dtype=np.float64)
    meta_rows = []

    i = 0
    for refi in refi_vals:
        for ltv in ltv_vals:
            row = np.array([
                refi,
                MEDIANS['borrower_credit_score'],
                MEDIANS['original_ltv'],
                ltv,
                MEDIANS['original_upb'],
                MEDIANS['loan_age_months'],
                MEDIANS['dti'],
                MEDIANS['loan_purpose_enc'],
                MEDIANS['property_type_enc'],
            ])
            raw_grid[i] = row
            meta_rows.append({'refi_incentive': refi, 'current_ltv': ltv, 'grid_idx': i})
            i += 1

    meta = pd.DataFrame(meta_rows)

    # Scale features using production scaler
    # numpy arrays produce a FutureWarning about feature names — harmless, works by position
    scaled = scaler.transform(raw_grid).astype(np.float32)

    # Build padded sequences: each grid point → SEQ_LEN-step steady-state sequence
    sequences = np.zeros((n_total, MAX_SEQ, N_FEATURES), dtype=np.float32)
    masks     = np.zeros((n_total, MAX_SEQ), dtype=bool)
    for j in range(SEQ_LEN):
        sequences[:, j, :] = scaled    # same features at every timestep
        masks[:, j]        = True

    return sequences, masks, meta


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, sequences: np.ndarray, masks: np.ndarray,
                  batch_size: int = 256) -> np.ndarray:
    """Run model in per-timestep mode, return hazard at last real position.

    Returns h: (N_GRID,) float32 — predicted monthly prepay hazard rate.
    """
    n      = len(sequences)
    h_vals = np.zeros(n, dtype=np.float32)
    last_t = int(SEQ_LEN) - 1    # last real timestep index (0-indexed)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            bs = torch.tensor(sequences[i:i+batch_size], device=DEVICE)
            bm = torch.tensor(masks[i:i+batch_size],     device=DEVICE)
            logits = model(bs, mask=bm, return_per_timestep=True)  # (B, MAX_SEQ)
            h_pt   = torch.sigmoid(logits[:, last_t]).cpu().numpy()
            h_vals[i:i+batch_size] = h_pt

    return h_vals


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(meta: pd.DataFrame, h_vals: np.ndarray, out_prefix: str) -> None:
    """Save annotated 2D heatmap of predicted hazard rate."""
    meta = meta.copy()
    meta['h'] = h_vals * 100    # convert to monthly prepay % for readability

    # Pivot: rows=LTV, cols=refi_incentive (so y-axis = LTV ascending bottom-to-top)
    pivot = meta.pivot(index='current_ltv', columns='refi_incentive', values='h')
    # Sort index descending so LTV=30 is at top (high equity at top)
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        pivot.values,
        aspect='auto',
        cmap='RdYlGn_r',   # green=low hazard, red=high hazard
        vmin=0,
        vmax=meta['h'].quantile(0.97),   # clip top 3% for colour range
        interpolation='bilinear',
    )

    # Axes labels
    refi_ticks = list(pivot.columns)
    ltv_ticks  = list(pivot.index)

    # Sparse ticks — every 1pp for refi, every 10 LTV units
    refi_tick_pos   = [i for i, r in enumerate(refi_ticks) if r % 1.0 == 0]
    refi_tick_labels = [f'{r:+.0f}' for r in refi_ticks if r % 1.0 == 0]
    ltv_tick_pos    = [i for i, l in enumerate(ltv_ticks) if l % 10 == 0]
    ltv_tick_labels = [str(int(l)) for l in ltv_ticks if l % 10 == 0]

    ax.set_xticks(refi_tick_pos, refi_tick_labels, fontsize=9)
    ax.set_yticks(ltv_tick_pos,  ltv_tick_labels,  fontsize=9)
    ax.set_xlabel('Rate Incentive: Note Rate − PMMS (percentage points)', fontsize=11)
    ax.set_ylabel('Current LTV (%) — lower = more equity', fontsize=11)
    ax.set_title(
        'Transformer Model: Predicted Monthly Prepay Hazard (%)\n'
        'Rate Incentive × Equity Position — Steady-State Synthetic Sequences',
        fontsize=12, fontweight='bold',
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Monthly Prepay Hazard (%)', fontsize=10)

    # Reference lines
    # LTV=80 row index
    ltv_80_idx = [i for i, l in enumerate(ltv_ticks) if l == 80]
    ltv_100_idx = [i for i, l in enumerate(ltv_ticks) if l == 100]
    refi_0_idx  = [i for i, r in enumerate(refi_ticks) if r == 0.0]

    for idx in ltv_80_idx:
        ax.axhline(idx - 0.5, color='white', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text(0.02, (idx) / len(ltv_ticks), 'LTV=80', transform=ax.transAxes,
                color='white', fontsize=8, va='center', alpha=0.9)

    for idx in ltv_100_idx:
        ax.axhline(idx - 0.5, color='#ffcc00', linewidth=1.5, linestyle=':', alpha=0.9)
        ax.text(0.02, (idx) / len(ltv_ticks), 'LTV=100\n(underwater)', transform=ax.transAxes,
                color='#ffcc00', fontsize=7.5, va='center', alpha=0.95)

    for idx in refi_0_idx:
        ax.axvline(idx - 0.5, color='white', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text((idx) / len(refi_ticks), 0.97, 'Incentive=0',
                transform=ax.transAxes, color='white', fontsize=8,
                ha='center', va='top', alpha=0.9)

    # Data coverage note
    ax.text(
        0.98, 0.02,
        'Note: LTV>100 is ~0.1% of training data\n(2013–2023 vintages; post-GFC underwater\ncohort 2009–2012 not in sample)',
        transform=ax.transAxes, fontsize=7.5, color='lightyellow',
        ha='right', va='bottom', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.7),
    )

    plt.tight_layout()
    png_path = os.path.join(out_prefix + '.png')
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Heatmap: {png_path}', flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  default=MODEL_PATH,
                        help='Path to hazard model checkpoint')
    parser.add_argument('--scaler_path', default=SCALER_PATH,
                        help='Path to scaler.pkl')
    parser.add_argument('--out_dir',     default=OUT_DIR,
                        help='Output directory for PNG and CSV')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, 'diag_equity_incentive')

    # ── Load model + scaler ───────────────────────────────────────────────────
    model = load_model(args.model_path)

    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f'Loaded scaler: {args.scaler_path}', flush=True)
    print(f'  Scaler means (refi,ltv): '
          f'{scaler.mean_[0]:.3f}, {scaler.mean_[3]:.3f}', flush=True)

    # ── Build synthetic grid ──────────────────────────────────────────────────
    print(f'\nBuilding grid: {len(REFI_GRID)} refi × {len(LTV_GRID)} LTV '
          f'= {len(REFI_GRID)*len(LTV_GRID)} points...', flush=True)
    sequences, masks, meta = build_synthetic_grid(scaler)
    print(f'  Sequence array: {sequences.shape}  (seq_len={SEQ_LEN}, padded to {MAX_SEQ})',
          flush=True)

    # ── Run inference ─────────────────────────────────────────────────────────
    print('\nRunning inference...', flush=True)
    h_vals = run_inference(model, sequences, masks)
    print(f'  Hazard range: [{h_vals.min():.4f}, {h_vals.max():.4f}]', flush=True)

    meta['h_monthly_pct'] = h_vals * 100
    meta['cpr_annualized'] = (1.0 - (1.0 - h_vals) ** 12) * 100

    # ── Save grid CSV ─────────────────────────────────────────────────────────
    csv_path = out_prefix + '.csv'
    meta.to_csv(csv_path, index=False)
    print(f'  Grid CSV: {csv_path}', flush=True)

    # ── Print summary table ───────────────────────────────────────────────────
    print('\nPredicted monthly hazard (%) — Equity × Incentive interaction:', flush=True)
    for ltv in [60, 80, 100, 120]:
        row_vals = []
        for refi in [-1.0, 0.0, 1.0, 2.0, 3.0]:
            match = meta[(meta['current_ltv'] == ltv) &
                         (np.abs(meta['refi_incentive'] - refi) < 0.01)]
            val = match['h_monthly_pct'].values[0] if len(match) else np.nan
            row_vals.append(f'{val:.2f}%')
        print(f'  LTV={ltv:3d}: refi=-1pp={row_vals[0]}, 0pp={row_vals[1]}, '
              f'+1pp={row_vals[2]}, +2pp={row_vals[3]}, +3pp={row_vals[4]}', flush=True)

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    print('\nPlotting heatmap...', flush=True)
    plot_heatmap(meta, h_vals, out_prefix)

    print(f'\nDone. Outputs: {out_prefix}.png / .csv', flush=True)


if __name__ == '__main__':
    main()
