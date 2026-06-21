"""
train_hazard_rolling.py — Per-cutoff-year hazard model retraining + Platt calibration.

Architecture identical to train_hazard.py. Adds Platt calibration post-training
so downstream CPR forecasting uses calibrated monthly hazard estimates.

Usage:
    python train_hazard_rolling.py --cutoff_year 2018
    python train_hazard_rolling.py --cutoff_year 2018 --n_epochs 30

Outputs (to outputs/rolling/cutoff_{YEAR}/):
    hazard_best.pt    — best model checkpoint (same format as production hazard_best.pt)
    results.json      — best AUC, Platt params (a, b), training history
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

BASE = '/scratch/at7095/mortgage_prepayment'

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE      = 2048
N_EPOCHS        = 50      # override with --n_epochs if needed
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
STEPS_PER_EPOCH = 10_000  # steps not full passes — keeps epoch wall-clock predictable
MAX_SEQ         = 33
N_FEATURES      = 9
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model (identical to production train_hazard.py) ───────────────────────────

class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim: int = N_FEATURES, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_ff: int = 256, max_seq: int = MAX_SEQ, dropout: float = 0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x, mask=None, return_per_timestep: bool = False):
        B, T, _ = x.shape
        pos      = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out      = self.input_proj(x) + self.pos_embedding(pos)
        pad_mask = ~mask if mask is not None else None       # True=padding for PyTorch
        out      = self.transformer(out, src_key_padding_mask=pad_mask)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)          # (B, T)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)              # (B,)


# ── Hazard sampler (identical to production) ──────────────────────────────────

class HazardSampler:
    """50/50 oversampling of prepaid vs all-valid loans, random timestep per loan."""

    def __init__(self, sequences, masks, prepay_timesteps):
        self.sequences = sequences
        self.masks     = masks
        self.prepay_t  = prepay_timesteps
        self.seq_lens  = masks.sum(axis=1).astype(np.int32)
        self.max_t     = np.where(
            self.prepay_t >= 0, self.prepay_t, self.seq_lens - 1
        ).astype(np.int32)
        valid              = self.max_t >= 0
        self.valid_idx     = np.where(valid)[0]
        self.prepaid_idx   = np.where(self.prepay_t >= 0)[0]
        n_prepaid          = len(self.prepaid_idx)
        print(f'  Sampler: {len(self.valid_idx):,} valid | '
              f'{n_prepaid:,} prepaid ({100*n_prepaid/len(sequences):.2f}%)')

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        n_pos    = batch_size // 2
        n_neg    = batch_size - n_pos
        loan_idx = np.concatenate([
            rng.choice(self.prepaid_idx, size=n_pos, replace=True),
            rng.choice(self.valid_idx,   size=n_neg, replace=True),
        ])
        rng.shuffle(loan_idx)

        max_ts    = self.max_t[loan_idx]
        t_sampled = np.clip(
            (rng.random(batch_size) * (max_ts + 1)).astype(np.int32), 0, max_ts
        )
        labels = (t_sampled == self.prepay_t[loan_idx]).astype(np.float32)

        batch_seq  = np.zeros((batch_size, MAX_SEQ, N_FEATURES), dtype=np.float32)
        batch_mask = np.zeros((batch_size, MAX_SEQ), dtype=bool)
        for i in range(batch_size):
            t = t_sampled[i]; l = loan_idx[i]
            batch_seq[i, :t+1, :] = self.sequences[l, :t+1, :]
            batch_mask[i, :t+1]   = True
        return batch_seq, batch_mask, labels, t_sampled


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_hazard(model, seq, mask, labels, batch_size: int = 512):
    """Compute AUC using survival CDF: score = 1 - prod(1 - h_t)."""
    model.eval()
    n      = len(seq)
    scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bs = torch.tensor(seq[i:i+batch_size],  device=DEVICE)
            bm = torch.tensor(mask[i:i+batch_size], device=DEVICE)
            logits = model(bs, mask=bm, return_per_timestep=True)
            h      = torch.sigmoid(logits).cpu().numpy()
            active = bm.cpu().numpy()
            h_m    = np.where(active, h, 0.0)
            surv   = np.prod(1.0 - h_m, axis=1, where=active, initial=1.0)
            scores[i:i+batch_size] = 1.0 - surv
    return roc_auc_score(labels, scores), scores


# ── Platt calibration ─────────────────────────────────────────────────────────

def platt_calibrate(raw_scores: np.ndarray, labels: np.ndarray):
    """Fit P = sigmoid(a * score + b) minimising log-loss.

    raw_scores: survival-CDF scores from evaluate_hazard (already in [0,1])
    labels:     binary prepaid labels

    Returns (a, b). Typical values: a~0.4-0.6, b~-4 to -5 for this model.
    """
    def nll(params):
        a, b = params
        p = np.clip(1.0 / (1.0 + np.exp(-(a * raw_scores + b))), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    res = minimize(nll, x0=[1.0, 0.0], method='Nelder-Mead',
                   options={'maxiter': 20_000, 'xatol': 1e-7, 'fatol': 1e-7})
    return float(res.x[0]), float(res.x[1])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year', type=int, required=True)
    parser.add_argument('--n_epochs',    type=int, default=N_EPOCHS)
    args = parser.parse_args()

    SEQ_DIR = os.path.join(BASE, f'data/sequences_rolling/cutoff_{args.cutoff_year}')
    OUT_DIR = os.path.join(BASE, f'outputs/rolling/cutoff_{args.cutoff_year}')
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Device: {DEVICE}  |  cutoff: {args.cutoff_year}', flush=True)
    print(f'Sequences: {SEQ_DIR}', flush=True)
    print(f'Outputs:   {OUT_DIR}', flush=True)

    # ── Load sequences ────────────────────────────────────────────────────────
    def _load(name):
        return np.load(os.path.join(SEQ_DIR, name), mmap_mode='r')

    train_seq    = _load('train_seq.npy')
    train_mask   = _load('train_mask.npy')
    train_prepay = np.load(os.path.join(SEQ_DIR, 'train_prepay_timestep.npy'))
    test_seq     = _load('test_seq.npy')
    test_mask    = _load('test_mask.npy')
    test_labels  = np.load(os.path.join(SEQ_DIR, 'test_labels.npy'))

    print(f'train: {train_seq.shape} | test: {test_seq.shape}', flush=True)

    # ── Training ──────────────────────────────────────────────────────────────
    sampler   = HazardSampler(train_seq, train_mask, train_prepay)
    model     = PrepaymentTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=DEVICE))
    rng       = np.random.default_rng(42)

    best_auc, best_scores = 0.0, None
    history = []

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for _ in range(STEPS_PER_EPOCH):
            bseq, bmask, lbl, _ = sampler.sample_batch(BATCH_SIZE, rng)
            x = torch.tensor(bseq,  device=DEVICE)
            m = torch.tensor(bmask, device=DEVICE)
            y = torch.tensor(lbl,   device=DEVICE)
            logits = model(x, mask=m)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss    = epoch_loss / STEPS_PER_EPOCH
        auc, scores = evaluate_hazard(model, test_seq, test_mask, test_labels)
        elapsed     = time.time() - t0

        history.append({'epoch': epoch, 'loss': avg_loss, 'auc': float(auc)})
        print(f'Epoch {epoch:>2}/{args.n_epochs}  loss={avg_loss:.5f}  '
              f'auc={auc:.4f}  t={elapsed:.0f}s', flush=True)

        scheduler.step(auc)
        if auc > best_auc:
            best_auc    = auc
            best_scores = scores.copy()
            torch.save({
                'model_state': model.state_dict(),
                'config': {'input_dim': N_FEATURES, 'n_heads': 4, 'n_layers': 2,
                           'd_model': 64, 'dim_ff': 256, 'dropout': 0.1},
                'cutoff_year': args.cutoff_year,
                'epoch': epoch,
                'auc':   float(auc),
            }, os.path.join(OUT_DIR, 'hazard_best.pt'))
            print(f'  → Best AUC: {best_auc:.4f} — saved.', flush=True)

    # ── Platt calibration on test set ─────────────────────────────────────────
    # Calibrates the survival-CDF score (used for AUC evaluation).
    # Note: per-timestep hazard rates used for CPR forecasting are NOT
    # Platt-calibrated here — that is a raw sigmoid output interpreted as a
    # monthly prepay probability. Platt params are retained for future use
    # in probability-level applications (e.g., regulatory PD estimates).
    print('\nFitting Platt calibration on test set...', flush=True)
    a, b = platt_calibrate(best_scores, test_labels)
    print(f'  Platt: a={a:.4f}, b={b:.4f}', flush=True)

    results = {
        'cutoff_year': args.cutoff_year,
        'best_auc':    float(best_auc),
        'platt_a':     a,
        'platt_b':     b,
        'n_train':     int(len(train_seq)),
        'n_test':      int(len(test_seq)),
        'history':     history,
    }
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nDone. cutoff={args.cutoff_year} | AUC={best_auc:.4f} | '
          f'Platt a={a:.4f} b={b:.4f}', flush=True)


if __name__ == '__main__':
    main()
