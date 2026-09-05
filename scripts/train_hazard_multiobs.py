"""
train_hazard_multiobs.py — Trains on prepare_sequences_multiobs_zbc.py output:
fixed-window (loan_id, ref_month) observations, one forward label each, built
specifically to separate "high refi incentive" from "high incentive but
already declined it repeatedly" (burnout/survivor selection) by observing the
same loan at several ages instead of once at its final age.

THIS IS A NEW FILE, NOT A COPY-EDIT OF train_hazard_rolling.py. The scoring
logic is deliberately different, not an oversight — see "Scoring" below.
Do not merge train_hazard_rolling.py's HazardSampler / evaluate_hazard back
in here; the multiobs builder's own docstring says pointing HazardSampler at
this output destroys the window/label alignment it constructs (random
per-epoch truncation on top of windows that are already fixed at prep time,
plus a second layer of 50/50 oversampling stacked on the mandatory-draw
oversampling already baked into which rows exist).

Model: PrepaymentTransformer, architecture and hyperparameters identical to
train_hazard_rolling.py / train_hazard.py.

Scoring: each multiobs observation is a fixed window ending at ref_month with
ONE forward label (prepay in (ref_month, ref_month+H]) — not an open-ended
sequence that needs per-timestep survival aggregation. So training and eval
both use the masked-mean-pool + classifier forward pass (model(seq, mask),
no return_per_timestep) giving one logit per observation, and evaluate()
scores with sigmoid(logit) directly. No survival-CDF product.

Batching: no HazardSampler, no random truncation — each observation's window
is already fixed at prep time by the builder. ObservationSampler below just
draws observation indices, either uniformly (natural ~8% positive rate) or,
if --pos_ratio is set, at a chosen positive/negative mix.

Do NOT load train_prepay_timestep / test_prepay_timestep — the builder's own
module docstring says these are meaningless for this output (filled with -1
for every observation, kept only for file-shape compatibility with the
trailing builder).

Usage:
    python train_hazard_multiobs.py --cutoff_year 2020
    python train_hazard_multiobs.py --cutoff_year 2020 --pos_ratio 0.5 --use_ipw

Outputs (to outputs/rolling/cutoff_{YEAR}_multiobs_k{K}_h{H}/):
    hazard_best.pt    — best model checkpoint
    results.json      — best AUC, Platt params (a, b), pos_ratio/use_ipw used, history
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

# ── Hyperparameters (identical to train_hazard_rolling.py) ────────────────────
BATCH_SIZE      = 2048
N_EPOCHS        = 50      # override with --n_epochs if needed
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
STEPS_PER_EPOCH = 10_000  # steps not full passes — keeps epoch wall-clock predictable
MAX_SEQ         = 33
N_FEATURES      = 9
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model (identical to train_hazard_rolling.py) ───────────────────────────────

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


# ── Batch sampler ───────────────────────────────────────────────────────────
# NOT a HazardSampler. Each observation's window is already fixed at prep
# time by the multiobs builder (right-aligned at ref_month, label already
# resolved) — there is no per-timestep truncation to do here, just draw
# observation indices.

class ObservationSampler:
    """Draws batch_size observation indices.

    pos_ratio=None (default): uniform random draw over ALL observations —
        trains on the natural positive rate as-is (~8.33% for k=5,h=1).
    pos_ratio=f: draw f*batch_size from label==1 indices and the rest from
        label==0 indices, both with replacement.
    """

    def __init__(self, seq, mask, labels, incl_prob, pos_ratio=None):
        self.seq       = seq
        self.mask      = mask
        self.labels    = labels
        self.incl_prob = incl_prob
        self.pos_ratio = pos_ratio
        self.n         = len(labels)

        if pos_ratio is None:
            print(f'  Sampler: uniform over {self.n:,} observations '
                  f'(natural {100 * float(labels.mean()):.2f}% positive rate)', flush=True)
        else:
            self.pos_idx = np.where(labels == 1)[0]
            self.neg_idx = np.where(labels == 0)[0]
            print(f'  Sampler: pos_ratio={pos_ratio} draw | '
                  f'{len(self.pos_idx):,} positive / {len(self.neg_idx):,} negative '
                  f'observations available', flush=True)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        if self.pos_ratio is None:
            idx = rng.integers(0, self.n, size=batch_size)
        else:
            n_pos = int(round(batch_size * self.pos_ratio))
            n_neg = batch_size - n_pos
            idx = np.concatenate([
                rng.choice(self.pos_idx, size=n_pos, replace=True),
                rng.choice(self.neg_idx, size=n_neg, replace=True),
            ])
            rng.shuffle(idx)

        batch_seq    = np.asarray(self.seq[idx])
        batch_mask   = np.asarray(self.mask[idx])
        batch_labels = np.asarray(self.labels[idx])
        batch_weight = np.asarray(self.incl_prob[idx]) if self.incl_prob is not None else None
        return batch_seq, batch_mask, batch_labels, batch_weight


# ── Evaluation ────────────────────────────────────────────────────────────────
# Deliberately NOT a survival-CDF aggregation (contrast with
# train_hazard_rolling.py's evaluate_hazard). Each multiobs observation is a
# single fixed window with one forward label baked in by the builder, so the
# model's single pooled logit for that observation IS the score. Do not
# re-add per-timestep survival aggregation here by copy-pasting
# evaluate_hazard from train_hazard_rolling.py — there is no open-ended
# sequence to aggregate over.

def evaluate(model, seq, mask, labels, batch_size: int = 512):
    """score = sigmoid(logit) directly. roc_auc_score(labels, score)."""
    model.eval()
    n      = len(labels)
    scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bs = torch.tensor(np.asarray(seq[i:i + batch_size]),  device=DEVICE)
            bm = torch.tensor(np.asarray(mask[i:i + batch_size]), device=DEVICE)
            logits = model(bs, mask=bm)                       # (B,) — no return_per_timestep
            scores[i:i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return roc_auc_score(labels, scores), scores


# ── Platt calibration (identical to train_hazard_rolling.py) ──────────────────

def platt_calibrate(raw_scores: np.ndarray, labels: np.ndarray):
    """Fit P = sigmoid(a * score + b) minimising log-loss.

    raw_scores: sigmoid scores from evaluate() (already in [0,1])
    labels:     binary prepaid-in-horizon labels

    Returns (a, b).
    """
    def nll(params):
        a, b = params
        p = np.clip(1.0 / (1.0 + np.exp(-(a * raw_scores + b))), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    res = minimize(nll, x0=[1.0, 0.0], method='Nelder-Mead',
                   options={'maxiter': 20_000, 'xatol': 1e-7, 'fatol': 1e-7})
    return float(res.x[0]), float(res.x[1])


# ── Training loop (factored out so a synthetic smoke test can drive it) ───────

def train_and_evaluate(
    train_seq, train_mask, train_labels, train_incl_prob,
    test_seq, test_mask, test_labels,
    n_epochs: int, pos_ratio, use_ipw: bool,
    out_dir: str | None = None,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
    max_seq: int = MAX_SEQ,
    seed: int = 42,
):
    print(f'Active options: pos_ratio={pos_ratio!r}  use_ipw={use_ipw}', flush=True)

    sampler   = ObservationSampler(train_seq, train_mask, train_labels, train_incl_prob,
                                    pos_ratio=pos_ratio)
    model     = PrepaymentTransformer(max_seq=max_seq).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss(reduction='none' if use_ipw else 'mean')
    rng       = np.random.default_rng(seed)

    best_auc, best_scores = 0.0, None
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for _ in range(steps_per_epoch):
            bseq, bmask, lbl, bweight = sampler.sample_batch(batch_size, rng)
            x = torch.tensor(bseq,  device=DEVICE)
            m = torch.tensor(bmask, device=DEVICE)
            y = torch.tensor(lbl,   device=DEVICE)
            logits = model(x, mask=m)

            if use_ipw:
                w = torch.tensor(bweight, device=DEVICE)
                per_sample = criterion(logits, y)
                loss = (per_sample * w).sum() / w.sum().clamp(min=1e-12)
            else:
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss    = epoch_loss / steps_per_epoch
        auc, scores = evaluate(model, test_seq, test_mask, test_labels)
        elapsed     = time.time() - t0

        history.append({'epoch': epoch, 'loss': avg_loss, 'auc': float(auc)})
        print(f'Epoch {epoch:>2}/{n_epochs}  loss={avg_loss:.5f}  '
              f'auc={auc:.4f}  t={elapsed:.0f}s', flush=True)

        scheduler.step(auc)
        if auc > best_auc:
            best_auc    = auc
            best_scores = scores.copy()
            if out_dir is not None:
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {'input_dim': N_FEATURES, 'n_heads': 4, 'n_layers': 2,
                               'd_model': 64, 'dim_ff': 256, 'dropout': 0.1,
                               'max_seq': max_seq},
                    'epoch': epoch,
                    'auc':   float(auc),
                }, os.path.join(out_dir, 'hazard_best.pt'))
            print(f'  → Best AUC: {best_auc:.4f}' + (' — saved.' if out_dir else ''), flush=True)

    print('\nFitting Platt calibration on test set...', flush=True)
    a, b = platt_calibrate(best_scores, test_labels)
    print(f'  Platt: a={a:.4f}, b={b:.4f}', flush=True)

    return {
        'best_auc': float(best_auc),
        'platt_a':  a,
        'platt_b':  b,
        'history':  history,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year',   type=int, default=2020)
    parser.add_argument('--k_draws',       type=int, default=5)
    parser.add_argument('--label_horizon', type=int, default=1)
    parser.add_argument('--n_epochs',      type=int, default=N_EPOCHS)
    parser.add_argument('--pos_ratio',     type=float, default=None,
                         help='Fraction of each batch drawn from label==1 observations. '
                              'Default: unset — uniform draw over all observations at '
                              'the natural positive rate.')
    parser.add_argument('--use_ipw',       action='store_true', default=False,
                         help='Weight the loss per-sample by train_incl_prob '
                              '(weighted mean instead of plain mean). Default: off.')
    args = parser.parse_args()

    SEQ_DIR = os.path.join(
        BASE, f'data/sequences_rolling/cutoff_{args.cutoff_year}_zbc_multiobs'
              f'_k{args.k_draws}_h{args.label_horizon}')
    OUT_DIR = os.path.join(
        BASE, f'outputs/rolling/cutoff_{args.cutoff_year}_multiobs'
              f'_k{args.k_draws}_h{args.label_horizon}')
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Device: {DEVICE}  |  cutoff: {args.cutoff_year}  |  '
          f'k={args.k_draws}  H={args.label_horizon}', flush=True)
    print(f'Sequences: {SEQ_DIR}', flush=True)
    print(f'Outputs:   {OUT_DIR}', flush=True)

    # ── Load sequences ────────────────────────────────────────────────────────
    # train_seq/train_mask are mmap'd (train_seq alone is 8.15GB). The
    # per-observation metadata arrays (labels, incl_prob) are small (~27MB
    # each for train) and loaded fully. train_prepay_timestep /
    # test_prepay_timestep are NEVER loaded — the builder fills them with -1
    # for every observation and its own docstring calls them meaningless here.
    def _mmap(name):
        return np.load(os.path.join(SEQ_DIR, name), mmap_mode='r')

    def _load(name):
        return np.load(os.path.join(SEQ_DIR, name))

    train_seq       = _mmap('train_seq.npy')
    train_mask      = _mmap('train_mask.npy')
    train_labels    = _load('train_labels.npy')
    train_incl_prob = _load('train_incl_prob.npy')

    test_seq    = _mmap('test_seq.npy')
    test_mask   = _mmap('test_mask.npy')
    test_labels = _load('test_labels.npy')

    print(f'train: {train_seq.shape} | test: {test_seq.shape}', flush=True)

    results = train_and_evaluate(
        train_seq, train_mask, train_labels, train_incl_prob,
        test_seq, test_mask, test_labels,
        n_epochs=args.n_epochs, pos_ratio=args.pos_ratio, use_ipw=args.use_ipw,
        out_dir=OUT_DIR,
    )

    results.update({
        'cutoff_year':    args.cutoff_year,
        'k_draws':        args.k_draws,
        'label_horizon':  args.label_horizon,
        'n_train':        int(len(train_seq)),
        'n_test':         int(len(test_seq)),
        'pos_ratio':      args.pos_ratio,
        'use_ipw':        args.use_ipw,
    })
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. cutoff={args.cutoff_year} | AUC={results['best_auc']:.4f} | "
          f"Platt a={results['platt_a']:.4f} b={results['platt_b']:.4f}", flush=True)


if __name__ == '__main__':
    main()
