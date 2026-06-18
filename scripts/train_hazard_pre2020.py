"""
Hazard Model Training — PRE-2020 (out-of-sample validation)
-----------------------------------------------------------
Identical architecture/config to train_hazard.py, but trains ONLY on
2018Q1-2019Q4 sequences. The model never sees the 2020-21 refi boom.
Used to forecast CPR into 2020-21 as a clean out-of-sample test (advisor option b).

Reads:  data/sequences_pre2020/
Writes: outputs/hazard_pre2020_best.pt, outputs/results_hazard_pre2020.json
Does NOT touch the 21-vintage model (outputs/hazard_best.pt).
"""

import numpy as np
import torch
import torch.nn as nn
import os
import json
import time

BASE      = "/scratch/at7095/mortgage_prepayment"
SEQ_DIR   = os.path.join(BASE, "data/sequences_pre2020")   # <-- pre-2020 sequences
OUTPUTS   = os.path.join(BASE, "outputs")

CKPT_NAME    = "hazard_pre2020_best.pt"        # <-- separate checkpoint
RESULTS_NAME = "results_hazard_pre2020.json"   # <-- separate results

TRAIN_SEQ    = os.path.join(SEQ_DIR, "train_seq.npy")
TRAIN_MASK   = os.path.join(SEQ_DIR, "train_mask.npy")
TRAIN_PREPAY = os.path.join(SEQ_DIR, "train_prepay_timestep.npy")
TEST_SEQ     = os.path.join(SEQ_DIR, "test_seq.npy")
TEST_MASK    = os.path.join(SEQ_DIR, "test_mask.npy")
TEST_LABELS  = os.path.join(SEQ_DIR, "test_labels.npy")
TEST_PREPAY  = os.path.join(SEQ_DIR, "test_prepay_timestep.npy")

MAX_SEQ    = 33
N_FEATURES = 9
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 2048
N_EPOCHS    = 50
LR          = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP   = 1.0
STEPS_PER_EPOCH = 10000


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


class HazardSampler:
    def __init__(self, sequences, masks, prepay_timesteps):
        self.sequences = sequences
        self.masks     = masks
        self.prepay_t  = prepay_timesteps
        self.n_loans   = len(sequences)
        self.seq_lens  = masks.sum(axis=1).astype(np.int32)
        self.max_t = np.where(self.prepay_t >= 0, self.prepay_t,
                              self.seq_lens - 1).astype(np.int32)
        valid = self.max_t >= 0
        self.valid_idx   = np.where(valid)[0]
        self.prepaid_idx = np.where(self.prepay_t >= 0)[0]
        print(f"  HazardSampler: {len(self.valid_idx):,} valid loans out of {self.n_loans:,}")
        prepaid = (self.prepay_t >= 0).sum()
        print(f"  Prepaid loans: {prepaid:,} ({100*prepaid/self.n_loans:.2f}%)")

    def sample_batch(self, batch_size, rng):
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos
        pos_idx = rng.choice(self.prepaid_idx, size=n_pos, replace=True)
        neg_idx = rng.choice(self.valid_idx,   size=n_neg, replace=True)
        loan_idx = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(loan_idx)
        max_ts    = self.max_t[loan_idx]
        t_sampled = (rng.random(batch_size) * (max_ts + 1)).astype(np.int32)
        t_sampled = np.clip(t_sampled, 0, max_ts)
        prepay_ts = self.prepay_t[loan_idx]
        labels    = (t_sampled == prepay_ts).astype(np.float32)
        batch_seq  = np.zeros((batch_size, MAX_SEQ, N_FEATURES), dtype=np.float32)
        batch_mask = np.zeros((batch_size, MAX_SEQ), dtype=bool)
        for i in range(batch_size):
            t = t_sampled[i]; l = loan_idx[i]
            batch_seq[i, :t+1, :] = self.sequences[l, :t+1, :]
            batch_mask[i, :t+1]   = True
        return batch_seq, batch_mask, labels, t_sampled


def evaluate_hazard(model, test_seq, test_mask, test_labels, test_prepay_t, batch_size=512):
    from sklearn.metrics import roc_auc_score
    model.eval()
    n_loans     = len(test_seq)
    loan_scores = np.zeros(n_loans, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_loans, batch_size):
            batch_seq  = torch.tensor(test_seq[i:i+batch_size],  device=DEVICE)
            batch_mask = torch.tensor(test_mask[i:i+batch_size], device=DEVICE)
            logits = model(batch_seq, mask=batch_mask, return_per_timestep=True)
            h = torch.sigmoid(logits).cpu().numpy()
            active   = batch_mask.cpu().numpy()
            h_masked = np.where(active, h, 0.0)
            survival = np.prod(1.0 - h_masked, axis=1, where=active, initial=1.0)
            loan_scores[i:i+batch_size] = (1.0 - survival).astype(np.float32)
    auc = roc_auc_score(test_labels, loan_scores)
    return auc, loan_scores


def main():
    print(f"Device: {DEVICE}")
    print(f"SEQ_DIR: {SEQ_DIR}")
    print("Loading sequences...", flush=True)
    train_seq    = np.load(TRAIN_SEQ,    mmap_mode='r')
    train_mask   = np.load(TRAIN_MASK,   mmap_mode='r')
    train_prepay = np.load(TRAIN_PREPAY)
    test_seq     = np.load(TEST_SEQ,     mmap_mode='r')
    test_mask    = np.load(TEST_MASK,    mmap_mode='r')
    test_labels  = np.load(TEST_LABELS)
    test_prepay  = np.load(TEST_PREPAY)
    print(f"train_seq: {train_seq.shape}, prepay: {train_prepay.shape}")
    print(f"test_seq:  {test_seq.shape},  labels: {test_labels.shape}")

    print("\nBuilding hazard sampler...", flush=True)
    sampler = HazardSampler(train_seq, train_mask, train_prepay)

    n_prepaid   = (train_prepay >= 0).sum()
    avg_seq_len = train_mask.sum(axis=1).mean()
    pos_weight_val = 1.0
    print(f"pos_weight: {pos_weight_val:.1f} (n_prepaid={n_prepaid:,}, avg_seq={avg_seq_len:.1f})")

    model     = PrepaymentTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                    factor=0.5, patience=5, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=DEVICE))

    rng = np.random.default_rng(42)
    best_auc = 0.0
    results  = []

    print(f"\nTraining for {N_EPOCHS} epochs × {STEPS_PER_EPOCH} steps...", flush=True)
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(STEPS_PER_EPOCH):
            batch_seq, batch_mask, labels, _ = sampler.sample_batch(BATCH_SIZE, rng)
            x    = torch.tensor(batch_seq,  device=DEVICE)
            mask = torch.tensor(batch_mask, device=DEVICE)
            y    = torch.tensor(labels,     device=DEVICE)
            logits = model(x, mask=mask)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / STEPS_PER_EPOCH
        elapsed  = time.time() - t0
        auc, _ = evaluate_hazard(model, test_seq, test_mask, test_labels, test_prepay)
        results.append({'epoch': epoch, 'loss': avg_loss, 'auc': auc})
        print(f"Epoch {epoch:>2}/{N_EPOCHS}  loss={avg_loss:.5f}  auc={auc:.4f}  t={elapsed:.0f}s",
              flush=True)
        scheduler.step(auc)
        if auc > best_auc:
            best_auc = auc
            torch.save({
                'model_state': model.state_dict(),
                'config': {'n_heads': 4, 'n_layers': 2, 'd_model': 64, 'dropout': 0.1},
                'epoch': epoch, 'auc': auc,
            }, os.path.join(OUTPUTS, CKPT_NAME))
            print(f"  → New best AUC: {best_auc:.4f} — saved.", flush=True)

    with open(os.path.join(OUTPUTS, RESULTS_NAME), 'w') as f:
        json.dump({'best_auc': best_auc, 'history': results}, f, indent=2)
    print(f"\nDone. Best AUC: {best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
