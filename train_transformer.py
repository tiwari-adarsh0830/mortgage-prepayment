"""
Job 2 — GPU only
Loads preprocessed .npy sequences from disk, trains Transformer.
Run this after prepare_sequences.py has completed.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import os
import json
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR   = '/scratch/at7095/mortgage_prepayment/data/sequences'
OUTPUT_DIR = '/scratch/at7095/mortgage_prepayment/outputs'

MAX_SEQ_LEN = 33
N_FEATURES  = 6


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class PrepayDataset(Dataset):
    def __init__(self, sequences, masks, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.masks     = torch.BoolTensor(masks)
        self.labels    = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx], self.labels[idx]


# ── Transformer Model ─────────────────────────────────────────────────────────
class PrepayTransformer(nn.Module):
    """
    Transformer encoder for mortgage prepayment prediction.

    Architecture:
        1. Linear projection: N_FEATURES -> d_model
        2. Learnable positional encoding (one embedding per timestep position)
        3. Transformer encoder (n_layers, n_heads)
        4. Mean pooling over real (unmasked) timesteps
        5. Linear classifier -> logit (sigmoid applied at inference)
    """
    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()

        self.input_proj    = nn.Linear(N_FEATURES, d_model)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)   # raw logit — BCEWithLogitsLoss applies sigmoid internally
        )

    def forward(self, x, mask):
        """
        x:    (batch, seq_len, N_FEATURES)
        mask: (batch, seq_len) bool — True=real, False=padding

        PyTorch src_key_padding_mask: True = IGNORE (padding), so we invert.
        """
        seq_len = x.shape[1]

        x = self.input_proj(x)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        padding_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Mean pool over real timesteps only
        mask_exp = mask.unsqueeze(-1).float()
        x_pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)

        return self.classifier(x_pooled).squeeze(-1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    # Load preprocessed sequences from disk
    print('Loading sequences from disk...', flush=True)
    train_seq    = np.load(os.path.join(SAVE_DIR, 'train_seq.npy'))
    train_mask   = np.load(os.path.join(SAVE_DIR, 'train_mask.npy'))
    train_labels = np.load(os.path.join(SAVE_DIR, 'train_labels.npy'))
    test_seq     = np.load(os.path.join(SAVE_DIR, 'test_seq.npy'))
    test_mask    = np.load(os.path.join(SAVE_DIR, 'test_mask.npy'))
    test_labels  = np.load(os.path.join(SAVE_DIR, 'test_labels.npy'))

    print(f'  Train: {train_seq.shape} | Test: {test_seq.shape}', flush=True)
    print(f'  Train prepay rate: {train_labels.mean()*100:.2f}%', flush=True)
    print(f'  Test prepay rate:  {test_labels.mean()*100:.2f}%', flush=True)

    # DataLoaders
    train_loader = DataLoader(
        PrepayDataset(train_seq, train_mask, train_labels),
        batch_size=512, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        PrepayDataset(test_seq, test_mask, test_labels),
        batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model    = PrepayTransformer(d_model=64, n_heads=4, n_layers=2, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel parameters: {n_params:,}', flush=True)

    # Loss — weight positive class for imbalance
    n_neg      = int((train_labels == 0).sum())
    n_pos      = int((train_labels == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print('\nTraining Transformer...', flush=True)
    best_auc = 0.0

    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for seqs, masks, labels in train_loader:
            seqs   = seqs.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(seqs, masks)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for seqs, masks, labels in test_loader:
                seqs  = seqs.to(device)
                masks = masks.to(device)
                probs = torch.sigmoid(model(seqs, masks)).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

        auc      = roc_auc_score(all_labels, all_probs)
        avg_loss = total_loss / len(train_loader)
        is_best  = auc > best_auc

        if is_best:
            best_auc = auc
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'auc':         best_auc,
                'config': {
                    'd_model':  64,
                    'n_heads':  4,
                    'n_layers': 2,
                    'dropout':  0.1,
                }
            }, os.path.join(OUTPUT_DIR, 'transformer_best.pt'))

        print(f'  Epoch {epoch+1:02d}/20 | loss: {avg_loss:.4f} | AUC: {auc:.4f}'
              f'{" [best]" if is_best else ""}', flush=True)

    print('\n' + '='*50)
    print('TRANSFORMER RESULT')
    print('='*50)
    print(f'  Best AUC: {best_auc:.4f}')
    print('='*50)

    results = {'Transformer': best_auc}
    with open(os.path.join(OUTPUT_DIR, 'results_transformer.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_DIR}/results_transformer.json')


if __name__ == '__main__':
    main()
