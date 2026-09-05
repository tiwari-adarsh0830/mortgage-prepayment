"""diag_age_stratified_auc.py — is the multiobs model's 0.7847 AUC driven by
age_at_ref instead of by the incentive/burnout signal it's meant to capture?

Loads hazard_best.pt (outputs/rolling/cutoff_2020_multiobs_k5_h1/) and scores
the test split from data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1/
with the exact forward pass evaluate() uses in train_hazard_multiobs.py:
pooled logit via model(seq, mask) (no return_per_timestep), score=sigmoid(logit).

Then:
  1. Overall AUC as a sanity check against results.json's 0.7847.
  2. age_at_ref quintiles (5 roughly-equal-count bins). Within-bin AUC for
     each, printed with age range / n / n positive.
  3. Within each bin: Spearman(score, incentive_at_ref) over all observations
     in the bin, and separately over just the label==1 observations in the
     bin.
  4. A verdict line on whether within-bin AUC survives age-stratification or
     collapses toward 0.5 (which would say age, not incentive/burnout, was
     doing the discriminating).

Read-only: loads hazard_best.pt and the test_* arrays, writes nothing,
retrains nothing. test_seq/test_mask are mmap'd (test_seq is ~2GB); nothing
else about the data pipeline is touched.

Needs a GPU-having node in practice — 1.7M observations through the full
transformer is a real inference workload, not a login-node smoke test (same
reasoning as diag_multiobs_onevintage.py's docstring), but falls back to CPU
if none is visible so it still runs correctly, just slower.

Run (on a GPU node, e.g. via srun/sbatch):
    cd /scratch/at7095/mortgage_prepayment
    python scripts/diag/diag_age_stratified_auc.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE      = '/scratch/at7095/mortgage_prepayment'
CKPT_DIR  = os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1')
SEQ_DIR   = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1')
N_FEATURES = 9
N_BINS     = 5
EVAL_BATCH = 2048
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model — identical architecture to train_hazard_multiobs.py ────────────────

class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim: int = N_FEATURES, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_ff: int = 256, max_seq: int = 33, dropout: float = 0.1):
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

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        pos      = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out      = self.input_proj(x) + self.pos_embedding(pos)
        pad_mask = ~mask if mask is not None else None
        out      = self.transformer(out, src_key_padding_mask=pad_mask)
        real     = mask.float().unsqueeze(-1)
        out      = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        return self.classifier(out).squeeze(-1)


def score_all(model, seq, mask, batch_size: int = EVAL_BATCH) -> np.ndarray:
    """Same scoring as train_hazard_multiobs.evaluate(): sigmoid(pooled logit),
    no return_per_timestep, no survival-CDF aggregation."""
    model.eval()
    n      = len(mask)
    scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bs = torch.tensor(np.asarray(seq[i:i + batch_size]),  device=DEVICE)
            bm = torch.tensor(np.asarray(mask[i:i + batch_size]), device=DEVICE)
            logits = model(bs, mask=bm)
            scores[i:i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return scores


def safe_auc(labels, scores) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def safe_spearman(x, y):
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None, None
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def main():
    print(f'Device: {DEVICE}', flush=True)

    ckpt = torch.load(os.path.join(CKPT_DIR, 'hazard_best.pt'),
                       map_location='cpu', weights_only=False)
    cfg  = ckpt['config']
    print(f"Checkpoint: epoch={ckpt['epoch']}  reported_auc={ckpt['auc']:.4f}  config={cfg}", flush=True)

    model = PrepaymentTransformer(
        input_dim=cfg['input_dim'], d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], dim_ff=cfg['dim_ff'], max_seq=cfg['max_seq'],
        dropout=cfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])

    def _mmap(name):
        return np.load(os.path.join(SEQ_DIR, name), mmap_mode='r')

    def _load(name):
        return np.load(os.path.join(SEQ_DIR, name))

    test_seq       = _mmap('test_seq.npy')
    test_mask      = _mmap('test_mask.npy')
    test_labels    = _load('test_labels.npy')
    test_age       = _load('test_age_at_ref.npy')
    test_incentive = _load('test_incentive_at_ref.npy')

    n = len(test_labels)
    print(f'Test set: {n:,} observations  |  positive rate: {test_labels.mean():.4%}', flush=True)

    scores = score_all(model, test_seq, test_mask)

    # ── 1. Overall AUC sanity check ────────────────────────────────────────
    overall_auc = roc_auc_score(test_labels, scores)
    print(f'\n=== Overall AUC ===')
    print(f'Overall AUC: {overall_auc:.4f}  (results.json reports 0.7847)')

    # ── 2/3. Age quintiles ──────────────────────────────────────────────────
    bin_edges = np.quantile(test_age, np.linspace(0, 1, N_BINS + 1))
    bin_edges = np.unique(bin_edges)  # guard against degenerate duplicate edges
    bin_idx   = np.digitize(test_age, bin_edges[1:-1], right=True)

    print(f'\n=== Age-stratified AUC ({len(bin_edges) - 1} bins) ===')
    print(f"{'bin':>3}  {'age range':>16}  {'n':>10}  {'n_pos':>8}  {'pos_rate':>9}  "
          f"{'AUC':>7}  {'spearman(score,incentive)':>27}  {'spearman|label=1':>18}")

    bin_aucs = []
    for b in range(len(bin_edges) - 1):
        mask_b = bin_idx == b
        n_b    = int(mask_b.sum())
        if n_b == 0:
            continue
        labels_b = test_labels[mask_b]
        scores_b = scores[mask_b]
        age_lo, age_hi = test_age[mask_b].min(), test_age[mask_b].max()
        n_pos = int(labels_b.sum())

        auc_b = safe_auc(labels_b, scores_b)
        bin_aucs.append(auc_b)

        rho_all, p_all = safe_spearman(scores_b, test_incentive[mask_b])

        pos_mask = labels_b == 1
        rho_pos, p_pos = safe_spearman(scores_b[pos_mask], test_incentive[mask_b][pos_mask])

        auc_str      = f'{auc_b:.4f}' if auc_b is not None else 'n/a (1 class)'
        rho_all_str  = f'{rho_all:+.4f} (p={p_all:.1e})' if rho_all is not None else 'n/a'
        rho_pos_str  = f'{rho_pos:+.4f} (p={p_pos:.1e}, n={int(pos_mask.sum())})' if rho_pos is not None else f'n/a (n={int(pos_mask.sum())})'

        print(f'{b:>3}  {f"[{age_lo:.0f}, {age_hi:.0f}]":>16}  {n_b:>10,}  {n_pos:>8,}  '
              f'{n_pos / n_b:>8.2%}  {auc_str:>7}  {rho_all_str:>27}  {rho_pos_str:>18}')

    # ── 4. Verdict ───────────────────────────────────────────────────────────
    valid_aucs = [a for a in bin_aucs if a is not None]
    print(f'\n=== Verdict ===')
    print(f'Overall AUC:        {overall_auc:.4f}')
    print(f'Within-bin AUC min: {min(valid_aucs):.4f}' if valid_aucs else 'Within-bin AUC min: n/a')
    print(f'Within-bin AUC max: {max(valid_aucs):.4f}' if valid_aucs else 'Within-bin AUC max: n/a')
    print(f'Within-bin AUC mean: {np.mean(valid_aucs):.4f}' if valid_aucs else 'Within-bin AUC mean: n/a')
    if valid_aucs:
        collapsed = [a for a in valid_aucs if a < 0.55]
        if collapsed:
            print(f'{len(collapsed)}/{len(valid_aucs)} bin(s) have AUC < 0.55 '
                  f'(near-chance once age is held fixed).')
        else:
            print('All bins retain AUC well above 0.5 — discrimination is not purely an '
                  'age proxy effect within this stratification.')

    print('\nDone.')


if __name__ == '__main__':
    main()
