"""diag_bin4_incentive_split.py — follow-up on diag_age_stratified_auc.py's
bin 4 (age_at_ref in [37, 109]), where spearman(score, incentive_at_ref) over
all observations went negative (-0.0730) but stayed positive (+0.5022)
restricted to label==1.

Same model, same checkpoint, same bin construction (age quintiles via
np.quantile + np.digitize) as diag_age_stratified_auc.py, so bin 4 here is
the identical set of observations. Within bin 4 only:
  - split into label==0 / label==1
  - for each: n, mean incentive_at_ref, mean score, spearman(score, incentive)
  - among label==0: fraction with incentive_at_ref above the BIN'S median
    incentive ("should have refinanced but didn't" candidates), and their
    mean score vs. the rest of label==0

Read-only: loads hazard_best.pt and the test_* arrays, writes nothing,
retrains nothing. No interpretation printed — numbers only.

Run (GPU node):
    cd /scratch/at7095/mortgage_prepayment
    python scripts/diag/diag_bin4_incentive_split.py
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


def main():
    print(f'Device: {DEVICE}', flush=True)

    ckpt = torch.load(os.path.join(CKPT_DIR, 'hazard_best.pt'),
                       map_location='cpu', weights_only=False)
    cfg  = ckpt['config']

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

    scores = score_all(model, test_seq, test_mask)

    # ── Reproduce diag_age_stratified_auc.py's bin construction exactly ──────
    bin_edges = np.quantile(test_age, np.linspace(0, 1, N_BINS + 1))
    bin_edges = np.unique(bin_edges)
    bin_idx   = np.digitize(test_age, bin_edges[1:-1], right=True)

    bin4 = bin_idx == (len(bin_edges) - 2)  # last bin
    age_lo, age_hi = test_age[bin4].min(), test_age[bin4].max()
    n_bin4 = int(bin4.sum())
    print(f'\nBin 4: age_at_ref in [{age_lo:.0f}, {age_hi:.0f}]  n={n_bin4:,}')

    labels_b    = test_labels[bin4]
    scores_b    = scores[bin4]
    incentive_b = test_incentive[bin4]

    print(f'\n=== Label subsets within bin 4 ===')
    for lbl in (0, 1):
        m = labels_b == lbl
        n_l = int(m.sum())
        mean_incentive = float(incentive_b[m].mean())
        mean_score     = float(scores_b[m].mean())
        rho, p = spearmanr(scores_b[m], incentive_b[m])
        print(f'label={lbl}: n={n_l:,}  mean_incentive_at_ref={mean_incentive:.6f}  '
              f'mean_score={mean_score:.6f}  spearman(score,incentive)={rho:+.6f} (p={p:.3e})')

    # ── "Should have refinanced but didn't": label==0 with incentive above bin median ──
    bin_median_incentive = float(np.median(incentive_b))
    print(f'\nBin 4 median incentive_at_ref (all observations): {bin_median_incentive:.6f}')

    neg_mask       = labels_b == 0
    n_neg          = int(neg_mask.sum())
    above_med_mask = neg_mask & (incentive_b > bin_median_incentive)
    n_above_med    = int(above_med_mask.sum())
    frac_above_med = n_above_med / n_neg

    mean_score_above = float(scores_b[above_med_mask].mean())
    rest_neg_mask    = neg_mask & ~above_med_mask
    n_rest           = int(rest_neg_mask.sum())
    mean_score_rest  = float(scores_b[rest_neg_mask].mean())

    print(f'\n=== label=0 split by incentive relative to bin median ===')
    print(f'label=0 total: n={n_neg:,}')
    print(f'label=0 with incentive_at_ref > bin median ("should have refi but didn\'t"): '
          f'n={n_above_med:,}  fraction={frac_above_med:.6f}  mean_score={mean_score_above:.6f}')
    print(f'label=0 with incentive_at_ref <= bin median (rest): '
          f'n={n_rest:,}  mean_score={mean_score_rest:.6f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
