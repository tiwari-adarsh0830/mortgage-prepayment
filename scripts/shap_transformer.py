"""
SHAP GradientExplainer for Transformer Prepayment Model
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os

BASE        = "/scratch/at7095/mortgage_prepayment"
SEQ_DIR     = os.path.join(BASE, "data/sequences")
OUTPUTS     = os.path.join(BASE, "outputs")
CKPT_PATH   = os.path.join(OUTPUTS, "transformer_best.pt")
TEST_SEQ    = os.path.join(SEQ_DIR, "test_seq.npy")
TEST_LABELS = os.path.join(SEQ_DIR, "test_labels.npy")
SHAP_OUT    = os.path.join(OUTPUTS, "shap_values.npy")
HEAT_OUT    = os.path.join(OUTPUTS, "shap_mean_heatmap.png")
BAR_OUT     = os.path.join(OUTPUTS, "shap_feature_importance.png")

FEATURE_NAMES = [
    "refi_incentive", "borrower_credit_score", "original_ltv",
    "current_ltv", "original_upb", "loan_age_months",
]
MAX_SEQ    = 33
N_FEATURES = 6
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=64, n_heads=4, n_layers=2,
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
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )

    def forward(self, x, src_key_padding_mask=None):
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out = self.input_proj(x) + self.pos_embedding(positions)
        out = self.transformer(out, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            real = (~src_key_padding_mask).float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_flat):
        x = x_flat.view(-1, MAX_SEQ, N_FEATURES)
        return self.model(x).unsqueeze(-1)


def main():
    print(f"Device: {DEVICE}")
    print("Loading test sequences...")
    test_seq    = np.load(TEST_SEQ)
    test_labels = np.load(TEST_LABELS)
    print(f"test_seq shape:    {test_seq.shape}")
    print(f"test_labels shape: {test_labels.shape}")
    print(f"Prepaid rate:      {test_labels.mean():.4f}")

    np.random.seed(42)
    pos_idx = np.where(test_labels == 1)[0]
    neg_idx = np.where(test_labels == 0)[0]
    print(f"Positive samples: {len(pos_idx)}, Negative samples: {len(neg_idx)}")

    bg_idx  = np.concatenate([
        np.random.choice(pos_idx, 250, replace=False),
        np.random.choice(neg_idx, 250, replace=False),
    ])
    exp_idx = np.concatenate([
        np.random.choice(pos_idx, 1000, replace=False),
        np.random.choice(neg_idx, 1000, replace=False),
    ])

    bg_tensor  = torch.tensor(
        test_seq[bg_idx].reshape(500, MAX_SEQ * N_FEATURES).astype(np.float32),
        device=DEVICE
    )
    exp_tensor = torch.tensor(
        test_seq[exp_idx].reshape(2000, MAX_SEQ * N_FEATURES).astype(np.float32),
        device=DEVICE
    )

    print("Loading model...")
    ckpt   = torch.load(CKPT_PATH, map_location=DEVICE)
    config = ckpt["config"]
    print(f"Checkpoint config: {config}")
    model  = PrepaymentTransformer(
        n_heads=config["n_heads"], n_layers=config["n_layers"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    wrapper = ModelWrapper(model)

    print("Running SHAP GradientExplainer...")
    explainer   = shap.GradientExplainer(wrapper, bg_tensor)
    shap_values = explainer.shap_values(exp_tensor)

    # Debug: print raw shape before any manipulation
    if isinstance(shap_values, list):
        print(f"SHAP returned list of length {len(shap_values)}, first element shape: {np.array(shap_values[0]).shape}")
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    print(f"Raw SHAP output shape: {shap_values.shape}")

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    print(f"SHAP shape after squeeze: {shap_values.shape}")  # should be (2000, 198)

    shap_3d = shap_values.reshape(2000, MAX_SEQ, N_FEATURES)
    print(f"SHAP 3D shape: {shap_3d.shape}")  # should be (2000, 33, 6)
    np.save(SHAP_OUT, shap_3d)
    print(f"Saved shap_values.npy")

    mean_abs = np.abs(shap_3d).mean(axis=0)   # (33, 6)

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mean_abs.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(MAX_SEQ))
    ax.set_xticklabels([str(i+1) for i in range(MAX_SEQ)], fontsize=7)
    ax.set_yticks(range(N_FEATURES))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=10)
    ax.set_xlabel("Loan Age (months)")
    ax.set_title("Mean |SHAP| — Transformer Prepayment Model")
    plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(HEAT_OUT, dpi=150)
    plt.close()
    print(f"Heatmap saved: {HEAT_OUT}")

    # Bar chart
    global_imp = np.abs(shap_3d).mean(axis=(0, 1))
    sorted_idx = np.argsort(global_imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx[::-1]],
        global_imp[sorted_idx[::-1]],
        color=plt.cm.YlOrRd(np.linspace(0.4, 0.9, N_FEATURES))
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance — Transformer")
    plt.tight_layout()
    plt.savefig(BAR_OUT, dpi=150)
    plt.close()
    print(f"Bar chart saved: {BAR_OUT}")

    print("\n── Global Feature Importance (ranked) ──")
    for rank, i in enumerate(sorted_idx):
        print(f"  {rank+1}. {FEATURE_NAMES[i]:<25}  {global_imp[i]:.5f}")

    print("\n── Peak importance timestep per feature ──")
    for j, fname in enumerate(FEATURE_NAMES):
        peak_t = np.argmax(mean_abs[:, j]) + 1
        print(f"  {fname:<25}  peak at month {peak_t:>2}  (|SHAP|={mean_abs[peak_t-1,j]:.5f})")


if __name__ == "__main__":
    main()
