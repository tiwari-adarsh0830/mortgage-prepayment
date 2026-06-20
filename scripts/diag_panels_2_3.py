"""
Panels 2 & 3 — model-side evidence for the pre-2020 training-regime finding.

Panel 2: refi-incentive sweep run against BOTH models on one axis.
  - old 21-vintage model (hazard_best.pt + sequences/scaler.pkl)
  - new 2013-2019 model (hazard_best_extended.pt + sequences_extended/scaler.pkl)
  Each model uses ITS OWN scaler (the scaler it was trained with).
  Reports raw (uncalibrated) annualized CPR vs refi incentive.
  Expectation: old model monotonically rising S-curve; new model flat/inverted.

Panel 3: raw refi-incentive distribution in each training set.
  Reads feature 0 from train_seq.npy (scaled), inverse-transforms with the
  matching scaler to recover raw refi incentive in percentage points, and
  prints distribution summary + histogram counts for both cohorts.
  Explains WHY: pre-2020 training mass sits at/below zero incentive.
"""
import numpy as np, torch, torch.nn as nn, os, pickle
import warnings
warnings.filterwarnings("ignore")

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
MAX_SEQ, N_FEATURES = 33, 9
DEAD_COLS = [7, 8]
REP = dict(credit_score=740.0, orig_ltv=75.0, current_ltv=70.0,
           orig_upb=250000.0, dti=35.0,
           loan_purpose_enc=0.0, property_type_enc=0.0)

MODELS = {
    'old_21vintage': dict(
        ckpt=os.path.join(OUT, "hazard_best.pt"),
        scaler=os.path.join(BASE, "data/sequences/scaler.pkl"),
        train_seq=os.path.join(BASE, "data/sequences/train_seq.npy")),
    'new_2013_2019': dict(
        ckpt=os.path.join(OUT, "hazard_best_extended.pt"),
        scaler=os.path.join(BASE, "data/sequences_extended/scaler.pkl"),
        train_seq=os.path.join(BASE, "data/sequences_extended/train_seq.npy")),
}

class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, n_heads=4, n_layers=2,
                 dim_ff=256, max_seq=33, dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
              dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer  = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier   = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x, mask=None, return_per_timestep=False):
        B, T, _ = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out  = self.input_proj(x) + self.pos_embedding(pos)
        pad  = ~mask if mask is not None else None
        out  = self.transformer(out, src_key_padding_mask=pad)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(1) / real.sum(1).clamp(min=1)
        else:
            out = out.mean(1)
        return self.classifier(out).squeeze(-1)

def load_model(ckpt):
    ck  = torch.load(ckpt, map_location="cpu")
    cfg = ck.get("config", {})
    m   = PrepaymentTransformer(
        input_dim=N_FEATURES, d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.1))
    m.load_state_dict(ck["model_state"]); m.eval()
    return m

def build_batch(refi, n=500):
    s = np.zeros((n, MAX_SEQ, N_FEATURES), dtype=np.float32)
    s[:, :, 0] = refi
    s[:, :, 1] = REP["credit_score"]; s[:, :, 2] = REP["orig_ltv"]
    s[:, :, 3] = REP["current_ltv"];  s[:, :, 4] = REP["orig_upb"]
    s[:, :, 5] = np.arange(1, MAX_SEQ+1)[None, :]
    s[:, :, 6] = REP["dti"]
    s[:, :, 7] = REP["loan_purpose_enc"]; s[:, :, 8] = REP["property_type_enc"]
    return s

def raw_cpr(model, scaler, refi, n=500):
    seqs = build_batch(refi, n)
    flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(n, MAX_SEQ, N_FEATURES)
    for c in DEAD_COLS:
        flat[:, :, c] = 0.0
    x    = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(n, MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        logit = model(x, mask=mask, return_per_timestep=True).numpy()
    haz = (1.0 / (1.0 + np.exp(-logit))).mean()
    return 1.0 - (1.0 - haz) ** 12

def panel2():
    print("="*72)
    print("PANEL 2 — Raw (uncalibrated) model CPR vs refi incentive, both models")
    print("="*72)
    incentives = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    loaded = {k: (load_model(v['ckpt']), pickle.load(open(v['scaler'],'rb')))
              for k, v in MODELS.items()}
    print(f"{'refi%':>7} {'old_CPR%':>12} {'new_CPR%':>12}")
    print("-"*34)
    for ri in incentives:
        o = raw_cpr(*loaded['old_21vintage'], ri) * 100
        n = raw_cpr(*loaded['new_2013_2019'], ri) * 100
        print(f"{ri:>7.1f} {o:>12.3f} {n:>12.3f}")
    print("\n  old model should rise monotonically with refi incentive (correct S-curve)")
    print("  new model is expected flat/inverted (no refi signal in 2013-2019 training)")

def panel3():
    print("\n" + "="*72)
    print("PANEL 3 — Raw refi-incentive distribution in each training set")
    print("="*72)
    edges = np.array([-np.inf,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,np.inf])
    lbls  = ['<-2','-2..-1.5','-1.5..-1','-1..-0.5','-0.5..0',
             '0..0.5','0.5..1','1..1.5','1.5..2','>2']
    for name, cfg in MODELS.items():
        scaler = pickle.load(open(cfg['scaler'],'rb'))
        seq    = np.load(cfg['train_seq'], mmap_mode='r')
        mask   = np.load(cfg['train_seq'].replace('train_seq','train_mask'), mmap_mode='r')
        mu, sd = scaler.mean_[0], np.sqrt(scaler.var_[0])
        N = seq.shape[0]
        idx = np.sort(np.random.default_rng(0).choice(N, size=min(N, 200000), replace=False))
        z_all = seq[idx, :, 0].astype(np.float64)      # (n, 33) scaled refi
        m_all = mask[idx, :].astype(bool)              # (n, 33) True = real timestep
        z = z_all[m_all]                               # keep only real timesteps
        raw = z * sd + mu
        print(f"\nCohort: {name}   mean={raw.mean():+.3f}  std={raw.std():.3f}  "
              f"median={np.median(raw):+.3f}")
        cnt, _ = np.histogram(raw, bins=edges)
        frac = cnt / cnt.sum()
        for l, c, fr in zip(lbls, cnt, frac):
            bar = '#' * int(fr*50)
            print(f"  {l:>10}: {fr*100:5.1f}%  {bar}")
        pos = (raw > 0).mean()
        print(f"  share with POSITIVE refi incentive (in-the-money): {pos*100:.1f}%")

if __name__ == "__main__":
    panel2()
    panel3()
