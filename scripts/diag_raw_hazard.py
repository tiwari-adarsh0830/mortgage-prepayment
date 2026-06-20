"""
Diagnostic: does the extended-trained hazard model produce cross-incentive
spread in raw (uncalibrated) hazards? Reuses the exact synthetic-loan
construction from stage2_forecast_cpr_timeseries.py.

Prints, for a sweep of refi incentives:
  - raw mean per-timestep hazard (sigmoid of raw logit, no Platt)
  - raw mean logit
  - calibrated SMM and annualized CPR using the extended Platt (a,b)

If RAW hazard is flat across incentive  -> model is weak (retrain strategy).
If RAW hazard varies but CALIBRATED is flat -> Platt is the problem (refit).
"""
import numpy as np, torch, torch.nn as nn, os, json, pickle

BASE   = "/scratch/at7095/mortgage_prepayment"
OUT    = os.path.join(BASE, "outputs")
SEQ    = os.path.join(BASE, "data/sequences_extended")
CKPT   = os.path.join(OUT,  "hazard_best_extended.pt")
SCALER = os.path.join(SEQ,  "scaler.pkl")
CALIB  = os.path.join(OUT,  "hazard_calibration_extended.json")

MAX_SEQ, N_FEATURES = 33, 9
DEAD_COLS = [7, 8]
REP = dict(credit_score=740.0, orig_ltv=75.0, current_ltv=70.0,
           orig_upb=250000.0, dti=35.0,
           loan_purpose_enc=0.0, property_type_enc=0.0)

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

def load_model():
    ck  = torch.load(CKPT, map_location="cpu")
    cfg = ck.get("config", {})
    m   = PrepaymentTransformer(
        input_dim=N_FEATURES, d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.1))
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m

def build_batch_constant_refi(refi_incentive, n_paths=500):
    s = np.zeros((n_paths, MAX_SEQ, N_FEATURES), dtype=np.float32)
    s[:, :, 0] = refi_incentive
    s[:, :, 1] = REP["credit_score"]
    s[:, :, 2] = REP["orig_ltv"]
    s[:, :, 3] = REP["current_ltv"]
    s[:, :, 4] = REP["orig_upb"]
    s[:, :, 5] = np.arange(1, MAX_SEQ+1)[None, :]
    s[:, :, 6] = REP["dti"]
    s[:, :, 7] = REP["loan_purpose_enc"]
    s[:, :, 8] = REP["property_type_enc"]
    return s

def main():
    model  = load_model()
    scaler = pickle.load(open(SCALER, "rb"))
    calib  = json.load(open(CALIB))
    a, b   = calib["a"], calib["b"]
    print(f"Platt (extended): a={a:.4f}  b={b:.4f}\n")

    print(f"{'refi%':>7} {'raw_logit':>11} {'raw_hazard':>12} {'raw_CPR%':>10} "
          f"{'calib_SMM':>11} {'calib_CPR%':>11}")
    print("-" * 70)

    incentives = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    for ri in incentives:
        seqs = build_batch_constant_refi(ri, n_paths=500)
        flat = scaler.transform(seqs.reshape(-1, N_FEATURES)).reshape(500, MAX_SEQ, N_FEATURES)
        for c in DEAD_COLS:
            flat[:, :, c] = 0.0
        x    = torch.tensor(flat, dtype=torch.float32)
        mask = torch.ones(500, MAX_SEQ, dtype=torch.bool)
        with torch.no_grad():
            logit = model(x, mask=mask, return_per_timestep=True).numpy()

        raw_logit  = logit.mean()
        raw_hazard = (1.0 / (1.0 + np.exp(-logit))).mean()
        raw_cpr    = 1.0 - (1.0 - raw_hazard) ** 12
        calib_smm  = (1.0 / (1.0 + np.exp(-(a * logit + b)))).mean()
        calib_cpr  = 1.0 - (1.0 - calib_smm) ** 12

        print(f"{ri:>7.1f} {raw_logit:>11.4f} {raw_hazard:>12.6f} {raw_cpr*100:>9.3f} "
              f"{calib_smm:>11.6f} {calib_cpr*100:>10.3f}")

    print("\nInterpretation:")
    print("  RAW_CPR rising sharply with refi%  -> model learned the S-curve (good)")
    print("  RAW_CPR flat across refi%          -> model is weak (retrain)")
    print("  RAW varies but CALIB flat          -> Platt compression issue (refit)")

if __name__ == "__main__":
    main()
