"""
Health check on current_ltv (feature index 3) before the rolling t->t+1 build.
Confirms the equity signal is real so the equity x rate-incentive interaction
is learnable. Reads existing sequences, inverse-transforms feature 3 to raw LTV
using the matching scaler, and reports distribution + within-loan time variation.
"""
import numpy as np, pickle, os
import warnings; warnings.filterwarnings("ignore")

BASE = "/scratch/at7095/mortgage_prepayment"
SETS = {
    'production_21vintage': "data/sequences",
    'extended_2013_2019':   "data/sequences_extended",
}
LTV_IDX = 3   # FEATURE_COLS = [refi_incentive, credit_score, orig_ltv, current_ltv, ...]

for name, d in SETS.items():
    seqp = os.path.join(BASE, d, "train_seq.npy")
    mskp = os.path.join(BASE, d, "train_mask.npy")
    sclp = os.path.join(BASE, d, "scaler.pkl")
    if not (os.path.exists(seqp) and os.path.exists(sclp)):
        print(f"[{name}] missing files, skipping"); continue

    scaler = pickle.load(open(sclp, "rb"))
    mu, sd = scaler.mean_[LTV_IDX], np.sqrt(scaler.var_[LTV_IDX])
    seq = np.load(seqp, mmap_mode='r')
    msk = np.load(mskp, mmap_mode='r')

    N = seq.shape[0]
    idx = np.sort(np.random.default_rng(0).choice(N, size=min(N, 200000), replace=False))
    z   = seq[idx, :, LTV_IDX].astype(np.float64)   # (n,33) scaled
    m   = msk[idx, :].astype(bool)

    raw = z * sd + mu                               # recover raw current_ltv
    real = raw[m]                                   # real timesteps only

    print(f"\n=== {name} ===")
    print(f"scaler current_ltv mean={mu:.2f} std={sd:.2f}")
    print(f"raw current_ltv (real timesteps): "
          f"min={real.min():.1f} p5={np.percentile(real,5):.1f} "
          f"median={np.median(real):.1f} p95={np.percentile(real,95):.1f} max={real.max():.1f}")
    # equity buckets
    for lo, hi, lbl in [(-1e9,60,'<60 (high equity)'),(60,80,'60-80'),
                        (80,100,'80-100'),(100,1e9,'>100 (underwater)')]:
        frac = ((real>=lo)&(real<hi)).mean()*100
        print(f"  {lbl:>22}: {frac:5.1f}%")
    # within-loan time variation: std across a loan's own real timesteps
    n_show = min(5000, len(idx))
    within = []
    for i in range(n_show):
        vals = (z[i][m[i]] * sd + mu)
        if len(vals) >= 2:
            within.append(vals.std())
    within = np.array(within)
    print(f"within-loan LTV std (median over loans): {np.median(within):.2f}  "
          f"(frac loans with std>1: {(within>1).mean()*100:.1f}%)")
