"""
PROOF OF CONCEPT — population forward forecast (object C).

Goal: confirm that forecasting over a REAL loan population (instead of one
representative loan) with the ORIGINAL loan-level Platt closes most of the
boom under-prediction, WITHOUT any recalibration.

Design (honest DER forecast leg, given only month t's PMMS):
  For a target month t and coupon c:
    1. Sample N real loans from the test set whose note_rate maps to coupon c.
       (note_rate recovered per loan; see below.)
    2. Build each loan's 33-month sequence under a FLAT-rate assumption:
       incentive held at (note_rate_loan - PMMS[t]) for all timesteps, but
       all OTHER features (credit, ltv, upb, dti, purpose, ptype) taken from
       that loan's REAL origination values. Age runs 1..33 as usual.
       -> This is a forecast: "given today's rate, what CPR do these loans
          produce over the next 33 months if rates stay flat."
    3. Run model -> per-timestep hazard -> ORIGINAL Platt -> SMM -> annualize.
    4. Balance-weight across loans by original_upb to get coupon CPR.

Compare against v6 realized at the same (coupon, month) cells, for one boom
month and one trough month. If forecast lands in the right ballpark of
realized (same order of magnitude, correct cross-section) with NO recalibration,
object C is validated and we build the full version.

Recovering per-loan note_rate:
  incentive[t] (raw) = note_rate - PMMS[t].  The sequences store SCALED
  incentive. We invert: raw_inc[t] = scaled_inc[t]*scale0 + mean0, then
  note_rate = raw_inc[t] + PMMS[at that loan's calendar month t].
  Simpler & robust: recover note_rate from raw incentive at age 0 using the
  loan's KNOWN origination PMMS. But we don't have per-loan calendar dates in
  the sequence. Instead we recover note_rate directly: for each loan, the
  feature set is FIXED except incentive/age, so we read the loan's raw
  static features (credit/ltv/upb/dti) from the scaled seq via the scaler,
  and we recover note_rate from the FIRST-timestep raw incentive plus the
  loan's origination-month PMMS via test_loan_ids -> origination month.

  ** For the POC we avoid the calendar join: we recover note_rate from the
     median raw incentive + median PMMS over the panel, which is good enough
     to BUCKET loans into coupons for a sanity check. The full version will do
     the exact per-loan calendar join. **
"""
import os, sys, json, pickle
import numpy as np
import torch

BASE = "/scratch/at7095/mortgage_prepayment"
OUT  = os.path.join(BASE, "outputs")
SEQ  = os.path.join(BASE, "data/sequences")
sys.path.insert(0, os.path.join(BASE, "scripts"))

from stage2_forecast_cpr_gfee050 import load_model, MAX_SEQ, N_FEATURES, DEAD_COLS, GFEE

CALIB = os.path.join(OUT, "hazard_calibration.json")   # ORIGINAL loan-level Platt
N_SAMPLE = 3000          # loans per coupon bucket
COUPONS  = [3.0, 3.5, 4.0, 4.5, 5.0]

# Boom and trough test months to check, with their actual PMMS
# (PMMS values from pmms_monthly.csv; flat-rate forecast anchor)
CHECK = {
    "boom_2021-01":   2.74,   # Jan 2021 PMMS ~2.74%
    "trough_2023-01": 6.48,   # Jan 2023 PMMS ~6.48%
}

def annualize(smm):
    return 1.0 - (1.0 - smm) ** 12

def main():
    print("Loading model, scaler, ORIGINAL Platt...", flush=True)
    model  = load_model()
    scaler = pickle.load(open(os.path.join(SEQ, "scaler.pkl"), "rb"))
    cal    = json.load(open(CALIB)); a, b = cal["a"], cal["b"]
    print(f"  Platt a={a:.4f} b={b:.4f}  (original loan-level, UNCHANGED)", flush=True)

    mean0, scale0 = scaler.mean_[0], scaler.scale_[0]   # incentive
    print(f"  incentive scaler: mean={mean0:.3f} scale={scale0:.3f}", flush=True)

    print("Loading test sequences (mmap)...", flush=True)
    seq  = np.load(os.path.join(SEQ, "test_seq.npy"),  mmap_mode="r")
    mask = np.load(os.path.join(SEQ, "test_mask.npy"), mmap_mode="r")
    n    = seq.shape[0]

    # --- recover per-loan note_rate (POC approximation) ---
    # Use each loan's FIRST real timestep raw incentive + a panel-median PMMS
    # anchor just to BUCKET into coupons. (Full version: exact calendar join.)
    PMMS_ANCHOR = 3.0   # rough median PMMS over 2020-2023 for bucketing only
    SAMP = 400000       # subsample for speed in POC
    idx  = np.random.RandomState(0).choice(n, size=min(SAMP, n), replace=False)
    idx.sort()
    s0   = np.asarray(seq[idx, 0, 0])            # scaled incentive at age 0
    raw_inc0 = s0 * scale0 + mean0
    note_rate = raw_inc0 + PMMS_ANCHOR           # approx note rate per loan
    coupon_est = np.round((note_rate - GFEE) * 2) / 2   # nearest 0.5 coupon

    # static raw features per sampled loan (from scaled seq via scaler)
    static_scaled = np.asarray(seq[idx, 0, :])   # (S, 9) at age 0
    static_raw = static_scaled * scaler.scale_ + scaler.mean_
    upb_raw = static_raw[:, 4]

    print("Coupon bucket counts (POC approx):", flush=True)
    for c in COUPONS:
        print(f"  {c}: {int((coupon_est==c).sum())}", flush=True)

    def forecast_coupon(c, pmms_t):
        sel = np.where(coupon_est == c)[0]
        if len(sel) == 0:
            return np.nan
        take = np.random.RandomState(1).choice(sel, size=min(N_SAMPLE, len(sel)), replace=False)
        # Build flat-rate forecast sequences: real static features, age 1..33,
        # incentive constant at (note_rate_loan - pmms_t).
        nr   = note_rate[take]                       # (k,)
        inc  = (nr - pmms_t)[:, None]                # (k,1) raw incentive, flat
        k    = len(take)
        # start from each loan's real scaled static row, overwrite inc + age
        base = np.asarray(seq[idx[take], 0, :])[:, None, :].repeat(MAX_SEQ, axis=1)  # (k,33,9) scaled
        # overwrite incentive (col0) with flat raw->scaled
        base[:, :, 0] = (inc - mean0) / scale0
        # overwrite age (col5) with 1..33 scaled
        ages = np.arange(1, MAX_SEQ + 1)[None, :]
        base[:, :, 5] = (ages - scaler.mean_[5]) / scaler.scale_[5]
        for dc in DEAD_COLS:
            base[:, :, dc] = 0.0
        x  = torch.tensor(base, dtype=torch.float32)
        mk = torch.ones(k, MAX_SEQ, dtype=torch.bool)
        with torch.no_grad():
            logit = model(x, mask=mk, return_per_timestep=True).numpy()  # (k,33)
        smm = 1.0 / (1.0 + np.exp(-(a * logit + b)))
        cpr_loan = annualize(smm.mean(axis=1))       # per-loan annualized CPR
        w = upb_raw[take]; w = np.clip(w, 1, None)
        return float(np.average(cpr_loan, weights=w)) # balance-weighted

    for label, pmms_t in CHECK.items():
        print(f"\n=== {label}  PMMS={pmms_t}%  (ORIGINAL Platt, balance-weighted) ===", flush=True)
        print(f"{'coupon':>7} {'fcst_CPR':>9}", flush=True)
        for c in COUPONS:
            fc = forecast_coupon(c, pmms_t)
            print(f"{c:>7} {fc:>9.4f}", flush=True)

    print("\nCompare these to v6 realized:")
    print("  boom 2021-01 realized 3.5% ~0.34;  trough 2023-01 realized 3.5% ~0.07")
    print("If boom forecast is now ~0.15-0.30 (not 0.03), object C works with NO recalibration.")

if __name__ == "__main__":
    main()
