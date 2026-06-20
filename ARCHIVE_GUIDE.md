# Archive Guide — Complete Project Deliverable

This repository is the complete archive of the mortgage prepayment / DER
project: code, documentation, input data, trained models, and outputs.
The only items NOT in the repo are the large regenerable arrays and the
raw Fannie Mae files (see "Not included" below).

## Where things are

**Code** — `scripts/` (production pipeline), root `.py` (legacy/baseline)
  Key scripts:
  - prepare_sequences.py / prepare_sequences_extended.py — build model input
  - train_hazard.py — Transformer discrete-time hazard model
  - calibrate_hazard.py — Platt calibration
  - stage2_der_betas.py — DER time-varying betas (Eq. 5–6)
  - stage2_forecast_cpr_timeseries.py — forecast CPR per coupon
  - stage3_der_regression_v2.py — Fama-MacBeth cross-section
  - realized_cpr_v5.py — realized CPR by coupon (global 3-pass)
  - realized_cpr_by_refi_v1.py — realized CPR by refi-incentive bin
  - diag_raw_hazard.py / diag_panels_2_3.py — Phase 15 diagnostics

**Documentation** — `docs/` and `README.md`
  - README.md — full phase history (Phases 1–15), results, bug log
  - docs/DER_methodology_note.md — DER framework mapping, results, limitations
  - docs/economic_magnitude_comparison.md — lambda vs DER paper (in progress)

**Input data** — `data/`
  - pmms_monthly.csv — Freddie PMMS 30yr (refi incentive input)
  - zhvi_zip3.csv — Zillow ZHVI at zip3 (dynamic LTV input), covers 2000–2025
  - fncl_tba_prices_clean.xlsx — Bloomberg FNCL 2.5–6.5 TBA prices (Bobst pull)
  - treasury_yields_clean.xlsx — UST 5yr/10yr (Treasury hedge)
  - tba_roll_snapshot.xlsx — TBA roll/drop snapshot, June 2026
  - historicalweeklydata.xlsx — supporting historical series
  NOTE: the Bloomberg .xlsx files are hand-built terminal pulls and are
  not reproducible without terminal access — they are the authoritative copy.

**Trained models** — `outputs/`
  - hazard_best.pt — production 21-vintage model (AUC 0.7999)
  - hazard_best_extended.pt — Phase 15 pre-2020 model (AUC 0.7728)
  - hazard_calibration*.json — Platt coefficients

**Results / outputs** — `outputs/`
  - der_betas.csv, stage3_lambda_ts.csv — DER betas + Fama-MacBeth lambdas
  - forecast_cpr_timeseries.csv, forecast_vs_realized_cpr.csv — forecast vs realized
  - realized_cpr_by_coupon_v5.csv — realized CPR panel
  - realized_cpr_by_refi_v1.csv / _nocap.csv — Phase 15 refi-bin analysis
  - forecast_vs_realized_cpr_2020.png — headline plot

## Not included (regenerable)

- **Sequence arrays** (`data/sequences*/`, ~29 GB): model input tensors.
  Regenerate by running prepare_sequences.py (21-vintage) or
  prepare_sequences_extended.py (pre-2020) on the raw data.
- **Raw Fannie Mae data** (~231 GB): Single-Family Loan Performance Data,
  vintages 2013Q1–2023Q1. Download free from
  https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data
  (Primary Dataset, one zip per quarter; 113-col pipe-delimited, no header).

## Reproduction path

1. Download raw Fannie vintages (link above) to data/raw/
2. prepare_sequences.py  -> data/sequences/
3. train_hazard.py       -> outputs/hazard_best.pt
4. calibrate_hazard.py   -> outputs/hazard_calibration.json
5. stage2_*.py, stage3_* -> DER betas, forecast CPR, lambdas
   (PMMS, ZHVI, Bloomberg inputs already in data/)
