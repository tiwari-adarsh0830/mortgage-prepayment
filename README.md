# Mortgage Prepayment Prediction
**NYU Stern — RA Project | Advisor: Prof. Arpit Gupta**

---

## Project Overview
Predicting mortgage prepayment using Fannie Mae Single-Family Loan Performance Data. The project builds a sequence of increasingly sophisticated models — from logistic regression through Transformer-based architectures — and is now applying the Diep-Eisfeldt-Richardson (DER) framework to explain the cross-section of TBA MBS returns using hazard-model-implied prepayment risk loadings.

**Contribution angle:** The DER framework uses Bloomberg dealer survey forecasts as the prepayment forecast leg. We substitute our ML hazard model as the forecast, removing dependence on proprietary survey data.

---

## Repository Structure
```
mortgage_prepayment/
├── data/
│   ├── raw/                        # Raw Fannie Mae CSVs (not tracked in git)
│   ├── sequences/                  # Preprocessed padded sequences (not tracked)
│   ├── pmms_monthly.csv            # Freddie Mac PMMS 30yr rates
│   ├── zhvi_zip3.csv               # Zillow ZHVI at zip3 level
│   ├── treasury_yields.csv         # Treasury par yields (FRED, May 22 2026)
│   ├── fncl_tba_prices_clean.xlsx  # Bloomberg FNCL TBA prices (Jan 2018–May 2026)
│   ├── treasury_yields_clean.xlsx  # Bloomberg UST 5yr/10yr yields (Jan 2018–May 2026)
│   └── tba_roll_snapshot.xlsx      # TBA roll/drop snapshot (June 2026)
├── notebooks/                      # Exploration and analysis
├── outputs/                        # Model checkpoints, results, plots
├── logs/                           # SLURM job logs
├── docs/
│   └── DER_methodology_note.md     # DER framework documentation
├── scripts/
│   ├── train_hazard.py                 # Discrete hazard model training
│   ├── run_hazard.sbatch
│   ├── oas_engine.py                   # Monte Carlo OAS cashflow engine
│   ├── oas_solver.py                   # OAS spread solver (brentq)
│   ├── risk_neutral_rates.py           # Treasury bootstrap + drift correction
│   ├── train_ddpm_conditional.py       # Conditional DDPM rate simulation
│   ├── shap_transformer.py             # SHAP interpretability
│   ├── stage2_coupon_cpr.py            # CPR extraction by coupon bucket
│   ├── stage2_der_betas.py             # DER beta_x, beta_y computation (Eq. 5-6)
│   ├── stage3_der_regression_v2.py     # Fama-MacBeth cross-sectional regression
│   ├── realized_cpr_v4.py              # Realized CPR by coupon by month (CORRECT)
│   └── run_cpr_v4.sbatch               # SLURM job for realized CPR
├── prepare_sequences.py            # Data pipeline: raw CSV → padded sequences
├── work_log.txt                    # Hourly work log
└── README.md
```

---

## Data

### Fannie Mae Loan Performance Data
Source: https://loanperformancedata.fanniemae.com  
Format: Pipe-delimited (|), no header row (col0 = empty due to leading pipe).  
Pre-2020 files: 113 cols. Post-2020 files: 110 cols. Key field positions unchanged.

| Vintage | Loans | Rate Environment |
|---------|-------|-----------------|
| 2018Q1–Q4 | ~1.8M | ~4.5–5% |
| 2019Q1–Q4 | ~2.1M | ~3.5–4.5% |
| 2020Q1–Q4 | ~2.4M | ~2.7–3.5% (COVID low) |
| 2021Q1–Q4 | ~2.4M | ~2.7–3.5% |
| 2022Q1–Q4 | ~4.7M | ~3.5–7% (rising) |
| 2023Q1 | ~105K | ~6.5–7% (high) |
| **Total** | **~15.7M unique loans** | **Full rate cycle covered** |

**Sequence data:**
- Train: 6,295,960 × 33 × 9
- Test: 1,573,990 × 33 × 9
- Mask convention: True = real timestep throughout; inverted inside forward() for PyTorch attention

### Bloomberg TBA Data (pulled June 2026, Bobst terminal)
- FNCL 2.5–6.5 Mtge: Monthly last price, Jan 2018–May 2026, 32nds converted to decimal
- USGG5YR / USGG10YR Index: Monthly yields, same period
- TBA Monitor: Roll/drop snapshot for UMBS coupons (June 2026, point-in-time)
- Verified against Bloomberg-reported HIGH values — all 9 coupons match exactly

---

## Features
| Feature | Description |
|---------|-------------|
| `refi_incentive` | `original_interest_rate - pmms_rate_at_reporting_period` |
| `borrower_credit_score` | FICO at origination |
| `original_ltv` | LTV at origination |
| `current_ltv` | Dynamic: `original_upb / (orig_home_value × zhvi_now/zhvi_orig) × 100` |
| `original_upb` | Original unpaid principal balance |
| `loan_age_months` | Age in months |
| `dti` | Debt-to-income at origination |
| `loan_purpose_enc` | N=0 (purchase), Y=1 (refi/cash-out) |
| `property_type_enc` | P=0 (PUD), R=1 (row), C=2 (condo) |

---

## Results

### Phase 1–7 — Model Progression
| Phase | Model | AUC |
|-------|-------|-----|
| 1 | Logistic Regression (single vintage) | 0.7765 |
| 2–4 | XGBoost / LightGBM (multi-vintage, time-varying) | 0.8306 |
| 5 | Transformer (full sequence) | 0.8431 |
| 6 | +zip3 covariate (XGBoost) | +0.006 |
| 7 | Segmentation: Transformer wins all buckets | — |

### Phase 8 — SHAP Interpretability
| Feature | Mean |SHAP| |
|---------|------|
| loan_age_months | 0.040 |
| borrower_credit_score | 0.031 |
| refi_incentive | 0.029 |
| original_ltv | 0.025 |
| current_ltv | 0.024 |

Peak activation month 28. Burnout signal: negative loan_age SHAP at month 28.

### Phase 9 — Discrete Hazard Model
**Test AUC: 0.8181** (9 features). Architecture: Transformer with BCEWithLogitsLoss, 50% prepaid oversampling, ReduceLROnPlateau.

### Phase 10–11 — DDPM + Risk-Neutral Rates
- Conditional DDPM: paths start at today's PMMS (6.18%), conditioned on `start_rate` embedding
- Treasury zero-coupon curve bootstrapped from FRED par yields; drift correction → ZCB error < 2.6bp
- PMMS paths for refi incentive; Treasury paths for discounting. Historical spread: 1.89%

### Phase 12 — OAS Pricing
Monte Carlo OAS pipeline. **Median model price: 99.06% of par** ✅. OAS solver: brentq (±0.1bp).

### Phase 13 — TBA Return Cross-Section (DER Framework)

Following Diep, Eisfeldt, Richardson (2021) *Journal of Finance*:

**Model:** $E[R^{e,i}] = \lambda_x \beta^i_x + \lambda_y \beta^i_y$

where $\beta^i_x = \frac{r_t - c^i}{(r_t + \phi^i)(\phi^i + c^i)}$ and $\beta^i_y = \beta^i_x \cdot \max(0, m^i - r_t)$

- $r_t$ = PMMS (par rate), $c^i$ = coupon, $\phi^i$ = mean CPR from hazard model
- Betas are **time-varying**: recomputed each month using that month's PMMS
- Treasury-hedged excess return: TBA total return minus duration-matched UST return (D_mod = 6.5yr blended)
- Market type: DM = PMMS > 3.5% (WAC proxy), PM = PMMS < 3.5%

**Fama-MacBeth results (100 months, Jan 2019–May 2026):**

| Market | Months | λ_x mean | t-stat | p-value | Sign correct? |
|--------|--------|----------|--------|---------|---------------|
| Discount (DM) | 76 | +0.000016 | 0.25 | 0.81 | ✅ |
| Premium (PM) | 24 | −0.000651 | −2.35 | **0.028** | ✅ |

DER prediction confirmed in PM (2020–21): λ_x < 0 when market is premium-heavy.
DM result correct sign but insignificant — attributed to compressed CPR cross-section from one-sided loan panel.

**Known limitation:** Fannie Mae panel (2020Q1–2023Q1) is discount-heavy (rates only rose). Hazard model CPR spread: 0.74–1.46% vs realized 1–39%. Full DM identification requires earlier vintages spanning the 2020–21 premium regime.

---

## Key Files (outputs/)
| File | Description |
|------|-------------|
| `transformer_best.pt` | Binary classifier (AUC 0.8431) |
| `hazard_best.pt` | Discrete hazard model (AUC 0.7999, 21 vintages) |
| `ddpm_conditional.pt` | Conditional DDPM checkpoint |
| `pmms_cond_paths.npy` | PMMS rate paths (refi incentive) |
| `treasury_cond_paths.npy` | Treasury paths (discounting) |
| `oas_cashflows.npy` | Cashflow matrix (1000 × 1000 × 33) |
| `oas_spreads.npy` | OAS spreads per loan |
| `stage2_coupon_cpr.json` | Hazard model CPR by coupon bucket |
| `der_betas.json` | DER β_x, β_y per coupon (time-varying) |
| `stage3_lambda_ts.csv` | Monthly λ_x, λ_y from Fama-MacBeth |
| `stage3_excess_returns.csv` | Treasury-hedged TBA excess returns |
| `realized_cpr_by_coupon_v5.csv` | Realized CPR v5 (global cross-file detection, 21 vintages, 2018-2025) |

---

## Infrastructure
**HPC:** NYU Torch (`login.torch.hpc.nyu.edu`)
**Working dir:** `/scratch/at7095/mortgage_prepayment/`
**Conda env:** `/scratch/at7095/conda_envs/mortgage_env`
**SLURM account:** `torch_pr_932_general` (submit without `--partition`)
**GitHub:** `tiwari-adarsh0830/mortgage-prepayment`

```bash
ssh-keygen -R login.torch.hpc.nyu.edu
ssh at7095@login.torch.hpc.nyu.edu
```

---

## Key Engineering Decisions & Bugs Fixed
| Issue | Fix |
|-------|-----|
| Data leakage in current_ltv | Use `original_upb` not `current_actual_upb` |
| Column misalignment | Dict-based col_map sorted by file position index |
| Mask convention | True=real throughout; invert inside forward() |
| Hazard class imbalance | 50% prepaid oversampling per batch |
| OAS price too low (57%) | Add terminal value (remaining UPB at month 33) |
| PMMS path (wrong file) | Must use conditional not unconditional DDPM paths |
| realized_cpr bug (v1) | Used extra_13 (wrong col) + cumulative count → monotonic CPR |
| realized_cpr bug (v2–3) | col41=Modification Flag, not prepayment indicator |
| realized_cpr fix (v4) | UPB=0 in last appearance = prepayment month; two-pass chunked |
| TBA beta time-invariant | Beta_x/y must use that month's PMMS as r_t, not current PMMS |
| SLURM partition rejection | Submit without --partition flag |
| ZHVI coverage gap (2018 loans) | Rebuilt zhvi_zip3.csv to 2015+ (was 2019+); 2019+ values unchanged |
| realized_cpr cross-file bug (v4) | Global Pass 0 across all files finds true last appearance per loan |
| Calibrate/forecast on login node | Login node kills heavy CPU jobs; use SLURM run_calibrate.sbatch or nohup |

---

## References
1. Diep, Eisfeldt, Richardson — "The Cross Section of MBS Returns" — *Journal of Finance* 76(5), 2021 (NBER w22851)
2. Gabaix, Krishnamurthy, Vigneron — "Limits of Arbitrage: Theory and Evidence from the Mortgage-Backed Securities Market" — 2007
3. Boyarchenko, Fuster, Lucca — "Understanding Mortgage Spreads" — NY Fed SR674
4. Ho et al. — "Denoising Diffusion Probabilistic Models" — arxiv 2006.11239
5. arxiv 2410.18897 — DDPM + wavelet for synthetic financial time series
6. arxiv 2511.17892 — HJM no-arbitrage neural yield curve
7. Fuster et al. — "Predictably Unequal?" — SSRN 3072038

### Phase 14 — Full Vintage Expansion + Forecast Validation (June 13-15, 2026)

**Data expanded to 21 vintages (2018Q1–2023Q1):**
- Downloaded 2018Q1–Q4, 2019Q1–Q4, 2022Q1–Q4 from capitalmarkets.fanniemae.com
- Fixed ZHVI coverage gap: zhvi_zip3.csv only covered 2019+, causing 2018 loans to silently drop (NaN current_ltv). Rebuilt to cover 2015–2026; 2019+ values byte-identical to original (max diff = 0.0000)
- Sequences rebuilt: train 6,295,960×33×9, test 1,573,990×33×9
- Hazard model retrained: AUC 0.7999 (best at epoch 3, then overfits on larger dataset)
- Platt recalibration: a=0.4934, b=−4.840

**Forecast vs. realized CPR validation (core contribution):**
- Built time-varying forecast CPR: ran hazard model with each historical month's actual PMMS as refi incentive
- Before 2018-19 vintages: model underestimated premium-regime CPR by 4-7x
- After expansion: model tracks realized CPR across full rate cycle
  - Peak 2020-21 (premium): FNCL 4.5% forecast 4.6% vs realized 4.5% — near exact
  - Trough 2022-23 (discount): FNCL 6.5% forecast 2.7% vs realized 2.7% — exact
- Root cause of old gap: model needed 2018-19 loans (their first 33 months cover the 2020-21 refi boom)
- Files: forecast_cpr_timeseries.csv, forecast_vs_realized_cpr.csv

**Updated Fama-MacBeth results (21-vintage model):**
| Market | Months | λ_x mean | t-stat | p-value |
|--------|--------|----------|--------|---------|
| Discount (DM) | 76 | +0.000016 | 0.20 | 0.84 |
| Premium (PM) | 24 | −0.000639 | −2.15 | **0.042** |

PM result robust across all three model versions (9/13/21-vintage). DM insignificance is structural — in current rate environment all 9 coupons are discount, insufficient sign variation in beta_x to identify lambda_x.

**Realized CPR bug history:**
- v1: wrong column (cumulative flag) → monotonically increasing CPR
- v2-3: col41 = Modification Flag (Y/N), not prepayment
- v4: UPB=0 in last appearance per file → cross-file bug (Dec 2018 spikes for multi-file loans)
- v5 (current): global cross-file Pass 0 finds true last appearance per loan across all 21 files; remaining Dec 2018 artifact under investigation (early-month UPB reporting lag)

---

### Phase 15 — Pre-2020 Extended Training + 2020-21 OOS Holdout (June 18-19, 2026)

**Objective (Gupta, June 18):** Pull pre-2020 vintages (~2010 back), train on the extended panel, hold out 2020-2021 as a clean out-of-sample test of the hazard forecast.

**Data expanded to 2013Q1–2023Q1:**
- Downloaded 2013Q1–2017Q4 (20 vintages) from capitalmarkets.fanniemae.com (manual browser download; portal Cloudflare-blocks server-side requests)
- Schema confirmed identical across 2013–2023 (113 cols; loan_purpose R/P/C, property_type SF/PU/CO/MH, DTI populated) — no pipeline changes needed
- ZHVI rebuilt back to 2000 (was 2015+); 2015+ values byte-identical
- `prepare_sequences_extended.py`: train = 2013Q1–2019Q4 (5.56M loans, 2.54% prepay), OOS = 2020Q1–2021Q4 (9.58M loans, 1.07% prepay)
- Hazard retrained: `hazard_best_extended.pt`, AUC 0.7728 (best epoch 18, vs 0.7999 on 21-vintage)
- Platt recalibration: a=0.1032, b=−6.0877 (anomalously low slope; old a=0.4934)

**Key finding — pre-2020-only model learns an INVERTED refi S-curve.**
Diagnostic refi-incentive sweep (raw uncalibrated annualized CPR, same architecture + synthetic loans for both models):

| refi% | OLD 21-vintage | NEW 2013–2019 |
|-------|----------------|---------------|
| -2.0  | 22.2% | 11.8% |
| 0.0   | 46.1% | 0.001% |
| +0.5  | 63.7% | 0.001% |
| +1.0  | 76.5% | 0.001% |
| +3.0  | 96.2% | 0.02% |

Old model: correct monotonic S-curve. New model: collapses to ~0 exactly where prepayment should peak. Sweep values fall within the scaler's fitted range (z −1.19 to +2.43), so this is not extrapolation. The difference is purely the training-data regime — the 2013–2019 window contains no refi boom to learn from.

**Not a sample-size effect.** Raw refi-incentive distribution (mask-filtered): pre-2020 training set is 52.5% in-the-money (mean −0.09%) vs 25.8% for the 21-vintage set (mean −1.74%). The pre-2020 set has MORE in-the-money mass yet learns the relationship worse.

**Realized CPR by refi-incentive bin (age≤33) — the complication.** Both cohorts are hump-shaped, peaking at 0..+0.5% incentive then falling; pre2020 peak 2.77% CPR vs boom 1.53% (pre2020 higher in most bins). Neither shows a monotonic rising limb under this windowed static binning. Cause: the age≤33 cap + burnout selection suppress the high-incentive limb for both eras (high-incentive 33-month survivors are burned-out non-responders); the boom cohort's exposure is dominated by 2020–21 ultra-low-rate loans deeply out-of-the-money (197.9M at-risk in <−1.5 bin, 4 prepays).

**Open design question — the 33-month window.** 33 was inherited from the original 2018–2023 data (max common observation length). For 2013–2017 originations, the first 33 months fall entirely pre-boom, structurally excluding the 2020–21 response even from loans that lived through it. This confounds (a) origination-era effect vs (b) window-truncation effect. Next experiment: rerun realized-CPR-by-refi without the cap (window≈120mo) for the pre2020 cohort to test whether full-lifetime histories recover the high-incentive limb. If so, the fix is longer sequences — a cheaper intermediate step than full rolling estimation.

**Conclusion.** The clean pre-2020-only OOS test does not work as hoped: a model trained purely on 2013–2019 cannot represent the boom-era refi response. Confirms the concern from the June 17 email; points to longer observation windows or the rolling estimation Gupta called the ideal next step.

**New scripts:** prepare_sequences_extended.py, diag_raw_hazard.py, diag_panels_2_3.py, realized_cpr_by_refi_v1.py, check_schema_2013_2017.py

---
