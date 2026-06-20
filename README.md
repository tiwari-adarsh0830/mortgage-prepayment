# Mortgage Prepayment Prediction
**NYU Stern — RA Project**

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
│   ├── realized_cpr_v5.py              # Realized CPR by coupon (global 3-pass; current)
│   ├── realized_cpr_v4.py              # superseded by v5 (cross-file last-appearance bug)
│   ├── realized_cpr_by_refi_v1.py      # Realized CPR by refi-incentive bin (Phase 15)
│   ├── diag_raw_hazard.py              # refi-incentive sweep, single model (Phase 15)
│   ├── diag_panels_2_3.py             # refi sweep + distribution, both models (Phase 15)
│   └── check_schema_2013_2017.py       # schema validation for added vintages
├── prepare_sequences.py            # Data pipeline: raw CSV → padded sequences (production)
├── prepare_sequences_extended.py   # 2013–2019 train + 2020–2021 OOS holdout (Phase 15)
├── ARCHIVE_GUIDE.md                # Archive layout + reproduction path
├── work_log.txt                    # Hourly work log
└── README.md
```

---

## Data

### Fannie Mae Loan Performance Data
Source: https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data
(portal blocks server-side downloads via Cloudflare; download manually in browser, then transfer to HPC.)
Format: Pipe-delimited (|), no header row (col0 = empty due to leading pipe).
Vintages 2013Q1–2023Q1 all carry 113 columns with identical key field positions; categorical codes (loan_purpose R/P/C, property_type SF/PU/CO/MH, DTI) populated throughout.

| Vintage | Rate Environment | Use |
|---------|------------------|-----|
| 2013Q1–2017Q4 | ~3.5–4.5% (post-crisis, stable) | Pre-2020 extension (Phase 15) |
| 2018Q1–Q4 | ~4.5–5% | Production train |
| 2019Q1–Q4 | ~3.5–4.5% | Production train |
| 2020Q1–Q4 | ~2.7–3.5% (COVID low) | Production train / Phase 15 OOS holdout |
| 2021Q1–Q4 | ~2.7–3.5% | Production train / Phase 15 OOS holdout |
| 2022Q1–Q4 | ~3.5–7% (rising) | Production train |
| 2023Q1 | ~6.5–7% (high) | Production train |

The production hazard model uses 21 vintages (2018Q1–2023Q1, ~15.7M unique loans).
Phase 15 adds 2013Q1–2017Q4 for the pre-2020 training experiment.

**Sequence data:**
- Production (21-vintage): train 6,295,960 × 33 × 9, test 1,573,990 × 33 × 9
- Extended (2013–2019, Phase 15): train 5,558,998, test 1,389,750 (2.54% prepay)
- OOS holdout (2020–2021, Phase 15): 9,584,630 loans (1.07% prepay)
- Mask convention: True = real timestep throughout; inverted inside forward() for PyTorch attention
- Sequence arrays are not tracked in git (~29GB); regenerate via prepare_sequences*.py

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
| `loan_purpose_enc` | **Inactive** — see note below |
| `property_type_enc` | **Inactive** — see note below |

> **Note on the two categorical features (known issue).** In `prepare_sequences.py`,
> `loan_purpose` is mapped with `{'N':0,'Y':1}` and `property_type` with
> `{'P':0,'R':1,'C':2}`. The raw Fannie data actually uses `loan_purpose` codes
> R/P/C and `property_type` codes SF/PU/CO/MH, so neither mapping matches — both
> resolve to the `.fillna(0)` default and are effectively constant zero. The
> diagnostics treat them as dead (`DEAD_COLS=[7,8]`), and all reported results
> were produced with these two features inert. **Fix opportunity:** remap to the
> real codes (loan_purpose R/P/C → 0/1/2; property_type SF/PU/CO/MH → 0/1/2/3)
> and retrain to add two genuinely live features. Net effect on current results
> is nil since the model never used them.

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

**Fama-MacBeth results — 13-vintage model (superseded; see Phase 14 for current 21-vintage figures):**

| Market | Months | λ_x mean | t-stat | p-value | Sign correct? |
|--------|--------|----------|--------|---------|---------------|
| Discount (DM) | 76 | +0.000016 | 0.25 | 0.81 | ✅ |
| Premium (PM) | 24 | −0.000651 | −2.35 | **0.028** | ✅ |

DER prediction confirmed in PM (2020–21): λ_x < 0 when market is premium-heavy.
DM result correct sign but insignificant — attributed to compressed CPR cross-section from one-sided loan panel.
The current production figures (21-vintage model) are in Phase 14 below: PM λ_x = −0.000639, t = −2.15, p = 0.042.

**Known limitation:** Fannie Mae panel (2020Q1–2023Q1) is discount-heavy (rates only rose). Hazard model CPR spread: 0.74–1.46% vs realized 1–39%. Full DM identification requires earlier vintages spanning the 2020–21 premium regime.

---

## Key Files (outputs/)
Tracked in git (models + result tables):
| File | Description |
|------|-------------|
| `hazard_best.pt` | Production hazard model (AUC 0.7999, 21 vintages) |
| `hazard_best_extended.pt` | Phase 15 pre-2020 model (AUC 0.7728, 2013–2019) |
| `hazard_calibration.json` / `_extended.json` | Platt coefficients (a, b) per model |
| `der_betas.csv` | DER β_x, β_y per coupon (time-varying) |
| `stage3_lambda_ts.csv` | Monthly λ_x, λ_y from Fama-MacBeth |
| `stage3_excess_returns.csv` | Treasury-hedged TBA excess returns |
| `stage3_robustness_orthog.csv` | Orthogonalized-λ_y robustness check |
| `forecast_cpr_timeseries.csv` / `forecast_vs_realized_cpr.csv` | Forecast vs realized CPR |
| `realized_cpr_by_coupon_v5.csv` | Realized CPR by coupon (global 3-pass, 2018–2025) |
| `realized_cpr_by_refi_v1.csv` / `_nocap.csv` | Phase 15 realized CPR by refi-incentive bin |
| `forecast_vs_realized_cpr_2020.png` | Headline forecast-vs-realized plot |

Not tracked (large; regenerable): `*_seq.npy`, `oas_cashflows.npy`, DDPM/OAS path arrays.
JSON variants of some tables (`der_betas.json`, `stage2_coupon_cpr.json`) also exist; the CSVs are the primary form.

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

**Objective (advisor, June 18):** Pull pre-2020 vintages (~2010 back), train on the extended panel, hold out 2020-2021 as a clean out-of-sample test of the hazard forecast.

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

**Conclusion.** The clean pre-2020-only OOS test does not work as hoped: a model trained purely on 2013–2019 cannot represent the boom-era refi response. Confirms the concern from the June 17 email; points to longer observation windows or the rolling estimation the advisor called the ideal next step.

**New scripts:** prepare_sequences_extended.py, diag_raw_hazard.py, diag_panels_2_3.py, realized_cpr_by_refi_v1.py, check_schema_2013_2017.py

---
