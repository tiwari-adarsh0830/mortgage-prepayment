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
| FM sample restriction leak (stage3_der_factor_shocks.py) | fama_macbeth() received full returns panel instead of factor-coverage months; silently inflated n back to full-sample count in both full-sample (77->72) and rolling (77->48) runs |
| Rolling calibration fallback (stage2_forecast_cpr_rolling.py) | cutoff_2020/2021 had no own Platt file, silently fell back to OAS Platt (b=-4.840) instead of cohort-CPR Platt; forced cohort-CPR onto all four cutoffs |

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

## Phase 16 — Rolling t→t+1 Estimation + Equity×Incentive Diagnostic (June 20, 2026)

Implements the rolling real-time OOS design (per advisor guidance, June 20): train
through Dec Y, forecast Jan–Dec Y+1, roll forward. Directly addresses the
equity×rate-incentive interaction previously flagged (high-leverage post-GFC loans
did not refinance; low-leverage 2020–21 loans did).

### Equity×incentive diagnostic (`scripts/diag_equity_incentive.py`)
Confirms the transformer learned the equity gate on refinancing. Sweeping rate
incentive (−2 to +4pp) × current LTV (30–130) on the production model, holding all
else at median:
- LTV=80: monthly prepay hazard rises 0.22% → 17.35% as incentive goes 0 → +3pp (S-curve fires)
- LTV=120 (underwater): same +3pp incentive only reaches 6.87%, with a much flatter curve
- current_ltv is a live, time-varying model feature (index 3 of 9, ZHVI-adjusted each month)
- Caveat: LTV>100 is ~0.1% of the 2013–2023 panel (2009–2012 underwater cohort absent),
  so the underwater corner is extrapolation; the interaction is well-identified for LTV 60–100.
- Output: `outputs/diag_equity_incentive.png`, `.csv`

### Rolling pipeline (`prepare_sequences_rolling.py`, `train_hazard_rolling.py`, `forecast_rolling_cpr.py`)
- Calendar-truncated, expanding-window prep per cutoff year; per-cutoff scaler + Platt calibration.
- Train through Dec Y on GPU array; forecast Jan–Dec Y+1 CPR vs realized per coupon.

### ~~Key finding — the t→t+1 design only has signal from cutoff_2020 onward~~ (RETRACTED 2026-08-29)
**This finding was an artifact of reading the wrong column. See "Label column defect" below.**
Calendar-censoring at any cutoff ≤ Dec 2019 yields a training set with 0.00% prepay events.
Quantified: cutoff_2019 across 13.9M loans → 0.00% prepay. Every prepayment in the 2013–2023
panel occurs in the 2020–21 refi boom. This is the same regime-concentration result from the
June 17 analysis, now measured at cutoff level. Usable cutoffs:
- cutoff_2020 (~0.5–1.0% prepay) → forecast 2021
- cutoff_2021 (1.47% prepay)     → forecast 2022
- cutoff_2022                    → forecast 2023
- cutoff_2023                    → forecast 2024

### Bug fixes vs production pipeline (all in rolling scripts)
- MMYYYY→YYYYMM sort: Fannie's MMYYYY int is non-monotone across years (Dec-2018=122018 > Jan-2019=12019),
  which ordered sequences January-first across years. Fixed via `mmyyyy_to_yyyymm()`.
- Dead categoricals: `loan_purpose_enc`/`property_type_enc` were all-zero from wrong code maps.
  Fixed to R/C/P and SF/PU/CO/MH.
- Prepay-label lookahead: labels now derived only from rows within the cutoff window.
- Pass-2 scaler speedup: sampled fit (50k train rows/vintage) replaces full re-read; ~2hr → ~5min.

### Diagnostics added
~~`scripts/diag_zbc_column.py`, `scripts/diag_prepay_vanish.py` — confirmed zero_balance_code is
at col 106 across all vintages and isolated the 0% prepay to the cutoff filter (genuine
regime concentration, not a column/label bug).~~

**RETRACTED 2026-08-29.** Both diagnostics read col 106 themselves, so neither could detect
the error. `diag_zbc_column.py`'s own docstring in fact states the opposite of what this
line claims. Col 106 is Alternative Delinquency Resolution Count; the zero-balance code is
usecols 43. See "Label column defect" below.

## Phase 16 (cont.) — Rolling forecast completion + pipeline hardening (June 21–22, 2026)

Recovered and completed the rolling t→t+1 pipeline after a series of SLURM/memory issues. Key fixes:
- ~~**cutoff ≤ 2019 has zero prepay signal** (confirmed: cutoff_2019 = 13.9M loans, 0.00% prepay; all prepayments are in the 2020–21 boom).~~ **RETRACTED 2026-08-29 — artifact of the label column defect; col 106 is only populated from July 2020, so every pre-2020 cutoff necessarily read 0.00%. Corrected, cutoff_2018 = 23.31%.** Rolling estimation runs cutoffs 2020–2021: cutoff_2020 (0.90% prepay) → forecast 2021; cutoff_2021 (1.47%) → forecast 2022.
- **Trained AUCs**: cutoff_2020 = 0.7006, cutoff_2021 = 0.7159 (below production 0.7999). Likely depressed by the first-33-month window vs full-cutoff-window label mismatch in the eval set — the model trains on in-window-33 positives but eval labels count any in-window prepay. Forecast-vs-realized CPR is the primary metric, not AUC.
- **Pipeline hardening**: (1) resume guards skip completed passes (loan-IDs, scaler, train/test sequence shards) so timed-out jobs restart where they stopped; (2) single-read prep — each vintage read once, train+test built together, per-vintage shard checkpoints (was 2× full reads, ~6h → ~3h); (3) forecast load_panel pre-filters vintages by origination window (skips files that can't contribute), and writes CPR output incrementally per month to avoid end-of-run OOM.

### SLURM operational notes
- `--time=4:00:00` routes to `cpu_short` (backfill, fast scheduling) but caps at 4h — too short for full sequence-building passes. Use `--time=8:00:00` on general partitions for full prep (Pass 1–4 ≈ 5–6h); short walltime only for jobs provably under ~3h.
- FairShare was 1.0 throughout (not deprioritized); walltime, not priority, governed scheduling.

---

## Phase 17 — Rolling OOS Extension + DER Factor-Shock Pipeline (June 24–26, 2026)

### Rolling cutoff_2022 / cutoff_2023

Extended the rolling pipeline to cutoff_2022 (forecasts 2023) and cutoff_2023
(forecasts 2024). Trained AUCs: cutoff_2022=0.7070, cutoff_2023=0.7165.
Platt calibration written manually from training logs (trainer does not auto-save):
cutoff_2022 a=2.3598 b=−5.2993; cutoff_2023 a=2.2815 b=−5.1419.

**Rolling diagnostic findings (all 5 models, calibration-independent):**
The incentive S-curve diagnostic uses raw σ(logit) with no Platt scaling — since
Platt is monotonic it cannot reverse the direction of the hazard-vs-incentive
relationship, so shape verdicts are independent of calibration.

| Model | Shape | Mechanism |
|---|---|---|
| production | Correct S-curve (rises monotonically, near-zero at −2pp to ~0.25 at +4pp) | Trained on full rate cycle |
| cutoff_2020 | Null / flat (near-zero throughout, < 0.02) | 0.90% in-window prepay; no refi signal |
| cutoff_2021 | U-shaped / distorted | Boom overfit: activation at age 28–33 × incentive >1.5pp |
| cutoff_2022 | Flat near zero | Turnover learned (age 3–6 months); refi channel closed |
| cutoff_2023 | Flat near zero | Same as cutoff_2022; equity gate inverted above LTV=100 |

**Equity gate (production model, confirmed):**
- LTV=80: monthly hazard 0.22% → 17.35% as incentive 0 → +3pp (strong S-curve)
- LTV=120 (underwater): same incentive only reaches 6.87% (gate suppresses refi)
- Gate survives in cutoff_2021 but weakens in cutoff_2022/2023 as rate-driven
  signal disappears from the training window

**Core finding:** Rolling models only learn rate-driven prepayment responsiveness
when their training window contains a refi wave. Outside that window (pre-boom or
post-boom), the model either has no signal (cutoff_2020) or learns turnover-at-
young-age which is incentive-insensitive (cutoff_2022/2023). This is a fundamental
data-availability constraint, not a modeling failure — and it explains why DER use
a forward-looking dealer survey rather than a backward-fit model for the forecast leg.

**New scripts:** `scripts/stage2_forecast_cpr_rolling.py` (5-model dispatch,
2020–2024 window), `scripts/diag_rolling_incentive_scurve.py` (S-curve + age×
incentive + equity×incentive heatmaps), `slurm/prep_rolling_array.slurm`.

---

### DER factor-shock pipeline (`scripts/stage3_der_factor_shocks.py`)

Implements DER (NBER w22851) Eqs 15–18: empirical prepayment-surprise factors
replacing the analytical price-formula betas used in stage3_der_regression_v2.py.

**Factor construction (DER Eqs 15–18, verified against paper):**
Each month, run separate OLS of forecast and realized CPR on `max(0, note_rate − PMMS)`
across the 9 FNCL coupons. Factor innovations = difference in regression coefficients:
- f_level[t] = x̂_realized − x̂_forecast  (level/turnover surprise)
- f_slope[t] = ŷ_realized − ŷ_forecast  (rate-sensitivity surprise)

Empirical betas estimated by time-series regression of TBA excess returns on
(f_level, f_slope). Fama-MacBeth cross-sectional regression gives lambda_x, lambda_y.
DER multicollinearity guard: drop months where corr(b_x, b_y) > 0.90.
Single-factor fallback: when all months are collinear (discount-heavy sample),
report lambda_x only (lambda_y unidentified).

**GFEE alignment (critical):** factor-shock pipeline uses GFEE=0.50 throughout
to match realized_cpr_by_coupon_v6 bucketing. Separate script
`scripts/stage2_forecast_cpr_gfee050.py` generates the aligned forecast.
The production timeseries uses GFEE=0.75 — do not mix these.

**Corrected results (v6 realized panel; 2026-07-03):**
- corr(b_x, b_y) = 0.402 → two-factor mode, both lambda_x and lambda_y identified.
  The v5->v6 realized-CPR fix (not just the forecast leg) is what unlocks this --
  v5's MMYYYY-sort bug was compressing the cross-section enough to force DER's own
  single-factor collapse (corr=0.935, above).
- Full-sample (theta_full): lambda_x=0.057 t=2.35 n=72, lambda_y=0.169 t=1.58 n=72
  (an earlier run reported t=2.52 n=77 -- FM sample-restriction bug, see bug table below)
- Rolling t->t+1 (theta_t-, genuine OOS across cutoff_2020..2023): lambda_x=0.149
  t=3.04 n=48, corr(b_x,b_y)=0.390 -- both survive the OOS test, lambda_x strengthens
- AR(1) robustness (DER's own test, Sec IV.B.1, replicated on full-sample factors):
  rho_x=0.911 rho_y=0.573. Unlike DER ("nearly identical"), our lambda_x is NOT
  robust to this: t drops 2.35->1.08. Real finding -- full-sample forecast leg
  (one fixed hazard model) carries more persistent/forecastable structure than
  DER's dealer-survey panel does.
- Per-cutoff-model debias of the rolling shock (3 attempts: additive, log-space,
  log-space ex-cutoff_2020) all broke the cross-section -- rolling shock is 53%
  time-driven / 8% coupon-driven with sign-reversing trend across cutoffs, a
  scalar bias per cutoff can't represent it. Correctly abandoned, open problem.
- lambda_y not currently reportable in the rolling design (0.169->1.263 jump,
  traceable to 2022-23 forecast/realized ratio blowups, same root cause as debias)

**UPB balance-weighting (2026-07-04):** rebuilt realized CPR with UPB weighting
(DER convention) via realized_cpr_v6_upb.py -- verified clean (2/13.77M prepaid
loans excluded for lacking a prior-month row; cpr_count matches v6.csv's cpr to
1.1e-4 max diff across all coupon-months). Fed through both forecast legs:

| Leg | Weight | lambda_x | t | n | lambda_y | t | corr(bx,by) |
|---|---|---|---|---|---|---|---|
| Full-sample | count | 0.057 | 2.35 | 72 | 0.169 | 1.58 | 0.402 |
| Full-sample | UPB   | 0.071 | 2.23 | 72 | 0.156 | 1.56 | 0.493 |
| Rolling     | count | 0.149 | 3.04 | 48 | 1.263 | 1.52 | 0.390 |
| Rolling     | UPB   | 0.175 | 3.02 | 48 | 1.299 | 1.53 | 0.377 |

lambda_x positive and significant (p<0.05) in all four combinations; UPB raises
the coefficient ~20-25% on both legs. corr(bx,by) stays well under DER's 0.90
threshold throughout -- two-factor identification unaffected by weighting choice.

---

### realized_cpr_v6.py — two bug fixes to the realized CPR panel

**Bug 1 (boundary failure):** v5 found each loan's global-last row using `idxmax`
on raw MMYYYY integers. MMYYYY is non-monotonic as integers (122020 > 62024, but
Dec-2020 precedes Jun-2024). Loans whose payoff month had a numerically smaller
MMYYYY int than earlier months got the wrong last row → UPB>0 → missed payoff →
2024–2025 realized CPR all-zero.

**Bug 2 (at-risk denominator):** same ordering error kept paid-off loans in the
at-risk pool past their true payoff month, inflating denominators and depressing
CPR even in 2018–2023.

**Fix:** convert MMYYYY→YYYYMM before all ordering/comparisons. Prepayment
detection stays as UPB==0 at true-last-row (zbc==1 was investigated but rejected —
col 106 persists for many months post-payoff, not a one-time event stamp).

**v6 also adds:** Pass 0 checkpoint (saves prepay_month/rate_map dict to pkl so
SLURM restarts skip the global scan). Script: `scripts/realized_cpr_v6.py`.
Scan running as of June 26; output: `outputs/realized_cpr_by_coupon_v6.csv`.

---

## Phase 18 — UPB Default Throughout + AR(1) Persistence on Rolling Series (July 5, 2026)

### UPB-weighting made the pipeline standard

`scripts/stage3_der_factor_shocks.py` previously required an explicit
`--realized-col cpr_upb` flag to use balance-weighted realized CPR. Both defaults
now point to UPB (`--realized-col` defaults to `cpr_upb`, `--realized` defaults to
`realized_cpr_by_coupon_v6_upb.csv`), so UPB-weighting is the standing convention
rather than a robustness check. Verified: a bare `python stage3_der_factor_shocks.py`
now reproduces the previously-confirmed UPB result (lambda_x=0.071, t=2.23, n=72)
with zero flags.

### AR(1)/persistence test extended to the rolling series

New script: `scripts/stage3_ar1_test.py`. Fits `f[t] = alpha + rho*f[t-1] + eps[t]`
on the factor series, replaces `f_level`/`f_slope` with the AR(1) residual
(innovation-only series), and reruns empirical betas + Fama-MacBeth on the
residualized factors.

**Note on the Phase 17 full-sample AR(1) result:** the original rho_x=0.911,
rho_y=0.573, t: 2.35->1.08 finding was run ad hoc and the script no longer exists
(confirmed via shell-history search) — only the result was recorded above. The new
`stage3_ar1_test.py` is the versioned, reproducible implementation going forward;
its full-sample count-weighted number differs slightly (t=1.26 vs the earlier 1.08)
but the qualitative conclusion — significance collapses under AR(1) residualization
— is unchanged.

**OOS-only fix:** the rolling series must be filtered to `is_oos == True` before
the AR(1) test (excludes the in-sample 2020 production-model months). An initial
run without this filter gave n=60 and a full collapse (rolling t: 2.20 -> -0.59);
after the fix, n=48, matching the genuine-OOS sample used throughout Phase 17.

**Results (`outputs/ar1_persistence_test_results.json`):**

| | Weight | RAW t | AR(1)-resid t | Survives? |
|---|---|---|---|---|
| Full-sample | count | 2.348 | 1.260 | collapses |
| Full-sample | UPB | 2.232 | 2.098 | mostly holds |
| Rolling OOS | count | 3.035 | 2.896 | holds |
| Rolling OOS | UPB | 3.020 | 2.877 | holds |

Two-factor mode stays intact in all four cases after residualizing (no silent
fallback to single-factor; corr(b_x,b_y) never exceeds the 0.90 threshold).

**Caveat found and diagnosed:** corr(b_x,b_y) in the rolling AR(1)-residualized
case flips sign (0.39 -> -0.53 count-weighted, 0.38 -> -0.51 UPB-weighted), unlike
full-sample (0.40 -> 0.43, 0.49 -> 0.59), which stays positive. Inspected the
per-coupon beta table directly: not a single-outlier artifact — every coupon's R^2
degrades broadly (e.g. one coupon's fit falls from 0.042 to 0.018) after
residualizing f_level's rho~0.9 persistence out of only 47 months split across 9
coupons. Read as an estimation-precision issue at this sample size, not a genuine
economic reversal — flagged rather than smoothed over.

## Phase 19 — Rolling AR(1) Robustness: Cutoff_2020 Exclusion (July 6, 2026)

### Request

Advisor asked for one more robustness cut on the Phase 18 rolling AR(1) result:
re-run the AR(1)-residualized rolling Fama-MacBeth excluding the `cutoff_2020`
forecast leg (drops the 2020-21 forecast-year months), running on the remaining
36 months from `cutoff_2021` onward. Report point estimates, t-stats, and both
lambdas.

### Implementation

Patched `scripts/stage3_ar1_test.py` (additive only, verified via diff against
pre-patch backup):
- Added `exclude_cutoffs` param to `run()`, filtering on the `model_used` column
  in `rolling_forecast_cpr_timeseries.csv` before the OOS-only filter
- Added `lambda_y` mean/t-stat reporting alongside the existing `lambda_x` output
  (previously only `lambda_x` was surfaced)
- New `results["rolling_ex_cutoff_2020"]` entry in the output JSON

**Data source correction:** initial run failed — the default realized-CPR file
(`realized_cpr_by_coupon_v6.csv`) only has count-weighted `cpr`, not `cpr_upb`.
The UPB-weighted column lives in a separate file, `realized_cpr_by_coupon_v6_upb.csv`
(built by `scripts/realized_cpr_v6_upb.py`, previously uncommitted — added this
phase). Corrected invocation passes `--realized-path` explicitly.

### Results (`outputs/ar1_persistence_test_results.json`, UPB-weighted)

| | lambda_x mean | t-stat | n |
|---|---|---|---|
| RAW | 0.0486 | 2.586 | 36 |
| AR(1)-residualized | 0.0318 | 2.310 | 35 |

Holds up: significant both before and after AR(1) residualization, though the
correction takes a larger relative bite here (t drops ~11%) than in the full
48-month rolling series (~5% drop, Phase 18).

### lambda_y not identified in this window

`rho(b_x, b_y)` across the 9 coupons rises to **0.986** once `cutoff_2020` is
excluded (vs. 0.39 with it included), tripping the pipeline's existing
`rho_max=0.90` single-factor fallback in `fama_macbeth()` — same collinearity
mechanism as DER's own result, not a bug. Confirmed via standalone diagnostic
against `empirical_betas()` output directly. Ruled out one hypothesis (all-discount
market months): 31/36 months have at least one premium coupon, so it isn't simply
a one-sided-market identification issue like 2023 was.

### Robustness check on the RAW lambda_x result

- Sign consistency: 25/36 months positive
- Leave-one-out: t-stat ranges from 2.32 to 3.14 across all 36 single-month
  exclusions (full-sample t=2.586 sits inside this range) — no single month
  drives the result

Sent to advisor July 6.

## Phase 20 — Standardized (Unit-Variance) Price of Risk: With/Without-2020 Comparison (July 7, 2026)

### Request

Following the Phase 19 result, advisor asked to rescale each surprise series
(f_level, f_slope) to unit variance within its own window before estimating
betas, so that lambda is denominated in "premium per one-SD exposure" in
every specification -- making the with/without-cutoff_2020 comparison
directly comparable, and separating whether 2020-21 was carrying magnitude
as opposed to just significance.

### Implementation

Patched `scripts/stage3_ar1_test.py` (additive only):
- Added `standardize_factors()`: z-scores f_level/f_slope using that
  specification's own mean/std, applied immediately before
  `empirical_betas()`, for both the RAW and AR(1)-residualized legs
- Fixed the `--realized-path` default (was `None`, silently falling back to
  the count-weighted `realized_cpr_by_coupon_v6.csv`); now defaults to the
  UPB file to match the `--realized-col=cpr_upb` default
- Added per-month standardized lambda_x CSV export
  (`ar1_std_lambda_x_<slug>.csv`) to support leave-one-out checks without
  re-running the full pipeline

**Analytical note, confirmed both by derivation and on synthetic data:**
because `empirical_betas()`/`fama_macbeth()` are linear in the factor
columns, this rescaling cannot change any t-stat -- only the reported
lambda magnitude. Confirmed against live output: all six t-stats
(full-sample, rolling, rolling-ex-cutoff_2020 x RAW/AR(1)-resid) matched
the previously-reported values exactly.

### Results (standardized, AR(1)-residualized, UPB-weighted)

| | lambda_x (per 1-SD) | t-stat | n |
|---|---|---|---|
| Rolling (with cutoff_2020) | 2.745 | 2.877 | 47 |
| Rolling ex-cutoff_2020 | 1.834 | 2.310 | 35 |

Ratio (without/with) = 0.668, a 33% drop. RAW series (no AR(1) filter) gives
a consistent direction: 1.389 -> 0.964, a 31% drop.

std(f_level) itself: 0.029 (with 2020) vs 0.017 (without), AR(1)-resid;
0.126 vs 0.050, RAW. rho(f_level): 0.92 (with 2020) vs 0.76 (without).

### Robustness check

Leave-one-out (jackknife) on the standardized AR(1)-resid lambda_x series:
with-2020 means range [2.44, 3.09] across single-month exclusions,
without-2020 means range [1.54, 2.15] -- ranges do not overlap.

Sent results-only to advisor (no interpretation of the
stable-vs-scale-artifact question -- left open per his framing) July 7.

## Phase 21 — Post-Residualization Autocorrelation Check + Beta Spread/Sharpe (July 17, 2026)

### Request

Advisor's three-part reply to Phase 20: (1) confirm the quoted rho values
(0.92/0.76) are from the raw pre-standardization factor series, and confirm
post-AR(1)-residualization autocorrelation is near zero; (2) report the
cross-sectional spread of standardized betas across coupons, translate into
an implied premium gap in bps/yr between the most- and least-exposed coupon,
plus the factor portfolio's Sharpe next to DER's; (3) scope a pre-2013
Fannie Mae data investigation ahead of a redesigned historical retrain.
This section covers (1) and (2).

### Implementation

Ask (1), rho sourcing: confirmed directly from code -- `ar1_residualize()`
runs on `factor_ts['f_level']`/`factor_ts['f_slope']` (the raw series)
before `standardize_factors()` is ever applied.

Ask (1), post-residualization check: this wasn't actually being verified
before (only the AR(1) coefficient on the raw series was reported, never
whether the leftover residual itself is white noise). Added
`lag1_autocorr()` to `scripts/stage3_ar1_test.py`, wired into `run()` to
report it for all three specs alongside the existing rho.

New script `scripts/stage3_beta_spread_sharpe.py` (ask 2): reuses the
AR(1)-residualize + standardize pipeline from Phase 20, then for each spec:
(a) takes max-min of the standardized b_x across the 9 coupons, (b)
multiplies by lambda_x and annualizes to bps/yr, (c) builds a
long-highest-beta/short-lowest-beta zero-cost portfolio from realized
excess returns and reports its annualized Sharpe.

New script `scripts/stage3_beta_spread_loo.py`: leave-one-out on the
ex-cutoff_2020 spec's bps/yr and Sharpe (the only spec with a real
monotonic beta profile -- see Results).

### Results

Post-residualization autocorrelation (lag-1, on the AR(1) residual itself,
not the raw-series rho):

| | rho (raw series) | resid autocorr | n | ~SE (1/sqrt(n)) |
|---|---|---|---|---|
| Full-sample | 0.916 | -0.278 | 71 | 0.119 |
| Rolling (with cutoff_2020) | 0.922 | -0.152 | 47 | 0.146 |
| Rolling ex-cutoff_2020 | 0.764 | -0.177 | 35 | 0.169 |

Both rolling specs are within ~1 SE of zero (genuinely near-white-noise).
Full-sample sits at ~2.3 SE -- a real residual autocorrelation, not clean.

Cross-sectional beta profile monotonicity (Spearman rho between coupon and
standardized b_x):

| | spearman rho | p-value | mean per-coupon R2 |
|---|---|---|---|
| Full-sample | +0.450 | 0.224 (n.s.) | 0.014 |
| Rolling (with cutoff_2020) | -0.217 | 0.576 (n.s.) | 0.038 |
| Rolling ex-cutoff_2020 | +0.933 | <0.001 | 0.034 |

Only rolling ex-cutoff_2020 has a statistically real, monotonic exposure
gradient. The other two specs' "most/least exposed coupon" would be
reading noise as signal -- not reported as a spread there.

For rolling ex-cutoff_2020: standardized b_x spread = 0.0035 (coupon 6.5
high / 3.0 low), lambda_x = 1.834, implied gap = 769.7 bps/yr. Realized
long-6.5/short-3.0 portfolio: Sharpe = 0.918 (n=35).

DER's own Sharpe benchmarks (Table XII) for comparison: full-sample
Max-Min=0.44/PRP=0.76; discount-market Max-Min=-0.47/PRP=0.47. Our
ex-cutoff_2020 window (cutoff_2021 onward, i.e. 2022-24) is a
discount-market period per the existing DM/PM classification, so the
discount-market row is the relevant comparison, not full-sample. Their
portfolios are vol-scaled/equal-leg-weighted over ~20 years; ours is a raw
monthly return difference over 35 months -- not directly comparable
methodology, caveated as such.

### Robustness check

Leave-one-out on the ex-cutoff_2020 headline (36 folds, one month dropped
each time, everything downstream re-estimated):
- bps/yr: range [572.9, 905.2] around 769.7, zero sign flips across all 36
  folds -- stable.
- Sharpe: range [-1.349, 1.185] around 0.918 -- **flips sign** when July
  2022 is dropped. That single month's exclusion changes which two coupons
  are identified as most/least exposed (6.5/3.0 -> 2.5/5.0), so the
  "Sharpe" isn't even comparing the same pair of coupons in that fold. Root
  cause: per-coupon betas are closely spaced and noisily estimated (R^2
  ~2-5% each), so an argmax/argmin over 9 coupons is fragile in a way the
  cross-sectional mean (lambda_x) isn't.

bps/yr reported to advisor as solid; Sharpe explicitly flagged as not
stable enough to report as a clean number, with the one-month mechanism
explained. Sent results-only July 17.

## Pre-2013 historical data (verified 2026-07-19)

`data_pre2013_raw/` — Fannie Mae Single-Family Loan Performance, 2000Q1-2012Q4,
52 quarters, 29,130,527 unique loans. Same layout as the existing pipeline:
113 pipe-delimited columns with a leading pipe (field N = awk $(N+1)), MMYYYY
dates, 2-decimal original UPB.

Field positions verified against a 2018Q1 control (11 of 113 checked, i.e. the
ones the pipeline consumes): $2 loan_id, $3 month, $8 rate, $10 original_upb,
$13 term, $14 origination_date, $24 borrower_credit_score,
$25 coborrower_credit_score, $32 msa, $33 zip3, $107 zero_balance_code.

The Jan-2015 single-score -> dual-score change does not appear here; co-borrower
fill rates are 45.2%/59.7%/43.3% for 2005Q4/2012Q4/2018Q1. Historical files
appear to have been restated into the current layout (inferred from fill rates,
not confirmed against Fannie Mae documentation).

`data_harp_raw/` — HARPLPPub.csv (25.9GB, same 113-col layout) and
Loan_Mapping.txt (comma-delimited, no header, 1,035,452 rows).
Mapping direction verified empirically: **col1 = original loan_id,
col2 = post-refi HARP loan_id**.

Open design questions (with advisor): sampling balance between the 2003 wave
(5,659,815 loans) and pre-wave history (2000-2002, 7,862,013), and whether a
HARP-refinanced loan is one continuous loan life or two events for labelling.

## Known issue: ar1_residualize() positional shift

`stage3_ar1_test.ar1_residualize()` does `reset_index(drop=True)` then
`shift(1)` — positional, not date-aware. Contiguous series are fine. In
leave-one-out folds, dropping a middle month makes the step treat the two
months either side of the gap as consecutive, and dropping either of the first
two months yields identical post-residualization date sets (35 unique Sharpes
across 36 folds in beta_spread_loo_ex_cutoff_2020.json). Headline estimates are
unaffected; fold values are slightly off.

## Known issue: fixed duration hedge in load_excess_returns() (found 2026-07-20)

`stage3_der_factor_shocks.py` line 51 sets `D_MOD_AVG = 6.5` — a single blended
5y/10y modified duration in YEARS — and applies it to every coupon at line 85.
TBA duration varies strongly by coupon (prepayment shortens premium coupons), so
this leaves residual rate exposure in `excess_return`.

`scripts/stage3_hedge_diagnostic.py` quantifies it. Per-coupon regression of
excess returns on 5y/10y rate changes:

- Ex-2020 window (2022-01..2024-12): 9/9 coupons significant on at least one leg.
  Full 100-month sample: 4/9.
- Implied duration (`D_c = D_MOD_AVG - 100*coef` on dy_avg) runs 8.05y at coupon
  2.5 down to 1.97y at coupon 6.5, vs 6.50 assumed. Spearman(coupon,
  implied_duration) = -1.000 in both samples.
- R2 is U-shaped in coupon, min 0.375 at coupon 4.0 — the coupon whose implied
  duration (6.58) is closest to the constant. Residual exposure is smallest where
  the fixed hedge happens to fit.
- At coupon 4.0, t(dy5) = +4.38 and t(dy10) = -4.45: unhedged curve exposure
  persists even where the level hedge fits, because one blended duration cannot
  match both key rates.
- Long-6.5/short-3.0: net mismatch 5.83y (t=13.56). Rate-driven component is
  599.9 of 790.4 bps/yr; residual intercept 190.5 bps/yr with t=1.00.

IMPACT: everything downstream of `load_excess_returns()` inherits this — the
lambda_x estimates, AR(1) residualization work, and beta spread / bps / Sharpe
results from Phases 18-21. Fix is per-coupon hedge ratios fixed at the beginning
of each month; estimation method (trailing-window empirical vs OAS model
duration, blended vs separate 5y/10y) pending.

## Hedge rebuild (2026-07-24)

### OAS engine cannot produce key-rate durations

`oas_engine.py` discounts on a 33-month grid (`MAX_SEQ`), and
`risk_neutral_rates.bootstrap_zero_curve()` interpolates the zero curve onto
`m/12 for m in 1..33` — max 2.75 years. Bumping the 5yr or 10yr par node moves
the discount curve by exactly zero (0/33 months change on +1bp at either tenor;
a 2yr control bump moves 21/33, confirming the test works). Both key-rate
durations come out identically zero. The Monte Carlo engine is therefore not a
viable source of per-tenor hedge ratios.

### Deterministic replacement: scripts/model_hedge_krd.py

Bootstraps the par curve to 360 monthly nodes at each month-end, prices each
coupon as a 30y pass-through (note rate = coupon + GFEE 0.50) amortizing at a
CPR path from the hazard model, bumps +/-25bp at the 5y and 10y points, passes
the bump through to the mortgage rate, recomputes the refi incentive, re-runs
the hazard model for a new CPR path under each bump, reprices, and takes the
two-sided difference. Ratios use data through the prior month-end.

Conversion (derived, matches a level position of equal parts 5y/10y plus a
long-10y/short-5y slope position):

    dP/P = -KRD5*dy5 - KRD10*dy10
         = -(KRD5+KRD10)*level - ((KRD10-KRD5)/2)*slope
    D_level = KRD5 + KRD10 ; D_slope = (KRD10 - KRD5)/2
    level = (dy5+dy10)/2 ; slope = dy10 - dy5

Calibration is `config/hazard_calibration_cpr_forecast.json` (a=0.4559,
b=-3.1376) — the cohort-CPR pair, never the OAS loan-level pair.

Note: all rows built by `build_batch_constant_refi()` are identical (constant
refi incentive, fixed representative loan), so `n_paths=1` is exact and the
500-path mean is redundant. `cpr_path()` uses 1 and memoizes on the incentive.

### Bump shape

Standard localized key-rate taper (0 at 3y, 1 at 5y, 0 at 7y; 0 at 7y, 1 at 10y,
0 at 20y) selects a single node on this par grid, since the taper endpoints
coincide with adjacent nodes. Two localized bumps capture only ~1/3 of effective
duration. `--spanning` uses partition-of-unity weights instead (w5 = 1 for T<=5,
linear taper to 0 at 10y; w10 = 1 - w5), so D_level equals effective duration by
construction — verified to 0.009% over 60 random coupon-months.

### Prepayment response works; verification still fails at discounts

`krd10` and `D_slope` turn negative at high coupons (krd10 = -0.97 at coupon
6.5): genuine negative convexity, absent from any fixed-CPR pricer.

Verification (regress hedged returns on level and slope per coupon; want all
coefficients zero, no cross-coupon pattern), spanning bumps, 99 months:

- coupon 6.5: t(level) = +0.12, R2 = 0.03 — passes
- degrades monotonically to t(level) = -12.70 at coupon 2.5

Residual duration implied by those coefficients (`-100*b_level`) plus the model
`D_level` reproduces the regression-implied duration at every coupon to within
~0.2y. Pricing is therefore correct; the hedge removes too little at discounts.

### Root cause: CPR beyond the 33-month forecast horizon

The forecast path is still ramping steeply at month 33 (m33/m12 = 5.1x to 7.5x
across coupons), so holding the terminal value flat to month 360 assumes a very
high permanent prepayment rate. Flat lifetime CPR needed to reproduce the
regression-implied duration, vs what the model assumes:

    coupon 2.5:  model 13.98%  needed  3.42%
    coupon 4.0:  model 14.43%  needed  6.80%
    coupon 6.0:  model 24.09%  needed 24.96%
    coupon 6.5:  model 28.91%  needed 37.83%

Crossover near coupon 6.0 — exactly where the verification test starts passing.

The path is also not a clean seasoning ramp: high at m1, trough near m12, then
climbing. Likely an artifact of the constant-incentive synthetic-loan setup,
which makes extending the terminal value fragile regardless of level.

Long-run CPR policy past the forecast horizon is unresolved and is the current
blocker. `extend_cpr()` isolates it.

### Superseded

- `scripts/krd_pricer.py` — first deterministic pricer, static CPR (no
  prepayment response) and spanning bumps only. Kept for reference; use
  `model_hedge_krd.py`.
- `scripts/build_hedge_panel.py` — per-coupon empirical level/slope hedge fit on
  realized returns. Neutralizes rate exposure (out-of-basis 2yr control t drops
  from 5-8 to <=0.25) but the betas are fit in sample, and they are regime
  dependent: fit on 2018-2026 the slope duration sign-flips for high coupons
  relative to a 2022-24 fit. Retained as the comparison that motivated the model
  hedge; not part of the pipeline.

### Pipeline state

`stage3_der_factor_shocks.load_excess_returns()` is UNCHANGED and still uses
`D_MOD_AVG = 6.5`. No corrected hedge has been wired in, pending the long-run
CPR decision. All Phase 18-21 results still carry the fixed-duration issue.

## Hedge rebuild, part 2: tent bumps + terminal S-curve (2026-07-29)

### Bump shape

Tent functions spanning the whole curve: the 5y leg is flat at full height for
T <= 5y then tapers linearly to zero at 10y; the 10y leg is the complement,
rising from zero at 5y to full height at 10y and staying there to 360 months.
The pair sums to exactly 1 at all 360 monthly nodes.

A strictly triangular 5y leg (rising from zero at T=0) does NOT span: below 5y
nothing holds the short end, so the weights sum to 0.017 at one month and only
reach 1 at month 60. Flat-below-peak is required for the parallel-shift
property. Identical to the earlier `--spanning` option.

### Terminal CPR: S-curve fitted to realized CPR

`extend_cpr()` no longer holds month-33 flat to 360. Months 34-360 use

    CPR(inc) = floor + (sat - floor) / (1 + exp(-k*(inc - x0)))

evaluated at the BUMPED incentive, so the terminal shifts with the bump.

The model CAN be queried at seasoned ages -- loan age is feature index 5 and can
be set independently of sequence position; `MAX_SEQ` only caps positions -- but
it extrapolates badly (`scripts/diag/diag_age_extrapolation.py`). Mean CPR at
incentive -3.0 RISES with age (0.063 at age 1, 0.146 at 61, 0.264 at 121) where
lock-in implies it should fall toward the realized 0.035-0.055, and the incentive
response collapses (sat/floor ratio 3.47 at age 1 to 1.42 at 121), which breaks
the requirement that the terminal preserve the CPR-rate relationship. Seasoned
ages are far outside the training range: age 61 maps to z=1.89..3.62 and age 121
to z=5.14..6.88 against a training span of z=-1.37..0.37. Even at month 33 the
model is ~4x realized at deep discounts (0.140 vs 0.035 at incentive -4) and
peaks at incentive 0.00 where realized peaks near +0.7. Hence realized data, not
the model, anchors the terminal.

Fitted per month on an expanding window (realized CPR strictly before the ratio
month) so ratios use prior data only. Restricted to coupons 2.5-6.5. Across the
99 monthly fits: floor 0.0364-0.0610, sat 0.1897-0.2518, x0 0.397-0.559,
n 350-1037. Cached per cutoff in `scurve_params_asof()`.

FIT SCOPE (important): `realized_cpr_by_coupon_v6_upb.csv` spans 2013-07 to
2025-12 and coupons 1.0-8.0 -- wider than the 2018Q1-2023Q1 vintage framing used
elsewhere. Coupons outside 2.5-6.5 were 25.5% of the unrestricted fit sample and
pre-2018 was 32.4%. Scope comparison (incentive -4..+2):

    ALL                   n=1392  floor=0.0517 sat=0.2204 x0=0.400 R2=0.429
    coupons 2.5-6.5       n=1037  floor=0.0546 sat=0.2492 x0=0.493 R2=0.515
    2018+                 n=941   floor=0.0534 sat=0.2207 x0=0.344 R2=0.400
    2.5-6.5 AND 2018+     n=694   floor=0.0573 sat=0.2740 x0=0.483 R2=0.567

Restricted to 2.5-6.5 is used: better specified (R2 0.43->0.52) though it makes
the verification marginally worse. A further 2018+ cut fits best but leaves ~9
observations at the panel start, so it is incompatible with the expanding window.
Scope was chosen on data relevance, NOT on verification outcome.

Expanding vs full-sample fit: coupon 2.5 level t -6.98 vs -6.91, so the earlier
full-sample result was not leaning on look-ahead.

### Verification (advisor's test: hedged returns on level and slope)

Level t-statistics, 99 months:

    coupon   flat-m33   S-curve (all cpn)   S-curve (2.5-6.5)
      2.5     -12.70          -6.91              -7.15
      3.5     -10.30          -7.70              -8.00
      5.0      -5.32          -2.75              -2.82
      5.5      -3.82          -2.01              -1.96
      6.0      -1.72          -1.62              -1.54
      6.5      +0.12          -1.32              -1.37

Inside |t| < 2: level at 5.5, 6.0, 6.5; slope at 4.5 through 6.5. Worst is now
coupon 3.5, not either end. Duration spread capture 44% -> 72% (model 4.241y vs
regression-implied 5.878y). Residual duration plus model duration reconstructs
the regression-implied duration to within 0.399y at every coupon (worst at 3.0),
short at all nine.

### Known limitations

- The aggregated CPR file has no age column, so the fit is across all ages. Age
  IS computable -- `realized_cpr_v6_upb.py` reads raw loan-level files and selects
  only 4 columns (COL_LOAN=1, COL_MONTH=2, COL_RATE=7, COL_UPB=11); origination
  date is in the rows. A seasoned-only fit needs the aggregation re-run with age
  as a key (~2-3h). Not yet done.
- Fit capped at incentive +2.0, terminal flat above it; 18.4% of panel
  coupon-months sit there (max +4.32). Not fitted beyond +2.0 because that region
  is dominated by the 2020-21 refi wave (realized CPR 0.33-0.35 in 2020-21 vs
  0.03-0.17 in 2014-19 at the same incentive, bucket sds 0.16-0.26).
- Realized CPR is non-monotone in incentive (dips ~+1.7 to +2.7 then rises). A
  monotone logistic was chosen, so this is not captured; a non-monotone form is
  fittable but adds a parameter and was not attempted.
- Weighting the fit on bucket means changes the floor by 16bp; checked, not adopted.
- `pmms_key='10yr'`: only the 10y bump moves the mortgage rate, so the whole
  prepayment response sits in KRD10 by construction. PMMS - 10yr measures 190bp
  over the full available history (2001-07+) and 189bp from 2003, matching the
  assumed figure; it is 215bp from 2018 and 247bp from 2022, so the panel window
  is wider than the long-run average. Does not affect the ratios, since the
  pass-through uses the bump and the PMMS level comes from data.
- `stage3_der_factor_shocks.load_excess_returns()` remains UNCHANGED at
  `D_MOD_AVG = 6.5`. No corrected hedge is wired into the pipeline.

## Phase 22 — Age-Keyed Realized CPR, Spread Control, Terminal Floor Refit (July 30 – August 3, 2026)

### Requests

Advisor, July 30, in reply to the hedge verification results: add a mortgage
spread control (the PMMS minus 10-year spread change); yes to re-running the
realized-CPR aggregation with loan age as a key; and yes, one terminal curve in
incentive evaluated per coupon, rather than a separate curve fitted per coupon.

Advisor, August 3, after those results: refit the terminal curve using a floor
of 0.0459 rather than the fitted 0.0546; check whether the model's prepayment
response is too flat to rate incentive by comparing the model S-curve against
realized; run the verification regression on level, slope AND spread change for
nine coupons; report the annualized vol of the hedged coupon-spread portfolio;
report residual duration in years. (The last bullet ends mid-sentence in the
original, so it was answered on a reading of intent and flagged as such.)

### Age-keyed aggregation — and a bug the validation could not see

`scripts/realized_cpr_v6_upb_byage.py` adds a seasoning key to Pass 1 of the
UPB-weighted aggregation: levels 0 (age < 60mo), 60 (60–119mo), 120 (120+).
Levels 60 and 120 together are the advisor's "age > 5yr" cut. Pass 0 is
untouched and its 2026-07-03 checkpoint (25,769,042 loans) is reused, since
nothing Pass 0 produces depends on age.

**The bug.** v2 read `LOAN_AGE` (0-based col 15) off each row. That field is
blank on the payoff row, so every prepayment landed in the missing-age bucket:
100% of `upb_prepay` in `age_group == -1`, zero in every real level. Seasoned
CPR came out identically zero, and the S-curve diagnostic downstream died with
a ZeroDivisionError on a zero-variance R² denominator.

**Why the validation missed it.** `verify_byage_totals.py` summed over all age
levels and compared against the baseline panel. That reconciles exactly whether
or not the numerator and denominator are split correctly — the partition was
intact, only the association between prepayments and ages was broken. The check
tested the wrong invariant. It now also asserts that `upb_prepay` is nonzero in
the real age levels.

**The fix.** v3 derives age from the origination date (col 13, MMYYYY, constant
within a loan) as `(Y2-Y1)*12 + (M2-M1)`, which is well-defined on every row
including the payoff row. Verified first on a synthetic raw file constructed to
reproduce the bug, then on the real panel: prepay mass in real age levels went
0.00% → 100.00%, and `n_prepay` now reconciles against the baseline exactly
(max abs diff 0.00, max rel diff 0.000e+00; was 1106 and 1.000 under v2).

The file's `LOAN_AGE` is not months-since-origination. Accumulating
`derived_age - LOAN_AGE` across the scan gives +1 at 95.45%, +2 at 2.97%, +0 at
1.51%, with a tail to +11 at ~0.05% — 99.93% within ±1 of the dominant
convention, immaterial at a 60-month boundary, and measured rather than assumed.

Runtime note: the first v3 run took 27 min/file and would have overrun a 12h
wall. The cost was `Counter(diff.tolist())` in the offset cross-check, a Python
loop over up to 2M ints per chunk. `np.unique` brought it to 7.4 min/file.

### Spread control — a negative result on the hypothesis

`scripts/diag/diag_spread_control.py`. Adding the PMMS − 10yr spread change as
a third regressor makes the level exposure LARGER, not smaller: coupon 2.5 goes
-7.12 → -8.20, worst coupon -7.89 → -9.13 (coupon 3.5), and coupons inside
|t| < 2 on level fall from three to two.

**A timing trap on the way.** The panel's `pmms` column is keyed to the
information date, not the return month: `corr(panel pmms, ret_month pmms lagged
one month) = 1.0000` exactly. Differencing it against a contemporaneous
Treasury change gives a spread series misaligned by one month — and that
misaligned version APPEARS to work, taking coupon 2.5 from -7.15 to -3.97. Its
VIF is 2.4, so much of the apparent improvement is standard errors widening
rather than the coefficient falling. Any spec mixing lagged panel PMMS with
contemporaneous external data is misaligned.

**Mechanism.** The spread coefficient is significant at coupons 2.5 through 5.0
(t between -2.81 and -4.49) and insignificant at 5.5, 6.0, 6.5 — present where
the hedge fails, absent where it passes. `corr(d_level, d_spread) = -0.601`
with a negative spread coefficient means omitting the spread biases the level
coefficient TOWARD zero. So the residual reads as unhedged level exposure that
the spread was partly offsetting in the estimate, not as spread contamination.
(Sign of the bias is shown; the interpretation is not separately tested.)

### Seasoned terminal curve — a wash, and why

Fitting the S-curve to seasoned loans only moves the level exposure around
rather than fixing it: coupon 2.5 goes -7.15 → -8.16, coupon 3.5 -8.00 → -7.56,
so the worst coupon shifts from 3.5 to 2.5 and the |t| < 2 count is unchanged.

The reason is that the seasoned restriction does not change the data where the
floor is identified. In the half-point incentive buckets from -4.0 to -1.5 the
seasoned and all-loan samples have IDENTICAL observation counts (28, 40, 41,
44, 53) — every deep-discount coupon-month cell already contains seasoned
balance. The restriction only thins the middle of the range (86→69, 122→86,
137→100 between -1.5 and 0), which distorts curvature.

The seasoned fit reports floor 0.0700 against 0.0546 all-loan, but that is a
fitting artifact: realized seasoned CPR below incentive -2.5 is 0.0516 (bootstrap
SE 0.0010), so the fitted floor sits 18.4 SE above what seasoned loans actually
do at depth, and a fit restricted to inc <= -1.0 gives 0.0483. The full-range
fit absorbs mid-range curvature into the floor parameter.

### Terminal floor modes

`model_hedge_krd.py` gains three alternatives to the fitted floor, selected by
`--floor-mode`, with the mode in the output filename so runs do not overwrite
each other. Default behaviour is unchanged and reproduces the prior t-statistics
exactly (verified as a control).

| mode | floor | sat | x0 | note |
|---|---|---|---|---|
| fitted (default) | 0.0546 | 0.2492 | 0.493 | full-range logistic, all-loan, expanding window |
| seasoned-fit | 0.0700 | 0.1875 | 0.365 | advisor's literal request; artifact, see above |
| pinned-seasoned | 0.0514 | 0.2509 | 0.484 | floor = realized seasoned mean at inc <= -2.5 |
| pinned-fixed | 0.0459 | 0.2545 | 0.473 | floor = realized all-loan mean; **has look-ahead** |

`pinned-seasoned` fails on the expanding window before 2018-02 (n=0 deep-discount
seasoned observations) — there were no deep discounts at all in the 2013–2018
window, so the floor is not estimable there. `pinned-fixed` applies a full-sample
statistic at every cutoff, which is look-ahead by construction; fine as a
diagnostic, not a production spec.

**Result of the 0.0459 refit.** Level t improves at all coupons 2.5–5.0
(2.5: -7.15 → -6.57; 3.5: -8.00 → -7.35, still worst), degrades slightly at
5.5/6.0/6.5, and 5.5 crosses out of the band at -2.07 so the |t| < 2 count falls
from three coupons to two. Duration capture 72.2% → 74.1%. Real improvement,
small.

### Verification outputs (advisor's requested table)

Three-regressor spec, hedged return on level, slope and spread change, pinned
floor panel, 99 months:

| cpn | t_level | t_slope | t_spread | resid_dur (2reg) | model_D |
|---|---|---|---|---|---|
| 2.5 | -7.73 | -3.24 | -3.62 | 1.588 | 5.463 |
| 3.0 | -8.06 | -2.72 | -3.63 | 1.536 | 4.752 |
| 3.5 | -8.74 | -2.77 | -4.11 | 1.538 | 4.257 |
| 4.0 | -8.08 | -2.82 | -4.67 | 1.194 | 3.857 |
| 4.5 | -5.93 | -2.02 | -3.66 | 0.866 | 3.500 |
| 5.0 | -3.87 | -0.51 | -2.80 | 0.560 | 3.107 |
| 5.5 | -2.53 | 0.17 | -1.42 | 0.384 | 2.498 |
| 6.0 | -1.32 | 0.09 | 0.19 | 0.384 | 1.630 |
| 6.5 | -0.32 | -1.14 | 1.50 | 0.313 | 1.106 |

Inside |t| < 2: coupons 6.0 and 6.5 on level, 5.0–6.5 on slope. The
three-regressor spec is a harsher test than the two-regressor one at every
coupon from 2.5 to 5.5.

**Portfolio vol.** Fixed pair, long 6.5 / short 2.5, on hedged returns over 99
months: 2.87% annualized (3.01% before the floor change). The 8.0417% figure
from the July 20 email is a DIFFERENT construction — beta-ranked pair on
unhedged excess returns over 35 months — so the two are not comparable and the
difference should not be read as the hedge improving.

All panel numbers were recomputed through a second code path (normal equations
rather than lstsq, `verify_before_email.py`) and matched exactly.

### Model vs realized S-curve — peak slope is not a usable statistic

The advisor's hypothesis was that the model's prepayment response is too flat to
rate incentive. **This has no stable answer as posed**, and three different
headlines were produced from the same data before that was caught:

| measurement | model/realized ratio, age 61 | reading |
|---|---|---|
| realized bucketed at 0.5 | 1.027 | not flat |
| bucket-free local linear | 0.536 | too flat |
| total CPR range | 2.292 | steeper than realized |

Peak slope moves monotonically with bucket width (age 61: 0.511 / 1.027 / 1.674
at widths 0.25 / 0.50 / 1.00) because realized CPR is noisy and non-monotone in
incentive, so its own peak slope moves by ~2x between quarter- and half-point
buckets. **Do not build a claim on peak slope with this data.**

What IS stable across every measurement, because it involves no derivative and
no binning:

- **Level.** Model CPR at incentive -4.0 is 0.140 (age 33), 0.091 (61), 0.267
  (121) against realized 0.0459 all-loan and 0.0516 seasoned below -2.5 — three
  to five times realized at deep discounts.
- **Position.** Model steepest at 0.00 (age 33), -0.75 (61), -0.50 (121);
  realized steepest at +0.55 for both populations under every bucketing tried.
  The model reacts hardest 0.5–1.25 points below where loans actually respond.
- **Age response is wrong-signed.** Deep-discount CPR rises from age 61 to age
  121 where lock-in implies it should fall. Consistent with
  `diag_age_extrapolation.py`; seasoned ages are far outside the training range.

### Open with the advisor

Portfolio definition (fixed pair on hedged returns vs beta-ranked); the
truncated residual-duration sentence; and whether to fit the terminal curve's
x0 to the realized peak (~+0.55) rather than letting the full-range fit place it
at ~0.47. The last is a proposal, not a request.

### New scripts

`scripts/realized_cpr_v6_upb_byage.py`,
`scripts/diag/verify_byage_totals.py`,
`scripts/diag/diag_spread_control.py`,
`scripts/diag/diag_seasoned_vs_all_scurve.py`,
`scripts/diag/diag_seasoned_floor_check.py`,
`scripts/diag/diag_advisor_outputs.py`,
`scripts/diag/diag_model_vs_realized_scurve.py`,
`scripts/diag/diag_flatness_range.py`,
`scripts/diag/verify_before_email.py`,
`scripts/patches/patch_floor_modes.py`,
`scripts/patches/patch_pinned_fixed.py`.

## Phase 23 — CPR Mapping, and Where the Residual Actually Lives (August 4–5, 2026)

### Request

Advisor, August 4: the transformer does not aggregate well into pool-level
predictions. For each month t, take history through t-1; for every coupon-month
cell fit realized CPR as a function of (model CPR, incentive) — suggested form, a
regression of log realized against log model with coefficients varying by
incentive. Then every CPR path, baseline and each bumped path, goes through that
mapping before it is priced. Expanding-window throughout so no look-ahead.

### The mapping works as a forecast correction

`scripts/cpr_mapping.py`, diagnostics in `scripts/diag/diag_cpr_mapping_v2.py`.
UPB-weighted, 36-month burn-in, 1-month reporting lag, 524 scored cells over 59
cutoffs. OOS log RMSE 0.4761 with no mapping against 0.3461 under a logit link —
a 27.3% reduction. Deep-discount ratio (realized/model below -2.5 incentive)
0.718 -> 0.912.

Model side is `forecast_cpr_timeseries_gfee050.csv`, which
`model_hedge_krd.py` already imports, so it is the same construction as
`cpr_path` at the same GFEE. Realized side is `cpr_upb`, matching
`scurve_params_asof`, so the mapped months 1-33 and the terminal months 34-360
share a weighting convention. `forecast_vs_realized_cpr_gfee050.csv` is NOT used:
its realized column is count-weighted (matches `cpr_count` to 1.1e-4, `cpr_upb`
only to 0.524), predating the 2026-07-06 UPB rebuild.

### It does not fix the hedge, in any of three application modes

`--map-mode {off,scalar,pointwise,frozen}`; `off` reproduces the prior
t-statistics exactly and is the control. Level t-statistics, pinned-fixed floor,
spanning bumps, 99 months:

| coupon | off | frozen | scalar | pointwise |
|---|---|---|---|---|
| 2.5 | -6.57 | -5.99 | -6.36 | -7.87 |
| 3.5 | -7.35 | -6.94 | -7.36 | -9.11 |
| 5.5 | -2.07 | -2.67 | -4.76 | -6.15 |
| 6.5 | -1.49 | -3.52 | -6.17 | -5.56 |
| **inside \|t\|<2** | **2** | **0** | **0** | **0** |

The fitted logit slope is 1.922 (sd 0.071, min 1.747), above 1 at every cutoff,
so the mapping amplifies the CPR response to a bump rather than only correcting
its level. That shortens model durations; scalar drives `D_level` at coupon 6.5
to -0.084, a premium MBS gaining value when the whole curve sells off.

`frozen` was built to test whether the degradation is an artifact of application:
it fixes the scale factor at the unbumped incentive so a bump moves the model
path only. It is the least bad mode and still leaves zero coupons in the band, so
the effect is structural — correcting the CPR level changes cashflow timing, and
that changes duration.

**Capture moves the other way**: 74.1% (off) -> 82.6% (frozen) -> 94.1%
(scalar). Capture is a range over argmax/argmin, the same fragile construction
that made the Phase 20 Sharpe unreportable, and scalar reaches 94.1% by
overshooting at 6.5 rather than fitting better. Reported to the advisor
alongside the t-statistics rather than omitted.

### Two departures from the literal specification, both measured

**Logit rather than log.** Log-log stays inside (0,1) on observed cells (peak
0.816) but exceeds 1.0 once extrapolated past the observed model-CPR range,
which is what a bump does. `price_path` computes
`1-(1-clip(cpr,0,0.99))**(1/12)`, so an out-of-range CPR is silently clamped and
priced wrongly with no error raised.

**Single slope rather than incentive-varying coefficients.** The bucketed form
gives a negative model->realized slope in some bucket at 44 of 59 cutoffs, which
inverts the KRD sign under a bump. In the logit family incentive terms also score
slightly worse (0.3461 plain against 0.3548 with incentive).

### Zero-cell handling is load-bearing under OLS and not under WLS

The 33 realized-zero cells hold 0.0001% of at-risk UPB — median 2.91e6 against
1.41e11, 48 loans against 754,961. At 48 loans an observed count of zero is the
modal draw, not evidence of zero prepayment. Unweighted, dropping them versus
flooring them flipped the headline (+12% against -10.8%). UPB-weighted, drop and
floor agree to four decimals and the `--min-upb` sweep is flat from 0 to 1e10, so
the size filter is redundant. Weighting is also correct on its own terms:
realized CPR is UPB-weighted, DER's convention is UPB-weighted, and the pricer
values balance rather than loan counts.

### The residual is not spanned by level and slope

`scripts/diag/diag_duration_gap.py`. Fitting level/slope durations by regression
on past returns — sized as well as the data allows — and testing residual
exposure to the 2-year change, which is outside the level/slope span:

- **Expanding-window fitted durations still leave |t(dy2)| > 2 at seven of nine
  coupons.** If the residual were spanned by level and slope, optimally-sized
  durations would drive out-of-basis exposure toward zero. They do not. Something
  is missing from the two-factor set, and no CPR correction can fix it. This
  explains why the seasoned curve, the spread control, the floor refit and the
  mapping have all left the level t-statistics roughly where they were.
- `hedge_panel_validation.csv`'s t_dy2 = -0.12 is in-sample flattery — its
  coefficients are fitted on the same 36 months they are evaluated against.

### Two sizing findings, separate from the above

**Under spanning, durations are uniformly ~1.36x too small, shape correct.**
Median `D_fit/D_model` 1.36, flat across coupons (Spearman -0.367, p=0.33),
unaffected by floor choice. The Phase 21 rebuild therefore fixed the
cross-sectional shape that `D_MOD_AVG = 6.5` destroyed, and left a uniform scale
error. Phase 21's Spearman of -1.000 was over implied duration *levels*, not this
ratio; the two agree where they measure the same thing (implied 8.05 -> 1.97
there, D_fit 7.41 -> 1.53 here).

**Localized key rates are unusable on this par-node grid.** At matched vintage
and floor: ratio 3.46, `D_level` negative at coupons 6.0 and 6.5, and
`t_dy2_model` indistinguishable from unhedged at every coupon. The par nodes sit
at 3, 5 and 7 years, so a standard taper zero at 3y and 7y touches the 5y node
only. Spanning is required, not preferred. (An earlier comparison against
`model_hedge_panel_10.csv`, dated July 24, overstated this at 4-7x by confounding
bump shape with the July 31 terminal-curve construction.)

### Method notes worth keeping

- **A guard that only warns will be reasoned past.** The duration diagnostic
  merged Treasury changes on `info_date`; the correct key is `ret_month`
  (correlation 0.994 against 0.025). The reconstruction check fired and printed a
  warning, and the table below it was read anyway. It now raises. A second guard
  was added: unhedged returns must show significant dy2 exposure, or the
  alignment is wrong whatever else passed.
- **An epsilon is a modelling choice.** Flooring zero cells at 1e-4 puts
  log(1e-4) = -9.21 into the response and dominates every score: identity RMSE
  1.1768 under floor against 0.4860 under drop, with identical predictions.
- **Check a safety grid against the empirical support.** The first in-range check
  tested model CPR 0.60 at incentive -5.0, a combination that never occurs;
  restricted to the observed support, in-range pass rates went 0% -> 100%.
- **189bp vs 216bp is not an error.** 189bp is `risk_neutral_rates.py`'s
  2001-07-onward average (daily join; month-end gives 190bp); 216bp is the
  2018-02..2026-04 window. Post-2020 widening. Nothing in pricing uses either —
  `krd_pair` takes the contemporaneous monthly `pmms`.
- **`FIXED_FLOOR = 0.0459` is not reproducible from the current panel.** Every
  filter tried gives 0.045452; the value likely predates the July 31 age-keyed
  rebuild. No t-statistic depends on it. It is quoted as the specified value, not
  as a recomputed statistic.

### Open

The mapping's slope of ~1.92 is close to a direct measure of how much less the
model responds to incentive than realized CPR does. The model sees 2018 onward —
one refi cycle. The pre-2013 files are unzipped at `data_pre2013_raw/` and the
layout is verified compatible, so the expanding-window design (train through
2002 predict 2003, through 2011 predict 2012-13, through 2019 predict 2020-21)
would address the flatness at source rather than after the fact. Vintage sampling
balance and HARP one-life-vs-two-events labelling remain undecided, and the scan
needs rebuilding at roughly double scale.

The larger open question is what the missing factor is. Level and slope do not
span the residual, and that is prior to any CPR or duration work.

## Phase 24 — Third Tent Tested; 2yr Blindness Found, Not Yet Fixed (August 7, 2026)

### Request

Advisor, August 6, replying to Phase 23: the missing factor is a separate 2yr
rate component. Fix: a third tent, flat below 2yr, peaking at 2yr, falling to
zero at 5yr, with the 5yr leg starting to rise at 2yr instead of flat from
zero. Reparameterize into level/slope/curvature (curvature = "the middle
moving against the two ends"). Separately: the Phase 23 finding that durations
are uniformly ~1.36x too small "looks mechanical" — try scaling durations by
1.36 directly as a diagnostic. Defer the pre-2013 historical work until this is
nailed down.

### The 1.36 scaling test does not pass

Non-circular test (dy2 was never used to fit the scalar): scaling the existing
level/slope durations by 1.36 does not zero out residual dy2 exposure — it
overshoots. At k=1.36, t(dy2) is positive at every coupon (+1.06 to +3.76),
having crossed zero somewhere below it. The scalar that actually zeros t(dy2)
sits near 1.15-1.20, a different number from what the level t-statistic itself
wants (that grid-searched value is circular and only used as context, not
reported as a finding). Not one clean mechanical scale factor.

### The tent is built and geometrically exact

`key_rate_weights3()`, `krd_triple()`, `--bump-shape tents3` in
`model_hedge_krd.py`. Verified against the pricer's actual node grid, not just
algebraically: the sum of the three tents equals 1 at every node (max error
1e-10), and w2+w5 under the new construction exactly reproduces the old
spanning w5, w10 reproduces the old spanning w10. The three-tent version is
therefore a strict refinement of the existing spanning pair, not a new
construction — it splits the old 5y leg into a genuine 2y piece and 5y piece.

Curvature is built as `2*dy5 - dy2 - dy10`, exactly twice the advisor's literal
"the middle moving against the two ends" (`dy5 - (dy2+dy10)/2`) — confirmed on
synthetic data so the check can't inherit a real-data bug. The
level/slope/curvature reparameterization (`D_level=K2+K5+K10`,
`D_slope=(K10-K2)/2`, `D_curve=(2*K5-K2-K10)/6`) round-trips exactly.

### The three-factor hedge does not outperform the two-factor one

Expanding-window fitted durations (level, slope, curve — sized as well as the
data allows, not just the pricer's own output) leave `|t(dy2)| > 2` at seven of
nine coupons, identical to the two-factor count from Phase 23. Curvature is not
absorbing the residual that dy2 was flagging.

### Root cause: the 2yr leg was designed to never move PMMS

Decision made in this phase, not from the advisor's email: PMMS was assumed to
track only the long end, so `dp=0` unconditionally for the 2yr tenor. That
means the incentive fed to the CPR model (`note - pmms`) is identical under a
+25bp and a −25bp 2yr bump, so `krd2` can only reflect discounting of near-term
cashflows — it structurally cannot respond to prepayment risk, which is the
dominant channel of MBS curve exposure.

Confirmed directly: `krd2` correlates with realized dy2 at only 0.03–0.17
across coupons (rising modestly toward premium coupons, not flat), and is a
small share of total duration at discount coupons (9.3% at 2.5). Its share
rises to 88% at coupon 6.5, but that tracks krd5+krd10 collapsing toward zero
at premium coupons (the Phase 23 duration-scaling gap), not genuine 2yr
sensitivity — checked and distinguished from the real effect.

### Empirical PMMS/2yr sensitivity — a range, not settled

Regressing monthly PMMS changes on the three Treasury legs:

- **Univariate** (dy2 alone): 0.44 contemporaneous, 0.57 at a one-month lag.
  The lag gap was checked against a month-alignment bug — the same class of
  error that produced the Phase 22 spread-misalignment trap — by rerunning
  under two different resample conventions (month-end, month-start). Both give
  *identical* results (0.442/t=6.05 and 0.570/t=8.81, to three decimals), so
  the lag effect appears to be a genuine one-month PMMS reporting lag rather
  than an artifact.
- **Multivariate** (all three legs, dy5/dy10 controlled): 0.73 pooled
  2018-2026, but leave-one-out stable (std 0.033, no single month responsible)
  while a chronological half-sample split is NOT stable — 0.05 (t=0.20) in
  2018–early 2022, 1.09 (t=2.90) in 2022–2026. A 24-month rolling window shows
  this is not a clean regime break at a single date either; it is a noisy,
  mostly-insignificant relationship through most of the sample that only
  became reliably significant (t>2.4) in roughly the most recent 15 months.

No single number was proposed to the advisor as settled. Reported as a range
(0.4–1.1) with the instability stated explicitly, and the choice — rebuild with
a specific pass-through now, or pin the estimate down further first — was left
to him.

### Verification discipline

Before emailing, every claim above was re-derived independently in one pass
(`verify_all_claims_final.py`) from raw source files, without importing or
trusting any of the diagnostic scripts that produced the original numbers:
tent geometry (numeric, against the real grid), curvature formula (synthetic
data), `dp=0` (read from the live pricer source, not memory), krd2
magnitude/correlation (rebuilt dy2 from scratch with its own alignment guard),
the 7-of-9 count (fresh regression, not reused from `verify_tents3.py`),
control-panel byte-identity (direct file diff), and the MS/ME resample
equivalence (both conventions run side by side in one script). All seven
checks passed.

### Open

Whether to rebuild `krd_triple` with a nonzero PMMS pass-through on the 2yr
leg, and at what value — awaiting the advisor's reply. If he wants the
estimate pinned down further before choosing a number, the natural next step is
an expanding-window PMMS/2yr sensitivity (mirroring the Phase 23 CPR mapping
design) rather than a single fixed constant, given the regime instability found
above.

### Two threads not yet followed up (flagged 2026-08-07, not started)

**"1.36 looks mechanical" — his named candidates not individually checked.**
The 1.36 test performed was the uniform-scalar sweep he explicitly suggested as
a diagnostic, and it failed (overshoots, wrong scalar for dy2 vs level t-stat).
But his email named three *specific* candidate mechanisms — the duration
denominator price (model vs market), bump normalization, or a term correction
in discounting future MBS cashflows — and only the first was checked (Phase 23:
market/model price ratio is 0.945-0.960, wrong direction and wrong magnitude to
explain 1.36). Bump normalization and the discounting term correction remain
unexamined. If he comes back with one of those two in mind specifically, that
is separate, not-yet-started work, not a re-read of what's already done.

**Why does PMMS/2yr sensitivity break down pre- vs post-2022?** The regime
instability (t=0.20 pre-2022, t=2.90 post-2022, no clean single-date break)
was found and reported as a range, but no economic explanation was tested.
Candidate: PMMS is a lender-survey rate that may be sticky/administered in
calm periods and start tracking the front end more closely when curve moves
are fast and repricing risk becomes urgent — plausible, untested. Worth
checking whether the instability correlates with realized rate volatility
(e.g. rolling std of dy2 or dy10) rather than calendar date, which would
support a vol-regime story over an arbitrary split-sample artifact.

## Phase 25 — 2yr Leg Fixed in Hedge Construction; Level Still Fails at 7/9 (August 8, 2026)

Advisor's Aug 8 reply to the Aug 7 email asked directly whether the hedged
return subtracts a two-year Treasury position. It didn't, even in the
tents3 run: `d_level`/`d_slope` in `model_hedge_krd.py` were still built as
`(dy5+dy10)/2` and `dy10-dy5`, the original two-tent definitions. `D_curve`
was computed from krd2/krd5/krd10 but never entered the `hedged` formula.
So the Phase 24 three-factor result was, at the point that mattered, still
a two-factor hedge with a durations basis (`D_level`, `D_slope`) that
included krd2 while the shocks it was paired with did not.

Fixed for tents3 only: `d_level = (dy2+dy5+dy10)/3`, `d_slope = dy10-dy2`,
`hedged` now includes `D_curve*d_curve`. Span (two-tent) path untouched —
verified byte-identical against the Aug 7 `_span_pinnedfixed.csv` after
patching (control run before and after every edit).

While tracing shock sources, found `treasury_yields_clean.xlsx` and
`treasury_yields.csv` disagree by a few bp at matching dates — mean-zero,
mean-reverting (diff-of-diffs autocorrelation -0.5/-0.6), not a level or
convention offset; ruled out simple resampling explanations by direct
test. Added `--shock-source {clean,daily}` to the pricer to isolate this
from the composition fix. Three arms run, same 99-month panel:

  A: span,   clean source  (control, reproduces Phase 24 exactly)
  B: span,   daily source  (isolates source effect only)
  C: tents3, daily source  (composition fix, this session)

|t_lvl|>2 count is 7/9 in all three arms — unchanged. Source swap (A->B)
moves t_lvl by <0.5 at every coupon; composition fix (B->C) accounts for
essentially all the improvement, largest at low coupons (2.5: -6.57 to
-5.15; 3.0: -6.99 to -5.83), fading to near zero by coupon 5.0-6.5.
Curvature not significant anywhere (|t_crv| < 1, all nine coupons).

Not yet explained: R2 on the hedged return rose under the corrected basis
at nearly every coupon (2.5: 0.317->0.360; 6.5: 0.030->0.102) rather than
falling. A hedge that's working better should explain less residual
variance, not more. Flagged to advisor, not investigated further this
session.

All three arms' t-stats independently re-derived via raw normal equations
(`/tmp/verify_final.py`, not committed — reran from scratch next session
if needed) against the pricer's own `ols()` output before anything went
to the advisor. Matched exactly.

Verdict sent to advisor: the 2yr omission was real and the fix is
correctly scoped, but it isn't sufficient — six of nine coupons remain
significant post-fix. Of his three named candidates for the 1.36 (Aug 6
email — price denominator, bump normalization, discounting term
correction), only price-denominator has been tested (Phase 24, ruled
out). Asked him which of the remaining two to check next rather than
guessing.

**Open, unchanged from Phase 24:** pre-2013 data investigation deferred
per advisor's Aug 6 instruction. PMMS/2yr pass-through range (0.4-1.1)
not needed for now — advisor's Aug 8 reply resolved this by pointing at
the hedge construction instead.

**New open items:**
- Bump normalization and discounting term correction — untested, his
  remaining two candidates for the 1.36.
- R2-rises-under-correct-hedge anomaly — no working hypothesis yet.
- clean.xlsx vs daily-file Treasury discrepancy — confirmed negligible
  for this result (<0.5 t-stat) but source unreconciled; worth resolving
  independent of the hedge work since it's a standing data-quality gap.
- `build_hedge_panel.py`'s dy2 out-of-basis control is no longer valid
  once dy2 is inside the hedge basis (this session's fix) — needs a
  replacement control (1yr and/or 30yr proposed, not yet built).

## Phase 26 — Three-Instrument Regression Hedge; Duration Scaling Clears Most of the Residual (August 17, 2026)

Advisor's Aug 16 reply resolved the R2 question (not a puzzle — expected
under the new normalization), endorsed the treasury fix, and set three
tasks: rerun the regression hedge for all three instruments, try a
`(0, 0.3, 0.7)` PMMS pass-through instead of `(0, 0, 1)`, and try
everything with durations scaled by 1.36.

**Regression hedge, three instruments.** `diag_duration_gap.py` extended
to fit on level/slope/curve when the panel carries `d_curve`. Because dy2
is now inside the fitted basis, the dy2 out-of-basis test is orthogonal by
construction and useless; switched to 1yr and 30yr. Caveat recorded and
reported: corr(dy1,dy2)=0.892, corr(dy30,dy10)=0.954, so these controls
partly proxy the fitted legs and the test is weaker than dy2-vs-{5,10}.

The answer splits on fitting method, and both were reported rather than
one being chosen:
  - in-sample (full window): residual exposure GONE, all 18 t-stats
    between -0.60 and -0.09
  - expanding-window (36mo burn-in): residual exposure REMAINS,
    t_dy1 significant 7/9, t_dy30 9/9

The in-sample arm is fitted on the returns it is then tested against, so
the expanding-window arm is the honest one. Verified this is not an
estimation artifact: burn-in swept 24/36/48/60 months, counts invariant
(7-8 of 9 and 9 of 9 throughout) while coefficient CV falls 0.309->0.102.
The durations stabilize and the exposure survives anyway.

**Duration scaling — this is the one that works.** Sum of squared
out-of-basis t-stats across 18 tests: 370.7 unscaled, 20.3 at the
advisor's 1.36, minimum 17.4 at 1.33. At 1.36, t_dy30 insignificant at
all nine coupons and t_dy1 at 7 of 9. Leave-one-out across coupons gives
1.31-1.34 — stable, unlike the Phase 20 Sharpe argmax. The objective is
smooth and single-minimum over 1.00-1.60. It does NOT reach zero:
typical |t| at the optimum is ~1.0, so a small residual persists.
Applied as a scalar on the model durations inside the hedged return;
the pricer itself was not rerun with scaled durations.

**`(0, 0.3, 0.7)` pass-through — not the explanation.** New
`--pmms-passthrough` flag (default None preserves the prior pmms-key
path; control byte-identical). Reallocates duration across legs
(krd5 1.398->1.061, krd10 1.228->1.562, krd2 unchanged at 0.726 since
its weight is still 0) but total D_level is 3.352->3.349 and the scale
factor stays at 1.34. Modest improvement in residual fit at the optimum
(17.4 -> 14.5). The reallocation direction is consistent with negative
convexity — a leg that moves PMMS has its duration offset by faster
prepayment — but that mechanism was NOT tested and is flagged as a guess.

**Verification.** Curvature leg scale checked independently: the
`D_curve*d_curve` term is 4.2% of the level term's sd (slope is 19.5%),
so the /6 in the curvature duration is correctly placed and the leg is
small for economic reasons, not a scaling bug. All headline figures
re-derived through normal equations in code that does not import the
scripts that produced them. Every patch control-run to byte-identical
reproduction before treatment.

**Open:**
- Residual does not fully vanish at the optimal scale (~1.0 typical |t|)
  — unexplained.
- The 1.33-1.36 scalar itself has no derivation; it is fitted, not
  mechanical. Root cause of the duration under-sizing still unknown.
- Out-of-basis controls are correlated with the fitted legs; a genuinely
  orthogonal control does not exist among Treasury tenors once the basis
  spans 2-10yr.
- `build_hedge_panel.py` still uses the two-instrument in-sample hedge
  and `clean.xlsx` treasury source; not updated this session.

## Phase 27 — Pricer-Side Search Exhausted; Bootstrap Defect Found and Fixed (August 18–20, 2026)

Advisor's remaining two candidates for the 1.36 were tested and both
ruled out. A separate, genuine bug was found in the curve construction
along the way, fixed, and validated externally — but it is not the cause
either. Every test in this phase held CPR fixed unless stated.

**Bump normalization — ruled out.** The three tents each peak at exactly
1.0 (worked through from the piecewise definitions, not read off the
weight function) and their key-rate durations sum to a single parallel
bump at the PRICE level, ratio 0.99991–0.99997 across four curve shapes x
three coupons x two CPR levels. The prior check in Phase 24 was on the
weights only; this is the stronger claim, and it holds at CPR=0.12 where
K10 goes negative at coupon 6.5.

**Discounting term — ruled out, after one invalid attempt.** The first
test bumped the PAR curve and compared against a closed-form duration.
That test is worthless and the record should say so: Macaulay equals
modified only under a parallel ZERO shift, and the bootstrap redistributes
a par bump according to the curve slope. Its deviations tracked curve
shape (0.93–1.09) and proved nothing. Redone by bumping the zero curve
directly, max |ratio-1| = 4.19e-04 across three zero levels x three
coupons x two CPR levels.

**Bootstrap defect — real, not on his list, not the cause.**
 does not reprice its own par inputs: feeding a real
curve in and pricing the par bonds back off the resulting zeros gives
errors up to 3.34pt at the 20yr node.  interpolates zero
rates linearly between solved nodes and clamps flat above the last solved
one, so the ~19 coupon PVs between 10yr and 20yr are priced off a guess
and the solved long node absorbs the error. Flat synthetic curves hide it
entirely — linear interpolation of a constant is exact — which is why it
survived this long.

 (log-DF interpolation) FAILED, 3.34 -> 2.99, and is kept in
the repo as a documented dead end: it still extrapolates past the node
being solved, which is the same disease.  solves each node
by root-find on -ln(DF) so coupons inside the gap price off a forward
consistent with the node being solved; 1mo/3mo/6mo treated as bills, 1yr+
solved. Par repricing 2.3e-13 across 299 month-end curves.

Externally validated against QuantLib 1.43: v3 sits within ~1bp past 20yr
while the original swings +18bp to -11bp on the same curve, sign-flipping
between the 240 and 360 month nodes. Note the aggregate mean|diff| is
MISLEADING (1.94 for v3 vs 2.00 for the original) because it is dominated
by short-end noise; the signal is only visible in the maturity split.

Duration impact measured three ways, all ~1.00: frozen CPR 0.997–0.999,
independent re-derivation 0.9989–0.9990, live pipeline 0.9983–1.0089 per
coupon. Legs reallocate (krd2 0.947, krd5 1.027) but the total barely
moves. v3 is OFF BY DEFAULT behind ; the default path
is byte-identical (md5 ecf4d2d2) and t_lvl is unchanged. Do not enable it
in production without asking.

**Short-end convention — open but immaterial.** v3 and the original BOTH
differ from QuantLib by up to 16bp at months 1–6; the convention question
is shared and unresolved. Four conventions (simple / disc360 / bond-
equivalent / continuous) move durations by 0.008%, and the forward-kink
test at the 6mo/1yr boundary does not discriminate between them
(0.27–0.32bp for all four).

**CPR response — cannot reach the target.** Terminal S-curve saturation
was checked first: 18.4% of coupon-months sit above incentive +2.0 where
 is flat-extrapolated at  and its derivative is exactly
zero. But saturation is 0.0% at coupons 2.5–4.0, which are the four
worst-hedged, so it cannot be the explanation. The strong pct_sat vs
D_level correlation is mechanical — incentive rises with coupon by
construction and premium coupons prepay fast.

Decomposition on coupons 2.5–4.0 by which segment sees the bumped
incentive: segments are separable (interaction 0.003–0.012, under 0.3%).
Prepayment feedback SHORTENS duration by 9–20%, rising with coupon
(BOTH/NONE 0.9103, 0.8496, 0.8200, 0.8006). This is the wrong direction
and insufficient in magnitude: durations are ~33% too small and feedback
makes them smaller, so switching the response off entirely recovers only
9–20%. At coupon 2.5, production 5.47 vs 6.01 with zero response.

Advisor's Aug 20 request — keep the transformer baseline path but take the
bump response from the realized S-curve — was run under both readings of
'apply the change', additive and multiplicative. Duration ratios 0.9868
and 0.9814. The cause is that the two responses are already close in size:
per 25bp the model mean over months 1–33 is 0.00802 and the S-curve 0.00824.
Per coupon the S-curve responds ~1.5x more at 2.5–3.0 and ~0.7–0.8x at
4.5–5.5, and durations shift accordingly — shorter at the discount end,
longer in the middle, which is the wrong direction for the coupons needing
the largest increase. Clip on the grafted path binds on 0.10% of elements.

**Two traps recorded so they are not repeated.**

 and  default at module level and are only set
inside . Any diagnostic that imports  bypasses
that and silently runs a different configuration. This corrupted a full
night of decomposition numbers before it was caught; all diagnostics now
set  explicitly.

The panel  corresponds to a PARALLEL bump with PMMS moving, not
 with PMMS on the 10yr leg. A reconstruction using the latter
came out 0.6–1.0 low at every coupon. Validate any reconstruction against
the panel values (5.478 4.751 4.252 3.850 3.495 3.110 2.513 1.645 1.110)
before trusting derived ratios.

**Verification.** Every elimination was re-derived in code that does not
import  — own tent weights from the piecewise
definitions, own cashflow recursion, own discounting — and reproduced
every number. The decomposition BOTH column and the graft run parallel-
bump baseline agree exactly (5.4686 4.7492 4.2558 3.8557) from separately
written scripts, and both reproduce the panel D_level to within 0.02 at
every coupon. Control runs byte-identical before every treatment run.

**State at close.** The pricer-side search is exhausted: price denominator
(Ph24), pass-through (Ph26), bump normalization, discounting term,
bootstrap, short-end convention, CPR response. The pricer computes correct
durations for the cashflows it is given and the cashflows respond
sensibly. Advisor's Aug 23 reply concludes the error is not in the model
but in the comparison — return construction, factor scaling, Treasuries —
and proposes running the machinery on an instrument of known duration
(a 5yr Treasury should return ~5 years), or that the spread control is
broken and TBA empirical durations genuinely exceed cashflow durations
because spreads move with rates. These two point opposite ways: the first
says the machinery is broken, the second says nothing is and the 1.33 is
economics.

**Open:**
- Root cause of the 1.33 under-sizing still unknown; the whole pricer side
  is now eliminated.
- Known-duration test on a 5yr Treasury — not built. Must go through the
  identical  and hedge-regression path, not a
  parallel reimplementation.
- Spread control / spreads-move-with-rates hypothesis — untested.
-  is off by default; enabling it means re-running
  everything downstream.
- Short-end convention vs QuantLib (~16bp at months 1–6) unresolved.
- The 1.33 scalar is still fitted full-sample; the expanding-window
  version was offered to the advisor and is not built.
-  still on the two-instrument in-sample hedge and
  the  treasury source.

## Phase 28 — Comparison-Side Search: Machinery Clears, Spread and Roll and Anchoring Do Not (August 23, 2026)

Advisor's Aug 23 reply concluded the error is not in the model but in the
comparison — return construction, factor scaling, Treasuries — and proposed
running the machinery on an instrument of known duration, or that the spread
control is broken and TBA empirical durations genuinely exceed cashflow
durations because spreads move with rates. All four candidates below came back
negative. The gap is unchanged at ~1.35 and its cause remains unknown.

### Known-duration test — the machinery is sound

A par Treasury put through the panel's own shocks and the same hedge
regression recovers its closed-form modified duration: 2yr 0.946, 5yr 0.971,
10yr 0.983 as fitted/analytic ratios. The shortfall is maturity-dependent and
was traced to a convention in the test, not the machinery — duration was
compared at start-of-month maturity against a return on a bond that had aged
one month. Evaluating duration at the aged maturity gives 0.988 / 0.988 /
0.993, a spread of 0.005 across tenors against 0.038 before. The residual ~1%
is not accounted for; candidates are the start-of-month yield versus the
realised end-of-month return, and convexity, neither tested.

The same harness returns a median 1.349 on the nine coupons. So the regression
path recovers a known duration and does not recover the TBA one.

Scope limit worth stating: the Treasury return was constructed here from
yields, so this tests the shock construction and the regression, NOT the FNCL
price series or the TBA total-return formula. `load_excess_returns()` is not
on this path at all — it still carries `D_MOD_AVG` and is the legacy Phase
18-21 route; the 1.33 lives in the panel path.

### Spread control — widens the gap

Adding the PMMS minus 10yr change as a fourth regressor moves the median
ratio from 1.352 to 1.381, rising at eight of nine coupons (6.5 is the
exception, falling 1.374 to 1.278). The spread coefficient is negative and
`corr(d_level, d_spread) = -0.519`, so omitting it biased the level
coefficient toward zero — the same sign mechanism Phase 22 recorded when the
spread control made the level t-statistic worse.

Note -0.519 here against Phase 22's -0.601: different statistics, not a
discrepancy. Phase 22 ran on the span panel where `d_level = (dy5+dy10)/2`
from clean.xlsx; this is tents3 where `d_level = (dy2+dy5+dy10)/3` from the
daily file.

### Roll/drop — bounded, and it fails where the gap is largest

The panel's TBA return is `(P_curr + c/12 - P_prev)/P_prev`, which omits the
drop. Using the June 2026 roll snapshot, the drop would have to swing 12.8x
its own level per 25bp to supply the missing duration at coupon 2.5 and 9.2x
at 3.0. At 5.5-6.5 the required multiple is 0.6-0.8, within what a drop can
do over a cycle — but those are the coupons where the gap is smallest, and
the gap is flat across coupons while the roll's capacity to explain it varies
twentyfold. Not the mechanism.

LIMITATION: one snapshot, nine coupons, treated as representative of 99
months. Bounds plausibility; does not measure the realised roll series.

### PMMS pass-through anchoring — total duration is invariant

Phase 26 tested `(0, 0.3, 0.7)`, which moves weight only between the 5yr and
10yr legs. Phase 24 had found the 2yr leg structurally excluded (`dp=0`), so
what a short-end anchor does to the TOTAL was never tested. Two arms,
`(0.3,0.4,0.3)` and `(0.33,0.34,0.33)`, reprice through the pricer rather
than rescaling after the fact:

    mean D_level : control 3.3520 | 3.3474 | 3.3475
    coupon 4.0   : krd2 0.668->0.351, krd10 1.758->2.397

Large reallocation, 0.14% change in the total, and the two arms agree to
three decimals. The total is insensitive to the split, not merely to one
split. Direction is consistent with the negative-convexity guess from Phase
26 (a leg that moves PMMS has its duration offset by faster prepayment) —
still untested, still a guess.

Control run reproduced `_srcdaily.csv` at md5 ecf4d2d2 before the arms ran.

### Ratio has no shape across coupons

Weighted fits on the per-coupon ratio with standard errors: constant-only
chi2 = 4.98 on 8 dof, linear t = -1.84, quadratic t = -0.24. The 1.21-1.43
range is inside estimation error (SEs 0.063 to 0.206, widening toward
premiums). A single scalar is the right description.

Temporal stability: expanding-window fits give median-over-fits 1.314 and
last-24-window mean 1.371 against full-sample 1.349. The final expanding
window equals the full sample exactly (0.00e+00), confirming the loop.

### Two retractions from this session

Both were stated as findings before being tested, and both are withdrawn.

A U-SHAPE IN THE RATIO ACROSS COUPONS. Raised on eyeballing 1.21 at coupon
5.0 against 1.43 at 3.5. The shape test above says noise. Spearman is blind
to a U, so Phase 23's -0.367/p=0.33 was never evidence either way — the
quadratic fit is the right test and it is flat.

A TEMPORAL DRIFT IN THE SCALAR. The expanding-window MEAN is 1.245, which
was read as the scalar drifting. It is early-window noise: minimum fitted
ratios run 0.317-1.10 in the short windows. Median and last-24 both sit at
the full-sample value.

### Trap: the panel's d_level spans FORWARD from info_date

`model_hedge_krd.py` pairs `prev = clean["Date"][i-1]` with `curr = Date[i]`
and writes `info_date = prev`, `ret_month = curr`, while taking
`d_level3.iloc[i]`, which is the diff from `prev` to `curr`. So the shock on
a row spans from that row's `info_date` to the NEXT row's `info_date`.
Joining external data on `info_date` and differencing in place is off by one
month.

Confirmed numerically: joining the panel's `d_level` against a rebuilt series
on the row's own `info_date` gives max |diff| 1.107; joining on the NEXT
`info_date` gives 9.4e-17.

This cost four wrong calls in one session before the join settled it. A
misaligned spread produces `corr(d_level, d_spread) = -0.0099` and a delta of
0.000 — a plausible-looking null. The correctly aligned series gives -0.519
and +0.029. Assert the window against the panel before using it; do not infer
alignment from a correlation.

Note also that Phase 22's identity `dy10 = d_level + d_slope/2` holds only on
the two-tent span panel. On tents3 the mean gap is 0.28pp, so pointing
`diag_spread_control.py` at a tents3 panel silently produces a wrong spread.

### Literature note, not a test

Secondary sources (MSCI, Salomon's effective-vs-empirical duration work)
report the standard finding as TBA empirical durations coming in SHORTER than
model durations, attributed to spreads tightening as rates rise. Ours are
longer, i.e. the opposite sign to what that channel produces — consistent
with the spread control widening the gap rather than closing it. Read but not
tested; much of that literature also uses swap rather than Treasury shocks.

### Open

- Root cause of the ~1.35 under-sizing still unknown. Pricer side exhausted
  (Phase 27); machinery, spread, roll and anchoring now added.
- The FNCL price series and the TBA total-return formula are NOT covered by
  the known-duration test.
- The broad reading of the advisor's second hypothesis — that empirical
  duration genuinely exceeds cashflow duration through a spread-response
  channel the pricer structurally cannot have — is untested. Under it, 1.33
  is an answer rather than a bug.
- Residual ~1% in the Treasury recovery unexplained.
- Carried forward: bootstrap_v3 off by default; short-end convention vs
  QuantLib; scalar still fitted full-sample; build_hedge_panel.py unchanged.

### Addendum, same session — two gaps closed after the section above was written

TBA RETURN FORMULA — cleared. The known-duration test built the Treasury
return from yields and so never touched the FNCL price series or the return
formula; that limit is now removed. `diag_tba_return_check.py` reproduces the
panel's stated formula to 1e-17 and compares against the workbook's own
`Raw_MoM_Returns` sheet: correlation 0.9997-0.9999 at all nine coupons, and
the mean difference tracks c/12 exactly (22.80bp observed vs 20.83 at coupon
2.5; 47.30 vs 50.00 at 6.0), i.e. the workbook return is price-only and ours
is total. Critically the difference does NOT load on the level shock —
t between -0.13 and -1.18, implied duration error 0.001-0.009 years against a
~1.9y gap. Carry that is near-constant month to month cannot masquerade as
duration.

SPREAD CHANNEL — measured, and it predicts the opposite sign. The Phase 28
test asked whether a spread REGRESSOR absorbs the gap. The prior question is
what the channel predicts. If dP/P = -D_cash*(d_level + d_spread) and
d_spread = beta*d_level, then regressing on d_level alone recovers
D_emp = D_cash*(1+beta).

`diag_spread_channel_sign.py`: beta = -0.4734, t = -5.95, n = 98. The spread
TIGHTENS as rates rise, so the channel predicts D_emp/D_cash = 0.527. We
observe 1.35. Stable in sign across halves (0.379, 0.573).

Two consequences. First, the advisor's Aug 23 second hypothesis — that
empirical durations exceed cashflow durations because spreads move with rates
— predicts the gap in the OPPOSITE direction on this data, so it cannot be
the explanation. This also supersedes the "literature note" above: the point
is now measured here, not read from secondary sources. Second, the unexplained
residual is LARGER than 1.35 implies, since a well-identified channel should
be pulling the ratio below one and something is overcoming it.

Caveats: the arithmetic assumes the spread enters the discount rate one for
one, a first-order framing rather than a derivation; and PMMS is the primary
mortgage rate, not the TBA's own spread, which is the object that properly
belongs there. Neither affects the sign.

## Phase 29 — Secondary Market Spread: Sign Flips as Predicted, Gap Narrows but Does Not Close (August 27, 2026)

Phase 28 closed by naming its own limitation: PMMS is the primary mortgage
rate, not the TBA's own spread. The advisor's Aug 27 reply made that the
diagnosis — the item discounting the TBA is the secondary market spread, and
the two move differently. He asked for a secondary series (Urban Institute
dealer OAS, or Bloomberg current coupon), a regression of the spread change on
the level shock expecting a large positive beta, and then that spread swapped
in as the control in place of PMMS − 10yr.

All three were run. The beta comes out positive as predicted and the control
moves the ratio in the right direction, but it does not close the gap.

### Current coupon from the FNCL price grid

The secondary rate is built as the FNCL coupon that prices at par, interpolated
each month between the two coupons that BRACKET par. The bracketing is done in
coupon order, never by sorting on price: FNCL prices are non-monotonic in
coupon at the premium end — 2018-01 has 6.0 at 111.33 and 6.5 at 109.61 — so a
price sort scrambles coupon order and corrupts the interpolation.

The current coupon is NOT IDENTIFIED in 26 of 101 months, almost all of them
2020-01 through 2021-12. In those months every quoted coupon is above par and
the 2.5/3.0 price slope is flat or inverted, so extrapolating below 2.5 gives
implied coupons ranging from −8.09 to +22.06 with a standard deviation of 6.20
on a quantity that should sit near 2%. Those months are DROPPED, not
extrapolated. The working sample is therefore 69 months and EXCLUDES the QE
window.

`secondary_spread = current_coupon − 10yr`. Mean level 1.186 against 2.244 for
PMMS − 10yr on the same months.

### The channel — sign flips, and it is the definition not the sample

`diag_secondary_spread.py`: beta of d_spread on d_level = **+0.3042, t = +6.05,
n = 69**, against the Phase 28 PMMS result of −0.4734.

The obvious objection is that the sample changed. It did not do the work: PMMS
re-run on the SAME 69 months gives −0.4354, t = −4.31. corr(d_level, d_spread)
is +0.5944 for the secondary spread and −0.4658 for PMMS on identical months,
and corr between the two spread changes is −0.3301. The sign flip is
attributable to the spread definition.

Implied D_emp/D_cash = 1 + beta = 1.3042, against an observed 1.35 and a
required +0.35. The advisor's mechanism now predicts the right direction and
close to the right magnitude.

### The control — 1.349 → 1.197

`verify_secondary_spread_effect.py`, cloned from `verify_spread_effect_v2.py`
with only the spread series changed:

| control | median ratio3 | median ratio_spr | change |
|---|---|---|---|
| PMMS − 10yr (reduced sample) | 1.349 | 1.382 | +0.033 |
| current coupon − 10yr | 1.349 | **1.197** | **−0.153** |

The PMMS arm reproduces the Phase 28 result (1.352 → 1.381) on the reduced
sample, so the comparison is like-for-like. The t on the spread regressor runs
−8.26 to −4.08 at coupons 2.5 through 5.5, against roughly −2 to 0 for PMMS.

### Verification

- ASSERT 1 kept verbatim from v2: d_level rebuilt from the two window
  endpoints matches the panel column to 9.4e-17, so the spread is measured
  over the same forward window as d_level.
- ASSERT 2 could NOT be carried over. v2 checked start-of-window PMMS against
  the panel's own `pmms` column; the secondary spread has no panel counterpart,
  so there is nothing to assert against. Replaced by a coverage guard, which is
  a weaker guarantee and is labelled as such in the script.
- Current coupon re-derived by a SECOND construction — quadratic through the
  three coupons nearest par, solved for price = 100 — giving beta +0.3107
  (t +6.16) and ratio 1.193. Max disagreement with the bracketing method is
  2.8bp, mean 0.8bp, corr 0.999983, identical 75-month coverage.
- Leave-one-coupon-out: change ranges −0.150 to −0.160 across all nine drops.
  Mean instead of median gives −0.148. Excluding 6.0/6.5 gives 1.346 → 1.188.
- Input file structure checked directly: 103 rows = title row + header row +
  101 months, no all-NaN rows, no unparseable dates, coupon labels read from
  the header and spot-checked against raw values.

### What does NOT work

**It does not close to 1.0.** 1.197 leaves roughly 60% of the excess standing.
The advisor's expectation was that this "should finally close the issue"; it
does not.

**It fails where the gap is worst.** At coupons 6.0 and 6.5 the ratio is 1.428
and 1.856 and the spread t is −1.81 and −0.55 — the control does essentially
nothing there. This is the SECOND independent test to fail at the premium
coupons after the Phase 28 roll bound, which also could not reach them. Two
tests failing in the same place is a pattern, not a coincidence, and it
suggests the premium residual is a different mechanism.

**Pre-2020 the channel is not identified.** beta +0.0347, t +0.48, n 21,
95% CI [−0.106, +0.175], against 2022+ beta +0.3385, t +5.55, CI
[+0.219, +0.458]. The CIs do not overlap, but sd(d_level) is 0.1621 pre-2020
against 0.3102 after 2022 — rates barely moved. This is a low-variation period
rather than a demonstrated regime break, and should not be written up as
"the channel is post-2022 only."

**Circularity caveat.** With beta = +0.30 measured against the same d_level,
a controlled ratio near 1.35/1.30 is close to what the arithmetic already
implies. The control test is not fully independent evidence of the channel.

### Repo anomaly — shock-source robustness arm could not be run

`outputs/model_hedge_panel_10_tents3_pinnedfixed_srcdaily.csv` is
BYTE-IDENTICAL to the base panel (both md5 `ecf4d2d2`), so re-running the
result on the alternative Treasury source tests nothing. The flag itself works:
the span pair (`27dc2d07` vs `7b1d9af8`) genuinely differs, and the two
`srcdaily_pt*` variants from the same Aug 23 session have distinct md5s. No
committed sbatch combines tents3 with daily shocks —
`run_hedge_srcB.sbatch` passes `--spanning`, not `--bump-shape tents3`. That
file was hand-launched and dropped `--shock-source daily`. Regenerating it is
open work; until then no source-robustness claim should be made for tents3.

### Scripts

- `scripts/diag/diag_secondary_spread.py` — builds the current coupon and
  measures the channel; clone of `diag_spread_channel_sign.py` with the spread
  series swapped.
- `scripts/diag/verify_secondary_spread_effect.py` — the control test; clone of
  `verify_spread_effect_v2.py`, runs both spread definitions on the same
  reduced sample.

## Known Defect — Prepayment Label Column (found August 28, 2026)

Found while starting the historical buildout. The hazard model's training
label has been read from the wrong column in both sequence builders. Recorded
here rather than as a Phase because it is a pipeline defect, not a step in the
duration investigation.

### What is wrong

Both builders map the label the same way — `prepare_sequences_rolling.py` at
line 123 and `prepare_sequences_extended.py` at line 106 both use
`index('extra_13') + 1` for `zero_balance_code_actual`, then set `prepaid`
from `== 1.0` on it. But `extra_13` is usecols 106 (awk field 107) and is not
the zero-balance code. The zero-balance code is usecols 43 (awk field 44).

### Evidence

2013Q1, full file, censored at the cutoff exactly as the rolling builder does
it (MMYYYY converted to YYYYMM before any comparison):

| cutoff | loans | zero_balance_code == 01 | extra_13 == 1 |
|---|---|---|---|
| Dec 2018 | 681,364 | 236,823 (34.8%) | 0 (0.00%) |
| Dec 2020 | 681,364 | 354,363 (52.0%) | 3,421 (0.50%) |
| Dec 2022 | 681,364 | 459,279 (67.4%) | 6,791 (1.00%) |

Censoring affects both columns identically, so it does not explain the gap.

`extra_13` is not an under-inclusive subset of true prepayments either. On
2016Q1 at cutoff 202212, of the 7,940 loans it flags only 2,681 (34%) are also
`zero_balance_code == 01`. Mean borrower credit score is 751.7 for the
population, 751.5 for true prepayments, and 723.1 for `extra_13` loans, so
true prepayments look like the population while `extra_13` selects roughly 28
points lower.

Existing artefacts are consistent with this: rolling cutoffs 2020 through 2023
have label rates 0.0090 / 0.0147 / 0.0155 / 0.0166, and `sequences_extended`
0.0254. Rebuilding cutoff_2020 with the correct column gives 52.15% on 2013Q1
and 53.85% on 2013Q2.

What `extra_13` actually is remains unidentified. It is systematic — roughly
7,000 to 8,000 loans per vintage regardless of vintage size, co-occurring with
C/7/P/D codes in the adjacent field. It should not be named without evidence.

### Downstream signature — consistent, not demonstrated

`outputs/forecast_vs_realized_cpr_gfee050.csv`, 2020 onward, binned by refi
incentive, forecast divided by realized:

| incentive | ratio |
|---|---|
| -2.5 to -0.5 | 0.86 to 0.95 |
| -0.5 to +0.5 | 0.95 to 0.98 |
| +0.5 to +1.5 | 0.70, 0.60 |
| +1.5 to +2.5 | 0.64, 0.67 |

The model captures about 60% of realized CPR precisely where refinancing
happens, and tracks well where it does not. That is a shape distortion rather
than a level shift, so no single Platt scalar repairs it. It matches the
earlier finding that the model peaks 0.5 to 1.25 incentive points below where
realized loans respond, and is consistent with the credit-score skew, since
lower-score borrowers refinance less readily at a given incentive.

This is a consistent signature, NOT a demonstrated causal link. The test is a
retrain on corrected labels, which has not yet been run.

### Scope

Affected: the training target in both builders, therefore the production model
and every rolling cutoff.

Not affected: realized CPR, which `realized_cpr_v6_upb.py` derives from UPB
disappearance without touching this column; the DER regression's realized leg;
and the Phase 29 duration and spread work, which uses TBA prices and Treasury
yields only.

### Trap for anyone fixing this

Do not swap in a name lookup. `_ALL_COLS` holds 109 names for 113 fields and
drifts, so `_ALL_COLS.index('zero_balance_code') + 1` returns usecols 42,
which is not the verified column. Hardcode 43 with a comment. All other mapped
columns were checked against 2018Q1 values and the features are correct — rate
4.250, loan_age 0/1/2, credit score 791, origination date 012018 — so only the
label is affected.

### Status

`scripts/prepare_sequences_rolling_zbc.py` is a copy with the one-line label
fix, writing to `data/sequences_rolling/cutoff_{year}_zbc/`. A cutoff-2020
rebuild is running at `--sample_frac 0.3`, which subsets unique loan IDs at
discovery and so does not affect the label logic. The retrain and the
forecast comparison have not yet been run.

## Label column defect (Aug 29, 2026) — supersedes the Phase 16 "no signal before 2020" finding

**What was wrong.** Both sequence builders (`prepare_sequences_rolling.py` line 123,
`prepare_sequences_extended.py` line 106) and the realized leg of
`forecast_rolling_cpr.py` read the prepayment label from `_ALL_COLS.index('extra_13')+1`
= usecols 106. That is not the zero-balance code.

**What col 106 actually is.** Field position 107 in the vendor's published file layout is
Alternative Delinquency Resolution Count. Verified in data: every non-empty value
co-occurs with P/C/D/7 in field 106 (Alternative Delinquency Resolution), and the dominant
pair is `1|C` — one COVID-19 payment deferral — at 546,498 rows in 2018Q1. The counts run
1, 2, 3. So the models were trained to predict how many payment deferrals a loan received.

**The correct column.** Field position 44 = usecols 43, code 01 = Prepaid. Confirmed
against the published layout and by direct read across 2000Q1 / 2012Q4 / 2018Q1.
Hardcode 43 — do NOT use a name lookup: `_ALL_COLS` holds 109 names for 113 fields and
`index('zero_balance_code')+1` returns 42.

**Why this produced the false "no signal before 2020" result.** Field 107 is only populated
from the July 2020 activity period. It does not exist in earlier windows, so every pre-2020
cutoff necessarily measured 0.00% prepay. That was read as regime concentration.

Corrected, cutoff_2018 gives **23.31%** pooled prepay across 1,178,894 loans, per-vintage
35.40% (2015Q1) declining monotonically to 0.19% (2018Q4) — the right shape for a
cumulative ever-prepaid-by-cutoff label. Corroborated by `realized_cpr_v6_upb`, which
derives payoff from UPB disappearance and never reads this column: annual CPR
6.2 / 12.3 / 15.5 / 9.3 / 7.8 / 13.3% for 2014–2019.

**Second defect, found while fixing the first.** `loan_age` is blank on every payoff row
(71,559 of 71,559 zbc==1 rows in 2015Q1). Since `loan_age_months` is in `FEATURE_COLS`,
the `dropna` in `load_vintage_filtered` deleted 100% of prepayment rows, leaving
`prepay_timestep` all -1 while the loan-level label — computed before the dropna —
survived. Same root cause as the Aug 5 age-keyed realized CPR bug. `loan_age_months` is
now derived from origination date minus a one-month offset (measured: 382,207 of ~400k
non-null rows at derived-minus-field == 1; a ~4.4% tail sits at 0/2/6/9), clipped at 0.

**Also corrected.** `zero_balance_code` at usecols 43 IS a one-time stamp (71,559 loans,
min/median/max rows per loan all 1), so `.min()` in `build_sequences` is the right reducer.
This supersedes the June 26 note that "col 106 persists for many months post-payoff" —
true of the deferral counter, not of field 44.

**Scope.** Affects the training target in both builders, the production model, all rolling
cutoffs, and the realized leg every forecast-vs-realized comparison was scored against.
Does NOT affect `realized_cpr_v6_upb.py`, the DER realized leg, the pre-2013 event count
table (`count_prepay_events_pre2013.py` already reads col 43), or Phase 29.

**Retrain on corrected labels.** cutoff_2020: AUC 0.5966, Platt a=2.4245, b=-2.4348.
Weak, and NOT comparable to the prior 0.7006 — that number measured deferral prediction,
a different and easier task. These Platt params are a third calibration and must not be
mixed with the OAS loan-level (0.4934 / -4.840) or cohort-CPR forecast (0.4559 / -3.1376)
sets.

**Open design question — the 33-month window.** 242,289 of 571,561 prepaid loans at
cutoff_2020 prepay outside the 33-month sequence window and are correctly treated as
censored non-events (58% of positives placeable at cutoff_2020, 69.5% at cutoff_2018).
The sampler draws its target from `prepay_t`, not from the label array, so this is proper
discrete-time censoring rather than mislabeling. But it means the model estimates
early-life prepayment hazard only. The window was flagged as an open question in June and
held at 33 to keep an old-vs-new model comparison clean; that rationale no longer applies.
Not yet resolved.

## Prior-shift correction to the rolling forecast (Aug 30, 2026)

With corrected labels, the first forecast run on `cutoff_2020_zbc` gave forecast CPR of
31–86% against realized 6–36% — wrong by 2–8x at every coupon. The cause is not the
label fix. It is that `forecast_rolling_cpr.py` compounds the raw sigmoid without any
correction for the training sampler's oversampling.

**The fix was verified as necessary before it was written.** The old deferral-trained
`outputs/rolling/cutoff_2020/rolling_cpr_forecast.csv` (Jun 23) uses the same script and
the same construction, and shows the same inflation: 8.5% forecast against 0.30% realized
at coupon 2.0, 99.8% against 14.6% at coupon 6.0. The raw-sigmoid path has never produced
calibrated levels. Note this file is NOT the same object as
`outputs/rolling_forecast_vs_realized.csv`, which is the stage2 synthetic
representative-loan construction and does carry a calibration.

**The correction.** `HazardSampler` draws half its loans from `prepaid_idx`, then samples
one timestep per loan and labels it `t == prepay_t`, so most draws in the positive half
land on non-event timesteps. The effective positive rate must be measured, not assumed —
an initial attempt using 0.5 overshot and produced 1.42% against an 11.55% target.
Simulating the sampler's own draw gives **p_train = 0.04732** against a per-person-month
**p_true = 0.01017**, for a logit offset of **−1.5758**.

`prior_shift_offset()` derives both rates from the training arrays at runtime. It has no
free parameters and nothing fitted to realized CPR. This was deliberate: a two-parameter
calibration against realized would land the forecast almost exactly on target and thereby
make "calibration against realized CPR" circular as a criterion for choosing between
window lengths. `--no_prior_shift` reproduces the uncorrected path.

**Result** (job 16616280, `outputs/rolling/cutoff_2020_zbc/rolling_cpr_forecast.csv`).
Pooled over the seven coupons with ≥5,000 loans (229,017 loans): forecast **22.78%**
against realized **26.39%**, ratio **0.863**.

| coupon | forecast | realized | ratio | n_loans |
|---|---|---|---|---|
| 2.0 | 21.75 | 11.33 | 1.92 | 22,858 |
| 2.5 | 22.69 | 15.03 | 1.51 | 38,641 |
| 3.0 | 20.02 | 26.11 | 0.77 | 70,896 |
| 3.5 | 21.12 | 33.84 | 0.62 | 39,043 |
| 4.0 | 25.49 | 35.17 | 0.72 | 41,497 |
| 4.5 | 32.73 | 35.33 | 0.93 | 10,628 |
| 5.0 | 35.44 | 36.01 | 0.98 | 5,454 |

Coupons 1.0 and 1.5 (21 and 786 loans) are too thin to characterise and are excluded.

**What remains is shape, not level.** The error changes sign across the curve — too high
at 2.0–2.5, too low at 3.0–4.0, calibrated at 4.5–6.0. Forecast CPR spans 1.79x across
coupons 2.0–6.0 where realized spans 3.21x. The model's incentive response is too flat.
This is consistent with the long-standing finding that the model peaks 0.5–1.25 incentive
points below where realized loans respond, but that was measured on the pre-fix model and
does not transfer automatically.

**Sampler defect found and fixed before it could run.** `HazardSampler.sample_batch`
allocated batches at the module constant `MAX_SEQ` rather than at the array width, so a
48-month run would have trained on 33-wide batches under a 48-row embedding with 15 rows
never updated — silent, and only visible as an unexplained result later. Width is now
`sequences.shape[1]`, and `train_hazard_rolling.py` asserts `--max_seq` against the loaded
array width.

**Sequence cap parameterised.** `--max_seq_len` on both prep builders, `--max_seq` on the
trainer, `max_seq` written into the saved checkpoint config, and `load_model` reads it via
`cfg.get('max_seq', MAX_SEQ)` so pre-existing checkpoints fall back to 33 unchanged.
Non-default caps append `_L{n}` to output directories. Note the constant is `MAX_SEQ_LEN`
in the prep scripts and `MAX_SEQ` in the model-side ones, and it appears in ~40 files —
only these four are on this path; the rest hold independent literals and are unaffected.

**Three hypotheses raised and killed by testing.** Mask-based label leakage (sequence
length alone gives AUC 0.415 / 0.545 against the label, near chance — the exact match
between `prepay_t >= 0` and the label below the cap is a definitional tautology, not an
information channel); a last-timestep sampling bias in `infer_test_set` (hazard at the
last real timestep is 0.97x the all-timestep mean, not hotter); and the assumption that
the sampler's positive rate is 0.5. Recorded because each looked convincing before it was
measured.

## Time-varying inference (Aug 31, 2026) — implemented, one cutoff validated, one blocked

`forecast_rolling_cpr.py --time_varying` replaces the single-hazard extrapolation with a
twelve-pass forward loop. For each forecast month it recomputes `refi_incentive` from
contemporaneous PMMS, `current_ltv` from contemporaneous ZHVI, and `loan_age_months`,
scales them with the training `scaler.pkl`, substitutes them into the sequence's last
valid timestep, and compounds the twelve monthly hazards into `1 - prod(1 - h_m)`. The
training window is untouched — the builder still truncates at the cutoff, so only the
inference inputs move. Substitution into the last slot was chosen over extending the
sequence because extension would index position embeddings beyond the trained `max_seq`
and require a retrain.

`zip3` and `origination_date` were added to `_RAW_COL_MAP` for this, with a runtime range
check that asserts zip3 in [1,999] and a decodable month in [1,12] before the values are
used anywhere.

### cutoff_2020 → 2021: the fix did not improve calibration

Pooled over the seven coupons with at least 5,000 loans (229,017 loans), forecast/realized
moved 0.8631 → 0.7691 — further from one, not closer. Every coupon's forecast moved down.
That helped where the model over-forecast (2.0: 1.920 → 1.676; 2.5: 1.510 → 1.206) and hurt
where it already under-forecast (3.0: 0.766 → 0.618; 3.5: 0.624 → 0.588). Coupon 4.0 and 5.0
are unchanged to three decimals. Dispersion across the seven tightened (sd 0.4752 → 0.3845,
spread 1.297 → 1.088), so the bias is more uniform, but a more uniform bias that is further
from one is not a calibration win and is not reported as one. Why the shift is uniformly
downward rather than concentrated near the money is NOT established.

### cutoff_2022 built; its forecast is INVALID

`cutoff_2022_zbc` was prepped (job 16637540) and trained (job 16653509, best AUC 0.7627 at
epoch ~45, epoch 50 ended 0.7480) because `cutoff_2020_zbc` was the only cutoff with a
corrected-label model, which silently blocked any multi-cutoff validation. Sequences:
train (20391761, 33, 9), test (5097941, 33, 9). Both splits report prepay 45.37% — identical
by construction, not coincidence: `train_test_split(..., stratify=labels_1p)` at line 441
of the zbc builder forces it. Train/test loan-id overlap measured at 0.

The forecast output in `outputs/rolling/cutoff_2022_zbc_tv/` is NOT usable. See below.

### Invariant — `loan_age_months` is window-relative, not calendar age

**This is the defect that invalidated the cutoff_2022 time-varying forecast, and it is the
kind of convention that is expensive to relearn.** In the training sequences, `loan_age_months`
is measured from the start of each loan's observation window, not from origination. Printed
directly from `train_seq.npy`: row 0 runs 1,2,3,…,33; row 1 runs 0,1,…,18; row 2 runs 0,1,…,32.
A loan originated December 2012 has sequence age starting near 0. The feature is a window
position bounded by the cap, not a calendar age. Measured distribution at the last timestep:
min −0.0, max 82.0, p99 34.0, median 26.0.

The `--time_varying` path computed calendar age from `origination_date` instead, producing
128–142 months (median 130) for seasoned loans — roughly 4x the p99 of the training range.
The model extrapolates to near-zero hazard there. Signature: coupons 2.0–4.0 forecast
0.41–0.56% CPR against realized 5.45–6.60%, while the frozen run over-forecast the same
coupons by up to 2.9x. Pooled over all 14 coupons (2,773,315 loans), frozen 1.198 vs
time-varying 0.156.

That the training data supports a real floor here was checked independently: the empirical
per-person-month rate in the `[-4,-2)` incentive bucket is 3.1168% (n=6,892,113), which under
the same prior-shift offset implies ~7.65% annual CPR — close to realized, and an order of
magnitude above what the model produced. So the collapse is not the model faithfully
reporting an absent floor.

`current_ltv` was checked separately and is NOT a second defect of this kind: the training
convention is a true LTV in percent units declining from `original_ltv` by amortization
(row 0: 80.0 → 69.7 → … → 53.9), the same scale the `--time_varying` path computes. However,
the recomputed Dec-2023 median (33.3) sits well below the training median (63.6) while
remaining inside the training range (min 2.4), and whether that is correct for seasoned
loans after 2020–23 house-price appreciation is NOT established.

### Two root causes proposed and refuted before the real one

Recorded because both were argued confidently from structure before anything was printed.

**A raw-file field offset.** Proposed on the reasoning that the raw rows begin with a leading
delimiter, so `_ALL_COLS.index(name) + 1` would be off by one. Refuted by reading the columns
back: `usecols=13` returns `122012` for a loan whose reporting period is `022013`, and `zip3`
at `usecols=32` returns a valid prefix. The `+1` is correct for these fields; the prep script
uses the identical expressions and its features are sound.

**loan-id reuse across vintages.** Proposed to explain an apparent 85–87 month age gap. Not
supported: the gap is fully explained by comparing a window-relative age against a calendar
age, with no reuse required. The diagnostic written to test it was itself broken (it matched
`$1`, but the leading delimiter puts `loan_id` in `$2`) and returned empty for every id; it
was deleted rather than committed.

### Status

Blocked: `loan_age_months` must be fed as a window-relative value continuing from the last
observed timestep, not as calendar age, and the cutoff_2022 forecast rerun both ways. The
33/48/60 window comparison stays blocked behind that, since all three windows would inherit
the same defect.

## Sequence window anchoring (Aug 31, 2026) — supersedes the `loan_age_months` invariant in the section above, and the message of commit 5629725

**What was wrong.** The section above states that `loan_age_months` in the training
sequences is "window-relative, not calendar age" — a window position bounded by the cap.
That is backwards. The feature IS calendar age: `prepare_sequences_rolling_zbc.py` lines
300-306 derive it as months from `origination_date`, minus a measured one-month offset,
clipped at 0. There is no second convention.

**Why the ages nonetheless look low.** `build_sequences` takes the first `MAX_SEQ_LEN`
reporting rows present for each loan — line 343 ("Takes the FIRST MAX_SEQ_LEN months per
loan chronologically") enforced by `cumcount` at line 353. Windows are anchored at the
start of the loan's data, not at the cutoff. Ages appear bounded only because the window
is. Verified: random 20,000 of the 374,182 cutoff_2020 test sequences show first-timestep
age at ~0 for 99.8% of loans and `corr(last-first, L-1) = 0.9994`.

**Evidence that settles it.** `logs/diag_origdate_16679693.out` lists loans with sequence
age 32-33 whose origination dates are Nov 2012 - Mar 2013 and whose true calendar age at
the Dec 2022 cutoff is 116-120 months. Both numbers are correct and describe different
moments: the window covers roughly 2013-2015, the forecast month is Dec 2023. A loan's
window can end years before the cutoff.

**The 82-month tail is not a gap artifact.** Age advances by exactly 1 per timestep:
0.0% of 50,000 sampled training rows contain any step greater than 1. The highest-age loan
sampled runs 22, 23, ..., 54 contiguously — it starts at 22 because its rows in the vintage
file start 22 months after origination, not because months are missing. So the window is
the first `MAX_SEQ_LEN` rows PRESENT for that loan, wherever its data begins. Why some
loans' data starts late is NOT established. (The 82 figure comes from
`diag_feature_ranges_16676478.out`; the sample here maxes at 54.)

**The prescribed fix does not work.** The Status paragraph above says `loan_age_months`
must be fed "as a window-relative value continuing from the last observed timestep." Since
the last observed value already IS calendar age, continuing it forward reproduces the same
128-142 months that caused the collapse. Clipping to the training range instead would feed
a fabricated age alongside real forecast-date rates, producing a plausible number with no
support. Neither is a fix.

**What this actually is.** Not a coding defect. It is the 33-month window question, open
since June, appearing at inference: a loan seasoned past the window has no in-range age at
any forecast date, so the model has never seen a seasoned loan at a seasoned age with
contemporaneous rates. There is no local patch in `infer_test_set_time_varying`. Note the
consequence for the 33/48/60 comparison: a 2013 loan at a 2022 cutoff is ~120 months old,
so widening to 60 does not reach it either. Whether widening improves forecast-date
calibration is NOT tested; the comparison should be judged on calibration at the forecast
date rather than on training-window event coverage alone.

**Commit message 5629725 is wrong** and cannot be edited without a history rewrite. It
asserts the window-relative invariant and names it as the root cause. Trust this section
over that message.

**What survives from the section above, unchanged.** All of it except the invariant and the
Status prescription. The cutoff_2020 results (pooled 0.8631 -> 0.7691, per-coupon moves,
dispersion tightening), the cutoff_2022 collapse (frozen 1.198 vs time-varying 0.156,
coupons 2.0-4.0 at 0.41-0.56% against realized 5.45-6.60%), the `[-4,-2)` bucket check
(3.1168% per person-month, n=6,892,113, implying ~7.65% annual CPR) showing the floor is
real, the implementation description, and both refuted hypotheses all stand. The loan-id
reuse refutation holds, but its stated reasoning changes: the age gap is explained by the
window sitting years before the forecast month, not by a window-relative vs calendar
mismatch.

## Trailing-window anchor test (Sep 1, 2026) — large AUC gain, calibration regression

**Why this was run.** The advisor's July 3 message specified the training design as "the full
rolling prediction window (predict t+1 every period based on date-t information)." The shipped
builder does not implement that: `build_sequences` takes the FIRST `MAX_SEQ_LEN` rows per loan
(line 343/353), so a loan originated in 2013 and still alive at a 2020 cutoff is scored from a
2013-2015 window. `prepare_sequences_trailing_zbc.py` is a copy of the zbc builder with the window
selection changed to the LAST `MAX_SEQ_LEN` rows (`cumcount(ascending=False)`), writing to
`data/sequences_rolling/cutoff_{YEAR}_zbc_trail`. Nothing else differs.

**No event truncation is needed.** Fannie stops reporting a loan after its zero-balance row:
121 of 121 payoff loans in a 2015Q1 slice have zero rows after payoff. So for a prepaid loan the
last row IS the payoff row, and a trailing window terminates at the event automatically.

**Window verified before training.** Job 16697085 on one vintage: last kept row equals the loan's
max month for every loan; first-row age median 33 against ~0 for the origination-anchored build.
On the full build (job 16697151), the highest-age sampled loan runs 65, 66, ..., 97 — 33
consecutive months ending at the cutoff, a loan being scored at age 97 that the origination-anchored
build could only ever show at age <= 33. Young loans (62% of the sample) still start near age 0
because they have not lived 33 months; that is correct, not a failed flip.

**Result 1 — discrimination improves substantially.** Identical loans, identical split
(n_train 1,496,727 / n_test 374,182 in both), identical labels, identical architecture and epoch
count. Best AUC 0.5966 -> 0.7553. The trajectories matter more than the headline: the
origination-anchored model starts at 0.5849 and ends at 0.5713 — fifty epochs and the last epoch
is worse than the first, i.e. it never learned. The trailing model starts at 0.6823 (already above
anything the other reached) and climbs to 0.7238 by epoch 50. Late-epoch AUC oscillates roughly
0.72-0.76, so 0.7553 is the best draw rather than a stable level.

**Result 2 — coupon-level calibration gets WORSE.** Per-coupon forecast/realized, seven coupons
with n >= 5,000:

| coupon | realized | origination | trailing | orig ratio | trail ratio |
|---|---|---|---|---|---|
| 2.0 | 11.33 | 21.75 | 37.28 | 1.920 | 3.291 |
| 2.5 | 15.03 | 22.69 | 39.10 | 1.510 | 2.602 |
| 3.0 | 26.11 | 20.02 | 31.07 | 0.766 | 1.190 |
| 3.5 | 33.84 | 21.12 | 21.08 | 0.624 | 0.623 |
| 4.0 | 35.17 | 25.49 | 19.66 | 0.725 | 0.559 |
| 4.5 | 35.33 | 32.73 | 34.20 | 0.926 | 0.968 |
| 5.0 | 36.01 | 35.44 | 44.82 | 0.984 | 1.245 |

Realized CPR rises monotonically 11.3 -> 35.2 across coupons 2.0-4.0. The trailing forecast runs
37.3 -> 39.1 -> 31.1 -> 21.1 -> 19.7 across the same range — declining where the truth rises.
The origination-anchored forecast was nearly flat (20-25% across 2.0-4.0), consistent with its
AUC of 0.57; the trailing forecast has structure but the structure is inverted through the middle.

Dispersion widened from 1.920..0.624 to 3.291..0.559. The loan-weighted pooled ratio moved
0.8631 -> 1.1273, which is nominally closer to one, but only because larger errors in opposing
directions cancel more completely. Applying the same standard used for the Aug 31 time-varying
result: a pooled number closer to one produced by LESS uniform bias is not a calibration win and
is not reported as one.

**What is ruled out as the cause.** The realized leg is byte-identical across both runs, so this
is entirely model-side. Same loans and same split, so it is not sample composition. The forecast
path uses raw sigmoid plus the prior-shift offset and never reads a Platt file (`grep -ic calib
scripts/forecast_rolling_cpr.py` returns 0), so the extreme trailing Platt fit is not acting here.
The prior-shift offset does differ (-0.8291 trailing vs -1.5758 origination-anchored, from
p_train 0.03954 vs p_true 0.01765), but a constant logit shift moves the level uniformly and
cannot invert a slope.

**Why discrimination and calibration move in opposite directions is NOT established.** AUC is
computed loan-level on the test window, which for the trailing build ends at the cutoff; the
forecast is about the following year. Ranking loans well within 2018-2020 need not carry into
2021 levels. That is a hypothesis, not a finding.

**Fourth Platt calibration — never mix.** Trailing zbc 2020: a=12.9671, b=-13.0827. The four
now in existence are OAS loan-level (0.4934 / -4.840), cohort-CPR forecast (0.4559 / -3.1376),
corrected-label zbc 2020 (2.4245 / -2.4348), and this one. Prior-shift logit offsets are a separate
mechanism again.

**Consequence for the 33/48/60 window comparison.** That comparison varies window LENGTH, not
ANCHOR. This result suggests anchor is the larger lever, and that it does not move calibration in
the helpful direction on its own. Whether length interacts with anchor is untested.

Artifacts: `scripts/prepare_sequences_trailing_zbc.py`, `scripts/diag_trailing_window.py`,
`slurm/{prep_trail_2020.sbatch, diag_trailing_window.slurm, rolling_train_2020_trail.slurm,
rolling_forecast_2020_trail.slurm}`. Jobs 16697085, 16697151, 16719369, 16733177.
Outputs under `data/sequences_rolling/cutoff_2020_zbc_trail/` and
`outputs/rolling/cutoff_2020_zbc_trail/`.
