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

### Key finding — the t→t+1 design only has signal from cutoff_2020 onward
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
`scripts/diag_zbc_column.py`, `scripts/diag_prepay_vanish.py` — confirmed zero_balance_code is
at col 106 across all vintages and isolated the 0% prepay to the cutoff filter (genuine
regime concentration, not a column/label bug).

## Phase 16 (cont.) — Rolling forecast completion + pipeline hardening (June 21–22, 2026)

Recovered and completed the rolling t→t+1 pipeline after a series of SLURM/memory issues. Key fixes:
- **cutoff ≤ 2019 has zero prepay signal** (confirmed: cutoff_2019 = 13.9M loans, 0.00% prepay; all prepayments are in the 2020–21 boom). Rolling estimation runs cutoffs 2020–2021: cutoff_2020 (0.90% prepay) → forecast 2021; cutoff_2021 (1.47%) → forecast 2022.
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
