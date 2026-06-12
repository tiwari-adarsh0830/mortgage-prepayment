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
**Source:** https://loanperformancedata.fanniemae.com
**Format:** Post-October 2020 single-file format, 110 fields, pipe-delimited (`|`), no header row (col0 = empty due to leading pipe).

**Critical column note:** Field positions in the Fannie Mae data dictionary are 1-indexed. In pandas (0-indexed), Field N = col(N-1). The file has a leading pipe so col0 is always empty.

| Vintage | Loans | Prepay Rate | Rate Environment |
|---------|-------|-------------|-----------------|
| 2020Q1 | 341,184 | 3.84% | ~3% (COVID low) |
| 2020Q2 | 617,442 | 1.44% | ~3% |
| 2020Q3 | 691,357 | 0.83% | ~3% |
| 2020Q4 | 755,874 | 0.71% | ~3% |
| 2021Q1 | 705,614 | 0.63% | ~3% |
| 2021Q2 | 656,891 | 0.70% | ~3% |
| 2021Q3 | 518,391 | 0.89% | ~3% |
| 2021Q4 | 505,505 | 0.88% | ~3% |
| 2023Q1 | 105,087 | 0.71% | ~6.5–7% (high) |
| **Total** | **4,897,345** | | |

**Sequence data:**
- Train: 3,917,876 × 33 × 9, Test: 979,432 × 33 × 9
- Mask convention: True = real timestep throughout; inverted inside `forward()` for PyTorch attention

### Bloomberg TBA Data (pulled June 2026, Bobst terminal)
- **FNCL 2.5–6.5 Mtge:** Monthly last price, Jan 2018–May 2026, 32nds converted to decimal
- **USGG5YR / USGG10YR Index:** Monthly yields, same period
- **TBA Monitor:** Roll/drop snapshot for UMBS coupons (June 2026, point-in-time)
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
| `hazard_best.pt` | Discrete hazard model (AUC 0.8181) |
| `ddpm_conditional.pt` | Conditional DDPM checkpoint |
| `pmms_cond_paths.npy` | PMMS rate paths (refi incentive) |
| `treasury_cond_paths.npy` | Treasury paths (discounting) |
| `oas_cashflows.npy` | Cashflow matrix (1000 × 1000 × 33) |
| `oas_spreads.npy` | OAS spreads per loan |
| `stage2_coupon_cpr.json` | Hazard model CPR by coupon bucket |
| `der_betas.json` | DER β_x, β_y per coupon (time-varying) |
| `stage3_lambda_ts.csv` | Monthly λ_x, λ_y from Fama-MacBeth |
| `stage3_excess_returns.csv` | Treasury-hedged TBA excess returns |
| `realized_cpr_by_coupon_v4.csv` | Correct monthly realized CPR by coupon (in progress) |

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

---

## References
1. Diep, Eisfeldt, Richardson — "The Cross Section of MBS Returns" — *Journal of Finance* 76(5), 2021 (NBER w22851)
2. Gabaix, Krishnamurthy, Vigneron — "Limits of Arbitrage: Theory and Evidence from the Mortgage-Backed Securities Market" — 2007
3. Boyarchenko, Fuster, Lucca — "Understanding Mortgage Spreads" — NY Fed SR674
4. Ho et al. — "Denoising Diffusion Probabilistic Models" — arxiv 2006.11239
5. arxiv 2410.18897 — DDPM + wavelet for synthetic financial time series
6. arxiv 2511.17892 — HJM no-arbitrage neural yield curve
7. Fuster et al. — "Predictably Unequal?" — SSRN 3072038
