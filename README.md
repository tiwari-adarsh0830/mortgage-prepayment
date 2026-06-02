# Mortgage Prepayment Prediction
**NYU Stern — RA Project | Advisor: Prof. Arpit Gupta**

---

## Project Overview
Predicting mortgage prepayment using Fannie Mae Single-Family Loan Performance Data. The project builds a sequence of increasingly sophisticated models — from logistic regression through Transformer-based architectures — and is now extending toward Option-Adjusted Spread (OAS) pricing using AI-driven rate simulation and discrete hazard models.

---

## Repository Structure
```
mortgage_prepayment/
├── data/
│   ├── raw/                  # Raw Fannie Mae CSVs (not tracked in git)
│   ├── sequences/            # Preprocessed padded sequences (not tracked)
│   ├── pmms_monthly.csv      # Freddie Mac PMMS 30yr rates
│   ├── zhvi_zip3.csv         # Zillow ZHVI at zip3 level
│   └── treasury_yields.csv   # Treasury par yields (FRED, May 22 2026)
├── notebooks/                # Exploration and analysis
├── outputs/                  # Model checkpoints, results, plots
├── logs/                     # SLURM job logs
├── scripts/
│   ├── train_hazard.py           # Discrete hazard model training
│   ├── run_hazard.sbatch         # SLURM job for hazard training
│   ├── oas_engine.py             # Monte Carlo OAS cashflow engine
│   ├── run_oas.sbatch            # SLURM job for OAS
│   ├── oas_solver.py             # OAS spread solver (brentq)
│   ├── risk_neutral_rates.py     # Treasury bootstrap + drift correction
│   ├── run_risk_neutral.sbatch   # SLURM job for risk-neutral paths
│   ├── train_ddpm_conditional.py # Conditional DDPM rate simulation
│   ├── run_ddpm_conditional.sbatch
│   ├── shap_transformer.py       # SHAP interpretability
│   ├── train_ddpm.py             # Unconditional DDPM rate simulation
│   ├── synthetic_rate_paths.py   # Hull-White synthetic paths
│   └── segmentation_analysis.py  # Transformer vs XGBoost by loan bucket
├── prepare_sequences.py      # Data pipeline: raw CSV → padded sequences
├── train_transformer.py      # Binary classifier Transformer
├── train_zip3.py             # Static models with zip3 covariate
├── run_prepare.sbatch        # SLURM job for data prep
├── work_log.txt              # Hourly work log
└── README.md
```

---

## Data
**Source:** Fannie Mae Single-Family Loan Performance Dataset
**Portal:** https://loanperformancedata.fanniemae.com

### Vintages Used
| Vintage | Loans | Prepay Rate | Rate Environment | Prepayment Driver |
|---------|-------|-------------|-----------------|-------------------|
| 2020Q1 | 341,184 | 3.84% | ~3% (COVID low) | Refi-driven |
| 2020Q2 | 617,442 | 1.44% | ~3% (COVID low) | Refi-driven |
| 2020Q3 | 691,357 | 0.83% | ~3% (COVID low) | Refi-driven |
| 2020Q4 | 755,874 | 0.71% | ~3% (COVID low) | Refi-driven |
| 2021Q1 | 705,614 | 0.63% | ~3% (COVID low) | Refi-driven |
| 2021Q2 | 656,891 | 0.70% | ~3% (COVID low) | Refi-driven |
| 2021Q3 | 518,391 | 0.89% | ~3% (COVID low) | Refi-driven |
| 2021Q4 | 505,505 | 0.88% | ~3% (COVID low) | Refi-driven |
| 2023Q1 | 105,087 | 0.71% | ~6.5–7% (high) | Turnover-driven |
| **Total** | **4,897,345** | **1.06%** | | |

**Format:** Post-October 2020 single-file format, 109 fields, pipe-delimited (`|`), no header.

### Sequence Data
- Train: 3,917,876 loans × 33 months × 9 features
- Test: 979,432 loans × 33 months × 9 features
- Stored as padded numpy arrays with boolean mask (True = real timestep)

---

## Features
| Feature | Description |
|---------|-------------|
| `refi_incentive` | `original_interest_rate - pmms_rate_at_reporting_period` (time-varying) |
| `borrower_credit_score` | FICO score at origination |
| `original_ltv` | Loan-to-value ratio at origination |
| `current_ltv` | Dynamic LTV: `original_upb / (orig_home_value × zhvi_now/zhvi_orig) × 100` |
| `original_upb` | Original unpaid principal balance |
| `loan_age_months` | Age of loan in months |
| `dti` | Debt-to-income ratio at origination |
| `loan_purpose_enc` | Loan purpose: N=0 (purchase), Y=1 (refi/cash-out) |
| `property_type_enc` | Property type: P=0 (PUD), R=1 (row house), C=2 (condo) |

**External data:**
- **Freddie Mac PMMS** — monthly 30yr fixed rates, matched by reporting period
- **Zillow ZHVI** — zip3-level monthly home values for dynamic LTV
- **FRED Treasury yields** — bootstrapped zero-coupon curve for OAS discounting

---

## Results

### Phase 1 — Single Vintage 2023Q1 (~105K loans)
| Model | AUC |
|-------|-----|
| Logistic Regression | **0.7765** |
| Random Forest | 0.7743 |
| XGBoost | 0.7713 |
| MLP | 0.7706 |
| LightGBM | 0.7688 |

### Phase 2–4 — Multi-Vintage 2020–2023 with Time-Varying Features
| Model | AUC |
|-------|-----|
| XGBoost | 0.8306 |
| LightGBM | 0.8306 |
| MLP | 0.8289 |
| Random Forest | 0.8174 |
| Logistic Regression | 0.8074 |

### Phase 5 — Transformer (Full Sequence Model)
| Model | AUC |
|-------|-----|
| **Transformer** | **0.8431** |
| XGBoost | 0.8306 |
| LightGBM | 0.8306 |
| MLP | 0.8289 |
| Random Forest | 0.8174 |
| Logistic Regression | 0.8074 |

**Architecture:** input 6 → d_model 64, learnable positional embeddings (MAX_SEQ=33), 2-layer encoder 4 heads dim_ff=256, mean pooling with mask, 64→32→1 classifier.

### Phase 6 — zip3 as Raw Covariate
| Model | Baseline | +zip3 | Delta |
|-------|----------|-------|-------|
| XGBoost | 0.8306 | 0.8367 | +0.006 |
| LightGBM | 0.8306 | 0.8361 | +0.006 |
| MLP | 0.8289 | 0.8230 | -0.006 |
| Random Forest | 0.8174 | 0.8111 | -0.006 |
| Logistic Regression | 0.8074 | 0.8051 | ~0 |

### Phase 7 — Segmentation Analysis
Transformer wins every loan bucket. Largest gaps: oldest loans (+0.024 AUC), near-zero refi incentive (+0.018 AUC).

### Phase 8 — SHAP Interpretability
| Feature | Mean |SHAP| |
|---------|------|
| loan_age_months | 0.040 |
| borrower_credit_score | 0.031 |
| refi_incentive | 0.029 |
| original_ltv | 0.025 |
| current_ltv | 0.024 |
| original_upb | 0.007 |

Peak activation month 28. Burnout signal: negative loan_age SHAP at month 28.

### Phase 9 — Hazard Model (Discrete-Time Survival)
Retrained Transformer as discrete hazard model — predicts P(prepay at month t | survived to t).

**Test AUC: 0.8181** (9 features: +0.022 vs 6-feature model at 0.7958)

Key fixes: mask convention (True=real), 50% prepaid loan oversampling per batch, ReduceLROnPlateau.

### Phase 10 — DDPM Rate Simulation

**Unconditional DDPM:** Trained on full PMMS history (660 obs). 1000 paths × 34 months.

**Conditional DDPM:** Conditioned on starting rate level. Paths start at today's rate (6.18%) and evolve realistically. Architecture adds `start_rate` embedding to DenoiseNet alongside diffusion timestep.

### Phase 11 — Risk-Neutral Rate Paths
- Bootstrapped Treasury zero-coupon curve from FRED par yields (May 22, 2026)
- Drift correction: shift each month's mean to implied forward rate → ZCB repricing error < 2.6bp
- PMMS separated from Treasury: historical spread 1.89% over 10yr Treasury
- Two path sets: `treasury_cond_paths.npy` (discounting), `pmms_cond_paths.npy` (refi incentive)

### Phase 12 — OAS Cashflow Engine
Monte Carlo OAS pipeline:
1. For each conditional DDPM rate path, recompute per-loan refi incentive (PMMS path)
2. Run hazard model → per-month prepayment probabilities
3. Compute scheduled cashflows adjusted for prepayments + terminal value
4. Discount at Treasury path → price per path
5. Average across 1000 paths → model fair price

**Results:** Mean price 85.07%, **Median price 99.06%** of par ✅

OAS solver implemented (brentq root-finding). Awaiting Bloomberg/ICE market prices from Prof. Gupta.

---

## Infrastructure
**HPC:** NYU Torch (`login.torch.hpc.nyu.edu`)
**Working directory:** `/scratch/at7095/mortgage_prepayment/`
**Conda env:** `/scratch/at7095/conda_envs/mortgage_env`
**SLURM account:** `torch_pr_932_general`
**GitHub:** `tiwari-adarsh0830/mortgage-prepayment`

### Connect to Torch
```bash
ssh-keygen -R login.torch.hpc.nyu.edu
ssh at7095@login.torch.hpc.nyu.edu
# Note: submit without --partition flag
```

### Key Paths
| Path | Description |
|------|-------------|
| `/data/sequences/train_seq.npy` | Train sequences (3,917,876×33×9) |
| `/data/sequences/test_seq.npy` | Test sequences (979,432×33×9) |
| `/data/sequences/scaler.pkl` | StandardScaler for 9 features |
| `/data/sequences/train_prepay_timestep.npy` | Prepayment timestep per loan |
| `/outputs/transformer_best.pt` | Binary classifier checkpoint |
| `/outputs/hazard_best.pt` | Hazard model checkpoint (AUC 0.8181) |
| `/outputs/ddpm_conditional_paths.npy` | Conditional DDPM paths (1000×34) |
| `/outputs/treasury_cond_paths.npy` | Risk-neutral Treasury paths for discounting |
| `/outputs/pmms_cond_paths.npy` | Risk-neutral PMMS paths for refi incentive |
| `/outputs/monthly_zero_rates.npy` | Bootstrapped zero-coupon curve |
| `/outputs/oas_loan_prices.npy` | OAS model prices (1000 loans × 1000 paths) |
| `/outputs/oas_cashflows.npy` | Cashflow matrix (1000 loans × 1000 paths × 33 months) |

---

## Key Engineering Decisions & Bugs Fixed
| Issue | Fix |
|-------|-----|
| Data leakage in current_ltv | Use `original_upb` not `current_actual_upb` |
| Column misalignment | Dict-based col_map sorted by file position index |
| MLP double sigmoid | Remove sigmoid from final layer with BCEWithLogitsLoss |
| OOM on 400M+ rows | Split into CPU data prep + GPU training SLURM jobs |
| Mask convention | True=real throughout; invert inside forward() for transformer |
| Hazard class imbalance | Oversample prepaid loans 50% per batch; pos_weight=1.0 |
| SLURM partition rejection | Submit without --partition flag |
| PMMS used for discounting | Separate Treasury (discounting) from PMMS (refi incentive) |
| P-measure rate paths | Drift correction anchors paths to today's Treasury term structure |
| OAS price too low (57%) | Add terminal value (remaining balance at month 33) |

---

## References
1. Ho et al. — "Denoising Diffusion Probabilistic Models" — arxiv 2006.11239
2. arxiv 2011.13456 — DDPM variant
3. arxiv 2410.18897 — DDPM + wavelet for synthetic financial time series
4. arxiv 2511.17892 — HJM no-arbitrage neural yield curve (AER penalty)
5. Fuster et al. — "Predictably Unequal?" — SSRN 3072038
6. Higham et al. — "Diffusion Models for Applied Mathematicians" — arxiv 2312.14977
7. Valencia et al. — "Data-Efficient Ensemble Forecasting with Diffusion Models" — arxiv 2509.11047
