# Mortgage Prepayment Prediction
**NYU Stern — RA Project | Advisor: Prof. Arpit Gupta**

---

## Project Overview
Predicting mortgage prepayment using Fannie Mae Single-Family Loan Performance Data. The project builds a sequence of increasingly sophisticated models — from logistic regression through Transformer-based architectures — and is now extending toward Option-Adjusted Spread (OAS) pricing using AI-driven rate simulation and discrete hazard models.

---

## Repository Structure
mortgage_prepayment/
├── data/
│   ├── raw/                  # Raw Fannie Mae CSVs (not tracked in git)
│   ├── sequences/            # Preprocessed padded sequences (not tracked)
│   ├── pmms_monthly.csv      # Freddie Mac PMMS 30yr rates
│   ├── zhvi_zip3.csv         # Zillow ZHVI at zip3 level
│   └── treasury_yields.csv   # Treasury zero-coupon curve (pending)
├── notebooks/                # Exploration and analysis
├── outputs/                  # Model checkpoints, results, plots
├── logs/                     # SLURM job logs
├── scripts/
│   ├── train_hazard.py       # Discrete hazard model training
│   ├── run_hazard.sbatch     # SLURM job for hazard training
│   ├── oas_engine.py         # Monte Carlo OAS cashflow engine
│   ├── run_oas.sbatch        # SLURM job for OAS
│   ├── shap_transformer.py   # SHAP interpretability
│   ├── train_ddpm.py         # DDPM rate path simulation
│   ├── synthetic_rate_paths.py # Hull-White synthetic paths
│   └── segmentation_analysis.py # Transformer vs XGBoost by loan bucket
├── prepare_sequences.py      # Data pipeline: raw CSV → padded sequences
├── train_transformer.py      # Binary classifier Transformer
├── train_zip3.py             # Static models with zip3 covariate
├── run_prepare.sbatch        # SLURM job for data prep
├── work_log.txt              # Hourly work log
└── README.md

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
- Train: 3,917,876 loans × 33 months × 6 features
- Test: 979,469 loans × 33 months × 6 features
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

**External data:**
- **Freddie Mac PMMS** — monthly 30yr fixed rates, matched by reporting period
- **Zillow ZHVI** — zip3-level monthly home values for dynamic LTV

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

All models converge ~0.77. Turnover-driven regime has low nonlinearity — LR wins.

### Phase 2–4 — Multi-Vintage 2020–2023 with Time-Varying Features
| Model | AUC |
|-------|-----|
| XGBoost | 0.8306 |
| LightGBM | 0.8306 |
| MLP | 0.8289 |
| Random Forest | 0.8174 |
| Logistic Regression | 0.8074 |

Tree models take the lead as refi nonlinearity enters the data.

### Phase 5 — Transformer (Full Sequence Model)
| Model | AUC |
|-------|-----|
| **Transformer** | **0.8431** |
| XGBoost | 0.8306 |
| LightGBM | 0.8306 |
| MLP | 0.8289 |
| Random Forest | 0.8174 |
| Logistic Regression | 0.8074 |

Transformer learns path dependence across 33-month sequences. Best overall model.

**Architecture:** input 6 → d_model 64, learnable positional embeddings (MAX_SEQ=33), 2-layer encoder 4 heads dim_ff=256, mean pooling with mask, 64→32→1 classifier. BCEWithLogitsLoss + pos_weight. Adam lr=1e-3, StepLR step=5 gamma=0.5, grad clip 1.0.

### Phase 6 — zip3 as Raw Covariate (Static Models)
| Model | Baseline | + zip3 | Delta |
|-------|----------|--------|-------|
| XGBoost | 0.8306 | 0.8367 | +0.006 |
| LightGBM | 0.8306 | 0.8361 | +0.006 |
| MLP | 0.8289 | 0.8230 | -0.006 |
| Random Forest | 0.8174 | 0.8111 | -0.006 |
| Logistic Regression | 0.8074 | 0.8051 | ~0 |

Tree models benefit; linear/neural models do not. Geographic signal is real but nonlinear — motivates zip3 embeddings.

### Phase 7 — Segmentation Analysis (Transformer vs XGBoost)
Transformer wins every loan attribute bucket. Largest gaps:
- Oldest loans (loan_age > 24mo): +0.024
- Near-zero refi incentive: +0.018

### Phase 8 — SHAP Interpretability
| Feature | Mean |SHAP| |
|---------|------|
| loan_age_months | 0.040 |
| borrower_credit_score | 0.031 |
| refi_incentive | 0.029 |
| original_ltv | 0.025 |
| current_ltv | 0.024 |
| original_upb | 0.007 |

Peak activation at month 28. Negative loan_age SHAP at month 28 = burnout signal. Discrete activation spikes at months 3, 16, 28.

### Phase 9 — Hazard Model (Discrete-Time Survival)
Retrained Transformer as discrete hazard model — predicts P(prepay at month t | survived to t).

**Test AUC: 0.7958**

Key fixes: mask convention (True=real), oversampling prepaid loans 50% per batch (only 0.036% of loan-month pairs are positive events), ReduceLROnPlateau scheduler.

### Phase 10 — DDPM Rate Simulation
Trained DDPM on full PMMS history (660 monthly observations) to generate synthetic rate paths. Outputs: 1000 paths × 34 months, rates 0.5–14%, mean ~6.8%.

### Phase 11 — OAS Cashflow Engine (In Progress)
Monte Carlo OAS pipeline:
1. For each DDPM rate path, recompute refi incentive per loan per month
2. Run hazard model → per-month prepayment probabilities
3. Compute scheduled cashflows adjusted for prepayments
4. Discount back at path rates → price per path
5. Average across 1000 paths → model fair price

**Current limitation:** PMMS embeds g-fee and primary-secondary spread — need separate risk-free curve (Treasury/SOFR) for discounting. DDPM paths are real-world (P) measure — need risk-neutral (Q) paths anchored to today's term structure.

**Next steps:**
- Pull Treasury zero-coupon curve from FRED
- Implement risk-neutral drift correction or conditional DDPM generation
- Plug in Bloomberg/ICE market prices from Gupta → compute OAS spread

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
| `/data/raw/2020Q1.csv` | Raw Fannie Mae (pipe-delimited, no header, 109 cols) |
| `/data/sequences/train_seq.npy` | Train sequences (3,917,876×33×6) |
| `/data/sequences/test_seq.npy` | Test sequences (979,469×33×6) |
| `/data/sequences/train_mask.npy` | Boolean mask True=real |
| `/data/sequences/train_prepay_timestep.npy` | Prepayment timestep per loan (-1 if none) |
| `/data/sequences/scaler.pkl` | StandardScaler for 6 features |
| `/outputs/transformer_best.pt` | Binary classifier checkpoint |
| `/outputs/hazard_best.pt` | Hazard model checkpoint |
| `/outputs/ddpm_rate_paths.npy` | 1000 DDPM rate paths (1000×34) |
| `/outputs/oas_loan_prices.npy` | OAS model prices (1000 loans × 1000 paths) |

---

## Key Engineering Decisions & Bugs Fixed
| Issue | Fix |
|-------|-----|
| Data leakage in current_ltv | Use `original_upb` not `current_actual_upb` |
| Column misalignment | Dict-based col_map sorted by file position index |
| MLP double sigmoid | Remove sigmoid from final layer with BCEWithLogitsLoss |
| OOM on 400M+ rows | Split into CPU data prep + GPU training SLURM jobs |
| Slow ZHVI merge | Vectorized pandas join instead of row-by-row apply |
| Mask convention | True=real throughout; invert inside forward() for transformer encoder |
| Hazard class imbalance | Oversample prepaid loans 50% per batch; pos_weight=1.0 |
| SLURM partition rejection | Submit without --partition flag |

---

## References
1. Ho et al. — "Denoising Diffusion Probabilistic Models" — arxiv 2006.11239
2. arxiv 2011.13456 — DDPM variant
3. arxiv 2410.18897 — DDPM + wavelet for synthetic financial time series
4. arxiv 2511.17892 — HJM no-arbitrage neural yield curve (AER penalty)
5. Fuster et al. — "Predictably Unequal?" — SSRN 3072038
6. Higham et al. — "Diffusion Models for Applied Mathematicians" — arxiv 2312.14977
7. Valencia et al. — "Data-Efficient Ensemble Forecasting with Diffusion Models" — arxiv 2509.11047
