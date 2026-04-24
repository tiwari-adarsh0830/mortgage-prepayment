# Mortgage Prepayment Prediction
**NYU Stern — RA Project | Advisor: Prof. Gupta**

---

## Project Overview
Predicting mortgage prepayment using Fannie Mae Single-Family Loan Performance Data. The goal is to build a sequence of increasingly sophisticated models — from logistic regression through transformer-based architectures — and understand what drives prepayment behavior across different interest rate regimes.

---

## Repository Structure
```
mortgage_prepayment/
├── data/
│   └── raw/              # Raw Fannie Mae CSVs (not tracked in git)
├── notebooks/            # Exploration and analysis notebooks
├── outputs/              # Model results, plots, saved models
├── logs/                 # SLURM job logs
├── train.py              # Main training script
├── run_train.sbatch      # SLURM job script
└── README.md
```

---

## Data
**Source:** Fannie Mae Single-Family Loan Performance Dataset
**Portal:** https://loanperformancedata.fanniemae.com

### Vintages Used
| Vintage | Loans | Prepay Rate | Rate Environment | Prepayment Driver |
|---------|-------|-------------|-----------------|-------------------|
| 2020Q1 | 683,383 | 3.56% | ~3% (COVID low) | Refi-driven |
| 2020Q2 | 1,235,302 | 1.32% | ~3% (COVID low) | Refi-driven |
| 2020Q3 | 1,383,813 | 0.75% | ~3% (COVID low) | Refi-driven |
| 2020Q4 | 1,513,322 | 0.64% | ~3% (COVID low) | Refi-driven |
| 2021Q1 | 1,413,004 | 0.58% | ~3% (COVID low) | Refi-driven |
| 2021Q2 | 1,315,673 | 0.66% | ~3% (COVID low) | Refi-driven |
| 2021Q3 | 1,038,463 | 0.81% | ~3% (COVID low) | Refi-driven |
| 2021Q4 | 1,012,946 | 0.81% | ~3% (COVID low) | Refi-driven |
| 2023Q1 | 210,545 | 0.68% | ~6.5–7% (high) | Turnover-driven |
| **Total** | **9,806,451** | **0.97%** | | |

**Note:** Data is organized by origination quarter. Each file contains static loan characteristics at origination plus dynamic monthly performance data through the most recent available quarter.

**Format:** Post-October 2020 single-file format, 108–109 fields, pipe-delimited (`|`), no header.

### Key Design Decisions
- Time-varying PMMS rate used for refi incentive — monthly average 30yr fixed rate from Freddie Mac matched to each loan's last reporting period
- Target variable: `prepaid = 1` if `zero_balance_code == 1.0` (voluntary prepayment), else 0
- One row per loan (last observed monthly record used for outcome)
- `original_upb` used in current_ltv numerator (not `current_actual_upb`) to avoid data leakage — current UPB drops to zero for prepaid loans in their final record

---

## Features
| Feature | Description |
|---------|-------------|
| `refi_incentive` | `original_interest_rate - pmms_rate_at_reporting_period` (time-varying) |
| `borrower_credit_score` | FICO score at origination |
| `original_ltv` | Loan-to-value ratio at origination |
| `current_ltv` | Dynamic LTV using Zillow ZHVI appreciation: `original_upb / (original_home_value × zhvi_last/zhvi_orig) × 100` |
| `original_upb` | Original unpaid principal balance (loan size) |
| `loan_age_months` | Age of loan in months |

**External data sources:**
- **Freddie Mac PMMS** — weekly 30yr fixed rates → monthly averages → matched by reporting period
- **Zillow ZHVI** — zip-level monthly home values → aggregated to zip3 → matched by zip3 + period

---

## Results

### Phase 1 — 2023 Q1 Only (~210K loans, 0.68% prepay rate)
| Model | AUC |
|-------|-----|
| Logistic Regression | **0.7765** |
| Random Forest (tuned) | 0.7743 |
| XGBoost | 0.7713 |
| MLP Neural Network | 0.7706 |
| LightGBM | 0.7688 |

**Key insight:** All models converge to ~0.77 AUC. LR wins because 2023Q1 loans originated at 6–7% with current rates at ~6.8% — virtually no refinancing incentive exists. Almost all prepayments are turnover-driven (home sales), which is a linear, low-complexity signal. The nonlinearity that tree-based models capture simply isn't present in this rate regime.

**FICO sign flip:** Higher FICO borrowers prepay *less* in this dataset — counterintuitive at first, but explained by turnover dynamics. Lower FICO borrowers may be more financially stressed and more likely to sell.

### Phase 2 — Multi-Vintage 2020–2023 (9.8M loans, 0.97% overall prepay rate) ✅
| Model | AUC | vs Phase 1 |
|-------|-----|------------|
| **XGBoost** | **0.8297** | +0.0584 |
| LightGBM | 0.8295 | +0.0607 |
| MLP Neural Network | 0.8192 | +0.0486 |
| Random Forest | 0.8189 | +0.0446 |
| Logistic Regression | 0.7979 | +0.0214 |

**Key insight:** Hypothesis confirmed. Once 2020–2021 refi-driven prepayments are included, the model rankings flip — XGBoost and LightGBM now lead, while LR falls to last. The S-curve nonlinearity of refinancing behavior is now present in the data, and tree-based models capture it better.

**FICO sign corrected:** With refi behavior present, higher FICO borrowers now correctly show *higher* prepayment probability — they are more able to refinance when rates drop.

**loan_age_months flipped negative:** In the refi regime, newer loans prepay more — they originated at low rates and face little burnout. Older loans may have already refinanced (burnout effect).

### Phase 3 — Time-Varying PMMS Rate ✅
| Model | AUC | vs Phase 2 |
|-------|-----|------------|
| LightGBM | **0.8310** | +0.0015 |
| XGBoost | 0.8308 | +0.0011 |
| MLP Neural Network | 0.8208 | +0.0016 |
| Random Forest | 0.8190 | +0.0001 |
| Logistic Regression | 0.8077 | +0.0098 |

**Key insight:** Time-varying rate helped LR most (+0.0098) — refi incentive is now properly measured. LightGBM edges ahead of XGBoost.

### Phase 4 — Dynamic LTV via Zillow ZHVI ✅
| Model | AUC | vs Phase 3 |
|-------|-----|------------|
| XGBoost | **0.8306** | -0.0002 |
| LightGBM | 0.8306 | -0.0004 |
| MLP Neural Network | 0.8289 | +0.0081 |
| Random Forest | 0.8174 | -0.0016 |
| Logistic Regression | 0.8074 | -0.0003 |

**Key insight:** Dynamic LTV helped MLP most (+0.008) but had minimal impact on tree-based models — they already captured the collateral signal through original LTV and loan age. Current LTV has low feature importance in RF (0.04) and negative coefficient in LR — likely multicollinearity with original LTV.

**avg current_ltv by vintage:**
- 2020Q1: 59.2% — large appreciation, lots of equity built
- 2023Q1: 71.5% — less appreciation, closer to original LTV

---

## Infrastructure
**HPC:** NYU Torch (`login.torch.hpc.nyu.edu`)
**Working directory:** `/scratch/at7095/mortgage_prepayment/`
**Conda env:** `mortgage_env` at `/scratch/at7095/conda_envs/mortgage_env`

### Connect to Torch
```bash
ssh-keygen -R login.torch.hpc.nyu.edu
ssh -o KexAlgorithms=curve25519-sha256 at7095@login.torch.hpc.nyu.edu
```

### Submit Training Job
```bash
sbatch /scratch/at7095/mortgage_prepayment/run_train.sbatch
```

### Monitor Job
```bash
squeue -u at7095
tail -f /scratch/at7095/mortgage_prepayment/logs/train_<JOBID>.out
```

---

## Roadmap

### Completed
- [x] Baseline models on 2023Q1 (LR, RF, XGBoost, LightGBM, MLP) — AUC ~0.77
- [x] EDA — prepayment by refi incentive, FICO, LTV buckets
- [x] Set up NYU Torch HPC environment + conda env
- [x] Transfer 2020Q1–Q4, 2021Q1–Q4, 2023Q1 to Torch (9.8M loans)
- [x] Multi-vintage training pipeline with chunk-based memory-efficient loading
- [x] Multi-vintage results — XGBoost leads at 0.8297
- [x] Time-varying PMMS rate — Freddie Mac monthly rates merged by reporting period
- [x] Dynamic LTV — Zillow ZHVI merged at zip3 level
- [x] Fixed data leakage — use original_upb not current_actual_upb in current_ltv
- [x] Fixed MLP sigmoid + BCEWithLogitsLoss double application bug

### Next Steps
- [ ] Restructure data to keep full monthly sequence per loan (for Transformer)
- [ ] Implement PyTorch Transformer with location embedding at zip3 level
- [ ] Read Fuster et al. (SSRN 3072038) — ML fairness in credit markets
- [ ] Study Higham et al. (arxiv 2312.14977) — diffusion model math primer
- [ ] Implement diffusion model (arxiv 2509.11047)
- [ ] Add 2018–2019 vintages for additional rate regime coverage
- [ ] Switch from binary classification to survival/hazard model for CPR output
- [ ] Ensemble averaging across models

---

## References
1. Fuster, Goldsmith-Pinkham, Ramadorai, Walther — "Predictably Unequal? The Effects of Machine Learning on Credit Markets" — *Journal of Finance* (SSRN 3072038)
2. Higham et al. — "Diffusion Models for Generative AI: An Introduction for Applied Mathematicians" — arxiv 2312.14977
3. Valencia et al. — "Data-Efficient Ensemble Forecasting with Diffusion Models" — arxiv 2509.11047
4. Zillow Research Data — ZHVI (Home Value Index) — https://www.zillow.com/research/data/

---

## Notes & Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| Apr 19, 2026 | Use 2020–2021 vintages | COVID-era low rates (~3%) create refi-driven prepayment — best contrast with 2023Q1 turnover regime |
| Apr 19, 2026 | Work in /scratch on Torch | Home dir quota too small for multi-GB CSV files |
| Apr 19, 2026 | No Q1-only restriction | Fannie Mae data is by origination quarter, not reporting quarter — using all quarters of 2020–2021 |
| Apr 20, 2026 | Chunk-based CSV loading | Full file loading caused OOM even with usecols — chunked reading at 500K rows resolves this |
| Apr 20, 2026 | Binary classification target | Simplification for Phase 1–4; longer term should move to survival/hazard model for CPR output |
| Apr 23, 2026 | Time-varying PMMS rate | Gupta suggestion — monthly average 30yr fixed rate more accurate than fixed 6.8% |
| Apr 23, 2026 | ZHVI at zip3 level | Fannie Mae masks geography to 3-digit zip prefix — must aggregate Zillow zip-level data to zip3 |
| Apr 23, 2026 | Use original_upb in current_ltv | current_actual_upb = 0 for prepaid loans in final record — would leak the target |
| Apr 23, 2026 | Remove sigmoid from MLP | BCEWithLogitsLoss applies sigmoid internally — double application was a bug |
| Apr 23, 2026 | Dict-based usecols mapping | Pandas returns usecols in file order not specified order — positional col_names assignment caused misalignment |
