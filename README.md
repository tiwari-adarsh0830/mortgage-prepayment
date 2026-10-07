## Current state
*(Oct 6, 2026 — rewritten each session, not appended to.)*

**Oct 5–6 update (new schema, `_seq`).** Everything below the next paragraph describes the Oct 1–3
`_30y` builds and is unchanged. Since then the pipeline moved to the `_seq` schema: `harp_eligible`
is a tenth feature column (all zeros — no eligibility logic yet), the 30-year rule is applied to the
loan's origination (earliest) row, and `scripts/slurm/submit_cutoff_chain.sh <year>` submits the whole
census → build → gate → smoke → 10 seeds → ensemble chain for one cutoff. `cutoff_2002_seq` is
done: 10-seed ensemble one-step pooled ratio **0.8798** (count) / **0.8573** (UPB), vs. 0.8982 /
0.8818 for the five-seed `_30y` ensemble (source files in the Oct 5–6 section's number audit). The
census is byte-identical to `_30y`. Cutoffs 2003, 2004, 2005 are submitted and still running as of
this writing (job ids in the Oct 5–6 section). The `_30y` numbers below remain the record for the
five-seed runs.

**Advisor's Oct 1 clarification.** "Rebuild the 2021 cutoff" means the existing `cutoff_2020`
design (train ≤ Dec 2020, forecast CY2021) — not a new train-≤-Dec-2021 cutoff. No new
`cutoff_2021` build is needed; `cutoff_2020` rebuilt under the same 30-year/post-mod rules as
`cutoff_2002` *is* the "2021" rebuild, and it's now done (below).

**Valid builds, both cutoffs rebuilt 30-year/post-mod.**
- `cutoff_2002` `_30y`: `data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y` —
  277,042 loans through the loader, 269,536 in train+test, 833,720/208,187 train/test obs. Five
  checkpoints `outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s{42,7,123,1001,2026}/`,
  AUC 0.7744–0.7759.
- `cutoff_2020` `_30y`: `data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_30y` —
  1,412,264 loans through the loader, 1,378,753 in train+test, 8,247,014/2,064,977 train/test obs.
  Five checkpoints `outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_cutoff_2020_30y_s{42,7,123,1001,2026}/`,
  AUC 0.7245–0.7268.

Both: **use `hazard_best.pt`, not `hazard_final.pt`** (see mistakes log). Both gates
(`check_build_30y.py`, `check_build_2020_30y.py`) and both census checks (`census_check_2002_30y.py`,
`census_check_2020_30y.py`) pass. Standing consistency test passes on both (below).

**Superseded: both cutoffs' pre-rebuild checkpoint sets.** `cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist{,_seed7,_seed123,_seed1001,_seed2026}`
and `cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33{,_seed7,_seed123,_seed1001_goldenbackup,_seed2026_goldenbackup}`
predate the 30-year filter and post-mod drop — non-360 loans included, no post-mod drop. Kept only
for the `cutoff_2002` matched-intersection model-vs-population comparison (below); not used for any
new forecast number.

**Pending, not started:** the two-realized-series split (voluntary code-01-30y vs. total payoffs,
for `realized_cpr_v6.py`); the HARP eligibility *logic* (the column now exists, all zeros, since Oct 5 —
see the Oct 5–6 section; the real eligibility rule is not written); the recency-weighting form; the advisor reply
email.

**Pending, added Oct 6.** (1) `outputs/hazard_best.pt`, `outputs/hazard_calibration.json` and
`outputs/transformer_best.pt` (dated April–June, from the old stage2/OAS pipeline; no current
writer) must be replaced before the DER regression is rerun with the new models. (2)
`docs/artifact_map.md` is an untracked provenance audit awaiting review.

**Standing test.** `scripts/tests/test_train_forecast_consistency.py` — both `_30y` cases now pass:
`cutoff_2002_seed42_30y` (Oct 2, Check 1 exact n=2,000, Check 2 verified at 3 months, negative
control (b) caught the frozen-window bug — control (a) structurally inapplicable, `TRAIL_SEQ_DIR`
is `cutoff_2020`-only) and `cutoff_2020_seed42_30y` (Oct 3, same Check 1/2 pattern, **both**
negative controls fire here since `TRAIL_SEQ_DIR` is a `cutoff_2020` build).

**Settled.** The Sep 29 "76.8% of the shortfall" and "2021... 37.3% of the shortfall" findings are
decompositions of the **all-term** models' shortfalls (`cutoff_2002` 360-pooled 0.9562/non-360
0.7049; `cutoff_2020` control 360-pooled 0.8990/non-360 0.7912) — both remain true as statements
about those model/population pairs. Neither is the 30-year-only models' own number: `_30y` pooled
ratios are **0.8982** (`cutoff_2002`, 323,203 loan-months) and **0.8441** (`cutoff_2020`,
1,723,631 loan-months) — both below their respective all-term 360-only subset figures. The 30-year
restriction fixed the incentive-measurement mismatch (15-year loans scored against the 30-year
PMMS) but did not close the predicted-vs-realized gap for either cutoff — see the Oct 2-3 section's
"what the two rebuilt tests say together."

**Open:** `cutoff_2011`/`cutoff_2019` not built; the matched-intersection model-vs-population
result for `cutoff_2002` (old ckpt 0.9723 vs new ckpt 0.9341 on a 4,911-loan held-out-for-both
population) runs opposite the full-population comparison (0.8734 old vs 0.8982 new), within a
single seed's typical noise band — not reconciled, reported as-is; no equivalent
matched-intersection decomposition run for `cutoff_2020` (not needed per this session's
instruction). Frozen-Dec-window 12-month forecast, superseded design, reported for continuity:
`cutoff_2002` 0.7137, `cutoff_2020` 1.0303; not comparable to the one-step numbers (0.8982, 0.8441)
and not an in-sample check.

**Next.** The two-realized-series split, HARP eligibility feature, recency weighting, advisor
reply email.

---

# Mortgage Prepayment Prediction
**NYU Stern — RA Project**

---

## Project Overview
Predicting mortgage prepayment using Fannie Mae Single-Family Loan Performance Data. The project builds a sequence of increasingly sophisticated models — from logistic regression through Transformer-based architectures — and is now applying the Diep-Eisfeldt-Richardson (DER) framework to explain the cross-section of TBA MBS returns using hazard-model-implied prepayment risk loadings.

**Contribution angle:** The DER framework uses Bloomberg dealer survey forecasts as the prepayment forecast leg. We substitute our ML hazard model as the forecast, removing dependence on proprietary survey data.

---

## Standing tests

`scripts/tests/test_train_forecast_consistency.py` checks that the training path and the
forecasting path produce identical sequences and predictions for the same (loan_id, ref_month)
observations, and that a multi-month forecast uses the window ending at each forecast month (not
an earlier, frozen one). It is run after any change to data prep, the sequence builder, feature
definitions, category maps, scalers, or any scoring/forecast script, and before any forecast
number is reported here or elsewhere.

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
>
> **This note describes `prepare_sequences.py` only.** The multiobs builder
> (`prepare_sequences_multiobs_zbc.py`) encodes both features live, with the
> CP/U fix (commit 2a5b283, 2026-09-19) — they are not inert there. The
> `cutoff_2020` control checkpoints (seeds 42/7/123, plus the retrained
> 1001/2026) were trained before that fix and use the pre-fix category maps
> (`--map_era prefix`); only `cutoff_2002`'s frozen hist build uses the
> post-fix maps (`--map_era fixed`).

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
| GPU training silently non-deterministic despite seeded RNG | `torch.manual_seed`/`cudnn.deterministic`/`use_deterministic_algorithms` alone are not sufficient — must also `export CUBLAS_WORKSPACE_CONFIG=:4096:8` in the sbatch environment before the job starts, or cuBLAS algorithm selection stays nondeterministic; seedcheck round 1 without it (jobs 17128745/17128746) differed, round 2 with it (jobs 17140541/17140542) was bit-identical |

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

**CORRECTION (Aug 29, 2026):** the field 107 = zero_balance_code identification
below is wrong. Field position 44 (usecols 43), code 01 = Prepaid, is the
correct column — confirmed by direct read across 2000Q1, 2012Q4, and 2018Q1.
See "Label column defect (Aug 29, 2026)" below.

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

## Multi-observation sampling (Sep 3, 2026)

**Why this was built.** The advisor's Sep 2 note raised burnout/survivor selection as a gap in
every builder so far: each of them emits exactly ONE observation per loan (its final age at the
cutoff), so the data can never separate "high refi incentive, about to prepay" from "high
incentive and has already declined it repeatedly" — those two loan-months are indistinguishable
when a loan is only ever observed once. `prepare_sequences_multiobs_zbc.py` is a copy of the
trailing builder that instead samples each loan at several `(loan_id, ref_month)` ages, so the
same loan can appear both before and after burnout sets in. It does not re-merge the trailing
builder's `build_sequences()` — the windowing semantics differ and must not be collapsed together.

**Eligibility rule — `t < term_t` is load-bearing.** For within-loan row index `t`, the window is
eligible only if `t >= min_hist - 1`, `t <= L - 1 - H`, and `t < term_t` (the index of the loan's
first `zbc==01` row for prepaid loans, else `L-1` for censored ones). The third condition is not
a refinement, it is required: without it, a reference month can land ON the payoff row itself,
which becomes the last timestep of the feature window while the forward label reads 0 because
there is nothing left to look ahead to — the event enters `X` and vanishes from `y` at the same
draw. This was found and fixed via synthetic testing, not by inspection; the synthetic suite
(`scripts/diag/test_multiobs_sampler.py`, 39 checks) includes `last_row_event`, a loan whose
event is the final row, specifically to keep this defect from recurring.

**Mandatory draw at `term_t - H` — this is our interpretation, not the advisor's literal words.**
The Sep 2 note asked for multiple observations per loan to expose burnout; it did not specify a
sampling scheme. The design implemented here fixes `k` draws per loan (default 5, length-bias
fix: a 120-month loan and an 8-month loan contribute the same observation count, not a rate), with
one MANDATORY draw at `t = term_t - H` (the last eligible pre-event month, never at `term_t`
itself — that is the degenerate/leaking case above) and the remaining `k-1` slots filled by
deterministic blake2b-hash selection (not `rng.choice`, for reproducibility across processes and
monotone growth in `k`). Uniform and incentive-stratified draw schemes are both implemented. This
particular scheme — one guaranteed terminal draw plus hash-sampled fill — is a design choice made
to satisfy the burnout goal, not something dictated by the email; flag it for review rather than
treating it as pre-approved.

**Known limitation — calendar-gap filtering.** `row_idx` is a POSITION, not a calendar index:
upstream `dropna(subset=FEATURE_COLS)` can remove an interior row, after which row_idx `i` and
`i+1` for that loan are no longer 1 calendar month apart even though they are still adjacent
positions. `select_observations` drops two kinds of candidates for this reason and prints both
counts per call: (1) feature window spans a calendar gap — the drawn `(s, t]` window would
silently splice two non-adjacent calendar months together; (2) label-window row/calendar distance
mismatch — for a real prepay, the calendar distance from `ref_month` to the event month must equal
`term_t - row_idx`, or `H` means a different number of calendar months for this loan than for
every other one. Reference point (2015Q1, cutoff_year=2020, k=5, H=1, uniform), from the
single-vintage diagnostic (job 16903515): out of 2,166,647 candidate observations from 435,443
loans, 18 (10 loans) hit the window-gap filter and 4 (of 435,443 terminal draws) hit the
label-mismatch filter — both rare but real; do not remove either filter assuming the gap can't
occur.

**cutoff_2020 build result (job 16906766, 1h28m, reused split/scaler from `cutoff_2020_zbc_trail`).**
Train: 6,861,377 observations, 8.15 GB, 1,460,803 unique loans (97.60% of the 1,496,727-loan
reused train split — the rest yielded zero eligible observations). Test: 1,715,642 observations,
2.04 GB, 365,146 unique loans (97.59% of 374,182). Sampled label rate 8.33% on both splits.
Reconciled exactly against the loan-level population: restricting the trailing builder's
per-loan ever-prepay ground truth to just the loans that survived into this build gives a 39.13%
ever-prepay rate (higher than the full split's 38.19%, since dropped loans skew censored), and the
mean realized draws per loan is 4.697, not 5 (short/pool-limited loans get fewer than `k` slots).
39.13% / 4.697 = 8.330%, matching the observed rate to three decimals on both splits. Every
positive observation is a terminal draw and every loan has at most one positive observation —
checked directly, not assumed — so the 8.33%-vs-naive-7.64% (38.19%/5) gap is fully explained by
population and filtering effects, not a sampling defect.

**Open item — trainer incompatibility, no model trained.** `train_hazard_rolling.py`'s
`HazardSampler` (lines 78-95) is incompatible with this output: its random per-epoch truncation
and 50/50 oversampling both assume the trailing builder's one-observation-per-loan shape and are
the wrong operations on multi-observation data. It needs its own trainer, not a flag on the
existing one. That trainer has not been built. **No model has been trained on this data** — this
section covers data preparation and validation only.

Artifacts: `scripts/prepare_sequences_multiobs_zbc.py`, `scripts/diag/test_multiobs_sampler.py`,
`scripts/diag/diag_multiobs_onevintage.py`, `scripts/run_diag_multiobs_onevintage.sbatch`,
`scripts/slurm/run_multiobs_2020.sbatch`. Jobs 16903515 (one-vintage diagnostic), 16906766
(full cutoff_2020 build). Output under
`data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1/`.

## Multi-observation trainer, first training run, and age-stratified diagnostics (Sep 4-5, 2026)

**Why this was built.** The section above left the multiobs sequences from Sep 3 with no
compatible trainer: `train_hazard_rolling.py`'s `HazardSampler` assumes the trailing builder's
one-observation-per-loan, open-ended-sequence shape (random per-epoch truncation, 50/50
prepaid/non-prepaid oversampling), both wrong for fixed windows that already have one resolved
label each. `train_hazard_multiobs.py` is a new trainer, not a copy-edit of the rolling one —
its scoring logic is deliberately different, documented loudly in its own header so it doesn't get
merged back in by mistake. It uses the masked-mean-pool + classifier forward pass
(`model(seq, mask)`, no `return_per_timestep`) and scores with `sigmoid(logit)` directly — no
survival-CDF aggregation, since each observation is a single fixed window with one forward label
already resolved by the builder, not an open-ended sequence. `ObservationSampler` replaces
`HazardSampler`: `--pos_ratio` (default `None`, uniform draw at the natural ~8.33% rate) and
`--use_ipw` (default off, would weight the loss by the builder's emitted `incl_prob`) are both
explicit CLI flags, neither applied silently. `train_prepay_timestep`/`test_prepay_timestep` are
never loaded, matching the builder's own docstring that calls them meaningless for this output.

**Validated before the real run.** A synthetic smoke test
(`scripts/diag/test_train_hazard_multiobs_smoke.py`, no cluster data) exercises four configs —
default, `pos_ratio=0.5`, `use_ipw`, and both combined — confirming the loop runs end to end, loss
decreases, and AUC is always finite and in `[0, 1]` before spending GPU time on the real data.

**Real run (job 16964657, natural sampling, no `--pos_ratio`/`--use_ipw`).** 50 epochs, ~2h wall
clock, best AUC 0.7847 at epoch 50 (`outputs/rolling/cutoff_2020_multiobs_k5_h1/`). Flagged before
trusting it: epoch 1 already reached 0.7790 — nearly all of the final discrimination was present
immediately, which is consistent with a model that mostly learned something coarse and easy (e.g.
age or terminal-draw structure) rather than a genuinely incentive-conditional hazard. This was
treated as a red flag to check, not dismissed.

**Age-stratified AUC diagnostic (job 17005892, `scripts/diag/diag_age_stratified_auc.py`)
rules out the coarse explanation.** Test observations were split into 5 `age_at_ref` quintiles and
AUC was recomputed within each bin separately, holding age (and therefore most of the
terminal/censored-draw structure) fixed:

| bin | age range | n | n_pos | pos rate | AUC |
|---|---|---|---|---|---|
| 0 | [0, 3] | 411,707 | 3,616 | 0.88% | 0.7129 |
| 1 | [4, 8] | 304,422 | 15,289 | 5.02% | 0.7009 |
| 2 | [9, 18] | 337,457 | 30,995 | 9.18% | 0.6840 |
| 3 | [19, 36] | 333,100 | 43,155 | 12.96% | 0.6798 |
| 4 | [37, 109] | 328,956 | 49,834 | 15.15% | 0.7318 |

Within-bin AUC ranges 0.6798–0.7318 (mean 0.7019), every bin well above chance. Age alone does not
manufacture the overall 0.7847 — real discrimination survives once age is held roughly fixed,
which weighs against the epoch-1 red flag being pure age/terminal-detection.

**A burnout-shaped signature, found in the oldest bin, not proven.** Within bin 4 (age 37–109),
`spearman(score, incentive_at_ref)` over all observations is -0.0730, but restricted to label=1
(terminal draws) it is +0.5022 — opposite signs in the same bin. A follow-up split (job 17006097,
`scripts/diag/diag_bin4_incentive_split.py`) isolated label=0 (still-alive) loans: among survivors,
`spearman(score, incentive)` = -0.1738 (p≈0, n=279,122) — higher-incentive survivors score LOWER.
Splitting further, label=0 loans with `incentive_at_ref` above the bin median ("should have
refinanced but didn't", 49.40% of label=0 in this bin) have mean score 0.1373 versus 0.1485 for the
rest of label=0 — lower score despite higher incentive, same direction, modest magnitude. This is
consistent with the model distinguishing burnout survivors (high incentive, already passed on it
repeatedly) from fresh high-incentive loans, which is exactly what the multiobs sampling design was
built to make learnable. It is flagged as **consistent-with, not proven** — not checked against
vintage or credit-score confounds that could produce the same correlation pattern for an unrelated
reason.

Artifacts: `scripts/train_hazard_multiobs.py`, `scripts/diag/test_train_hazard_multiobs_smoke.py`,
`scripts/slurm/run_train_multiobs_2020.sbatch`, `scripts/diag/diag_age_stratified_auc.py`,
`scripts/diag/diag_bin4_incentive_split.py`, `scripts/slurm/run_diag_age_stratified_auc.sbatch`,
`scripts/slurm/run_diag_bin4_incentive_split.sbatch`. Jobs 16964657 (training), 17005892
(age-stratified AUC), 17006097 (bin4 incentive split). Output under
`outputs/rolling/cutoff_2020_multiobs_k5_h1/`.

## Coupon-level CPR calibration: multiobs vs. origination vs. trailing, matched population (Sep 5, 2026)

**CORRECTION (Sep 25, 2026):** these forecast numbers came from scoring right-aligned multiobs
checkpoints on the left-aligned `TRAIL_SEQ_DIR` test set. For `_seedcheck_a`, the like-for-like
corrected values (the left-aligned reproduction matched the original exactly): epoch 4 ratio range
1.139..0.717 → 1.047..0.544, pooled 0.9091 → 0.9736; epoch 50 1.662..0.684 → 1.161..0.664, pooled
1.0233 → 1.0794; Jan 2021 un-annualized at coupons 4.5 / 5.0: 0.803 / 0.746 → 1.027 / 0.982. The Sep
5 non-IPW `k5_h1` numbers below are affected but were not rescored. The 1.79x/3.21x origination-model
benchmark elsewhere in this README is unaffected (`infer_test_set` uses the last real timestep). See
the "Fixed-fraction sampling rates (a_eff / b_eff), a sequence-alignment bug, and the cutoff_2002 2003
forward test" section (Sep 24-25, 2026) near the end of this file for the full measurement.

**Why this was run.** With a trained multiobs model (0.7847 AUC) in hand, the natural next question
was how its coupon-level CPR forecast compares to the already-documented origination (pooled ratio
0.8631) and trailing (1.1273) runs from the Aug 30 / Sep 1 sections above.

**Scoring the multiobs model required the TRAILING test set, not its own.** `aggregate()`
(`forecast_rolling_cpr.py`, imported unchanged) needs exactly one forward-looking window per loan
ending at the Dec 2020 cutoff; the multiobs test set has multiple sampled `ref_month`s per loan by
design and is not usable here. `scripts/forecast_matched_population_cpr.py` instead scores the
multiobs-trained model against `data/sequences_rolling/cutoff_2020_zbc_trail/`'s test sequences,
calling `model(seq, mask)` with **no** `return_per_timestep` — mirroring `train_hazard_multiobs.py`'s
own `evaluate()`, since that checkpoint's classifier was never trained on a per-timestep token and
`forecast_rolling_cpr.py`'s `infer_test_set()` (`return_per_timestep=True`, last-real-timestep
gather) would silently score it with numbers it was never trained to produce.

**Population mismatch found and matched by intersection.** Multiobs's own test set (365,146 loans)
is a strict 9,036-loan (2.4%) subset of origination/trailing's shared 374,182-loan test set — all
three share one `--reuse_from` split, but `select_observations()` can legitimately emit zero
sampled rows for a split-assigned loan with too little post-dropna history, dropping it from the
multiobs test set while origination/trailing (which need no such eligibility) keep it. All three
pooled/dispersion numbers below are recomputed on the 365,146-loan intersection so the comparison
isolates model differences, not population differences; origination/trailing's own pooled ratios
shift slightly on this matched population (0.8631→0.8735, 1.1273→1.1409) versus their full-population
values.

**`pooled_comparison()` — first committed, reusable version of math that only existed ad hoc
before.** `dispersion` = max(ratio)..min(ratio) across coupons 2.0–5.0 with n≥5,000, where
`ratio = forecast_cpr / realized_cpr`; `pooled_ratio` = n_loans-weighted `pooled_forecast /
pooled_realized` (re-pooling the underlying loan population, NOT a mean of the per-coupon ratios —
verified these give different numbers). Self-checked against the known origination (0.8631) and
trailing (1.1273) values before trusting it on the new multiobs output; both reproduced exactly.

**Base-rate correction gap — found, not resolved.** Origination and trailing both apply
`prior_shift_offset()`, a logit shift undoing `HazardSampler`'s known 50/50 prepaid/non-prepaid
draw. Multiobs has no analogous correction (this checkpoint trained with `--use_ipw=False`), and a
flat scalar shift is not theoretically justified for its sampling design: the mandatory/terminal
draw is always included when eligible (`incl_prob=1.0` by construction, not sampled at any rate),
while the remaining pool draws' `incl_prob` varies per observation by loan length/stratum size — a
single King-Zeng-style intercept cannot represent a per-observation inclusion probability. A
justified fix would need either an `--use_ipw` retrain or per-observation `incl_prob` reweighting at
inference; neither has been built. Multiobs's pooled ratio (2.2250) is reported but marked
explicitly NOT comparable to origination/trailing's.

**Mean monthly h_t (pre-annualization), matched population, per coupon:**

| coupon | realized | orig h_t | orig CPR | trail h_t | trail CPR | multi h_t | multi CPR |
|---|---|---|---|---|---|---|---|
| 2.0 | 12.20 | 0.0298 | 25.82 | 0.2249 | 44.25 | 0.3327 | 84.03 |
| 2.5 | 15.81 | 0.0283 | 25.22 | 0.1874 | 43.47 | 0.2744 | 79.66 |
| 3.0 | 26.36 | 0.0210 | 20.37 | 0.1016 | 31.62 | 0.1701 | 60.70 |
| 3.5 | 33.88 | 0.0211 | 21.15 | 0.0364 | 21.11 | 0.1145 | 49.69 |
| 4.0 | 35.17 | 0.0258 | 25.52 | 0.0261 | 19.68 | 0.0794 | 44.11 |
| 4.5 | 35.36 | 0.0336 | 32.77 | 0.0438 | 34.24 | 0.0983 | 54.81 |
| 5.0 | 36.02 | 0.0363 | 35.44 | 0.0551 | 44.83 | 0.0951 | 59.26 |

Origination's monthly hazard stays in the near-linear regime (0.021–0.036) across the whole range.
Trailing saturates hard at coupons 2.0–3.0 (0.10–0.22) but is near-linear from 3.5 up. Multiobs is
at or above 0.079 at **every** coupon in range and above 0.10 at four of the seven — essentially the
whole reported range sits inside or at the edge of `1-(1-h)^12`'s saturating region, worse than
either of the other two models at every coupon.

**Conclusion — directional finding stands, magnitude does not.** The inversion (forecast CPR
falling as coupon rises 2.0→4.0 while realized CPR rises 12.2%→35.2%) is present in multiobs's raw
monthly hazard, not only the annualized number — a defensible qualitative finding that multiobs
reproduces the same burnout-shaped miscalibration direction already documented for the trailing
model. The magnitude (dispersion 6.885..1.254, pooled ratio 2.2250) is NOT reported as a
calibration result: it is generated inside a coupon range where the uncorrected base rate is
already deep in the annualization's saturating region, so it cannot be separated from a genuine
burnout signal without the correction described above.

Artifacts: `scripts/forecast_matched_population_cpr.py`,
`scripts/slurm/run_forecast_matched_population_cpr.sbatch`. Jobs 17009376 (matched-population
run), 17010241 (rerun adding the monthly h_t table). Output:
`outputs/rolling/cutoff_2020_{zbc,zbc_trail,multiobs_k5_h1}/rolling_cpr_forecast_matched.csv`
(new files; the existing unrestricted CSVs are untouched).

## IPW correction and reproducibility (Sep 6-7, 2026)

**CORRECTION (Sep 25, 2026):** these forecast numbers came from scoring right-aligned multiobs
checkpoints on the left-aligned `TRAIL_SEQ_DIR` test set. For `_seedcheck_a`, the like-for-like
corrected values (the left-aligned reproduction matched the original exactly): epoch 4 ratio range
1.139..0.717 → 1.047..0.544, pooled 0.9091 → 0.9736; epoch 50 1.662..0.684 → 1.161..0.664, pooled
1.0233 → 1.0794; Jan 2021 un-annualized at coupons 4.5 / 5.0: 0.803 / 0.746 → 1.027 / 0.982. The Sep
5 non-IPW `k5_h1` numbers (earlier in this file) are affected but were not rescored. The 1.79x/3.21x origination-model
benchmark elsewhere in this README is unaffected (`infer_test_set` uses the last real timestep). See
the "Fixed-fraction sampling rates (a_eff / b_eff), a sequence-alignment bug, and the cutoff_2002 2003
forward test" section (Sep 24-25, 2026) near the end of this file for the full measurement.

**Inference-time IPW pooling was abandoned as the wrong estimand, not shelved as untested.**
The first attempt combined a loan's several multiobs-sampled `ref_month` observations into one
per-loan hazard via Horvitz-Thompson pooling, `h_loan = sum_i(score_i/incl_prob_i) /
sum_i(1/incl_prob_i)`. By the multiobs builder's own design, a loan's sampled reference months
deliberately span different ages/incentive regimes — that separation of incentive from burnout is
the entire point of multiobs sampling — so pooling them into one number estimates a
**lifetime-average** hazard over the sampled window, not the **cutoff-conditional** hazard
(`aggregate()`'s and origination/trailing's h_t both already are) that a CPR forecast needs.
Feeding a lifetime-average h_loan into `aggregate()` under the same column name would compare a
different estimand, not correct multiobs's calibration against a like-for-like target. The
formula's *direction* was still verified correct on a synthetic 3-observation case before the
estimand problem killed it (`scripts/diag/diag_multiobs_ipw_weight_direction.py`) — kept as a
record that the pooling math itself was sound, only wrongly applied — and the real fix was pushed
upstream: retrain with `--use_ipw` so the *loss* is IPW-corrected during training, and the trained
model's score needs no further per-loan reweighting at inference.

**The inverted-weight bug.** `train_hazard_multiobs.py`'s `--use_ipw` loss used
`w = torch.tensor(bweight, device=DEVICE)` — the batch's `incl_prob` values **directly** as the
per-sample weight — where Horvitz-Thompson weighting requires `w = 1/incl_prob`. This is backwards:
a rare pool draw (small `incl_prob`) stands in for more of its stratum's unsampled population and
must be *up*weighted, not down-weighted. Fixed by factoring the weight into its own function,
`ipw_weight(incl_prob) = 1.0 / torch.tensor(incl_prob)`, called at the loss site instead of the raw
tensor.

Batch evidence (`ObservationSampler.sample_batch`, seed=42, batch_size=2048, real
`cutoff_2020_zbc_multiobs_k5_h1` training data — reproduced from
`scripts/diag/diag_ipw_batch_weight_check.py`): the sampled batch had 163 label=1 and 1,885
label=0 observations. Label=1 observations got mean weight **1.0000** either way (their
`incl_prob` is always exactly 1.0 — see the premise check below). Label=0 observations got mean
weight **0.3806** under the buggy direct-`incl_prob` code — *lower* than the positives — and would
get mean weight **7.3614** under the correct `1/incl_prob` inversion, i.e. more than 19x higher
than what the buggy code actually used and, correctly, higher than the positives' weight rather
than lower.

The premise this diagnosis rests on was verified against the full training set rather than
assumed: **571,553 / 571,553** label=1 observations have `incl_prob == 1.0` exactly (the
mandatory/terminal draw is the only draw that can carry a positive label under this builder's H=1
window, and it is always included when eligible). The 6,289,824 label=0 observations have
`incl_prob` mean 0.3783, min 0.04255, max 1.0, with 21.04% sitting exactly at 1.0 (their own
mandatory-terminal draws that happened to still be censored). Index alignment (`batch_weight`
indexed by the same `idx` as `labels`/`seq`) and loss normalization were checked separately and
were both clean — neither was a contributing cause.

**The smoke test passed the inverted formula silently** because
`scripts/diag/test_train_hazard_multiobs_smoke.py` only asserted that the `--use_ipw` code path
*executed* (finite loss, decreasing over epochs) — never that the weight it computed pointed the
correct direction. A `check_ipw_weight_direction()` case was added, calling
`train_hazard_multiobs.ipw_weight()` directly (the real function `train_and_evaluate` uses, not a
hand-rolled duplicate), asserting that a smaller `incl_prob` produces a strictly larger weight and
that `incl_prob == 1.0` observations get the minimum weight of 1.0.

**Buggy run's numbers vs. the corrected run's, read from each run's own artifacts.** The buggy
run (`outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_buggy/`, preserved rather than overwritten)
peaked at epoch 8, `best_auc = 0.7740`, Platt `a=3.3898, b=-3.8723`. Its matched-population
(365,146-loan) coupon-level forecast is saturated at nearly every coupon — `forecast_cpr` of
90.3–99.3% across coupons 2.0–5.0 against realized 12.2–36.0% — giving dispersion **7.853..2.615**
and pooled ratio **3.4089**: calibration got strictly worse than the uncorrected (`--use_ipw=False`)
run, not better, which is what first flagged the direction bug rather than a training issue. The
corrected, seed-controlled run's best checkpoint (below) instead gives dispersion **1.139..0.717**
and pooled ratio **0.9091** — inside a plausible range rather than saturated.

**The seeding gap.** `train_hazard_multiobs.py` seeded only the batch sampler
(`np.random.default_rng`); model weight initialization and dropout were never seeded at all. Two
same-seed, post-bugfix `--use_ipw` reruns
(`outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw/hazard_best.pt` vs.
`..._ipw_epoch1_20260907/hazard_best.pt`, both epoch-1 best checkpoints) differed by up to **0.7919**
in a single parameter tensor (`transformer.layers.1.norm2.weight`), confirmed by direct tensor
comparison (`logs/forecast_ipw_epoch_cmp_17127037.out`) — despite near-equal AUCs (0.7124 vs.
0.7127) that would have looked like reproducibility if only the metric, not the weights, had been
compared. Adding `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)` alone reduced the max
per-parameter diff to **0.21** (documented in a `train_hazard_multiobs.py` code comment) — better,
but not bit-identical, because cuDNN algorithm selection and CUDA attention/reduction kernels
(flash/mem-efficient SDPA backward, autotuned convolutions) remain free to pick nondeterministic
implementations even with the RNG seeded. Bit-identical reproduction (0 of 31 parameter tensors
differing, confirmed by `scripts/diag/compare_seedcheck_checkpoints.py` on jobs 17140541/17140542,
run tags `_seedcheck_a`/`_seedcheck_b`) required all of: `torch.backends.cudnn.deterministic =
True`, `torch.backends.cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True,
warn_only=True)`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported in the sbatch environment before
the job starts.

**Epoch-4 vs. epoch-50, on the now-reproducible run.** With run-to-run variance eliminated as a
confound, `_seedcheck_a`'s `hazard_best.pt` (epoch 4, AUC 0.7181) and `hazard_final.pt` (epoch 50,
AUC 0.7090) were scored identically (trailing test sequences, same 365,146-loan matched population,
`logit_offset=0.0` for both — job 17149933):

| coupon | n_loans | realized | h_t (best) | CPR (best) | ratio (best) | h_t (final) | CPR (final) | ratio (final) |
|---|---|---|---|---|---|---|---|---|
| 2.0 | 19,255 | 12.205 | 0.0125 | 13.899 | 1.1388 | 0.0190 | 20.279 | 1.6616 |
| 2.5 | 34,755 | 15.808 | 0.0153 | 16.656 | 1.0537 | 0.0212 | 22.268 | 1.4087 |
| 3.0 | 69,663 | 26.361 | 0.0233 | 23.500 | 0.8914 | 0.0272 | 26.790 | 1.0163 |
| 3.5 | 38,981 | 33.876 | 0.0317 | 30.398 | 0.8973 | 0.0349 | 32.782 | 0.9677 |
| 4.0 | 41,460 | 35.174 | 0.0331 | 31.903 | 0.9070 | 0.0347 | 33.121 | 0.9417 |
| 4.5 | 10,615 | 35.356 | 0.0278 | 27.960 | 0.7908 | 0.0274 | 27.701 | 0.7835 |
| 5.0 | 5,453 | 36.017 | 0.0250 | 25.812 | 0.7167 | 0.0237 | 24.639 | 0.6841 |

Dispersion widens from **1.139..0.717** (best) to **1.662..0.684** (final), concentrated almost
entirely in coupons 2.0 and 2.5 — ratio 1.14→1.66 and 1.05→1.41 respectively — while coupons
3.0–5.0 hold roughly stable across the same 46 additional epochs of training. Pooled ratio moves
**0.9091 → 1.0233**, nominally closer to 1.0, but per this repo's own f35b0bc caveat that is **not**
evidence of better calibration when dispersion is simultaneously widening: a pooled ratio can
improve purely because low-coupon overshoot and high-coupon undershoot are canceling in the
n-weighted average, which is exactly the pattern here. Neither checkpoint saturates (h_t range
0.0125–0.0331 best, 0.0190–0.0349 final; 0 of 7 coupons above the 0.10 threshold in both cases), so
this is not a saturating-region story — additional training specifically distorts the low-coupon
end of the curve.

**Proposed mechanism — untested.** One candidate explanation: IPW's upweighted rare pool draws
(small `incl_prob`, weight up to ~7-24x per the distribution characterized in
`diag_multiobs_ipw_weight_direction.py`) are disproportionately long-lived, low-coupon loans that
survived many eligible reference months without prepaying, and additional training epochs let the
model increasingly fit those upweighted observations' burnout signature at the expense of
cutoff-conditional accuracy at low coupons specifically. This is a hypothesis only — not checked
against the actual coupon/incl_prob joint distribution, not checked against vintage or credit-score
confounds, and not the only mechanism consistent with the observed pattern.

**Two new Platt calibrations — never mix with any of the other four.** Read directly from each
run's `results.json`: the buggy run is `a=3.3897956287900497, b=-3.8722876743517958`; the
corrected, seed-controlled run (`_seedcheck_a`, identical to `_seedcheck_b`) is
`a=29.600035577849884, b=-2.90084783552794`. Together with the four already on record — OAS
loan-level (0.4934/-4.840), cohort-CPR forecast (0.4559/-3.1376), corrected-label zbc 2020
(2.4245/-2.4348), and trailing zbc 2020 (12.9671/-13.0827) — **six distinct Platt calibrations now
exist**. The two non-seed-controlled corrected reruns along the way (`_ipw`: a=25.3948, b=-2.8621;
`_ipw_epoch1_20260907`: a=25.5254, b=-2.8500) are not added to this list as permanent entries — they
are exactly the pre-determinism-fix nondeterminism this section exists to document, superseded by
the seed-controlled canonical calibration above.

**Residual top-end under-forecast.** Even on the best (epoch-4) checkpoint — the more calibrated of
the two — coupons 4.5 and 5.0 are under-forecast by 21% and 28% respectively (ratio 0.7908 and
0.7167), which is now the **largest** error in the table: bigger than coupon 2.0's 14% over-forecast
(ratio 1.1388), the next-largest deviation from 1.0. Not addressed by this session's work; flagged
as the next open question rather than treated as resolved by the IPW fix.

Artifacts: `scripts/train_hazard_multiobs.py` (`ipw_weight()`, seeding, `hazard_final.pt`,
`--run_tag`), `scripts/diag/test_train_hazard_multiobs_smoke.py`
(`check_ipw_weight_direction()`), `scripts/diag/diag_multiobs_ipw_weight_direction.py`,
`scripts/diag/diag_ipw_batch_weight_check.py`, `scripts/diag/compare_seedcheck_checkpoints.py`,
`scripts/forecast_multiobs_ipw_cpr.py`, `scripts/forecast_multiobs_ipw_epoch_compare.py`,
`scripts/forecast_multiobs_ipw_seedcheck_epoch_compare.py`. Jobs: buggy training 17068340,
corrected-but-non-seeded training reruns 17092647 (`_ipw_epoch1_20260907`) and 17118450 (`_ipw`,
adds `hazard_final.pt`), their checkpoint-identity/epoch comparison 17127037 (found the 0.7919 max
param diff), seedcheck round 1 (`manual_seed` only, not bit-identical) 17128745/17128746, seedcheck
round 2 (full determinism, bit-identical) training 17140541/17140542, seedcheck epoch-compare
17149933. Outputs under
`outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw{,_buggy,_epoch1_20260907,_seedcheck_a,_seedcheck_b}/`.

### Advisor follow-up: realized-rate weighting, non-prepay censoring, un-annualized calibration, and ESS (Sep 8, 2026)

**CORRECTION (Sep 25, 2026):** these forecast numbers came from scoring right-aligned multiobs
checkpoints on the left-aligned `TRAIL_SEQ_DIR` test set. For `_seedcheck_a`, the like-for-like
corrected values (the left-aligned reproduction matched the original exactly): epoch 4 ratio range
1.139..0.717 → 1.047..0.544, pooled 0.9091 → 0.9736; epoch 50 1.662..0.684 → 1.161..0.664, pooled
1.0233 → 1.0794; Jan 2021 un-annualized at coupons 4.5 / 5.0: 0.803 / 0.746 → 1.027 / 0.982. The Sep
5 non-IPW `k5_h1` numbers (earlier in this file) are affected but were not rescored. The 1.79x/3.21x origination-model
benchmark elsewhere in this README is unaffected (`infer_test_set` uses the last real timestep). See
the "Fixed-fraction sampling rates (a_eff / b_eff), a sequence-alignment bug, and the cutoff_2002 2003
forward test" section (Sep 24-25, 2026) near the end of this file for the full measurement.

**`realized_cpr` carries no training-time weight.** Traced end to end in
`forecast_rolling_cpr.py`: `read_coupon_and_realized()` builds `prepaid_set` from raw
`zero_balance_code_actual == 1.0` membership only, and `aggregate()` computes
`df['realized'] = df['loan_id'].isin(prepaid_set).astype(int)` then
`realized_cpr = round(g['realized'].mean() * 100, 4)` (lines 492/509) — a plain unweighted
mean of a 0/1 indicator per coupon. `incl_prob`/`ipw_weight()` exist only inside
`train_hazard_multiobs.py`'s loss and never appear in this file. The realized side of every
forecast/realized comparison in this README is therefore unweighted by construction, independent
of whatever IPW correction (or bug) is live on the forecast side.

**Non-prepay terminations are coded as censored survivors, not a distinct outcome — a
simplification, not yet validated as harmless.** In `prepare_sequences_multiobs_zbc.py`'s
`_prepare_panel()` (lines 466–470), `zbc_idx` is built by filtering to
`zero_balance_code_actual == 1.0` before the groupby, so `is_prepaid` is `True` only for that
code; every other terminal code falls through the `.fillna(df['L'] - 1)` branch and is treated
identically to an ordinary loan still performing at the cutoff. Counted directly on the
365,146-loan matched population (fresh raw pass, job 17230371): 142,889 loans terminate zbc==1
(prepaid), 221,644 have no terminal code (still current), and **613 loans (0.17%) terminate via
zbc 2/3/6/9/15/16** (third-party sale 64, short sale 27, repurchase 349, REO 132, note sale 5,
reperforming-loan sale 36) — all 613 silently coded as right-censored survivors up to their last
row. This is a modeling simplification stated plainly as such: it has not been checked for whether
these loans share a distress signature with true prepayments that would bias the hazard if
mislabeled as censored, only confirmed that the code treats them as censored and that the affected
population is small in this cutoff/vintage window.

**Un-annualized comparison: the 4.5/5.0 under-forecast is not a `1-(1-h)^12` artifact.**
Scored the epoch-4 `_seedcheck_a` checkpoint (`hazard_best.pt`) against realized prepayment in
January 2021 specifically — the single calendar month the H=1, Dec-2020-cutoff-anchored model
actually predicts — instead of the annualized 12-month realized_cpr (job 17230237):

| coupon | n_loans | mean h_t (monthly) | realized, Jan 2021 | ratio |
|---|---|---|---|---|
| 2.0 | 19,255 | 0.01250 | 0.00784 | 1.594 |
| 2.5 | 34,755 | 0.01530 | 0.01240 | 1.233 |
| 3.0 | 69,663 | 0.02328 | 0.02773 | 0.839 |
| 3.5 | 38,981 | 0.03169 | 0.03650 | 0.868 |
| 4.0 | 41,460 | 0.03312 | 0.03582 | 0.925 |
| 4.5 | 10,615 | 0.02777 | 0.03457 | 0.803 |
| 5.0 | 5,453 | 0.02505 | 0.03356 | 0.746 |

The under-forecast at 4.5/5.0 (ratio 0.803, 0.746) is present at the single-month level, before any
annualization — this rules out `1-(1-h)^12` saturation as the explanation. **Caveat:** January is a
seasonal prepayment trough, so this comparison also implicitly nets out whatever seasonal
adjustment the annual realized_cpr embeds; it isolates the annualization-vs-genuine-miscalibration
question but does not by itself validate the full-year forecast.

**ESS/n (relative weight concentration) tracks which coupons drifted between epoch 4 and 50; raw
ESS and mean weight do not.** Weight = `1/incl_prob` on the full training set (6,861,377
observations, 1,460,803 unique loans), coupon recovered from `incentive_at_ref + PMMS(ref_month)`
(validated: only 4 of 1.4M loans show a cross-observation spread above 0.01, the rest is float32
noise):

| coupon | n | raw ESS | ESS/n | drifted (epoch 4→50)? |
|---|---|---|---|---|
| 2.0 | 210,278 | 72,758 | 34.6% | yes |
| 2.5 | 857,682 | 338,509 | 39.5% | yes |
| 3.0 | 1,585,475 | 804,664 | 50.8% | no |
| 3.5 | 1,866,555 | 1,104,091 | 59.2% | no |
| 4.0 | 1,493,411 | 898,297 | 60.2% | no |
| 4.5 | 653,920 | 369,728 | 56.5% | no |
| 5.0 | 150,526 | 91,169 | 60.6% | no |

The two drifted coupons are exactly the two lowest ESS/n in this range, with an ~11-point gap down
to the stable 3.0–5.0 band (50.8–60.6%) — consistent with the drift being driven by relative weight
concentration at low coupons. This does **not** hold for raw (absolute) ESS: 2.5 has more effective
observations (338,509) than 5.0 (91,169), yet 2.5 drifted and 5.0 held steady — if scarcity of
absolute effective samples were the mechanism, 5.0 should be at least as vulnerable, and it isn't.
Mean weight is similarly uninformative (2.0 mean 5.46 vs. 5.0 mean 5.06 — nearly identical despite
opposite drift outcomes). The scoped conclusion is specifically about concentration, not scarcity:
a smaller fraction of the nominal sample at 2.0/2.5 is carried by disproportionately high-weight
rows relative to that coupon's own sample size, not that those coupons have too little data in
absolute terms.

**Points toward, but does not yet implement, the advisor's fixed-fraction resampling proposal.**
If relative weight concentration (not raw scarcity) is driving the epoch-4→50 drift at 2.0/2.5,
capping or resampling to a fixed fraction of eligible pool draws per loan (flattening the ESS/n
profile across coupons) is the targeted next build — this has not been built or tested; it is
recorded here as the next open question pending the advisor's input, not as a completed fix.

Artifacts: `scripts/diag/diag_advisor_unannualized_epoch4.py` (job 17230237),
`scripts/diag/diag_advisor_zbc_terminal_counts.py` (job 17230371),
`scripts/slurm/run_diag_advisor_unannualized_epoch4.sbatch`,
`scripts/slurm/run_diag_advisor_zbc_counts.sbatch`. Outputs:
`outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a/unannualized_comparison_epoch4_202101.csv`,
`outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a/terminal_zbc_matched_population.csv`.

## Fixed-fraction resampling: coupon-level preview and mechanism verification (Sep 9-10, 2026)

Implements the fixed-fraction resampling proposal recorded as the open next step at the end of
the Sep 8 advisor follow-up above: resample to a fixed fraction of each loan's eligible pool,
instead of `fixed_k`'s k_draws-per-loan (which scales `incl_prob` directly with pool size and
drives the length bias documented in `census_panel_baseline.py`'s coupon-level `n_eligible`
spread).

**Formula** (`select_observations()`, `sampling_mode='fixed_fraction'`,
`prepare_sequences_multiobs_zbc.py:653-670`):
```
budget    = ceil(frac_draws * n_pool)          # non-mandatory slots
incl_prob (non-terminal) = min(budget, n_pool) / n_pool
mandatory/terminal draw: incl_prob = 1.0 always, independent of frac_draws or n_pool
```
No explicit `max(1, budget)` floor is needed — `ceil(frac_draws * n_pool)` is already >= 1 for
any `n_pool >= 1` and `frac_draws > 0`, since `frac_draws * n_pool > 0` and `ceil()` of a positive
number is always >= 1 (now stated explicitly in the docstring).

**Coupon-level preview** (`fixed_fraction_coupon_preview.py`, `frac_draws=0.2`, cutoff_2020, full
population, job 17295020):

| coupon | n_eligible_median (census) | incl_prob_mean | sampled/eligible ratio |
|---|---|---|---|
| 2.0 | 3 | 0.2550 | 0.2635 |
| 2.5 | 5 | 0.2381 | 0.2544 |
| 3.0 | 30 | 0.2156 | 0.2342 |
| 3.5 | 35 | 0.2126 | 0.2327 |
| 4.0 | 32 | 0.2126 | 0.2331 |
| 4.5 | 24 | 0.2161 | 0.2411 |
| 5.0 | 23 | 0.2192 | 0.2484 |
| 5.5 | 21 | 0.2245 | 0.2607 |
| 6.0 | 22 | 0.2250 | 0.2620 |
| 6.5 | 20 | 0.2342 | 0.2757 |

(coupons 1.0/1.5 excluded from this table for readability only — `n_eligible_median=1`,
tiny/degenerate pools; the full 12-bucket table including them is in
`outputs/fixed_fraction_coupon_preview_cutoff_2020_f0.2.csv`.)

**Mechanism confirmed exactly, population-wide — not inferred from the shape of this table.**
Two follow-up checks: job 17299525 (per-loan spot check, vintage 2013Q1) and job 17300560 (full
41-vintage population check, `fixed_fraction_population_check.py`):

- *Per-loan*: 10/10 sampled loans (coupons 2.0 and 3.5, vintage 2013Q1) matched
  `budget = ceil(0.2*n_pool)` exactly.
- *Population, by n_pool*: grouping every non-terminal draw across all 41 vintages by `n_pool`
  (94 distinct values) and comparing observed `incl_prob` to `ceil(0.2*n_pool)/n_pool`, max
  deviation across every group's min/max is **0.0**; max deviation of the group mean from the
  formula is **5.55e-17** (float64 machine epsilon).
  `outputs/fixed_fraction_population_check_by_npool_f0.2.csv`.
- *Population, by coupon*: each coupon's `incl_prob_mean` matches exactly what its own `n_pool`
  distribution predicts via the formula (max deviation 5.55e-17).
  `outputs/fixed_fraction_population_check_by_coupon_f0.2.csv`.

This rules out the originally hypothesized mechanism: `incl_prob_mean` in the preview table is
computed on `obs[~obs['is_terminal']]` only, so the mandatory/terminal draw's `incl_prob=1.0`
never enters it — it cannot be "the terminal draw dominating and diluting out." The actual, fully
confirmed driver is `ceil()` rounding up the non-mandatory budget itself:
`incl_prob = ceil(0.2*n_pool)/n_pool >= 0.2` always, with the excess shrinking as `n_pool` grows
and vanishing exactly at multiples of 5.

**The range itself, no derived multiplier.** Deliberately not framed as a single "X-fold collapses
to Y-fold" number: `n_pool` can legitimately be summarized per coupon at least three different
ways, and they disagree with each other by roughly 3.2x to 11.7x depending on which is used,
because each weights loans differently — Step 1's census-derived per-loan median `n_eligible`
(coupon 2.0 -> 6.5 range 3 -> 35, an 11.7x spread), this check's own saved row-weighted mean
`n_pool` (range 19.0 -> 61.06, a 3.2x spread), and a loan-weighted median `n_pool` recovered from
this check's per-vintage checkpoints (range 4 -> 34, an 8.5x spread). Picking one of these to
headline would assert a precision the underlying comparison doesn't have. What's directly comparable and
exact is `incl_prob_mean` itself, read straight from `outputs/fixed_fraction_population_check_by_coupon_f0.2.csv`:
it spans **0.2126 (coupon 4.0) to 0.2550 (coupon 2.0), a ~1.20x range**, against `fixed_k`'s
`incl_prob = k_draws/n_pool`, which scales directly and unboundedly with pool size (a loan with
10x the pool gets 1/10th the inclusion probability, with no ceiling-driven compression at all).

**Monotonicity — not strictly monotonic under the check's own row-weighted statistic; a
follow-up loan-weighting test explains most, not all, of the break.** Ranking coupons by this
check's own saved `mean_n_pool` (`outputs/fixed_fraction_population_check_by_coupon_f0.2.csv`,
row/observation-weighted — each loan's `n_pool` is weighted by how many sampled rows it
contributed, i.e. by roughly its own `n_pool`), `incl_prob_mean` does **not** monotonically
decrease: coupons **2.0, 2.5, and 3.5** all show an increase where the "larger pool -> lower
incl_prob" pattern predicts a decrease. Recomputing the same statistic **loan-weighted** instead
(each loan's `n_pool` weighted once, by loan count, recovered exactly as
`n_loans = n_rows / ceil(0.2*n_pool)` per `(coupon, n_pool)` cell — verified exact,
`max|n_rows - n_loans*budget| = 0.0`) restores strict monotonic decrease across 2.0 -> 2.5 -> 3.0
-> 4.0:

| coupon | row-weighted mean n_pool | loan-weighted mean n_pool | incl_prob_mean |
|---|---|---|---|
| 2.0 | 61.06 | 22.06 | 0.25502 |
| 2.5 | 56.76 | 23.67 | 0.23808 |
| 3.0 | 57.88 | 34.81 | 0.21560 |
| 4.0 | 48.75 | 35.38 | 0.21257 |
| 3.5 | 50.45 | 35.63 | 0.21261 |

This **confirms the row-weighting-artifact explanation for coupons 2.0 and 2.5**: the
row-weighted mean over-weights each loan by roughly its own `n_pool`, which is why coupon 2.0
appears to have a *larger* "mean" pool (61.06) than coupon 3.0 (57.88) despite having far fewer
large-pool loans in absolute terms — switching to loan-weighting corrects this and both coupons
fall into their expected rank. It does **not** explain coupon 3.5: even loan-weighted, 3.5
(35.63) still sits fractionally above 4.0 (35.38) in pool size while also sitting fractionally
above it in `incl_prob_mean` (0.21261 vs 0.21257, a 0.000046 / ~0.02% difference) — a real,
non-floating-point difference under the `diff > 1e-9` tolerance used to flag it, but small enough
that it may simply reflect the two coupons' pool-size distributions being nearly identical rather
than a distinct driving factor. This residual is not yet explained and is left as such, not
attributed to weighting or anything else.

The two smallest-sample coupons, 1.0 and 1.5 (430 and 18,298 loans respectively, median `n_pool`
1-2), sit far outside this range (`incl_prob_mean` 0.71-0.76, since `n_pool=1` forces
`incl_prob=1.0`) and are reported separately rather than folded into the headline range.

### SLURM operational notes
Job 17299525 (single-vintage per-loan inspection, 2013Q1 alone, ~50M rows) OOM'd at 48G; needed
96G to complete. A single large vintage's raw CSV load can exceed a "just inspect a few loans"
job's naive memory footprint — size diag jobs touching a full vintage at the same 96G/8-cpu
envelope as the full 41-vintage runs, not down.

Artifacts: `scripts/diag/fixed_fraction_coupon_preview.py` (job 17295020),
`scripts/diag/inspect_fixed_fraction_examples.py` (job 17299525, resubmitted at 96G after an
OOM at 48G), `scripts/diag/fixed_fraction_population_check.py` (job 17300560),
`scripts/slurm/run_fixed_fraction_coupon_preview_2020.sbatch`,
`scripts/slurm/run_inspect_fixed_fraction_examples.sbatch`,
`scripts/slurm/run_fixed_fraction_population_check.sbatch`. Outputs:
`outputs/fixed_fraction_coupon_preview_cutoff_2020_f0.2.csv`,
`outputs/fixed_fraction_population_check_by_npool_f0.2.csv`,
`outputs/fixed_fraction_population_check_by_coupon_f0.2.csv`.

## Window-length comparison: training results and a reproducibility gap (Sep 10-11, 2026)

**Census validation of the training population (Step 3, closes the advisor's standing "always
validate against the census panel" instruction).** `check_sampler_vs_census.py` (job 17308661,
cutoff Dec 2020, `frac_draws=0.2`) confirmed the IPW-weighted `fixed_fraction` sampler reproduces
the census panel's totals overall and per coupon — pooled eligible-loan-months, prepay-event
counts, and the weight-direction check all passed within the 1% tolerance. A follow-up script,
`check_sampler_vs_census_by_vintage.py`, re-ran the same comparison broken out **per vintage
instead of pooled**, to rule out offsetting errors hiding inside the pooled pass: **41/41
vintages pass both checks, `rel_dev` exactly 0.0 at every single vintage** — a bit-for-bit exact
match, not an approximate one. Output: `outputs/check_sampler_vs_census_by_vintage_f0.2.csv`.

**Three full training runs at L=33/48/60, same population.** All three use `fixed_fraction`
sampling (`f=0.2`), the same cutoff_2020 population, and the same train/test split — confirmed by
comparing `train_loan_ids_split.npy`/`test_loan_ids_split.npy`/`scaler.pkl` across the three
sequence directories, which are byte-identical. Each ran 50 epochs with `--use_ipw`:

| window (L) | job | best_auc |
|---|---|---|
| 33 | 17355084 | 0.7164 |
| 48 | 17355085 | 0.7170 |
| 60 | 17355086 | 0.7160 |

**The spread cannot be distinguished from single-seed noise.** The L33-L60 best_auc spread is
~0.001 (0.7170 - 0.7160). Over the last 15-20 epochs of each run, near convergence, epoch-to-epoch
AUC already varies by 0.0015-0.0020 within a *single* run — larger than the spread between the
three window lengths. On this evidence, L=33/48/60 cannot be ranked; the apparent best (L=48)
could just as easily be noise. **The window-length question is left open, pending a second-seed
comparison, per the email sent to the advisor.**

**The CUBLAS_WORKSPACE_CONFIG gap.** Full determinism for these runs requires
`CUBLAS_WORKSPACE_CONFIG` to be set in the sbatch job's environment, on top of the code-level
`torch.manual_seed` / `cudnn.deterministic` calls already in `train_hazard_multiobs.py` — the
env var alone is necessary but not present by default. It was missing from all three of today's
job scripts (`run_train_multiobs_2020_f0.2_L33/L48/L60.sbatch`). A repeat of the L33 run with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` set (job 17378466, `run_train_multiobs_2020_f0.2_L33_repeat.sbatch`)
reproduced a `results.json` byte-for-byte identical to the original L33 run (best_auc
0.7163846603462104 both times, 0.0 delta). That confirms the repeat run itself was deterministic —
it does **not** retroactively establish whether the original three runs were deterministic, since
the env var was absent when they ran. That determinism status is recorded plainly as unknown, not
asserted either way.

**Fix.** `train_hazard_multiobs.py` now prints the resolved `CUBLAS_WORKSPACE_CONFIG` value at the
start of every training run, so a missing env var shows up in the job log instead of silently
producing an unverifiable-determinism gap again.

## No-history control (max_seq_len=1): slicing approach, result, and the same noise-floor caveat (Sep 12, 2026)

**Setup.** Built the `max_seq_len=1` "no-history" control by slicing the last timestep off the
already-built L33 multiobs data (`scripts/diag/slice_l33_to_l1.py`), instead of running a fresh
`prepare_sequences_multiobs_zbc.py --max_seq_len 1` prep job. Source:
`data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1` (L33); dest: the same directory name
with an `_L1` suffix.

**Why slicing is valid, not an approximation.** `build_sequences_multiobs` is right-aligned:
index `MAX_SEQ_LEN-1` is always `ref_month` itself, for every observation — confirmed empirically
(L33's `test_mask` last column is 100% `True` across all 2,808,691 rows). Eligibility, selection,
labels, and `incl_prob` have zero dependence on `MAX_SEQ_LEN` (the window-gap filter is the only
`MAX_SEQ_LEN`-dependent term, and it is vacuously `False` at L=1 anyway). So `train_seq[:, -1:, :]`
**is** the `--max_seq_len 1` build for every observation the L33 run already has, with
labels/`incl_prob` carried over unchanged — not a stand-in for it.

**Training run.** Same `fixed_fraction` sampling (`f=0.2`), same cutoff_2020 population and
train/test split as the window-length runs above, 50 epochs, `--use_ipw`
(`run_train_multiobs_2020_f0.2_L1.sbatch`, job 17520583):

| window (L) | job | best_auc |
|---|---|---|
| 1 | 17520583 | 0.7145 |
| 33 | 17355084 | 0.7164 |

**The spread cannot be distinguished from single-seed noise, by the same standard already applied
to the window-length comparison.** Same statistic as the L33/L48/L60 noise floor above — AUC range
(max − min) over the last 15/20 epochs, not epoch-to-epoch deltas: the L=1 run's range is **0.0018**
(identical for both windows, since the min (epoch 39) and max (epoch 43) both fall inside the last
15 epochs), in the same 0.0015–0.0019 band as L33 (0.0015), L48 (0.0016–0.0019), and L60 (0.0019).
The L33-vs-L1 best_auc spread (0.7164 − 0.7145 = 0.0019) sits inside that band. **This result does
not confirm a history effect, and it does not rule one out either** — on this evidence alone,
no-history (L=1) cannot be distinguished from "a different draw of the same noisy training
process." Both the window-length question and this history question are left open, pending a
proper multi-seed comparison — not another email to the advisor asking
him to decide with an open question again.

## Seed-replication comparison resolves the L33-vs-L1 question; the noise-floor proxy above was too conservative (Sep 13, 2026)

**Setup.** Reran both L33 and L1 (same `fixed_fraction f=0.2`, same cutoff_2020 population/split,
50 epochs, `--use_ipw`) at two additional seeds (7, 123), giving n=3 per condition alongside the
existing seed=42 baselines (jobs 17355084/17520583). New jobs: 17594751 (L33 seed7), 17594752
(L33 seed123), 17594753 (L1 seed7), 17594754 (L1 seed123) — all COMPLETED (0:0), seeds confirmed
against each run's `results.json` `seed` field and against the (unseeded) `--seed` flag in each
run's own sbatch script.

| condition | seed=42 | seed=7 | seed=123 | mean | sd (n−1) | range |
|---|---|---|---|---|---|---|
| L33 best_auc | 0.716385 | 0.716581 | 0.716961 | 0.716642 | 0.000293 | 0.000576 |
| L1 best_auc | 0.714495 | 0.714702 | 0.714942 | 0.714713 | 0.000224 | 0.000447 |
| L33 mean_auc_last10 | 0.716000* | 0.715865 | 0.716511 | 0.716125 | 0.000341 | 0.000646 |
| L1 mean_auc_last10 | 0.713960* | 0.713915 | 0.714643 | 0.714173 | 0.000408 | 0.000728 |

*seed=42's `results.json` predates the `mean_auc_last10` field; computed manually from its stored
`history` (mean of epochs 41–50), same definition the current script writes.

**The gap is real, not noise.** Gap (L33 − L1 mean) = 0.001929 on best_auc, 0.001953 on
mean_auc_last10. Expressed against each condition's own seed-to-seed spread (8 ratios: 2 metrics
× {L33, L1} × {sd, range}), the gap is consistently larger than the within-group noise, ranging
from **2.68× to 8.62×** — precisely (not "3-9x" or any rounder figure): best_auc gap/L33 sd
6.58×, gap/L1 sd 8.62×, gap/L33 range 3.35×, gap/L1 range 4.32×; mean_auc_last10 gap/L33 sd 5.73×,
gap/L1 sd 4.79×, gap/L33 range 3.02×, gap/L1 range 2.68×. This supersedes the "cannot be
distinguished from noise" conclusion two sections above for L33-vs-L1 specifically: with real
n=3 seed replication instead of the single-run epoch-swing proxy, history vs. no-history is a
distinguishable effect, not indistinguishable-from-noise.

**The epoch-swing proxy used above overestimated true seed-to-seed noise by 2.6×–4.0×.** The
proxy quoted for L33/L1 above (AUC range over the last 15/20 epochs of a *single* run) was 0.0015
(L33) and 0.0018 (L1) — reproduced here from the same seed=42 histories to confirm apples-to-apples.
The real, measured seed-to-seed range from n=3 is 0.000576 (L33) and 0.000447 (L1) — smaller by
2.6× (0.0015/0.000576) and 4.0× (0.0018/0.000447) respectively. The proxy was a conservative
overestimate, not an underestimate. Consequence: the L33/L48/L60 window-length comparison above
(spread ≈0.001, judged "cannot be ranked" against a 0.0015–0.0019 proxy band) used a noise floor
that this replication shows was too large — a spread of ≈0.001 against a real per-seed range
closer to ~0.0005 is roughly 2× the real noise, not safely inside it. That comparison is not
retroactively resolved (L48/L60 were never rerun at other seeds) but its "indistinguishable from
noise" verdict should be treated as unconfirmed pending their own seed replication, not settled.

**Population identity confirmed, not assumed.** `train_hazard_multiobs.py`'s `--seed` reaches
`torch.manual_seed`/`cuda.manual_seed_all` (model init) and a `np.random.default_rng(seed)` used
only inside `ObservationSampler.sample_batch` for `rng.integers(0, self.n, size=batch_size)` —
uniform-with-replacement minibatch draws over the already-fixed `self.n` rows. It never reaches
`prepare_sequences_multiobs_zbc.py`, which has no `--seed` argument at all; its observation
selection is a deterministic `hashlib.blake2b` hash of `loan_id` (`_loan_base_hash`,
`_mix_hash`), independent of any seed. All four seed-replication runs' logs point at the exact
same `Sequences:` directories as their seed=42 baselines (`cutoff_2020_zbc_multiobs_f0.2_h1` for
L33, `..._h1_L1` for L1) — no prep job reran. The Step 3 census-panel validation
(`scripts/diag/check_sampler_vs_census.py`, which reruns `select_observations()` directly against
`census_panel_baseline_cutoff_2020.json`, entirely upstream of the training script) therefore
already covers all four seed-replication runs without needing to be rerun per seed.

## Pre-2013 historical extension: data verification, cell-grid sampler, builder patches, and a CP/U miscoding finding (Sep 18-19, 2026)

Per the advisor's Sep 16 reply ("stick with L=33, and go forward with the historical builout"),
work began extending the multiobs pipeline back to 2000Q1. This section covers verification of
the pre-2013 raw data, the cell-grid loan sampler built to make a 29.1M-loan historical corpus
tractable, three patches to `prepare_sequences_multiobs_zbc.py` needed to consume it, a test build
at `cutoff_2002`, and a categorical-encoding bug the extension surfaced that also affects
previously-reported modern-era results.

### Pre-2013 data verification

Before drawing any historical sample, the prepay label itself was checked for maturity
conflation — does `zero_balance_code==01` (the label) fire on loans simply reaching the end of
their term, rather than genuine prepayment? `scripts/diag/term_maturity_scan_2002q3_2011q2.py`
(job 17952495) checked two files spanning the historical range: of all `ZBC==01` terminations,
loans ending within 2 months of their `original_loan_term` are **1.03%** (2002Q3, n=731,581) and
**0.80%** (2011Q2, n=258,431) of the total. The label is not materially conflated with maturity.

This builds on the pre-2013 column verification already recorded above ("Pre-2013 historical data
(verified 2026-07-19)") and on `count_prepay_events_pre2013.py`'s existing column-choice fix
(`zero_balance_code` at usecols 43, not `extra_13`, which is nearly empty pre-2013) — not
re-verified here, only extended with the maturity check.

### Cell-grid loan sampler

The pre-2013 corpus is 29,130,522 loans across 52 acquisition-quarter files (2000Q1-2012Q4) —
too large to process whole for every build. `scripts/build_cell_grid_sample_pre2013.py` selects a
loan-level (not row-level) subsample, gated on the same (`vintage_quarter`, `coupon`) cell grid
`count_prepay_events_pre2013.py` already measured: cells with ≤`TARGET_CAP` (5000) prepayment
events keep every loan; cells above it keep a hash-ranked `ceil(5000/n_events * n_loans)`-loan
subset, chosen by the same deterministic `blake2b`-based hash (`_loan_base_hash`) the modern-era
pipeline already uses for its own sampling — so downsampling preserves each cell's observed event
rate in expectation without a second RNG convention entering the repo. Checkpointed per input file
(same discipline as `count_prepay_events_pre2013.py`, after a prior long scan there was lost to a
SLURM timeout holding state in memory).

**Validated on a 2-quarter subset before the full run.** `scripts/diag/test_cell_grid_sample_pre2013.py`
(job 17959384, 2002Q3+2011Q2, 1,033,118 loans): cell assignment matches an independent replica of
the counting logic exactly (same cells, same per-cell loan/event counts), and loan selection is
deterministic — identical across two independent runs and after a fresh re-scan. All checks
passed. `scripts/diag/check_cell_grid_downsample_deviation_pre2013.py` (job 17960864, same 2
files) then checked how far the actual selected-loan event count lands from each downsampled
cell's 5000-event target: of 17 cells that trigger downsampling in this 2-file population, **max
deviation is 1.62%, mean 0.53%, zero cells exceed 10% or 15%** — the `ceil()`-based budget formula
tracks its target closely in practice, not just in the algebra.

**Full 52-quarter run** (job 17961041, 1h21m): **1,677,060 of 29,130,522 loans selected (5.8%)**,
729 (vintage_quarter, coupon) cells touched, 257 downsampled.

**Why 1.68M is reasonable against the spec's original "2-4M loans" ballpark.** That figure, in
`count_prepay_events_pre2013.py`'s own docstring, was written *before* the actual event
distribution had been measured — explicitly a pre-measurement guess ("the actual event
distribution has to be measured first rather than the oversampling weights guessed"). The measured
reality: most of the 729 cells don't reach the 5000-event cap at all (a large share sit below a
~1000-event floor and are kept whole, contributing their full but individually small loan counts;
`build_cell_grid_sample_pre2013.py`'s own docstring puts 400 of 729 cells below that floor holding
only 15,583 loans combined), and only 257 cells need downsampling down toward the cap. A rough
upper-bound guess made before measurement landing above the number the actual, measured
distribution produces is the expected direction of error, not a discrepancy to explain away.

### Builder extension: three patches to `prepare_sequences_multiobs_zbc.py`

1. **Cutoff-based file filter.** `RELEVANT_VINTAGES` skips opening any vintage file whose
   acquisition quarter provably starts after the build's cutoff (`_vintage_quarter_start_yyyymm`).
   Margin verified empirically, not assumed: an independent awk single-pass min-scan of
   `monthly_reporting_period` against 12 boundary files (2000Q1, 2002Q3/Q4, 2003Q1/Q2, 2011Q2/Q3,
   2012Q4, 2013Q1, 2019Q4, 2020Q1/Q4) confirms **delta=0 at all 12** — no file holds a row before
   its nominal acquisition-quarter start, so no safety margin is needed on top of it.
2. **Per-vintage checkpointing.** Pass 1 (loan discovery) and Pass 2 (scaler fit) each checkpoint
   after every vintage and resume from it on restart, same discipline as the cell-grid sampler
   above. **Interrupt-tested, not just read**: a from-scratch Pass 1+2 build on 4 small pre-2013
   vintages was killed mid-Pass-1 (resume printed `RESUME Pass 1 from vintage 2/4` and picked up
   correctly) and mid-Pass-2 (resume printed `RESUME Pass 2 from vintage 1/4`). The resumed run's
   final state — train/test loan-id split and fitted scaler — was compared against an uninterrupted
   baseline over the same 4 vintages: **splits byte-identical, scaler `mean_`/`var_`/`scale_`/
   `n_samples_seen_` all exactly equal.**
3. **Categorical-encoding fix** — see the CP/U finding below; this is the more consequential of the
   three.

### `cutoff_2002` test build

Job 17980942 (`--cutoff_year 2002 --include_pre2013`, 7h04m, unattended overnight): **304,447
total loans** (Train 243,557 / Test 60,890), train shape `(906,877, 33, 9)` prepay=9.03%, test
shape `(227,078, 33, 9)` prepay=9.02%. **Caveat, not swept under the rug:** this build ran with the
CP/U category maps still unfixed (the fix landed afterward — see below), so its
`property_type_enc`/`loan_purpose_enc` feature values carry the same miscoding described next. It
is a valid test of the checkpointing/cutoff-filter machinery and of the sampler's loan selection,
not yet a "clean" data artifact — rebuilding it after the fix is open follow-up work, not done
here.

**CORRECTION (2026-09-24): the caveat above is stale — the rebuild described as follow-up was
done, and this section never got updated to say so.** Job 17980942's output directory was renamed
to `cutoff_2002_zbc_multiobs_f0.2_h1_hist_PRECPUFIX_STALE` and was never trained on. `cutoff_2002`
was rebuilt post-fix as job 18054077, landing at
`data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist`. Verified directly:
`scaler.pkl` in that directory has `property_type_enc` mean **0.275936** (post-fix), vs. **0.252335**
in the `_PRECPUFIX_STALE` copy (pre-fix). Both `cutoff_2002` seed checkpoints — seed 42 (job
18097822) and seed 7 (job 18138546) — trained on the post-fix rebuild:
`train_hazard_multiobs.py`'s `SEQ_DIR` path resolution (cutoff_year=2002, `f0.2`, `h1`, no `_L`
suffix since `max_seq_len`=33 is default, `_hist` since `--include_pre2013`) resolves to
`cutoff_2002_zbc_multiobs_f0.2_h1_hist`, not the `_STALE` directory, for both training sbatch
files. Anything scored against these checkpoints must apply the CP/U fix, using the same category
maps as the rebuild.

### A categorical-encoding bug the extension surfaced, affecting previously-reported modern-era results

Building the historical extension required, for the first time, counting *unmapped* categorical
codes explicitly instead of letting them fall back to 0 silently. That check (added as part of the
patches above) immediately found that `property_type=='CP'` (Co-op) and `loan_purpose=='U'`
(Unknown) — both real, documented Fannie Mae codes — were falling back to code 0, landing
indistinguishably in the same bucket as a genuine Single-Family/Purchase row, in **every
modern-era (2013+) build this pipeline has ever produced.**

Two independent measurements, kept on separate denominators rather than conflated: **(1)** job
17974041 (a cutoff_2020 rebuild that ran with the new warning in place but before the fix), raw
loan-month rows in the full panel — CP present in every one of the 32 modern (2013Q1-2020Q4)
vintages, from 5,300 to 318,574 rows per vintage; U present sparsely in 3 of them (96/82/24 rows,
2013Q1/2014Q1/2014Q4). **(2)** job 18022825 (a cutoff_2020 regression check run after the fix), the
*sampled* `fixed_fraction` training population — CP is 0.49% of train / 0.52% of test observations
(55,028 / 14,652); U drew zero sampled observations in that population, consistent with (1)'s much
sparser raw count simply not being drawn by the sampler, not with U being absent from the modern
era.

**This is small enough to be very unlikely to have changed any prior conclusion, but it is real
miscoding present in data behind results already reported** in this document — the window-length
(L33 vs. L1) comparison, the no-history control, and the n=3 seed-replication run (Sep 10-13
sections above) all trained on data with this miscoding present. Not retracted, since the scale
argues against it mattering; flagged plainly rather than silently absorbed into the historical
extension. Full incident detail, including the reused-scaler interaction this fix creates (the
Sep 19 regression check's scaler was fit *before* the fix existed, so CP's new code-4 value is an
extrapolation outside what that scaler was fit on — not wrong, but not natively calibrated either,
and a candidate for a fresh scaler fit whenever the modern-era split/scaler is next rebuilt from
scratch) is in `docs/mistakes_and_lessons.md`.

## cutoff_2002: seed replication, a calibration-check methodology correction, and an S-curve check ruling out the Phase 15 failure mode (Sep 21, 2026)

**CORRECTION (Sep 23, 2026):** the 1.16x/1.23x pooled-monthly dispersion figures in the table below
are wrong. `forecast_raw_h1`/`forecast_platt_h1` were computed as a plain, unweighted `.mean()` over
the sampled test rows, then compared against `ipw_debiased_rate_h1`, which is IPW-weighted
(`sum(label/incl_prob)/sum(1/incl_prob)`) — one side reweighted, the other not. With BOTH sides
IPW-weighted (`scripts/diag/ipw_consistent_gap.py`, jobs 18312946/18312947), the pooled gap drops to
**1.0269x / logit +0.0269** (cutoff_2020) and **1.0696x / logit +0.0693** (cutoff_2002, seed 42). A
seed-7 replication of cutoff_2002 gives **1.0212x / logit +0.0215** — noticeably smaller than the
seed-42 figure, so this residual is not yet established as seed-stable. For cutoff_2020, the
sample's own IPW-weighted realized rate (0.012051) matches the external census rate (0.012078); the
model's IPW-weighted mean prediction is 0.012376, i.e. **1.0246x census**. Grouping by row
characteristics rather than by outcome (months before cutoff, which is not one of the model's
features, and loan_age, which is): rows 1 and 2-3 months before the cutoff are **not** over-predicted
in either cutoff (if anything, slightly under-predicted) — so there is no sign of an effect
concentrated at the cutoff edge. None of the model's 9 features encodes calendar time or distance to
the cutoff, so this is weak evidence on its own; the stronger evidence is that IPW already assigns
censored terminal rows their true inclusion probability and the model's weighted
mean prediction is within about 2.5% of the census rate. The residual shows up on the training set
almost as strongly as on test (cutoff_2020: train +0.0260 vs test +0.0269; cutoff_2002: train
+0.0604 vs test +0.0693, seed 42), i.e. it is not purely test-set sampling noise. Full numbers:
`scripts/diag/ipw_consistent_gap.py` (jobs 18312946/18312947) and
`scripts/diag/ipw_gap_feature_breakdown.py`.

### Seed-7 replication

`outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed7/results.json` (completed
2026-09-20 22:14 EDT): `best_auc=0.7745741145291616` (seed=7) against the seed=42 baseline's
`0.7743897241456773` — spread **0.000184 (~0.0002)**. For comparison, the existing n=3
seed-replication for cutoff_2020's L33 (`mistakes_and_lessons.md`, seeds 42/7/123: best_auc
0.71638/0.71658/0.71696) has spread **0.000576** — cutoff_2002's spread is **~3.1x tighter** than
that. Caveat, not smoothed over: this is n=2 (42, 7), not the n=3 that grounded the L33-vs-L1
noise-floor correction — a weaker replication than that comparison, not an equivalent one, and not
yet evidence about cross-seed *calibration* stability (Platt a/b were not compared across seeds).

### Calibration-check methodology: which Platt does the reporting pipeline actually use

A same-population-holdout calibration check was run against cutoff_2002's own internal Platt fit
(`results.json`: `platt_a=21.7239, platt_b=-3.2331`, seed=42) — same style as the check that caught
the `ipw_buggy` direction bug. Per-coupon, that Platt-calibrated forecast overshot an IPW-debiased
realized rate (Horvitz-Thompson, `1/incl_prob` on the run's own stored `incl_prob`) by **3.0x-6.0x**
depending on coupon, and the same check against cutoff_2020's main non-buggy `..._ipw_f0.2_L33`
result showed the same pattern, **3.5x-5.6x**. Before reporting that as a finding, traced whether
either Platt fit is actually what the CPR-forecast-reporting pipeline consumes:

- **`train_hazard_multiobs.py`'s internal Platt** (`platt_calibrate(best_scores, test_labels)`,
  line 396) fits log-loss with no `incl_prob` argument at all — directly against the **as-sampled**
  test labels, which are inflated above the true population rate because the mandatory/terminal
  draw is always included (`incl_prob=1.0`) regardless of the fixed-fraction downsampling applied
  to the rest of the pool. By construction, mean(Platt score) ≈ mean(raw sampled label) — the 3-6x
  gap above is a property of that internal fit, not evidence about the model itself.
- **`forecast_matched_population_cpr.py`** (the script whose output this document actually reports,
  e.g. the Sep 6-7 "dispersion 1.139..0.717" result) calls `aggregate(ids_multi, h_multi, ...,
  logit_offset=off_multi)` with `h_multi` = the model's **raw** `sigmoid(logit)` from
  `score_multiobs_model()` — no Platt applied anywhere on that path — and `off_multi=0.0` hardcoded,
  per that module's own `MULTIOBS_CAVEAT`: no King-Zeng-style scalar shift is theoretically
  justified for multiobs's per-observation `incl_prob` design, and none is applied. `aggregate()`
  itself (`forecast_rolling_cpr.py:481`) has no Platt parameter at all, only the optional additive
  `logit_offset`.
- A second, unrelated Platt-shaped pair, **a=0.4559, b=-3.1376** ("cohort_CPR_calibration_for_forecast_leg"),
  lives in `config/hazard_calibration_cpr_forecast.json` and in
  `outputs/rolling/cutoff_{2020,2021,2022,2023}/hazard_calibration.json` — all five files are
  **byte-identical** (`md5sum` confirmed) and written in the same second (2026-07-03 11:50:13), i.e.
  one shared constant, not four independent per-model fits. Traced to
  `scripts/diag/recalibrate_forecast_cpr.py`: fit by minimizing squared error between annualized
  forecast CPR and `realized_cpr_by_coupon_v6.csv` over a 2022-06..2023-12 trough window, scoring
  the separate **"production"** model (`outputs/hazard_best.pt`) on synthetic representative-loan
  paths via `return_per_timestep=True` — a different model, different (synthetic) population, and a
  different scoring convention than multiobs. `grep -rl "multiobs" scripts/stage2_*.py
  scripts/model_hedge_krd.py` returns nothing: this pipeline never touches multiobs.

**Neither Platt is what a multiobs CPR forecast actually reports.** Re-ran the check using the raw
score (`forecast_raw_h1`, matching `aggregate()`'s real `logit_offset=0.0` path) against the same
IPW-debiased realized rate — used here strictly as a diagnostic to identify the right calibration
target, never written to any `results.json`/config/CSV as a reported value:

| run | coupon range used | dispersion (raw/IPW-debiased) | pooled monthly ratio | pooled annualized ratio |
|---|---|---|---|---|
| cutoff_2002 (seed 42) | 5.5-9.5 (this cohort's own note-rate range — 2000-2002 originations, not 2.0-5.0) | 1.10x-1.37x | 1.23x | 1.19x |
| cutoff_2020 (`..._ipw_f0.2_L33`) | 2.0-5.0 | 0.97x-1.20x | 1.16x | 1.14x |

Both are reasonably calibrated by the metric that actually matches the reporting pipeline —
nothing like the 3-6x gap the internal-Platt check produced. This is an **in-sample** check (the
model's own held-out test observations, all within the calendar-truncated window); it is not, and
should not be conflated with, the still-separate, still-undone "predict the 2003 wave"
out-of-sample forward test described above (`n_test` clarification). Scripts:
`scripts/diag/calibration_check_cutoff_2002.py`, `scripts/diag/calibration_check_cutoff_2020_ipw_L33.py`.

### Incentive S-curve check: ruling out the Phase 15 failure mode

**CORRECTION (Sep 23, 2026):** `model_raw_h_t` below (the "model's raw hazard" column) was
originally computed as a plain, unweighted `.mean()` over sampled test rows, in the same per-bin
table as the IPW-weighted `ipw_debiased_rate_h1` column — the same weighting mismatch as the
calibration-check methodology correction above. Fixed and reran (job 18376180,
`scripts/diag/incentive_scurve_check_cutoff_2002.py`): with `model_raw_h_t` IPW-weighted the same
way, the "~29x" rise becomes **~27.3x** (0.001905 at incentive -2.0 to -1.0, up to 0.051952 at +2.0
to +3.0; full range 0.00109-0.05195, vs the unweighted 0.00109-0.05505 below). The shape conclusion
is unchanged: still 3 of 12 bin-to-bin steps are non-monotone (one is now the thin-tail (-4.0,-3.0]
to (-3.0,-2.0] step instead of (-3.0,-2.0] to (-2.0,-1.0] — both n<100 — plus the same two
top-incentive burnout bins in the wrinkle noted below), so this does not change the "not a
reoccurrence of Phase 15's collapse" conclusion. Comparing the IPW-weighted model hazard to the
IPW-weighted realized rate bin by bin, the model runs about 1.26x realized at incentive -2.0 to -1.0
but about 1.04x at +2.0 to +3.0, and realized rises 33.1x across those bins (0.001514 to 0.050110)
versus 27.3x for the model. In-sample and single seed, so not a finding, but it is the same direction
as the flatness documented in Phase 23, only milder.

Phase 15 (above) found that a model trained on a window with no in-window refi boom
(2013-2019) learns a collapsed/inverted S-curve — hazard pinned near 0 exactly where prepayment
should peak, because the training regime never showed the model a real incentive-driven response.
Checked the same premise for cutoff_2002 directly on real data
(`scripts/diag/incentive_scurve_check_cutoff_2002.py`, job 18225125) rather than assuming the
expanding-window framing rules it out by construction:

- **PMMS moved substantially inside the training window itself**: 8.515% (early 2000) down to
  6.048% (Dec 2002), a **2.468pp** decline entirely within cutoff_2002's own calendar-truncated
  window — the opposite condition from Phase 15's flat, boom-free 2013-2019 window.
  77.34% of test observations are in-the-money (mean incentive +0.75pp, p95 +2.39pp).
- **The model's raw hazard rises ~29x across the informative incentive range** on real test data:
  0.00189 (incentive -2.0 to -1.0) up to a peak 0.05505 (incentive +2.0 to +3.0), tracking both the
  raw-sampled and IPW-debiased realized rates rising in step over the same range — the opposite
  shape from Phase 15's collapse-to-zero failure.
- **One honest wrinkle, not smoothed over**: the top two incentive bins (+3.0 to +6.0, only 2,145 of
  227,078 obs) show hazard falling rather than continuing to rise — non-monotone, but on thin tail
  data and consistent with ordinary burnout (which multiobs's multi-age sampling design exists to
  capture), not a re-occurrence of Phase 15's every-positive-incentive-bin collapse.

This addresses curve *shape* only; it does not bear on the calibration *level* question above —
those are separate axes and are not meant to offset one another.

## Fixed-fraction sampling rates (a_eff / b_eff), a sequence-alignment bug, and the cutoff_2002 2003 forward test (Sep 24-25, 2026)

### a_eff / b_eff: the fixed-fraction 10% loan subsample's event and exposure sampling rates

`cutoff_2020`'s multiobs training data is drawn from a uniform 10% loan subsample
(`prepare_sequences_rolling_zbc.py`'s `--sample_frac`, applied at Pass-1 loan-ID discovery; the
multiobs builder's own `reuse_from` inherits this population unchanged). After removing the uniform
10% loan subsample (scaling by the sampling ratio `r`), events enter at a_eff = 0.9985 and non-events
at b_eff = 0.2277, above the nominal 0.2 because censored loans' last months are drawn with
probability 1, and per-row IPW accounts for this.

- **Census population loans:** 18,709,686 — `outputs/census_panel_baseline_cutoff_2020.json` →
  `overall.n_loans`.
- **Sample population loans:** 1,870,909 — `logs/prep_trail_2020_16697151.log:79`
  (`Total loans: 1,870,909 | Prepay rate: 38.19%`, end of Pass 1); confirmed independently by
  counting unique IDs in `data/sequences_rolling/cutoff_2020_zbc_trail/{train,test}_loan_ids.npy`
  (1,496,727 + 374,182, zero overlap).
- **r = 1,870,909 / 18,709,686 = 0.099997** — within 0.003% of the nominal `--sample_frac 0.1`.
- **Census prepay events:** 7,155,149 — same JSON, `overall.n_prepay_events`.
- **Census non-event loan-months:** 585,243,056 — `overall.n_eligible_loan_months` (592,398,205)
  minus `overall.n_prepay_events` (7,155,149), same JSON.
- **Sample events / non-events:** 714,442 / 13,324,780 — summed `train_labels.npy` +
  `test_labels.npy` in `data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP/`
  (`y.sum()` / `(y==0).sum()`).
- **a_eff = (714,442 / 7,155,149) / r = 0.9985** — the sample's event rate scales with `r` almost
  exactly as expected against the census population (0.15% off 1.0).
- **b_eff = (13,324,780 / 585,243,056) / r = 0.2277** — the sample's non-event (exposure)
  loan-month rate does **not** scale the same way; it is about 4.4x under what pure uniform
  loan-sampling would predict. This is expected, not a defect: the multiobs builder does not draw
  every eligible loan-month per sampled loan, only a fixed fraction of the pool plus the mandatory
  terminal draw (`--sampling_mode fixed_fraction --frac_draws 0.2`, `run_multiobs_2020_f0.2_L33.sbatch`),
  so exposure rows are deliberately thinned relative to a full loan-month census while events are not
  (the terminal/mandatory draw is always included). `a_eff` and `b_eff` are answering different
  questions — event-rate representativeness vs. row-count representativeness — and are not expected
  to agree.
- **log(a_eff / b_eff) = 1.478.**
- **No constant offset exists in the fixed-fraction pipeline.** The correction for the fixed-fraction
  design is the per-row IPW loss weight (`ipw_weight(incl_prob) = 1/incl_prob`, documented in the Sep
  6-7 section above) applied during training, not any inference-time additive/multiplicative offset;
  `forecast_matched_population_cpr.py`'s `aggregate()` call already runs with `off_multi=0.0`
  hardcoded per its `MULTIOBS_CAVEAT` (documented in the Sep 21 section above). `--pos_ratio` is a
  separate, unrelated batch-composition knob (`train_hazard_multiobs.py`, controls the
  positive/negative draw mix within a training batch) and has no offset role either.

### Sequence window alignment: right-aligned checkpoints scored on left-aligned windows

`measure_alignment_effect.py` (job 18479425) scored the `cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33`
seed-42 checkpoint (`hazard_best.pt`, AUC 0.7163846603462104) two ways on the loan intersection of
its LEFT-aligned test set (`TRAIL_SEQ_DIR`, the population every Sep 5 / Sep 6-7 / Sep 8 forecast
above actually scored) and a freshly built RIGHT-aligned control (a single Dec-2020-anchored forward
window, this checkpoint's own scaler and pre-fix CP/U maps): intersection 230,670 loans (left
n=374,182, right n=230,670).

- **Full history (all 33 months populated, no left-padding):** 100,127 loans. The two alignments
  agree **within 1e-4 relative** on every one of them — expected, since a completely full trailing
  window is identical under either anchoring convention. `mean_h_left=0.03574 mean_h_right=0.03574
  mean_ratio=1.0000 median_ratio=1.0000`.
- **Partial history (<33 months, left-padded under the left alignment):** 130,543 loans —
  **56.59%** of the 230,670-loan intersection. The two alignments diverge materially here:
  `mean_h_left=0.02707 mean_h_right=0.02386 mean_ratio=1.1765 median_ratio=0.8092`.
- **`alignment_effect_vs_realized.py`** (reuses the cached combined pass; run fresh this session)
  added realized CY2021 prepayment (count-weighted, same `active_set`/`prepaid_set` definition as
  `forecast_rolling_cpr.read_coupon_and_realized`/`aggregate`) to the same split. Partial-history
  per-coupon table:

  | coupon | n_loans | cpr_left | cpr_right | realized_cpr | abs_err_left | abs_err_right | closer_to_realized |
  |---|---|---|---|---|---|---|---|
  | 1.0 | 21 | 26.4507 | 1.7587 | 9.5238 | 16.9269 | 7.7651 | right |
  | 1.5 | 785 | 30.1716 | 1.9942 | 5.8599 | 24.3117 | 3.8657 | right |
  | 2.0 | 19,425 | 32.0965 | 5.0421 | 9.6885 | 22.4080 | 4.6464 | right |
  | 2.5 | 30,720 | 31.5192 | 10.9418 | 13.4603 | 18.0589 | 2.5185 | right |
  | 3.0 | 38,420 | 25.1516 | 26.1685 | 25.4945 | 0.3429 | 0.6740 | left |
  | 3.5 | 15,115 | 23.2310 | 36.7497 | 35.8187 | 12.5877 | 0.9310 | right |
  | 4.0 | 14,539 | 24.7639 | 38.8819 | 39.1980 | 14.4341 | 0.3161 | right |
  | 4.5 | 6,363 | 24.7070 | 37.1473 | 38.9125 | 14.2055 | 1.7652 | right |
  | 5.0 | 4,333 | 24.5969 | 33.4256 | 36.9259 | 12.3290 | 3.5003 | right |
  | 5.5 | 665 | 24.4028 | 29.9007 | 36.2406 | 11.8378 | 6.3399 | right |
  | 6.0 | 157 | 25.2117 | 29.5026 | 36.3057 | 11.0940 | 6.8031 | right |

  **Right-aligned is closer to realized at 10 of 11 partial-history coupons.** The single exception
  is coupon 3.0, where left is closer by a narrow margin (0.3429 vs. 0.6740 absolute error) — not a
  systematic counter-pattern, given every other coupon favors right by a wide margin.
- **Practical implication:** every forecast reported in the Sep 5 / Sep 6-7 / Sep 8 sections above
  scored a right-aligned (single Dec-cutoff-window-trained) multiobs checkpoint against the
  left-aligned `TRAIL_SEQ_DIR` test set — the wrong alignment for the 56.59% of that population with
  partial history. See the CORRECTION notes at the top of those three sections for the like-for-like
  corrected numbers.

Artifacts: `scripts/measure_alignment_effect.py` (job 18479425),
`scripts/diag/alignment_effect_vs_realized.py`, `scripts/score_matched_population_right_aligned.py`
(jobs 18486581/18486582, the seedcheck_a like-for-like correction cited in the three CORRECTION notes
above). Outputs: `outputs/rolling/alignment_effect_cutoff_2020_seed42/alignment_effect_loans.csv`,
`outputs/rolling/matched_population_right_aligned_seedcheck_a/left_vs_right_comparison.csv`.

### The frozen Dec-window 2003 test, and why frozen scoring was the wrong design

Before building the one-step-ahead pipeline below, the first `cutoff_2002` → 2003 forward test
scored one frozen Dec-2002 window/checkpoint against every forecast month (Dec 2002 - Nov 2003)
without updating each loan's contemporaneous incentive month to month. **PMMS moved from 6.0475%
(Dec 2002) to 5.23% (Jun 2003)** —
`outputs/rolling/dec_window_cutoff_2002_build/pmms_series_2002_2003.csv` — an 82bp decline over
six months, entirely inside the window the frozen design holds fixed. A design that freezes the
reference rate (and therefore each loan's `incentive_at_ref`) at the Dec-2002 cutoff cannot capture
that shift, which is exactly the signal a rate-path forecast needs; the advisor's original Aug 30
design specifies contemporaneous incentive each month for this reason (see the mistakes-log entry
added this session). The one-step-ahead redesign below restores that: each of the 12 forecast
months gets its own one-month-forward window and its own contemporaneous incentive.

Headline seed-42 pooled-level results from the frozen design, for the record (source:
`.claude_tmp/report_2003_and_control.txt`; not the corrected design — see one-step-ahead below):

| cohort | count unweighted fcst | count unweighted realized | count cell-reweighted fcst | count cell-reweighted realized | UPB unweighted fcst | UPB unweighted realized | UPB cell-reweighted fcst | UPB cell-reweighted realized | UPB n_excluded |
|---|---|---|---|---|---|---|---|---|---|
| cutoff_2002 seed42 | 31.4832 | 43.1878 | 29.6619 | 45.7531 | 41.8719 | 53.0965 | 44.2222 | 58.6476 | 10,385 (25.85%) |
| cutoff_2020 control seed42 | 27.6829 | 26.3528 | 27.6829 | 26.3528 | 30.4542 | 28.3103 | 30.4542 | 28.3103 | 2 (0.00%) |

Dispersion (PRIMARY = max/min), count- and UPB-weighted, frozen design:

| cohort | count fcst | count realized | count ratio | UPB fcst | UPB realized | UPB ratio |
|---|---|---|---|---|---|---|
| cutoff_2002 seed42 | 8.1118 | 2.6046 | 3.1144 | 5.5673 | 2.4091 | 2.3110 |
| cutoff_2020 control seed42 | 5.4052 | 3.1799 | 1.6998 | 6.4347 | 3.5944 | 1.7902 |

Incentive-bin slope, frozen design, seed42: `cutoff_2002` forecast_slope 10.1832 / realized_slope
2.6281 / ratio 3.8748 (lowest_bin `(-1.0, -0.5]` n=3,373, highest_bin `(2.0, 3.0]` n=4,273);
`cutoff_2020` control forecast_slope 5.5753 / realized_slope 3.3468 / ratio 1.6659 (lowest_bin
`(-0.5, 0.0]` n=13,405, highest_bin `(3.0, 4.0]` n=1,403). Note the two cohorts' auto-selected
endpoint bins are entirely different pairs — see the caveat at the end of this section.

### One-step-ahead forecasts: cutoff_2002 → 2003 and the cutoff_2020 → 2021 control

One window per loan per forecast month, contemporaneous incentive each month, replacing the frozen
design above. `cutoff_2002`: Dec 2002 - Nov 2003, seeds 42/7. `cutoff_2020` control: Dec 2020 - Nov
2021, seeds 42/7/123. All five runs' Dec-month consistency checks against the frozen `dec_window_scores`
passed (max abs diff 3.7e-09 to 7.4e-09, all < 1e-6). Source: `.claude_tmp/report_rolling_2003_and_control.txt`.

**Pooled predicted/realized, per seed:**

| cohort | seed | count pred monthly | count real monthly | count pred ann | count real ann | UPB pred monthly | UPB real monthly | UPB pred ann | UPB real ann | UPB n_excluded |
|---|---|---|---|---|---|---|---|---|---|---|
| cutoff_2002 | 42 | 0.04230 | 0.04629 | 0.4047 | 0.4338 | 0.05269 | 0.05832 | 0.4777 | 0.5138 | 36,475 |
| cutoff_2002 | 7 | 0.03896 | 0.04629 | 0.3793 | 0.4338 | 0.04874 | 0.05832 | 0.4510 | 0.5138 | 36,475 |
| cutoff_2020 control | 42 | 0.02348 | 0.02536 | 0.2481 | 0.2652 | 0.02592 | 0.02761 | 0.2704 | 0.2854 | 5 |
| cutoff_2020 control | 7 | 0.02299 | 0.02536 | 0.2435 | 0.2652 | 0.02606 | 0.02761 | 0.2716 | 0.2854 | 5 |
| cutoff_2020 control | 123 | 0.01997 | 0.02536 | 0.2150 | 0.2652 | 0.02216 | 0.02761 | 0.2358 | 0.2854 | 5 |

**Dispersion — PRIMARY (max/min of the per-group monthly rate, n_groups=10) and SECONDARY (bounds
of the per-group predicted/realized ratio), count- and UPB-weighted:**

| cohort | seed | weight | PRIMARY pred | PRIMARY real | PRIMARY ratio | SECONDARY max | SECONDARY min |
|---|---|---|---|---|---|---|---|
| cutoff_2002 | 42 | count | 10.6269 | 8.4863 | 1.2523 | 1.1255 | 0.5760 |
| cutoff_2002 | 42 | UPB | 9.1159 | 6.4247 | 1.4189 | 1.1732 | 0.5612 |
| cutoff_2002 | 7 | count | 11.1193 | 8.4863 | 1.3103 | 1.0025 | 0.4913 |
| cutoff_2002 | 7 | UPB | 8.9388 | 6.4247 | 1.3913 | 1.0068 | 0.4916 |
| cutoff_2020 control | 42 | count | 10.7397 | 7.3843 | 1.4544 | 0.9762 | 0.6416 |
| cutoff_2020 control | 42 | UPB | 12.8834 | 8.4062 | 1.5326 | 0.9954 | 0.6234 |
| cutoff_2020 control | 7 | count | 9.5308 | 7.3843 | 1.2907 | 0.9703 | 0.7013 |
| cutoff_2020 control | 7 | UPB | 12.1029 | 8.4062 | 1.4397 | 1.0107 | 0.6586 |
| cutoff_2020 control | 123 | count | 10.8737 | 7.3843 | 1.4726 | 0.8417 | 0.5599 |
| cutoff_2020 control | 123 | UPB | 13.5819 | 8.4062 | 1.6157 | 0.8766 | 0.5242 |

**Incentive-bin pattern (count-weighted, seed42 shown in full; the other seeds within each cohort
show the same directional shape — predicted below realized at negative/low incentive, predicted
near or above realized at high incentive):**

`cutoff_2002` seed42 (monthly rates):

| bin | n | predicted | realized |
|---|---|---|---|
| (-4.0, -3.0] | 3 | 0.001622 | 0.000000 |
| (-3.0, -2.0] | 82 | 0.004004 | 0.000000 |
| (-2.0, -1.0] | 2,180 | 0.006977 | 0.010092 |
| (-1.0, -0.5] | 20,235 | 0.007702 | 0.010872 |
| (-0.5, 0.0] | 39,382 | 0.013480 | 0.019527 |
| (0.0, 0.5] | 56,818 | 0.026517 | 0.035746 |
| (0.5, 1.0] | 64,884 | 0.044121 | 0.051554 |
| (1.0, 1.5] | 58,596 | 0.059539 | 0.065039 |
| (1.5, 2.0] | 58,918 | 0.059136 | 0.057860 |
| (2.0, 3.0] | 57,271 | 0.054607 | 0.054006 |
| (3.0, 4.0] | 14,832 | 0.042288 | 0.040520 |
| (4.0, 6.0] | 1,643 | 0.034354 | 0.031041 |

`cutoff_2020` control seed42 (monthly rates):

| bin | n | predicted | realized |
|---|---|---|---|
| (-2.0, -1.0] | 2,082 | 0.003403 | 0.005764 |
| (-1.0, -0.5] | 48,862 | 0.004673 | 0.007409 |
| (-0.5, 0.0] | 348,329 | 0.008204 | 0.010777 |
| (0.0, 0.5] | 551,904 | 0.014473 | 0.016921 |
| (0.5, 1.0] | 588,640 | 0.027041 | 0.027699 |
| (1.0, 1.5] | 444,957 | 0.034529 | 0.036044 |
| (1.5, 2.0] | 273,254 | 0.034519 | 0.036329 |
| (2.0, 3.0] | 132,457 | 0.032441 | 0.036374 |
| (3.0, 4.0] | 6,786 | 0.028752 | 0.034335 |

Incentive-bin slope (predicted vs. realized), all five runs — realized slope is identical within
each cohort since the population and realized outcomes don't change across seeds, only the model:

| cohort | seed | predicted slope | realized slope |
|---|---|---|---|
| cutoff_2002 | 42 | 4.9236 | 3.0759 |
| cutoff_2002 | 7 | 3.4932 | 3.0759 |
| cutoff_2020 control | 42 | 8.4497 | 5.9572 |
| cutoff_2020 control | 7 | 7.6278 | 5.9572 |
| cutoff_2020 control | 123 | 8.9122 | 5.9572 |

**Month-by-month summary (seed42, representative — the population and consistency-check results
are identical across seeds within each cohort):**

`cutoff_2002` seed42 (12 months, Dec 2002 - Nov 2003):

| ref_month | predicted_rate_annualized | realized_rate_annualized | non_prepay_term_count | n_excluded_by_forward_gap |
|---|---|---|---|---|
| 200212 | 0.357060 | 0.345414 | 19 | 1 |
| 200301 | 0.369426 | 0.358316 | 13 | 0 |
| 200302 | 0.386858 | 0.402753 | 21 | 1 |
| 200303 | 0.409486 | 0.483833 | 20 | 0 |
| 200304 | 0.417018 | 0.456452 | 19 | 0 |
| 200305 | 0.452000 | 0.497271 | 19 | 1 |
| 200306 | 0.494281 | 0.578572 | 26 | 0 |
| 200307 | 0.475469 | 0.571210 | 17 | 0 |
| 200308 | 0.408955 | 0.466582 | 16 | 0 |
| 200309 | 0.366178 | 0.358846 | 14 | 0 |
| 200310 | 0.348988 | 0.286545 | 16 | 0 |
| 200311 | 0.347345 | 0.293667 | 25 | 0 |

`cutoff_2020` control seed42 (12 months, Dec 2020 - Nov 2021):

| ref_month | predicted_rate_annualized | realized_rate_annualized | non_prepay_term_count | n_excluded_by_forward_gap |
|---|---|---|---|---|
| 202012 | 0.297680 | 0.271484 | 9 | 0 |
| 202101 | 0.306439 | 0.283647 | 10 | 0 |
| 202102 | 0.294790 | 0.330907 | 6 | 0 |
| 202103 | 0.273758 | 0.279517 | 14 | 0 |
| 202104 | 0.248924 | 0.242081 | 9 | 0 |
| 202105 | 0.240689 | 0.259656 | 20 | 0 |
| 202106 | 0.226292 | 0.241051 | 8 | 0 |
| 202107 | 0.223010 | 0.275945 | 21 | 0 |
| 202108 | 0.224092 | 0.264435 | 7 | 0 |
| 202109 | 0.208552 | 0.255417 | 19 | 0 |
| 202110 | 0.195170 | 0.231105 | 10 | 0 |
| 202111 | 0.184001 | 0.221493 | 23 | 0 |

Artifacts: `scripts/score_rolling_one_step.py`, `scripts/report_multiobs_dec_window.py`,
`scripts/score_multiobs_dec_window.py`; sbatch files `run_rolling_onestep_2002_seed{42,7}.sbatch`,
`run_rolling_onestep_2020_control_seed{42,7,123}.sbatch`. Jobs 18493059/18493061 (`cutoff_2002`),
18493064/18493067/18493069 (`cutoff_2020` control). Full printed output for all five runs:
`.claude_tmp/report_rolling_2003_and_control.txt`.

### Caveat: the top/bottom slope metric uses different endpoint bins across runs

The incentive-bin slope reported throughout this section (and the frozen-design section above) is
computed between whichever two bins happen to be the lowest/highest with enough observations in each
run — not a fixed pair of bins. `cutoff_2002`'s frozen-design slope spans `(-1.0, -0.5]` to
`(2.0, 3.0]`; `cutoff_2020` control's frozen-design slope spans `(-0.5, 0.0]` to `(3.0, 4.0]` — a
different pair of bins entirely, because the two cohorts' coupon/incentive distributions differ. The
one-step-ahead tables above show the same asymmetry (`cutoff_2002` has usable bins from `(-4.0,
-3.0]` to `(4.0, 6.0]`; `cutoff_2020` control only from `(-2.0, -1.0]` to `(3.0, 4.0]`, since it has
no observations in the negative tail beyond -2.0). A slope number is therefore not comparable
across runs or cohorts by itself — compare the per-bin predicted/realized rates directly instead,
using the full bin tables above.

## Five-seed ensembles, a train/forecast consistency test, and a house-price / responsiveness check on the low-incentive gap — cutoff_2002 (2003) and cutoff_2020 control (2021) (Sep 26, 2026)

Extends the one-step-ahead forecasts above from 2/3 seeds per cohort to a full five-seed ensemble
(42, 7, 123, 1001, 2026 for `cutoff_2002`; same five for the `cutoff_2020` control), adds a
standing train/forecast consistency test with two negative controls, and tests whether zip3
house-price growth explains the low-incentive predicted-vs-realized gap found in both years.
Numbers below are read from saved CSVs under `outputs/rolling/ensemble_onestep_cutoff_{2002,2020_control}/`
and `.claude_tmp/*.log`; nothing here was recomputed for this write-up. Per standing instruction,
every cutoff_2020 computation this session ran under `srun`, never the login node.

### Two control seeds were trained on the wrong data; retrained and verified by content

Seeds 1001/2026 for the `cutoff_2020` control were first submitted against the plain
`cutoff_2020_zbc_multiobs_f0.2_h1` sequence dir, which had been silently overwritten in place on
Sep 19 with post-CP/U-fix category codes — five days after seeds 42/7/123 trained on the pre-fix
version. Caught by decoding `property_type_enc`'s content, not by directory name: the plain
(post-rewrite) dir has codes `{0,1,2,3,4}` (CP present), `..._GOLDEN_BACKUP` (pre-fix, frozen) has
only `{0,1,2,3}`. The two mismatched runs were renamed `..._POSTFIX_MISMATCH` and excluded; both
seeds were retrained pointed explicitly at `..._GOLDEN_BACKUP` (`best_auc` 0.71621 seed1001,
0.71641 seed2026 — see full detail in `docs/mistakes_and_lessons.md`). Full detail: see the
"Current state" data warning above.

### Train/forecast consistency test made a standing test, with two negative controls per cutoff

`scripts/tests/test_train_forecast_consistency.py` (srun job 18613794,
`.claude_tmp/consistency_srun.log`) — **all checks passed, both cases:**

- **`cutoff_2002_seed42_hist`** (random vintage 2002Q4): candidate-observation count, exact
  sequence/mask reconstruction (n=2,000, max diff 0.000e+00), and prediction match against the
  stored checkpoint (AUC 0.7743897241456773, max diff 0.000e+00) all passed. Negative control (b) —
  a frozen Dec-cutoff window's last timestep checked against June-of-forecast-year's actual
  features — **correctly disagreed** on 32,122/32,122 loan-months (max abs diff 2.331), confirming
  the test can catch a real misalignment, not just pass trivially.
- **`cutoff_2020_f0.2_seed42_GOLDEN_BACKUP`** (random vintage 2016Q3): same sequence/mask/prediction
  checks passed (checkpoint AUC 0.7163846603462104, max diff 0.000e+00). Negative control (a) — the
  left-aligned `TRAIL_SEQ_DIR` sequences checked against the correct right-aligned train
  sequences — **correctly disagreed** on 2,000/2,000 shared observations (max abs diff 3.603).
  Negative control (b) (same frozen-window check as above) **correctly disagreed** on
  200,871/200,871 loan-months (max abs diff 1.191).

Both negative controls exist specifically to catch the two historical bugs this project already
made once each (frozen-window scoring, left/right sequence-alignment mismatch) — the test is
useless if it can only pass, so both are designed to fail loudly if either bug reappears.

### Five-seed ensemble: pooled ratios and dispersion at min_n=1,000

**Pooled predicted/realized ratio, count- and UPB-weighted, ensemble and all five seeds:**

| cohort | seed | count ratio | UPB ratio |
|---|---|---|---|
| cutoff_2002 | ensemble | 0.8734 | 0.8644 |
| cutoff_2002 | 42 | 0.9138 | 0.9035 |
| cutoff_2002 | 7 | 0.8416 | 0.8357 |
| cutoff_2002 | 123 | 0.8013 | 0.7889 |
| cutoff_2002 | 1001 | 0.8964 | 0.8960 |
| cutoff_2002 | 2026 | 0.9142 | 0.8979 |
| cutoff_2020 control | ensemble | 0.8750 | 0.8955 |
| cutoff_2020 control | 42 | 0.9260 | 0.9390 |
| cutoff_2020 control | 7 | 0.9067 | 0.9440 |
| cutoff_2020 control | 123 | 0.7877 | 0.8027 |
| cutoff_2020 control | 1001 | 0.7999 | 0.8120 |
| cutoff_2020 control | 2026 | 0.9546 | 0.9797 |

**Dispersion at min_n=1,000** (`dispersion_stats_minn1000.csv`): `cutoff_2002` coupons used
[4.5-9.0 by 0.5] (10 groups); `cutoff_2020` control coupons used [1.5-6.0 by 0.5] (10 groups,
derived from `per_coupon_ratio_disagreement.csv` filtered to n≥1,000 — matches n_groups=10).

| cohort | model | PRIMARY predicted disp | PRIMARY realized disp | PRIMARY ratio |
|---|---|---|---|---|
| cutoff_2002 | ensemble | 11.525 | 8.486 | 1.358 |
| cutoff_2002 | seed42 | 10.627 | 8.486 | 1.252 |
| cutoff_2002 | seed7 | 11.119 | 8.486 | 1.310 |
| cutoff_2002 | seed123 | 10.066 | 8.486 | 1.186 |
| cutoff_2002 | seed1001 | 13.568 | 8.486 | 1.599 |
| cutoff_2002 | seed2026 | 13.151 | 8.486 | 1.550 |
| cutoff_2020 control | ensemble | 10.272 | 7.384 | 1.391 |
| cutoff_2020 control | seed42 | 10.740 | 7.384 | 1.454 |
| cutoff_2020 control | seed7 | 9.531 | 7.384 | 1.291 |
| cutoff_2020 control | seed123 | 10.874 | 7.384 | 1.473 |
| cutoff_2020 control | seed1001 | 10.350 | 7.384 | 1.402 |
| cutoff_2020 control | seed2026 | 10.871 | 7.384 | 1.472 |

Realized dispersion is identical across seeds within a cohort by construction (same realized
outcomes, only the model predictions vary). Seed disagreement (min/max/spread across the 5 seeds,
`disagreement_summary.csv`): pooled_ratio spread 0.167 count / 0.177 UPB for `cutoff_2020` control
(the wider of the two cohorts).

### House-price test: does zip3 house-price growth explain the low-incentive gap

**Why zip3 ZHVI, not state-level Case-Shiller.** `data/zhvi_zip3.csv` is already zip3-level and
already joined to every loan via the model's own `current_ltv` feature pipeline — Case-Shiller is
published at the MSA/national level, too coarse to construct the per-zip3 `x` this test needs, and
would collapse most of the within-state heterogeneity the test is trying to detect.

**Unit of x:** percent, not fraction — `scripts/house_price_test_2002.py:67`:
`return 100.0 * (now / prior - 1.0)`.

**Pooled low-incentive gap** (pooled ensemble realized monthly rate minus pooled ensemble predicted
monthly rate, over the same incentive≤0.5 loan-months the house-price test's WLS regression uses;
`scripts/house_price_gap_quintiles.py`, run directly for `cutoff_2002`, via `srun` job 18619302 for
the `cutoff_2020` control):

| cohort | weight | pooled predicted | pooled realized | gap | n |
|---|---|---|---|---|---|
| cutoff_2002 (2003) | count | 0.017522 | 0.026135 | 0.008613 | 104,267 |
| cutoff_2002 (2003) | upb | 0.023258 | 0.034501 | 0.011243 | 82,782 |
| cutoff_2020 control (2021) | count | 0.010939 | 0.014171 | 0.003232 | 945,306 |
| cutoff_2020 control (2021) | upb | 0.011345 | 0.014564 | 0.003219 | 945,302 |

**x distribution, low-incentive group** (per-zip3 mean trailing-12mo ZHVI growth, %):

| cohort | weight | n zip3s | min | 25th | median | 75th | max | IQR |
|---|---|---|---|---|---|---|---|---|
| cutoff_2002 (2003) | count | 342 | -4.91 | 3.31 | 6.08 | 12.61 | 43.97 | 9.30 |
| cutoff_2002 (2003) | upb | 301 | -4.66 | 3.34 | 6.71 | 13.18 | 59.75 | 9.84 |
| cutoff_2020 control (2021) | count/upb | 758 | -2.21 | 11.63 | 14.54 | 16.98 | 37.89 | 5.34 |

**Quintile tables** (zip3s split into 5 equal-count groups by x; count-weighted shown, UPB-weighted
in the saved CSV `house_price_zip3_quintiles.csv`):

`cutoff_2002` low-incentive:

| quintile | x range | n zip3s | n loan-months | predicted | realized | gap |
|---|---|---|---|---|---|---|
| 1 (lowest x) | -4.91 to 2.99 | 69 | 18,785 | 0.016803 | 0.023210 | 0.006407 |
| 2 | 3.00 to 4.76 | 68 | 19,722 | 0.016278 | 0.021803 | 0.005525 |
| 3 | 4.96 to 8.27 | 68 | 17,507 | 0.016294 | 0.029074 | 0.012780 |
| 4 | 8.31 to 13.86 | 68 | 20,851 | 0.017886 | 0.026953 | 0.009067 |
| 5 (highest x) | 13.89 to 43.97 | 69 | 27,402 | 0.019416 | 0.028757 | 0.009341 |

`cutoff_2002` placebo:

| quintile | x range | n zip3s | n loan-months | predicted | realized | gap |
|---|---|---|---|---|---|---|
| 1 | -6.19 to 2.93 | 75 | 22,610 | 0.046478 | 0.040823 | -0.005655 |
| 2 | 2.93 to 4.31 | 75 | 23,755 | 0.049707 | 0.047822 | -0.001886 |
| 3 | 4.31 to 7.52 | 75 | 23,610 | 0.047075 | 0.049089 | 0.002014 |
| 4 | 7.57 to 13.18 | 75 | 23,233 | 0.058353 | 0.061507 | 0.003155 |
| 5 | 13.20 to 38.86 | 75 | 25,274 | 0.064803 | 0.068133 | 0.003331 |

Not monotone in the low-incentive group (quintile 3, not 5, has the largest gap) and quintile 1's
gap (0.006407) is not near zero — 74% of the overall 0.008613 gap. The placebo group, which
shouldn't show an incentive-driven relationship, itself shows a monotone gap rising with x — a
wrinkle, not a clean confirmation.

`cutoff_2020` control (2021) low-incentive:

| quintile | x range | n zip3s | n loan-months | predicted | realized | gap |
|---|---|---|---|---|---|---|
| 1 (lowest x) | -2.21 to 10.70 | 152 | 145,617 | 0.011097 | 0.012993 | 0.001896 |
| 2 | 10.71 to 13.56 | 151 | 206,258 | 0.010652 | 0.013086 | 0.002434 |
| 3 | 13.56 to 15.42 | 152 | 195,411 | 0.011141 | 0.014713 | 0.003571 |
| 4 | 15.44 to 17.63 | 151 | 199,146 | 0.010985 | 0.014803 | 0.003818 |
| 5 (highest x) | 17.63 to 37.89 | 152 | 198,874 | 0.010876 | 0.014994 | 0.004118 |

`cutoff_2020` control (2021) placebo:

| quintile | x range | n zip3s | n loan-months | predicted | realized | gap |
|---|---|---|---|---|---|---|
| 1 (lowest x) | -3.33 to 10.76 | 132 | 65,799 | 0.030535 | 0.031201 | 0.000666 |
| 2 | 10.79 to 13.22 | 131 | 90,171 | 0.031715 | 0.036264 | 0.004549 |
| 3 | 13.26 to 14.98 | 131 | 79,640 | 0.031214 | 0.036226 | 0.005012 |
| 4 | 14.98 to 16.80 | 131 | 83,338 | 0.030707 | 0.035662 | 0.004955 |
| 5 (highest x) | 16.81 to 35.09 | 131 | 83,011 | 0.033648 | 0.041741 | 0.008094 |

Both `cutoff_2020` control tables are monotone in x with the lowest-x quintile's gap smallest
(low-incentive: 0.0019, ~59% of the overall 0.003232 gap; placebo: 0.0007, close to zero) — a
cleaner pattern than `cutoff_2002` shows.

**Slope × IQR(x) as a share of the low-incentive gap**, with a range of (slope ± 2×cluster_se) ×
IQR / gap (clustering on zip3's first digit, 10 clusters):

| cohort | weight | slope | cluster_se | slope×IQR | share of gap | range |
|---|---|---|---|---|---|---|
| cutoff_2002 (2003) | count | 0.0001880 | 0.0002167 | 0.001748 | 20.3% | [-26.5%, 67.1%] |
| cutoff_2002 (2003) | upb | 0.0002031 | 0.0002717 | 0.001999 | 17.8% | [-29.8%, 65.3%] |
| cutoff_2020 control (2021) | count | 0.0001659 | 0.0000797 | 0.000887 | 27.4% | [1.1%, 53.8%] |
| cutoff_2020 control (2021) | upb | 0.0002421 | 0.0001022 | 0.001294 | 40.2% | [6.3%, 74.2%] |

Point estimates put house-price growth at roughly a fifth to two-fifths of the low-incentive gap,
but the interval straddles zero for `cutoff_2002` — consistent with `cutoff_2002`'s clustered
t-stats not clearing significance (below). **House-price growth does not explain most of the 2003
gap** either way, even at the high end of the range.

**Clustered t-stats against the correct critical value.** With 10 clusters (9 df), the two-sided
5% critical value is **t₉,0.975 ≈ 2.262**, not 1.96:

| cohort | group | weight | cluster_t | clears 2.262 |
|---|---|---|---|---|
| cutoff_2002 | low_incentive | count | 0.867 | No |
| cutoff_2002 | low_incentive | upb | 0.748 | No |
| cutoff_2002 | placebo | count | 1.649 | No |
| cutoff_2002 | placebo | upb | 1.931 | No |
| cutoff_2020 control | low_incentive | count | 2.082 | No |
| cutoff_2020 control | low_incentive | upb | 2.368 | Yes |
| cutoff_2020 control | placebo | count | 3.274 | Yes |
| cutoff_2020 control | placebo | upb | 3.087 | Yes |

Only 3 of 8 clustered slopes clear the correct threshold, all in the `cutoff_2020` control cohort;
none in `cutoff_2002`.

### Responsiveness: too low and under-responsive at low-to-moderate incentive, within the training range

`cutoff_2002`'s training incentive range is [-1.306, 2.967] (1st-99th percentile of the training
set's own `incentive_at_ref`). **4.58%** of the 2003 one-step-ahead population (17,185/374,850
loan-months) falls outside that range — for the `cutoff_2020` control, the corresponding figure is
3.76% (90,084/2,397,271) against its own [-1.63, 2.156] range.

**Per-coupon slope ratios with 95% CI** (`responsiveness_slope_se_ci.csv`):

`cutoff_2002`:

| coupon | ratio pred/real | 95% CI |
|---|---|---|
| 5.0 | 0.213 | [0.095, 0.332] |
| 5.5 | 0.415 | [0.258, 0.572] |
| 6.0 | 0.482 | [0.364, 0.600] |
| 6.5 | 0.572 | [0.439, 0.705] |
| 7.0 | 1.124 | [0.112, 2.136] |
| 7.5 | 2.355 | [-4.219, 8.929] |
| 8.0 | 1.622 | [-3.419, 6.664] |
| 8.5 | 0.948 | [-6.635, 8.532] |
| 9.0 | 0.529 | [-3.570, 4.629] |

`cutoff_2020` control:

| coupon | ratio pred/real | 95% CI |
|---|---|---|
| 1.5 | 0.267 | [0.067, 0.468] |
| 2.0 | 0.355 | [-1.175, 1.885] |
| 2.5 | 2.631 | [-5.736, 10.999] |
| 3.0 | 1.330 | [0.721, 1.940] |
| 3.5 | 1.407 | [0.591, 2.223] |
| 4.0 | 2.568 | [-1.487, 6.624] |
| 4.5 | 20.809 | [-319.699, 361.317] |
| 5.0 | -2.238 | [-9.471, 4.996] |
| 5.5 | -1.004 | [-5.996, 3.987] |

Coupons 5.0-6.5 (`cutoff_2002`, within its training range) are the cleanest signal: all four ratios
are well below 1 with 95% CIs that exclude 1 — **the model is too low and under-responsive to
incentive at low-to-moderate coupons**. The high-coupon and `cutoff_2020` control ratios are mostly
uninformative (wide or sign-flipping CIs off near-zero realized slopes), not evidence of the same
or a different pattern.

**1-month-lag ratios** (`responsiveness_lag_check.csv`, lag1_ratio column):

`cutoff_2002`: 5.0→0.237, 5.5→0.425, 6.0→0.515, 6.5→0.502, 7.0→0.623, 7.5→0.881, 8.0→0.452,
8.5→0.398, 9.0→0.134.

`cutoff_2020` control: 1.5→0.191, 2.0→4.691, 2.5→7.357, 3.0→1.155, 3.5→0.970, 4.0→1.411,
4.5→1.652, 5.0→2.766, 5.5→1.555.

The lagged version doesn't change the `cutoff_2002` low-to-moderate-coupon conclusion (still
well below 1 at 5.0-6.5).

**Crossover shares** (`responsiveness_itm_crossover.csv`, share of loan-months with
incentive ≤ 0):

`cutoff_2002`, Dec 2002 vs Jun 2003, coupons 5.0-7.0: 5.0 → 1.000 to 0.000 (Δ -1.000); 5.5 → 0.647
to 0.000 (Δ -0.647); 6.0/6.5/7.0 → 0.000 to 0.000 (Δ 0).

`cutoff_2020` control, Dec 2020 vs Jun 2021, coupons 1.5-5.5: 1.5 → 1.000 to 1.000 (Δ 0); 2.0 →
0.586 to 1.000 (Δ +0.414); 2.5 → 0.000 to 0.377 (Δ +0.377); 3.0-5.5 → 0.000 to 0.000 (Δ 0).

Artifacts: `scripts/house_price_test_{2002,2020_control}.py`,
`scripts/house_price_cluster_se_{2002,2020_control}.py`, `scripts/house_price_gap_quintiles.py`
(new this session), `scripts/responsiveness_{2002,2020_control}.py`,
`scripts/responsiveness_extras_{2002,2020_control}.py`, `scripts/ensemble_onestep_{2002,2020_control}.py`,
`scripts/tests/test_train_forecast_consistency.py`. Logs: `.claude_tmp/consistency_srun.log`,
`.claude_tmp/ensemble_2020_srun.log`, `.claude_tmp/house_price_2020.log`,
`.claude_tmp/house_price_cse_2020.log`, `.claude_tmp/responsiveness_2020.log`,
`.claude_tmp/responsiveness_extras_2020.log`, `.claude_tmp/gap_quintiles_{2002,2020}.log`. Jobs:
18613794 (consistency), 18614134 (ensemble 2020), 18616110/18616335 (house-price 2020),
18616339/18616475 (responsiveness 2020), 18619302 (gap/quintiles 2020).

---

## Same-incentive comparison resolves the low-incentive gap: an out-of-time shift, not incentive extrapolation (Sep 27, 2026)

**SUPERSEDED (Sep 29): the same-incentive gap is mostly a loan-term effect; see the Sep 29 section.**

The "Responsiveness" section above (Sep 26) read the 2003 low-incentive gap as the model being
"too low and under-responsive" within `cutoff_2002`'s training range, based on a per-coupon
comparison. This session scored the same five-seed ensemble one-step-ahead on `cutoff_2002`
TEST-split loans for Jan2001-Nov2002 (23 months, same frozen scaler and explicit-obs path as the
2003 test — reused the same cached combined pass, no new raw scan; Dec-2002 anchor check passed
for all 5 seeds, max diff ≤7.4e-9 against the frozen 2003-run scores) to test whether the model
under-responds in-sample too. It does, mildly, by coupon — but the coupon-level comparison
against 2003 turned out to be confounded, and is **superseded** by the same-incentive comparison
below.

### The coupon-level comparison was confounded by S-curve region

Market rates fell about 1.5pp between the two windows, so the same coupon sat on a different part
of the S-curve in each period:

| Coupon | 2001-02 mean-incentive range (23 months) | 2003 mean-incentive range (12 months) |
|---|---|---|
| 5.0 | -1.49 to -0.57 | -0.79 to +0.24 |
| 5.5 | -1.13 to -0.05 | -0.26 to +0.78 |
| 6.0 | -0.66 to +0.42 | +0.22 to +1.25 |
| 6.5 | -0.15 to +0.90 | +0.70 to +1.73 |

In 2001-02 these coupons sat mostly out-of-the-money; the same coupons in 2003 crossed through and
past zero incentive. This is why the coupon-level read above looked like "responsive in-sample,
shift in 2003" even though the in-sample ratios were themselves below 1 — see
`docs/mistakes_and_lessons.md`. Full range, all coupons:
`outputs/rolling/ensemble_onestep_insample_cutoff_2002/same_incentive_coupon_ranges.csv`.

### Same-incentive bin comparison (pooled, bootstrap 95% CI over loans, 500 draws, seed 20260927)

| Incentive bin | 2001-02 n | 2001-02 pred/real/ratio (95% CI) | 2003 n | 2003 pred/real/ratio (95% CI) |
|---|---|---|---|---|
| (-2,-1] | 20,985 | .0018/.0016/**1.124** [0.833, 1.556] | 2,180 | .0067/.0101/**0.664** [0.453, 1.099] |
| (-1,-0.5] | 54,829 | .0032/.0029/**1.134** [0.979, 1.314] | 20,235 | .0072/.0109/**0.659** [0.587, 0.751] |
| (-0.5,0] | 85,509 | .0064/.0068/**0.939** [0.871, 1.030] | 39,382 | .0123/.0195/**0.627** [0.591, 0.676] |
| (0,0.5] | 126,944 | .0157/.0151/**1.035** [0.991, 1.079] | 56,818 | .0247/.0357/**0.692** [0.665, 0.723] |
| (0.5,1] | 134,344 | .0303/.0296/**1.023** [0.990, 1.054] | 64,884 | .0419/.0516/**0.813** [0.786, 0.841] |
| (1,1.5] | 115,014 | .0461/.0479/**0.962** [0.939, 0.988] | 58,596 | .0571/.0650/**0.878** [0.852, 0.908] |
| (1.5,2] | 87,236 | .0498/.0475/**1.048** [1.018, 1.080] | 58,918 | .0579/.0579/**1.000** [0.967, 1.035] |
| (2,3] | 59,305 | .0506/.0507/**0.999** [0.969, 1.037] | 57,271 | .0526/.0540/**0.975** [0.944, 1.006] |
| (3,4] | 5,705 | .0368/.0337/**1.092** [0.958, 1.272] | 14,832 | .0384/.0405/**0.948** [0.876, 1.023] |

At matched incentive, 2001-02 ratios sit at/near 1 across the whole -2 to 4 range (every CI
straddles or exceeds 1 except the mildly-low (1,1.5] bin, 0.962). 2003 ratios are tightly and
significantly below 1 for every bin from -1 to 1.5 (0.63-0.88, every CI excludes 1), converging
back to ~1 with the in-sample period beyond incentive 1.5.

### Fixed-band slope comparison, incentive in (-0.5, 1.5]

| Period | months | loan-months | realized slope (SE) | predicted slope (SE) | ratio (95% CI) |
|---|---|---|---|---|---|
| 2001-02 | 23 | 461,811 | 0.0213 (0.0153) | 0.0185 (0.0113) | 0.865 [0.441, 1.289] |
| 2003 | 12 | 219,680 | 0.1234 (0.0528) | 0.0252 (0.0236) | 0.204 [-0.063, 0.471] |

Coupons supplying the band — 2001-02: 7.0 (183,394), 6.0 (95,125), 6.5 (70,298), 7.5 (50,302),
8.0 (43,389), 5.5 (14,124), 5.0 (5,179). 2003: 6.0 (91,763), 5.0 (37,639), 5.5 (32,580),
6.5 (31,091), 7.0 (26,190), 4.5 (405), 4.0 (12) — materially overlapping coupon sets, so the
slope result isn't a coupon-composition artifact either.

### Conclusion: an out-of-time shift, not incentive extrapolation

At the same incentive, the model is calibrated in 2001-02 (predicted/realized 0.94-1.13 for
incentive -1 to 1.5) but 0.63-0.88 in 2003 over that same range, with every interval below 1, and
about 1 above 1.5. Realized slope in that band was about 6x steeper in 2003 (0.123 vs 0.021) while
predicted barely moved (0.018 to 0.025). Both periods populate the same incentive bins with large,
comparable loan-month counts (e.g. (-0.5,0]: 85,509 in 2001-02 vs 39,382 in 2003) — the incentive
*values* were well within what the training-era population covered, so this is not incentive
extrapolation. What changed is the realized rate-incentive relationship at those same incentive
values between the two periods, which the model — calibrated on the earlier relationship — did
not track into 2003. Open question: why prepayment at a given incentive was higher in 2003 than in
2001-02; one untested candidate is a response to rates reaching new lows (PMMS 30yr: 6.05% Dec
2002, the low for 2000-2002, vs 5.23% June 2003, a new low within 2003).

Artifacts: `scripts/score_rolling_onestep_insample_2002.py`, `scripts/ensemble_onestep_insample_2002.py`,
`scripts/responsiveness_insample_2002.py`, `scripts/responsiveness_same_incentive_2002.py`.
Outputs: `outputs/rolling/rolling_onestep_insample_cutoff_2002_seed{42,7,123,1001,2026}/`,
`outputs/rolling/ensemble_onestep_insample_cutoff_2002/same_incentive_*.csv`. Jobs: 18669329-18669333
(five-seed in-sample scoring, cache hits, <1 min each).

## Advisor's full-training-sequence plan: timing, cost, a 10-epoch ensemble, and a voluntary-prepayment audit (Sep 28, 2026)

Fact-gathering for a proposed expanding-window training sequence. No pipeline code changed; new
artifacts are five 10-epoch training sbatch files and this README. All numbers below are read from
saved outputs (`outputs/zbc_audit/`, `outputs/master_audit/`, `outputs/rolling/`) and `sacct`.

**Training time — all 10 f0.2 L33 50-epoch runs, on one L40S GPU (1 GPU / 4 CPU / 40G).** From
`sacct` elapsed:

| cutoff | seed | job | node | elapsed |
|---|---|---|---|---|
| 2002 | 42 | 18097822 | gl017 | 2:02:54 |
| 2002 | 7 | 18138546 | gl022 | 2:02:05 |
| 2002 | 123 | 18593821 | gl006 | 2:01:44 |
| 2002 | 1001 | 18593824 | gl042 | 2:05:45 |
| 2002 | 2026 | 18593826 | gl063 | 2:03:04 |
| 2020 | 42 | 17355084 | gl020 | 2:17:05 |
| 2020 | 7 | 17594751 | gl036 | 2:14:52 |
| 2020 | 123 | 17594752 | gl007 | 2:14:22 |
| 2020 | 1001gb | 18606771 | gl011 | 2:15:12 |
| 2020 | 2026gb | 18606772 | gl026 | 2:15:55 |

**Fixed steps per epoch.** `train_hazard_multiobs.py:74` sets `STEPS_PER_EPOCH = 10,000`
(× `BATCH_SIZE = 2048` = 20,480,000 draws/epoch, sampled with replacement), so work per epoch is
constant regardless of data size. 12.4× more data (11.2M vs 907K obs) costs only ~10% more wall
(2:15 vs 2:03) — the residual is per-epoch `evaluate()` over a 12× larger test set, not training.
A cutoff_2002 epoch sees each observation ~22.6× per epoch; a cutoff_2020 epoch ~1.82×.

**10-epoch vs 50-epoch, 5-seed ensemble.** Trained a 10-epoch replicate of all five cutoff_2002
seeds (`_ep10`; jobs 18750843, 18754368/70/72/73; each ~0:25:1x wall). Test AUC: 50-epoch mean
0.7742 (0.7734-0.7746), 10-epoch mean 0.7723 (0.7717-0.7728). Same-incentive comparison with the
difference divided by the standard error of two 5-seed means, √(sd₁₀²/5 + sd₅₀²/5):

| period | pooled 50ep | pooled 10ep | pooled t | max \|t\| over bins |
|---|---|---|---|---|
| 2003 (out-of-sample) | 0.873 | 0.922 | +1.40 | 1.72 (bin (-2,-1]) |
| 2001-02 (in-sample) | 1.008 | 1.049 | +1.26 | 1.64 (bin (3,4]) |

No incentive bin, and neither pooled figure, reaches |t| > 2 in either period — the 10-epoch
ensemble is indistinguishable from the 50-epoch ensemble at the seed level. The 10-epoch seeds are,
however, systematically noisier across seeds (per-bin sd ~0.07-0.15 vs 0.03-0.10). (An earlier
single-seed claim that "10 epochs worsens the low-incentive tail" was retracted: it was one seed vs
a 5-seed ensemble, inside the 0.09-0.41 per-bin seed spread.)

**Voluntary-prepayment label audit.** Training label (`prepare_sequences_multiobs_zbc.py:527`) and
every forecast test use `zero_balance_code_actual == 1` (code 01) only. But `realized_cpr_v6*.py`
defines a prepayment as UPB==0 at the true last row — *any* balance-to-zero ending. Column positions
were verified empirically on both raw dirs (113 fields each): loan_id `$2`, MRP `$3`, current_upb
`$12`, delinquency `$40`, modification flag `$42` (Y, persists post-mod), ZBC `$44`. Terminations by
code, all vintages 2000-2025 (source `outputs/zbc_audit/`): 01 = 40,820,568; 09 = 453,043;
16 = 145,072; 03 = 107,905; 06 = 93,656; 02 = 68,850; 15 = 45,959. Non-voluntary (non-01) share of
balance-to-zero endings, by count and UPB-weighted (last nonzero balance):

| year | non-vol % (count) | non-vol % (UPB-wt) |
|---|---|---|
| 2008 | 2.68 | 2.99 |
| 2009 | 2.58 | 2.69 |
| 2010 | 5.11 | 5.28 |
| 2011 | 5.17 | 5.50 |
| 2018 | 3.24 | 4.06 |
| 2020 | 0.65 | 0.45 |
| 2024 | 2.27 | 3.01 |

Peak contamination is 2010-11 (crisis: REO code 09 + short sales code 03); across the DER window
(2018-25) it is 0.45-4.1% UPB-weighted, driven mostly by code 16 (reperforming/NPL loan sales).
UPB-weighting raises it in recent years (larger-balance non-voluntary endings) and lowers it pre-2008.

**Modifications.** Ever-modified share by vintage year (`$42==Y`): crisis vintages peak — 2007 7.56%,
2006 5.94%, 2008 5.11%, 2005 4.27% (HAMP era) — vs ~0.4-1.4% for normal vintages; 1.33% overall
across 54.9M loans. Post-modification rate divergence (source `outputs/master_audit/`, current rate
`$9` vs original `$8` on post-mod months): 67.7% of post-mod months carry a rate change, of which
**99.4% are cuts**, mean 2.50pp / median 2.25pp; crisis vintages 2005-08 have 78.9-91.0% of post-mod
months changed at 2.3-2.8pp. Because `refi_incentive = original_rate − market` uses `$8`, incentive
is **overstated** for rate-reduced modified loans (a cut lowers the current rate, hence the true
incentive).

**Maturities inside code 01.** Share of code-01 terminations within 2 months of scheduled maturity
(origination date + original term): 0.97% overall by count, near-zero before 2010, rising to a
**7.66% peak in 2018** (the 2003 15-year refi cohort reaching term); COVID window (2020-21) only
~0.5%. So code 01 is a near-clean voluntary label except where an older cohort matures.
**UPB-weighted, this is negligible** — near-maturity loans have amortized to tiny balances, so by
balance the maturity share is 0.012% overall and 0.097% even in the 2018 peak (job 18762696):

| year | count-share % | UPB-weighted % |
|---|---|---|
| 2013 | 1.38 | 0.019 |
| 2017 | 3.36 | 0.042 |
| 2018 | 7.66 | 0.097 |
| 2023 | 2.62 | 0.036 |
| 2025 | 3.72 | 0.056 |
| ALL | 0.97 | 0.012 |

**Estimated LTV.** An empirical search across all 93 files found no field carrying genuine
current-LTV values: the only fields with LTV-range values are decoys (`$73` step-mod counts,
`$79` the constant "7"), and the one mid-range field (`$49`) holds non-LTV rising values (7% in-band).
File-native estimated LTV is effectively unpopulated; the pipeline uses the derived `current_ltv`
(original_upb × ZHVI ratio) instead.

**Cost estimates (ESTIMATES, from the measured times above).** Expanding-window retrain 2002→2024,
50 epochs from scratch: yearly (23 retrains) ≈ 51 GPU-h at 1 seed / ≈ 253 GPU-h at 5 seeds; monthly
(276 retrains) ≈ 607 / ≈ 3,035 GPU-h. Warm-starting ~10 epochs (justified by the ensemble result
above) cuts these ~5×. The real bottleneck is the CPU-side data build (cold 1.5-7h, warm 0.5-1.2h)
per retrain, and sequence-array memory scales with observation count (the current 11.2M-obs / 40G
profile will need to grow for an expanding window) — GPU *time* will not, because steps are fixed.
`l40s_public` has no per-user GPU cap (272 GPUs physical in the partition, QOS pool cap 208 GPUs
shared across all users); `cpu_short` caps a user at 32 CPU / 120G.

Artifacts: `scripts/slurm/run_train_multiobs_2002_f0.2_L33_hist_ep10.sbatch` and the four
`_seed{7,123,1001,2026}_ep10` variants. Outputs:
`outputs/rolling/{rolling_onestep,rolling_onestep_insample}_cutoff_2002_seed*ep10/`,
`outputs/zbc_audit/`, `outputs/master_audit/`. Consistency test passed 2026-09-28 (srun 18753120).

## Advisor's Sep 29 decisions; loan-term mix resolves the 2003 gap; modification, cutoff_2021, and compute facts (Sep 29, 2026)

Fact-gathering for the advisor's Sep 29 decisions. No pipeline code changed; new artifacts are two
analysis-only scripts (`scripts/diag/term_pass_worker.py`, `scripts/diag/term_report.py`), one
sbatch file (`scripts/diag/run_term_pass.sbatch`), and this README section. All numbers below trace
to `.claude_tmp/term_pass/` (`report.txt`, `gap_decomp_out.txt`, `sacct_l40s_raw.txt`,
`compute_budget_facts.txt`) — a local, not-committed scratch dir — plus the two committed scripts
that produced them.

### Advisor's decisions (verbatim intent)

30-year loans only, throughout — label, features, both realized series, census panel, and sampler.
Yearly retraining (quarterly possibly later). Drop all post-modification loan-months. Skip the HARP
two-loan linkage for now; build only the eligibility feature. Two realized series: **voluntary**
(zero_balance_code 01, 30-year only — what the model forecasts and is validated against) and
**total payoffs** (all terminations, for pricing). The post-month-33 terminal curve is fit on
voluntary refis only. Rebuild the 2002 and "2021" cutoffs first (which cutoff "2021" means is being
clarified with him).

### Method

One CPU pass over all 93 raw vintage files (`data_pre2013_raw/2000Q1.csv` .. `data/raw/2023Q1.csv`,
~908 GB combined) built a per-loan table: `original_loan_term` (`$13`), `original_upb` (`$10`),
`note_rate` (`$8`), origination month (`$14`), the earliest month `modification_flag` (`$42`) reads
`Y`, the earliest non-blank `zero_balance_code` (`$44`) and its month, and the loan's last reporting
month + balance. `MMYYYY`→`YYYYMM` converted before every ordering/comparison (standing lesson).
Field positions re-verified empirically on both `data/raw` and `data_pre2013_raw` before use;
`$13`=360 on the first rows of both 2000Q1 and 2013Q1. Run as a 93-way SLURM array
(`scripts/diag/run_term_pass.sbatch`, job 18844165, `cpu_short`, ~40 min wall, 0 failures) producing
54,899,569 unique loans. `modification_flag` is monotone — checked directly (107 ever-`Y` loans in a
2013Q1 sample, 0 reverted to `N`; the existing `outputs/zbc_audit/` shows `nmod`≈`nmod_persist` per
vintage throughout) — so post-modification is exactly `ref_month ≥ first-Y month` per loan; no
loan-month-level raw join was needed. `scripts/diag/term_report.py` then joins this table against
the five scored populations' `loan_id`s (0 unmatched in every population) to produce everything
below.

### 1. Loan-term mix

**(a) Original-term distribution by origination year**, loan count and original UPB, 360/240/180/other:

| year | 360 (n) | 240 | 180 | other | total | non-360 % | 360 UPB ($B) | non-360 % (UPB) |
|---|---|---|---|---|---|---|---|---|
| 2000 | 1,064,868 | 26,151 | 155,233 | 21,640 | 1,267,892 | 16.0 | 140.3 | 12.7 |
| 2001 | 2,331,641 | 111,932 | 827,245 | 101,052 | 3,371,870 | 30.9 | 347.6 | 26.5 |
| 2002 | 2,374,635 | 160,304 | 1,165,252 | 157,076 | 3,857,267 | 38.4 | 372.1 | 34.1 |
| 2003 | 2,990,406 | 258,513 | 1,571,822 | 286,582 | 5,107,323 | **41.5** | 494.1 | **36.5** |
| 2004 | 1,183,793 | 95,709 | 371,887 | 93,108 | 1,744,497 | 32.1 | 199.3 | 27.3 |
| 2005 | 1,123,820 | 71,171 | 201,886 | 49,305 | 1,446,182 | 22.3 | 207.1 | 17.9 |
| 2006 | 889,390 | 44,493 | 122,362 | 24,557 | 1,080,802 | 17.7 | 171.4 | 13.7 |
| 2007 | 1,056,344 | 50,035 | 118,734 | 27,312 | 1,252,425 | 15.7 | 216.5 | 11.9 |
| 2008 | 1,173,656 | 52,565 | 223,859 | 41,608 | 1,491,688 | 21.3 | 261.0 | 17.1 |
| 2009 | 1,751,247 | 98,619 | 425,321 | 87,976 | 2,363,163 | 25.9 | 416.0 | 20.3 |
| 2010 | 1,195,965 | 121,212 | 491,711 | 142,643 | 1,951,531 | 38.7 | 294.5 | 31.9 |
| 2011 | 1,000,142 | 94,727 | 425,264 | 141,640 | 1,661,773 | 39.8 | 234.6 | 34.4 |
| 2012 | 1,704,837 | 147,440 | 662,260 | 165,605 | 2,680,142 | 36.4 | 416.4 | 31.5 |
| 2013 | 1,518,439 | 81,112 | 486,920 | 120,944 | 2,207,415 | 31.2 | 355.2 | 26.5 |
| 2014 | 1,091,133 | 48,848 | 249,311 | 60,358 | 1,449,650 | 24.7 | 247.2 | 20.7 |
| 2015 | 1,404,425 | 74,932 | 319,792 | 69,971 | 1,869,120 | 24.9 | 333.5 | 21.2 |
| 2016 | 1,719,029 | 136,857 | 409,872 | 88,002 | 2,353,760 | 27.0 | 423.4 | 23.7 |
| 2017 | 1,571,179 | 100,097 | 288,795 | 54,015 | 2,014,086 | 22.0 | 377.6 | 18.2 |
| 2018 | 1,514,613 | 71,271 | 170,441 | 31,695 | 1,788,020 | 15.3 | 368.8 | 12.2 |
| 2019 | 1,814,200 | 78,682 | 263,435 | 55,943 | 2,212,260 | 18.0 | 494.5 | 15.0 |
| 2020 | 3,636,583 | 313,840 | 831,778 | 209,370 | 4,991,571 | 27.2 | 1,074.3 | 23.3 |
| 2021 | 3,302,258 | 289,303 | 838,127 | 221,917 | 4,651,605 | 29.0 | 1,001.5 | 23.9 |
| 2022 | 1,509,170 | 55,335 | 181,159 | 44,296 | 1,789,960 | 15.7 | 476.5 | 11.6 |
| 2023 | 125,139 | 2,560 | 6,871 | 1,016 | 135,586 | 7.7 | 39.5 | 5.6 |

Non-360 (15-year the bulk) peaks at both boom periods — 2002-03 (rate-driven refi wave into 15-year)
and 2010-11 (post-crisis refi into shorter terms) — and is lowest in the purchase-heavy/high-rate
years 2018-19 and 2022-23. The UPB share is consistently lower than the count share (15-year loans
are smaller). Full parquet-level table is in `.claude_tmp/term_pass/report.txt`.

**(b) Non-360 share of loan-months in the five scored populations** (join clean — 0 unmatched loans
in any population):

| population | loan-months | non-360 share |
|---|---|---|
| cutoff_2002 train | 906,877 | 30.8% |
| cutoff_2002 test | 227,078 | 30.8% |
| 2001-02 in-sample | 690,358 | 31.5% |
| **2003 one-step-ahead** | 374,850 | **41.0%** |
| cutoff_2020 train | 11,230,531 | 26.3% |
| cutoff_2020 test | 2,808,691 | 26.3% |
| **2021 one-step-ahead** | 2,397,271 | **27.6%** |

**(c) Same-incentive bin ratios split 360 vs non-360, pooled decomposition.** Ratio = Σ predicted
event-probability (`h`) / Σ realized events, count-weighted; bin edges `[-2,-1,-0.5,0,0.5,1,1.5,2,3,4]`
(same as the Sep 27 same-incentive comparison).

*2003 one-step-ahead — 374,850 loan-months, exact split 221,001 (360) + 153,849 (non-360):*

| group | n | Σh | Σrealized | ratio | shortfall (real−pred) |
|---|---|---|---|---|---|
| 360 | 221,001 | 11,127.69 | 11,637 | **0.9562** | 509.31 |
| non-360 | 153,849 | 4,028.31 | 5,715 | **0.7049** | 1,686.69 |
| pooled | 374,850 | 15,156.00 | 17,352 | 0.8734 | 2,196.00 |

**Non-360 is 41.0% of loan-months but 76.8% of the total shortfall.** Per-bin (360 / non-360, exact
counts):

| incentive bin | 360 n | 360 ratio | non-360 n | non-360 ratio |
|---|---|---|---|---|
| (-2,-1] | 173 | n/a (real=0) | 2,007 | 0.635 |
| (-1,-0.5] | 575 | 0.402 | 19,660 | 0.672 |
| (-0.5,0] | 3,526 | 1.027 | 35,856 | 0.596 |
| (0,0.5] | 18,429 | 1.046 | 38,389 | 0.577 |
| (0.5,1] | 38,512 | 0.925 | 26,372 | 0.672 |
| (1,1.5] | 45,209 | 0.893 | 13,387 | 0.827 |
| (1.5,2] | 50,161 | 1.001 | 8,757 | 0.995 |
| (2,3] | 49,760 | 0.977 | 7,511 | 0.960 |
| (3,4] | 13,099 | 0.978 | 1,733 | 0.738 |

(Bin totals sum to 373,116, not 374,850 — 1,734 loan-months fall outside `[-2,4]` incentive and are
excluded from the bin table but included in the pooled totals above.) 360-term is essentially
calibrated (0.89-1.05) everywhere populated; non-360 sits at 0.58-0.83 through incentive 1.5, then
converges to ~1 above it — the same shape the Sep 27 note attributed to an out-of-time shift, now
resolved as concentrated in the 15-year population.

*2021 one-step-ahead / cutoff_2020 control — 2,397,271 loan-months, split 1,736,441 (360) +
660,830 (non-360):*

| group | n | Σh | Σrealized | ratio | shortfall |
|---|---|---|---|---|---|
| 360 | 1,736,441 | 42,445.72 | 47,212 | **0.8990** | 4,766.29 |
| non-360 | 660,830 | 10,741.70 | 13,576 | **0.7912** | 2,834.30 |
| pooled | 2,397,271 | 53,187.42 | 60,788 | 0.8750 | 7,600.58 |

Non-360 is 27.6% of loan-months and **37.3%** of the shortfall — same direction, much milder than
2003. Full 2021 per-bin table (360/non-360) is in `.claude_tmp/term_pass/gap_decomp_out.txt`.

**(d) Code-01 near-maturity share** (`|age_at_termination − original_term| ≤ 2`), by term, by
termination year:

- **360-term: 0.00% in every year 2001-2025.** No 30-year loan in this data window (acquisitions
  start 2000, data through Dec 2025) has reached scheduled maturity.
- **Non-360:** ramps from ~0 pre-2010 to a **20.0% peak in 2018** (98,879 of 493,691 non-360 code-01
  terminations) — the 2003 15-year cohort maturing (2003+15=2018) — with 2017 at 9.2%, and a later
  rise (2023 7.4%, 2024 6.8%, 2025 10.0%) from the 2008-10 15-year cohorts.
- **Pooled, both terms: 7.66% in 2018** — reproduces the figure already in this README (job
  18762696) exactly, and confirms it is **entirely a non-360 phenomenon**.

### 2. Modification scope

Post-modification loan-month share, all five populations: cutoff_2002 train 0.063%, cutoff_2002
test 0.065%, 2001-02 in-sample 0.061%, **2003 one-step 0.254%**, cutoff_2020 train 0.221%, cutoff_2020
test 0.228%, **2021 one-step 0.496%**. Dropping post-mod months removes well under 1% of every
population — the change is essentially free in sample size.

### 3. Cutoff 2021 status

No `cutoff_2021_zbc_multiobs` build exists. The only `cutoff_2021*` artifact is
`data/sequences_rolling/cutoff_2021/` — the old June-21 **plain pipeline** (non-ZBC, non-multiobs),
not comparable to the current cutoff_2002/cutoff_2020 multiobs design. Everything scored to date
used `cutoff_2020` (train ≤ Dec 2020, forecast CY2021). Realized data for CY2022 exists — raw
performance rows run through Dec 2025 (`outputs/realized_cpr_by_coupon_v6.csv` has months through
2025-09) — so either a relabeled cutoff_2020 or a genuine new train-≤-Dec-2021/forecast-CY2022
cutoff_2021 is buildable; which one the advisor means is the open clarifying question.

### 4. Compute budget

**Account limits — no hard cap.** `torch_pr_932_general` under QOS `normal`: no `GrpTRESMins`,
`MaxJobs`, `MaxSubmit`, or GPU-hour cap (`sacctmgr show qos normal`, `sacctmgr show assoc` — both
blank). Scheduling is fair-share only; `sshare -A torch_pr_932_general` gives FairShare 0.233 as of
Sep 29 23:37 EDT (a live metric that moves with usage, not a fixed limit — it read 0.236 a few hours
earlier the same day). Full output in `.claude_tmp/term_pass/compute_budget_facts.txt`.

**GPU queue-wait distribution (measured, full population, not a sample).** All 53 `l40s_public`
jobs submitted since Sep 20 (`.claude_tmp/term_pass/sacct_l40s_raw.txt`), submit-to-start wait:
**min 0.02 min, median 1.45 min, max 54.5 min** (job 18479425 `measure_alignment_effect`). Seven
jobs waited 13-55 min; all seven were submitted in batches of 5-6 jobs at once and queued behind
each other for a small number of GPUs — self-contention, not external queue pressure. `l40s_public`
itself is deep (695 jobs, 506 pending, snapshot Sep 29 23:37) but jobs submitted 1-2 at a time
started in under 3 minutes throughout.

**Wall-clock for yearly retraining 2002-2024 × 5 seeds (~115 runs, ~250 GPU-h) — ESTIMATES beyond
the two measured points.** Measured: one 50-epoch run = 2:01-2:17 wall on one L40S (work per epoch
is fixed by `STEPS_PER_EPOCH`, independent of data size); a cutoff_2002-scale data build = 31 min
CPU wall (job 18054077). *Estimated:* ~26 h wall with 10 concurrent L40S, ~51 h with 5 concurrent
(12/23 waves × ~2.2 h); the cutoff_2020-scale build time was not separately measured this session
(previously noted as 1.5-7h cold / 0.5-1.2h warm in the Sep 28 section, likely still applicable but
not re-verified). Given the account has no hard cap and recent queue waits are dominated by
self-batching rather than external pressure, nothing in (a) or the queue snapshot blocks this run —
the main risk is queue-wait stretching wall-clock if we submit a large burst, not a hard stop.

### 5. Implementation scope and order

**(i) 30-year filter at raw-read.** Add `original_loan_term` (`$13`) to the column map and filter
`== 360` in: `prepare_sequences_multiobs_zbc.py` (`_COL_MAP`/`load_vintage_filtered`) and siblings
`prepare_sequences_rolling_zbc.py`, `prepare_sequences_trailing_zbc.py`; the cell-grid sampler
`scripts/build_cell_grid_sample_pre2013.py` (`scan_file`); the census panel
`scripts/diag/census_panel_baseline.py`; `scripts/realized_cpr_v6.py` (+ `_upb`, `_upb_byage`).
**Must land first** — the sampler and census panel both consume its output, so the sampler must be
rebuilt before the census/sequence builds.

**(ii) Drop post-modification rows.** Read `$42` in the same raw loaders; drop each loan's rows with
`ref_month ≥ first-Y month`. **Must land before window/age computation** — dropping interior rows
changes `L`, `loan_age`, and `term_t`, all of which are computed from the row-indexed panel.

**(iii) Incentive from current rate `$9`** *(not part of the Sep 29 decision list above — carried
over from the Sep 28 open item, still awaiting explicit confirmation)*: change
`original_interest_rate`→`current_interest_rate` in `prepare_sequences_multiobs_zbc.py:458` and in
every script that recomputes incentive for scoring/binning (`ensemble_onestep_2002.py`,
`ensemble_onestep_2020_control.py`, `ensemble_onestep_insample_2002.py`, `forecast_rolling_cpr.py`,
`realized_cpr_by_refi_v1.py`, `house_price_*`, the `diag_*` incentive scripts). Interacts with (ii):
current rate differs from note rate almost only on rate-reduced modified loans, so once (ii) drops
those rows this becomes close to a no-op — decide (ii) first.

**(iv) Two realized series from `realized_cpr_v6.py`.** Split into **voluntary** (zbc `01` AND
30-year, per (i)) and **total payoffs** (any UPB→0 / any zbc, all terms — for pricing). The
post-month-33 terminal curve fit (`scripts/diag/fit_terminal_scurve.py`) switches to the voluntary
series. **Depends on (i)** landing first.

**(v) HARP eligibility feature.** Build the eligibility flag from `data_harp_raw/` (fields located;
two-loan linkage via `Loan_Mapping.txt` explicitly skipped per today's decision). Add to
`FEATURE_COLS` in the builders — a new sequence channel, so it changes the input dimension consumed
by `train_hazard_multiobs.py` and the scoring path. **Cannot share a training run with models that
omit it** — build the eligibility map and join it in the builder before any run that includes it,
and do not mix HARP/non-HARP checkpoints in the same ensemble.

Artifacts: `scripts/diag/term_pass_worker.py`, `scripts/diag/term_report.py`,
`scripts/diag/run_term_pass.sbatch`. Outputs: `.claude_tmp/term_pass/parquet/` (93 per-vintage
parquets, 710M, local/not committed), `.claude_tmp/term_pass/report.txt`,
`.claude_tmp/term_pass/gap_decomp_out.txt`, `.claude_tmp/term_pass/sacct_l40s_raw.txt`,
`.claude_tmp/term_pass/compute_budget_facts.txt`. Jobs: 18844165 (93-way term-pass array, `cpu_short`,
~40 min, 0 failures), 18846369/18848299 (srun report/decomposition runs, `cpu_short`). Consistency
test not re-run today (no pipeline code changed); last pass remains 2026-09-28 (srun 18753120).

---

## 30-year/post-mod rebuild of cutoff_2002, first 2002 census, five seeds, 2003 one-step 0.898, matched-intersection decomposition (Oct 1-2, 2026)

### Advisor's decisions and Oct 1 clarification

Per the Sep 29 section above: 30-year loans only throughout (label, features, both realized series,
census panel, sampler); yearly retraining; drop all post-modification loan-months; skip HARP
two-loan linkage, build only the eligibility feature (not done this session); two realized series
(voluntary code-01-30y, total payoffs — not done this session). **Oct 1 clarification:** "rebuild
the 2021 cutoff" means the existing `cutoff_2020` design (train ≤ Dec 2020, forecast CY2021), not a
new train-≤-Dec-2021 cutoff — no new `cutoff_2021` build is needed.

### 30-year filter and post-modification implementation

**Loan-level term rule.** `load_vintage_filtered`'s `term_filter: int | None = 360` (default 360):
drops a loan ENTIRELY (every row) if ANY observed `original_loan_term` across its own rows differs
from 360, including blank/unparseable values — a loan-level filter, not row-level (found necessary
2026-10-01: 2 loans in 2002Q1 and 1 in 2018Q1 carry more than one distinct nonblank term across
their own rows). Applied in `prepare_sequences_multiobs_zbc.py`, `prepare_sequences_rolling_zbc.py`,
`prepare_sequences_trailing_zbc.py`, `build_cell_grid_sample_pre2013.py`'s `scan_file`, and
`census_panel_baseline.py` — all five readers, same rule.

**Sticky post-mod rule.** Drop every row from a loan's first `modification_flag=='Y'` month
onward ("first Y onward"). **This is the advisor's design choice, not an empirical property of the
flag** — see the mistakes log. `scripts/diag/verify_term_mod_filters.py` (job 18986341) checked the
monotonicity assumption directly rather than trusting the 2013Q1-sample check from before this
session, and found: 142 of 594,425 30-year loans in 2002Q1 revert `Y`→`N` for at least one month,
140 of those 142 reverting in the same single reporting month (April 2020); 64 2018Q1 loans revert,
spread across six months from June 2020 through August 2025. The April-2020 concentration is
consistent with (not confirmed as) a COVID-era servicer reporting change. D.4
(`.claude_tmp/verify_term_mod_filters_18992408.log`, not committed): of the term==360 population's
589,311 total nonblank-zbc terminations (2002Q1) and 278,641 (2018Q1), the loan-months the sticky
post-mod rule drops that are themselves terminations are **4,416/589,311 = 0.75%** (2002Q1) and
**2,193/278,641 = 0.79%** (2018Q1) — code-01 specifically: 2,268/577,984 = 0.39% and
1,285/276,472 = 0.46%.

### Sampler rerun

`build_cell_grid_sample_pre2013.py` rerun under the fixed loaders (729-cell vintage_quarter×coupon
grid, `outputs/pre2013_cell_sample_30y_loans.csv`, job not separately logged beyond
`logs/build_cell_grid_sample_30y_19004194.log`): 19,763,814 term==360 loan records scanned, **694 of
729 cells nonempty**, **1,482,004 of 19,763,814 loans selected (7.5%)**, **1,292,272 events**
selected (summed `n_events_selected` from `outputs/pre2013_cell_sample_30y_summary.csv`).

### First cutoff_2002 census

`census_panel_baseline.py --include_pre2013 --out_tag _30y` (job 19015357, the first `cutoff_2002`
census ever run — the script previously scanned 41 modern-only vintage files, all empty after the
cutoff filter, producing a silent empty census for any pre-2013 cutoff; see mistakes log):
**277,042 loans, 3,595,798 loan-months, 99,227 events, 2.7595% raw rate**
(`outputs/census_panel_baseline_cutoff_2002_30y.json`, `overall.*`).

**a_eff/b_eff** (`scripts/diag/census_check_2002_30y.py`, srun cpu_short): sample events/census
events = 99,227/99,227, **a_eff = 1.000**; sample non-events/census non-events = 942,680/3,496,571,
**b_eff = 0.2696**; **log(a_eff/b_eff) = 1.311**. IPW-weighted sample rate 0.027595, matching the
census rate to displayed precision. **Caveat:** `a_eff=1.0` is by construction (every terminal/prepay
draw is mandatory, `incl_prob=1.0`), not evidence of a matched population — sample n_loans
(269,536) ≠ census n_loans (277,042). Because the builder's non-mandatory `incl_prob` is the
REALIZED per-loan `k_actual/n_eligible` ratio (not the nominal `frac_draws=0.2`), the IPW-rate match
is a **near-identity**, not an independent check — see mistakes log STANDING TESTS.

### Build

`prepare_sequences_multiobs_zbc.py --cutoff_year 2002 --include_pre2013 --sampling_mode
fixed_fraction --frac_draws 0.2 --cell_sample outputs/pre2013_cell_sample_30y_loans.csv --run_tag
_30y` (job 19016008, 37m21s): **277,042 loans through the loader** (matches the census loan count
exactly — same population), Train 221,633 / Test 55,409 loans. Gate (`check_build_30y.py`, job
19016010 first run — FAILED on a path-string-compare bug, masked as `0:0` by the missing `set -e`;
fixed and rerun as job 19053097 — **PASSED**): **269,536 loans in train+test** (fewer than 277,042
— some loans get zero sampled observations under the calendar filters), **833,720 train / 208,187
test observations** (1,041,907 total), 99,227 of which are terminal&prepay (exactly the census event
count).

### Five seeds

`outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s{42,7,123,1001,2026}/` — jobs
19016089/19016121/19016131/19016132/19016133, 50 epochs each, ~2h01m: **AUC 0.7744–0.7759**
(`hazard_best.pt`'s `results.json.best_auc`; seed42 0.7744, seed7 0.7757, seed123 0.7759, seed1001
0.7758, seed2026 0.7751 — see §5 numbers below for the exact `hazard_best.pt` AUCs used downstream).

### Consistency test

`scripts/tests/test_train_forecast_consistency.py --case cutoff_2002_seed42_30y`, srun
(cpu_short, 40G, 30 min). **Passed**, `hazard_best.pt`: Check 1 (vintage 2001Q1, n=2,000)
max_seq_diff=0.0, mask exact, max_pred_diff=0.0 (AUC=0.7747443717718066). Check 2: window-end
verified at months 200212/200306/200311 (max abs diff 1.17e-07 to 1.20e-07, scaled space).
Negative control (b): frozen Dec-cutoff window disagrees with June-of-forecast-year's actual
features on 27,733/27,733 loan-months (max abs diff 2.629). Negative control (a) structurally
inapplicable (`TRAIL_SEQ_DIR` is `cutoff_2020`-only).

### 2003 one-step result, five seeds

`scripts/score_rolling_one_step.py` per seed (jobs 19055843/44/45/46/47, each depending on its own
dec-window scoring job 19055681-685 for the Dec-consistency check, max abs diff 7.3e-9 to 1.1e-8),
pooled via `scripts/ensemble_onestep_2002.py --run_tag _30y` (job 19056038):

**Pooled, n=323,203 loan-months:** count ratio **0.8982**, UPB ratio **0.8818**. Per-seed count
ratios: seed42 0.9273, seed7 0.8365, seed123 0.8870, seed1001 0.8697, seed2026 0.9706 — range
0.8365–0.9706. Pooled events: Σh=14,413.0, Σrealized=16,046.0, **shortfall = 1,633.0 (10.2% of
16,046 realized events)**.

Incentive-bin table (ensemble, count-weighted monthly, edges `[-6,-4,-3,-2,-1,-0.5,0,0.5,1,1.5,2,3,4,6]`):

| bin | n | predicted | realized |
|---|---|---|---|
| (-2.0,-1.0] | 259 | 0.003198 | 0.003861 |
| (-1.0,-0.5] | 2,605 | 0.006026 | 0.008061 |
| (-0.5,0.0] | 14,874 | 0.013285 | 0.015867 |
| (0.0,0.5] | 52,911 | 0.021696 | 0.025722 |
| (0.5,1.0] | 67,760 | 0.041972 | 0.052332 |
| (1.0,1.5] | 58,187 | 0.059051 | 0.067678 |
| (1.5,2.0] | 57,058 | 0.059298 | 0.058239 |
| (2.0,3.0] | 54,725 | 0.052296 | 0.054619 |
| (3.0,4.0] | 13,431 | 0.036486 | 0.043556 |
| (4.0,6.0] | 1,300 | 0.027166 | 0.034615 |

Month-by-month (ensemble, annualized):

| ref_month | n | predicted | realized |
|---|---|---|---|
| 200212 | 35,242 | 0.3680 | 0.3574 |
| 200301 | 33,948 | 0.3777 | 0.3739 |
| 200302 | 32,626 | 0.4022 | 0.4365 |
| 200303 | 31,089 | 0.4284 | 0.4959 |
| 200304 | 29,346 | 0.4370 | 0.4855 |
| 200305 | 27,731 | 0.4728 | 0.5278 |
| 200306 | 26,030 | 0.5228 | 0.6066 |
| 200307 | 24,052 | 0.5164 | 0.5956 |
| 200308 | 22,287 | 0.4275 | 0.4850 |
| 200309 | 21,073 | 0.3652 | 0.3891 |
| 200310 | 20,206 | 0.3578 | 0.3090 |
| 200311 | 19,573 | 0.3459 | 0.3148 |

### Three-way pooled-ratio comparison

| population | pooled count ratio |
|---|---|
| Sep 26 ensemble, old all-term build (374,850 loan-months, 360+non-360 mixed) | **0.8734** |
| Sep 29 `term_report.py` decomposition, old build's 360-only subset (221,001 loan-months) | **0.9562** |
| `_30y` build, pure 360-term by construction (323,203 loan-months) | **0.8982** |

**Mark clearly:** the Sep 29 "76.8% of the shortfall" finding is a decomposition of the **all-term**
model's shortfall (360 pooled 0.9562 / non-360 pooled 0.7049, both on the old mixed-term build) —
still true as a statement about that model/population pair. It is superseded, for the
30-year-only model, by the `_30y` build's own number: **0.898**, not 0.956 — these differ because
the `_30y` build changes more than term filtering alone (post-mod drop, calendar-alignment filters,
a different/fresher cell sample, and a different checkpoint set), not because 0.956 was wrong for
what it measured.

### Matched-intersection model-vs-population decomposition

Intersection: loans in the TEST split of both the old (`_hist`) and new (`_30y`) `cutoff_2002`
builds, same reference months — **44,604 (loan_id, ref_month) pairs, 4,911 loans**
(`outputs/diag/matched_intersection_2003_pairs.csv`), ~12-14% of either population (each build
independently re-randomizes its train/test split; old-test ∩ new-test = 8,301 loans out of
60,890/55,409). Four cells (`scripts/score_matched_intersection_2003.py`, jobs
19059011/19061232/19061233/19061234): pass axis (old-pass vs. new-pass raw combined-pass) confirmed
a pure no-op — byte-identical pooled ratios across `{old,new}ckpt_{old,new}pass` pairs, as expected
since both passes run the same current loader and intersection loans survive both cell samples by
construction.

**Model axis: old checkpoints 0.9723 vs. new `_30y` checkpoints 0.9341** — the opposite direction
from the full-population comparison (0.8734 old vs. 0.8982 new). Per-seed count ratios: old
{1.0163, 0.9380, 0.8866, 0.9964, 1.0242}, new {0.9667, 0.8735, 0.9217, 0.9057, 1.0030} — both
5-seed ranges span roughly ±0.09 around their respective means, i.e. the 0.038 gap between the two
ensembles' pooled ratios is within a single seed's typical noise band for either checkpoint set.
**Both population (which 44,604-pair population is being scored) and model (which 5 checkpoints
score it) were held fixed in turn by this design** — population contributes nothing (no-op,
confirmed empirically) and the full difference traces to the model axis, though that difference is
not large relative to seed-to-seed noise. Not reconciled against the full-population direction this
session — reported as-is.

**Dispersion note.** `pooled_comparison()`'s (`forecast_matched_population_cpr.py`) `n_loans≥5000`
(distinct-loan) restriction, applied to the full `_30y` ensemble's per-coupon table, leaves only
**2 of 12 coupons** (6.0, 7.0) — the dispersion/max-min-ratio metric is not usable at this
threshold for this window; the broader `n_loan_months≥100` convention used elsewhere in this
project's `dispersion_stats.csv` is a materially looser restriction on a different unit (row count,
not distinct loans).

### Number audit (this section)

| number | file | key |
|---|---|---|
| 142/140, 64 reversions | `logs/verify_term_mod_18986341.out` (job 18986341) + user-supplied per-loan breakdown | monotonicity violation detail |
| D.4 4,416/403,889, 2,193/195,749 | `logs/verify_term_mod_18986341.out` | `D.4` lines, both vintages |
| 694/729 cells, 1,482,004/19,763,814 loans, 1,292,272 events | `logs/build_cell_grid_sample_30y_19004194.log`, `outputs/pre2013_cell_sample_30y_summary.csv` | log tail + summed `n_events_selected`/`n_selected` |
| 277,042 loans, 3,595,798 loan-months, 99,227 events, 2.7595% | `outputs/census_panel_baseline_cutoff_2002_30y.json` | `overall.{n_loans,n_eligible_loan_months,n_prepay_events,raw_prepay_rate}` |
| a_eff=1.000, b_eff=0.2696, log=1.311 | this session's `scripts/diag/census_check_2002_30y.py` stdout | printed a_eff/b_eff/log_ratio |
| 269,536 train+test loans, 833,720/208,187 obs | `logs/check_build_30y_19053097.out` | gate output (passing run) |
| AUC 0.7744-0.7759 | `outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s{seed}/results.json` | `best_auc` |
| consistency test diffs | this session's srun output (job 19064330) | printed `max_seq_diff`/`max_pred_diff`/window-check lines |
| pooled 0.8982/0.8818, seeds 0.8365-0.9706 | `outputs/rolling/ensemble_onestep_cutoff_2002_30y/pooled_stats.csv` | rows `model=ensemble`/`model=seed{N}` |
| n=323,203, shortfall 1,633/16,046=10.2% | same file, derived (`pred_rate×n`, `real_rate×n`) | — |
| incentive-bin table | `outputs/rolling/ensemble_onestep_cutoff_2002_30y/per_bin_ratio_disagreement.csv` | `n_loan_months` col |
| month-by-month table | `outputs/rolling/ensemble_onestep_cutoff_2002_30y/month_by_month.csv` | rows `model=ensemble` |
| 0.8734 (old all-term) | README.md Sep 26 section | pooled ratio table, `cutoff_2002/ensemble/count` |
| 0.9562 (old 360-only) | README.md Sep 29 section | "2003 one-step-ahead" table, `group=pooled` |
| intersection 44,604/4,911 | `outputs/diag/matched_intersection_2003_pairs.csv` | row count / distinct `loan_id` |
| old-test∩new-test = 8,301 | this session's npy intersection, stdout | `test_loan_ids_split.npy` set intersection |
| cell pooled ratios 0.9723/0.9341 | `outputs/rolling/matched_intersection_2003/summary_4cell_pooled.csv` | per-cell rows |
| per-seed ranges both cells | derived from `outputs/rolling/matched_intersection_2003/{cell}/all_months.csv` | `h_seed{N}.mean()/realized_event.mean()` |
| dispersion n_loans≥5000 → 2 coupons | `outputs/rolling/ensemble_onestep_cutoff_2002_30y/per_coupon_by_distinct_loans.csv` | rows with `n_loans≥5000` |

Artifacts: `scripts/diag/census_check_2002_30y.py`, `scripts/score_matched_intersection_2003.py`,
`scripts/diag/old_pass_postmod_effect.py`, `scripts/diag/check_build_30y.py` (fixed),
`scripts/score_rolling_one_step.py`/`scripts/score_multiobs_dec_window.py`/`scripts/ensemble_onestep_2002.py`
(`--cell_sample`/`--run_tag` threaded), `scripts/tests/test_train_forecast_consistency.py` (`_30y`
case added). Consistency test passed (above) — current as of this section.

---

## cutoff_2020 control rebuilt 30-year: census, build, five seeds, consistency test, 2021 one-step 0.844, in-sample calibration on both cutoffs (Oct 2-3, 2026)

Same chain as `cutoff_2002`'s `_30y` rebuild above, run for the `cutoff_2020` control. All jobs
submitted with `set -euo pipefail` and `--dependency=afterok` chaining (census → build → gate →
five trainings → five dec-window scorings → five one-step scorings → ensemble); none polled after
submission, all verified against real output (not just exit codes) once complete.

### Census: old vs new

| | old 2020 census | new 2020 `_30y` census |
|---|---|---|
| n_loans | 18,709,686 | 14,122,835 |
| n_prepay_events | 7,155,149 | 5,646,103 |
| non-event loan-months | 585,243,056 | 427,396,836 |
| raw_prepay_rate | 1.2081% | 1.3038% |

Source: `outputs/census_panel_baseline_cutoff_2020_30y.json` (`overall.*`), job 19067717.

### a_eff/b_eff, 10% subsample divided out

`scripts/diag/census_check_2020_30y.py` (committed `0c709d0`), srun cpu_short 48G. Unlike
`cutoff_2002`'s cell-grid stratified sample, `cutoff_2020`'s population is a uniform 10% LOAN
subsample (`load_vintage_filtered`'s `--sample_frac`, applied at Pass-1 discovery, before the
per-loan fixed_fraction `incl_prob` logic) — so `a_eff`/`b_eff` are divided by
`r = sample_loans / census_loans` first, the same step used for the original Sep 25 check:
```
r = 1,378,753 / 14,122,835 = 0.097626
a_eff_raw = 564,939 / 5,646,103 = 0.100058
b_eff_raw = 9,747,052 / 427,396,836 = 0.022806
a_eff = a_eff_raw / r = 1.024916
b_eff = b_eff_raw / r = 0.233602
log(a_eff / b_eff) = 1.478745
IPW-weighted sample prepay rate = 0.013048   (census raw_prepay_rate = 0.013038)
```
Closely matches the Sep 25 pre-fix figures (a_eff=0.9985, b_eff=0.2277, log=1.478) — the 30-year/
post-mod filtering changed the population substantially (see census table above) but left this
particular diagnostic's shape essentially unchanged.

### Build: old vs new

| | old build (trail/GOLDEN_BACKUP) | new `_30y` build |
|---|---|---|
| total loans | 1,870,909 | 1,412,264 |
| train / test loans | 1,496,727 / 374,182 | 1,129,811 / 282,453 |
| train / test obs | 11,230,531 / 2,808,691 | 8,247,014 / 2,064,977 |

Gate (`check_build_2020_30y.py`, job 19067733) **PASSED** on first run (no masked-failure repeat —
`set -e` present from the start this time). One note from the gate log: `loan_purpose_enc` codes
are `[0, 1, 2]` in both train and test, not `[0, 1, 2, 3]` — a subset check (`⊆ {0,1,2,3}`) so this
still passes, it just means no `U`-coded loan survived into this build's sampled population (`U`
is already known to be rare in the modern era — see the Sep 19 CP/U-fix section above, "U drew zero
sampled observations").

### Five seeds, old vs new best_auc

| seed | new `_30y` best_auc | old control best_auc |
|---|---|---|
| 42 | 0.72565 | 0.71638 |
| 7 | 0.72678 | 0.71658 |
| 123 | 0.72450 | 0.71696 |
| 1001 | 0.72674 | 0.71621 |
| 2026 | 0.72527 | 0.71641 |

**The populations differ, so this is not a clean model-only comparison**: new checkpoints trained
on 1,129,811/282,453 train/test loans (360-only, post-mod dropped, fresh 10% subsample); old
checkpoints trained on 1,496,727/374,182 train/test loans (pre-fix, no term/post-mod filtering).
Source: `outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_{cutoff_2020_30y_s{N},f0.2_L33{,_seed7,_seed123,_seed1001_goldenbackup,_seed2026_goldenbackup}}/results.json`, `best_auc`.

### Consistency test — PASSED

`scripts/tests/test_train_forecast_consistency.py --case cutoff_2020_seed42_30y` (case added this
session), srun cpu_short 40G 30min:
```
Check 1 (positive): vintage 2016Q1, n=2,000, max_seq_diff=0.000e+00, max_pred_diff=0.000e+00 (AUC=0.7256461)
Check 1 (negative control a) vs TRAIL_SEQ_DIR: [PASS (bug caught)] 162/162 shared observations disagree (max abs diff 4.144e+00)
Check 2 (positive): months 202012/202106/202111, max abs diff 1.17e-07 to 2.28e-07
Check 2 (negative control b): [PASS (bug caught)] 144,274/144,274 loan-months disagree (max abs diff 1.251e+00)
Total elapsed: 726.0s. All checks passed (1 cases).
```
Unlike the `cutoff_2002` cases, `TRAIL_SEQ_DIR` **is** a `cutoff_2020` build, so negative control
(a) applies here and fired correctly — both negative controls exercised for this cutoff, not just
one.

### 2021 one-step result in full

`scripts/score_rolling_one_step.py` per seed (jobs 19067955/57/59/60/61, each depending on its own
dec-window scoring job 19067928/29/30/31/32 for the Dec-consistency check — no cache-read errors in
any of the 5 dec-window logs, confirmed by `grep -i "cache hit|traceback|error"`), pooled via
`scripts/ensemble_onestep_2020_control.py --run_tag _30y` (job 19067981):

**Pooled, n=1,723,631 loan-months:**

| model | weight | predicted monthly | realized monthly | ratio |
|---|---|---|---|---|
| ensemble | count | 0.02307 | 0.02733 | **0.8441** |
| ensemble | upb | 0.02490 | 0.02937 | **0.8479** |
| seed42 | count | 0.02611 | 0.02733 | 0.9553 |
| seed7 | count | 0.02221 | 0.02733 | 0.8126 |
| seed123 | count | 0.02161 | 0.02733 | 0.7907 |
| seed1001 | count | 0.02026 | 0.02733 | 0.7413 |
| seed2026 | count | 0.02516 | 0.02733 | 0.9206 |

Pooled events (count-weighted): Σh=39,762.3, Σrealized=47,106.0, **shortfall = 7,343.7 (15.6% of
47,106.0 realized events)** — derived from `outputs/rolling/ensemble_onestep_cutoff_2020_control_30y/pooled_stats.csv`
(`pred_rate_monthly×n`, `real_rate_monthly×n`, row `model=ensemble, weight=count`), same method as
`cutoff_2002`'s 1,633.0/16,046.0=10.2% figure above.

**Independent audit's B3 result** (fresh Claude Code session, Oct 3, no README consulted as
evidence — see mistakes log and STANDING TESTS): of 1,555,596 forecast-year-2021 one-step rows
matched 1:1 against the raw `current_interest_rate` field, **11 (0.001%)** have
`current_interest_rate != original_interest_rate` — i.e. the loan's note rate changed between
origination and the 2021 scoring window (rate-modified survivors) on a negligible share of the
`cutoff_2020` `_30y` population. The equivalent `cutoff_2002` figure is **305 of 287,961 (0.106%)**
2003 one-step rows. Source: `logs/audit_partB_19093910.log`, `--- B3 ---` sections, both cutoffs.

**Incentive-bin table** (edges `[-6,-4,-3,-2,-1,-0.5,0,0.5,1,1.5,2,3,4,6]`, count-weighted monthly):

| bin | n | predicted | realized |
|---|---|---|---|
| (-1.0,-0.5] | 9,886 | 0.003012 | 0.005058 |
| (-0.5,0.0] | 170,654 | 0.005050 | 0.007102 |
| (0.0,0.5] | 344,428 | 0.011686 | 0.014781 |
| (0.5,1.0] | 446,997 | 0.025410 | 0.028622 |
| (1.0,1.5] | 376,160 | 0.031784 | 0.037311 |
| (1.5,2.0] | 245,363 | 0.031466 | 0.037145 |
| (2.0,3.0] | 123,459 | 0.029493 | 0.037049 |
| (3.0,4.0] | 6,662 | 0.025487 | 0.035425 |

**Month-by-month** (ensemble, annualized):

| ref_month | n | predicted | realized |
|---|---|---|---|
| 202012 | 168,035 | 0.3111 | 0.2874 |
| 202101 | 163,331 | 0.3035 | 0.3146 |
| 202102 | 158,227 | 0.2853 | 0.3561 |
| 202103 | 152,485 | 0.2480 | 0.2957 |
| 202104 | 148,061 | 0.2256 | 0.2647 |
| 202105 | 144,274 | 0.2249 | 0.2737 |
| 202106 | 140,428 | 0.2192 | 0.2541 |
| 202107 | 136,971 | 0.2239 | 0.2843 |
| 202108 | 133,167 | 0.2315 | 0.2829 |
| 202109 | 129,502 | 0.2233 | 0.2696 |
| 202110 | 126,105 | 0.1975 | 0.2507 |
| 202111 | 123,045 | 0.1840 | 0.2262 |

**Per-coupon with `n_loans`** (distinct loans):

| coupon | n_loans | n_loan_months | predicted | realized | ratio |
|---|---|---|---|---|---|
| 1.5 | 14 | 159 | 0.002244 | 0.012579 | 0.1784 |
| 2.0 | 8,544 | 100,131 | 0.003889 | 0.005683 | 0.6843 |
| 2.5 | 22,514 | 257,829 | 0.007631 | 0.009948 | 0.7671 |
| 3.0 | 51,191 | 530,902 | 0.022458 | 0.025555 | 0.8788 |
| 3.5 | 32,735 | 318,990 | 0.031063 | 0.035694 | 0.8703 |
| 4.0 | 37,171 | 361,192 | 0.030992 | 0.036820 | 0.8417 |
| 4.5 | 9,835 | 95,499 | 0.029544 | 0.037445 | 0.7890 |
| 5.0 | 5,219 | 51,008 | 0.026774 | 0.036288 | 0.7378 |
| 5.5 | 647 | 6,276 | 0.024333 | 0.037444 | 0.6499 |
| 6.0 | 165 | 1,645 | 0.023999 | 0.034043 | 0.7050 |

**Dispersion, both restrictions:**

| version | restriction | n_groups | PRIMARY pred disp | PRIMARY real disp | PRIMARY ratio | SECONDARY max | SECONDARY min | pooled_ratio |
|---|---|---|---|---|---|---|---|---|
| as saved | n_loan_months≥100 | 10 | 13.844 | 6.590 | 2.101 | 0.879 | 0.178 | 0.8441 (full-pop) |
| `pooled_comparison()`-style | distinct n_loans≥5000 | **7** | 7.988 | 6.590 | 1.212 | 0.879 | 0.684 | 0.8457 (n=167,209) |

Unlike `cutoff_2002` (2/12 coupons survived the stricter threshold), **7 of 10 coupons clear
n_loans≥5000 here** (2.0-5.0) — the larger `cutoff_2020` population makes the stricter restriction
usable.

### Three-way comparison

| population | pooled count ratio |
|---|---|
| Sep 26 ensemble, old all-term control (2,397,271 loan-months, 360+non-360 mixed) | **0.8750** |
| Sep 29 `term_report.py` decomposition, old control's 360-only subset (1,736,441 loan-months) | **0.8990** |
| `_30y` control, pure 360-term by construction (1,723,631 loan-months) | **0.8441** |

Same pattern as `cutoff_2002`'s three-way comparison (0.8734/0.9562/0.8982): the new 30-year-only
number sits *below* both old figures here, rather than between them as it did for `cutoff_2002` —
a different relative position, not a contradiction, since both the population and the checkpoint
set changed together in the rebuild (not term filtering alone).

### Frozen-Dec-window 12-month forecast (superseded design), both cutoffs, five seeds

From the existing dec-window scoring outputs — no new scoring needed. **This is not an in-sample
check and not comparable to the one-step numbers.** It is the frozen-December-features 12-month
forecast — window frozen at Dec of the cutoff year, realized over the next 12 months — the design
the Sep 24 one-step-ahead redesign superseded (frozen incentive can't capture a within-year rate
move; see "The frozen Dec-window 2003 test, and why frozen scoring was the wrong design" above).
Reported here for continuity only. Each `dec_window_scores_seed{N}.csv` already saves a realized
side (`realized_prepay`, whether the loan prepaid — `zero_balance_code_actual==1` — anywhere in the
forecast calendar year) alongside the predicted side (`h_t`, monthly hazard;
`annual_pp = 1-(1-h_t)**12`, the proper annualized conversion). Both sides are simple
count-weighted means (`.mean()` over the frozen Dec-cutoff population) — weighted the same way on
both sides.

| cutoff | n | pred (annual_pp) | realized | ratio |
|---|---|---|---|---|
| `cutoff_2002` `_30y`, seed42 | 35,243 | 0.3268 | 0.4553 | 0.7177 |
| `cutoff_2002` `_30y`, seed7 | 35,243 | 0.3149 | 0.4553 | 0.6915 |
| `cutoff_2002` `_30y`, seed123 | 35,243 | 0.3336 | 0.4553 | 0.7327 |
| `cutoff_2002` `_30y`, seed1001 | 35,243 | 0.3222 | 0.4553 | 0.7077 |
| `cutoff_2002` `_30y`, seed2026 | 35,243 | 0.3273 | 0.4553 | 0.7188 |
| `cutoff_2002` `_30y`, **ensemble** | 35,243 | 0.3249 | 0.4553 | **0.7137** |
| `cutoff_2020` `_30y`, seed42 | 168,035 | 0.2978 | 0.2803 | 1.0623 |
| `cutoff_2020` `_30y`, seed7 | 168,035 | 0.2884 | 0.2803 | 1.0287 |
| `cutoff_2020` `_30y`, seed123 | 168,035 | 0.2779 | 0.2803 | 0.9914 |
| `cutoff_2020` `_30y`, seed1001 | 168,035 | 0.2798 | 0.2803 | 0.9980 |
| `cutoff_2020` `_30y`, seed2026 | 168,035 | 0.3002 | 0.2803 | 1.0709 |
| `cutoff_2020` `_30y`, **ensemble** | 168,035 | 0.2888 | 0.2803 | **1.0303** |

Source for every row: `outputs/rolling/dec_window_cutoff_{2002,2020}_30y_seed{N}/dec_window_scores_seed{N}.csv`, columns `annual_pp`/`realized_prepay`, computed this session (not previously saved as a table anywhere).

**Next to the Sep 23 `ipw_consistent_gap.py` figures** (both sides IPW-weighted, on the TRAIN/TEST
SAMPLE population — a different population and a different weighting than the frozen Dec-window
table above, not a direct replication): `cutoff_2020` = 1.0269x; `cutoff_2002` seed42 = 1.0696x,
seed7 = 1.0212x (README, "cutoff_2002: seed replication..." section, Sep 21/corrected Sep 23).
`cutoff_2020`'s new Dec-window ensemble figure (1.0303) lands almost exactly on top of its Sep 23
one-step-sample figure (1.0269) despite the population/weighting difference. `cutoff_2002`'s does
not (0.7137 new Dec-window vs. 1.02-1.07 old one-step-sample) — a real divergence, not a
computation error (re-verified: using `h_t` directly instead of `annual_pp` gives nonsense ~0.08
ratios from comparing a monthly rate to an annual flag, which is the wrong-units mistake to rule
out first; `annual_pp` is the correct conversion and what's reported above).

**Frozen-Dec-window 12-month forecast, superseded design, reported for continuity: `cutoff_2002`
0.7137, `cutoff_2020` 1.0303; not comparable to the one-step numbers (`cutoff_2002` one-step pooled
0.8982, `cutoff_2020` one-step pooled 0.8441) and not an in-sample check.**

### What the two rebuilt tests say together

`cutoff_2002`'s 2003 one-step shortfall concentrates at low-to-moderate incentive (bins up to
~1.5) and is roughly calibrated above it (bin table above, Oct 1-2 section). `cutoff_2020`'s 2021
one-step shortfall, by contrast, shows up in **every** incentive bin and at **both** coupon ends
(per-coupon table above: ratio 0.18 at the sparse 1.5 coupon tail, climbing to ~0.88 at 3.0, falling
back to 0.65-0.70 at 5.5-6.0 — a shortfall on both sides, not concentrated at one end the way
`cutoff_2002`'s is at the low-incentive end). Both pooled ratios (0.898, 0.844) sit **below** the
corresponding all-term model's own 360-only subset ratio (0.956, 0.899) — the 30-year restriction
changes which population is being scored and, for `cutoff_2020`, the Dec-window frozen-window
calibration looks fine (1.03) even though the one-step rolling test (0.844) does not, the opposite
pattern from `cutoff_2002` (Dec-window 0.71, one-step 0.898) — and the matched-intersection
decomposition for `cutoff_2002` already showed the model-vs-population gap is within a single
seed's typical noise band (seed ranges 0.74-0.96 and 0.79-1.07 here are comparably wide). **The
30-year restriction fixed the incentive-measurement mismatch (15-year loans scored against the
30-year PMMS) that the Sep 29 term-mix finding diagnosed, but it did not close the predicted-vs-
realized gap** — the gap is still there under the cleaner, term-matched population; it has simply
moved to look different (concentrated vs. uniform across bins, Dec-window-fine vs. Dec-window-off)
between the two cutoffs rather than disappearing.

### Number audit (this section)

| number | file | key |
|---|---|---|
| census totals (14,122,835 / 5,646,103 / 427,396,836 / 1.3038%) | `outputs/census_panel_baseline_cutoff_2020_30y.json` | `overall.*` |
| a_eff=1.025, b_eff=0.234, log=1.479, r=0.0976 | this session's `scripts/diag/census_check_2020_30y.py` stdout | printed lines |
| build loans/obs (1,412,264 / 1,129,811+282,453 / 8,247,014+2,064,977) | `logs/multiobs_2020_30y_19067719.out` | "Total loans"/"Train:"/"Test:"/shape lines |
| gate n_loans/n_obs (1,378,753 / 10,311,991) | `logs/check_build_2020_30y_19067733.out` | gate output |
| loan_purpose_enc codes [0,1,2] | `logs/check_build_2020_30y_19067733.out` | "distinct loan_purpose_enc codes" lines |
| five best_auc (0.72450-0.72678) | `outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_cutoff_2020_30y_s{seed}/results.json` | `best_auc` |
| five old best_auc (0.71621-0.71696) | `outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33{...}/results.json` | `best_auc` |
| consistency test diffs/counts | this session's srun output (job 19086048) | printed Check1/Check2/negative-control lines |
| dec-window AUC verifications (5x PASSED) | `logs/score_decwin_2020_30y_s{seed}_*.out` | "Checkpoint AUC verification" lines |
| pooled 0.8441/0.8479, seeds 0.7413-0.9553 | `outputs/rolling/ensemble_onestep_cutoff_2020_control_30y/pooled_stats.csv` | rows `model=ensemble`/`model=seed{N}` |
| n=1,723,631 | `logs/ensemble_2020_30y_19067981.out` | "Population identity check PASSED" |
| incentive-bin table | `outputs/rolling/ensemble_onestep_cutoff_2020_control_30y/per_bin_ratio_disagreement.csv` | `n_loan_months` col |
| month-by-month table | `outputs/rolling/ensemble_onestep_cutoff_2020_control_30y/month_by_month.csv` | rows `model=ensemble` |
| per-coupon n_loans table | `outputs/rolling/ensemble_onestep_cutoff_2020_control_30y/per_coupon_by_distinct_loans.csv` | full table, computed this session |
| dispersion n_loans≥5000 → 7 coupons | same file | rows with `n_loans≥5000` |
| 0.8750/0.8990 (old all-term/360-only) | README.md Sep 26/Sep 29 sections | pooled ratio tables |
| Dec-window in-sample table (both cutoffs) | `outputs/rolling/dec_window_cutoff_{2002,2020}_30y_seed{N}/dec_window_scores_seed{N}.csv` | `annual_pp`/`realized_prepay` columns, computed this session |
| Sep 23 one-step-sample figures (1.0269/1.0696/1.0212) | README.md "cutoff_2002: seed replication..." section | `ipw_consistent_gap.py` correction text |
| D.4 denominators (589,311/278,641) | `.claude_tmp/verify_term_mod_filters_18992408.log` (not committed) | `D.4` lines, both vintages |


## Oct 5–6, 2026 — seq schema, cache fingerprint, 2002 `_seq` result, 2003–2005 launch

**What changed in the pipeline.**
- **`harp_eligible` as the tenth feature column (all zeros).** `FEATURE_COLS` in the three `_zbc`
  builders gains `harp_eligible` as its last entry (commit e96857f); no eligibility logic. The
  trainer now takes `input_dim` from `train_seq.shape[-1]` instead of a hardcoded 9. This is the
  only change to the training script between the `_30y` and `_seq` runs.
- **30-year rule by origination row (189ddd0).** A loan is kept iff `original_loan_term` on its
  earliest reporting-period row is 360, replacing "drop if any row is non-360". A loan whose term
  changes at modification is now kept. The smoke test's kept counts on 2002Q1 and 2018Q1 (594,424
  and 367,395) match the prior rule's reference numbers.
- **Cache-key fingerprint (17d5120).** The decwin raw-pass cache was keyed on a hand-bumped
  `CACHE_VERSION` that was not bumped when `harp_eligible` landed, so all ten `_seq` decwin jobs read
  the Oct 2 pre-`harp_eligible` pickle and died with `KeyError: ['harp_eligible'] not in index`. The
  key now includes a blake2b fingerprint of `FEATURE_COLS` (cache filenames gain a `_feat<8 hex>`
  suffix), so a schema change selects a new file instead of silently reusing a stale one.
- **`SCHEMA_SMOKE` stage (7a5b1af, default on).** After the gate, the driver submits a 1-epoch
  seed-42 train and the train/forecast consistency test on that checkpoint; the ten real trainings
  depend on the consistency-test job, not the gate. A schema divergence is caught before ten GPU
  trainings are spent on it.

**`cutoff_2002_seq` result (10 seeds: 42, 7, 123, 1001, 2026, 3, 11, 77, 314, 999).**
- Ensemble one-step pooled ratio: **0.8798** by count, **0.8573** by UPB.
- Per-seed range: count 0.8050 (seed 77) to 0.9830 (seed 999); UPB 0.7806 (seed 77) to 0.9696
  (seed 999).
- SE of the ten per-seed ratios (sd/√10, computed from the per-seed rows): 0.0212 count, 0.0220 UPB.
- Seeds 42, 7, 123, 1001, 2026 only (the five that also exist as `_30y`): mean per-seed ratio 0.8883
  count / 0.8661 UPB, vs. the `_30y` five-seed ensemble **0.8982** count / **0.8818** UPB. The other
  five seeds average 0.8714 count / 0.8486 UPB.
- Month by month (annualized, ensemble, 12 reference months 200212–200311): predicted is below
  realized in 9 of 12 (200301–200309); above in 200212, 200310, 200311.

**Same data, different initialization — measured vs. inferred.** *Measured:* the census JSON and CSV
are byte-identical to `_30y` (`cmp`); the `_seq` and `_30y` decwin raw-pass caches share the same
`pop_hash` (`73fab0fbc4de189c`, in the cache filenames); and `scripts/diag/compare_seq_30y_vs_seq_2002.py`
(job 19313234, `logs/compare_seq_30y_vs_seq_2002_19313234.out`) found `train_seq.npy` (833,720 × 33 × 10
vs. 833,720 × 33 × 9) and `test_seq.npy` (208,187 × 33 × 10 vs. 208,187 × 33 × 9) identical on the first
nine columns (max abs diff 0, all chunks), the tenth column all zeros, and the other 24 `.npy`
arrays (mask, labels, loan ids and splits, ref_month, incl_prob, etc.) identical. *Inferred:* the
0.8883-vs-0.8982 gap on the same five seeds is therefore initialization noise from the changed
parameter shapes, by elimination — the data are identical and the only training-script change is
`input_dim` (e96857f); the seed-42 `_30y` and `_seq` trainings use the same arguments apart from the
run tag. Not tested directly (no 9-column retrain on the new code, no second 10-column draw per seed).

**Cache-contamination check (verdict as of Oct 6).** No feature-value change landed between the
Oct 2 cache build and the `_30y` jobs; 189ddd0 came three days later (Oct 5). The `_seq` results
were scored from the fresh `..._feat6faa879d.pkl` cache (file time Oct 6 11:35), not the stale
pickle that crashed the first ten decwin jobs.

**Cutoffs 2003, 2004, 2005 launched (Oct 6).**
- Driver verified before launch: no hardcoded 2002, run tag, or output path in the driver or in any
  script it calls (the 2002 mentions are docstrings, usage examples, and a named-case table the
  driver does not use); all paths are keyed by cutoff year and seed. `--include_pre2013` prepends
  `PRE2013_VINTAGES` to the vintage list and the cutoff then drops vintages starting after Dec of
  the cutoff year, so a pre-2013 cutoff uses only 2000Q1 through the cutoff year's Q4. Census and
  build for every cutoff use `outputs/pre2013_cell_sample_30y_loans.csv`.
- Partition: census and build on `cpu_short`, `--time=5:45:00`. Basis: `cutoff_2002_seq` census took
  33:12 and build 46:13 (sacct, jobs 19255256 and 19255257); times three is 1:39:36 and 2:18:39,
  under the 5:30:00 threshold. Peak RSS was 66.4 GB (census) and 87.1 GB (build) against 96 GB
  requested. A timed-out build resumes on resubmission into the same directory (Pass 1/2 per-vintage
  pickles, Pass 3 per-vintage shards).
- Job ids (ranges are first to last of each chain; other users' ids fall inside them):
  2003 = 19312977–19313023 (census 19312977, build 19312978, gate 19312979, ensemble 19313023);
  2004 = 19313024–19313093 (census 19313024, build 19313025, gate 19313026, ensemble 19313093);
  2005 = 19313095–19313156 (census 19313095, build 19313096, gate 19313099, ensemble 19313156).
  Each chain is 36 jobs. For one train job per cutoff (seed 42: 19312982, 19313033, 19313105),
  `scontrol` shows `Dependency=afterok:` the cutoff's own smoke-test job.
- State when written: 2003 census running; the 2004 and 2005 censuses pending on
  `QOSMaxMemoryPerUser` (96 GB each), so the three CPU stages run effectively one after another.

### Number audit (this section)

| number | file | key |
|---|---|---|
| 0.8798 / 0.8573 (10-seed ensemble count / UPB) | `outputs/rolling/ensemble_onestep_cutoff_2002_seq/pooled_stats.csv` | rows `model=ensemble` |
| per-seed ranges 0.8050–0.9830 / 0.7806–0.9696 | same file | `ratio`, rows `model=seed{N}` |
| SE 0.0212 / 0.0220 | same file | sd(ddof=1)/√10 of the ten `ratio` values, computed this session |
| 0.8883 / 0.8661, 0.8714 / 0.8486 | same file | means of seeds {42,7,123,1001,2026} / the other five, computed this session |
| 0.8982 / 0.8818 | `outputs/rolling/ensemble_onestep_cutoff_2002_30y/pooled_stats.csv` | rows `model=ensemble` |
| 9 of 12 months below | `outputs/rolling/ensemble_onestep_cutoff_2002_seq/month_by_month.csv` | rows `model=ensemble`, predicted vs. realized annualized |
| 26 arrays identical; shapes; tenth column zeros | `logs/compare_seq_30y_vs_seq_2002_19313234.out` | per-array lines and VERDICT |
| census byte-identical | `outputs/census_panel_baseline_cutoff_2002_{30y,seq}.{json,csv}` | `cmp`, this session |
| pop_hash 73fab0fbc4de189c | `outputs/rolling/_dec_window_raw_cache/_raw_combined_pass_73fab0fbc4de189c_trunc200312_v2_upb{,_feat6faa879d}.pkl` | filenames |
| 33:12 / 46:13; 66.4 GB / 87.1 GB | `sacct -j 19255256`, `sacct -j 19255257` | Elapsed; `.batch` MaxRSS (66370368K, 87075292K) |
| fresh cache used by `_seq` scoring | `logs/decwin_2002_seq_s42_19291350.out` | "Cache hit (combined)" line |
| 594,424 / 367,395 | commit message of 189ddd0 | smoke-test line |
| job ids | `.claude_tmp/submit_{2003,2004,2005}.out` (not committed) | driver stdout |
