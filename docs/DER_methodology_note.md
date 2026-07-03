# Diep–Eisfeldt–Richardson (2021): Methodology Note

*Reference: "The Cross Section of MBS Returns," Journal of Finance 76(5), 2093–2151.
NBER WP 22851 (free full text). Companion: Gabaix, Krishnamurthy, Vigneron (2007).*

*Last updated: July 3, 2026 — reflects factor-shock Fama-MacBeth (full-sample + rolling t->t+1), AR(1) robustness test, and debias-attempt diagnosis.*

---

## 1. The Core Model

Expected **treasury-hedged excess return** on coupon *i* is linear in two prepayment-risk
loadings times their prices of risk:

```
E[R_e^i] = λx · β_x^i  +  λy · β_y^i
```

- `β_x^i` = loading on the **level / "turnover"** factor (prepayment shifts up/down across all coupons)
- `β_y^i` = loading on the **rate-sensitivity / "refi"** factor (borrower responsiveness to rate incentive, conditional on moneyness)
- `λx, λy` = prices of risk; **they change sign with market composition** (discount-heavy vs premium-heavy)

Securities are defined by **relative coupon** = `c^i − c_par`, NOT absolute coupon.
Discount: `c^i − c_par < 0`. Premium: `c^i − c_par > 0`.

### Economic Intuition (Why Signs Flip)
- **Discount** MBS (price < par) is prepaid at par → prepayment is **value-increasing** → `β_x > 0`, `β_y = 0`.
- **Premium** MBS (price > par) is prepaid at par below market value → prepayment is **value-decreasing** → `β_x < 0`, `β_y < 0`.
- Loadings are **monotonic in |relative coupon|**: bigger discount/premium → bigger |loading|.

---

## 2. How the Factors Are Built (Eq. 15–18)

The factors are **prepayment forecast errors**, decomposed each month by a cross-coupon regression.

**Step A — Cross-coupon regression on FORECAST prepayments (each month):**
```
ppmt_forecast_t[i] = x_forecast_t + y_forecast_t · max(0, m^i − m_PMMS)
```
- `m^i` = WAC (weighted average coupon = borrower note rate) of MBS coupon i
- `m_PMMS` = Freddie Mac Primary Mortgage Market Survey rate (monthly)
- Intercept `x̂_forecast` = forecast level; slope `ŷ_forecast` = forecast rate-sensitivity

**Step B — Same regression on REALIZED prepayments:**
```
ppmt_realized_t[i] = x_realized_t + y_realized_t · max(0, m^i − m_PMMS)
```

**Step C — Factor shocks are forecast errors:**
```
x_t = x̂_realized_t − x̂_forecast_t      (Eq 17)   level shock
y_t = ŷ_realized_t − ŷ_forecast_t      (Eq 18)   rate-sensitivity shock
```
Properties in DER data: corr(x,y) ≈ 0.13; autocorr x ≈ 0.78, y ≈ 0.66.

---

## 3. Estimating Loadings and Prices of Risk

**First stage (Eq. 19) — time-series regression, per coupon:**
```
R_e^i_t = a^i + β_x^i · x_t + β_y^i · y_t + ε^i_t
```
Regress each coupon's hedged excess return on the two factor shocks → β_x^i, β_y^i.

**Second stage (Eq. 20) — Fama–MacBeth cross-section, split by market type:**
```
R_e^i_t = a_t + λx · β̂_x^i + λy · β̂_y^i + ε^i_t
```
- **Discount market (DM):** PMMS > WAC proxy (3.5%); rates above par coupon
- **Premium market (PM):** PMMS < WAC proxy; borrowers hold in-the-money refi options
- Prediction (Hypothesis 1): PM → λx < 0, λy < 0;  DM → λx > 0, λy < 0
- Multicollinearity fix: drop months where cross-section corr of loadings > 0.90

---

## 4. DER Data Sources (for Reference)

| Series | Source | Frequency |
|---|---|---|
| Hedged excess returns by coupon | Bloomberg Barclays Hedged MBS Return indices (Fannie 30yr) | Monthly, since 1994 |
| Forecast prepayments by coupon | Bloomberg median dealer survey (base rate scenario) | Monthly |
| Realized prepayments by coupon | eMBS | Monthly |
| WAC by coupon | Collected per coupon | Monthly |
| Current mortgage rate | Freddie Mac PMMS | Weekly → monthly avg |
| Market composition (%RPB discount) | From coupon prices vs par | Monthly |

---

## 5. How Our Pipeline Maps to the DER Framework

### Our Contribution: ML Hazard Model as Dealer Survey Substitute

The paper uses the **Bloomberg median dealer survey** as the forecast leg of factor construction.
Our trained Transformer hazard model produces **predicted CPR by coupon** — substituting for
the proprietary dealer survey as `ppmt_forecast`. This is the core methodological contribution.

```
x_t, y_t  =  (realized CPR cross-coupon regression)  −  (hazard model CPR cross-coupon regression)
```

**Current implementation (analytical betas, not factor shocks):**

Rather than building factor shocks via the two-stage regression (Eq. 17–19), the current
Stage 2/3 pipeline uses the **price-formula betas** derived analytically from DER Lemma 1:

```
β_x^i = (r_t − c^i) / [(r_t + φ^i)(φ^i + c^i)]
β_y^i = β_x^i · max(0, m^i − r_t)
```

where `r_t` = PMMS (time-varying, per month), `c^i` = coupon, `φ^i` = mean CPR from hazard model,
`m^i` = note rate = coupon + 0.75. These are recomputed each month using that month's actual PMMS.

| Component | DER Paper | Current Implementation |
|---|---|---|
| Factors x, y | Realized − forecast CPR (cross-coupon regression) | **Built** — `stage3_der_factor_shocks.py` (2026-07-03) |
| β_x, β_y | Time-series regression of returns on x, y shocks | Empirical betas, per DER Eq. 19 (both analytical and empirical versions now maintained side by side) |
| Forecast leg | Bloomberg dealer survey | Hazard model CPR — full-sample (θ_full) and rolling t→t+1 (θ_{t-}), both implemented |
| Realized leg | eMBS | Fannie Mae panel (v6) |
| TBA returns | Bloomberg Barclays indices | Bloomberg FNCL TBA prices (Bobst) |

Note: analytical betas (Lemma 1, §5 above) remain in production for the DER Eq. 22
regression; the empirical/factor-shock betas below are a separate, additional
estimation, not a replacement.

---

## 6. Current Results (Stage 3 Fama-MacBeth, 21-Vintage Model)

**Data:** 100 months, Jan 2019–May 2026. FNCL coupons 2.5–6.5% (9 securities).
Treasury hedge: D_mod = 6.5yr, blended 5yr/10yr UST. Market split: PMMS vs WAC proxy 3.5%.

| Market | Months | λ_x mean | t-stat | p-value | Sign correct? |
|--------|--------|-----------|--------|---------|---------------|
| Discount (DM) | 76 | +0.000016 | +0.20 | 0.84 | ✅ |
| Premium (PM) | 24 | −0.000639 | −2.15 | 0.042 | ✅ |

**PM result is robust** across all three model versions (9/13/21-vintage).

**DM result:** correct sign but not significant (p ≈ 0.84). This is structural — in the current
rate environment all 9 FNCL coupons trade at a discount, leaving insufficient cross-sectional
variation in β_x to identify λ_x. This is a data-scope limitation, not a model flaw.

### Forecast vs. Realized CPR Validation (Core Contribution)

The 21-vintage expansion (2018Q1–2023Q1) enables validation across the full rate cycle:

| Coupon | Period | Forecast CPR | Realized CPR |
|--------|--------|-------------|--------------|
| FNCL 4.5% | Peak 2020–21 (PM) | 4.6% | 4.5% |
| FNCL 6.5% | Trough 2022–23 (DM) | 2.7% | 2.7% |

Prior to adding 2018–19 vintages, the model underestimated premium-regime CPR by 4–7×.
Root cause: the hazard model required 2018–19 originations whose first 33 months of
performance span the 2020–21 refi boom. Vintage composition directly determines
what rate regimes the model can generalize to.

### Factor-Shock Fama-MacBeth (DER Eq. 15–19, empirical betas)

Built 2026-07-03. `shock[c,t] = realized_CPR[c,t] − forecast_CPR[c,t]`, decomposed each
month into level (f_level) and rate-sensitivity (f_slope) innovations via DER's cross-
sectional regression (Eq. 15–18), then empirical betas via time-series regression of
returns on the two factors, then monthly Fama-MacBeth on those betas.

| Forecast leg | n (months) | λ_x | t(λ_x) | λ_y | t(λ_y) | corr(b_x,b_y) |
|---|---|---|---|---|---|---|
| Full-sample (θ_full) | 72 | 0.057 | 2.35 | 0.169 | 1.58 | 0.402 |
| Rolling t→t+1 (θ_{t-}) | 48 | 0.149 | **3.04** | 1.263 | 1.52 | 0.390 |

*Note: an earlier full-sample run reported t=2.52, n=77 (sent 2026-06-27) — a bug in the
Fama-MacBeth restriction let 5 months without valid factor coverage get priced against
betas that never saw them. Corrected above; coefficient and conclusions are essentially
unchanged, significance is slightly weaker but still significant at 5%.*

**Rolling result is the genuine ex-ante test advisor requested 2026-07-03**: θ_{t-} only ever
uses information available before the forecast month (cutoff_2020→2021, cutoff_2021→2022,
cutoff_2022→2023, cutoff_2023→2024). n drops from 72 to 48 (only OOS months have a valid
rolling forecast) — the honest cost of a clean test. λ_x **survives and strengthens**
(t: 2.35→3.04) despite the smaller sample. The λ_x/λ_y decorrelation (0.40 vs. 0.70+ in
DER's own paper, cited as the differentiator vs. DER's collapsed single-factor result) is
essentially unchanged (0.402→0.390) under the rolling test. Pre-2020-only training was
considered and rejected (see Limitation 2) — full rolling is the only viable clean-OOS design
given this dataset.

**λ_y is not currently reportable as a finding.** It moves 0.169→1.263 between full-sample
and rolling — an implausible jump traceable to the 2022–23 rolling forecast overshooting
realized CPR by 10–40× in specific discount-regime coupon-months (see Debiasing below).
λ_x is the headline result; λ_y needs the debiasing problem resolved before its magnitude
means anything.

**Debiasing (advisor's request, 2026-07-03):** DER's own paper (Sec. IV.B.1) reports x_t, y_t
are autocorrelated (ρ=0.78, 0.66) but treats them as legitimate "surprises" anyway — dealer
forecast models are structurally slow-to-update, so persistence is expected, not
contamination. DER's own robustness check is an AR(1) residualization of the pooled shock
series, reported as "nearly identical" to the raw result.

We replicated this on the full-sample series: ρ_x=0.911, ρ_y=0.573 (broadly in line with
DER's own ρ's). **Unlike DER, our result is NOT robust to this test**: λ_x drops from
t=2.35 (raw) to t=1.08 (AR(1)-residualized, n=71, mean=0.039, p=0.28). This suggests a
real share of the full-sample λ_x significance is attributable to persistent/forecastable
structure in a single fixed hazard model, rather than genuine period-to-period surprise —
directly responsive to advisor's concern, and a genuine finding rather than a null result.

Four attempts at a per-cutoff-model debias of the **rolling** shock series (additive,
log-space, log-space excluding cutoff_2020) were tried and abandoned — each broke the
cross-section (corr(b_x,b_y): 0.39→0.51→−0.58). Diagnosis: rolling shock variance is 53%
time-driven, only 8% coupon-driven, with the trend **reversing sign** across cutoffs
(cutoff_2021 shock decays −1.4 log-points over its forecast year; cutoff_2023 shock
*rises* +0.35). A single scalar bias per cutoff cannot represent this — it is a
specification mismatch with DER's implicit stationary-AR(1) precondition, which holds on
DER's 250-month continuous survey series but not on our four independent 12-month rolling
segments. This is an open problem, not a bug; correctly left unresolved rather than shipped
broken.

---

## 7. Known Limitations and Next Steps

### Limitation 1: Analytical Betas vs. Factor Shocks — RESOLVED 2026-07-03
Factor shocks (realized − forecast CPR, cross-coupon regression) and empirical betas are
now built (`stage3_der_factor_shocks.py`); see §6. Analytical (Lemma 1) betas remain in
production for the Eq. 22 regression, retained separately rather than replaced. Open
item: λ_y magnitude is not yet reliable pending resolution of the rolling-series debias
problem (see §6, Debiasing).

### Limitation 2: Out-of-Sample Validation — PARTIALLY RESOLVED 2026-07-03
Option (b), pre-2020-only training, was attempted and formally ruled out: calendar-censoring
at Dec 2019 gives ~0% in-window prepayment events (2026-06-17), and an extended 2013–2019
panel reproduces the same failure — an inverted/flat refi S-curve, because in-the-money
loans in 2013–2019 mostly didn't refinance; the rate-driven response only exists in the
2020–21 boom the test is trying to hold out (2026-06-20). This is a genuine data constraint,
not a modeling gap — no pre-2020-only model can learn the behavior it needs to forecast.

Rolling t→t+1 estimation (predict each period from date-t information only) is now built
and validated: four cutoff models (2020–2023), each forecasting the following year. See §6
for the resulting factor-shock Fama-MacBeth (λ_x survives and strengthens under this
genuine OOS design). Option (a), projecting into 2023Q2+, remains open pending additional
vintage data (Limitation 4).

### Limitation 3: DM Identification
With 8 of 9 FNCL coupons in the discount regime throughout 2022–2026, there is insufficient
cross-sectional beta spread to identify λ_x in the DM subsample. Full DM identification
requires data spanning a market with mixed discount/premium composition.

### Limitation 4: Missing Vintages
2023Q2–Q4 vintages not yet downloaded. These would extend the realized CPR series and
improve OOS coverage.

---

## 8. Infrastructure

| Component | Detail |
|---|---|
| Hazard model | `scripts/train_hazard.py` — PrepaymentTransformer, AUC 0.7999 (21 vintages) |
| Platt calibration | a=0.4934, b=−4.840 — must recompute after every retrain |
| Stage 2 betas | `scripts/stage2_der_betas.py` — φ range 0.011–0.033, all Lemma checks pass |
| Stage 3 regression | `scripts/stage3_der_regression_v2.py` — Fama-MacBeth (analytical betas), 100 months |
| Stage 3 factor shocks | `scripts/stage3_der_factor_shocks.py` — Fama-MacBeth (empirical betas), full-sample + rolling |
| Rolling forecast leg | `scripts/stage2_forecast_cpr_rolling.py` — per-month PMMS re-scoring through θ_{cutoff(t)} |
| Rolling checkpoints | `outputs/rolling/cutoff_{2020,2021,2022,2023}/hazard_best.pt` |
| Cohort-CPR Platt | `config/hazard_calibration_cpr_forecast.json` (a=0.4559, b=−3.1376) — forecast-leg only, never mix with OAS Platt |
| Realized CPR | `scripts/realized_cpr_v6.py` — global cross-file Pass 0, 21 vintages (v5 has a known MMYYYY-sort bug, do not use) |
| Forecast CPR | `outputs/forecast_cpr_timeseries_gfee050.csv` (full-sample), `outputs/rolling_forecast_oos_gfee050.csv` (rolling) |
| Results | `outputs/factor_shock_results.json`, `outputs/factor_shock_lambda_ts.csv`, `outputs/forecast_vs_realized_cpr.csv` |
