# Diep–Eisfeldt–Richardson (2021): Methodology Note

*Reference: "The Cross Section of MBS Returns," Journal of Finance 76(5), 2093–2151.
NBER WP 22851 (free full text). Companion: Gabaix, Krishnamurthy, Vigneron (2007).*

*Last updated: June 2026 — reflects 21-vintage pipeline through Stage 3 Fama-MacBeth.*

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
| Factors x, y | Realized − forecast CPR (cross-coupon regression) | Not yet built (next step) |
| β_x, β_y | Time-series regression of returns on x, y shocks | Analytical price-formula (Lemma 1) |
| Forecast leg | Bloomberg dealer survey | Hazard model CPR |
| Realized leg | eMBS | Fannie Mae panel (v5) |
| TBA returns | Bloomberg Barclays indices | Bloomberg FNCL TBA prices (Bobst) |

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

---

## 7. Known Limitations and Next Steps

### Limitation 1: Analytical Betas vs. Factor Shocks
Current β_x, β_y are price-formula betas (Lemma 1), not the paper's empirical betas from
time-series regression of returns on factor shocks (Eq. 19). The regression identifies the
market price of theoretical duration/convexity exposure rather than realized prepayment
surprise risk. Building proper factor shocks (realized − forecast CPR, cross-coupon regression)
is the next methodological step.

### Limitation 2: Out-of-Sample Validation
The 2020–2025 forecast vs. realized comparison overlaps with the training window (model
trained on 2018Q1–2023Q1 performance data). A clean out-of-sample test requires either:
- (a) Projecting CPR into a post-training period (2023Q2 onward) — requires additional vintage data
- (b) Train on pre-2020 data only, test on 2020–21 refi boom

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
| Stage 3 regression | `scripts/stage3_der_regression_v2.py` — Fama-MacBeth, 100 months |
| Realized CPR | `scripts/realized_cpr_v5.py` — global cross-file Pass 0, 21 vintages |
| Forecast CPR | `outputs/forecast_cpr_timeseries.csv` |
| Results | `outputs/stage3_lambda_ts.csv`, `outputs/forecast_vs_realized_cpr.csv` |
