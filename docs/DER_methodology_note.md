# Diep–Eisfeldt–Richardson (2021): Methodology Note for TBA Return Project

*Reference: "The Cross Section of MBS Returns," Journal of Finance 76(5), 2093–2151.
NBER WP 22851 (free full text). Companion: Gabaix, Krishnamurthy, Vigneron (2007).*

---

## 1. The core model

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

### Economic intuition (why signs flip)
- **Discount** MBS (price < par) is prepaid at par → prepayment is **value-increasing** → `β_x > 0`, `β_y = 0`.
- **Premium** MBS (price > par) is prepaid at par below market value → prepayment is **value-decreasing** → `β_x < 0`, `β_y < 0`.
- Loadings are **monotonic in |relative coupon|**: bigger discount/premium → bigger |loading|.

---

## 2. How the factors are built (Eq. 15–18) — the key procedure

The factors are **NOT** from a prepayment model. They are **prepayment forecast errors**,
decomposed each month by a cross-coupon regression.

**Step A — cross-coupon regression, run separately each month, on FORECAST prepayments:**
```
ppmt_forecast_t[i] = x_forecast_t + y_forecast_t · max(0, m^i − m_PMMS)
```
- `m^i` = WAC (weighted average coupon = borrower loan rate) of MBS coupon i
- `m_PMMS` = Freddie Mac Primary Mortgage Market Survey rate (monthly avg of weekly)
- intercept `x̂_forecast` = forecast level; slope `ŷ_forecast` = forecast rate-sensitivity

**Step B — same regression on REALIZED prepayments:**
```
ppmt_realized_t[i] = x_realized_t + y_realized_t · max(0, m^i − m_PMMS)
```

**Step C — the factor shocks are forecast errors:**
```
x_t = x̂_realized_t − x̂_forecast_t      (Eq 17)   level shock
y_t = ŷ_realized_t − ŷ_forecast_t      (Eq 18)   rate-sensitivity shock
```
Properties in their data: corr(x,y) ≈ 0.13; autocorr x ≈ 0.78, y ≈ 0.66.

---

## 3. Estimating loadings and prices of risk

**First stage (Eq. 19) — time-series regression, per coupon:**
```
R_e^i_t = a^i + β_x^i · x_t + β_y^i · y_t + ε^i_t
```
Regress each coupon's hedged excess return on the two factor shocks → β_x^i, β_y^i.

**Second stage (Eq. 20) — Fama–MacBeth cross-section, split by market type:**
```
R_e^i_t = a_t + λx · β̂_x^i + λy · β̂_y^i + ε^i_t
```
- **Discount market (DM):** ≥50% of remaining principal balance (RPB) trades below par
- **Premium market (PM):** otherwise
- Prediction (Hypothesis 1): PM → λx<0, λy<0;  DM → λx>0, λy<0.
- Multicollinearity fix: drop months where cross-section corr of loadings > 0.90.

**Robustness alternatives:**
- Single-characteristic version (Eq. 22): use relative moneyness `(c^i − r)` directly as the factor.
- Pooled panel with interactions (Eq. 24): interact β with `(%RPB_disc − 50%)`.

---

## 4. Their data sources (for reference)

| Series | Source | Frequency |
|---|---|---|
| Hedged excess returns by coupon | Bloomberg Barclays Hedged MBS Return indices (Fannie 30yr) | monthly, since 1994 |
| Forecast prepayments by coupon | Bloomberg median **dealer survey** (base rate scenario) | monthly |
| Realized prepayments by coupon | eMBS | monthly |
| WAC by coupon | (collected per coupon) | monthly |
| Current mortgage rate | Freddie Mac PMMS | weekly → monthly avg |
| Market composition (%RPB discount) | from coupon prices vs par | monthly |

---

## 5. WHERE OUR PROJECT MAPS — and where today's Stage 2 diverges

**What we built today (Stage 2):** model-implied `∂CPR/∂refi` per coupon from the hazard model.
**This is NOT the paper's β.** The paper's β comes from regressing realized hedged *returns*
on prepayment forecast-error *shocks*. Different objects:

| Component | Paper | Our Stage 2 today |
|---|---|---|
| Factors x, y | realized − forecast prepayment (cross-coupon regression) | not built (no forecast series) |
| β_x, β_y | time-series regression of returns on x,y shocks | `∂CPR/∂refi` from hazard model |
| Inputs needed | TBA returns + dealer forecasts + realized CPR | hazard model only |

**Three data series by coupon we still need for the paper's pipeline:**
1. Hedged TBA excess returns (Stage 1 — Bloomberg/WRDS).
2. Forecast prepayments by coupon (Bloomberg dealer survey).
3. Realized prepayments (CPR) by coupon by month — **buildable from our Fannie Mae data**.

---

## 6. THE CONTRIBUTION ANGLE — hazard model replaces the dealer forecast

The paper uses the **Bloomberg median dealer survey** as the forecast leg of the factor
construction (Step A). Our trained hazard model produces **predicted CPR by coupon** — it can
**replace the dealer survey** as `ppmt_forecast`:

```
x_t, y_t  =  (realized CPR, cross-coupon regression)  −  (OUR MODEL CPR, cross-coupon regression)
```

This is:
- **Novel** — prior work uses dealer surveys or structural-model-implied forecasts; we substitute an ML forecast.
- **"AI-forward"** — exactly the kind of substitution Gupta favors.
- **Self-contained on the forecast side** — removes dependence on the proprietary Bloomberg survey; we only need realized CPR (we have it) + TBA returns (Stage 1).

---

## 7. Concrete next steps (Friday, when cluster + ideally WRDS are back)

1. **Build realized CPR by coupon by month** from Fannie Mae data (group loans by coupon /
   relative coupon, compute monthly SMM → CPR). This is the realized leg (Eq 16), and a clean
   aggregation independent of any market data.
2. **Define securities by relative coupon** `c^i − c_par` using WAC, not the assumed +0.75 spread.
3. **Generate model-forecast CPR by coupon** from the (calibrated) hazard model → forecast leg.
4. **Form factor shocks** x_t, y_t = realized − model-forecast (cross-coupon regression each month).
5. **Stage 1 dependency:** hedged TBA excess returns + %RPB-discount for market-type split —
   needs Bloomberg/WRDS TBA data. This is the gating external input.
6. Then run Eq. 19 (loadings) and Eq. 20 (prices of risk, DM vs PM split).

---

## 8. Important caveat carried over from Stage 2

Our 2020Q1–2023Q1 Fannie Mae panel is **one-sided** (rates only rose; book is deep-discount,
mean refi incentive −2.34%). The model never observed the fast-refi premium regime, so the
realized-CPR cross-section it can produce is compressed (Stage 2 CPR spread only 0.74–1.46%).
For a full DM-vs-PM test we need either (a) earlier vintages observed through the 2020–21 refi
boom, or (b) the market TBA data, which embeds the full cross-section directly. This is the
honest limitation to raise with Gupta.
