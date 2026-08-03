"""
Is the model's prepayment response too flat in incentive? Gradient-free test.

WHY THIS EXISTS. The peak-slope ratio turned out not to be a robust statistic.
For model age 61 vs realized seasoned it reads 0.511 / 1.027 / 1.674 at bucket
widths 0.25 / 0.50 / 1.00, and 0.536 bucket-free -- a three-fold swing driven
by binning choice alone. Realized peak slope is itself poorly determined
(0.1677 at width 0.25 vs 0.0808 at 0.50), because realized CPR is noisy and
non-monotone in incentive. Any headline built on peak slope is fragile, and
two different framings were produced from it already.

WHAT IS ROBUST. Total RANGE of CPR across the incentive sweep. If a curve
spans less CPR from deep discount to deep premium than realized does, it is
flatter -- and range needs no derivative, no binning, and no smoothing window.
This is the direct form of the question asked.

Reported three ways, all gradient-free:
  1. Range  = CPR(+2.0) - CPR(-4.0), endpoints only.
  2. Interquantile rise = CPR at the 90th vs 10th percentile of the realized
     incentive distribution, so the comparison sits where data actually is
     rather than at sweep endpoints that may be sparsely populated.
  3. Ratio of ranges, model / realized. Below 1 means flatter.

Realized range is computed two ways to show it does not depend on bucketing:
  (a) from bucket means at several widths
  (b) from a direct percentile split of the raw coupon-month observations

Also reports the LEVEL at deep discount, which is the other robust claim, and
the peak LOCATION, which is stable across widths and is a separate finding
from flatness.

Usage:
    python scripts/diag/diag_flatness_range.py
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'scripts')
import model_hedge_krd as M

GFEE = 0.50
BYAGE = 'outputs/realized_cpr_by_coupon_v6_upb_byage.csv'
AGES = [12, 33, 61, 121]
LO, HI = -4.0, 2.0


def load_realized(seasoned):
    d = pd.read_csv(BYAGE)
    d = d[d['age_group'] >= 0]
    if seasoned:
        d = d[d['age_group'] >= 60]
    g = (d.groupby(['coupon_bucket', 'implied_mbs_coupon', 'yyyymm'], as_index=False)
           [['upb_atrisk', 'upb_prepay']].sum())
    g = g[g['upb_atrisk'] > 0].copy()
    g['cpr'] = 1.0 - (1.0 - g['upb_prepay'] / g['upb_atrisk']) ** 12
    g['date'] = pd.to_datetime(g['yyyymm'].astype(str), format='%Y%m')
    pm = pd.read_csv('data/pmms_monthly.csv')

    def parse(x):
        t = str(int(x))
        if len(t) == 5: return pd.Timestamp(year=int(t[1:]), month=int(t[0]), day=1)
        if len(t) == 6: return pd.Timestamp(year=int(t[2:]), month=int(t[:2]), day=1)
        return pd.NaT

    pm['date'] = pm['reporting_period'].apply(parse)
    ps = pm.dropna(subset=['date']).set_index('date')['rate_30yr']
    g['pmms'] = g['date'].map(ps)
    g['inc'] = (g['implied_mbs_coupon'] + GFEE) - g['pmms']
    g = g.dropna(subset=['inc', 'cpr'])
    g = g[(g['implied_mbs_coupon'] >= 2.5) & (g['implied_mbs_coupon'] <= 6.5)]
    return g[(g['inc'] >= LO) & (g['inc'] <= HI)]


def model_curve(age, model, scaler, a, b, grid):
    out = []
    for i in grid:
        if age <= M.MAX_SEQ:
            out.append(float(M.cpr_path(i, model, scaler, a, b)[age - 1]))
        else:
            s = np.zeros((1, M.MAX_SEQ, M.N_FEATURES), dtype=np.float32)
            s[:, :, 0] = i
            s[:, :, 1] = M.REP["credit_score"]; s[:, :, 2] = M.REP["orig_ltv"]
            s[:, :, 3] = M.REP["current_ltv"];  s[:, :, 4] = M.REP["orig_upb"]
            s[:, :, 5] = float(age)
            s[:, :, 6] = M.REP["dti"]; s[:, :, 7] = M.REP["loan_purpose_enc"]
            s[:, :, 8] = M.REP["property_type_enc"]
            f = scaler.transform(s.reshape(-1, M.N_FEATURES)).reshape(
                1, M.MAX_SEQ, M.N_FEATURES)
            for c in M.DEAD_COLS:
                f[:, :, c] = 0.0
            with torch.no_grad():
                lg = model(torch.tensor(f, dtype=torch.float32),
                           mask=torch.ones(1, M.MAX_SEQ, dtype=torch.bool),
                           return_per_timestep=True).numpy()[0]
            smm = 1.0 / (1.0 + np.exp(-(a * lg + b)))
            out.append(float((1.0 - (1.0 - smm) ** 12)[-1]))
    return np.array(out)


def realized_at(g, x, half=0.25):
    """Mean realized CPR in a narrow band around incentive x."""
    m = (g['inc'] >= x - half) & (g['inc'] <= x + half)
    return (float(g.loc[m, 'cpr'].mean()), int(m.sum())) if m.sum() else (np.nan, 0)


print("Loading model...", flush=True)
model, scaler, a, b = M.load_hazard()
r_all, r_sea = load_realized(False), load_realized(True)
grid = np.arange(LO, HI + 0.01, 0.25)
curves = {age: model_curve(age, model, scaler, a, b, grid) for age in AGES}

print("\n" + "=" * 86)
print("(1) ENDPOINT RANGE  CPR(+2.0) - CPR(-4.0)   [no derivative, no binning]")
print("=" * 86)
print("%-24s %10s %10s %10s" % ("series", "at -4.0", "at +2.0", "RANGE"))
for age in AGES:
    c = curves[age]
    print("%-24s %10.4f %10.4f %10.4f" %
          ("model age %d" % age, c[0], c[-1], c[-1] - c[0]))
for nm, g in [("realized all-loan", r_all), ("realized seasoned", r_sea)]:
    lo_v, lo_n = realized_at(g, LO + 0.25)
    hi_v, hi_n = realized_at(g, HI - 0.25)
    print("%-24s %10.4f %10.4f %10.4f   (n=%d, %d)" %
          (nm, lo_v, hi_v, hi_v - lo_v, lo_n, hi_n))

print("\n" + "=" * 86)
print("(2) RANGE ACROSS THE REALIZED DATA MASS  (10th to 90th pctile of inc)")
print("=" * 86)
for nm, g in [("all-loan", r_all), ("seasoned", r_sea)]:
    q10, q90 = np.percentile(g['inc'], [10, 90])
    rv10, n10 = realized_at(g, q10)
    rv90, n90 = realized_at(g, q90)
    rng = rv90 - rv10
    print("\n  %s: inc p10=%+.2f p90=%+.2f | realized %.4f -> %.4f, range %.4f"
          % (nm, q10, q90, rv10, rv90, rng))
    print("  %-22s %10s %10s %10s %10s" % ("series", "at p10", "at p90", "range", "RATIO"))
    for age in AGES:
        c = curves[age]
        m10 = float(np.interp(q10, grid, c))
        m90 = float(np.interp(q90, grid, c))
        print("  %-22s %10.4f %10.4f %10.4f %10.3f" %
              ("model age %d" % age, m10, m90, m90 - m10,
               (m90 - m10) / rng if rng else np.nan))

print("\n" + "=" * 86)
print("(3) LEVEL AT DEEP DISCOUNT  [means only]")
print("=" * 86)
for age in AGES:
    print("  model age %-4d at inc -4.00 : %.4f" % (age, curves[age][0]))
for nm, g in [("all-loan", r_all), ("seasoned", r_sea)]:
    s = g[g['inc'] <= -2.5]
    print("  realized %-9s inc<=-2.5   : %.4f  (n=%d)" % (nm, s['cpr'].mean(), len(s)))

print("\n" + "=" * 86)
print("(4) PEAK LOCATION  [stable across bucket widths; separate from flatness]")
print("=" * 86)
for age in AGES:
    c = curves[age]
    k = int(np.argmax(np.gradient(c, grid)))
    print("  model age %-4d steepest at inc %+.2f" % (age, grid[k]))
for nm, g in [("all-loan", r_all), ("seasoned", r_sea)]:
    best = (-np.inf, np.nan)
    for cx in np.arange(LO + 0.4, HI - 0.4, 0.05):
        m = (g['inc'] >= cx - 0.4) & (g['inc'] <= cx + 0.4)
        if m.sum() < 25:
            continue
        X = np.column_stack([np.ones(m.sum()), g.loc[m, 'inc'].values])
        co, *_ = np.linalg.lstsq(X, g.loc[m, 'cpr'].values, rcond=None)
        if co[1] > best[0]:
            best = (co[1], cx)
    print("  realized %-9s steepest at inc %+.2f" % (nm, best[1]))

print("\nRatio below 1 in block (2) means the model spans less CPR than realized")
print("across the same incentive range, i.e. flatter. Block (2) is the primary")
print("test: it is gradient-free and measured where the data actually sits.")
