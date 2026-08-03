"""
Advisor ask (2026-08-03), item 2:

  "Another thing that could be happening here is that the attention prepayment
   response is too flat to rate incentive. To check this, compare the model
   sensitivity of CPR to rate incentive (the 'S' curve) against realized data"

The claim is about SLOPE -- dCPR/d(incentive) -- not level. Two curves can sit
at different levels with identical slopes, or overlap in level while one is
much flatter. Those imply different fixes, so slope is the primary table here
and level is supporting.

WHAT IS COMPARED

  Model: scripts/model_hedge_krd.cpr_path(inc) is called directly, so this
  exercises the same code path the pricer uses -- same scaler, same DEAD_COLS
  zeroing, same Platt pair (cohort-CPR a=0.4559 b=-3.1376, NOT the OAS pair).
  cpr_path returns a 33-length path with age in feature index 5 running 1..33,
  so cpr_path(inc)[k-1] is model CPR at age k. Ages beyond 33 are produced by
  writing age directly into index 5, which is outside the training range and
  is labelled as extrapolation wherever it appears.

  Raw (pre-Platt) is shown alongside calibrated. Platt is monotonic so it
  cannot reverse the shape, but it compresses slope -- and if the flatness is
  Platt rather than the transformer, that is a different fix. Calibrated is
  primary since it is what the pricer consumes.

  Realized: both all-loan and seasoned (age > 60mo), from the age-keyed panel.
  Model output at ages 61/121 is compared against SEASONED realized, since
  comparing a seasoned-age model query against an all-loan average would mix
  populations. All-loan is also shown because it is what the terminal S-curve
  is currently fitted to.

  Ages 12, 33, 61, 121. A single-age sweep would answer half the question:
  the hedge queries the model within 33 months, but the terminal covers ages
  34-360, and diag_age_extrapolation.py already found the age response is
  wrong-signed at depth (CPR RISING with age at incentive -3.0).

HEADLINE. Peak slope over the steep region, model vs realized, plus the
incentive at which each peaks. If peak slopes are comparable but located at
different incentives, the problem is horizontal displacement, not flatness.
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'scripts')
import model_hedge_krd as M

GFEE = 0.50
INC = np.arange(-4.0, 2.01, 0.25)
AGES = [12, 33, 61, 121]
BYAGE = 'outputs/realized_cpr_by_coupon_v6_upb_byage.csv'


def model_cpr_at_age(inc, age, model, scaler, a, b, calibrated=True):
    """CPR at a given age for constant incentive. age<=33 reads the path
    position; age>33 writes age into feature index 5 (extrapolation)."""
    if age <= M.MAX_SEQ:
        path = M.cpr_path(inc, model, scaler, a, b)
        if calibrated:
            return float(path[age - 1])
        s = _raw_logit(inc, np.arange(1, M.MAX_SEQ + 1), model, scaler)
        smm = 1.0 / (1.0 + np.exp(-s))
        return float((1.0 - (1.0 - smm) ** 12)[age - 1])
    ages = np.full(M.MAX_SEQ, float(age))
    lg = _raw_logit(inc, ages, model, scaler)
    if calibrated:
        smm = 1.0 / (1.0 + np.exp(-(a * lg + b)))
    else:
        smm = 1.0 / (1.0 + np.exp(-lg))
    return float((1.0 - (1.0 - smm) ** 12)[-1])


def _raw_logit(inc, ages, model, scaler):
    s = np.zeros((1, M.MAX_SEQ, M.N_FEATURES), dtype=np.float32)
    s[:, :, 0] = inc
    s[:, :, 1] = M.REP["credit_score"]; s[:, :, 2] = M.REP["orig_ltv"]
    s[:, :, 3] = M.REP["current_ltv"];  s[:, :, 4] = M.REP["orig_upb"]
    s[:, :, 5] = np.asarray(ages, dtype=np.float32)[None, :]
    s[:, :, 6] = M.REP["dti"]; s[:, :, 7] = M.REP["loan_purpose_enc"]
    s[:, :, 8] = M.REP["property_type_enc"]
    flat = scaler.transform(s.reshape(-1, M.N_FEATURES)).reshape(1, M.MAX_SEQ, M.N_FEATURES)
    for c in M.DEAD_COLS:
        flat[:, :, c] = 0.0
    x = torch.tensor(flat, dtype=torch.float32)
    mask = torch.ones(1, M.MAX_SEQ, dtype=torch.bool)
    with torch.no_grad():
        return model(x, mask=mask, return_per_timestep=True).numpy()[0]


def realized_curve(seasoned):
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
    b = np.arange(-4.0, 2.26, 0.5)
    g['bk'] = pd.cut(g['inc'], bins=b)
    q = (g.groupby('bk', observed=True)
           .agg(inc=('inc', 'mean'), cpr=('cpr', 'mean'), n=('cpr', 'size'))
           .reset_index(drop=True))
    q = q[q['n'] >= 5].reset_index(drop=True)
    q['slope'] = np.gradient(q['cpr'].values, q['inc'].values)
    return q


print("Loading model...", flush=True)
model, scaler, a, b = M.load_hazard()
print("  Platt (cohort-CPR): a=%.4f b=%.4f" % (a, b))

rc_all = realized_curve(False)
rc_sea = realized_curve(True)

print("\n" + "=" * 92)
print("MODEL S-CURVE BY AGE (calibrated), and its slope")
print("=" * 92)
tab = {}
for age in AGES:
    cal = np.array([model_cpr_at_age(i, age, model, scaler, a, b, True) for i in INC])
    raw = np.array([model_cpr_at_age(i, age, model, scaler, a, b, False) for i in INC])
    tab[age] = dict(cal=cal, raw=raw,
                    scal=np.gradient(cal, INC), sraw=np.gradient(raw, INC))

hdr = "%7s" % "inc"
for age in AGES:
    hdr += " %9s" % ("age%d" % age)
hdr += "  |"
for age in AGES:
    hdr += " %9s" % ("slp%d" % age)
print(hdr)
for j, i in enumerate(INC):
    if abs((i * 4) % 2) > 1e-9:
        continue
    line = "%7.2f" % i
    for age in AGES:
        line += " %9.4f" % tab[age]['cal'][j]
    line += "  |"
    for age in AGES:
        line += " %9.4f" % tab[age]['scal'][j]
    print(line)
print("  ages > 33 are outside the training range (extrapolation)")

print("\n" + "=" * 92)
print("REALIZED, all-loan and seasoned (age > 60mo)")
print("=" * 92)
print("%8s %9s %9s %7s   %9s %9s %7s" %
      ("inc", "cpr_all", "slope", "n", "cpr_seas", "slope", "n"))
for k in range(max(len(rc_all), len(rc_sea))):
    ra = rc_all.iloc[k] if k < len(rc_all) else None
    rs = rc_sea.iloc[k] if k < len(rc_sea) else None
    print("%8.2f %9.4f %9.4f %7d   %9.4f %9.4f %7d" % (
        ra['inc'] if ra is not None else np.nan,
        ra['cpr'] if ra is not None else np.nan,
        ra['slope'] if ra is not None else np.nan,
        ra['n'] if ra is not None else 0,
        rs['cpr'] if rs is not None else np.nan,
        rs['slope'] if rs is not None else np.nan,
        rs['n'] if rs is not None else 0))

print("\n" + "=" * 92)
print("HEADLINE: peak slope and where it occurs")
print("=" * 92)
print("%-26s %12s %14s %12s" % ("series", "peak_slope", "at_incentive", "cpr_there"))


def line(nm, x, y, s):
    k = int(np.argmax(s))
    print("%-26s %12.4f %14.2f %12.4f" % (nm, s[k], x[k], y[k]))
    return s[k], x[k]


ms = {}
for age in AGES:
    ms[age] = line("model age %d (cal)" % age, INC, tab[age]['cal'], tab[age]['scal'])
for age in AGES:
    line("model age %d (raw)" % age, INC, tab[age]['raw'], tab[age]['sraw'])
ra = line("realized all-loan", rc_all['inc'].values, rc_all['cpr'].values,
          rc_all['slope'].values)
rs = line("realized seasoned", rc_sea['inc'].values, rc_sea['cpr'].values,
          rc_sea['slope'].values)

print("\nmodel-to-realized peak slope ratio (calibrated):")
for age in AGES:
    ref = rs if age > 33 else ra
    nm = "seasoned" if age > 33 else "all-loan"
    print("  age %-4d : %.3f  (vs %s; model peaks at %+.2f, realized at %+.2f)" %
          (age, ms[age][0] / ref[0], nm, ms[age][1], ref[1]))
print("\n  ratio < 1 means the model response is flatter than realized.")
print("  A ratio near 1 with different peak locations is horizontal")
print("  displacement rather than flatness -- a different problem.")
