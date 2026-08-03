"""
Pre-email verification. Two things I have NOT checked and one of them is
load-bearing.

(A) BUCKET-WIDTH SENSITIVITY OF THE PEAK-SLOPE RATIO.
    diag_model_vs_realized_scurve.py swept the model on a 0.25 incentive grid
    but bucketed realized data at 0.5 and took np.gradient on each. Coarser
    bucketing smooths a peak, so realized peak slope may be biased LOW, which
    would bias the model-to-realized ratio HIGH. The headline claim -- "not
    flat, ratio 1.06 at age 61" -- turns entirely on that number, so it needs
    to be shown stable across grids before it goes in an email.

    Recomputes realized peak slope at bucket widths 0.25, 0.33, 0.5 and 1.0,
    and recomputes the model peak slope on matching grids, so like is compared
    with like. Also reports a bucket-free estimate: a local linear fit of cpr
    on inc over a sliding window, which does not depend on binning at all.

    Peak LOCATION and the deep-discount LEVEL gap do not involve gradients and
    should be insensitive to this; that is checked rather than assumed.

(B) INDEPENDENT RECOMPUTE OF THE PANEL NUMBERS.
    The t-stats, residual durations and capture for the pinned panel came from
    one script (diag_advisor_outputs.py). Recomputes them here with a different
    code path -- statsmodels-free explicit normal equations, and residual
    duration from the covariance formula rather than lstsq -- so two
    independent computations either agree or they do not.

Usage:
    python scripts/diag/verify_before_email.py
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'scripts')
import model_hedge_krd as M

GFEE = 0.50
BYAGE = 'outputs/realized_cpr_by_coupon_v6_upb_byage.csv'
PIN = 'outputs/model_hedge_panel_10_span_pinnedfixed.csv'
FIT = 'outputs/model_hedge_panel_10_span.csv'
AGES = [33, 61, 121]


# --------------------------------------------------------------------- shared
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
    return g[(g['implied_mbs_coupon'] >= 2.5) & (g['implied_mbs_coupon'] <= 6.5)]


def binned_peak(g, w, min_n=5):
    b = np.arange(-4.0, 2.0 + w, w)
    x = g.copy()
    x['bk'] = pd.cut(x['inc'], bins=b)
    q = (x.groupby('bk', observed=True)
           .agg(inc=('inc', 'mean'), cpr=('cpr', 'mean'), n=('cpr', 'size'))
           .reset_index(drop=True))
    q = q[q['n'] >= min_n].reset_index(drop=True)
    if len(q) < 3:
        return np.nan, np.nan, 0
    s = np.gradient(q['cpr'].values, q['inc'].values)
    k = int(np.argmax(s))
    return float(s[k]), float(q['inc'].values[k]), len(q)


def local_linear_peak(g, half=0.4, step=0.05, min_n=25):
    """Bucket-free: slope of a local OLS of cpr on inc inside a sliding window."""
    xs, ys = g['inc'].values, g['cpr'].values
    grid = np.arange(-4.0 + half, 2.0 - half, step)
    best = (-np.inf, np.nan, 0)
    for c in grid:
        m = (xs >= c - half) & (xs <= c + half)
        if m.sum() < min_n:
            continue
        X = np.column_stack([np.ones(m.sum()), xs[m]])
        co, *_ = np.linalg.lstsq(X, ys[m], rcond=None)
        if co[1] > best[0]:
            best = (float(co[1]), float(c), int(m.sum()))
    return best


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
            f = scaler.transform(s.reshape(-1, M.N_FEATURES)).reshape(1, M.MAX_SEQ, M.N_FEATURES)
            for c in M.DEAD_COLS:
                f[:, :, c] = 0.0
            with torch.no_grad():
                lg = model(torch.tensor(f, dtype=torch.float32),
                           mask=torch.ones(1, M.MAX_SEQ, dtype=torch.bool),
                           return_per_timestep=True).numpy()[0]
            smm = 1.0 / (1.0 + np.exp(-(a * lg + b)))
            out.append(float((1.0 - (1.0 - smm) ** 12)[-1]))
    return np.array(out)


print("Loading model...", flush=True)
model, scaler, a, b = M.load_hazard()
r_all, r_sea = load_realized(False), load_realized(True)

print("\n" + "=" * 90)
print("(A1) REALIZED PEAK SLOPE vs BUCKET WIDTH")
print("=" * 90)
print("%10s | %12s %12s %8s | %12s %12s %8s" %
      ("width", "all_peak", "at_inc", "nbins", "seas_peak", "at_inc", "nbins"))
for w in [0.25, 1.0 / 3, 0.5, 1.0]:
    pa, ia, na = binned_peak(r_all, w)
    ps, is_, ns = binned_peak(r_sea, w)
    print("%10.3f | %12.4f %12.2f %8d | %12.4f %12.2f %8d" %
          (w, pa, ia, na, ps, is_, ns))

la = local_linear_peak(r_all)
ls = local_linear_peak(r_sea)
print("\nbucket-free local linear (half-width 0.40):")
print("  all-loan : peak slope %.4f at inc %+.2f  (n in window %d)" % la)
print("  seasoned : peak slope %.4f at inc %+.2f  (n in window %d)" % ls)

print("\n" + "=" * 90)
print("(A2) MODEL PEAK SLOPE ON MATCHING GRIDS, and the ratio")
print("=" * 90)
print("%6s %8s | %12s %12s | %12s %10s" %
      ("age", "grid", "model_peak", "at_inc", "realized_pk", "RATIO"))
for age in AGES:
    ref_series, ref_name = (r_sea, "seasoned") if age > 33 else (r_all, "all-loan")
    for w in [0.25, 0.5, 1.0]:
        grid = np.arange(-4.0, 2.0 + w / 2, w)
        mc = model_curve(age, model, scaler, a, b, grid)
        ms = np.gradient(mc, grid)
        k = int(np.argmax(ms))
        rp, ri, _ = binned_peak(ref_series, w)
        ratio = ms[k] / rp if rp and np.isfinite(rp) else np.nan
        print("%6d %8.2f | %12.4f %12.2f | %12.4f %10.3f" %
              (age, w, ms[k], grid[k], rp, ratio))
    lp = ls if age > 33 else la
    grid = np.arange(-4.0, 2.01, 0.05)
    mc = model_curve(age, model, scaler, a, b, grid)
    ms = np.gradient(mc, grid)
    k = int(np.argmax(ms))
    print("%6d %8s | %12.4f %12.2f | %12.4f %10.3f   (vs %s, bucket-free)" %
          (age, "loclin", ms[k], grid[k], lp[0], ms[k] / lp[0], ref_name))

print("\n" + "=" * 90)
print("(A3) CLAIMS THAT DO NOT DEPEND ON GRADIENTS")
print("=" * 90)
for age in AGES:
    grid = np.arange(-4.0, 2.01, 0.25)
    mc = model_curve(age, model, scaler, a, b, grid)
    print("  model age %-4d CPR at inc -4.00 : %.4f" % (age, mc[0]))
deep_all = r_all[r_all['inc'] <= -2.5]['cpr'].mean()
deep_sea = r_sea[r_sea['inc'] <= -2.5]['cpr'].mean()
print("  realized all-loan mean, inc <= -2.5 : %.4f (n=%d)" %
      (deep_all, (r_all['inc'] <= -2.5).sum()))
print("  realized seasoned mean, inc <= -2.5 : %.4f (n=%d)" %
      (deep_sea, (r_sea['inc'] <= -2.5).sum()))
print("  -> these are means and levels, no gradient, so bucket width is irrelevant")

print("\n" + "=" * 90)
print("(B) INDEPENDENT RECOMPUTE OF PANEL NUMBERS (normal equations)")
print("=" * 90)


def recompute(path, label):
    p = pd.read_csv(path)
    p['ret_month'] = pd.to_datetime(p['ret_month'])
    p = p.sort_values(['coupon', 'ret_month'])
    print("\n%s" % label)
    print("%6s %10s %10s %12s %12s" % ("cpn", "t_lvl", "t_slp", "resid_dur", "model_D"))
    md, ed = [], []
    for c, g in p.groupby('coupon'):
        g = g.dropna(subset=['hedged', 'd_level', 'd_slope'])
        X = np.column_stack([np.ones(len(g)), g['d_level'].values, g['d_slope'].values])
        y = g['hedged'].values
        XtX = X.T @ X
        co = np.linalg.solve(XtX, X.T @ y)          # normal equations, not lstsq
        r = y - X @ co
        s2 = float(r @ r) / (len(y) - 3)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
        yu = g['tba_total_return'].values - g['income'].values
        cu = np.linalg.solve(XtX, X.T @ yu)
        print("%6.1f %10.2f %10.2f %12.3f %12.3f" %
              (c, co[1] / se[1], co[2] / se[2], -100 * co[1], g['D_level'].mean()))
        md.append(g['D_level'].mean()); ed.append(-100 * cu[1])
    md, ed = np.array(md), np.array(ed)
    print("  capture: %.3f / %.3f = %.1f%%" %
          (md.max() - md.min(), ed.max() - ed.min(),
           100 * (md.max() - md.min()) / (ed.max() - ed.min())))


recompute(FIT, "fitted floor 0.0546")
recompute(PIN, "pinned floor 0.0459")
print("\nThese must match diag_advisor_outputs.py. Any disagreement is a bug in one of them.")
