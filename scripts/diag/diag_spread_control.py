"""
Verification regression with a mortgage-spread control.

Advisor (2026-07-30): "If you add a mortgage spread control (ie the PMMS -
10 year spread change) this should address the remaining part."

This changes the TEST, not the hedged returns. p['hedged'] is untouched; the
series feeding load_excess_returns() is identical with and without this. The
claim being tested is that the residual level exposure is mortgage-spread
contamination rather than unhedged Treasury exposure.

Reports both specifications side by side, because the level coefficient means
different things in each: unconditional exposure vs exposure holding mortgage
spread fixed.

Two guards before trusting the result:
  (a) TIMING. The panel's `pmms` is the as-of value used for the bump and may
      be keyed to info_date, while d_level/d_slope are keyed to the return
      month. If those are offset by a month, pmms.diff() is misaligned. So
      d_spread is built two ways -- from the panel column, and from a fresh
      PMMS join on ret_month -- and the two are compared.
  (b) COLLINEARITY. If corr(d_level, d_spread) is high the two regressors are
      fighting each other and a collapsing level t means little on its own.
      VIF is reported alongside.

Identity used: d_y10 = d_level + d_slope/2, since d_level = (dy5+dy10)/2 and
d_slope = dy10 - dy5. Avoids a second date-keyed merge for the Treasury leg.
"""
import numpy as np, pandas as pd

PANEL = 'outputs/model_hedge_panel_10_span.csv'


def ols(y, X):
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ co
    n, k = X.shape
    se = np.sqrt(np.diag(float(r @ r) / (n - k) * np.linalg.pinv(X.T @ X)))
    return co, se


p = pd.read_csv(PANEL)
p['ret_month'] = pd.to_datetime(p['ret_month'])
p = p.sort_values(['coupon', 'ret_month']).reset_index(drop=True)

# --- d_spread, version A: from the panel's own pmms column -------------------
p['d_pmms_panel'] = p.groupby('coupon')['pmms'].diff()
p['d_y10'] = p['d_level'] + p['d_slope'] / 2.0
p['d_spread_A'] = p['d_pmms_panel'] - p['d_y10']

# --- d_spread, version B: fresh PMMS join keyed to ret_month -----------------
pm = pd.read_csv('data/pmms_monthly.csv')


def parse(x):
    s = str(int(x))
    if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT


pm['date'] = pm['reporting_period'].apply(parse)
pms = pm.dropna(subset=['date']).set_index('date')['rate_30yr']
p['pmms_ret'] = p['ret_month'].dt.to_period('M').dt.to_timestamp().map(pms)
p['d_pmms_ret'] = p.groupby('coupon')['pmms_ret'].diff()
p['d_spread_B'] = p['d_pmms_ret'] - p['d_y10']

print("=== TIMING GUARD: panel pmms vs ret_month-keyed pmms ===")
chk = p.dropna(subset=['d_spread_A', 'd_spread_B'])
print("  n compared           : %d" % len(chk))
print("  corr(A, B)           : %.4f" % chk['d_spread_A'].corr(chk['d_spread_B']))
print("  mean abs diff (bp)   : %.2f" % (100 * (chk['d_spread_A'] - chk['d_spread_B']).abs().mean()))
lag = p.groupby('coupon').apply(
    lambda g: g['pmms'].corr(g['pmms_ret'].shift(1)), include_groups=False).mean()
same = p.groupby('coupon').apply(
    lambda g: g['pmms'].corr(g['pmms_ret']), include_groups=False).mean()
print("  corr(panel pmms, ret pmms)        : %.4f" % same)
print("  corr(panel pmms, ret pmms lag 1)  : %.4f" % lag)
print("  -> if the lagged corr is the higher one, the panel pmms is info_date-keyed")
print("     and version B is the one to use.\n")

for tag, col in [('A: panel pmms', 'd_spread_A'), ('B: ret_month pmms', 'd_spread_B')]:
    print("=" * 74)
    print("SPEC %s" % tag)
    print("=" * 74)
    print("%4s %9s %9s %9s %9s %9s %8s" %
          ("cpn", "t_lvl_no", "t_lvl_yes", "t_slp_yes", "t_spr", "corr_ls", "vif_lvl"))
    keep = []
    for c, g in p.groupby('coupon'):
        g = g.dropna(subset=['hedged', 'd_level', 'd_slope', col])
        if len(g) < 10:
            continue
        X0 = np.column_stack([np.ones(len(g)), g['d_level'], g['d_slope']])
        co0, se0 = ols(g['hedged'].values, X0)
        X1 = np.column_stack([np.ones(len(g)), g['d_level'], g['d_slope'], g[col]])
        co1, se1 = ols(g['hedged'].values, X1)

        r = np.corrcoef(g['d_level'], g[col])[0, 1]
        Xv = np.column_stack([np.ones(len(g)), g['d_slope'], g[col]])
        cv, _ = ols(g['d_level'].values, Xv)
        resid = g['d_level'].values - Xv @ cv
        r2 = 1 - float(resid @ resid) / float(
            ((g['d_level'] - g['d_level'].mean()) ** 2).sum())
        vif = 1.0 / max(1 - r2, 1e-12)

        print("%4s %9.2f %9.2f %9.2f %9.2f %9.3f %8.1f" %
              (c, co0[1] / se0[1], co1[1] / se1[1], co1[2] / se1[2],
               co1[3] / se1[3], r, vif))
        keep.append(dict(c=c, t_no=co0[1] / se0[1], t_yes=co1[1] / se1[1],
                         t_slp=co1[2] / se1[2], t_spr=co1[3] / se1[3]))

    d = pd.DataFrame(keep)
    print("\n  |t_lvl|<2  without control : %s" % list(d.loc[d['t_no'].abs() < 2, 'c']))
    print("  |t_lvl|<2  with control    : %s" % list(d.loc[d['t_yes'].abs() < 2, 'c']))
    print("  |t_slp|<2  with control    : %s" % list(d.loc[d['t_slp'].abs() < 2, 'c']))
    print("  spread coef significant at : %s" % list(d.loc[d['t_spr'].abs() > 2, 'c']))
    print("  max |t_lvl| with control   : %.2f at coupon %s\n" %
          (d['t_yes'].abs().max(), d.loc[d['t_yes'].abs().idxmax(), 'c']))
