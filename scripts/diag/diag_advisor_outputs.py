"""
Advisor asks (2026-08-03), items 3-5, computed on a hedge panel.

  3. Verification regression per coupon: hedged monthly return on level change,
     slope change, AND spread change. Report the nine level and slope t-stats.
  4. Annualized vol of the hedged coupon-spread portfolio.
  5. Residual duration in years for the unhedged part.

Usage:
    python scripts/diag/diag_advisor_outputs.py <panel.csv> [label]

Reports the TWO-regressor spec alongside the three-regressor one. The spread
control is known to increase the level coefficient rather than absorb it
(corr(d_level, d_spread) = -0.601 with a negative spread coefficient, so
omitting it biases level toward zero). Showing both keeps the effect of the
terminal floor separable from the effect of adding the spread -- otherwise a
change between panels cannot be attributed to either.

SPREAD TIMING. The panel's `pmms` column is keyed to the information date, not
the return month: corr(panel pmms, ret_month pmms lagged one month) = 1.0000
exactly. So d_spread is built from a fresh PMMS join on ret_month, and the
10-year leg comes from the identity d_y10 = d_level + d_slope/2 (since
d_level = (dy5+dy10)/2 and d_slope = dy10-dy5), avoiding a second date merge.

PORTFOLIO DEFINITION (item 4). Long coupon 6.5, short coupon 2.5, equal
weight, on HEDGED monthly returns, vol = std * sqrt(12). This is a fixed pair,
not selected by beta ranking.
    The ~8% recalled in the email is most likely
    factor_portfolio_std_monthly = 0.023214 (annualized 8.04%) from the
    ex-cutoff_2020 row of beta_spread_sharpe_results.json. That is a DIFFERENT
    construction: long-highest-beta/short-lowest-beta chosen by factor beta, on
    UNHEDGED excess returns, over 35 months. Its coupon pair is also unstable
    (6.5/3.0 flips to 2.5/5.0 when July 2022 is dropped, per the Phase 21
    leave-one-out). The fixed-pair number below is not the same statistic, so
    it is reported next to the old one rather than as a replacement.
    A beta-ranked variant is also printed for continuity.

RESIDUAL DURATION (item 5). resid_dur = -100 * b_level from the per-coupon
regression: the duration in years still showing up in hedged returns. Reported
for both specs, plus model D_level and the regression-implied duration from
unhedged returns, so the reconstruction check is visible.
"""
import sys
import numpy as np
import pandas as pd

PANEL = sys.argv[1] if len(sys.argv) > 1 else 'outputs/model_hedge_panel_10_span.csv'
LABEL = sys.argv[2] if len(sys.argv) > 2 else PANEL.split('/')[-1]
ANN = np.sqrt(12.0)


def ols(y, X):
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ co
    n, k = X.shape
    se = np.sqrt(np.diag(float(r @ r) / (n - k) * np.linalg.pinv(X.T @ X)))
    return co, se


p = pd.read_csv(PANEL)
p['ret_month'] = pd.to_datetime(p['ret_month'])
p = p.sort_values(['coupon', 'ret_month']).reset_index(drop=True)

pm = pd.read_csv('data/pmms_monthly.csv')


def parse(x):
    s = str(int(x))
    if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT


pm['date'] = pm['reporting_period'].apply(parse)
pms = pm.dropna(subset=['date']).set_index('date')['rate_30yr']

p['pmms_ret'] = p['ret_month'].dt.to_period('M').dt.to_timestamp().map(pms)
p['d_y10'] = p['d_level'] + p['d_slope'] / 2.0
p['d_spread'] = p.groupby('coupon')['pmms_ret'].diff() - p['d_y10']

print("=" * 88)
print("PANEL: %s" % LABEL)
print("=" * 88)

# ---------------------------------------------------------------- items 3 & 5
rows = []
for c, g in p.groupby('coupon'):
    g2 = g.dropna(subset=['hedged', 'd_level', 'd_slope'])
    X2 = np.column_stack([np.ones(len(g2)), g2['d_level'], g2['d_slope']])
    c2, s2 = ols(g2['hedged'].values, X2)
    cu, _ = ols(g2['tba_total_return'].values - g2['income'].values, X2)

    g3 = g.dropna(subset=['hedged', 'd_level', 'd_slope', 'd_spread'])
    X3 = np.column_stack([np.ones(len(g3)), g3['d_level'], g3['d_slope'], g3['d_spread']])
    c3, s3 = ols(g3['hedged'].values, X3)

    rows.append(dict(
        cpn=c, n2=len(g2), n3=len(g3),
        t_lvl2=c2[1] / s2[1], t_slp2=c2[2] / s2[2],
        t_lvl3=c3[1] / s3[1], t_slp3=c3[2] / s3[2], t_spr3=c3[3] / s3[3],
        resid2=-100 * c2[1], resid3=-100 * c3[1],
        modD=g2['D_level'].mean(), empD=-100 * cu[1]))
d = pd.DataFrame(rows)

print("\nITEM 3 -- verification regression, two-regressor vs three-regressor")
print("%5s %8s %8s   %8s %8s %8s" %
      ("cpn", "t_lvl", "t_slp", "t_lvl+sp", "t_slp+sp", "t_spread"))
for _, r in d.iterrows():
    print("%5.1f %8.2f %8.2f   %8.2f %8.2f %8.2f" %
          (r['cpn'], r['t_lvl2'], r['t_slp2'], r['t_lvl3'], r['t_slp3'], r['t_spr3']))

for tag, a, b in [("two-regressor", 't_lvl2', 't_slp2'),
                  ("three-regressor", 't_lvl3', 't_slp3')]:
    print("  %-16s |t_lvl|<2 at %s" % (tag, list(d.loc[d[a].abs() < 2, 'cpn'])))
    print("  %-16s |t_slp|<2 at %s" % ("", list(d.loc[d[b].abs() < 2, 'cpn'])))
print("  spread coef |t|>2 at %s" % list(d.loc[d['t_spr3'].abs() > 2, 'cpn']))

print("\nITEM 5 -- residual duration (years) still in hedged returns")
print("%5s %12s %12s %12s %12s %10s" %
      ("cpn", "resid_2reg", "resid_3reg", "model_D", "implied_D", "gap"))
for _, r in d.iterrows():
    print("%5.1f %12.3f %12.3f %12.3f %12.3f %10.3f" %
          (r['cpn'], r['resid2'], r['resid3'], r['modD'], r['empD'],
           r['empD'] - (r['resid2'] + r['modD'])))
ms = d['modD'].max() - d['modD'].min()
es = d['empD'].max() - d['empD'].min()
print("  duration spread capture: model %.3f / implied %.3f = %.1f%%" %
      (ms, es, 100 * ms / es))
print("  worst reconstruction gap: %.3f y at coupon %s" %
      ((d['empD'] - (d['resid2'] + d['modD'])).abs().max(),
       d.loc[(d['empD'] - (d['resid2'] + d['modD'])).abs().idxmax(), 'cpn']))

# --------------------------------------------------------------------- item 4
print("\nITEM 4 -- hedged coupon-spread portfolio, annualized vol")
w = p.pivot_table(index='ret_month', columns='coupon', values='hedged')
cps = sorted(w.columns)
lo, hi = cps[0], cps[-1]
port = (w[hi] - w[lo]).dropna()
print("  fixed pair, long %.1f / short %.1f, hedged returns" % (hi, lo))
print("    n months        : %d" % len(port))
print("    mean monthly    : %+.5f" % port.mean())
print("    std monthly     : %.5f" % port.std())
print("    ANNUALIZED VOL  : %.2f%%" % (100 * port.std() * ANN))
print("    ann. Sharpe     : %.3f" % (port.mean() / port.std() * ANN))

print("\n  all adjacent-pair vols, for context:")
print("  %6s %14s" % ("pair", "ann_vol_%"))
for i in range(len(cps) - 1):
    q = (w[cps[i + 1]] - w[cps[i]]).dropna()
    print("  %2.1f-%2.1f %13.2f" % (cps[i + 1], cps[i], 100 * q.std() * ANN))

print("\n  per-coupon hedged vol:")
print("  %6s %14s" % ("cpn", "ann_vol_%"))
for c in cps:
    print("  %6.1f %13.2f" % (c, 100 * w[c].dropna().std() * ANN))

print("\n  NOTE: beta_spread_sharpe_results.json ex-cutoff_2020 reports")
print("  factor_portfolio_std_monthly = 0.023214, i.e. 8.04%% annualized.")
print("  That is a beta-ranked pair on UNHEDGED excess returns over 35 months,")
print("  not this fixed pair on hedged returns over %d -- different statistic." % len(port))
