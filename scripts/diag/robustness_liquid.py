import pandas as pd, numpy as np
from scipy import stats

LIQ = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

betas = pd.read_csv('outputs/factor_shock_betas.csv')
betas = betas[betas['coupon'].isin(LIQ)][['coupon', 'b_x', 'b_y']]

ret = pd.read_csv('outputs/stage3_excess_returns.csv')
ret['Date'] = pd.to_datetime(ret['Date'])
ret = ret[ret['coupon'].isin(LIQ)]
ret = ret[ret['Date'] >= '2020-01-01']   # match factor-shock START_DATE

# month-by-month cross-sectional OLS: R^e = lx*b_x + ly*b_y (no intercept, per DER)
lam_x, lam_y = [], []
for d, g in ret.groupby('Date'):
    g = g.merge(betas, on='coupon')
    if len(g) < 4:
        continue
    X = g[['b_x', 'b_y']].values
    y = g['excess_return'].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    lam_x.append(coef[0]); lam_y.append(coef[1])

lam_x = np.array(lam_x); lam_y = np.array(lam_y)

def fm(l):
    m = l.mean()
    t = m / (l.std(ddof=1) / np.sqrt(len(l)))
    p = 2 * (1 - stats.t.cdf(abs(t), len(l) - 1))
    return m, t, p, len(l)

mx, tx, px, nx = fm(lam_x)
my, ty, py, ny = fm(lam_y)
print('LIQUID-ONLY (2.5-5.5), drop 6.0/6.5')
print(f'lambda_x: mean={mx:.5f}  t={tx:.3f}  p={px:.4f}  n={nx}')
print(f'lambda_y: mean={my:.5f}  t={ty:.3f}  p={py:.4f}  n={ny}')
print('(full-panel: lambda_x=0.0578 t=2.52 p=0.014 | lambda_y=0.1726 t=1.68 p=0.097)')
