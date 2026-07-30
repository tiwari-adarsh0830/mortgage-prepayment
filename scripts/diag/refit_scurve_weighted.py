import numpy as np, pandas as pd, json
from scipy.optimize import curve_fit
GFEE = 0.50

r = pd.read_csv('outputs/realized_cpr_by_coupon_v6_upb.csv')
r['date'] = pd.to_datetime(r['date'])
pm = pd.read_csv('data/pmms_monthly.csv')
def parse(p):
    s = str(int(p))
    if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm['date'] = pm['reporting_period'].apply(parse)
pms = pm.dropna(subset=['date']).set_index('date')['rate_30yr']
r['pmms'] = r['date'].map(pms)
r['inc'] = (r['implied_mbs_coupon'] + GFEE) - r['pmms']
r = r.dropna(subset=['inc','cpr_upb']).copy()

print("=== realized incentive coverage vs what the pricer needs ===")
p = pd.read_csv('outputs/model_hedge_panel_10_span.csv')
p['inc'] = (p['coupon'] + GFEE) - p['pmms']
print("pricer incentive range: %.2f to %.2f" % (p['inc'].min(), p['inc'].max()))
print("realized data range:    %.2f to %.2f" % (r['inc'].min(), r['inc'].max()))
for cap in [2.0, 2.5, 3.0]:
    n = int((p['inc'] > cap).sum())
    print("  pricer coupon-months above +%.1f: %d of %d (%.1f%%)" % (cap, n, len(p), 100*n/len(p)))

r['b'] = pd.cut(r['inc'], bins=np.arange(-4, 3.5, 0.5))
g = r.groupby('b', observed=True).agg(inc=('inc','mean'), cpr=('cpr_upb','mean'),
                                      n=('cpr_upb','size')).reset_index(drop=True)
g = g[g['n'] >= 20]
print("\n=== bucket means used for the weighted fit ===")
print(g.round(4).to_string(index=False))

def scurve(x, floor, sat, k, x0):
    return floor + (sat - floor)/(1.0 + np.exp(-k*(x - x0)))

popt, _ = curve_fit(scurve, g['inc'].values, g['cpr'].values,
                    p0=[0.04, 0.22, 3.0, 0.4],
                    bounds=([0.005, 0.05, 0.2, -2.0], [0.15, 0.60, 10.0, 3.0]),
                    maxfev=40000)
floor, sat, k, x0 = popt
pred = scurve(g['inc'].values, *popt)
ss = 1 - float(((g['cpr']-pred)**2).sum())/float(((g['cpr']-g['cpr'].mean())**2).sum())
print("\nweighted-on-bucket-means fit: floor=%.4f sat=%.4f k=%.3f x0=%.3f  R2(buckets)=%.4f"
      % (floor, sat, k, x0, ss))

old = json.load(open('config/terminal_scurve.json'))
print("\n%8s %10s %10s %10s" % ("inc","realized","old_fit","new_fit"))
for _, row in g.iterrows():
    o = scurve(row['inc'], old['floor'], old['sat'], old['k'], old['x0'])
    print("%8.2f %10.4f %10.4f %10.4f" % (row['inc'], row['cpr'], o, scurve(row['inc'], *popt)))

json.dump(dict(floor=float(floor), sat=float(sat), k=float(k), x0=float(x0),
               fit_on="bucket means (n>=20), 0.5pp incentive bins",
               r2_buckets=float(ss),
               source="realized_cpr_by_coupon_v6_upb.csv, all ages (no age column)",
               note="monotone; burnout past +1.5 not fitted; flat extrapolation above fit range"),
          open('config/terminal_scurve_weighted.json','w'), indent=2)
print("\nSaved: config/terminal_scurve_weighted.json (NOT yet wired in)")
