import sys; sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
from model_hedge_krd import load_hazard, cpr_path, GFEE

model, scaler, a, b = load_hazard()

print("=== model month-33 CPR vs refi incentive (implied terminal S-curve) ===")
print("%10s %10s %10s %10s" % ("incentive","m33_CPR","m24_CPR","m12_CPR"))
incs = np.arange(-4.0, 3.01, 0.25)
m33 = []
for inc in incs:
    p = cpr_path(float(inc), model, scaler, a, b)
    m33.append(p[32])
    print("%10.2f %10.4f %10.4f %10.4f" % (inc, p[32], p[23], p[11]))
m33 = np.array(m33)
print("\nfloor (inc=-4.00): %.4f   saturation (inc=+3.00): %.4f   ratio: %.2fx"
      % (m33[0], m33[-1], m33[-1]/m33[0]))
d = np.gradient(m33, incs)
print("max slope %.4f per pp, at incentive %.2f" % (d.max(), incs[int(np.argmax(d))]))

print("\n=== realized CPR by incentive bucket (empirical reference) ===")
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
r = r.dropna(subset=['inc','cpr_upb'])
r['bucket'] = pd.cut(r['inc'], bins=np.arange(-4, 3.5, 0.5))
print(r.groupby('bucket', observed=True)['cpr_upb'].agg(['mean','count']).round(4).to_string())
