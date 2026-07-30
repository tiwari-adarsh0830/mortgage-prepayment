"""Fit terminal-incentive S-curve to REALIZED CPR.

Anchored to realized data, not the model's month-33 output: the model's terminal
sits ~4x above realized at deep discounts (0.140 vs 0.035) and its steepest
response is at incentive 0.00 vs ~+1.0 realized, so extracting the curve from
the model would carry both defects into months 34-360.

CPR(inc) = floor + (sat - floor) / (1 + exp(-k*(inc - x0)))
Shifts with the bump because incentive is the argument. Monotone: the observed
downturn past +1.5 (burnout) is NOT fitted -- thin buckets, and a monotone
terminal is the conservative choice. Flagged, not hidden.
"""
import numpy as np, pandas as pd, json
from scipy.optimize import curve_fit
GFEE = 0.50
CAP  = 2.0

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
r['inc']  = (r['implied_mbs_coupon'] + GFEE) - r['pmms']
r = r.dropna(subset=['inc','cpr_upb'])
d = r[(r['inc'] >= -4.0) & (r['inc'] <= CAP)]
print(f"fit sample: {len(d)} coupon-months, incentive {d['inc'].min():.2f} to {d['inc'].max():.2f}")

def scurve(x, floor, sat, k, x0):
    return floor + (sat - floor)/(1.0 + np.exp(-k*(x - x0)))

p0 = [0.04, 0.25, 2.0, 0.8]
popt, _ = curve_fit(scurve, d['inc'].values, d['cpr_upb'].values, p0=p0,
                    bounds=([0.005, 0.05, 0.2, -2.0], [0.15, 0.60, 10.0, 3.0]),
                    maxfev=20000)
floor, sat, k, x0 = popt
pred = scurve(d['inc'].values, *popt)
ss_res = float(((d['cpr_upb'].values - pred)**2).sum())
ss_tot = float(((d['cpr_upb'].values - d['cpr_upb'].mean())**2).sum())
print(f"\nfitted: floor={floor:.4f} sat={sat:.4f} k={k:.3f} x0={x0:.3f}  R2={1-ss_res/ss_tot:.4f}")

print(f"\n{'inc':>6} {'fitted':>9} {'realized':>10} {'model_m33':>10}")
mm = {-4.0:0.1401, -3.0:0.1440, -2.0:0.1480, -1.0:0.1520,
       0.0:0.2200, 1.0:0.2900, 2.0:0.3500}
d['b'] = pd.cut(d['inc'], bins=np.arange(-4, 2.5, 0.5))
emp = d.groupby('b', observed=True)['cpr_upb'].mean()
for x in np.arange(-4.0, 2.01, 0.5):
    lab = [i for i in emp.index if i.left <= x < i.right or (x == i.right)]
    e = emp[lab[0]] if lab else np.nan
    print(f"{x:>6.1f} {scurve(x,*popt):>9.4f} {e:>10.4f} {mm.get(x,np.nan):>10.4f}")

json.dump(dict(floor=float(floor), sat=float(sat), k=float(k), x0=float(x0),
               n=int(len(d)), r2=float(1-ss_res/ss_tot), cap=CAP,
               source="realized_cpr_by_coupon_v6_upb.csv, all ages (no age column available)",
               note="monotone; observed burnout past +1.5 not fitted"),
          open('config/terminal_scurve.json','w'), indent=2)
print("\nSaved: config/terminal_scurve.json")
