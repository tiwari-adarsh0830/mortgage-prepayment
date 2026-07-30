import numpy as np, pandas as pd
GFEE = 0.50

r = pd.read_csv('outputs/realized_cpr_by_coupon_v6_upb.csv')
r['date'] = pd.to_datetime(r['date'])
print("columns:", list(r.columns))

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

# is there an age/vintage column to isolate seasoned cohorts?
agecols = [c for c in r.columns if 'age' in c.lower() or 'vint' in c.lower() or 'wala' in c.lower()]
print("age/vintage columns found:", agecols)

r['bucket'] = pd.cut(r['inc'], bins=np.arange(-4, 3.5, 0.5))
print("\n=== all observations ===")
print(r.groupby('bucket', observed=True)['cpr_upb'].agg(['mean','std','count']).round(4).to_string())

# proxy for seasoning: later calendar dates = older pools given 2018-2023 vintages
for lo, lab in [('2022-01-01','2022+'), ('2023-01-01','2023+'), ('2024-01-01','2024+')]:
    s = r[r['date'] >= lo]
    print(f"\n=== {lab} only (n={len(s)}) ===")
    print(s.groupby('bucket', observed=True)['cpr_upb'].agg(['mean','count']).round(4).to_string())
