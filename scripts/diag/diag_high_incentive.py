import numpy as np, pandas as pd
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

r['b'] = pd.cut(r['inc'], bins=np.arange(-4, 4.6, 0.5))
g = r.groupby('b', observed=True).agg(inc=('inc','mean'), cpr=('cpr_upb','mean'),
                                      sd=('cpr_upb','std'), n=('cpr_upb','size'))
print("=== FULL realized range, no cap ===")
print(g.round(4).to_string())

print("\n=== which calendar months populate incentive > +2.0? ===")
hi = r[r['inc'] > 2.0]
print("n=%d, dates %s to %s" % (len(hi), hi['date'].min().date(), hi['date'].max().date()))
print(hi.groupby(hi['date'].dt.year)['cpr_upb'].agg(['mean','size']).round(4).to_string())
print("\ncoupons appearing above +2.0:")
print(hi.groupby('implied_mbs_coupon')['cpr_upb'].agg(['mean','size']).round(4).to_string())
