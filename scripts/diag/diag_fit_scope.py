import numpy as np, pandas as pd
from scipy.optimize import curve_fit
GFEE=0.50
r=pd.read_csv('outputs/realized_cpr_by_coupon_v6_upb.csv'); r['date']=pd.to_datetime(r['date'])
pm=pd.read_csv('data/pmms_monthly.csv')
def parse(x):
    s=str(int(x))
    if len(s)==5: return pd.Timestamp(year=int(s[1:]),month=int(s[0]),day=1)
    if len(s)==6: return pd.Timestamp(year=int(s[2:]),month=int(s[:2]),day=1)
    return pd.NaT
pm['date']=pm['reporting_period'].apply(parse)
pms=pm.dropna(subset=['date']).set_index('date')['rate_30yr']
r['pmms']=r['date'].map(pms); r['inc']=(r['implied_mbs_coupon']+GFEE)-r['pmms']
r=r.dropna(subset=['inc','cpr_upb'])

def sc(x,f,s,k,x0): return f+(s-f)/(1+np.exp(-k*(x-x0)))
def fit(d,lab):
    d=d[(d['inc']>=-4)&(d['inc']<=2)]
    if len(d)<40: print("%-34s n=%-5d INSUFFICIENT"%(lab,len(d))); return
    po,_=curve_fit(sc,d['inc'],d['cpr_upb'],p0=[0.04,0.22,3.0,0.4],
                   bounds=([0.005,0.05,0.2,-2.0],[0.15,0.60,10.0,3.0]),maxfev=40000)
    pr=sc(d['inc'].values,*po)
    r2=1-float(((d['cpr_upb']-pr)**2).sum())/float(((d['cpr_upb']-d['cpr_upb'].mean())**2).sum())
    print("%-34s n=%-5d floor=%.4f sat=%.4f k=%.2f x0=%.3f R2=%.3f"%(lab,len(d),*po,r2))

print("=== fit scope sensitivity (all anchored to realized CPR) ===")
fit(r, "ALL (current fit)")
fit(r[r['implied_mbs_coupon'].between(2.5,6.5)], "coupons 2.5-6.5 only")
fit(r[r['date']>='2018-01-01'], "2018+ only")
fit(r[(r['implied_mbs_coupon'].between(2.5,6.5))&(r['date']>='2018-01-01')], "2.5-6.5 AND 2018+")

print("\n=== how much of the current fit sample is outside the hedge coupon range? ===")
d=r[(r['inc']>=-4)&(r['inc']<=2)]
out=d[~d['implied_mbs_coupon'].between(2.5,6.5)]
print("outside 2.5-6.5: %d of %d (%.1f%%)"%(len(out),len(d),100*len(out)/len(d)))
print(out.groupby('implied_mbs_coupon').size().to_string())
print("\npre-2018: %d of %d (%.1f%%)"%(int((d['date']<'2018-01-01').sum()),len(d),
                                       100*(d['date']<'2018-01-01').mean()))
