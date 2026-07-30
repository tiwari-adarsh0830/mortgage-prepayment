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
print("realized CPR file: %s to %s, n=%d" % (r['date'].min().date(), r['date'].max().date(), len(r)))

def sc(x,f,s,k,x0): return f+(s-f)/(1+np.exp(-k*(x-x0)))

p=pd.read_csv('outputs/model_hedge_panel_10_span.csv')
months=sorted(p['ret_month'].unique())
print("\n%10s %7s %8s %8s %8s %8s %8s" % ("ret_month","n_hist","inc_min","inc_max","floor","sat","x0"))
for m in months[:6]+months[len(months)//2:len(months)//2+3]+months[-3:]:
    cutoff=pd.Timestamp(m+"-01")
    h=r[(r['date']<cutoff)&(r['inc']>=-4)&(r['inc']<=2)]
    if len(h)<40:
        print("%10s %7d   INSUFFICIENT" % (m,len(h))); continue
    try:
        po,_=curve_fit(sc,h['inc'].values,h['cpr_upb'].values,p0=[0.04,0.22,3.0,0.4],
                       bounds=([0.005,0.05,0.2,-2.0],[0.15,0.60,10.0,3.0]),maxfev=40000)
        print("%10s %7d %8.2f %8.2f %8.4f %8.4f %8.3f" %
              (m,len(h),h['inc'].min(),h['inc'].max(),po[0],po[1],po[3]))
    except Exception as e:
        print("%10s %7d   FAILED %s" % (m,len(h),e))
