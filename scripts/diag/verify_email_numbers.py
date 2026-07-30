import numpy as np, pandas as pd, json
GFEE=0.50

def ols(y,X):
    co,*_=np.linalg.lstsq(X,y,rcond=None)
    r=y-X@co; n,k=X.shape
    se=np.sqrt(np.diag(float(r@r)/(n-k)*np.linalg.pinv(X.T@X)))
    return co,se

p=pd.read_csv('outputs/model_hedge_panel_10_span.csv')
print("=== reconstruction + capture, computed on the panel ===")
print("%4s %8s %8s %10s %9s %8s %7s" % ("cpn","t_lvl","t_slp","resid_dur","model_D","emp_D","gap"))
rows=[]
for c,g in p.groupby('coupon'):
    g=g.dropna(subset=['hedged','d_level','d_slope'])
    X=np.column_stack([np.ones(len(g)),g['d_level'],g['d_slope']])
    co,se=ols(g['hedged'].values,X)
    co2,_=ols(g['tba_total_return'].values-g['income'].values,X)
    emp=-100*co2[1]; mod=g['D_level'].mean(); res=-100*co[1]
    rows.append(dict(c=c,t=co[1]/se[1],ts=co[2]/se[2],mod=mod,emp=emp,gap=emp-(res+mod)))
    print("%4s %8.2f %8.2f %10.3f %9.3f %8.3f %7.3f" % (c,co[1]/se[1],co[2]/se[2],res,mod,emp,emp-(res+mod)))
d=pd.DataFrame(rows)
print("\nworst gap = %.3f y at coupon %s ; all short = %s" %
      (d['gap'].abs().max(), d.loc[d['gap'].abs().idxmax(),'c'], bool((d['gap']>0).all())))
ms=d['mod'].max()-d['mod'].min(); es=d['emp'].max()-d['emp'].min()
print("capture: model spread %.3f / emp spread %.3f = %.1f%%" % (ms,es,100*ms/es))
print("|t_lvl|<2 at coupons: %s" % list(d.loc[d['t'].abs()<2,'c']))
print("|t_slp|<2 at coupons: %s" % list(d.loc[d['ts'].abs()<2,'c']))

print("\n=== does the S-curve preserve the CPR/rate relationship? ===")
q=json.load(open('config/terminal_scurve.json'))
def sc(x): return q['floor']+(q['sat']-q['floor'])/(1+np.exp(-q['k']*(x-q['x0'])))
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
b=r[(r['inc']>=-4)&(r['inc']<=2)].copy()
b['bk']=pd.cut(b['inc'],bins=np.arange(-4,2.5,0.5))
g=b.groupby('bk',observed=True).agg(inc=('inc','mean'),cpr=('cpr_upb','mean')).reset_index(drop=True)
g['dr_emp']=np.gradient(g['cpr'],g['inc'])
g['fit']=sc(g['inc']); g['dr_fit']=np.gradient(g['fit'],g['inc'])
print("%8s %9s %9s %11s %11s" % ("inc","realized","fitted","dCPR/dinc_emp","dCPR/dinc_fit"))
for _,x in g.iterrows():
    print("%8.2f %9.4f %9.4f %11.4f %11.4f" % (x['inc'],x['cpr'],x['fit'],x['dr_emp'],x['dr_fit']))
print("steepest realized at inc=%.2f (slope %.4f) ; fitted x0=%.3f (slope %.4f)"
      % (g.loc[g['dr_emp'].idxmax(),'inc'],g['dr_emp'].max(),q['x0'],g['dr_fit'].max()))

print("\n=== non-monotonicity in realized (full range) ===")
f=r.copy(); f['bk']=pd.cut(f['inc'],bins=np.arange(-4,4.6,0.5))
gg=f.groupby('bk',observed=True).agg(inc=('inc','mean'),cpr=('cpr_upb','mean'),n=('cpr_upb','size'))
gg=gg[gg['n']>=10]
print(gg.round(4).to_string())
dec=[(a,bb) for a,bb in zip(gg['cpr'][:-1],gg['cpr'][1:]) if bb<a]
print("segments where CPR falls: %d" % len(dec))
