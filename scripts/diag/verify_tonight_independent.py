"""Re-derives tonight's claims WITHOUT importing model_hedge_krd.

Every check tonight imported that module, so a defect inside it would be
inherited by all of them and still report clean. Weights, cashflow recursion and
discounting are all rewritten from the Fannie/pass-through definitions here.
Only bootstrap_v3 is imported, and it is the thing under test.

C1  tent partition, from the piecewise definitions, not key_rate_weights3
C2  tent-KRD sum vs parallel bump, own pricer
C3  duration vs closed form on a parallel ZERO bump, own pricer
C4  v3 vs original duration ratio, own pricer
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from bootstrap_v3 import bootstrap_zeros_v3

LAB = ['1mo','3mo','6mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
YRS = np.array([1/12,3/12,6/12,1,2,3,5,7,10,20,30])
N, GFEE, H = 360, 0.50, 0.25

def tents(T):
    T = np.asarray(T, float)
    w2 = np.where(T<=2, 1.0, np.where(T<=5, (5-T)/3, 0.0))
    w5 = np.where(T<=2, 0.0, np.where(T<=5, (T-2)/3, np.where(T<=10, (10-T)/5, 0.0)))
    w10 = np.where(T<=5, 0.0, np.where(T<=10, (T-5)/5, 1.0))
    return w2, w5, w10

def price(coupon, cpr, zeros):
    """Independent 30y pass-through pricer, written from the definition."""
    nm, im = (coupon+GFEE)/1200.0, coupon/1200.0
    smm = 1.0-(1.0-np.clip(cpr,0,0.99))**(1/12)
    disc = np.exp(-zeros/100.0*(np.arange(1,N+1)/12.0))
    bal, pv = 100.0, 0.0
    pmt = bal*nm/(1.0-(1.0+nm)**(-N))
    for t in range(N):
        if bal <= 1e-12: break
        sp = max(min(pmt-bal*nm, bal), 0.0)
        pp = (bal-sp)*smm[t]
        pv += (bal*im+sp+pp)*disc[t]
        bal -= sp+pp
    return pv

def price_mac(coupon, cpr, zeros):
    nm, im = (coupon+GFEE)/1200.0, coupon/1200.0
    smm = 1.0-(1.0-np.clip(cpr,0,0.99))**(1/12)
    ty = np.arange(1,N+1)/12.0
    disc = np.exp(-zeros/100.0*ty)
    bal, pv, wt = 100.0, 0.0, 0.0
    pmt = bal*nm/(1.0-(1.0+nm)**(-N))
    for t in range(N):
        if bal <= 1e-12: break
        sp = max(min(pmt-bal*nm, bal), 0.0)
        pp = (bal-sp)*smm[t]
        cf = bal*im+sp+pp
        pv += cf*disc[t]; wt += ty[t]*cf*disc[t]
        bal -= sp+pp
    return pv, wt/pv

def orig_boot(par):
    """Original algorithm, transcribed independently from its definition."""
    from scipy.interpolate import interp1d
    z = {T: float(par[l]) for T, l in zip(YRS, LAB) if T <= 1.0}
    def gz(T):
        k = sorted(z); v = [z[t] for t in k]
        if T<=k[0]: return v[0]
        if T>=k[-1]: return v[-1]
        return float(interp1d(k, v)(T))
    for T, l in zip(YRS, LAB):
        if T<=1.0: continue
        c = float(par[l])/100.0
        pv = sum((c/2)*100*np.exp(-gz(t)/100.0*t) for t in np.arange(0.5,T,0.5))
        z[T] = -np.log((100-pv)/(c/2*100+100))/T*100
    k = sorted(z); v = [z[t] for t in k]
    f = interp1d(k, v, fill_value='extrapolate')
    return np.array([float(f(m/12.0)) for m in range(1,N+1)])

def dur(coupon, par, cpr, w, bs):
    p0 = price(coupon, cpr, bs(par))
    px = {}
    for s in (+1,-1):
        px[s] = price(coupon, cpr, bs({l: par[l]+s*H*wi for l, wi in zip(LAB, w)}))
    return (px[-1]-px[+1])/(2.0*p0*(H/100.0))

d = pd.read_csv("data/treasury_yields.csv").dropna(subset=LAB)
d['DATE'] = pd.to_datetime(d['DATE'])
me = d.groupby(d['DATE'].dt.to_period('M')).last()
me = me[me.index >= '2018-01']
curves = [{l: float(me.iloc[i][l]) for l in LAB}
          for i in (0, len(me)//2, len(me)-1)]

mg = np.arange(1,N+1)/12.0
w2,w5,w10 = tents(mg)
print("C1 tent partition: max|sum-1| = %.3e" % np.abs(w2+w5+w10-1).max())

print("\nC2 tent sum vs parallel (own pricer)")
wp = np.ones(len(LAB)); t2,t5,t10 = tents(YRS)
worst=0
for ci,par in enumerate(curves):
    for c in (2.5,4.0,6.5):
        z = np.full(N,0.12)
        s = sum(dur(c,par,z,w,bootstrap_zeros_v3) for w in (t2,t5,t10))
        p = dur(c,par,z,wp,bootstrap_zeros_v3)
        worst=max(worst,abs(s/p-1)); print("  curve%d cpn%.1f ratio %.6f"%(ci,c,s/p))
print("  max|ratio-1| = %.3e"%worst)

print("\nC3 zero-bump duration vs closed form (own pricer)")
worst=0
for zl in (3.0,5.0):
    for c in (2.5,6.5):
        z=np.full(N,zl); cp=np.full(N,0.12)
        p0=price(c,cp,z); dz=H/100.0
        db=(price(c,cp,z-H)-price(c,cp,z+H))/(2.0*p0*dz)
        _,dc=price_mac(c,cp,z)
        worst=max(worst,abs(db/dc-1)); print("  z%.1f cpn%.1f ratio %.6f"%(zl,c,db/dc))
print("  max|ratio-1| = %.3e"%worst)

print("\nC4 v3 vs original duration ratio (own pricer)")
for c in (2.5,4.0,6.5):
    rs=[]
    for par in curves:
        z=np.full(N,0.12)
        rs.append(dur(c,par,z,wp,bootstrap_zeros_v3)/dur(c,par,z,wp,orig_boot))
    print("  cpn%.1f  mean ratio %.5f  (range %.5f-%.5f)"%(c,np.mean(rs),min(rs),max(rs)))
print("\n~1.33 would mean bootstrap explains it; ~1.00 means it does not")
