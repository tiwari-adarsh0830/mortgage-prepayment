"""
Realized CPR by note-rate (coupon) bucket by month, from raw Fannie Mae data.
Realized leg of DER Eq.16. Uses the +1 offset (matches prepare_sequences col_map).
"""
import pandas as pd, numpy as np, glob, os, sys
sys.path.insert(0, "/scratch/at7095/mortgage_prepayment")
from prepare_sequences import all_cols

BASE="/scratch/at7095/mortgage_prepayment"
RAW=os.path.join(BASE,"data/raw"); OUT=os.path.join(BASE,"outputs")

NEED=['original_interest_rate','monthly_reporting_period','extra_13']
POS={c: all_cols.index(c)+1 for c in NEED}          # +1 offset (verified)
positions=sorted(POS.values())
names_in_order=[c for c,_ in sorted(POS.items(), key=lambda kv: kv[1])]
print("[cols]", POS, "positions", positions, flush=True)

GFEE_SERVICING=0.50
CHUNK=3_000_000

def main():
    files=sorted(glob.glob(os.path.join(RAW,"*.csv")))
    print(f"[files] {len(files)} vintages", flush=True)
    parts=[]
    for f in files:
        frows=0; fpp=0
        for chunk in pd.read_csv(f, sep='|', header=None, usecols=positions,
                                 names=names_in_order, chunksize=CHUNK, engine='c'):
            note=pd.to_numeric(chunk['original_interest_rate'],errors='coerce')
            month=pd.to_numeric(chunk['monthly_reporting_period'],errors='coerce')
            zbc=pd.to_numeric(chunk['extra_13'],errors='coerce')
            m=note.notna() & month.notna()
            if not m.any(): continue
            cb=(np.round(note[m].values*2)/2.0).astype(np.float32)
            mo=month[m].values.astype(np.int64)
            pp=(zbc[m].values==1.0)
            d=pd.DataFrame({'cb':cb,'month':mo,'pp':pp})
            agg=d.groupby(['cb','month'],observed=True).agg(
                    n_atrisk=('pp','size'), n_prepay=('pp','sum')).reset_index()
            parts.append(agg); frows+=int(m.sum()); fpp+=int(pp.sum())
        print(f"  {os.path.basename(f):12s} rows={frows:>10d} prepays={fpp:>8d}", flush=True)

    allagg=pd.concat(parts,ignore_index=True)
    out=allagg.groupby(['cb','month'],observed=True).agg(
            n_atrisk=('n_atrisk','sum'), n_prepay=('n_prepay','sum')).reset_index()
    out['smm']=out['n_prepay']/out['n_atrisk']
    out['cpr']=1-(1-out['smm'])**12
    out['implied_mbs_coupon']=(out['cb']-GFEE_SERVICING).round(3)
    out=out.rename(columns={'cb':'coupon_bucket'})[
        ['coupon_bucket','implied_mbs_coupon','month','n_atrisk','n_prepay','smm','cpr']]
    out.to_csv(os.path.join(OUT,"realized_cpr_by_coupon.csv"),index=False)
    print(f"\n[saved] ({len(out)} cells)", flush=True)
    print("\n[summary] at-risk-weighted mean CPR by coupon bucket:", flush=True)
    for cb,grp in out.groupby('coupon_bucket'):
        w=grp['n_atrisk'].sum()
        wcpr=(grp['cpr']*grp['n_atrisk']).sum()/w if w>0 else np.nan
        print(f"  note {cb:>4.1f}% (MBS~{cb-GFEE_SERVICING:>4.2f}%): meanCPR={wcpr:.4f}  atrisk={int(w):>11d}", flush=True)

if __name__=="__main__": main()
