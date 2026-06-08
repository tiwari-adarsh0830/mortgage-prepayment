"""
Calibrate hazard model for the 50% prepaid oversampling.
Fits Platt scaling sigmoid(a*logit + b) mapping model logit -> true per-timestep
hazard, on a representative (un-rebalanced) sample of real test loans.
Saves outputs/hazard_calibration.json and prints a reliability check.
"""
import numpy as np, torch, torch.nn as nn, os, json
from sklearn.linear_model import LogisticRegression

BASE="/scratch/at7095/mortgage_prepayment"
SEQ=os.path.join(BASE,"data/sequences"); OUT=os.path.join(BASE,"outputs")
CKPT=os.path.join(OUT,"hazard_best.pt")
MAX_SEQ,N_FEATURES=33,9
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLE=150000            # uniform subsample of test loans (preserves true base rate)
BATCH=8192
SEED=0

class PrepaymentTransformer(nn.Module):
    def __init__(self,input_dim=9,d_model=64,n_heads=4,n_layers=2,
                 dim_ff=256,max_seq=33,dropout=0.1):
        super().__init__()
        self.input_proj=nn.Linear(input_dim,d_model)
        self.pos_embedding=nn.Embedding(max_seq,d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
            dim_feedforward=dim_ff,dropout=dropout,batch_first=True)
        self.transformer=nn.TransformerEncoder(enc,num_layers=n_layers)
        self.classifier=nn.Sequential(nn.Linear(d_model,32),nn.ReLU(),
            nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x,mask=None,return_per_timestep=False):
        B,T,_=x.shape
        pos=torch.arange(T,device=x.device).unsqueeze(0).expand(B,-1)
        out=self.input_proj(x)+self.pos_embedding(pos)
        pad=~mask if mask is not None else None
        out=self.transformer(out,src_key_padding_mask=pad)
        if return_per_timestep: return self.classifier(out).squeeze(-1)
        if mask is not None:
            real=mask.float().unsqueeze(-1); out=(out*real).sum(1)/real.sum(1).clamp(min=1)
        else: out=out.mean(1)
        return self.classifier(out).squeeze(-1)

def main():
    ck=torch.load(CKPT,map_location="cpu"); cfg=ck.get("config",{})
    m=PrepaymentTransformer(input_dim=N_FEATURES,d_model=cfg.get("d_model",64),
        n_heads=cfg.get("n_heads",4),n_layers=cfg.get("n_layers",2),
        dropout=cfg.get("dropout",0.1))
    m.load_state_dict(ck["model_state"]); m.eval().to(DEVICE)

    seq=np.load(os.path.join(SEQ,"test_seq.npy"))                 # scaled
    mask=np.load(os.path.join(SEQ,"test_mask.npy")).astype(bool)  # True=real
    ptime=np.load(os.path.join(SEQ,"test_prepay_timestep.npy"))
    labels=np.load(os.path.join(SEQ,"test_labels.npy"))
    N=seq.shape[0]
    rng=np.random.default_rng(SEED)
    idx=rng.choice(N, size=min(N_SAMPLE,N), replace=False)
    seq,mask,ptime,labels=seq[idx],mask[idx],ptime[idx],labels[idx]
    print(f"[sample] {len(idx)} loans, prepaid frac={labels.mean():.4f}")

    # per-timestep logits
    logits=np.zeros((len(idx),MAX_SEQ),dtype=np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BATCH):
            xb=torch.tensor(seq[s:s+BATCH],dtype=torch.float32,device=DEVICE)
            mb=torch.tensor(mask[s:s+BATCH],dtype=torch.bool,device=DEVICE)
            logits[s:s+BATCH]=m(xb,mask=mb,return_per_timestep=True).cpu().numpy()

    # build at-risk labels: y=1 at the prepay timestep, over mask=True positions
    tgrid=np.arange(MAX_SEQ)[None,:]
    y=(tgrid==ptime[:,None]).astype(np.int8)          # 1 only at prepay month
    valid=mask                                         # at-risk = observed
    # SANITY: #positives within valid must equal #prepaid loans
    pos_in_valid=int((y[valid]==1).sum()); n_prepaid=int(labels.sum())
    print(f"[sanity] positive (loan,t) in valid={pos_in_valid}  prepaid loans={n_prepaid}  "
          f"{'OK' if pos_in_valid==n_prepaid else 'MISMATCH!'}")

    L=logits[valid].reshape(-1,1); Y=y[valid].ravel()
    print(f"[calib set] pairs={len(Y)}  positives={Y.sum()}  rate={Y.mean():.5f}")

    clf=LogisticRegression(C=1e6,max_iter=1000)        # near-unregularized Platt
    clf.fit(L,Y)
    a=float(clf.coef_[0,0]); b=float(clf.intercept_[0])
    print(f"[platt] a={a:.4f}  b={b:.4f}  (a~1 => pure prior shift)")

    # reliability: bin by calibrated prob, compare mean pred vs actual
    p=1/(1+np.exp(-(a*L.ravel()+b)))
    print("[reliability]  bin  mean_pred  mean_actual  n")
    edges=np.quantile(p,np.linspace(0,1,11))
    for i in range(10):
        lo,hi=edges[i],edges[i+1]; sel=(p>=lo)&(p<=hi if i==9 else p<hi)
        if sel.sum()==0: continue
        print(f"            {i:>3}   {p[sel].mean():.5f}    {Y[sel].mean():.5f}   {sel.sum()}")
    print(f"[overall] mean_pred={p.mean():.5f}  mean_actual={Y.mean():.5f}")

    json.dump({"a":a,"b":b,"note":"calibrated_SMM=sigmoid(a*logit+b)",
               "sample":len(idx),"calib_rate":float(Y.mean())},
              open(os.path.join(OUT,"hazard_calibration.json"),"w"),indent=2)
    print("Saved outputs/hazard_calibration.json")

if __name__=="__main__": main()
