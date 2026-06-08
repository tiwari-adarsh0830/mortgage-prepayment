"""
Stage 2 (calibrated): coupon-bucket CPR + rate sensitivity from the hazard model.
Applies Platt calibration (outputs/hazard_calibration.json) so CPR levels are physical.
"""
import numpy as np, torch, torch.nn as nn, pickle, os, json, csv

BASE="/scratch/at7095/mortgage_prepayment"
SEQ_DIR=os.path.join(BASE,"data/sequences"); OUT=os.path.join(BASE,"outputs")
CKPT=os.path.join(OUT,"hazard_best.pt"); SCALER=os.path.join(SEQ_DIR,"scaler.pkl")
PATHS=os.path.join(OUT,"ddpm_conditional_paths.npy")
CALIB=os.path.join(OUT,"hazard_calibration.json")

MAX_SEQ,N_FEATURES=33,9
GFEE_SERVICING=0.75
COUPONS=[2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5]
DELTA=0.25
DEAD_COLS=[7,8]                       # loan_purpose_enc, property_type_enc -> scaled 0
REP=dict(credit_score=740.0,orig_ltv=75.0,current_ltv=70.0,
         orig_upb=250000.0,dti=35.0,loan_purpose_enc=0.0,property_type_enc=0.0)

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

def load_model():
    ck=torch.load(CKPT,map_location="cpu"); cfg=ck.get("config",{})
    m=PrepaymentTransformer(input_dim=N_FEATURES,d_model=cfg.get("d_model",64),
        n_heads=cfg.get("n_heads",4),n_layers=cfg.get("n_layers",2),
        dropout=cfg.get("dropout",0.1))
    m.load_state_dict(ck["model_state"]); m.eval(); return m

def build_batch(refi):
    n=refi.shape[0]; s=np.zeros((n,MAX_SEQ,N_FEATURES),dtype=np.float32)
    s[:,:,0]=refi; s[:,:,1]=REP["credit_score"]; s[:,:,2]=REP["orig_ltv"]
    s[:,:,3]=REP["current_ltv"]; s[:,:,4]=REP["orig_upb"]
    s[:,:,5]=np.arange(1,MAX_SEQ+1)[None,:]
    s[:,:,6]=REP["dti"]; s[:,:,7]=REP["loan_purpose_enc"]; s[:,:,8]=REP["property_type_enc"]
    return s

def cpr_from_refi(refi,model,scaler,a,b):
    n=refi.shape[0]; seqs=build_batch(refi)
    flat=scaler.transform(seqs.reshape(-1,N_FEATURES)).reshape(n,MAX_SEQ,N_FEATURES)
    for c in DEAD_COLS: flat[:,:,c]=0.0          # match training (dead features)
    x=torch.tensor(flat,dtype=torch.float32); mask=torch.ones(n,MAX_SEQ,dtype=torch.bool)
    with torch.no_grad():
        logit=model(x,mask=mask,return_per_timestep=True).numpy()
    smm=1.0/(1.0+np.exp(-(a*logit+b)))            # CALIBRATED monthly hazard
    cpr=1.0-(1.0-smm)**12
    return cpr.mean(0), float(cpr.mean())

def main():
    model=load_model()
    scaler=pickle.load(open(SCALER,"rb"))
    cal=json.load(open(CALIB)); a,b=cal["a"],cal["b"]
    print(f"[calib] a={a:.4f} b={b:.4f}")
    paths=np.load(PATHS); pmms=paths[:,1:1+MAX_SEQ]
    print(f"[paths] forward t0={pmms[:,0].mean():.3f}% end={np.median(pmms[:,-1]):.3f}%")

    atm_curve,atm=cpr_from_refi(np.zeros((pmms.shape[0],MAX_SEQ),dtype=np.float32),
                                model,scaler,a,b)
    print(f"[turnover baseline] ATM CPR={atm:.4f}")
    rows,curves=[],{}
    for c in COUPONS:
        nr=c+GFEE_SERVICING
        base_curve,base=cpr_from_refi(nr-pmms,model,scaler,a,b)
        _,up=cpr_from_refi((nr+DELTA)-pmms,model,scaler,a,b)
        _,dn=cpr_from_refi((nr-DELTA)-pmms,model,scaler,a,b)
        d=(up-dn)/(2*DELTA)
        tail="(upper-tail)" if c>=6.0 else ""
        rows.append(dict(coupon=c,note_rate=round(nr,3),mean_cpr=round(base,4),
                         turnover_cpr_atm=round(atm,4),rate_sens_dcpr_drefi=round(d,4)))
        curves[str(c)]=base_curve.tolist()
        print(f"coupon {c:>4}: CPR={base:.4f}  dCPR/drefi={d:.4f}  {tail}")
    json.dump(rows,open(os.path.join(OUT,"stage2_coupon_cpr.json"),"w"),indent=2)
    np.save(os.path.join(OUT,"stage2_cpr_curves.npy"),curves)
    with open(os.path.join(OUT,"stage2_coupon_cpr.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("Saved stage2_coupon_cpr.{json,csv}, stage2_cpr_curves.npy")

if __name__=="__main__": main()
