"""Can the hazard model be queried at seasoned ages?

MAX_SEQ=33 caps sequence POSITIONS, but loan age is feature index 5 and can be
set independently. This checks whether the model produces a sensible seasoned
S-curve when age is pushed past the training range (1-33 months).
"""
import sys, os, pickle, json; sys.path.insert(0,'scripts')
import numpy as np, torch
import importlib
m = importlib.import_module("stage2_forecast_cpr_gfee050")

BASE="/scratch/at7095/mortgage_prepayment"
MAX_SEQ, N_FEAT, DEAD = 33, 9, [7,8]
REP = dict(cs=740.0, oltv=75.0, cltv=70.0, upb=250000.0, dti=35.0, lp=0.0, pt=0.0)

model  = m.load_model()
scaler = pickle.load(open(os.path.join(BASE,"data/sequences/scaler.pkl"),"rb"))
cal    = json.load(open(os.path.join(BASE,"config/hazard_calibration_cpr_forecast.json")))
a = float(cal.get("a", cal.get("platt_a"))); b = float(cal.get("b", cal.get("platt_b")))

print("scaler age (feature 5): mean=%.3f scale=%.3f  -> training ages 1-33 map to z %.2f..%.2f"
      % (scaler.mean_[5], scaler.scale_[5],
         (1-scaler.mean_[5])/scaler.scale_[5], (33-scaler.mean_[5])/scaler.scale_[5]))
for age0 in [61, 121]:
    print("  age %d-%d maps to z %.2f..%.2f (EXTRAPOLATION)"
          % (age0, age0+32, (age0-scaler.mean_[5])/scaler.scale_[5],
             (age0+32-scaler.mean_[5])/scaler.scale_[5]))

def cpr_at(inc, age_start):
    s = np.zeros((1, MAX_SEQ, N_FEAT), dtype=np.float32)
    s[:,:,0] = inc
    s[:,:,1] = REP["cs"]; s[:,:,2] = REP["oltv"]; s[:,:,3] = REP["cltv"]; s[:,:,4] = REP["upb"]
    s[:,:,5] = np.arange(age_start, age_start+MAX_SEQ)[None,:]
    s[:,:,6] = REP["dti"]; s[:,:,7] = REP["lp"]; s[:,:,8] = REP["pt"]
    f = scaler.transform(s.reshape(-1,N_FEAT)).reshape(1,MAX_SEQ,N_FEAT)
    for c in DEAD: f[:,:,c] = 0.0
    with torch.no_grad():
        lg = model(torch.tensor(f), mask=torch.ones(1,MAX_SEQ,dtype=torch.bool),
                   return_per_timestep=True).numpy()
    smm = 1.0/(1.0+np.exp(-(a*lg+b)))
    return (1.0-(1.0-smm)**12)[0]

print("\n=== mean CPR over the window, by starting age and incentive ===")
ages = [1, 13, 25, 37, 61, 85, 121]
incs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
print("%6s" % "age", "".join("%9.1f" % i for i in incs))
for ag in ages:
    row = [cpr_at(i, ag).mean() for i in incs]
    flag = "" if ag <= 1 else ("  <-- extrapolated" if ag > 33 else "")
    print("%6d" % ag, "".join("%9.4f" % v for v in row), flag)

print("\n=== floor and saturation by age (inc -3.0 vs +2.0) ===")
print("%6s %10s %10s %8s" % ("age","floor","sat","ratio"))
for ag in ages:
    lo, hi = cpr_at(-3.0, ag).mean(), cpr_at(2.0, ag).mean()
    print("%6d %10.4f %10.4f %8.2f" % (ag, lo, hi, hi/lo))
print("\nrealized reference: floor ~0.035-0.055 at deep discount, sat ~0.21 at +2")
