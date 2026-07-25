import sys; sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from model_hedge_krd import (load_hazard, cpr_path, price_path,
                             bootstrap_zeros, GFEE, N_MONTHS, COUPONS)

model, scaler, a, b = load_hazard()
daily = pd.read_csv("data/treasury_yields.csv", index_col=0, parse_dates=True).sort_index()
par = daily.iloc[-1].to_dict()
pmms = 6.60

print("=== CPR path by coupon (m1, m6, m12, m24, m33) ===")
print("%4s %6s %8s %8s %8s %8s %8s" % ("cpn","inc","m1","m6","m12","m24","m33"))
paths = {}
for c in COUPONS:
    inc = (c + GFEE) - pmms
    p = cpr_path(inc, model, scaler, a, b)
    paths[c] = p
    print("%4s %6.2f %8.4f %8.4f %8.4f %8.4f %8.4f"
          % (c, inc, p[0], p[5], p[11], p[23], p[32]))

print()
print("=== still ramping at m33?  (m33/m12) ===")
for c in COUPONS:
    print("  cpn %s: m33/m12 = %.3f" % (c, paths[c][32] / paths[c][11]))

print()
print("=== flat lifetime CPR needed to reproduce empirical duration ===")
emp = {2.5:7.407, 3.0:6.718, 3.5:6.113, 4.0:5.366, 4.5:4.631,
       5.0:3.806, 5.5:3.011, 6.0:2.222, 6.5:1.529}
z0 = bootstrap_zeros(par)

def dur_flat(c, cpr, h=0.25):
    up = {k: v + h for k, v in par.items()}
    dn = {k: v - h for k, v in par.items()}
    arr = np.full(N_MONTHS, cpr)
    p0 = price_path(c, arr, z0)
    return (price_path(c, arr, bootstrap_zeros(dn))
            - price_path(c, arr, bootstrap_zeros(up))) / (2 * p0 * (h / 100.0))

print("%4s %16s %18s %9s" % ("cpn","model_term_CPR","CPR_to_match_emp","emp_dur"))
for c in COUPONS:
    lo, hi = 0.001, 0.60
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if dur_flat(c, mid) > emp[c]:
            lo = mid
        else:
            hi = mid
    print("%4s %16.4f %18.4f %9.3f" % (c, paths[c][32], (lo + hi) / 2.0, emp[c]))
