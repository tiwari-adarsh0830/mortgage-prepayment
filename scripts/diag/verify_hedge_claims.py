import sys; sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from model_hedge_krd import load_hazard, cpr_path, GFEE, COUPONS

def ols(y, X):
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ co; n, k = X.shape
    se = np.sqrt(np.diag(float(r @ r) / (n - k) * np.linalg.pinv(X.T @ X)))
    return co, se

for tag, label in [("10", "LOCALIZED (his literal spec)"), ("10_span", "SPANNING (variant)")]:
    p = pd.read_csv("outputs/model_hedge_panel_%s.csv" % tag)
    print("=" * 70)
    print("%s   n_months=%d" % (label, p["ret_month"].nunique()))
    print("%4s %10s %10s %10s %10s %8s" %
          ("cpn", "t_lvl", "resid_dur", "model_D", "sum", "emp_D"))
    worst = 0.0
    for c, g in p.groupby("coupon"):
        g = g.dropna(subset=["hedged", "d_level", "d_slope"])
        X = np.column_stack([np.ones(len(g)), g["d_level"], g["d_slope"]])
        co, se = ols(g["hedged"].values, X)
        resid_dur = -100.0 * co[1]
        model_D = g["D_level"].mean()
        # empirical benchmark on EXACTLY these months
        raw = g["tba_total_return"].values - g["income"].values
        co2, _ = ols(raw, X)
        emp_D = -100.0 * co2[1]
        s = resid_dur + model_D
        worst = max(worst, abs(s - emp_D))
        print("%4s %10.2f %10.3f %10.3f %10.3f %8.3f" %
              (c, co[1] / se[1], resid_dur, model_D, s, emp_D))
    print("  worst |sum - emp| = %.4f years" % worst)
    print()

print("=" * 70)
print("PANEL-WIDE terminal CPR actually assumed (not single-scenario)")
model, scaler, a, b = load_hazard()
p = pd.read_csv("outputs/model_hedge_panel_10_span.csv")
print("%4s %10s %10s %10s %10s" % ("cpn", "mean", "min", "max", "n"))
for c, g in p.groupby("coupon"):
    term = [cpr_path((c + GFEE) - pm, model, scaler, a, b)[32] for pm in g["pmms"]]
    term = np.array(term)
    print("%4s %10.4f %10.4f %10.4f %10d" % (c, term.mean(), term.min(), term.max(), len(term)))

print()
print("PMMS range in panel: %.2f to %.2f" % (p["pmms"].min(), p["pmms"].max()))
