#!/usr/bin/env python3
"""verify_all_claims_final.py -- independent re-derivation of every claim that
could go in the email to the advisor. Nothing here imports or trusts the
output of any earlier diagnostic script tonight; every number is recomputed
from raw source files. Each check prints PASS/FAIL/RANGE explicitly.
"""
import os
import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
os.chdir(BASE)

print("="*78)
print("CLAIM 1: three-tent weights are a strict refinement of the old spanning")
print("pair -- w2(T)+w5(T) equals old spanning w5(T), and new w10(T) equals old")
print("spanning w10(T), at every node in MAT_YEARS.")
print("="*78)

MAT_YEARS = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
T = np.asarray(MAT_YEARS, float)

w5_old = np.clip((10.0 - T)/5.0, 0.0, 1.0)
w10_old = 1.0 - w5_old

w2_new = np.where(T <= 2.0, 1.0, np.where(T <= 5.0, (5.0-T)/3.0, 0.0))
w5_new = np.where(T <= 2.0, 0.0,
                  np.where(T <= 5.0, (T-2.0)/3.0,
                           np.where(T <= 10.0, (10.0-T)/5.0, 0.0)))
w10_new = np.where(T <= 5.0, 0.0, np.where(T <= 10.0, (T-5.0)/5.0, 1.0))

sum3 = w2_new + w5_new + w10_new
err_w5 = np.abs((w2_new + w5_new) - w5_old).max()
err_w10 = np.abs(w10_new - w10_old).max()
err_unity = np.abs(sum3 - 1.0).max()

print("  T:        " + "  ".join("%6.2f" % t for t in T))
print("  old w5:   " + "  ".join("%6.3f" % v for v in w5_old))
print("  new w2+w5:" + "  ".join("%6.3f" % v for v in (w2_new+w5_new)))
print("  old w10:  " + "  ".join("%6.3f" % v for v in w10_old))
print("  new w10:  " + "  ".join("%6.3f" % v for v in w10_new))
print("  max |old_w5 - (new_w2+new_w5)| = %.10f" % err_w5)
print("  max |old_w10 - new_w10|        = %.10f" % err_w10)
print("  max |sum of three tents - 1|   = %.10f" % err_unity)
print("  PASS" if (err_w5 < 1e-9 and err_w10 < 1e-9 and err_unity < 1e-9) else "  FAIL")


print("\n" + "="*78)
print("CLAIM 2: the curvature definition used, d_curve = 2*dy5 - dy2 - dy10,")
print("equals 2x the advisor's literal words 'the middle moving against the two")
print("ends' = dy5 - (dy2+dy10)/2. Reparameterization L,S,C inverts exactly.")
print("="*78)

rng = np.random.default_rng(0)
dy2, dy5, dy10 = rng.normal(size=(3, 5000))
C_mine = 2*dy5 - dy2 - dy10
C_theirs = dy5 - (dy2+dy10)/2.0
ratio = C_mine / C_theirs
print("  ratio C_mine/C_theirs: min %.6f max %.6f (should be exactly 2.0)"
      % (ratio.min(), ratio.max()))

L = (dy2+dy5+dy10)/3.0
S = dy10 - dy2
C = 2*dy5 - dy2 - dy10
dy2_r = L - S/2 - C/6
dy5_r = L + C/3
dy10_r = L + S/2 - C/6
err = max(np.abs(dy2_r-dy2).max(), np.abs(dy5_r-dy5).max(), np.abs(dy10_r-dy10).max())
print("  reparam round-trip max abs error: %.2e" % err)
print("  PASS" if (abs(ratio.min()-2.0) < 1e-9 and abs(ratio.max()-2.0) < 1e-9
                   and err < 1e-9) else "  FAIL")


print("\n" + "="*78)
print("CLAIM 3: the 2yr leg cannot respond to prepayment risk, because the")
print("incentive fed to cpr_path is identical under +25bp and -25bp on the 2yr")
print("tenor (dp=0 for that leg). Checked by reading the actual source, not by")
print("re-deriving the pricer's logic from memory.")
print("="*78)

src = open("scripts/model_hedge_krd.py").read()
i = src.find("def krd_triple")
block = src[i:i+2200] if i >= 0 else "krd_triple NOT FOUND"
print(block)
found_zero = ("ten == '2yr'" in block) and ("dp = 0.0" in block)
print("\n  source shows dp=0.0 unconditionally when ten=='2yr': %s" % found_zero)
print("  PASS -- confirmed from source" if found_zero else "  FAIL -- re-check source manually")


print("\n" + "="*78)
print("CLAIM 4: krd2 is small relative to krd5/krd10 at low/mid coupons, and its")
print("correlation with realized dy2 is near zero. Re-derived independently from")
print("the raw saved panel and a freshly rebuilt dy2 series (not reused).")
print("="*78)

PANEL = "outputs/model_hedge_panel_10_tents3_pinnedfixed.csv"
if not os.path.exists(PANEL):
    print("  PANEL MISSING: %s -- cannot check" % PANEL)
else:
    p = pd.read_csv(PANEL)
    t = pd.read_csv("data/treasury_yields.csv", parse_dates=["DATE"])
    me = t.set_index("DATE")[["2yr","5yr","10yr"]].sort_index().resample("ME").last()
    me["dy2"] = me["2yr"].diff()
    me = me.reset_index(); me["key"] = me.DATE.dt.to_period("M")
    p["key"] = pd.PeriodIndex(p.ret_month, freq="M")
    p = p.merge(me[["key","dy2"]], on="key", how="left")

    chk = p.dropna(subset=["dy2"])
    recon_check = None
    if "d_level" in p.columns:
        me5 = t.set_index("DATE")[["5yr","10yr"]].sort_index().resample("ME").last()
        me5["dl"] = (me5["5yr"]+me5["10yr"]).div(2).diff()
        me5 = me5.reset_index(); me5["key"] = me5.DATE.dt.to_period("M")
        cc = p.merge(me5[["key","dl"]], on="key", how="left").dropna(subset=["dl","d_level"])
        recon_check = float(np.corrcoef(cc.dl, cc.d_level)[0,1])
    print("  merge-key sanity (d_level reconstruction corr): %.4f (must be >0.95)"
          % (recon_check if recon_check is not None else float("nan")))

    g = p.groupby("coupon")[["krd2","krd5","krd10"]].mean()
    g["krd2_pct"] = 100*g.krd2/(g.krd2+g.krd5+g.krd10)
    print(g.round(3).to_string())

    print("\n  corr(krd2, dy2) by coupon:")
    for c in sorted(p.coupon.unique()):
        gc = p[p.coupon==c].dropna(subset=["krd2","dy2"])
        print("    %.1f: %.4f" % (c, gc.krd2.corr(gc.dy2)))

    ok4 = (recon_check is not None and recon_check > 0.95
           and g.loc[2.5,"krd2_pct"] < 15)
    print("\n  PASS (krd2 small at low coupons, alignment verified)" if ok4
          else "  CHECK MANUALLY -- see numbers above")


print("\n" + "="*78)
print("CLAIM 5: adding curvature as a third fitted regressor does not reduce the")
print("count of coupons with |t_dy2|>2 versus two-factor (both should be 7 of 9).")
print("Re-derived fresh, not reusing verify_tents3.py's printed conclusion.")
print("="*78)

BASELINE = "outputs/model_hedge_panel_10_span_pinnedfixed.MAPOFF.csv"
MIN_WINDOW = 36

def ols(y, X):
    XtX = X.T @ X
    co = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ co
    se = np.sqrt(np.diag(float(r@r)/(len(y)-X.shape[1])*np.linalg.inv(XtX)))
    return co, se

def dy2_series(panel):
    t = pd.read_csv("data/treasury_yields.csv", parse_dates=["DATE"])
    me = t.set_index("DATE")[["2yr"]].sort_index().resample("ME").last()
    me["dy2"] = me["2yr"].diff()
    me = me.reset_index(); me["key"] = me.DATE.dt.to_period("M")
    p = panel.copy(); p["key"] = pd.PeriodIndex(p.ret_month, freq="M")
    return p.merge(me[["key","dy2"]], on="key", how="left")

if os.path.exists(PANEL) and os.path.exists(BASELINE):
    p3 = dy2_series(pd.read_csv(PANEL))
    p2 = dy2_series(pd.read_csv(BASELINE))
    n2, n3 = 0, 0
    for c in sorted(p3.coupon.unique()):
        for panel_df, factors, counter_name in [(p2, ["d_level","d_slope"], "2f"),
                                                  (p3, ["d_level","d_slope","d_curve"], "3f")]:
            g = panel_df[panel_df.coupon==c].dropna(
                subset=factors+["dy2","tba_total_return","income"]).sort_values("ret_month").reset_index(drop=True)
            y = (g.tba_total_return - g.income).values
            X = np.column_stack([np.ones(len(g))]+[g[f].values for f in factors])
            hed = np.full(len(g), np.nan)
            for i in range(MIN_WINDOW, len(g)):
                co,_ = ols(y[:i], X[:i])
                hed[i] = y[i] + (X[i,1:] @ (-100*co[1:]))/100.0
            m = ~np.isnan(hed) & ~g.dy2.isna()
            if m.sum() < 10: continue
            co, se = ols(hed[m], np.column_stack([np.ones(m.sum()), g.dy2[m].values]))
            tstat = co[1]/se[1]
            sig = abs(tstat) > 2
            if counter_name=="2f": n2 += int(sig)
            else: n3 += int(sig)
    print("  2-factor coupons with |t_dy2|>2: %d of 9" % n2)
    print("  3-factor coupons with |t_dy2|>2: %d of 9" % n3)
    print("  PASS (matches claimed 7-of-9, 7-of-9)" if (n2==7 and n3==7)
          else "  MISMATCH vs earlier claim -- do not use old numbers")
else:
    print("  PANEL(S) MISSING -- cannot check")


print("\n" + "="*78)
print("CLAIM 6: control panels are byte-identical to pre-tents3-patch outputs.")
print("Direct file comparison, not a re-run.")
print("="*78)
a = "outputs/model_hedge_panel_10_span_pinnedfixed.MAPOFF.csv"
b = "outputs/model_hedge_panel_10_span_pinnedfixed.csv"
if os.path.exists(a) and os.path.exists(b):
    da, db = pd.read_csv(a), pd.read_csv(b)
    if da.shape != db.shape:
        print("  SHAPE MISMATCH %s vs %s -- FAIL" % (da.shape, db.shape))
    else:
        num = da.select_dtypes("number").columns
        d = (da[num]-db[num]).abs().max().max()
        print("  max abs diff across all numeric columns: %.2e" % d)
        print("  PASS" if d < 1e-9 else "  FAIL")
else:
    print("  one or both files missing -- cannot check")


print("\n" + "="*78)
print("CLAIM 7: PMMS/2yr sensitivity is genuinely unresolved in the 0.4-1.1")
print("range -- MS vs ME resample gives near-identical results (ruled out as the")
print("explanation), but lag0 vs lag1 gives a persistent, real gap.")
print("="*78)
pm = pd.read_csv("data/pmms_monthly.csv")
pm["date"] = pd.to_datetime(pm.year.astype(str)+"-"+pm.month.astype(str).str.zfill(2))
pm = pm.set_index("date")[["rate_30yr"]].sort_index()
t = pd.read_csv("data/treasury_yields.csv", parse_dates=["DATE"])
for resample_code, lab in [("ME","month-end"), ("MS","month-start")]:
    me = t.set_index("DATE")[["2yr","5yr","10yr"]].sort_index().resample(resample_code).last()
    if resample_code == "ME":
        me.index = me.index.to_period("M").to_timestamp()
    m = pm.join(me, how="inner").dropna()
    m["dpmms"]=m.rate_30yr.diff(); m["dy2"]=m["2yr"].diff()
    m["dy5"]=m["5yr"].diff(); m["dy10"]=m["10yr"].diff()
    m = m.dropna()
    m2 = m[m.index >= "2018-01-01"]
    for lagname, s in [("lag0", m2.dy2), ("lag1", m2.dy2.shift(1))]:
        mm = m2.assign(x=s).dropna(subset=["x"])
        X = np.column_stack([np.ones(len(mm)), mm.x.values])
        co, se = ols(mm.dpmms.values, X)
        print("  %-11s %-4s beta=%.3f t=%.2f n=%d" % (lab, lagname, co[1], co[1]/se[1], len(mm)))
print("  If both resample conventions give similar lag0/lag1 gaps, resample")
print("  choice is confirmed NOT the explanation (matches earlier finding) --")
print("  report as an unresolved range, not a single number.")


print("\n" + "="*78)
print("SUMMARY -- read every PASS/FAIL/CHECK MANUALLY line above before drafting")
print("anything. Do not round any RANGE claim into a single point estimate.")
print("="*78)
