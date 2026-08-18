"""External validation of the bootstrap against QuantLib.

Par-repricing proves SELF-consistency, not correctness: a bootstrap can reprice
its own inputs exactly while reading the quotes under the wrong convention.
QuantLib is an independent implementation with independently-chosen conventions,
so agreement is evidence the conventions are right.

Runs BOTH bootstraps against it. If QL agrees with v3 and disagrees with the
original by roughly the magnitude the original fails par-repricing by, that is
two independent confirmations of the same defect.

Expected noise: day-count/calendar differences vs the FRED par series give ~1bp
scatter. That is NOT a bug. A systematic drift growing with maturity, or a
short-end offset, WOULD be.
"""
import numpy as np, pandas as pd, os, sys
import QuantLib as ql
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import model_hedge_krd as M
from bootstrap_v3 import bootstrap_zeros_v3

LAB, YRS = M.MAT_LABELS, M.MAT_YEARS
BILLS = [('1mo', ql.Period(1, ql.Months)), ('3mo', ql.Period(3, ql.Months)),
         ('6mo', ql.Period(6, ql.Months))]
BONDS = [('1yr', 1), ('2yr', 2), ('3yr', 3), ('5yr', 5),
         ('7yr', 7), ('10yr', 10), ('20yr', 20), ('30yr', 30)]

def ql_zeros(par, asof, n_months=360):
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    d = ql.Date(asof.day, asof.month, asof.year)
    ql.Settings.instance().evaluationDate = d
    dc_bill = ql.Actual360()
    dc_bond = ql.ActualActual(ql.ActualActual.Bond)
    helpers = []
    for lab, tenor in BILLS:
        helpers.append(ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(float(par[lab]) / 100.0)),
            tenor, 0, cal, ql.ModifiedFollowing, False, dc_bill))
    for lab, yrs in BONDS:
        sched = ql.Schedule(d, cal.advance(d, ql.Period(yrs, ql.Years)),
                            ql.Period(ql.Semiannual), cal,
                            ql.Unadjusted, ql.Unadjusted,
                            ql.DateGeneration.Backward, False)
        helpers.append(ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(100.0)), 0, 100.0, sched,
            [float(par[lab]) / 100.0], dc_bond, ql.Unadjusted))
    curve = ql.PiecewiseLogCubicDiscount(d, helpers, dc_bond)
    curve.enableExtrapolation()
    out = []
    for m in range(1, n_months + 1):
        t = m / 12.0
        tgt = d + ql.Period(m, ql.Months)
        df = curve.discount(tgt)
        yrs = dc_bond.yearFraction(d, tgt)
        out.append(-np.log(df) / max(yrs, 1e-9) * 100.0)
    return np.array(out)

d = pd.read_csv(os.path.join("data", "treasury_yields.csv")).dropna(subset=LAB)
d['DATE'] = pd.to_datetime(d['DATE'])
me = d.groupby(d['DATE'].dt.to_period('M')).last()
me = me[me.index >= '2018-01']
sel = [0, len(me) // 4, len(me) // 2, 3 * len(me) // 4, len(me) - 1]

print("=" * 76)
print("ZERO CURVE: QuantLib vs v3 vs original   (differences in bp)")
print("=" * 76)
agg_new, agg_old = [], []
for i in sel:
    r = me.iloc[i]
    asof = pd.Timestamp(str(me.index[i])) + pd.offsets.MonthEnd(0)
    par = {l: float(r[l]) for l in LAB}
    try:
        zq = ql_zeros(par, asof)
    except Exception as e:
        print("  [QL failed %s: %s]" % (me.index[i], e)); continue
    zn = bootstrap_zeros_v3(par)
    zo = M.bootstrap_zeros(par)
    dn, do = (zn - zq) * 100, (zo - zq) * 100
    agg_new.append(dn); agg_old.append(do)
    print("\n--- %s   (1yr %.2f  10yr %.2f  30yr %.2f) ---"
          % (me.index[i], par['1yr'], par['10yr'], par['30yr']))
    print("%8s%10s%10s%10s | %9s%9s" % (
        "month", "QL", "v3", "orig", "v3-QL", "orig-QL"))
    for m in (1, 6, 12, 60, 120, 240, 360):
        print("%8d%10.4f%10.4f%10.4f | %9.2f%9.2f" % (
            m, zq[m-1], zn[m-1], zo[m-1], dn[m-1], do[m-1]))

if agg_new:
    an, ao = np.concatenate(agg_new), np.concatenate(agg_old)
    print("\n" + "=" * 76)
    print("SUMMARY over %d curves x 360 nodes  (bp vs QuantLib)" % len(agg_new))
    print("=" * 76)
    print("%12s%12s%12s%12s" % ("", "mean", "mean|.|", "max|.|"))
    print("%12s%12.3f%12.3f%12.3f" % ("v3", an.mean(), np.abs(an).mean(), np.abs(an).max()))
    print("%12s%12.3f%12.3f%12.3f" % ("original", ao.mean(), np.abs(ao).mean(), np.abs(ao).max()))
    print("\nv3 close to QL and original not -> defect independently confirmed")
    print("both far from QL -> convention mismatch somewhere shared; investigate")
