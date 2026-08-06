#!/usr/bin/env python3
"""verify_map_modes.py -- second-path verification of the four map-mode panels.

Every t-statistic that goes in an email must be confirmed through code that did
not produce it. model_hedge_krd.py prints its verification table using ols() and
np.linalg.lstsq; verify_before_email.recompute() rebuilds the same quantities
from the saved panel using normal equations (np.linalg.solve on X'X). This script
imports recompute unchanged and points it at all four panels, so
verify_before_email.py itself stays untouched as the independent path.

Panels compared:
  .MAPOFF          --map-mode off        control, must reproduce the pre-mapping
                                         t-statistics exactly
  _mapscalar       --map-mode scalar     factor recomputed under each bump
  _mappointwise    --map-mode pointwise  mapping applied to each of the 33 months
  _mapfrozen       --map-mode frozen     factor fixed at the unbumped incentive

Guard: the panels must share ret_month coverage and coupon set, or the
t-statistics are not comparable across modes and the whole table is misleading.
Checked explicitly rather than assumed.
"""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd

BASE = "/scratch/at7095/mortgage_prepayment"
os.chdir(BASE)                      # recompute() uses repo-relative paths
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, os.path.join(BASE, "scripts", "diag"))

PANELS = [
    ("outputs/model_hedge_panel_10_span_pinnedfixed.MAPOFF.csv",
     "map off (control)"),
    ("outputs/model_hedge_panel_10_span_pinnedfixed_mapfrozen.csv",
     "map frozen"),
    ("outputs/model_hedge_panel_10_span_pinnedfixed_mapscalar.csv",
     "map scalar"),
    ("outputs/model_hedge_panel_10_span_pinnedfixed_mappointwise.csv",
     "map pointwise"),
]

# t-statistics from the Aug 3 email, pinned floor 0.0459, three-regressor spec
# excluded -- these are the two-regressor level t-stats the control must match.
CONTROL_EXPECTED = {2.5: -6.57, 3.0: -6.99, 3.5: -7.35, 4.0: -6.07, 4.5: -4.46,
                    5.0: -2.63, 5.5: -2.07, 6.0: -1.80, 6.5: -1.49}


def load_recompute():
    """Import verify_before_email without executing its module-level calls.

    The file ends with two recompute(...) invocations and a print; a plain import
    would run them. Read the source, strip everything from the first module-level
    recompute( call onward, and exec the remainder."""
    src = open("scripts/diag/verify_before_email.py").read()
    marker = "\nrecompute(FIT"
    if marker in src:
        src = src[:src.index(marker)]
    ns = {"__name__": "verify_before_email_partial", "__file__":
          os.path.join(BASE, "scripts/diag/verify_before_email.py")}
    exec(compile(src, "verify_before_email.py", "exec"), ns)
    if "recompute" not in ns:
        raise RuntimeError("recompute() not found after trimming module tail")
    return ns["recompute"]


def coverage_guard():
    frames = {}
    for path, label in PANELS:
        if not os.path.exists(path):
            print("MISSING: %s" % path)
            continue
        d = pd.read_csv(path)
        frames[label] = d
        print("  %-20s rows %4d  months %3d  coupons %d  (%s .. %s)"
              % (label, len(d), d.ret_month.nunique(), d.coupon.nunique(),
                 d.ret_month.min(), d.ret_month.max()))
    if len(frames) < 2:
        return frames
    ref_label, ref = next(iter(frames.items()))
    ok = True
    for label, d in frames.items():
        if (set(d.ret_month) != set(ref.ret_month)
                or set(d.coupon) != set(ref.coupon)):
            print("  MISMATCH: %s does not share coverage with %s" % (label, ref_label))
            ok = False
    print("  coverage identical across panels: %s" % ok)
    if not ok:
        print("  -> t-statistics are NOT comparable across modes; stop here.")
    return frames


def control_check(recompute):
    """The control panel must reproduce the pre-mapping level t-statistics."""
    path = PANELS[0][0]
    p = pd.read_csv(path)
    p["ret_month"] = pd.to_datetime(p["ret_month"])
    print("\n=== control check against the Aug 3 pinned-floor numbers ===")
    print("%6s %10s %10s %8s" % ("cpn", "recomputed", "expected", "diff"))
    worst = 0.0
    for c, g in p.sort_values(["coupon", "ret_month"]).groupby("coupon"):
        g = g.dropna(subset=["hedged", "d_level", "d_slope"])
        X = np.column_stack([np.ones(len(g)), g.d_level.values, g.d_slope.values])
        y = g.hedged.values
        XtX = X.T @ X
        co = np.linalg.solve(XtX, X.T @ y)
        r = y - X @ co
        se = np.sqrt(np.diag(float(r @ r) / (len(y) - 3) * np.linalg.inv(XtX)))
        t = co[1] / se[1]
        exp = CONTROL_EXPECTED.get(round(c, 1))
        d = abs(t - exp) if exp is not None else float("nan")
        worst = max(worst, 0.0 if np.isnan(d) else d)
        print("%6.1f %10.2f %10.2f %8.3f" % (c, t, exp if exp else float("nan"), d))
    print("  worst absolute difference: %.4f" % worst)
    print("  -> %s" % ("control reproduces the prior panel"
                       if worst < 0.005 else
                       "CONTROL DOES NOT MATCH -- investigate before using any of this"))


def main():
    print("=== panel coverage ===")
    coverage_guard()
    recompute = load_recompute()
    control_check(recompute)
    print("\n=== recompute() via normal equations, all four modes ===")
    for path, label in PANELS:
        if os.path.exists(path):
            recompute(path, label)
    print("\nCompare each t_lvl against the value printed by model_hedge_krd.py")
    print("for the same mode. Any disagreement is a bug in one of the two paths.")


if __name__ == "__main__":
    main()
