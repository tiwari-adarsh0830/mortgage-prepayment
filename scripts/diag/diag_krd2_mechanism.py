#!/usr/bin/env python3
"""diag_krd2_mechanism.py -- does krd2 have any prepayment channel, or is it
pure discounting?

Hypothesis: the 2yr bump was designed to never move PMMS (my decision --
PMMS is a 30yr survey rate tracking the long end). That means inc = note -
pmms is IDENTICAL under +25bp and -25bp 2yr bumps, so cpr_path() is called
with the same incentive both times and returns the same cached path. krd2 can
therefore only reflect discounting of near-term cashflows under a curve bump
that tapers to zero by 5yr -- structurally it cannot respond to prepayment
risk, which is the dominant source of MBS curve exposure. If true, that is a
mechanical, expected reason D_curve doesn't track realized dy2, not evidence
the tent geometry or the hazard model is broken.

Checked two ways: (1) is krd2 small relative to krd5/krd10, as pure short-end
discounting on a long-duration amortizing bond would predict; (2) does krd2
vary at all across coupon/month in a way consistent with discounting only
(should scale with the SHAPE of near-term cashflows, which differs by coupon
mainly through the CPR-driven principal paydown speed -- itself unaffected by
the 2yr bump, so krd2 should mostly track scheduled amortization + the
UNBUMPED CPR path, which does vary by coupon even though it can't respond to
the 2yr shock itself).
"""
import pandas as pd
import numpy as np

p = pd.read_csv("/scratch/at7095/mortgage_prepayment/outputs/model_hedge_panel_10_tents3_pinnedfixed.csv")

print("=== magnitude: krd2 vs krd5 vs krd10, by coupon ===")
g = p.groupby("coupon")[["krd2", "krd5", "krd10"]].mean()
g["krd2_pct_of_total"] = 100 * g.krd2 / (g.krd2 + g.krd5 + g.krd10)
print(g.round(3).to_string())

print("\n=== krd2 variation across months, by coupon (std, and std/mean) ===")
s = p.groupby("coupon")["krd2"].agg(["mean", "std"])
s["cv"] = s["std"] / s["mean"].abs()
print(s.round(4).to_string())

print("\n=== reading ===")
print("  krd2 as % of total duration tells us whether it's economically small")
print("  (consistent with discounting-only) or comparable to krd5/krd10 (would")
print("  argue against the pure-discounting explanation).")
print("  If krd2 is both small AND its variation across months doesn't track")
print("  dy2 (checked already: corr ~0), the story holds: krd2 measures")
print("  something real but small and mechanically disconnected from 2yr risk")
print("  as currently wired, because CPR can't see the 2yr bump at all.")
