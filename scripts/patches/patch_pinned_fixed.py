"""
Add a fourth floor mode: pinned-fixed at 0.0459.

Advisor (2026-08-03): "Yes, I agree it makes sense to refit using the floor
(0.0459) rather than the current (0.0546)."

0.0459 is the realized ALL-LOAN mean CPR at incentive <= -2.5 over the full
sample (bootstrap SE 0.0010), from diag_seasoned_floor_check.py. Note he chose
the all-loan value, not the seasoned 0.0516.

LOOK-AHEAD, to be disclosed rather than silently worked around. Every other
mode fits on an expanding window (data strictly before the cutoff) so the
ratios use prior data only. A fixed 0.0459 is a full-sample statistic applied
at every cutoff, including months before enough deep-discount data existed to
estimate it. This is what was asked for and is fine as a diagnostic, but it is
not a production spec. An expanding-window all-loan version would avoid this;
it may also hit the n=0 problem that killed pinned-seasoned, since there were
no deep-discount observations at all before 2018 in this panel.

Run from /scratch/at7095/mortgage_prepayment. Asserts exactly one match per
replacement; writes .bak only after all assertions pass.
"""
import os
import shutil

P = "scripts/model_hedge_krd.py"
assert os.path.exists(P), "run from /scratch/at7095/mortgage_prepayment"
src = open(P).read()


def rep(s, old, new, label):
    n = s.count(old)
    assert n == 1, "%s: expected 1 match, got %d" % (label, n)
    return s.replace(old, new)


# 1. constant next to the existing globals
src = rep(src,
          'DEEP_INC = -2.5                # depth at which the pinned floor is estimated',
          'DEEP_INC = -2.5                # depth at which the pinned floor is estimated\n'
          'FIXED_FLOOR = 0.0459           # advisor 2026-08-03: realized all-loan CPR at inc <= -2.5\n'
          '                               # FULL-SAMPLE value -- introduces look-ahead by construction',
          "constant")

# 2. new branch, mirroring pinned-seasoned but with a constant floor
old2 = '''    if FLOOR_MODE == "pinned-seasoned":
        _s = _load_seasoned()'''
new2 = '''    if FLOOR_MODE == "pinned-fixed":
        _fl = FIXED_FLOOR

        def _sc_fix(x, sa, k, x0):
            return _fl + (sa - _fl) / (1.0 + np.exp(-k * (x - x0)))

        pof, _ = _cf(_sc_fix, h["inc"].values, h["cpr_upb"].values,
                     p0=[0.22, 3.0, 0.4],
                     bounds=([0.05, 0.2, -2.0], [0.60, 10.0, 3.0]), maxfev=40000)
        q = dict(floor=_fl, sat=float(pof[0]), k=float(pof[1]), x0=float(pof[2]),
                 n=int(len(h)))
    elif FLOOR_MODE == "pinned-seasoned":
        _s = _load_seasoned()'''
src = rep(src, old2, new2, "branch")

# 3. the trailing else must not also fire for pinned-fixed; it is already an
#    elif chain after the edit above, so only the choices list needs updating
src = rep(src,
          '                    choices=["fitted", "seasoned-fit", "pinned-seasoned"],',
          '                    choices=["fitted", "seasoned-fit", "pinned-seasoned",\n'
          '                             "pinned-fixed"],',
          "choices")

shutil.copy(P, P + ".bak2")
open(P, "w").write(src)
print("patched %s (backup at %s.bak2)" % (P, P))
print("modes: fitted | seasoned-fit | pinned-seasoned | pinned-fixed")
