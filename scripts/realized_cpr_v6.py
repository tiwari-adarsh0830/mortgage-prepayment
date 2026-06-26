"""
Realized CPR by coupon bucket by month — v6.

v6 = v5's UPB==0 prepayment detection + a correct MMYYYY->YYYYMM ordering fix.

WHY NOT zero-balance-code: col 106 (zero_balance_code) is NOT a one-time payoff
stamp in this data -- it persists for many months once set, so using zbc==1 as an
event collapses the numerator and inflates the at-risk denominator. v5's UPB==0
measure gives correct boom-era magnitudes (Dec-2020 3.5% ~ 40% CPR) and is kept.

THE BUG v6 FIXES: v5 computed each loan's "global last appearance" with idxmax on
RAW MMYYYY integers. MMYYYY is non-monotonic (122020 > 62024 as ints, but Dec-2020
precedes Jun-2024), so for any loan whose true last month is numerically smaller
than an earlier month, v5 picked the WRONG last row and checked UPB there. Near the
data boundary this silently missed payoffs -> realized CPR was spuriously 0 for all
of 2024-2025. v6 converts MMYYYY->YYYYMM before every ordering/comparison.

PASSES
  Pass 0 (global): per loan, note rate + global last appearance (YYYYMM) and its UPB;
                   prepaid if UPB==0 at that true-last row; prepay_month = that YYYYMM
  Pass 1 (per-file): count at-risk and new prepayments per (coupon_bucket, month)

OUTPUT: outputs/realized_cpr_by_coupon_v6.csv
"""

import pandas as pd
import pickle
import numpy as np
import glob
import os
from collections import defaultdict

BASE = "/scratch/at7095/mortgage_prepayment"
RAW  = os.path.join(BASE, "data/raw")
OUT  = os.path.join(BASE, "outputs")

COL_LOAN  = 1
COL_MONTH = 2
COL_RATE  = 7
COL_UPB   = 11
GFEE      = 0.50
CHUNK     = 2_000_000
CKPT_P0   = os.path.join(OUT, "realized_v6_pass0_checkpoint.pkl")


def mmyyyy_to_yyyymm(m):
    """MMYYYY int -> YYYYMM int for correct chronological ordering.
    122020 (Dec2020) -> 202012 ; 62024 (Jun2024) -> 202406."""
    yyyy = m % 10000
    mm   = m // 10000
    return yyyy * 100 + mm


def parse_date_yyyymm(v):
    yyyy = int(v) // 100
    mm   = int(v) % 100
    return pd.Timestamp(year=yyyy, month=mm, day=1)


def pass0_global_last(files):
    """Per loan: note rate + global last appearance (YYYYMM) and UPB there.
    Prepaid iff UPB==0 at the true (YYYYMM-ordered) last row."""
    print("Pass 0: global last appearance per loan (YYYYMM-ordered)...", flush=True)
    global_last = {}   # loan_id -> (last_ym, last_upb, rate)

    for fi, f in enumerate(files):
        fname = os.path.basename(f)
        print(f"  [{fi+1}/{len(files)}] {fname}", end=' ', flush=True)
        n_rows = 0
        for chunk in pd.read_csv(
                f, sep='|', header=None,
                usecols=[COL_LOAN, COL_MONTH, COL_RATE, COL_UPB],
                names=['loan_id', 'month', 'rate', 'upb'],
                chunksize=CHUNK, engine='c', dtype=str):
            chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
            chunk['rate']  = pd.to_numeric(chunk['rate'],  errors='coerce')
            chunk['upb']   = pd.to_numeric(chunk['upb'],   errors='coerce')
            chunk = chunk.dropna(subset=['loan_id', 'month', 'rate'])
            chunk['month'] = chunk['month'].astype(np.int64)
            chunk['ym']    = mmyyyy_to_yyyymm(chunk['month'].values)   # correct order
            n_rows += len(chunk)

            idx_max = chunk.groupby('loan_id')['ym'].idxmax()          # true last by YYYYMM
            last_rows = chunk.loc[idx_max].set_index('loan_id')
            for lid, row in last_rows.iterrows():
                ym = int(row['ym']); u = row['upb']; r = row['rate']
                if lid not in global_last or ym > global_last[lid][0]:
                    global_last[lid] = (ym, float(u) if not np.isnan(u) else np.nan, float(r))
        print(f"{n_rows:,} rows", flush=True)

    prepay_month = {}; rate_map = {}; n_prepaid = 0
    for lid, (last_ym, last_upb, rate) in global_last.items():
        rate_map[lid] = rate
        if not np.isnan(last_upb) and last_upb == 0.0:
            prepay_month[lid] = last_ym          # YYYYMM
            n_prepaid += 1
        else:
            prepay_month[lid] = -1
    print(f"\nPass 0 done: {len(global_last):,} unique loans, "
          f"{n_prepaid:,} prepaid ({100*n_prepaid/max(len(global_last),1):.2f}%)", flush=True)
    return prepay_month, rate_map


def pass1_aggregate(files, prepay_month, rate_map, atrisk, prepays):
    """Count at-risk and new prepayments per (coupon_bucket, YYYYMM)."""
    print("\nPass 1: aggregating at-risk and prepayments...", flush=True)
    for fi, f in enumerate(files):
        fname = os.path.basename(f)
        print(f"  [{fi+1}/{len(files)}] {fname}", flush=True)
        for chunk in pd.read_csv(
                f, sep='|', header=None,
                usecols=[COL_LOAN, COL_MONTH],
                names=['loan_id', 'month'],
                chunksize=CHUNK, engine='c', dtype=str):
            chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
            chunk = chunk.dropna(subset=['loan_id', 'month'])
            chunk['month'] = chunk['month'].astype(np.int64)
            chunk['ym']   = mmyyyy_to_yyyymm(chunk['month'].values)    # YYYYMM

            chunk['rate'] = chunk['loan_id'].map(rate_map)
            chunk = chunk.dropna(subset=['rate'])
            chunk['cb']  = (np.round(chunk['rate'] * 2) / 2.0).astype(np.float32)
            chunk['pm']  = chunk['loan_id'].map(prepay_month).fillna(-1).astype(np.int64)

            mo = chunk['ym'].values        # compare in YYYYMM
            pm = chunk['pm'].values        # YYYYMM (or -1)
            cb = chunk['cb'].values

            ar = (pm == -1) | (mo <= pm)   # at risk through payoff month
            pp = (mo == pm) & (pm != -1)   # new prepayment this month

            if ar.any():
                ar_df = pd.DataFrame({'cb': cb[ar], 'm': mo[ar]})
                for (c, m), n in ar_df.groupby(['cb', 'm']).size().items():
                    atrisk[(float(c), int(m))] += int(n)
            if pp.any():
                pp_df = pd.DataFrame({'cb': cb[pp], 'm': mo[pp]})
                for (c, m), n in pp_df.groupby(['cb', 'm']).size().items():
                    prepays[(float(c), int(m))] += int(n)


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    print(f"Found {len(files)} vintage files\n", flush=True)

    atrisk  = defaultdict(int)
    prepays = defaultdict(int)

    if os.path.exists(CKPT_P0):
        print(f"Pass 0: SKIPPED — loading checkpoint from {CKPT_P0}", flush=True)
        with open(CKPT_P0, "rb") as fh:
            prepay_month, rate_map = pickle.load(fh)
        print(f"  {len(rate_map):,} loans, {sum(v>0 for v in prepay_month.values()):,} prepaid", flush=True)
    else:
        prepay_month, rate_map = pass0_global_last(files)
        with open(CKPT_P0, "wb") as fh:
            pickle.dump((prepay_month, rate_map), fh)
        print(f"Pass 0 checkpoint saved: {CKPT_P0}", flush=True)
    pass1_aggregate(files, prepay_month, rate_map, atrisk, prepays)

    print("\nBuilding output...", flush=True)
    rows = []
    for (cb, ym) in sorted(set(atrisk.keys()) | set(prepays.keys())):
        n_at = atrisk.get((cb, ym), 0)
        n_pp = prepays.get((cb, ym), 0)
        if n_at == 0:
            continue
        smm = n_pp / n_at
        cpr = 1.0 - (1.0 - smm) ** 12
        rows.append(dict(
            coupon_bucket      = cb,
            implied_mbs_coupon = round(cb - GFEE, 2),
            yyyymm             = ym,
            n_atrisk           = n_at,
            n_prepay           = n_pp,
            smm                = round(smm, 8),
            cpr                = round(cpr, 8),
        ))

    out = pd.DataFrame(rows)
    out['date'] = out['yyyymm'].apply(parse_date_yyyymm)
    # keep a legacy MMYYYY 'month' column for backward compatibility
    out['month'] = out['yyyymm'].apply(lambda v: (int(v) % 100) * 10000 + int(v) // 100)
    out = out.sort_values(['coupon_bucket', 'yyyymm']).reset_index(drop=True)

    path = os.path.join(OUT, "realized_cpr_by_coupon_v6.csv")
    out.to_csv(path, index=False)
    print(f"Saved: {path} ({len(out)} rows)\n", flush=True)

    target = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    sub = out[out['implied_mbs_coupon'].isin(target)]
    print("=== Summary by coupon ===")
    print(sub.groupby('implied_mbs_coupon').agg(
        n_months=('yyyymm', 'nunique'), mean_cpr=('cpr', 'mean'),
        max_cpr=('cpr', 'max'), min_cpr=('cpr', 'min'),
        mean_atrisk=('n_atrisk', 'mean')).round(4))

    print("\n=== Recent-month recovery (v5 had these all-zero) ===")
    rec = sub[sub['date'] >= '2024-01-01']
    bym = rec.groupby('date')['cpr'].max()
    print(f"2024-01+ months: {len(bym)} | nonzero CPR: {(bym>0).sum()} | "
          f"last nonzero: {bym[bym>0].index.max() if (bym>0).any() else 'NONE'}")

    print("\n=== Boom check: coupon 3.5 peak should be 2020-2021, ~30-40% ===")
    c = sub[sub['implied_mbs_coupon'] == 3.5].sort_values('date')
    if len(c):
        peak = c.loc[c['cpr'].idxmax()]
        print(f"Peak CPR 3.5%: {peak['cpr']:.4f} in {peak['date'].strftime('%b %Y')}")


if __name__ == "__main__":
    main()
