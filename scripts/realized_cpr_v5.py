"""
Realized CPR by coupon bucket by month — v5 CORRECT FINAL.

Bug in v4: processed files independently, so a loan spanning multiple
vintage files had its "last appearance" computed per-file rather than
globally. E.g. loan in 2018Q4.csv shows UPB=0 (reporting lag) in Dec 2018
as its last row in THAT file, but continues in 2019Q1.csv with real UPB.
v4 incorrectly flagged Dec 2018 as the prepayment month.

Fix: THREE passes across ALL files:
  Pass 0 (global): find each loan's GLOBAL last month and UPB across all files
  Pass 1 (per-file): identify prepayment month = global last month IF UPB=0 there
  Pass 2 (per-file): count at-risk and new prepayments

This ensures a loan that spans multiple files is only flagged as prepaid
when its absolute final appearance has UPB=0.
"""

import pandas as pd
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


def parse_date(m):
    s = str(int(m))
    if len(s) == 5:
        return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    elif len(s) == 6:
        return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT


def pass0_global_last(files):
    """
    Scan ALL files to find each loan's global last month and its UPB.
    Returns: dict {loan_id -> (global_last_month_int, last_upb, note_rate)}
    """
    print("Pass 0: finding global last appearance per loan across all files...", flush=True)
    global_last = {}   # loan_id -> (last_month, last_upb, rate)

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
            n_rows += len(chunk)

            # Vectorized: find max month per loan in this chunk
            # and track the UPB and rate at that max month
            chunk_valid = chunk.copy()
            idx_max = chunk_valid.groupby('loan_id')['month'].idxmax()
            last_rows = chunk_valid.loc[idx_max].set_index('loan_id')

            for lid, row in last_rows.iterrows():
                m = int(row['month'])
                u = row['upb']
                r = row['rate']
                if lid not in global_last or m > global_last[lid][0]:
                    global_last[lid] = (m, float(u) if not np.isnan(u) else np.nan, float(r))

        print(f"{n_rows:,} rows", flush=True)

    # Determine prepay_month: global last month where UPB=0; else -1 (active/censored)
    prepay_month = {}
    rate_map     = {}
    n_prepaid = 0
    for lid, (last_m, last_upb, rate) in global_last.items():
        rate_map[lid] = rate
        if not np.isnan(last_upb) and last_upb == 0.0:
            prepay_month[lid] = last_m
            n_prepaid += 1
        else:
            prepay_month[lid] = -1

    print(f"\nPass 0 done: {len(global_last):,} unique loans, "
          f"{n_prepaid:,} prepaid ({100*n_prepaid/len(global_last):.2f}%)", flush=True)
    return prepay_month, rate_map


def pass1_aggregate(files, prepay_month, rate_map, atrisk, prepays):
    """
    Scan ALL files to count at-risk and new prepayments per (cb, month).
    Uses global prepay_month from Pass 0.
    """
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

            # Map rate and prepay month
            chunk['rate'] = chunk['loan_id'].map(rate_map)
            chunk = chunk.dropna(subset=['rate'])
            chunk['cb']  = (np.round(chunk['rate'] * 2) / 2.0).astype(np.float32)
            chunk['pm']  = chunk['loan_id'].map(prepay_month).fillna(-1).astype(np.int64)

            mo = chunk['month'].values
            pm = chunk['pm'].values
            cb = chunk['cb'].values

            # At-risk: present this month AND not yet prepaid before this month
            ar = (pm == -1) | (mo <= pm)
            # New prepayment: this is exactly the prepay month
            pp = (mo == pm) & (pm != -1)

            if ar.any():
                ar_df = pd.DataFrame({'cb': cb[ar], 'm': mo[ar]})
                for (c, m), n in ar_df.groupby(['cb','m']).size().items():
                    atrisk[(float(c), int(m))] += int(n)

            if pp.any():
                pp_df = pd.DataFrame({'cb': cb[pp], 'm': mo[pp]})
                for (c, m), n in pp_df.groupby(['cb','m']).size().items():
                    prepays[(float(c), int(m))] += int(n)


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    print(f"Found {len(files)} vintage files\n", flush=True)

    # Pass 0: global last appearance
    # NOTE: itertuples is slow for large files — use chunked groupby approach
    from collections import defaultdict
    atrisk  = defaultdict(int)
    prepays = defaultdict(int)

    prepay_month, rate_map = pass0_global_last(files)

    # Pass 1: aggregate
    pass1_aggregate(files, prepay_month, rate_map, atrisk, prepays)

    # Build output
    print("\nBuilding output...", flush=True)
    rows = []
    for (cb, month) in sorted(set(atrisk.keys()) | set(prepays.keys())):
        n_at = atrisk.get((cb, month), 0)
        n_pp = prepays.get((cb, month), 0)
        if n_at == 0:
            continue
        smm = n_pp / n_at
        cpr = 1.0 - (1.0 - smm) ** 12
        rows.append(dict(
            coupon_bucket      = cb,
            implied_mbs_coupon = round(cb - GFEE, 2),
            month              = month,
            n_atrisk           = n_at,
            n_prepay           = n_pp,
            smm                = round(smm, 8),
            cpr                = round(cpr, 8),
        ))

    out = pd.DataFrame(rows)
    out['date'] = out['month'].apply(parse_date)
    out = out.sort_values(['coupon_bucket', 'month']).reset_index(drop=True)

    path = os.path.join(OUT, "realized_cpr_by_coupon_v5.csv")
    out.to_csv(path, index=False)
    print(f"Saved: {path} ({len(out)} rows)\n", flush=True)

    # Sanity checks
    target = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    sub = out[out['implied_mbs_coupon'].isin(target)]

    print("=== Summary by coupon ===")
    print(sub.groupby('implied_mbs_coupon').agg(
        n_months    = ('month',    'nunique'),
        mean_cpr    = ('cpr',      'mean'),
        max_cpr     = ('cpr',      'max'),
        min_cpr     = ('cpr',      'min'),
        mean_atrisk = ('n_atrisk', 'mean'),
    ).round(4))

    print("\n=== CPR time series: coupon 4.0 (was most spiked) ===")
    c40 = sub[sub['implied_mbs_coupon'] == 4.0].sort_values('date')
    print(c40[['date','n_atrisk','n_prepay','cpr']].to_string(index=False))

    print("\n=== Validation ===")
    if len(c40) > 0:
        peak = c40.loc[c40['cpr'].idxmax()]
        print(f"Peak CPR for 4.0%: {peak['cpr']:.4f} in {peak['date'].strftime('%b %Y')}")
        print("Expected: peak in 2020-2021 (refi boom), NOT 2018-2019")
        y2018_max = c40[c40['date'].dt.year==2018]['cpr'].max()
        print(f"Max CPR in 2018: {y2018_max:.4f} (should be low, <10%)")


if __name__ == "__main__":
    main()
