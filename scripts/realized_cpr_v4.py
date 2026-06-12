"""
Realized CPR by coupon bucket by month — v4 FINAL.

Method: UPB-based prepayment detection (no ZBC column needed).
  - A loan prepays in month T if UPB==0 in month T (its last appearance)
  - A loan is right-censored if its last UPB > 0 (still active at panel end)
  - At-risk in month T = loans present in that month's data
  - New prepayment in month T = loans with UPB=0 in month T

Two-pass per file (handles 20GB files via chunked reading):
  Pass 1: find last month and last UPB per loan → determine prepay_month
  Pass 2: count at-risk and new prepayments per (coupon_bucket, month)

Columns (0-indexed in file, with leading pipe so Field N = col N-1):
  col1  = loan_id          (Field 2)
  col2  = month            (Field 3, MMYYYY)
  col7  = original rate    (Field 8)
  col11 = current UPB      (Field 12)
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

GFEE  = 0.50
CHUNK = 2_000_000


def parse_date(m):
    s = str(int(m))
    if len(s) == 5:
        return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    elif len(s) == 6:
        return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT


def pass1_find_prepay(f):
    """
    Chunked pass: find last month and last UPB per loan.
    Returns dict: loan_id -> prepay_month (or -1 if right-censored)
    """
    # Track: loan_id -> (last_month, last_upb, first_rate)
    loan_last  = {}   # loan_id -> (last_month, last_upb)
    loan_rate  = {}   # loan_id -> original_interest_rate

    for chunk in pd.read_csv(
            f, sep='|', header=None,
            usecols=[COL_LOAN, COL_MONTH, COL_RATE, COL_UPB],
            names=['lid', 'month', 'rate', 'upb'],
            chunksize=CHUNK, engine='c', dtype=str):

        chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
        chunk['rate']  = pd.to_numeric(chunk['rate'],  errors='coerce')
        chunk['upb']   = pd.to_numeric(chunk['upb'],   errors='coerce')
        chunk = chunk.dropna(subset=['lid', 'month', 'rate'])
        chunk['month'] = chunk['month'].astype(np.int64)

        # Vectorized: find max month per loan in this chunk
        # and track rate (first seen)
        chunk_valid = chunk.dropna(subset=['upb'])

        # Rate: first occurrence per loan
        first_rate = chunk.groupby('lid')['rate'].first()
        for lid, r in first_rate.items():
            if lid not in loan_rate:
                loan_rate[lid] = float(r)

        # Last month per loan in this chunk
        idx_last = chunk.groupby('lid')['month'].idxmax()
        last_rows = chunk.loc[idx_last][['lid','month','upb']].set_index('lid')
        for lid, row in last_rows.iterrows():
            m = int(row['month'])
            u = row['upb']
            if lid not in loan_last or m > loan_last[lid][0]:
                loan_last[lid] = (m, float(u) if not pd.isna(u) else np.nan)

    # Determine prepay month
    # prepay: last month has UPB == 0
    # right-censored: last month has UPB > 0 or UPB is nan
    prepay_month = {}
    for lid, (last_m, last_upb) in loan_last.items():
        if not np.isnan(last_upb) and last_upb == 0.0:
            prepay_month[lid] = last_m   # prepaid
        else:
            prepay_month[lid] = -1       # right-censored

    return prepay_month, loan_rate


def pass2_aggregate(f, prepay_month, loan_rate, atrisk, prepays):
    """
    Chunked pass: count at-risk and new prepayments per (cb, month).
    At-risk: loan present in month AND month <= prepay_month (or never prepaid)
    New prepay: month == prepay_month
    """
    for chunk in pd.read_csv(
            f, sep='|', header=None,
            usecols=[COL_LOAN, COL_MONTH],
            names=['lid', 'month'],
            chunksize=CHUNK, engine='c', dtype=str):

        chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
        chunk = chunk.dropna(subset=['lid', 'month'])
        chunk['month'] = chunk['month'].astype(np.int64)

        # Map rate and prepay month
        chunk['rate'] = chunk['lid'].map(loan_rate)
        chunk = chunk.dropna(subset=['rate'])
        chunk['cb']   = (np.round(chunk['rate'] * 2) / 2.0).astype(np.float32)
        chunk['pm']   = chunk['lid'].map(prepay_month).fillna(-1).astype(np.int64)

        mo = chunk['month'].values
        pm = chunk['pm'].values
        cb = chunk['cb'].values

        # At-risk: present this month AND not yet prepaid before this month
        ar = (pm == -1) | (mo <= pm)
        # New prepayment: this is exactly the prepay month
        pp = (mo == pm) & (pm != -1)

        # Aggregate
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
    print(f"Found {len(files)} files\n", flush=True)

    atrisk  = defaultdict(int)
    prepays = defaultdict(int)

    for fi, f in enumerate(files):
        fname = os.path.basename(f)
        print(f"[{fi+1}/{len(files)}] {fname}", flush=True)

        print("  Pass 1: finding prepayment months...", flush=True)
        prepay_month, loan_rate = pass1_find_prepay(f)
        n_prepaid = sum(1 for v in prepay_month.values() if v != -1)
        n_censored = sum(1 for v in prepay_month.values() if v == -1)
        print(f"  Pass 1 done: {n_prepaid:,} prepaid, {n_censored:,} right-censored", flush=True)

        print("  Pass 2: aggregating...", flush=True)
        pass2_aggregate(f, prepay_month, loan_rate, atrisk, prepays)
        print("  Pass 2 done", flush=True)

        del prepay_month, loan_rate

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

    path = os.path.join(OUT, "realized_cpr_by_coupon_v4.csv")
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

    print("\n=== CPR time series: coupon 3.5 ===")
    c35 = sub[sub['implied_mbs_coupon'] == 3.5].sort_values('date')
    print(c35[['date', 'n_atrisk', 'n_prepay', 'smm', 'cpr']].to_string(index=False))

    print("\n=== Validation ===")
    if len(c35) > 0:
        peak = c35.loc[c35['cpr'].idxmax()]
        print(f"Peak CPR for 3.5%: {peak['cpr']:.4f} in {peak['date'].strftime('%b %Y')}")
        print("Expected: peak mid-2020 to mid-2021 (refi boom)")
        post22 = c35[c35['date'] >= pd.Timestamp('2022-06-01')]
        if len(post22) > 0:
            low = post22.loc[post22['cpr'].idxmin()]
            print(f"Post-2022 trough: {low['cpr']:.4f} in {low['date'].strftime('%b %Y')}")
            print("Expected: much lower than peak (rate rise kills refi)")


if __name__ == "__main__":
    main()
