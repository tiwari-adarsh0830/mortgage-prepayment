"""
Age-keyed realized CPR aggregation (UPB-weighted). v3 -- age from origination.

WHY v3 EXISTS. v2 read LOAN_AGE (0-based col 15) off each row. That field is
blank on the payoff row, so every prepayment landed in the missing-age bucket:
100% of upb_prepay in age_group == -1, zero in every real bucket. Numerators
separated from denominators and seasoned CPR was identically zero.

verify_byage_totals.py did NOT catch this, because summing over all age levels
restores the baseline exactly -- the partition was intact. The validation tested
the wrong invariant. v3's output must additionally satisfy: upb_prepay is
nonzero in the real age levels.

FIX. Age is derived from the origination date (0-based col 13, MMYYYY, constant
within a loan) rather than read from the row:

    age = (Y_cur - Y_orig) * 12 + (M_cur - M_orig)

This is well-defined on every row including the payoff row, and removes the
missing-age bucket except where the origination date itself is unparseable.

OFFSET NOTE. The file's LOAN_AGE does not equal months-since-origination. In
2013Q1, month 022013 with origination 122012 is 2 months elapsed but LOAN_AGE
reads 1. Rather than assume a convention, this script accumulates the
distribution of (derived_age - LOAN_AGE) on rows where LOAN_AGE parses, and
prints it. A one-month offset is immaterial at a 60-month boundary, but it is
measured rather than assumed.

Seasoning key, three levels:
    0   = age <  60 months
    60  = age 60-119 months
    120 = age >= 120 months
Levels 60 and 120 together are the advisor's "age > 5yr" cut.

Pass 0 is UNCHANGED and skipped when its checkpoint exists. Nothing Pass 0
produces depends on age, so the 2026-07-03 checkpoint remains valid.

BEFORE RUNNING: delete the stale Pass 1 checkpoint, or this will resume a v2
scan and silently reproduce the bug:
    rm -f outputs/realized_v6_upb_byage_pass1_checkpoint.pkl
"""
import pandas as pd
import pickle
import numpy as np
import glob
import os
from collections import Counter

BASE = "/scratch/at7095/mortgage_prepayment"
RAW  = os.path.join(BASE, "data/raw")
OUT  = os.path.join(BASE, "outputs")

COL_LOAN  = 1
COL_MONTH = 2
COL_RATE  = 7
COL_UPB   = 11
COL_ORIG  = 13          # origination date, MMYYYY, constant within a loan
COL_AGE   = 15          # LOAN_AGE -- cross-check only, blank at payoff
GFEE      = 0.50
CHUNK     = 2_000_000

CKPT_P0 = os.path.join(OUT, "realized_v6_upb_pass0_checkpoint.pkl")
CKPT_P1 = os.path.join(OUT, "realized_v6_upb_byage_pass1_checkpoint.pkl")
OUTFILE = os.path.join(OUT, "realized_cpr_by_coupon_v6_upb_byage.csv")

KEYS = ['cb', 'ym', 'age_grp']
VALS = ['upb_atrisk', 'upb_prepay', 'n_atrisk', 'n_prepay']


def mmyyyy_to_yyyymm(m):
    yyyy = m % 10000
    mm   = m // 10000
    return yyyy * 100 + mm


def parse_date_yyyymm(v):
    yyyy = int(v) // 100
    mm   = int(v) % 100
    return pd.Timestamp(year=yyyy, month=mm, day=1)


def months_between(ym_cur, ym_orig):
    """Both YYYYMM ints. Returns integer months elapsed; NaN-safe via float."""
    y1, m1 = ym_orig // 100, ym_orig % 100
    y2, m2 = ym_cur // 100, ym_cur % 100
    return (y2 - y1) * 12 + (m2 - m1)


def seasoning_group(age):
    """0 = <60mo, 60 = 60-119mo, 120 = 120+mo, -1 = missing/negative."""
    a = np.asarray(age, dtype=float)
    out = np.full(a.shape, -1, dtype=np.int32)
    ok = np.isfinite(a) & (a >= 0)
    v = a[ok]
    g = np.zeros(v.shape, dtype=np.int32)
    g[v >= 60]  = 60
    g[v >= 120] = 120
    out[ok] = g
    return out


def collapse(frames):
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=KEYS + VALS)
    return (pd.concat(frames, ignore_index=True)
              .groupby(KEYS, as_index=False)[VALS].sum())


def pass1(files, prepay_month, rate_map, payoff_balance):
    print("\nPass 1 (UPB, age from origination): aggregating...", flush=True)

    start_idx = 0
    totals = pd.DataFrame(columns=KEYS + VALS)
    offsets = Counter()

    if os.path.exists(CKPT_P1):
        with open(CKPT_P1, "rb") as fh:
            start_idx, totals, offsets = pickle.load(fh)
        offsets = Counter(offsets)
        print(f"  RESUMING from file index {start_idx} "
              f"({len(totals):,} cells so far)", flush=True)

    n_bad_orig = 0

    for fi, f in enumerate(files):
        if fi < start_idx:
            continue
        print(f"  [{fi+1}/{len(files)}] {os.path.basename(f)}", flush=True)
        frames = []

        for chunk in pd.read_csv(
                f, sep='|', header=None,
                usecols=[COL_LOAN, COL_MONTH, COL_UPB, COL_ORIG, COL_AGE],
                names=['loan_id', 'month', 'upb', 'orig', 'file_age'],
                chunksize=CHUNK, engine='c', dtype=str):

            chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
            chunk['upb']   = pd.to_numeric(chunk['upb'],   errors='coerce')
            chunk['orig']  = pd.to_numeric(chunk['orig'],  errors='coerce')
            chunk['file_age'] = pd.to_numeric(chunk['file_age'], errors='coerce')
            chunk = chunk.dropna(subset=['loan_id', 'month'])
            if chunk.empty:
                continue
            chunk['month'] = chunk['month'].astype(np.int64)
            chunk['ym']    = mmyyyy_to_yyyymm(chunk['month'].values)

            chunk['rate'] = chunk['loan_id'].map(rate_map)
            chunk = chunk.dropna(subset=['rate'])
            if chunk.empty:
                continue
            chunk['cb'] = (np.round(chunk['rate'] * 2) / 2.0).astype(np.float32)
            chunk['pm'] = chunk['loan_id'].map(prepay_month).fillna(-1).astype(np.int64)
            chunk['payoff_bal'] = chunk['loan_id'].map(payoff_balance)

            # --- age from origination, valid on every row incl. payoff -------
            ov = chunk['orig'].values
            good = np.isfinite(ov) & (ov > 0)
            n_bad_orig += int((~good).sum())
            age = np.full(len(chunk), np.nan)
            if good.any():
                orig_ym = mmyyyy_to_yyyymm(ov[good].astype(np.int64))
                age[good] = months_between(chunk['ym'].values[good], orig_ym)
            chunk['age_grp'] = seasoning_group(age)

            # --- cross-check against the file's LOAN_AGE ---------------------
            fa = chunk['file_age'].values
            both = np.isfinite(age) & np.isfinite(fa)
            if both.any():
                diff = (age[both] - fa[both]).astype(np.int64)
                for k, v in Counter(diff.tolist()).items():
                    offsets[int(k)] += int(v)

            mo  = chunk['ym'].values
            pm  = chunk['pm'].values
            upb = chunk['upb'].values
            payoff_bal = chunk['payoff_bal'].values

            is_payoff = (mo == pm) & (pm != -1)
            is_active = ((pm == -1) | (mo < pm))

            weight = np.where(is_payoff, payoff_bal, upb)
            valid  = ~np.isnan(weight) & (weight >= 0)

            chunk['w']   = weight
            chunk['_ar'] = ((is_active | is_payoff) & valid)
            chunk['_pp'] = (is_payoff & valid)

            sub = chunk.loc[chunk['_ar'] | chunk['_pp'],
                            KEYS + ['w', '_ar', '_pp']].copy()
            if sub.empty:
                continue

            sub['upb_atrisk'] = np.where(sub['_ar'], sub['w'], 0.0)
            sub['upb_prepay'] = np.where(sub['_pp'], sub['w'], 0.0)
            sub['n_atrisk']   = sub['_ar'].astype(np.int64)
            sub['n_prepay']   = sub['_pp'].astype(np.int64)

            frames.append(sub.groupby(KEYS, as_index=False)[VALS].sum())

        totals = collapse([totals, collapse(frames)])

        with open(CKPT_P1, "wb") as fh:
            pickle.dump((fi + 1, totals, dict(offsets)), fh)
        print(f"    [checkpoint saved at file {fi+1}/{len(files)}; "
              f"{len(totals):,} cells]", flush=True)

    if n_bad_orig:
        print(f"  NOTE: {n_bad_orig:,} rows had unparseable origination date",
              flush=True)
    return totals, offsets


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    print(f"Found {len(files)} vintage files\n", flush=True)

    if not os.path.exists(CKPT_P0):
        raise SystemExit(f"Pass 0 checkpoint not found at {CKPT_P0}.")
    print(f"Pass 0: SKIPPED -- loading checkpoint from {CKPT_P0}", flush=True)
    with open(CKPT_P0, "rb") as fh:
        prepay_month, rate_map, payoff_balance = pickle.load(fh)
    print(f"  {len(rate_map):,} loans in rate_map", flush=True)

    t, offsets = pass1(files, prepay_month, rate_map, payoff_balance)

    print("\n=== derived_age - LOAN_AGE (rows where both parse) ===", flush=True)
    tot = sum(offsets.values())
    for k in sorted(offsets, key=lambda z: -offsets[z])[:8]:
        print("  offset %+d : %s rows (%.2f%%)" %
              (k, f"{offsets[k]:,}", 100.0 * offsets[k] / max(tot, 1)))

    print("\nBuilding output...", flush=True)
    t = t[t['upb_atrisk'] > 0].copy()
    t['smm_upb'] = t['upb_prepay'] / t['upb_atrisk']
    t['cpr_upb'] = 1.0 - (1.0 - t['smm_upb']) ** 12
    n_at = t['n_atrisk'].replace(0, np.nan)
    t['smm_count'] = t['n_prepay'] / n_at
    t['cpr_count'] = 1.0 - (1.0 - t['smm_count']) ** 12

    t = t.rename(columns={'cb': 'coupon_bucket', 'ym': 'yyyymm',
                          'age_grp': 'age_group'})
    t['implied_mbs_coupon'] = (t['coupon_bucket'] - GFEE).round(2)
    t['date'] = t['yyyymm'].apply(parse_date_yyyymm)
    t = t.sort_values(['coupon_bucket', 'yyyymm', 'age_group']).reset_index(drop=True)

    cols = ['coupon_bucket', 'implied_mbs_coupon', 'yyyymm', 'age_group', 'date',
            'n_atrisk', 'n_prepay', 'upb_atrisk', 'upb_prepay',
            'smm_upb', 'cpr_upb', 'smm_count', 'cpr_count']
    t[cols].to_csv(OUTFILE, index=False)
    print(f"Saved: {OUTFILE} ({len(t)} rows)\n", flush=True)

    print("=== at-risk UPB share by seasoning level ===")
    tot_at = t['upb_atrisk'].sum()
    print((t.groupby('age_group')['upb_atrisk'].sum() / tot_at * 100).round(2).to_string())

    print("\n=== PREPAY UPB by seasoning level (v2's bug: all mass fell in -1) ===")
    print(t.groupby('age_group')['upb_prepay'].sum().to_string())
    real = t.loc[t['age_group'] >= 0, 'upb_prepay'].sum()
    allp = t['upb_prepay'].sum()
    print("\n  prepay mass in real age levels: %.2f%% of total" %
          (100.0 * real / max(allp, 1e-12)))
    print("  (v2 gave 0.00%. Anything near 100%% means the fix worked.)")


if __name__ == "__main__":
    main()
