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
CKPT_P0   = os.path.join(OUT, "realized_v6_upb_pass0_checkpoint.pkl")


def mmyyyy_to_yyyymm(m):
    yyyy = m % 10000
    mm   = m // 10000
    return yyyy * 100 + mm


def parse_date_yyyymm(v):
    yyyy = int(v) // 100
    mm   = int(v) % 100
    return pd.Timestamp(year=yyyy, month=mm, day=1)


def _merge_top2(existing, candidates):
    pool = {ym: upb for ym, upb in existing}
    for ym, upb in candidates:
        if ym not in pool:
            pool[ym] = upb
    top = sorted(pool.items(), key=lambda x: -x[0])[:2]
    return top


def pass0_global_top2(files):
    print("Pass 0 (UPB): global top-2 appearances per loan (YYYYMM-ordered)...", flush=True)

    ckpt_progress = CKPT_P0 + ".partial"
    start_idx = 0
    global_top2 = {}
    rate_map    = {}
    if os.path.exists(ckpt_progress):
        with open(ckpt_progress, "rb") as fh:
            start_idx, global_top2, rate_map = pickle.load(fh)
        print(f"  RESUMING from file index {start_idx} "
              f"({len(rate_map):,} loans tracked so far)", flush=True)

    for fi, f in enumerate(files):
        if fi < start_idx:
            continue
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
            chunk['ym']    = mmyyyy_to_yyyymm(chunk['month'].values)
            n_rows += len(chunk)

            for lid, r in chunk.drop_duplicates('loan_id').set_index('loan_id')['rate'].items():
                if lid not in rate_map:
                    rate_map[lid] = float(r)

            chunk_sorted = chunk.sort_values(['loan_id', 'ym'], ascending=[True, False])
            chunk_top2 = chunk_sorted.groupby('loan_id').head(2)
            for lid, grp in chunk_top2.groupby('loan_id'):
                cand = [(int(row['ym']), float(row['upb']) if not np.isnan(row['upb']) else np.nan)
                        for _, row in grp.iterrows()]
                existing = global_top2.get(lid, [])
                global_top2[lid] = _merge_top2(existing, cand)
        print(f"{n_rows:,} rows", flush=True)

        # checkpoint every 5 files so a timeout doesn't lose all progress
        if (fi + 1) % 5 == 0 or (fi + 1) == len(files):
            with open(ckpt_progress, "wb") as fh:
                pickle.dump((fi + 1, global_top2, rate_map), fh)
            print(f"  [checkpoint saved at file {fi+1}/{len(files)}]", flush=True)

    prepay_month  = {}
    payoff_balance = {}
    n_prepaid = 0
    for lid, top2 in global_top2.items():
        last_ym, last_upb = top2[0]
        if not np.isnan(last_upb) and last_upb == 0.0:
            prepay_month[lid] = last_ym
            n_prepaid += 1
            if len(top2) >= 2:
                payoff_balance[lid] = top2[1][1]
            else:
                payoff_balance[lid] = np.nan
        else:
            prepay_month[lid] = -1

    n_no_prior = sum(1 for lid in prepay_month if prepay_month[lid] != -1
                      and np.isnan(payoff_balance.get(lid, np.nan)))
    print(f"\nPass 0 done: {len(global_top2):,} unique loans, {n_prepaid:,} prepaid "
          f"({100*n_prepaid/max(len(global_top2),1):.2f}%)", flush=True)
    if n_no_prior:
        print(f"  WARNING: {n_no_prior:,} prepaid loans have no prior-month row "
              f"(only ever observed at payoff) -- excluded from UPB-weighted panel, "
              f"still included in v6's loan-count panel.", flush=True)
    return prepay_month, rate_map, payoff_balance


def pass1_aggregate_upb(files, prepay_month, rate_map, payoff_balance,
                         atrisk_upb, prepay_upb, atrisk_n, prepay_n):
    print("\nPass 1 (UPB): aggregating balance-weighted at-risk and prepayments...", flush=True)
    for fi, f in enumerate(files):
        fname = os.path.basename(f)
        print(f"  [{fi+1}/{len(files)}] {fname}", flush=True)
        for chunk in pd.read_csv(
                f, sep='|', header=None,
                usecols=[COL_LOAN, COL_MONTH, COL_UPB],
                names=['loan_id', 'month', 'upb'],
                chunksize=CHUNK, engine='c', dtype=str):
            chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
            chunk['upb']   = pd.to_numeric(chunk['upb'],   errors='coerce')
            chunk = chunk.dropna(subset=['loan_id', 'month'])
            chunk['month'] = chunk['month'].astype(np.int64)
            chunk['ym']    = mmyyyy_to_yyyymm(chunk['month'].values)

            chunk['rate'] = chunk['loan_id'].map(rate_map)
            chunk = chunk.dropna(subset=['rate'])
            chunk['cb'] = (np.round(chunk['rate'] * 2) / 2.0).astype(np.float32)
            chunk['pm'] = chunk['loan_id'].map(prepay_month).fillna(-1).astype(np.int64)
            chunk['payoff_bal'] = chunk['loan_id'].map(payoff_balance)

            mo = chunk['ym'].values
            pm = chunk['pm'].values
            cb = chunk['cb'].values
            upb = chunk['upb'].values
            payoff_bal = chunk['payoff_bal'].values

            is_payoff_month = (mo == pm) & (pm != -1)
            is_active_month = ((pm == -1) | (mo < pm))

            weight = np.where(is_payoff_month, payoff_bal, upb)
            valid_weight = ~np.isnan(weight) & (weight >= 0)

            ar_mask = (is_active_month | is_payoff_month) & valid_weight
            if ar_mask.any():
                df = pd.DataFrame({'cb': cb[ar_mask], 'm': mo[ar_mask], 'w': weight[ar_mask]})
                for (c, m), w in df.groupby(['cb', 'm'])['w'].sum().items():
                    atrisk_upb[(float(c), int(m))] += float(w)
                for (c, m), n in df.groupby(['cb', 'm']).size().items():
                    atrisk_n[(float(c), int(m))] += int(n)

            pp_mask = is_payoff_month & valid_weight
            if pp_mask.any():
                df = pd.DataFrame({'cb': cb[pp_mask], 'm': mo[pp_mask], 'w': weight[pp_mask]})
                for (c, m), w in df.groupby(['cb', 'm'])['w'].sum().items():
                    prepay_upb[(float(c), int(m))] += float(w)
                for (c, m), n in df.groupby(['cb', 'm']).size().items():
                    prepay_n[(float(c), int(m))] += int(n)


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    print(f"Found {len(files)} vintage files\n", flush=True)

    atrisk_upb, prepay_upb = defaultdict(float), defaultdict(float)
    atrisk_n, prepay_n     = defaultdict(int), defaultdict(int)

    if os.path.exists(CKPT_P0):
        print(f"Pass 0: SKIPPED -- loading checkpoint from {CKPT_P0}", flush=True)
        with open(CKPT_P0, "rb") as fh:
            prepay_month, rate_map, payoff_balance = pickle.load(fh)
    else:
        prepay_month, rate_map, payoff_balance = pass0_global_top2(files)
        with open(CKPT_P0, "wb") as fh:
            pickle.dump((prepay_month, rate_map, payoff_balance), fh)
        print(f"Pass 0 checkpoint saved: {CKPT_P0}", flush=True)

    pass1_aggregate_upb(files, prepay_month, rate_map, payoff_balance,
                        atrisk_upb, prepay_upb, atrisk_n, prepay_n)

    print("\nBuilding output...", flush=True)
    rows = []
    for (cb, ym) in sorted(set(atrisk_upb.keys()) | set(prepay_upb.keys())):
        upb_at = atrisk_upb.get((cb, ym), 0.0)
        upb_pp = prepay_upb.get((cb, ym), 0.0)
        n_at   = atrisk_n.get((cb, ym), 0)
        n_pp   = prepay_n.get((cb, ym), 0)
        if upb_at <= 0:
            continue
        smm_upb = upb_pp / upb_at
        cpr_upb = 1.0 - (1.0 - smm_upb) ** 12
        smm_n   = n_pp / n_at if n_at > 0 else np.nan
        cpr_n   = 1.0 - (1.0 - smm_n) ** 12 if n_at > 0 else np.nan
        rows.append(dict(
            coupon_bucket=cb, implied_mbs_coupon=round(cb - GFEE, 2), yyyymm=ym,
            n_atrisk=n_at, n_prepay=n_pp,
            upb_atrisk=round(upb_at, 2), upb_prepay=round(upb_pp, 2),
            smm_upb=round(smm_upb, 8), cpr_upb=round(cpr_upb, 8),
            smm_count=round(smm_n, 8) if n_at > 0 else np.nan,
            cpr_count=round(cpr_n, 8) if n_at > 0 else np.nan,
        ))

    out = pd.DataFrame(rows)
    out['date'] = out['yyyymm'].apply(parse_date_yyyymm)
    out = out.sort_values(['coupon_bucket', 'yyyymm']).reset_index(drop=True)

    path = os.path.join(OUT, "realized_cpr_by_coupon_v6_upb.csv")
    out.to_csv(path, index=False)
    print(f"Saved: {path} ({len(out)} rows)\n", flush=True)

    target = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    sub = out[out['implied_mbs_coupon'].isin(target)]
    print("=== UPB-weighted vs loan-count CPR, by coupon (sanity check) ===")
    print(sub.groupby('implied_mbs_coupon').agg(
        mean_cpr_upb=('cpr_upb', 'mean'),
        mean_cpr_count=('cpr_count', 'mean'),
        n_months=('yyyymm', 'nunique')).round(4))


if __name__ == "__main__":
    main()
