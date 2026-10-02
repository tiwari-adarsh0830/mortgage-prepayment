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
COL_TERM  = 12   # original_loan_term, awk $13
COL_MOD   = 41   # modification_flag, awk $42 (HARDCODED, no name list here)
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


def pass0_global_top2(files, term_filter=360):
    """30-YEAR FILTER + POST-MOD (advisor's Sep 29 decision) -- same
    approach as realized_cpr_v6.py's pass0_global_last: a loan with ANY row
    whose original_loan_term != term_filter -- including blank/unparseable
    (NaN) -- is dropped from rate_map entirely (loan-level, not row-level;
    fixed 2026-10-01, see realized_cpr_v6.py's docstring for why the prior
    first-observed-term check was insufficient); an ever-modified loan's
    prepay_month is forced to -1 (censored, unconditional sticky design
    choice -- does NOT require modification_flag to be monotone; that
    assumption was found FALSE for 2002Q1, see scripts/diag/
    verify_term_mod_filters.py's decisive decode, 2026-10-01).
    See realized_cpr_v6.py for the full rationale -- not repeated here.
    """
    print("Pass 0 (UPB): global top-2 appearances per loan (YYYYMM-ordered)...", flush=True)

    ckpt_progress = CKPT_P0 + ".partial"
    start_idx = 0
    global_top2 = {}
    rate_map    = {}
    bad_term_loans = set()  # loan_id -> has ANY row with term != term_filter (incl. blank/NaN)
    first_mod_ym = {}
    if os.path.exists(ckpt_progress):
        with open(ckpt_progress, "rb") as fh:
            start_idx, global_top2, rate_map, bad_term_loans, first_mod_ym = pickle.load(fh)
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
                usecols=[COL_LOAN, COL_MONTH, COL_RATE, COL_UPB, COL_TERM, COL_MOD],
                names=['loan_id', 'month', 'rate', 'upb', 'term', 'mod'],
                chunksize=CHUNK, engine='c', dtype=str):
            chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
            chunk['rate']  = pd.to_numeric(chunk['rate'],  errors='coerce')
            chunk['upb']   = pd.to_numeric(chunk['upb'],   errors='coerce')
            chunk['term']  = pd.to_numeric(chunk['term'],  errors='coerce')
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

            # ANY row with term != term_filter disqualifies the loan --
            # including blank/unparseable term (NaN != x is True elementwise)
            # -- not just a dropna'd subset. Loan-level, not row-level.
            if term_filter is not None:
                bad_rows = chunk.loc[chunk['term'] != term_filter]
                bad_term_loans.update(bad_rows['loan_id'].tolist())

            y_rows = chunk.loc[chunk['mod'] == 'Y']
            if not y_rows.empty:
                y_min = y_rows.groupby('loan_id')['ym'].min()
                for lid, ym in y_min.items():
                    ym = int(ym)
                    if lid not in first_mod_ym or ym < first_mod_ym[lid]:
                        first_mod_ym[lid] = ym
        print(f"{n_rows:,} rows", flush=True)

        # checkpoint every 5 files so a timeout doesn't lose all progress
        if (fi + 1) % 5 == 0 or (fi + 1) == len(files):
            with open(ckpt_progress, "wb") as fh:
                pickle.dump((fi + 1, global_top2, rate_map, bad_term_loans, first_mod_ym), fh)
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

    if term_filter is not None:
        n_before = len(rate_map)
        n_dropped = len(bad_term_loans & rate_map.keys())
        for lid in bad_term_loans:
            rate_map.pop(lid, None)
            prepay_month.pop(lid, None)
            payoff_balance.pop(lid, None)
        print(f"  term filter (=={term_filter}, loan-level): dropped {n_dropped:,} "
              f"of {n_before:,} loans", flush=True)

    n_censored = 0
    for lid in first_mod_ym:
        if prepay_month.get(lid, -1) != -1:
            prepay_month[lid] = -1
            n_censored += 1
    print(f"  post-mod: {len(first_mod_ym):,} ever-modified loans, "
          f"{n_censored:,} had their terminal payoff censored", flush=True)

    return prepay_month, rate_map, payoff_balance, first_mod_ym


def pass1_aggregate_upb(files, prepay_month, rate_map, payoff_balance, first_mod_ym,
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
            chunk['fm'] = chunk['loan_id'].map(first_mod_ym).fillna(np.inf)

            mo = chunk['ym'].values
            pm = chunk['pm'].values
            cb = chunk['cb'].values
            upb = chunk['upb'].values
            payoff_bal = chunk['payoff_bal'].values
            fm = chunk['fm'].values

            is_payoff_month = (mo == pm) & (pm != -1)
            # `& (mo < fm)`: an ever-modified loan is forced to pm==-1 in
            # Pass 0, so without this it would stay at-risk forever,
            # including every post-mod month.
            is_active_month = ((pm == -1) | (mo < pm)) & (mo < fm)

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

    # NOTE: CKPT_P0's pickle format changed from a 3-tuple to a 4-tuple
    # (added first_mod_ym) when the 30-year filter / post-mod drop landed.
    # A pre-existing checkpoint from before that change will fail to
    # unpack here -- delete it and re-run Pass 0, do not catch/ignore.
    # realized_cpr_v6_upb_byage.py's main() reads this SAME checkpoint file
    # and must be kept in sync with this format.
    if os.path.exists(CKPT_P0):
        print(f"Pass 0: SKIPPED -- loading checkpoint from {CKPT_P0}", flush=True)
        with open(CKPT_P0, "rb") as fh:
            prepay_month, rate_map, payoff_balance, first_mod_ym = pickle.load(fh)
    else:
        prepay_month, rate_map, payoff_balance, first_mod_ym = pass0_global_top2(files)
        with open(CKPT_P0, "wb") as fh:
            pickle.dump((prepay_month, rate_map, payoff_balance, first_mod_ym), fh)
        print(f"Pass 0 checkpoint saved: {CKPT_P0}", flush=True)

    pass1_aggregate_upb(files, prepay_month, rate_map, payoff_balance, first_mod_ym,
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
