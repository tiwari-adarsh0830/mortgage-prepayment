"""
Panel 1 — Realized CPR by REFI INCENTIVE bin, for two origination cohorts,
restricted to each loan's first 33 months (the model's training window).

Purpose: show whether the refi-incentive -> prepayment S-curve is present in
the data each model trained on. Reuses the validated v5 prepay detection
(global last appearance; UPB=0 at global last month => prepaid) so the
prepayment definition is identical to what the advisor already accepted.

refi_incentive (per loan-month) = original_note_rate - PMMS_30yr[that month]
  positive  => borrower rate ABOVE market => incentive to refinance => high CPR
Window: loan_age <= 33 (matches sequence construction MAX_SEQ_LEN=33).

Two cohorts (by origination file):
  COHORT_PRE2020 : 2013Q1..2019Q4   (extended-model training set)
  COHORT_BOOM    : 2018Q1..2023Q1   (21-vintage model training set)

Output: outputs/realized_cpr_by_refi_v1.csv  with a 'cohort' column.
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse
from collections import defaultdict

BASE = "/scratch/at7095/mortgage_prepayment"
RAW  = os.path.join(BASE, "data/raw")
OUT  = os.path.join(BASE, "outputs")
PMMS = os.path.join(BASE, "data/pmms_monthly.csv")

COL_LOAN  = 1     # 0-indexed
COL_MONTH = 2
COL_RATE  = 7
COL_UPB   = 11
COL_AGE   = 15    # loan_age, field 16 -> 0-indexed 15
CHUNK     = 2_000_000
MAX_AGE   = 33   # default; overridable via --max-age

# Refi-incentive bins (percentage points). Edges chosen to span the
# diagnostic sweep (-2 .. +3) used in diag_raw_hazard.py.
BIN_EDGES = np.array([-np.inf, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf])
BIN_LBL   = ['<-1.5','-1.5..-1','-1..-0.5','-0.5..0','0..0.5',
             '0.5..1','1..1.5','1.5..2','>2']

COHORTS = {
    'pre2020': [f'{y}Q{q}' for y in range(2013, 2020) for q in range(1, 5)],
    'boom':    [f'{y}Q{q}' for y in range(2018, 2023) for q in range(1, 5)] + ['2023Q1'],
}


def load_pmms():
    df = pd.read_csv(PMMS)
    # key: reporting_period (MYYYY int) -> rate_30yr
    return dict(zip(df['reporting_period'].astype(np.int64), df['rate_30yr'].astype(float)))


def pass0_global_last(files):
    """Global last appearance per loan; prepay_month = global last month if UPB==0 there."""
    print("  Pass 0: global last appearance per loan...", flush=True)
    global_last = {}
    for fi, f in enumerate(files):
        print(f"    [{fi+1}/{len(files)}] {os.path.basename(f)}", end=' ', flush=True)
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
            idx_max = chunk.groupby('loan_id')['month'].idxmax()
            last_rows = chunk.loc[idx_max].set_index('loan_id')
            for lid, row in last_rows.iterrows():
                m = int(row['month']); u = row['upb']; r = row['rate']
                if lid not in global_last or m > global_last[lid][0]:
                    global_last[lid] = (m, float(u) if not np.isnan(u) else np.nan, float(r))
        print(f"{n_rows:,} rows", flush=True)

    prepay_month, rate_map = {}, {}
    n_prepaid = 0
    for lid, (last_m, last_upb, rate) in global_last.items():
        rate_map[lid] = rate
        if not np.isnan(last_upb) and last_upb == 0.0:
            prepay_month[lid] = last_m; n_prepaid += 1
        else:
            prepay_month[lid] = -1
    print(f"  Pass 0 done: {len(global_last):,} loans, {n_prepaid:,} prepaid "
          f"({100*n_prepaid/max(len(global_last),1):.2f}%)", flush=True)
    return prepay_month, rate_map


def pass1_aggregate(files, prepay_month, rate_map, pmms, atrisk, prepays):
    """Bin at-risk and new-prepay counts by refi-incentive bin, age<=33 only."""
    print("  Pass 1: aggregating by refi-incentive bin (age<=33)...", flush=True)
    for fi, f in enumerate(files):
        print(f"    [{fi+1}/{len(files)}] {os.path.basename(f)}", flush=True)
        for chunk in pd.read_csv(
                f, sep='|', header=None,
                usecols=[COL_LOAN, COL_MONTH, COL_AGE],
                names=['loan_id', 'month', 'age'],
                chunksize=CHUNK, engine='c', dtype=str):
            chunk['month'] = pd.to_numeric(chunk['month'], errors='coerce')
            chunk['age']   = pd.to_numeric(chunk['age'],   errors='coerce')
            chunk = chunk.dropna(subset=['loan_id', 'month', 'age'])
            chunk['month'] = chunk['month'].astype(np.int64)
            chunk['age']   = chunk['age'].astype(np.int64)

            # Training window: first 33 months only
            chunk = chunk[chunk['age'] <= MAX_AGE]
            if chunk.empty:
                continue

            chunk['rate'] = chunk['loan_id'].map(rate_map)
            chunk = chunk.dropna(subset=['rate'])
            chunk['pmms'] = chunk['month'].map(pmms)
            chunk = chunk.dropna(subset=['pmms'])
            chunk['refi'] = chunk['rate'] - chunk['pmms']
            chunk['bin']  = pd.cut(chunk['refi'], bins=BIN_EDGES, labels=BIN_LBL)
            chunk = chunk.dropna(subset=['bin'])

            chunk['pm'] = chunk['loan_id'].map(prepay_month).fillna(-1).astype(np.int64)
            mo = chunk['month'].values
            pm = chunk['pm'].values

            ar = (pm == -1) | (mo <= pm)              # at risk this month
            pp = (mo == pm) & (pm != -1)              # new prepay this month

            bins = chunk['bin'].astype(str).values
            if ar.any():
                for bcat, n in pd.Series(bins[ar]).value_counts().items():
                    atrisk[str(bcat)] += int(n)
            if pp.any():
                for bcat, n in pd.Series(bins[pp]).value_counts().items():
                    prepays[str(bcat)] += int(n)


def run_cohort(name, vintages, pmms):
    files = [os.path.join(RAW, f'{v}.csv') for v in vintages
             if os.path.exists(os.path.join(RAW, f'{v}.csv'))]
    print(f"\n=== Cohort '{name}': {len(files)} files ===", flush=True)
    atrisk, prepays = defaultdict(int), defaultdict(int)
    prepay_month, rate_map = pass0_global_last(files)
    pass1_aggregate(files, prepay_month, rate_map, pmms, atrisk, prepays)

    rows = []
    for lbl in BIN_LBL:
        n_at = atrisk.get(lbl, 0); n_pp = prepays.get(lbl, 0)
        if n_at == 0:
            continue
        smm = n_pp / n_at
        cpr = 1.0 - (1.0 - smm) ** 12
        rows.append(dict(cohort=name, refi_bin=lbl,
                         n_atrisk=n_at, n_prepay=n_pp,
                         smm=round(smm, 8), cpr=round(cpr, 6)))
    return rows


def main():
    global MAX_AGE
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-age', type=int, default=33,
                    help='cap on loan_age (months) for the at-risk/prepay window')
    ap.add_argument('--cohorts', type=str, default='pre2020,boom',
                    help='comma-separated subset of cohort names to run')
    ap.add_argument('--tag', type=str, default='',
                    help='suffix appended to the output CSV filename')
    args = ap.parse_args()
    MAX_AGE = args.max_age
    selected = [c.strip() for c in args.cohorts.split(',') if c.strip()]
    cohorts_to_run = {k: v for k, v in COHORTS.items() if k in selected}
    print(f"Running cohorts={list(cohorts_to_run)}  MAX_AGE={MAX_AGE}", flush=True)

    pmms = load_pmms()
    all_rows = []
    for name, vintages in cohorts_to_run.items():
        all_rows += run_cohort(name, vintages, pmms)

    out = pd.DataFrame(all_rows)
    # order refi_bin as categorical for clean output
    out['refi_bin'] = pd.Categorical(out['refi_bin'], categories=BIN_LBL, ordered=True)
    out = out.sort_values(['cohort', 'refi_bin']).reset_index(drop=True)
    path = os.path.join(OUT, f"realized_cpr_by_refi_v1{args.tag}.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}", flush=True)

    print("\n=== Realized CPR by refi-incentive bin (age<=33) ===")
    for name in COHORTS:
        sub = out[out['cohort'] == name]
        print(f"\nCohort: {name}")
        print(sub[['refi_bin','n_atrisk','n_prepay','cpr']].to_string(index=False))


if __name__ == "__main__":
    main()
