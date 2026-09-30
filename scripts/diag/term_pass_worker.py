"""term_pass_worker.py -- one raw vintage file -> one per-loan parquet.

Single pass over a Fannie vintage CSV producing a per-loan record for the
loan-term / modification / maturity analysis (advisor Sep 29 facts round).
NO pipeline data changes: writes only to an analysis output dir.

Per-loan fields captured (loan-level statics take first sighting; the rest are
reduced across the loan's rows within the file -- one loan lives in exactly one
vintage file, so per-file is complete):
  loan_id, term ($13), orig_upb ($10), note_rate ($8), orig_ym (YYYYMM from $14),
  first_modY_ym (min YYYYMM where mod_flag $42 == 'Y', else 0),
  zbc_code (first non-blank $44 by earliest YYYYMM), zbc_ym,
  last_ym (max YYYYMM $3), last_upb (current_actual_upb $12 at last_ym).

Field positions verified empirically on both data dirs; awk $N = usecols N-1.
MMYYYY -> YYYYMM conversion applied before every ordering/comparison
(standing lesson: the raw date field is non-monotonic as a plain int).

Usage: python term_pass_worker.py <path-to-csv> <out-parquet>
"""
import sys
import numpy as np
import pandas as pd

USECOLS = [1, 2, 7, 9, 11, 12, 13, 41, 43]   # 0-based; = awk $2 $3 $8 $10 $12 $13 $14 $42 $44
NAMES   = ['loan_id', 'mrp', 'note_rate', 'orig_upb', 'cur_upb',
           'term', 'orig_date', 'mod_flag', 'zbc']
CHUNK = 4_000_000


def mmyyyy_to_yyyymm(m):
    """MMYYYY int -> YYYYMM int. 122020 -> 202012 ; 62024 -> 202406."""
    yyyy = m % 10000
    mm = m // 10000
    return yyyy * 100 + mm


def main():
    csv_path, out_path = sys.argv[1], sys.argv[2]

    static_parts = []   # loan_id -> term, orig_upb, note_rate, orig_ym (first sighting)
    modY_parts = []     # loan_id -> min ym where mod=='Y'
    zbc_parts = []      # loan_id, zbc_ym, zbc_code (non-blank rows)
    last_parts = []     # loan_id, ym, cur_upb (all rows; reduced to max ym at end)

    for chunk in pd.read_csv(csv_path, sep='|', header=None, usecols=USECOLS,
                             names=NAMES, chunksize=CHUNK, engine='c', dtype=str):
        chunk['loan_id'] = pd.to_numeric(chunk['loan_id'], errors='coerce')
        chunk['mrp'] = pd.to_numeric(chunk['mrp'], errors='coerce')
        chunk = chunk.dropna(subset=['loan_id', 'mrp'])
        chunk['loan_id'] = chunk['loan_id'].astype(np.int64)
        chunk['ym'] = mmyyyy_to_yyyymm(chunk['mrp'].astype(np.int64).values)

        chunk['term'] = pd.to_numeric(chunk['term'], errors='coerce')
        chunk['orig_upb'] = pd.to_numeric(chunk['orig_upb'], errors='coerce')
        chunk['note_rate'] = pd.to_numeric(chunk['note_rate'], errors='coerce')
        chunk['cur_upb'] = pd.to_numeric(chunk['cur_upb'], errors='coerce')
        chunk['orig_date'] = pd.to_numeric(chunk['orig_date'], errors='coerce')
        chunk['orig_ym'] = mmyyyy_to_yyyymm(chunk['orig_date'].fillna(0).astype(np.int64).values)

        g = chunk.groupby('loan_id')
        static_parts.append(g[['term', 'orig_upb', 'note_rate', 'orig_ym']].first())

        my = chunk.loc[chunk['mod_flag'] == 'Y']
        if len(my):
            modY_parts.append(my.groupby('loan_id')['ym'].min())

        z = chunk.loc[chunk['zbc'].notna() & (chunk['zbc'].str.strip() != '')]
        if len(z):
            # earliest non-blank zbc within the chunk, per loan
            zc_chunk = z.sort_values('ym').groupby('loan_id', as_index=False).first()
            zbc_parts.append(zc_chunk[['loan_id', 'ym', 'zbc']])

        # reduce to last row per loan WITHIN the chunk (keeps partials small)
        last_idx = chunk.groupby('loan_id')['ym'].idxmax()
        last_parts.append(chunk.loc[last_idx, ['loan_id', 'ym', 'cur_upb']])

    # ---- reduce across chunks ----
    static = pd.concat(static_parts).groupby(level=0).first()

    if modY_parts:
        modY = pd.concat(modY_parts).groupby(level=0).min().rename('first_modY_ym')
    else:
        modY = pd.Series(dtype='int64', name='first_modY_ym')

    if zbc_parts:
        zc = pd.concat(zbc_parts).sort_values('ym').groupby('loan_id').first()
        zc = zc.rename(columns={'ym': 'zbc_ym', 'zbc': 'zbc_code'})
    else:
        zc = pd.DataFrame(columns=['zbc_ym', 'zbc_code'])

    last = pd.concat(last_parts).sort_values('ym').groupby('loan_id').last()
    last = last.rename(columns={'ym': 'last_ym', 'cur_upb': 'last_upb'})

    out = static.join(last, how='left').join(modY, how='left').join(zc, how='left')
    out = out.reset_index()
    out['first_modY_ym'] = out['first_modY_ym'].fillna(0).astype(np.int64)
    out['zbc_ym'] = out['zbc_ym'].fillna(0).astype(np.int64)
    out['zbc_code'] = out['zbc_code'].fillna('')
    out['last_ym'] = out['last_ym'].fillna(0).astype(np.int64)
    out['orig_ym'] = out['orig_ym'].fillna(0).astype(np.int64)

    out.to_parquet(out_path, index=False)
    print(f"{csv_path}: {len(out):,} loans -> {out_path}", flush=True)


if __name__ == '__main__':
    main()
