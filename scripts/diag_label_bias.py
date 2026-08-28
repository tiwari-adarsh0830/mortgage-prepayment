"""
Is the extra_13 label wrong uniformly, or does it catch a non-random subset?

Established already (2013Q1, full file, cutoff-censored the same way the
rolling builder does it):
    cutoff Dec2018  zbc==01  236,823 (34.8%)   extra_13==1        0 (0.00%)
    cutoff Dec2020  zbc==01  354,363 (52.0%)   extra_13==1    3,421 (0.50%)
    cutoff Dec2022  zbc==01  459,279 (67.4%)   extra_13==1    6,791 (1.00%)
Censoring hits both columns identically, so it cannot explain the gap.

The question here is narrower and decides whether calibration can rescue it.
If extra_13 flags a RANDOM subset of true prepayments, the training target is
a scaled-down version of the truth and the Platt step may recover the shape.
If it flags a SELECTED subset -- concentrated in particular coupons, credit
scores, or loan ages -- the bias is not a level shift and calibration cannot
fix it.

Columns are pandas 0-indexed (= awk field - 1):
    1 loan_id | 2 monthly_reporting_period (MMYYYY) | 7 original_interest_rate
    13 origination_date (MMYYYY) | 23 borrower_credit_score | 15 loan_age
    43 zero_balance_code | 106 extra_13

MMYYYY -> YYYYMM before ANY date compare. Both date fields go through the
same conversion; no raw MMYYYY integer is ever compared.
"""
import os
import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
RAW = os.path.join(BASE, 'data/raw')
OUT = os.path.join(BASE, 'outputs')
CUTOFF = 202212          # YYYYMM
VINTAGES = ['2013Q1', '2014Q1', '2015Q1', '2016Q1']
CHUNK = 2_000_000


def mmyyyy_to_yyyymm(s):
    """Series of MMYYYY strings -> YYYYMM ints. Never compare raw MMYYYY."""
    z = s.astype(str).str.zfill(6)
    return z.str[2:].astype(int) * 100 + z.str[:2].astype(int)


def scan(vintage):
    path = os.path.join(RAW, vintage + '.csv')
    rate, fico, age = {}, {}, {}
    zbc_set, ex_set, all_set = set(), set(), set()

    for ch in pd.read_csv(path, sep='|', header=None,
                          usecols=[1, 2, 7, 15, 23, 43, 106],
                          names=['loan_id', 'mrp', 'rate', 'age',
                                 'fico', 'zbc', 'ex13'],
                          dtype=str, chunksize=CHUNK):
        ch['yyyymm'] = mmyyyy_to_yyyymm(ch['mrp'])
        ch = ch[ch['yyyymm'] <= CUTOFF]
        if ch.empty:
            continue
        ch['zbc'] = pd.to_numeric(ch['zbc'], errors='coerce')
        ch['ex13'] = pd.to_numeric(ch['ex13'], errors='coerce')
        ch['rate'] = pd.to_numeric(ch['rate'], errors='coerce')
        ch['fico'] = pd.to_numeric(ch['fico'], errors='coerce')
        ch['age'] = pd.to_numeric(ch['age'], errors='coerce')

        all_set.update(ch['loan_id'].unique())
        zbc_set.update(ch.loc[ch['zbc'] == 1.0, 'loan_id'].unique())
        ex_set.update(ch.loc[ch['ex13'] == 1.0, 'loan_id'].unique())

        first = ch.dropna(subset=['rate']).drop_duplicates('loan_id')
        for lid, r, fc in zip(first['loan_id'], first['rate'], first['fico']):
            if lid not in rate:
                rate[lid] = float(r)
                fico[lid] = float(fc) if pd.notna(fc) else np.nan

        # age at the terminal row, for loans that terminate
        term = ch.dropna(subset=['age'])
        term = term.loc[(term['zbc'] == 1.0) | (term['ex13'] == 1.0)]
        for lid, a in zip(term['loan_id'], term['age']):
            if lid not in age:
                age[lid] = float(a)

    return all_set, zbc_set, ex_set, rate, fico, age


def main():
    rows = []
    for v in VINTAGES:
        allb, zb, ex, rate, fico, age = scan(v)
        print('\n===== %s (cutoff %d) =====' % (v, CUTOFF), flush=True)
        print('loans %d | zbc01 %d (%.4f) | extra13 %d (%.4f) | overlap %d'
              % (len(allb), len(zb), len(zb) / max(len(allb), 1),
                 len(ex), len(ex) / max(len(allb), 1), len(zb & ex)))

        # is extra_13 a subset of the true prepayments at all?
        print('extra13 loans that are ALSO zbc01 : %d of %d'
              % (len(zb & ex), len(ex)))

        df = pd.DataFrame({
            'loan_id': list(allb),
        })
        df['cpn'] = df['loan_id'].map(lambda x: rate.get(x, np.nan))
        df['cpn'] = (np.round(df['cpn'] * 2) / 2.0)
        df['fico'] = df['loan_id'].map(lambda x: fico.get(x, np.nan))
        df['zbc'] = df['loan_id'].isin(zb).astype(int)
        df['ex13'] = df['loan_id'].isin(ex).astype(int)

        print('\nprepay rate by coupon, both labels:')
        g = df.groupby('cpn')[['zbc', 'ex13']].agg(['mean', 'sum', 'size'])
        print(g.to_string())

        print('\nmean FICO: all %.1f | zbc01 %.1f | extra13 %.1f'
              % (df['fico'].mean(),
                 df.loc[df.zbc == 1, 'fico'].mean(),
                 df.loc[df.ex13 == 1, 'fico'].mean()))

        a_zbc = [age[l] for l in zb if l in age]
        a_ex = [age[l] for l in ex if l in age]
        if a_zbc:
            print('terminal loan_age: zbc01 median %.0f mean %.1f (n=%d)'
                  % (np.median(a_zbc), np.mean(a_zbc), len(a_zbc)))
        if a_ex:
            print('terminal loan_age: extra13 median %.0f mean %.1f (n=%d)'
                  % (np.median(a_ex), np.mean(a_ex), len(a_ex)))

        rows.append({'vintage': v, 'loans': len(allb), 'zbc01': len(zb),
                     'extra13': len(ex), 'overlap': len(zb & ex)})

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'diag_label_bias.csv'),
                              index=False)
    print('\nwrote outputs/diag_label_bias.csv')


if __name__ == '__main__':
    main()
