"""term_report.py -- Q1a/1b/1c/1d + Q2 from the per-loan parquet table.

Reads .claude_tmp/term_pass/parquet/*.parquet (one per vintage file) into a
single per-loan table, then answers the advisor-Sep-29 facts round. No
pipeline data changes; prints compact summaries for transcription into the reply.
"""
import glob
import os
import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
PQ = os.path.join(BASE, '.claude_tmp/term_pass/parquet')
SEQ = os.path.join(BASE, 'data/sequences_rolling')
ENS = os.path.join(BASE, 'outputs/rolling')
BIN_EDGES = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]


def term_bucket(t):
    return np.where(t == 360, '360', np.where(t == 180, '180',
                    np.where(t == 240, '240', 'other')))


def load_loans():
    files = sorted(glob.glob(os.path.join(PQ, '*.parquet')))
    print(f'loading {len(files)} parquets', flush=True)
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    # a loan_id should be unique across files; guard anyway (keep first)
    df = df.drop_duplicates('loan_id', keep='first')
    df['orig_year'] = df['orig_ym'] // 100
    df['tb'] = term_bucket(df['term'].values)
    print(f'  {len(df):,} unique loans', flush=True)
    return df


def q1a(df):
    print('\n===== Q1a: original-term distribution by vintage YEAR (orig_date) =====')
    d = df[(df['orig_year'] >= 2000) & (df['orig_year'] <= 2023)]
    cnt = d.pivot_table(index='orig_year', columns='tb', values='loan_id',
                        aggfunc='count', fill_value=0)
    upb = d.pivot_table(index='orig_year', columns='tb', values='orig_upb',
                        aggfunc='sum', fill_value=0)
    for name, tab in [('LOAN COUNT', cnt), ('ORIG UPB ($)', upb)]:
        tab = tab.reindex(columns=['360', '240', '180', 'other'], fill_value=0)
        tab['total'] = tab.sum(axis=1)
        tab['non360_share'] = 1 - tab['360'] / tab['total']
        print(f'\n-- {name} --')
        with pd.option_context('display.float_format', lambda x: f'{x:,.4f}' if x < 1 else f'{x:,.0f}'):
            print(tab.to_string())


def _pop_terms(df, loan_ids):
    """map term bucket onto an array of loan_ids (loan-month rows)."""
    m = df.set_index('loan_id')['tb']
    tb = pd.Series(loan_ids).map(m)
    return tb


def q1b_q2(df):
    print('\n===== Q1b / Q2: non-360 share & post-mod share of LOAN-MONTHS per population =====')
    modmap = df.set_index('loan_id')['first_modY_ym']
    pops = []
    # training arrays
    for tag, sub in [('cutoff_2002 TRAIN', 'cutoff_2002_zbc_multiobs_f0.2_h1_hist/train'),
                     ('cutoff_2002 TEST', 'cutoff_2002_zbc_multiobs_f0.2_h1_hist/test'),
                     ('cutoff_2020 TRAIN', 'cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP/train'),
                     ('cutoff_2020 TEST', 'cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP/test')]:
        lid = np.load(os.path.join(SEQ, sub + '_loan_ids.npy'))
        rm = np.load(os.path.join(SEQ, sub + '_ref_month.npy'))
        pops.append((tag, lid, rm))
    # ensemble CSVs
    for tag, path in [('2003 one-step', 'ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv'),
                      ('2001-02 in-sample', 'ensemble_onestep_insample_cutoff_2002/ensemble_merged_all_months.csv'),
                      ('2021 one-step', 'ensemble_onestep_cutoff_2020_control/ensemble_merged_all_months.csv')]:
        e = pd.read_csv(os.path.join(ENS, path), usecols=['loan_id', 'ref_month'])
        pops.append((tag, e['loan_id'].values, e['ref_month'].values))

    rows = []
    for tag, lid, rm in pops:
        tb = _pop_terms(df, lid)
        n = len(lid)
        n_missing = tb.isna().sum()
        non360 = (tb.values != '360') & (~pd.isna(tb.values))
        share_non360 = non360.sum() / n
        # post-mod: ref_month >= first_modY_ym (and modY>0)
        fm = pd.Series(lid).map(modmap).fillna(0).values
        postmod = (fm > 0) & (rm >= fm)
        rows.append((tag, n, int(n_missing), non360.sum(), share_non360,
                     int(postmod.sum()), postmod.sum() / n))
    r = pd.DataFrame(rows, columns=['population', 'n_loan_months', 'n_term_missing',
                                    'n_non360', 'non360_share', 'n_postmod', 'postmod_share'])
    with pd.option_context('display.float_format', lambda x: f'{x:.5f}' if abs(x) < 1 else f'{x:,.0f}'):
        print(r.to_string(index=False))


def q1c(df):
    print('\n===== Q1c: same-incentive bin ratios (Sum h / Sum realized), split 360 vs non-360 =====')
    tmap = df.set_index('loan_id')['tb']
    for tag, path in [('2003 (cutoff_2002)', 'ensemble_onestep_cutoff_2002/ensemble_merged_all_months.csv'),
                      ('2021 (cutoff_2020 control)', 'ensemble_onestep_cutoff_2020_control/ensemble_merged_all_months.csv')]:
        e = pd.read_csv(os.path.join(ENS, path),
                        usecols=['loan_id', 'incentive_at_ref', 'realized_event', 'h'])
        e['tb'] = e['loan_id'].map(tmap)
        e['grp'] = np.where(e['tb'] == '360', '360', 'non-360')
        e['bin'] = pd.cut(e['incentive_at_ref'], bins=BIN_EDGES)
        print(f'\n-- {tag} --')
        for grp in ['360', 'non-360']:
            g = e[e['grp'] == grp].dropna(subset=['bin'])
            agg = g.groupby('bin', observed=True).agg(
                n=('h', 'size'), sum_h=('h', 'sum'), sum_real=('realized_event', 'sum'))
            agg['ratio'] = agg['sum_h'] / agg['sum_real'].replace(0, np.nan)
            print(f'  [{grp}]  total loan-months={len(g):,}')
            for b, row in agg.iterrows():
                print(f'    {str(b):>14}  n={int(row.n):>8,}  ratio={row.ratio:.3f}  '
                      f'(pred={row.sum_h/row.n:.5f} real={row.sum_real/row.n:.5f})')


def q1d(df):
    print('\n===== Q1d: code-01 near-maturity share (|age_at_term - term|<=2), split by term, by TERMINATION year =====')
    c = df[df['zbc_code'] == '01'].copy()
    c = c[(c['zbc_ym'] > 0) & (c['orig_ym'] > 0) & (c['term'] > 0)]
    oy, om = c['orig_ym'] // 100, c['orig_ym'] % 100
    ty, tm = c['zbc_ym'] // 100, c['zbc_ym'] % 100
    c['age_at_term'] = (ty * 12 + tm) - (oy * 12 + om)
    c['term_year'] = ty
    c['near_mat'] = (c['age_at_term'] - c['term']).abs() <= 2
    c['grp'] = np.where(c['term'] == 360, '360', 'non-360')
    c = c[(c['term_year'] >= 2001) & (c['term_year'] <= 2025)]
    for grp in ['360', 'non-360', 'ALL']:
        sub = c if grp == 'ALL' else c[c['grp'] == grp]
        t = sub.groupby('term_year').agg(n_code01=('loan_id', 'size'),
                                         n_nearmat=('near_mat', 'sum'))
        t['share'] = t['n_nearmat'] / t['n_code01']
        print(f'\n-- {grp} --')
        with pd.option_context('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1 else f'{x:,.0f}'):
            print(t.to_string())


def main():
    df = load_loans()
    q1a(df)
    q1b_q2(df)
    q1c(df)
    q1d(df)


if __name__ == '__main__':
    main()
