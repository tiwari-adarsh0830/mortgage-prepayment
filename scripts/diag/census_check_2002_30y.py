"""census_check_2002_30y.py -- compare the cutoff_2002 _30y multiobs build's
sampled train+test observations against the full-population census baseline
in outputs/census_panel_baseline_cutoff_2002_30y.json.

Computes:
  - a_eff = sample events / census events
  - b_eff = sample non-events / census non-events
  - log(a_eff / b_eff)  (the choice-based-sampling intercept offset)
  - IPW-weighted prepay rate of the sample (sum(label/incl_prob) /
    sum(1/incl_prob)) next to the census raw_prepay_rate

"events" = observations with label==1 (the H=1 is_prepaid flag emitted by
prepare_sequences_multiobs_zbc.py); "non-events" = the rest. a_eff is only
a same-population check if the build's loan population matches the census
json's population exactly -- this script prints both n_loans figures so
that can be verified rather than assumed.
"""
import json
import os

import numpy as np

BASE = '/scratch/at7095/mortgage_prepayment'
BUILD_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y')
CENSUS_JSON = os.path.join(BASE, 'outputs/census_panel_baseline_cutoff_2002_30y.json')


def main():
    census = json.load(open(CENSUS_JSON))
    census_loans = census['overall']['n_loans']
    census_events = census['overall']['n_prepay_events']
    census_elig = census['overall']['n_eligible_loan_months']
    census_nonevents = census_elig - census_events
    census_rate = census['overall']['raw_prepay_rate']

    n_obs_total = 0
    n_events = 0
    sum_inv_p = 0.0
    sum_label_over_p = 0.0
    loan_ids_all = set()

    for split in ('train', 'test'):
        lbl  = np.load(os.path.join(BUILD_DIR, f'{split}_labels.npy'))
        incl = np.load(os.path.join(BUILD_DIR, f'{split}_incl_prob.npy'))
        lids = np.load(os.path.join(BUILD_DIR, f'{split}_loan_ids.npy'), allow_pickle=True)

        n_obs_total += lbl.shape[0]
        n_events += int((lbl == 1).sum())
        sum_inv_p += float((1.0 / incl).sum())
        sum_label_over_p += float((lbl / incl).sum())
        loan_ids_all.update(lids.tolist())

    n_nonevents = n_obs_total - n_events
    sample_loans = len(loan_ids_all)

    a_eff = n_events / census_events
    b_eff = n_nonevents / census_nonevents
    log_ratio = np.log(a_eff / b_eff)
    ipw_rate = sum_label_over_p / sum_inv_p

    print(f'Census json: {CENSUS_JSON}')
    print(f'  n_loans={census_loans:,}  n_eligible_loan_months={census_elig:,}  '
          f'n_prepay_events={census_events:,}  non_events={census_nonevents:,}  '
          f'raw_prepay_rate={census_rate:.6f}')
    print()
    print(f'Sample (train+test): {BUILD_DIR}')
    print(f'  n_loans (distinct)={sample_loans:,}  n_obs={n_obs_total:,}  '
          f'events (label==1)={n_events:,}  non_events={n_nonevents:,}')
    print()
    if sample_loans != census_loans:
        print(f'  NOTE: sample n_loans ({sample_loans:,}) != census n_loans ({census_loans:,}) -- '
              f'a_eff is not a strict same-population check; the build and the census json '
              f'are drawing on different loan populations (e.g. the multiobs build drops '
              f'loans with zero sampled/eligible observations under its calendar filters, '
              f'while the census json counts every loan that passed the loan-level filters).')
    else:
        print('  NOTE: sample n_loans == census n_loans -- same population.')
    print()
    print(f'a_eff = sample events / census events       = {n_events:,} / {census_events:,} = {a_eff:.6f}')
    print(f'b_eff = sample non-events / census non-events = {n_nonevents:,} / {census_nonevents:,} = {b_eff:.6f}')
    print(f'log(a_eff / b_eff) = {log_ratio:.6f}')
    print()
    print(f'IPW-weighted sample prepay rate = {ipw_rate:.6f}   (census raw_prepay_rate = {census_rate:.6f})')


if __name__ == '__main__':
    main()
