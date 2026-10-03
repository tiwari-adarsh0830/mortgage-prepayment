"""census_check_2020_30y.py -- compare the cutoff_2020 _30y multiobs build's
sampled train+test observations against the full-population census baseline
in outputs/census_panel_baseline_cutoff_2020_30y.json.

Cloned from census_check_2002_30y.py, with ONE methodological difference:
cutoff_2020's population is drawn via a uniform 10% LOAN subsample
(load_vintage_filtered's --sample_frac, applied at Pass-1 discovery,
BEFORE the per-loan fixed_fraction incl_prob logic runs on the sampled
loans) -- a different sampling mechanism from cutoff_2002's cell-grid
stratified sample. Comparing a_eff/b_eff directly against the census
(as the cutoff_2002 script does) would conflate the 10% loan-level
subsample rate with the fixed_fraction non-event draw rate. Instead,
divide both out by r = sample_loans / census_loans first (the same
"divide the 10% subsample out" step used for cutoff_2020's Sep 25
a_eff/b_eff check on the trailing-builder population) so a_eff/b_eff
are comparable to cutoff_2002's (which has no such uniform pre-filter).

Computes:
  - r = sample_loans / census_loans (the realized uniform-subsample rate)
  - a_eff = (sample events / census events) / r
  - b_eff = (sample non-events / census non-events) / r
  - log(a_eff / b_eff)
  - IPW-weighted prepay rate of the sample next to the census raw_prepay_rate

"events" = observations with label==1; "non-events" = the rest.
"""
import json
import os

import numpy as np

BASE = '/scratch/at7095/mortgage_prepayment'
BUILD_DIR = os.path.join(BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_30y')
CENSUS_JSON = os.path.join(BASE, 'outputs/census_panel_baseline_cutoff_2020_30y.json')


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
    r = sample_loans / census_loans

    a_eff_raw = n_events / census_events
    b_eff_raw = n_nonevents / census_nonevents
    a_eff = a_eff_raw / r
    b_eff = b_eff_raw / r
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
    print(f'r = sample_loans / census_loans = {sample_loans:,} / {census_loans:,} = {r:.6f}')
    print()
    print(f'a_eff_raw (not subsample-corrected) = {n_events:,} / {census_events:,} = {a_eff_raw:.6f}')
    print(f'b_eff_raw (not subsample-corrected) = {n_nonevents:,} / {census_nonevents:,} = {b_eff_raw:.6f}')
    print()
    print(f'a_eff = a_eff_raw / r = {a_eff:.6f}')
    print(f'b_eff = b_eff_raw / r = {b_eff:.6f}')
    print(f'log(a_eff / b_eff) = {log_ratio:.6f}')
    print()
    print(f'IPW-weighted sample prepay rate = {ipw_rate:.6f}   (census raw_prepay_rate = {census_rate:.6f})')


if __name__ == '__main__':
    main()
