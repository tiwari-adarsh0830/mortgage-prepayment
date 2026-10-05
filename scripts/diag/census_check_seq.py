"""census_check_seq.py -- generic a_eff/b_eff census check for a cutoff_{year}
..._seq multiobs build (any cutoff year, the 23-cutoff x 10-seed yearly-
sequence chain -- Part 3/4 of the Oct 5, 2026 prompt). Generalizes
census_check_2002_30y.py/census_check_2020_30y.py (both hardcoded to one
cutoff's BUILD_DIR/CENSUS_JSON) to an arbitrary --build_dir/--census_json.

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

--divide_out_subsample: pass this when the build's modern-era population was
drawn via a uniform --sample_frac (as every cutoff >= 2013 in this chain
is, per prepare_sequences_multiobs_zbc.py --sample_frac 0.1) -- divides
a_eff/b_eff by r = sample_loans/census_loans first, same correction
census_check_2020_30y.py applies, so the ratio isn't conflated with the
subsample rate. Do NOT pass it for a cutoff whose population is cell-grid-
STRATIFIED only (no modern vintages relevant, e.g. cutoff_2002-2012) --
that sample's non-uniformity doesn't behave like a flat subsample, and
dividing by r there would over-correct (see census_check_2002_30y.py,
which has no such division).

NOT a pass/fail gate (unlike check_build_seq.py) -- informational, run after
the census baseline (census_panel_baseline.py) and the build both exist.

Usage:
    python scripts/diag/census_check_seq.py \\
        --build_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_seq \\
        --census_json outputs/census_panel_baseline_cutoff_2002_seq.json
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build_dir', type=str, required=True)
    ap.add_argument('--census_json', type=str, required=True)
    ap.add_argument('--divide_out_subsample', action='store_true',
                     help='Divide a_eff/b_eff by r=sample_loans/census_loans -- use when '
                          'the modern-era population was drawn via a uniform --sample_frac '
                          '(see module docstring).')
    args = ap.parse_args()

    census = json.load(open(args.census_json))
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
        lbl  = np.load(os.path.join(args.build_dir, f'{split}_labels.npy'))
        incl = np.load(os.path.join(args.build_dir, f'{split}_incl_prob.npy'))
        lids = np.load(os.path.join(args.build_dir, f'{split}_loan_ids.npy'), allow_pickle=True)

        n_obs_total += lbl.shape[0]
        n_events += int((lbl == 1).sum())
        sum_inv_p += float((1.0 / incl).sum())
        sum_label_over_p += float((lbl / incl).sum())
        loan_ids_all.update(lids.tolist())

    n_nonevents = n_obs_total - n_events
    sample_loans = len(loan_ids_all)

    a_eff = n_events / census_events
    b_eff = n_nonevents / census_nonevents
    r = sample_loans / census_loans if census_loans else float('nan')
    if args.divide_out_subsample:
        a_eff, b_eff = a_eff / r, b_eff / r
    log_ratio = np.log(a_eff / b_eff)
    ipw_rate = sum_label_over_p / sum_inv_p

    print(f'Census json: {args.census_json}')
    print(f'  n_loans={census_loans:,}  n_eligible_loan_months={census_elig:,}  '
          f'n_prepay_events={census_events:,}  non_events={census_nonevents:,}  '
          f'raw_prepay_rate={census_rate:.6f}')
    print()
    print(f'Sample (train+test): {args.build_dir}')
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
    if args.divide_out_subsample:
        print(f'r = sample_loans / census_loans = {sample_loans:,} / {census_loans:,} = {r:.6f}')
        print(f'a_eff = (sample events / census events) / r = {a_eff:.6f}')
        print(f'b_eff = (sample non-events / census non-events) / r = {b_eff:.6f}')
    else:
        print(f'a_eff = sample events / census events       = {n_events:,} / {census_events:,} = {a_eff:.6f}')
        print(f'b_eff = sample non-events / census non-events = {n_nonevents:,} / {census_nonevents:,} = {b_eff:.6f}')
    print(f'log(a_eff / b_eff) = {log_ratio:.6f}')
    print()
    print(f'IPW-weighted sample prepay rate = {ipw_rate:.6f}   (census raw_prepay_rate = {census_rate:.6f})')


if __name__ == '__main__':
    main()
