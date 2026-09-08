"""
diag_multiobs_ipw_weight_direction.py — Diagnostic (not a forecasting
pipeline): verifies the DIRECTION of incl_prob-based inverse-probability
weighting, and characterizes the incl_prob distribution actually present in
the multiobs test set. No model, no GPU, no cluster job needed.

WHY THIS EXISTS, AND WHY IT IS NOT USED FOR CPR FORECASTING
-----------------------------------------------------------------
This module started life as forecast_multiobs_cpr_ipw.py, an attempt to
combine a loan's several multiobs-sampled (ref_month) observations into ONE
per-loan hazard via:

    h_loan = sum_i( sigmoid(logit_i) / incl_prob_i ) / sum_i( 1 / incl_prob_i )

and feed h_loan into forecast_rolling_cpr.py's aggregate() the same way
origination/trailing's h_t is fed in. THAT ATTEMPT WAS WRONG and was never
run on real data or submitted as an sbatch job -- shelved here instead of
deleted, because the two things it verified are still worth keeping:

  1. verify_ipw_weight_direction() -- confirms 1/incl_prob weighting
     genuinely UPweights a rare (small incl_prob) observation and
     DOWNweights a certain-inclusion (incl_prob=1.0) one. A synthetic
     3-observation check: incl_prob=[1.0, 1.0, 0.05] scores=[0.80,0.20,0.99]
     gives weights=[1.0, 1.0, 20.0] (the rare draw gets 90.9% of the total
     weight) and h_loan=0.9455 -- the rare observation's own score
     dominates. That IS the correct Horvitz-Thompson direction (a rarely-
     included observation stands in for more of its stratum's unsampled
     population, so it is upweighted, not downweighted) -- confirmed here,
     not assumed, since it is a one-line formula that is very easy to get
     backwards (e.g. weighting BY incl_prob instead of 1/incl_prob) and have
     it silently produce a plausible-looking number.

  2. characterize_incl_prob_distribution() -- real incl_prob values from
     data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1/test_incl_prob.npy
     (1,715,642 observations): range 0.043-1.0, resulting weight range
     1.0-23.5 (mean 6.9), 27.6% of observations are the mandatory
     (incl_prob=1.0, weight=1.0) draw. Confirms the dominance behavior in
     (1) is not a hypothetical edge case -- single-digit-percent incl_prob
     values, and hence >10x weight ratios between a loan's observations,
     are common in this dataset.

WHY THE ABANDONED APPROACH WAS THE WRONG ESTIMAND, NOT JUST UNTESTED
--------------------------------------------------------------------------
By the multiobs builder's own design (prepare_sequences_multiobs_zbc.py), a
loan's several sampled ref_months deliberately span DIFFERENT ages /
incentive regimes across its life -- that separation of incentive from
burnout is the entire point of multiobs sampling. Pooling all of a loan's
observations into ONE incl_prob-weighted number therefore estimates that
loan's IPW-weighted AVERAGE hazard across its sampled reference months, a
LIFETIME-AVERAGE hazard over whatever window select_observations() drew
from -- not a CUTOFF-CONDITIONAL hazard ("given this loan's state at the
Dec-2020 forecast cutoff, what is its monthly prepay probability"), which is
the estimand aggregate() and prior_shift_offset() were built around, and the
estimand origination's and trailing's h_t both already are. Feeding a
lifetime-average h_loan into aggregate() alongside those two would compare a
DIFFERENT estimand under the same column name, not correct multiobs's
calibration against a like-for-like target -- it would not have been a
meaningful CPR forecast no matter how correct the IPW arithmetic was.

If a cutoff-conditional, IPW-corrected multiobs forecast is wanted later,
the fix is upstream of this file: retrain with --use_ipw
(train_hazard_multiobs.py supports it --
scripts/slurm/run_train_multiobs_ipw_2020.sbatch) so the LOSS is
IPW-corrected during training and the trained model's cutoff-conditional
score needs no further per-loan reweighting at inference. Do not resurrect
the per-loan aggregation this file used to contain for that purpose.

Usage:
    python scripts/diag/diag_multiobs_ipw_weight_direction.py
"""
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
MULTIOBS_SEQ_DIR = os.path.join(
    BASE, 'data/sequences_rolling/cutoff_2020_zbc_multiobs_k5_h1')


def ipw_aggregate_to_loan(loan_ids: np.ndarray, scores: np.ndarray, incl_prob: np.ndarray):
    """Horvitz-Thompson-style per-loan reduction, kept ONLY so this
    diagnostic's synthetic check exercises the exact formula that was
    considered for (and rejected from) CPR forecasting -- see module
    docstring. Not called anywhere outside this file.

        h_loan = sum_i( score_i / incl_prob_i ) / sum_i( 1 / incl_prob_i )
    """
    assert np.all(incl_prob > 0), 'incl_prob must be > 0 (1/incl_prob would be inf/undefined).'
    assert np.all(incl_prob <= 1.0 + 1e-6), f'incl_prob > 1.0 found (max={incl_prob.max()}).'

    df = pd.DataFrame({'loan_id': loan_ids, 'score': scores, 'incl_prob': incl_prob})
    df['w']   = 1.0 / df['incl_prob']
    df['num'] = df['w'] * df['score']
    grouped   = df.groupby('loan_id').agg(num_sum=('num', 'sum'), w_sum=('w', 'sum'))
    h_loan    = (grouped['num_sum'] / grouped['w_sum']).astype(np.float64)
    return h_loan.index.to_numpy(), h_loan.to_numpy(dtype=np.float32)


def verify_ipw_weight_direction():
    """Synthetic, no cluster data, no GPU. One loan, three observations:

        obs A: incl_prob=1.0  (certain-inclusion, e.g. mandatory/terminal), score=0.80
        obs B: incl_prob=1.0  (a second certain-inclusion draw),            score=0.20
        obs C: incl_prob=0.05 (a rare pool draw),                           score=0.99

    Checks two things, reported plainly rather than forcing a conclusion:
      1) WEIGHT DIRECTION: incl_prob=0.05 must get the LARGEST weight
         (1/0.05=20), incl_prob=1.0 the SMALLEST (1.0 each) -- the
         Horvitz-Thompson direction. Asserted, not eyeballed.
      2) DOMINANCE: given that direction, does the rare observation's score
         end up dominating h_loan? Reported as fact -- this IS the expected,
         correct consequence of the formula when a loan has few observations
         and one sits in a rare stratum, not a bug.
    """
    loan_ids  = np.array([1, 1, 1])
    scores    = np.array([0.80, 0.20, 0.99], dtype=np.float64)
    incl_prob = np.array([1.0, 1.0, 0.05], dtype=np.float64)

    ids_out, h = ipw_aggregate_to_loan(loan_ids, scores, incl_prob)
    w     = 1.0 / incl_prob
    share = w / w.sum()

    print('=' * 78)
    print('SYNTHETIC IPW FORMULA-DIRECTION CHECK (no cluster data, no GPU)')
    print('=' * 78)
    print(f'  scores:                {scores.tolist()}')
    print(f'  incl_prob:             {incl_prob.tolist()}')
    print(f'  weights (1/incl_prob): {w.tolist()}')
    print(f'  weight share of total: {[round(float(s), 4) for s in share]}')
    print(f'  naive UNWEIGHTED mean score: {scores.mean():.4f}')
    print(f'  IPW-weighted h_loan:         {h[0]:.4f}')

    assert w[2] > w[0] and w[2] > w[1] and w[0] == w[1] == 1.0, (
        f'IPW weight direction is BACKWARDS: expected incl_prob=0.05 to have '
        f'the LARGEST weight and the two incl_prob=1.0 observations the '
        f'SMALLEST (equal to each other). Got w={w.tolist()}.')
    print('  [PASS] Weight direction correct: certain-inclusion (incl_prob=1.0) '
          'observations get the LOWEST weight (1.0 each); the rare '
          f'(incl_prob=0.05) observation gets the HIGHEST weight ({w[2]:.1f}).')

    dist_to_rare  = abs(h[0] - scores[2])
    dist_to_other = abs(h[0] - scores[:2].mean())
    dominates = dist_to_rare < dist_to_other
    print(f'  Rare observation\'s score dominates h_loan? {"YES" if dominates else "NO"} '
          f'(h_loan={h[0]:.4f} vs rare score {scores[2]:.2f} vs other-two mean '
          f'{scores[:2].mean():.2f}). Expected and correct for this formula with '
          f'few observations and one rare stratum -- see module docstring for why '
          f'this made the formula the wrong ESTIMAND for CPR forecasting anyway.')
    print('=' * 78, flush=True)
    return h[0]


def characterize_incl_prob_distribution(seq_dir: str = MULTIOBS_SEQ_DIR):
    """Real incl_prob values, multiobs test set. Reads a single ~6.8MB array
    directly -- no model, no GPU, no cluster job."""
    p = np.load(os.path.join(seq_dir, 'test_incl_prob.npy'))
    w = 1.0 / p

    print('=' * 78)
    print('REAL incl_prob DISTRIBUTION -- multiobs test set (cutoff_2020_zbc_'
          'multiobs_k5_h1)')
    print('=' * 78)
    print(f'  n observations: {len(p):,}')
    print(f'  incl_prob: min={p.min():.5f}  mean={p.mean():.5f}  max={p.max():.5f}')
    pct = np.percentile(p, [1, 5, 25, 50, 75, 95, 99])
    print(f'  incl_prob percentiles [1,5,25,50,75,95,99]: '
          f'{[round(float(x), 5) for x in pct]}')
    frac_mandatory = float((p == 1.0).mean())
    print(f'  fraction incl_prob==1.0 (mandatory/terminal draw): '
          f'{frac_mandatory * 100:.2f}%')
    print(f'  resulting weight (1/incl_prob): min={w.min():.2f}  '
          f'mean={w.mean():.2f}  max={w.max():.2f}')
    print(f'  Confirms dominance in verify_ipw_weight_direction() is not a '
          f'hypothetical edge case: {(p < 0.1).mean() * 100:.2f}% of real '
          f'observations have incl_prob < 0.1 (weight > 10x a mandatory draw).')
    print('=' * 78, flush=True)
    return {
        'n': len(p), 'min': float(p.min()), 'mean': float(p.mean()),
        'max': float(p.max()), 'frac_mandatory': frac_mandatory,
        'weight_min': float(w.min()), 'weight_mean': float(w.mean()),
        'weight_max': float(w.max()),
    }


def main():
    verify_ipw_weight_direction()
    print()
    characterize_incl_prob_distribution()


if __name__ == '__main__':
    main()
