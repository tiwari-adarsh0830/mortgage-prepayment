"""
One-off smoke test for the harp_eligible placeholder column (Part 2 of the
Oct 5 prompt). Builds a tiny synthetic panel (bypasses raw CSV I/O -- this
is a schema/shape check, not a data-correctness check) to confirm:
  1. FEATURE_COLS has 10 entries, N_FEATURES == 10, harp_eligible is last.
  2. StandardScaler.fit on data including the all-zero harp_eligible column
     does not raise and sets scale_ for that column to 1.0 (sklearn's
     zero-variance safeguard), so transform() returns exactly 0, not NaN/inf.
  3. build_sequences_multiobs's gather path produces (n_obs, MAX_SEQ_LEN, 10)
     arrays with column 9 all zero after unscaling.
  4. PrepaymentTransformer(input_dim=10) accepts a (B, T, 10) tensor.

Not a permanent test -- ad hoc, run once, not wired into any gate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import prepare_sequences_multiobs_zbc as m
from train_hazard_multiobs import PrepaymentTransformer

print(f'FEATURE_COLS ({len(m.FEATURE_COLS)}): {m.FEATURE_COLS}', flush=True)
assert len(m.FEATURE_COLS) == 10, f'expected 10 FEATURE_COLS, got {len(m.FEATURE_COLS)}'
assert m.FEATURE_COLS[-1] == 'harp_eligible', f'harp_eligible not last: {m.FEATURE_COLS}'
assert m.N_FEATURES == 10, f'N_FEATURES={m.N_FEATURES}, expected 10'
print('[1] FEATURE_COLS/N_FEATURES OK', flush=True)

# ── Synthetic panel: 5 loans, 6 months each, harp_eligible always 0 ─────────
rng = np.random.default_rng(0)
n_loans, n_months = 5, 6
rows = []
for lid in range(n_loans):
    for t in range(n_months):
        rows.append({
            'loan_id': lid, 'yyyymm': 200001 + t,
            'refi_incentive': rng.normal(), 'borrower_credit_score': rng.normal(700, 40),
            'original_ltv': rng.normal(80, 10), 'current_ltv': rng.normal(75, 10),
            'original_upb': rng.normal(200_000, 50_000), 'loan_age_months': float(t),
            'dti': rng.normal(35, 5), 'loan_purpose_enc': float(rng.integers(0, 3)),
            'property_type_enc': float(rng.integers(0, 4)), 'harp_eligible': 0.0,
            'row_idx': t, 'L': n_months, 'term_t': n_months - 1,
            'prepaid': 0, 'is_terminal': False, 'ref_month': 200001 + t,
        })
panel = pd.DataFrame(rows)

scaler = StandardScaler()
scaler.fit(panel[m.FEATURE_COLS])
harp_i = m.FEATURE_COLS.index('harp_eligible')
print(f'[2] scaler.mean_[{harp_i}]={scaler.mean_[harp_i]!r}  '
      f'scaler.scale_[{harp_i}]={scaler.scale_[harp_i]!r}', flush=True)
assert scaler.mean_[harp_i] == 0.0
assert scaler.scale_[harp_i] == 1.0, 'sklearn should set scale_=1 for zero-variance, not 0'
transformed = scaler.transform(panel[m.FEATURE_COLS])
assert np.all(transformed[:, harp_i] == 0.0), 'harp_eligible did not transform to exactly 0'
assert not np.any(np.isnan(transformed)), 'NaN in transformed features'
print('[2] StandardScaler zero-variance handling OK (scale_=1, transform=0, no NaN)', flush=True)

obs = pd.DataFrame({
    'loan_id': [0, 1], 't': [n_months - 1, n_months - 1],
    'ref_month': [200006, 200006], 'label': [0.0, 0.0],
    'is_terminal': [False, False], 'incl_prob': [1.0, 1.0],
    'n_eligible': [1, 1], 'k_actual': [1, 1],
    'age_at_ref': [5.0, 5.0], 'incentive_at_ref': [0.0, 0.0],
})
sequences, masks, labels, prepay_ts, loan_ids, extras = m.build_sequences_multiobs(
    panel, scaler, k_draws=1, H=1, min_hist=1, obs=obs)
print(f'[3] sequences.shape={sequences.shape}  masks.shape={masks.shape}', flush=True)
assert sequences.shape[-1] == 10, f'sequences last dim {sequences.shape[-1]}, expected 10'
harp_col = sequences[..., harp_i][masks]
assert np.all(harp_col == 0.0), 'harp_eligible column not all zero in built sequences'
print('[3] build_sequences_multiobs produces (n_obs, MAX_SEQ_LEN, 10), col 9 all zero', flush=True)

model = PrepaymentTransformer(input_dim=10, max_seq=m.MAX_SEQ_LEN)
x = torch.zeros(4, m.MAX_SEQ_LEN, 10)
mask_t = torch.ones(4, m.MAX_SEQ_LEN, dtype=torch.bool)
out = model(x, mask=mask_t)
print(f'[4] PrepaymentTransformer(input_dim=10) forward pass OK, out.shape={tuple(out.shape)}',
      flush=True)

print('\nALL SMOKE CHECKS PASSED', flush=True)
