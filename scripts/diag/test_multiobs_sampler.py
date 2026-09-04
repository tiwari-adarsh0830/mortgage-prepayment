"""test_multiobs_sampler.py — synthetic-panel unit tests for the multi-observation
sequence sampler in prepare_sequences_multiobs_zbc.py.

NO cluster data is touched. Everything here is an in-memory synthetic panel,
built to be tiny and fast enough to run on the login node -- and nothing else
should run on the login node. This only exercises select_observations() /
build_sequences_multiobs(); it does not read raw Fannie files, does not fit a
production scaler, and does not write anything to data/ or outputs/.

Run:
    cd /scratch/at7095/mortgage_prepayment
    python scripts/diag/test_multiobs_sampler.py
"""
import contextlib
import io
import re
import sys
sys.path.insert(0, 'scripts')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import prepare_sequences_multiobs_zbc as m

FEATURE_COLS = m.FEATURE_COLS
AGE_IDX = FEATURE_COLS.index('loan_age_months')

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail and not cond else ''))
    if not cond:
        FAILURES.append(name)


def add_months(yyyymm: int, n: int) -> int:
    y, mo = divmod(yyyymm, 100)
    mo += n
    y += (mo - 1) // 12
    mo = (mo - 1) % 12 + 1
    return y * 100 + mo


def make_loan(loan_id: str, n_rows: int, event_row: int | None,
              start_yyyymm: int = 201001) -> pd.DataFrame:
    """event_row: 0-based row index of the zero_balance_code_actual==1 row,
    or None for a censored loan (no event -- Fannie just stops reporting)."""
    rows = []
    for t in range(n_rows):
        rows.append({
            'loan_id': loan_id,
            'yyyymm': add_months(start_yyyymm, t),
            'zero_balance_code_actual': 1.0 if (event_row is not None and t == event_row) else np.nan,
            'refi_incentive': 0.5 * np.sin(t / 3.0),   # varies -- exercises incentive scheme too
            'borrower_credit_score': 720.0,
            'original_ltv': 80.0,
            'current_ltv': 75.0 + 0.1 * t,
            'original_upb': 300_000.0,
            'loan_age_months': float(t),
            'dti': 36.0,
            'loan_purpose_enc': 0.0,
            'property_type_enc': 0.0,
        })
    return pd.DataFrame(rows)


def make_loan_with_gap(loan_id: str, n_rows: int, event_row: int | None,
                        gap_before_row: int, gap_extra_months: int = 3,
                        start_yyyymm: int = 201001) -> pd.DataFrame:
    """Same as make_loan, but yyyymm jumps by (1 + gap_extra_months) months
    immediately before row_idx=gap_before_row, simulating an interior row
    that got dropna'd out upstream: row_idx stays 0..n_rows-1 (a contiguous
    POSITION sequence) while the underlying calendar month sequence gets a
    real hole at that point."""
    rows = []
    offset = 0
    for t in range(n_rows):
        if t == gap_before_row:
            offset += gap_extra_months
        rows.append({
            'loan_id': loan_id,
            'yyyymm': add_months(start_yyyymm, t + offset),
            'zero_balance_code_actual': 1.0 if (event_row is not None and t == event_row) else np.nan,
            'refi_incentive': 0.5 * np.sin(t / 3.0),
            'borrower_credit_score': 720.0,
            'original_ltv': 80.0,
            'current_ltv': 75.0 + 0.1 * t,
            'original_upb': 300_000.0,
            'loan_age_months': float(t),
            'dti': 36.0,
            'loan_purpose_enc': 0.0,
            'property_type_enc': 0.0,
        })
    return pd.DataFrame(rows)


# ── Synthetic panel ────────────────────────────────────────────────────────
# long_prepaid  : L=50, event at row 40 (term_t=40)      -- long, prepaid
# long_censored : L=35, no event       (term_t=34=L-1)   -- long, censored
# short_prepaid : L=8,  event at row 6 (term_t=6)         -- short but pool
#                 still large enough for a full k=5 draw
# last_row_event: L=10, event at row 9 (term_t=9=L-1)     -- event coincides
#                 with the last row; must not be mistaken for censored
# two_row       : L=2,  no event       (term_t=1=L-1)     -- pool too small
#                 for more than the mandatory draw
# one_row       : L=1,  no event       (term_t=0=L-1)     -- no eligible t
#                 at all under default H=1, min_hist=1

LOANS = {
    'long_prepaid':   dict(n_rows=50, event_row=40),
    'long_censored':  dict(n_rows=35, event_row=None),
    'short_prepaid':  dict(n_rows=8,  event_row=6),
    'last_row_event': dict(n_rows=10, event_row=9),
    'two_row':        dict(n_rows=2,  event_row=None),
    'one_row':        dict(n_rows=1,  event_row=None),
}


def build_panel() -> pd.DataFrame:
    return pd.concat(
        [make_loan(lid, **spec) for lid, spec in LOANS.items()],
        ignore_index=True,
    )


K_DRAWS, H, MIN_HIST = 5, 1, 1


def run():
    df = build_panel()

    obs = m.select_observations(df, K_DRAWS, H, MIN_HIST, draw_scheme='uniform')

    # ── 1. no observation from the 1-row loan ──────────────────────────────
    check('no observation from the 1-row loan',
          (obs['loan_id'] == 'one_row').sum() == 0,
          f"got {(obs['loan_id'] == 'one_row').sum()} rows")

    # ── 2. fixed k regardless of loan length (long loans of different L) ───
    for lid in ['long_prepaid', 'long_censored', 'short_prepaid']:
        k_actual = obs.loc[obs['loan_id'] == lid, 'k_actual']
        check(f'k_actual == k_draws for {lid} (length-invariant)',
              len(k_actual) > 0 and (k_actual == K_DRAWS).all(),
              f'k_actual values: {k_actual.unique().tolist()}')

    # two_row: pool too small, only the mandatory draw survives
    two_row_obs = obs[obs['loan_id'] == 'two_row']
    check('two_row loan gets exactly 1 observation (pool-limited)',
          len(two_row_obs) == 1, f'got {len(two_row_obs)}')

    # ── 3. exactly one terminal draw per loan, at term_t - H ───────────────
    for lid, spec in LOANS.items():
        if lid == 'one_row':
            continue
        sub = obs[obs['loan_id'] == lid]
        term_t = spec['n_rows'] - 1 if spec['event_row'] is None else spec['event_row']
        terminal = sub[sub['is_terminal']]
        check(f'{lid}: exactly one terminal draw',
              len(terminal) == 1, f'got {len(terminal)}')
        if len(terminal) == 1:
            check(f'{lid}: terminal draw at t = term_t - H',
                  int(terminal.iloc[0]['t']) == term_t - H,
                  f"t={terminal.iloc[0]['t']} expected={term_t - H}")

    # ── 4. one positive per prepaid loan at H=1, zero for censored ─────────
    for lid in ['long_prepaid', 'short_prepaid', 'last_row_event']:
        sub = obs[obs['loan_id'] == lid]
        check(f'{lid}: exactly one positive label',
              sub['label'].sum() == 1, f"sum={sub['label'].sum()}")
        # and it must be the terminal draw
        pos_row = sub[sub['label'] == 1]
        check(f'{lid}: the positive is the terminal draw',
              len(pos_row) == 1 and bool(pos_row.iloc[0]['is_terminal']))
    for lid in ['long_censored', 'two_row']:
        sub = obs[obs['loan_id'] == lid]
        check(f'{lid}: zero positives (censored)', sub['label'].sum() == 0)

    # ── 5. no observation at or after its loan's event month ───────────────
    for lid, spec in LOANS.items():
        if spec['event_row'] is None or lid == 'one_row':
            continue
        sub = obs[obs['loan_id'] == lid]
        check(f'{lid}: no observation at/after the event row',
              (sub['t'] < spec['event_row']).all(),
              f"max t={sub['t'].max()} event_row={spec['event_row']}")

    # ── 6. incl_prob: 1.0 for terminal, (k-1)/pool for uniform non-terminal ─
    check('incl_prob == 1.0 for all terminal draws',
          (obs.loc[obs['is_terminal'], 'incl_prob'] == 1.0).all())
    nonterm = obs[~obs['is_terminal']]
    expected = np.minimum(K_DRAWS - 1, nonterm['n_eligible']) / nonterm['n_eligible']
    check('incl_prob == (k-1)/pool for uniform non-terminal draws',
          np.allclose(nonterm['incl_prob'].to_numpy(), expected.to_numpy()))

    # ── 7. determinism across repeated calls ───────────────────────────────
    obs2 = m.select_observations(df, K_DRAWS, H, MIN_HIST, draw_scheme='uniform')
    key_cols = ['loan_id', 't']
    a = obs[key_cols].sort_values(key_cols).reset_index(drop=True)
    b = obs2[key_cols].sort_values(key_cols).reset_index(drop=True)
    check('determinism: identical (loan_id, t) set across repeated calls', a.equals(b))

    # ── 8. invariance to input row order ────────────────────────────────────
    df_shuffled = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    obs_shuf = m.select_observations(df_shuffled, K_DRAWS, H, MIN_HIST, draw_scheme='uniform')
    c = obs_shuf[key_cols].sort_values(key_cols).reset_index(drop=True)
    check('invariance to input row order (shuffled frame gives the same set)', a.equals(c))

    # ── 9. k=5 draws a strict subset of k=7 ─────────────────────────────────
    obs_k7 = m.select_observations(df, 7, H, MIN_HIST, draw_scheme='uniform')
    set5 = set(map(tuple, obs[key_cols].to_numpy()))
    set7 = set(map(tuple, obs_k7[key_cols].to_numpy()))
    check('k=5 selection is a subset of k=7 selection', set5.issubset(set7))
    check('k=5 selection is a STRICT subset of k=7 (k=7 adds new obs)', set5 < set7)

    # ── 10/11. mask length + last-timestep age via build_sequences_multiobs ─
    scaler = StandardScaler().fit(df[FEATURE_COLS])
    seq, mask, labels, prepay_t, loan_ids, extra = m.build_sequences_multiobs(
        df, scaler, K_DRAWS, H, MIN_HIST, draw_scheme='uniform')

    n_obs = seq.shape[0]
    check('build_sequences_multiobs observation count matches select_observations',
          n_obs == len(obs), f'{n_obs} vs {len(obs)}')

    mask_lengths = mask.sum(axis=1)
    # Recompute expected length in the same (deterministic) row order that
    # build_sequences_multiobs's internal call to select_observations produced.
    obs_unsorted = m.select_observations(df, K_DRAWS, H, MIN_HIST, draw_scheme='uniform')
    expected_len = np.minimum(m.MAX_SEQ_LEN, obs_unsorted['t'].to_numpy() + 1)
    check('mask length == min(MAX_SEQ_LEN, t+1) for every observation',
          np.array_equal(mask_lengths, expected_len))
    check('last timestep is always valid (mask[:, -1] all True)',
          bool(mask[:, -1].all()))

    unscaled_last = scaler.inverse_transform(seq[:, -1, :])
    age_last = unscaled_last[:, AGE_IDX]
    check("last timestep's UNSCALED loan_age_months equals age_at_ref",
          np.allclose(age_last, extra['age_at_ref'], atol=1e-3),
          f'max abs diff = {np.max(np.abs(age_last - extra["age_at_ref"])):.4f}')

    # prepay_timestep must be meaningless/-1 everywhere (file-shape only)
    check('prepay_timestep is -1 for every observation (meaningless by construction)',
          bool((prepay_t == -1).all()))

    # ── 12/13. calendar-gap filters ─────────────────────────────────────────
    # gap_before_event: prepaid, L=12, event at row 10 (term_t=10), with a
    # calendar gap inserted immediately before row 10 -- i.e. right where the
    # mandatory draw (t=term_t-H=9) would sit. Every eligible candidate's
    # calendar distance to the event is inflated by the same gap, so the
    # label-window filter must remove ALL of this loan's candidates: it
    # should contribute ZERO observations.
    gap_before_event = make_loan_with_gap(
        'gap_before_event', n_rows=12, event_row=10, gap_before_row=10, gap_extra_months=3)

    # gap_mid_window: prepaid, L=50, event at row 45 (term_t=45), with a
    # calendar gap early (before row 10). The mandatory draw sits at t=44
    # with window start=max(0,44-32)=12, which is PAST the gap (row 10) --
    # so the mandatory draw is untouched. Candidate t=12 has window
    # start=0, which DOES span the gap -- Filter 2 must drop t=12
    # specifically, regardless of hash-based selection odds.
    gap_mid_window = make_loan_with_gap(
        'gap_mid_window', n_rows=50, event_row=45, gap_before_row=10, gap_extra_months=3)

    df_gap = pd.concat([df, gap_before_event, gap_mid_window], ignore_index=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        obs_gap = m.select_observations(df_gap, K_DRAWS, H, MIN_HIST, draw_scheme='uniform')
    printed = buf.getvalue()
    print(printed, end='')  # surface it in the real run's output too

    nums = re.search(r'dropped ([\d,]+) obs .*?, ([\d,]+) obs', printed)
    n_window_gap_dropped = int(nums.group(1).replace(',', '')) if nums else -1
    n_label_mismatch_dropped = int(nums.group(2).replace(',', '')) if nums else -1

    check('calendar filters: window-gap counter fires (> 0 dropped)',
          n_window_gap_dropped > 0, f'parsed count={n_window_gap_dropped}')
    check('calendar filters: label-mismatch counter fires (> 0 dropped)',
          n_label_mismatch_dropped > 0, f'parsed count={n_label_mismatch_dropped}')

    gbe_obs = obs_gap[obs_gap['loan_id'] == 'gap_before_event']
    check('gap_before_event: label-window filter removes ALL of this loan\'s candidates',
          len(gbe_obs) == 0, f'got {len(gbe_obs)} observations, t values={gbe_obs["t"].tolist()}')

    gmw_obs = obs_gap[obs_gap['loan_id'] == 'gap_mid_window']
    check('gap_mid_window: mandatory draw survives (gap is outside its window)',
          bool((gmw_obs.loc[gmw_obs['is_terminal'], 't'] == 44).all())
          and gmw_obs['is_terminal'].sum() == 1)
    check('gap_mid_window: the gap-spanning candidate (t=12) is never selected',
          12 not in gmw_obs['t'].to_numpy())

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S): {FAILURES}')
        sys.exit(1)
    print(f'All checks passed ({len(obs)} observations from {len(LOANS)} synthetic loans).')


if __name__ == '__main__':
    run()
