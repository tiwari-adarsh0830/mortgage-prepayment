"""test_train_hazard_multiobs_smoke.py — synthetic smoke test for
train_hazard_multiobs.py's training loop.

NO cluster data is touched. Builds a small in-memory synthetic panel of
fixed-window observations (right-aligned mask, one forward label each — the
shape multiobs sequences actually have) and runs a handful of epochs through
train_and_evaluate() to confirm: the loop runs end to end, loss decreases,
AUC computes without error, and both --pos_ratio and --use_ipw code paths
(separately and combined) execute without crashing. This is NOT a
convergence test and does NOT touch data/ or outputs/ — it is small and fast
enough to run on the login node, and nothing else should run there.

Run:
    cd /scratch/at7095/mortgage_prepayment
    python scripts/diag/test_train_hazard_multiobs_smoke.py
"""
import sys
sys.path.insert(0, 'scripts')

import numpy as np
import torch

# Login node has many cores (28); PyTorch's default intra-op thread pool
# thrashes badly on the tiny tensors this smoke test uses, turning a
# few-hundred-row/few-epoch run into a multi-minute one. Cap it down.
torch.set_num_threads(2)

import train_hazard_multiobs as m

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail and not cond else ''))
    if not cond:
        FAILURES.append(name)


def check_ipw_weight_direction():
    """Regression check for the incl_prob -> weight DIRECTION bug (caught
    2026-09-07): the pre-fix code weighted BY incl_prob (w = incl_prob),
    which downweights rare pool draws instead of upweighting them via
    Horvitz-Thompson (w = 1/incl_prob) -- and passed this smoke test
    silently, because every other check here only verifies use_ipw=True
    EXECUTES, never that its weight points the right DIRECTION. Calls
    train_hazard_multiobs.ipw_weight() directly (the real function
    train_and_evaluate uses), not a hand-rolled duplicate, so a regression
    in that function is what this test is actually exercising."""
    incl_prob = np.array([1.0, 1.0, 0.1, 0.1], dtype=np.float32)
    w = m.ipw_weight(incl_prob).cpu().numpy()
    check('IPW weight direction: incl_prob=0.1 observations get a HIGHER '
          'weight than incl_prob=1.0 observations',
          w[2] > w[0] and w[3] > w[1],
          f'incl_prob={incl_prob.tolist()} -> weight={w.tolist()}')
    check('IPW weight direction: incl_prob=1.0 (mandatory/terminal draw) '
          'gets the MINIMUM weight (1.0)',
          w[0] == 1.0 and w[1] == 1.0,
          f'weight={w.tolist()}')


def make_split(n, seed):
    """Right-aligned fixed windows (mask[:, -1] always True, like the real
    multiobs builder), with the label weakly tied to a feature so the model
    has something learnable and loss/AUC aren't pure noise."""
    rng = np.random.default_rng(seed)
    T, F = m.MAX_SEQ, m.N_FEATURES
    seq = rng.normal(size=(n, T, F)).astype(np.float32)

    lengths = rng.integers(1, T + 1, size=n)
    mask = np.zeros((n, T), dtype=bool)
    for i in range(n):
        mask[i, T - lengths[i]:] = True

    signal = seq[np.arange(n), T - 1, 0]
    prob   = 1.0 / (1.0 + np.exp(-1.5 * signal))
    labels = (rng.random(n) < prob * 0.3).astype(np.float32)
    # force at least a few of each class so AUC is always defined
    labels[:3]  = 1.0
    labels[3:6] = 0.0

    incl_prob = rng.uniform(0.2, 1.0, size=n).astype(np.float32)
    return seq, mask, labels, incl_prob


def run():
    check_ipw_weight_direction()

    train_seq, train_mask, train_labels, train_incl_prob = make_split(400, seed=0)
    test_seq,  test_mask,  test_labels,  _                = make_split(150, seed=1)

    check('train positive rate in (0, 1)', 0.0 < train_labels.mean() < 1.0,
          f'mean={train_labels.mean():.4f}')
    check('test positive rate in (0, 1)', 0.0 < test_labels.mean() < 1.0,
          f'mean={test_labels.mean():.4f}')

    configs = [
        ('default (pos_ratio=None, use_ipw=False)', None, False),
        ('pos_ratio=0.5',                            0.5,  False),
        ('use_ipw=True',                              None, True),
        ('pos_ratio=0.3 + use_ipw=True (combined)',   0.3,  True),
    ]

    for label, pos_ratio, use_ipw in configs:
        print(f'\n--- {label} ---', flush=True)
        try:
            results = m.train_and_evaluate(
                train_seq, train_mask, train_labels, train_incl_prob,
                test_seq, test_mask, test_labels,
                n_epochs=4, pos_ratio=pos_ratio, use_ipw=use_ipw,
                out_dir=None, steps_per_epoch=15, batch_size=32, seed=7,
            )
        except Exception as e:
            check(f'{label}: runs without crashing', False, f'{type(e).__name__}: {e}')
            continue

        check(f'{label}: runs without crashing', True)
        check(f'{label}: history has one entry per epoch',
              len(results['history']) == 4, f"got {len(results['history'])}")

        losses = [h['loss'] for h in results['history']]
        aucs   = [h['auc']  for h in results['history']]
        # Compare the mean of the first two epochs against the mean of the
        # last two rather than epoch 1 vs epoch 4 directly: at this toy scale
        # (400 rows, 4 epochs, batch_size=32) a single epoch's loss is noisy
        # enough to bounce either direction, especially under use_ipw where
        # per-sample weights further shrink the effective batch. The real
        # 50-epoch cluster run (thousands of steps/epoch) declines smoothly;
        # this check only needs to catch a training loop that's broken, not
        # reproduce that smoothness on a handful of noisy toy epochs.
        check(f'{label}: loss trended down (first-2-epoch mean > last-2-epoch mean)',
              np.mean(losses[:2]) > np.mean(losses[-2:]), f'losses={losses}')
        check(f'{label}: all AUCs finite and in [0, 1]',
              all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in aucs), f'aucs={aucs}')
        check(f'{label}: best_auc in [0, 1]',
              0.0 <= results['best_auc'] <= 1.0, f"best_auc={results['best_auc']}")
        check(f'{label}: platt_a/platt_b are finite floats',
              np.isfinite(results['platt_a']) and np.isfinite(results['platt_b']),
              f"a={results['platt_a']} b={results['platt_b']}")

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S): {FAILURES}')
        sys.exit(1)
    print(f'All checks passed ({len(configs)} configs exercised).')


if __name__ == '__main__':
    run()
