"""
compare_seedcheck_checkpoints.py -- bit-for-bit comparison of the two
same-seed IPW training runs (run_tag=_seedcheck_a / _seedcheck_b), both
using the newly-added torch.manual_seed(42)/torch.cuda.manual_seed_all(42)
in train_hazard_multiobs.py. Compares hazard_best.pt and hazard_final.pt
independently -- best is saved whenever an epoch beats best_auc-so-far, so
the epoch number that "wins" could itself differ between runs even if the
underlying training trajectory is otherwise deterministic.

Usage:
    python scripts/diag/compare_seedcheck_checkpoints.py
"""
import os
import torch

BASE = '/scratch/at7095/mortgage_prepayment'
DIR_A = os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a')
DIR_B = os.path.join(BASE, 'outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_seedcheck_b')


def compare(name: str, path_a: str, path_b: str) -> bool:
    print(f'\n{"=" * 100}')
    print(f'{name}')
    print(f'  A: {path_a}')
    print(f'  B: {path_b}')
    print(f'{"=" * 100}')
    ckpt_a = torch.load(path_a, map_location='cpu')
    ckpt_b = torch.load(path_b, map_location='cpu')
    print(f'  A: epoch={ckpt_a.get("epoch","?")} auc={ckpt_a.get("auc","?")}')
    print(f'  B: epoch={ckpt_b.get("epoch","?")} auc={ckpt_b.get("auc","?")}')

    sd_a, sd_b = ckpt_a['model_state'], ckpt_b['model_state']
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    if keys_a != keys_b:
        print(f'  Parameter key sets DIFFER: only in A: {keys_a - keys_b} | only in B: {keys_b - keys_a}')
        return False

    all_identical = True
    max_diff_overall = 0.0
    for k in sorted(keys_a):
        ta, tb = sd_a[k], sd_b[k]
        if ta.shape != tb.shape:
            print(f'  {k}: SHAPE MISMATCH {tuple(ta.shape)} vs {tuple(tb.shape)}')
            all_identical = False
            continue
        identical = torch.equal(ta, tb)
        if not identical:
            all_identical = False
            diff = (ta.float() - tb.float()).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            print(f'  {k}: NOT identical, max abs diff = {diff:.3e}')

    print(f'\n  All {len(keys_a)} parameter tensors bit-identical: {all_identical}')
    if not all_identical:
        print(f'  Max abs param diff overall: {max_diff_overall:.3e}')
    return all_identical


def main():
    best_ok = compare('hazard_best.pt (A vs B)',
                       os.path.join(DIR_A, 'hazard_best.pt'), os.path.join(DIR_B, 'hazard_best.pt'))
    final_ok = compare('hazard_final.pt (A vs B)',
                        os.path.join(DIR_A, 'hazard_final.pt'), os.path.join(DIR_B, 'hazard_final.pt'))

    print(f'\n{"=" * 100}')
    print('VERDICT')
    print(f'{"=" * 100}')
    if best_ok and final_ok:
        print('  Both checkpoints are bit-identical between run A and run B. '
              'torch.manual_seed()/torch.cuda.manual_seed_all() closed the reproducibility gap -- '
              'same-seed training now produces the same model. Safe to proceed to the '
              'epoch1-vs-epoch50 comparison on a single canonical run.')
    else:
        print('  Checkpoints are NOT fully bit-identical -- seeding is still incomplete. '
              'Remaining nondeterminism is likely from non-deterministic CUDA/cuDNN kernels '
              '(e.g. attention, conv, atomic-add reductions) that ignore the RNG seed. '
              'Next step: set torch.backends.cudnn.deterministic=True, '
              'torch.backends.cudnn.benchmark=False, and/or torch.use_deterministic_algorithms(True) '
              'in train_and_evaluate() before model construction, then rerun the seedcheck.')


if __name__ == '__main__':
    main()
