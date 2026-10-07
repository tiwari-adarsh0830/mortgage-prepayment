#!/bin/bash
#SBATCH --job-name=preflight_f0.2_L33
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/preflight_f0.2_L33_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/preflight_f0.2_L33_%j.err

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "PRE-FLIGHT: multiobs train | cutoff=2020 h=1 | sampling_mode=fixed_fraction frac_draws=0.2 max_seq_len=33"
echo "3 epochs only -- confirms train_hazard_multiobs.py's new SEQ_DIR/max_seq logic (Step A) and"
echo "ipw_weight() actually run against real fixed_fraction data before committing to 50-epoch runs."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py \
    --cutoff_year 2020 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 33 \
    --label_horizon 1 \
    --n_epochs 3 \
    --use_ipw \
    --run_tag _preflight_f0.2

echo "Done: preflight cutoff=2020 multiobs f0.2 h=1 L=33 --use_ipw (3 epochs) | $(date)"
