#!/bin/bash
#SBATCH --job-name=train_f0.2_L33_2020_30y_seed1001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2020_30y_seed1001_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2020_30y_seed1001_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Multiobs train | cutoff=2020 h=1 | fixed_fraction f=0.2 | max_seq_len=33 | 50 epochs | --use_ipw | seed=1001 | _30y rebuild"
echo "map_era=fixed (post-CP/U-fix, built under the current loader -- NOT map_era=prefix like the old GOLDEN_BACKUP checkpoints)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py \
    --cutoff_year 2020 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 33 \
    --label_horizon 1 \
    --n_epochs 50 \
    --use_ipw \
    --seed 1001 \
    --seq_dir data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_30y \
    --run_tag _cutoff_2020_30y_s1001 \
    --ckpt_every 10

echo "Done: cutoff=2020 multiobs f0.2 h=1 L=33 --use_ipw seed=1001 _30y | $(date)"
