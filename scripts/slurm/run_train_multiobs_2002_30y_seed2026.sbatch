#!/bin/bash
#SBATCH --job-name=train_f0.2_L33_2002_30y_seed2026
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2002_30y_seed2026_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2002_30y_seed2026_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Multiobs train | cutoff=2002 h=1 | fixed_fraction f=0.2 | max_seq_len=33 | --include_pre2013 | 50 epochs | --use_ipw | seed=2026 | _30y sample"
echo "cutoff_2002 training run on the _30y cell-sample rebuild (data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y)"
echo "Checkpointing on (default --ckpt_every 10) -- train_ckpt.pt resumes automatically if this run is killed and resubmitted"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py \
    --cutoff_year 2002 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 33 \
    --label_horizon 1 \
    --n_epochs 50 \
    --use_ipw \
    --include_pre2013 \
    --seed 2026 \
    --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y \
    --run_tag _cutoff_2002_30y_s2026 \
    --ckpt_every 10

echo "Done: cutoff=2002 multiobs f0.2 h=1 L=33 --include_pre2013 --use_ipw seed=2026 _30y | $(date)"
