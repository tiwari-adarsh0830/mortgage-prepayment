#!/bin/bash
#SBATCH --job-name=rolling_2002_30y_s123
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/rolling_2002_30y_s123_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/rolling_2002_30y_s123_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "One-step-ahead rolling forecast | cutoff_2002 _30y | seed=123 | 12 months Dec2002-Nov2003"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_rolling_one_step.py \
    --cutoff_year 2002 \
    --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y \
    --ckpt_path outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s123/hazard_best.pt \
    --map_era fixed \
    --include_pre2013 \
    --cell_sample outputs/pre2013_cell_sample_30y_loans.csv \
    --seed_label seed123 \
    --frozen_scores outputs/rolling/dec_window_cutoff_2002_30y_seed123/dec_window_scores_seed123.csv \
    --out_dir outputs/rolling/rolling_onestep_cutoff_2002_30y_seed123

echo "Done: rolling one-step cutoff_2002 _30y seed=123 | $(date)"
