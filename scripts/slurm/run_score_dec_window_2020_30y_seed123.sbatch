#!/bin/bash
#SBATCH --job-name=score_decwin_2020_30y_s123
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2020_30y_s123_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2020_30y_s123_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Dec-2020 window scoring | _30y build | seed=123"
echo "Checkpoint: outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_cutoff_2020_30y_s123/hazard_best.pt"
echo "map_era=fixed. No --cell_sample: cutoff_2020 has no --include_pre2013, so cell_sample_path"
echo "is never consulted (it only gates PRE2013_VINTAGES membership)."
echo "Prerequisite for E123's --frozen_scores (score_rolling_one_step.py's Dec-month consistency check)."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2020 \
    --seq_dir data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_30y \
    --ckpt_path outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_cutoff_2020_30y_s123/hazard_best.pt \
    --map_era fixed \
    --seed_label seed123 \
    --out_dir outputs/rolling/dec_window_cutoff_2020_30y_seed123

echo "Done: Dec-2020 window scoring _30y seed=123 | $(date)"
