#!/bin/bash
#SBATCH --job-name=score_decwin_2002_30y_s2026
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2002_30y_s2026_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2002_30y_s2026_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Dec-2002 window scoring | _30y build | seed=2026"
echo "Checkpoint: outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s2026/hazard_best.pt"
echo "map_era=fixed | cell_sample=outputs/pre2013_cell_sample_30y_loans.csv"
echo "Reuses the raw-pass cache populated by the _30y consistency test run if the pop_hash matches."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2002 \
    --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_30y \
    --ckpt_path outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_cutoff_2002_30y_s2026/hazard_best.pt \
    --map_era fixed \
    --include_pre2013 \
    --cell_sample outputs/pre2013_cell_sample_30y_loans.csv \
    --seed_label seed2026 \
    --out_dir outputs/rolling/dec_window_cutoff_2002_30y_seed2026

echo "Done: Dec-2002 window scoring _30y seed=2026 | $(date)"
