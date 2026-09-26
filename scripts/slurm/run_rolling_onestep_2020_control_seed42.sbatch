#!/bin/bash
#SBATCH --job-name=rolling_2020_s42
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/rolling_2020_s42_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/rolling_2020_s42_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "One-step-ahead rolling forecast (MATCHED CONTROL) | cutoff_2020 | seed=42 | 12 months Dec2020-Nov2021"
echo "Reuses the existing cached combined pass (no new raw scan)."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_rolling_one_step.py \
    --cutoff_year 2020 \
    --seq_dir data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP \
    --ckpt_path outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33/hazard_best.pt \
    --map_era prefix \
    --seed_label seed42 \
    --frozen_scores outputs/rolling/dec_window_cutoff_2020_control_seed42/dec_window_scores_seed42.csv \
    --out_dir outputs/rolling/rolling_onestep_cutoff_2020_control_seed42

echo "Done: rolling one-step cutoff_2020 control seed=42 | $(date)"
