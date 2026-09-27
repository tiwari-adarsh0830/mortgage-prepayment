#!/bin/bash
#SBATCH --job-name=insample_2002_s123
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/insample_2002_s123_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/insample_2002_s123_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "In-sample one-step-ahead rolling scoring | cutoff_2002 | seed=123 | Jan2001-Nov2002 (23 months) + Dec2002 anchor"
echo "Reuses the existing cached combined pass (no new raw scan)."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_rolling_onestep_insample_2002.py \
    --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist \
    --ckpt_path outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed123/hazard_best.pt \
    --map_era fixed \
    --include_pre2013 \
    --seed_label seed123 \
    --anchor_scores outputs/rolling/rolling_onestep_cutoff_2002_seed123/rolling_month_200212.csv \
    --out_dir outputs/rolling/rolling_onestep_insample_cutoff_2002_seed123

echo "Done: in-sample one-step cutoff_2002 seed=123 | $(date)"
