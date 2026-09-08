#!/bin/bash
#SBATCH --job-name=forecast_ipw_seedcheck_cmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/forecast_ipw_seedcheck_cmp_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/forecast_ipw_seedcheck_cmp_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "IPW seedcheck_a epoch comparison: hazard_best.pt (epoch 4) vs hazard_final.pt (epoch 50)"
echo "Seed-controlled run -- run-to-run variance already ruled out (compare_seedcheck_checkpoints.py)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/forecast_multiobs_ipw_seedcheck_epoch_compare.py

echo "Done: $(date)"
