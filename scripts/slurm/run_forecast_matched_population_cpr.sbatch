#!/bin/bash
#SBATCH --job-name=forecast_matched_population_cpr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/forecast_matched_population_cpr_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/forecast_matched_population_cpr_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Matched-population CPR comparison: origination vs trailing vs multiobs(uncorrected)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/forecast_matched_population_cpr.py

echo "Done: $(date)"
