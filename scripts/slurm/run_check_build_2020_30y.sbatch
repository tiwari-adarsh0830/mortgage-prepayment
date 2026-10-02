#!/bin/bash
#SBATCH --job-name=check_build_2020_30y
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/check_build_2020_30y_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/check_build_2020_30y_%j.err
set -euo pipefail

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/diag/check_build_2020_30y.py \
    --build_log /scratch/at7095/mortgage_prepayment/logs/multiobs_2020_30y_19067719.out

echo "Job finished: $(date)"
