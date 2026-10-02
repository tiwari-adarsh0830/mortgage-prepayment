#!/bin/bash
#SBATCH --job-name=ensemble_2020_30y
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/ensemble_2020_30y_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/ensemble_2020_30y_%j.err
set -euo pipefail

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/ensemble_onestep_2020_control.py --run_tag _30y

echo "Job finished: $(date)"
