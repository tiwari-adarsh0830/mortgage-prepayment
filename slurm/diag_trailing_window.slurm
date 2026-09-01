#!/bin/bash
#SBATCH --job-name=diag_trail
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/diag_trailing_%j.out
#SBATCH --error=logs/diag_trailing_%j.err

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python scripts/diag_trailing_window.py
