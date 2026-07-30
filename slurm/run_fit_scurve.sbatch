#!/bin/bash
#SBATCH --job-name=fit_scurve
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --time=0:15:00
#SBATCH --mem=8G
#SBATCH --output=logs/fit_scurve_%j.out
#SBATCH --error=logs/fit_scurve_%j.err
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
mkdir -p config
python -u scripts/diag/fit_terminal_scurve.py
