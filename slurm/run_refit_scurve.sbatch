#!/bin/bash
#SBATCH --job-name=refit_sc
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --time=0:15:00
#SBATCH --mem=8G
#SBATCH --output=logs/refit_sc_%j.out
#SBATCH --error=logs/refit_sc_%j.err
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python -u scripts/diag/refit_scurve_weighted.py
