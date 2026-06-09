#!/bin/bash
#SBATCH --job-name=cpr
#SBATCH --account=torch_pr_932_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/cpr_%j.out
#SBATCH --error=logs/cpr_%j.err
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python3 scripts/realized_cpr_by_coupon.py
