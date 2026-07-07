#!/bin/bash
#SBATCH --job-name=cpr_upb
#SBATCH --account=torch_pr_932_general
#SBATCH --time=20:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/realized_cpr_v6_upb_%j.out
#SBATCH --error=logs/realized_cpr_v6_upb_%j.err

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

cd /scratch/at7095/mortgage_prepayment
python3 scripts/realized_cpr_v6_upb.py
