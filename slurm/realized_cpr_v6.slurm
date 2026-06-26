#!/bin/bash
#SBATCH --job-name=realized_v6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/realized_v6_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/realized_v6_%j.err
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python scripts/realized_cpr_v6.py
