#!/bin/bash
#SBATCH --job-name=ar1_std
#SBATCH --account=torch_pr_932_general
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage3_ar1_standardized_%j.out
#SBATCH --error=logs/stage3_ar1_standardized_%j.err

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

cd /scratch/at7095/mortgage_prepayment
python3 scripts/stage3_ar1_test.py --realized-col cpr_upb
