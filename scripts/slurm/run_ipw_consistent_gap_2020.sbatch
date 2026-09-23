#!/bin/bash
#SBATCH --job-name=ipw_gap_2020
#SBATCH --account=torch_pr_932_general
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2020_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/diag/ipw_consistent_gap.py --cutoff 2020

echo "Job finished: $(date)"
