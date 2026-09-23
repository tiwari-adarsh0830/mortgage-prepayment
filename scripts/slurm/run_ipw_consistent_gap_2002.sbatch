#!/bin/bash
#SBATCH --job-name=ipw_gap_2002
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2002_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2002_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/diag/ipw_consistent_gap.py --cutoff 2002

echo "Job finished: $(date)"
