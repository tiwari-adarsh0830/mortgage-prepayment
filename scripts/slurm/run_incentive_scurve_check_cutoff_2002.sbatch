#!/bin/bash
#SBATCH --job-name=scurve_check_2002
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/scurve_check_2002_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/scurve_check_2002_%j.err

export CLAUDE_CODE_TMPDIR=/scratch/at7095/mortgage_prepayment/.claude_tmp

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/diag/incentive_scurve_check_cutoff_2002.py

echo "Job finished: $(date)"
