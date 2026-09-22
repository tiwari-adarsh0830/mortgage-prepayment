#!/bin/bash
#SBATCH --job-name=calib_check_2020
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/calib_check_2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/calib_check_2020_%j.err

export CLAUDE_CODE_TMPDIR=/scratch/at7095/mortgage_prepayment/.claude_tmp

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/diag/calibration_check_cutoff_2020_ipw_L33.py

echo "Job finished: $(date)"
