#!/bin/bash
#SBATCH --job-name=census_panel_baseline_2020
#SBATCH --account=torch_pr_932_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/census_panel_baseline_2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/census_panel_baseline_2020_%j.err

export CLAUDE_CODE_TMPDIR=/scratch/at7095/mortgage_prepayment/.claude_tmp

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/diag/census_panel_baseline.py --cutoff_year 2020

echo "Job finished: $(date)"
