#!/bin/bash
#SBATCH --job-name=compare_seq_30y_2002
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/compare_seq_30y_vs_seq_2002_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/compare_seq_30y_vs_seq_2002_%j.err
set -euo pipefail
export CLAUDE_CODE_TMPDIR=/scratch/at7095/mortgage_prepayment/.claude_tmp
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh && conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python -u scripts/diag/compare_seq_30y_vs_seq_2002.py
