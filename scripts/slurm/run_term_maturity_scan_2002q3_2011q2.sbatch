#!/bin/bash
#SBATCH --job-name=term_mat_scan
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cs
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/term_mat_scan_%j.log

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python -u scripts/diag/term_maturity_scan_2002q3_2011q2.py
