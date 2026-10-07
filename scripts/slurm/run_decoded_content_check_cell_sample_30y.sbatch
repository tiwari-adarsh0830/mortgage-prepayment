#!/bin/bash
#SBATCH --job-name=decoded_check_30y
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/decoded_content_check_cell_sample_30y_%j.log

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python3 -u scripts/diag/decoded_content_check_cell_sample_30y.py
