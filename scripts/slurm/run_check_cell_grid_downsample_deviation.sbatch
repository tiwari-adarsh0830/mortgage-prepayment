#!/bin/bash
#SBATCH --job-name=check_downsample_dev
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cs
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/check_downsample_dev_%j.log

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python -u scripts/diag/check_cell_grid_downsample_deviation_pre2013.py
