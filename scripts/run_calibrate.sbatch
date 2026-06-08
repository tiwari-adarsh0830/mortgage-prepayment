#!/bin/bash
#SBATCH --job-name=calib
#SBATCH --account=torch_pr_932_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/calib_%j.out
#SBATCH --error=logs/calib_%j.err
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python3 scripts/calibrate_hazard.py
