#!/bin/bash
#SBATCH --job-name=realized_cpr_v4
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/cpr_v4_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/cpr_v4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_932_general

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

cd /scratch/at7095/mortgage_prepayment
mkdir -p logs
python scripts/realized_cpr_v4.py
