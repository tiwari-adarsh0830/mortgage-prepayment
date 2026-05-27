#!/bin/bash
#SBATCH --job-name=risk_neutral
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/risk_neutral_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/risk_neutral_%j.err

source /share/apps/anaconda3/2025.06/bin/activate /scratch/at7095/conda_envs/mortgage_env
python /scratch/at7095/mortgage_prepayment/scripts/risk_neutral_rates.py
