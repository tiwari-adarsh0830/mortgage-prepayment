#!/bin/bash
#SBATCH --job-name=oas_engine
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/oas_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/oas_%j.err

source /share/apps/anaconda3/2025.06/bin/activate /scratch/at7095/conda_envs/mortgage_env

python /scratch/at7095/mortgage_prepayment/scripts/oas_engine.py
