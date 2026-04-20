#!/bin/bash
#SBATCH --job-name=mortgage_train
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_%j.err

source /share/apps/anaconda3/2025.06/bin/activate /scratch/at7095/conda_envs/mortgage_env

python /scratch/at7095/mortgage_prepayment/train.py
