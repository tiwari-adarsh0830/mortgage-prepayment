#!/bin/bash
#SBATCH --job-name=hazard_train
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/hazard_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/hazard_%j.err

source /share/apps/anaconda3/2025.06/bin/activate /scratch/at7095/conda_envs/mortgage_env

python /scratch/at7095/mortgage_prepayment/scripts/train_hazard.py
