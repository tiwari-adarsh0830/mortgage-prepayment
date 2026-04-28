#!/bin/bash
#SBATCH --job-name=mortgage_transformer
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/transformer_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/transformer_%j.err

source /share/apps/anaconda3/2025.06/bin/activate /scratch/at7095/conda_envs/mortgage_env

python /scratch/at7095/mortgage_prepayment/train_transformer.py
