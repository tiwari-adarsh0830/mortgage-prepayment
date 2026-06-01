#!/bin/bash
#SBATCH --job-name=ddpm_cond
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/ddpm_cond_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/ddpm_cond_%j.err

source /share/apps/anaconda3/2025.06/bin/activate /scratch/at7095/conda_envs/mortgage_env
python /scratch/at7095/mortgage_prepayment/scripts/train_ddpm_conditional.py
