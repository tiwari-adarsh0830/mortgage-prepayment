#!/bin/bash
#SBATCH --job-name=hazard_pre2020
#SBATCH --account=torch_pr_932_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/hazard_pre2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/hazard_pre2020_%j.err

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

cd /scratch/at7095/mortgage_prepayment

python scripts/train_hazard_pre2020.py
