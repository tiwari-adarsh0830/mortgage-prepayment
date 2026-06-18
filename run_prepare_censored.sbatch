#!/bin/bash
#SBATCH --job-name=prepare_pre2020
#SBATCH --account=torch_pr_932_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/prepare_pre2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/prepare_pre2020_%j.err

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

cd /scratch/at7095/mortgage_prepayment

python prepare_sequences_censored2019.py
