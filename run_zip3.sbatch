#!/bin/bash
#SBATCH --job-name=zip3_models
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/zip3_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/zip3_%j.err

mkdir -p /scratch/at7095/mortgage_prepayment/logs

source /share/apps/anaconda3/2021.11/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

cd /scratch/at7095/mortgage_prepayment

python train_zip3.py
