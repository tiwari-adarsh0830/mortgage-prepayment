#!/bin/bash
#SBATCH --job-name=hedge_srcB
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/hedge_srcB_%j.out
#SBATCH --error=logs/hedge_srcB_%j.err
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python -u scripts/model_hedge_krd.py --pmms-key 10yr --spanning --floor-mode pinned-fixed --shock-source daily
