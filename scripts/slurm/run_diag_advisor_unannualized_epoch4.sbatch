#!/bin/bash
#SBATCH --job-name=diag_unannualized_e4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/diag_unannualized_e4_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/diag_unannualized_e4_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Un-annualized coupon comparison, epoch-4 seedcheck_a checkpoint"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/diag/diag_advisor_unannualized_epoch4.py

echo "Done: $(date)"
