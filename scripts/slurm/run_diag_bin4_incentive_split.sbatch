#!/bin/bash
#SBATCH --job-name=diag_bin4_incentive_split
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/diag_bin4_incentive_split_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/diag_bin4_incentive_split_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Bin 4 incentive/label split | cutoff_2020_multiobs_k5_h1"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/diag/diag_bin4_incentive_split.py

echo "Done: $(date)"
