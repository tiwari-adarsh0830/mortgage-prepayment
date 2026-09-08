#!/bin/bash
#SBATCH --job-name=train_multiobs_ipw_2020
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_multiobs_ipw_2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_multiobs_ipw_2020_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Multiobs train | cutoff=2020 k=5 h=1 | --use_ipw (per-sample loss weighted by train_incl_prob)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py --cutoff_year 2020 --k_draws 5 --label_horizon 1 --use_ipw

echo "Done: cutoff=2020 multiobs k=5 h=1 --use_ipw | $(date)"
