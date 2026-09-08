#!/bin/bash
#SBATCH --job-name=train_ipw_seedcheck_a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_ipw_seedcheck_a_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_ipw_seedcheck_a_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Seed-determinism check ROUND 2 (cudnn.deterministic + use_deterministic_algorithms), run A | cutoff=2020 k=5 h=1 --use_ipw, seed=42 (default)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py --cutoff_year 2020 --k_draws 5 --label_horizon 1 --use_ipw --run_tag _seedcheck_a

echo "Done: seedcheck run A | $(date)"
