#!/bin/bash
#SBATCH --job-name=rolling_train_2022_zbc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/rolling_train_2022_zbc_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/rolling_train_2022_zbc_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "=========================================="
echo "Train cutoff: 2022 (zbc-corrected labels)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python $BASE/scripts/train_hazard_rolling.py \
    --cutoff_year 2022 \
    --label_suffix _zbc

echo "Done: train cutoff=2022_zbc | $(date)"
