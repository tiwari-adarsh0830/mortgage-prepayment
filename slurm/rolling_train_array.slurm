#!/bin/bash
#SBATCH --job-name=rolling_train
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/rolling_train_%A_%a.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/rolling_train_%A_%a.err
#SBATCH --mail-type=FAIL

YEARS=(2016 2017 2018 2019 2020 2021 2022 2023)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID | Cutoff: $YEAR"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python $BASE/scripts/train_hazard_rolling.py \
    --cutoff_year $YEAR \
    --n_epochs 50

echo "Done: cutoff=$YEAR | $(date)"
