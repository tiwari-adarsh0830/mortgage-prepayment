#!/bin/bash
#SBATCH --job-name=rolling_train_2020_trail
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/rolling_train_2020_trail_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/rolling_train_2020_trail_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "=========================================="
echo "Cutoff: 2020 (zbc-corrected labels, TRAILING window)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python $BASE/scripts/train_hazard_rolling.py \
    --cutoff_year 2020 \
    --n_epochs 50 \
    --label_suffix _zbc_trail

echo "Done: cutoff=2020_zbc_trail | $(date)"
