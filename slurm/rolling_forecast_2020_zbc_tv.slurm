#!/bin/bash
#SBATCH --job-name=rolling_forecast_2020_zbc_tv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/rolling_forecast_2020_zbc_tv_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/rolling_forecast_2020_zbc_tv_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "=========================================="
echo "Cutoff: 2020 (zbc-corrected labels)  forecasts Jan-Dec 2021"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python $BASE/scripts/forecast_rolling_cpr.py \
    --cutoff_year 2020 \
    --label_suffix _zbc \
    --time_varying

echo "Done: cutoff=2020_zbc forecast=2021 | $(date)"
