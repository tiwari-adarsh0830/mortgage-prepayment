#!/bin/bash
#SBATCH --job-name=rolling_forecast
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/rolling_forecast_%A_%a.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/rolling_forecast_%A_%a.err
#SBATCH --mail-type=FAIL

YEARS=(2016 2017 2018 2019 2020 2021 2022 2023)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID | Cutoff: $YEAR  (forecasts Jan-Dec $((YEAR+1)))"
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

python $BASE/scripts/forecast_rolling_cpr.py \
    --cutoff_year $YEAR

echo "Done: cutoff=$YEAR forecast=$((YEAR+1)) | $(date)"
