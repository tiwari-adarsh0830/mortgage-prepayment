#!/bin/bash
#SBATCH --job-name=prep_rolling
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/prep_rolling_%A_%a.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/prep_rolling_%A_%a.err
#SBATCH --mail-type=FAIL
#
# CPU sequence builder for the next two rolling cutoffs.
#   --array=0-1%1 runs the two cutoffs SERIALLY (one at a time) so two 96G
#   tasks don't stack against QOSMaxMemoryPerUser.
#   --time=8:00:00 is the full-prep default (Pass 1+2+3+4 ~ 5-6h). If a task
#   times out, just resubmit: the shard + split + scaler resume guards in
#   prepare_sequences_rolling.py make it restart from the last completed pass.
#   No --gres: this is CPU-only; requesting a GPU would waste an allocation.

YEARS=(2022 2023)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID | Prep cutoff: $YEAR"
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

python $BASE/scripts/prepare_sequences_rolling.py --cutoff_year $YEAR

echo "Done: prep cutoff=$YEAR | $(date)"
