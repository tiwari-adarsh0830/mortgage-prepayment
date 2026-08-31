#!/bin/bash
#SBATCH --job-name=diag_feature_ranges
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/diag_feature_ranges_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/diag_feature_ranges_%j.err
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "Feature-range diagnostic | $(date)"
python $BASE/scripts/diag_feature_ranges.py
echo "Done | $(date)"
