#!/bin/bash
#SBATCH --job-name=prep_zbc_2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/prep_zbc_2022_%j.log
#SBATCH --mail-type=FAIL

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

echo "prep cutoff=2022 (zbc labels) | $(date)"
python $BASE/scripts/prepare_sequences_rolling_zbc.py --cutoff_year 2022
echo "Done | $(date)"
