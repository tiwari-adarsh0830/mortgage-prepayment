#!/bin/bash
#SBATCH --job-name=prep_zbc_2020
#SBATCH --account=torch_pr_932_general
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/prep_zbc_2020_%j.log

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python -u scripts/prepare_sequences_rolling_zbc.py --cutoff_year 2020 --sample_frac 0.3
