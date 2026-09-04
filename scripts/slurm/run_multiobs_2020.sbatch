#!/bin/bash
#SBATCH --job-name=run_multiobs_2020
#SBATCH --account=torch_pr_932_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/run_multiobs_2020_%j.out
#SBATCH --error=logs/run_multiobs_2020_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/prepare_sequences_multiobs_zbc.py \
    --cutoff_year 2020 \
    --k_draws 5 \
    --label_horizon 1 \
    --draw_scheme uniform \
    --reuse_from data/sequences_rolling/cutoff_2020_zbc_trail

echo "Job finished: $(date)"
