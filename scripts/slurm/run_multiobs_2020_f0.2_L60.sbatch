#!/bin/bash
#SBATCH --job-name=multiobs_f0.2_L60
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/multiobs_f0.2_L60_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/multiobs_f0.2_L60_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/prepare_sequences_multiobs_zbc.py \
    --cutoff_year 2020 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 60 \
    --label_horizon 1 \
    --draw_scheme uniform \
    --reuse_from data/sequences_rolling/cutoff_2020_zbc_trail

echo "Job finished: $(date)"
