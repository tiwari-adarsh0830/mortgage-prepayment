#!/bin/bash
#SBATCH --job-name=multiobs_2002_f0.2_L33_hist
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/multiobs_2002_f0.2_L33_hist_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/multiobs_2002_f0.2_L33_hist_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/prepare_sequences_multiobs_zbc.py \
    --cutoff_year 2002 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 33 \
    --label_horizon 1 \
    --draw_scheme uniform \
    --include_pre2013

echo "Job finished: $(date)"
