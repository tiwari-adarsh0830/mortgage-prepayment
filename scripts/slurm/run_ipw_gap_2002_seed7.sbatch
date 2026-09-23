#!/bin/bash
#SBATCH --job-name=ipwgap_2002_seed7
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/ipwgap_2002_seed7_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/ipwgap_2002_seed7_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

python -u scripts/diag/ipw_gap_feature_breakdown.py --cutoff 2002 --split test \
    --out_dir /scratch/at7095/mortgage_prepayment/outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed7

echo "Job finished: $(date)"
