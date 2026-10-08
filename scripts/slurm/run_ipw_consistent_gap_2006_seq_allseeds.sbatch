#!/bin/bash
#SBATCH --job-name=ipw_gap_2006_seq_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2006_seq_allseeds_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2006_seq_allseeds_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

for SEED in 42 7 123 1001 2026 3 11 77 314 999; do
    echo "=== ipw_consistent_gap cutoff_2006 _seq seed=$SEED hazard_best.pt | $(date)"
    python -u scripts/diag/ipw_consistent_gap.py \
        --cutoff 2006 \
        --seq_dir data/sequences_rolling/cutoff_2006_zbc_multiobs_f0.2_h1_hist_seq \
        --ckpt_dir outputs/rolling/cutoff_2006_multiobs_k5_h1_ipw_seq_s$SEED
done
