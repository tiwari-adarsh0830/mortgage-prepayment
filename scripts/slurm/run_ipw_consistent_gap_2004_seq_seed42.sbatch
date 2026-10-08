#!/bin/bash
#SBATCH --job-name=ipw_gap_2004_seq_s42
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2004_seq_s42_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/ipw_gap_2004_seq_s42_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "ipw_consistent_gap.py | cutoff_2004 _seq | seed=42 | $(date)"

python -u scripts/diag/ipw_consistent_gap.py \
    --cutoff 2004 \
    --seq_dir data/sequences_rolling/cutoff_2004_zbc_multiobs_f0.2_h1_hist_seq \
    --ckpt_dir outputs/rolling/cutoff_2004_multiobs_k5_h1_ipw_seq_s42

echo "Done: ipw_consistent_gap cutoff_2004 _seq seed=42 | $(date)"
