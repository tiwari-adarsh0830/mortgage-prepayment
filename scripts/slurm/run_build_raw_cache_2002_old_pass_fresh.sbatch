#!/bin/bash
#SBATCH --job-name=build_cache_2002_old_pass_fresh
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/build_cache_2002_old_pass_fresh_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/build_cache_2002_old_pass_fresh_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "CPU-only fresh rebuild: cutoff_2002 OLD test split, under the CURRENT loader"
echo "(bypasses the stale Sep-24 cache under CACHE_VERSION=v2_upb, which predates"
echo "bb8dece's post-mod/term-filter fix -- forces a genuine rescan)"
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2002 \
    --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist \
    --map_era fixed \
    --include_pre2013 \
    --build_cache_only \
    --cache_dir outputs/rolling/_dec_window_raw_cache_old_pass_fresh \
    --out_dir outputs/rolling/_dec_window_build_old_pass_fresh_throwaway

echo "Done: old-pass fresh cache rebuild | $(date)"
