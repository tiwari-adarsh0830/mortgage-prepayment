#!/bin/bash
#SBATCH --job-name=score_matched_seedcheck_a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/score_matched_seedcheck_a_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/score_matched_seedcheck_a_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Like-for-like correction: cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a, hazard_best.pt"
echo "(epoch4) and hazard_final.pt (epoch50), scored LEFT-aligned (TRAIL_SEQ_DIR, fresh"
echo "reproduction reusing the existing RAW_PASS_CACHE from Sep 5) and RIGHT-aligned (new"
echo "Dec-2020 single-window build, this checkpoint's own scaler + pre-fix CP/U maps),"
echo "on the SAME 365,146-loan matched population."
echo "DEPENDS on run_build_cache_matched_population_seedcheck_a.sbatch."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_matched_population_right_aligned.py

echo "Done: matched-population left-vs-right comparison | $(date)"
