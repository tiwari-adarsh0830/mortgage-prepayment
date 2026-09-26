#!/bin/bash
#SBATCH --job-name=build_cache_matched_seedcheck_a
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/build_cache_matched_seedcheck_a_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/build_cache_matched_seedcheck_a_%j.err

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "CPU-only build: right-aligned Dec-2020 raw-pass cache for the 365,146-loan matched"
echo "population (cutoff_2020_multiobs_k5_h1_ipw_seedcheck_a checkpoint, map_era=prefix)."
echo "Population-keyed per-vintage + combined cache (fixed 2026-09-25, see"
echo "docs/mistakes_and_lessons.md) -- this is a DIFFERENT population from the f0.2/L33"
echo "control build, so it gets its own cache regardless of any prior cache contents."
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

python -u scripts/score_matched_population_right_aligned.py --build_cache_only

echo "Done: matched-population (seedcheck_a) raw-pass cache build | $(date)"
