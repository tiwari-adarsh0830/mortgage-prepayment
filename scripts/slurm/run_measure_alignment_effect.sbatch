#!/bin/bash
#SBATCH --job-name=measure_alignment_effect
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/measure_alignment_effect_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/measure_alignment_effect_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Measuring the condition-4 alignment effect: cutoff_2020 f0.2/L33/IPW seed42 checkpoint,"
echo "scored LEFT-aligned (TRAIL_SEQ_DIR, forecast_matched_population_cpr.py's existing path)"
echo "vs RIGHT-aligned (new Dec-2020 control build, pre-fix maps, GOLDEN_BACKUP scaler),"
echo "on the loan intersection, split by full (33mo) vs partial history."
echo "See docs/mistakes_and_lessons.md Sec 8 (2026-09-24 entry) for the underlying finding."
echo "DEPENDS on run_build_raw_cache_2020_control.sbatch -- submit via submit_dec_window_pipeline.sh."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/measure_alignment_effect.py \
    --out_dir outputs/rolling/alignment_effect_cutoff_2020_seed42

echo "Done: alignment effect measurement | $(date)"
