#!/bin/bash
#SBATCH --job-name=build_cache_decwin_2002
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/build_cache_decwin_2002_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/build_cache_decwin_2002_%j.err

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "CPU-only build: cutoff_2002 Dec-window raw-pass cache (map_era=fixed, --include_pre2013)"
echo "Per-vintage checkpointed (scripts/score_multiobs_dec_window.py::build_combined_pass) --"
echo "safe to resubmit after a timeout/kill, already-written vintages are skipped."
echo "seed42/seed7 scoring jobs depend on this job (sbatch --dependency=afterok)."
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2002 \
    --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist \
    --map_era fixed \
    --include_pre2013 \
    --build_cache_only \
    --out_dir outputs/rolling/dec_window_cutoff_2002_build

echo "Done: cutoff_2002 raw-pass cache build | $(date)"
