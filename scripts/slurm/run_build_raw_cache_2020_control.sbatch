#!/bin/bash
#SBATCH --job-name=build_cache_decwin_2020
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/build_cache_decwin_2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/build_cache_decwin_2020_%j.err

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "CPU-only build: cutoff_2020 MATCHED CONTROL Dec-window raw-pass cache (map_era=prefix)"
echo "Per-vintage checkpointed -- safe to resubmit after a timeout/kill."
echo "seed42/seed7/seed123 control scoring jobs AND measure_alignment_effect.py depend on this"
echo "job (sbatch --dependency=afterok)."
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2020 \
    --seq_dir data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP \
    --map_era prefix \
    --build_cache_only \
    --out_dir outputs/rolling/dec_window_cutoff_2020_control_build

echo "Done: cutoff_2020 control raw-pass cache build | $(date)"
