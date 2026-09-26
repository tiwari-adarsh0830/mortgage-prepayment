#!/bin/bash
#SBATCH --job-name=score_decwin_2020_s42
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2020_s42_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2020_s42_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Dec-2020 window scoring (MATCHED CONTROL) | seed=42 | frozen hazard (no --time_varying)"
echo "Checkpoint: outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33/hazard_best.pt"
echo "map_era=prefix -- this checkpoint (2026-09-11) predates the CP/U fix (commit 2a5b283,"
echo "2026-09-19); GOLDEN_BACKUP confirms it trained on pre-fix category codes (property_type_enc"
echo "codes {0,1,2,3}, no CP=4). Do NOT use map_era=fixed for this checkpoint."
echo "seq_dir points at the GOLDEN_BACKUP copy -- the live cutoff_2020_zbc_multiobs_f0.2_h1 dir"
echo "was overwritten post-fix by job 18022825 on 2026-09-19 (test_seq.npy mtime 13:28, same day"
echo "as the fix, AFTER this checkpoint trained) but its scaler.pkl is byte-identical either way."
echo "DEPENDS on run_build_raw_cache_2020_control.sbatch -- submit via submit_dec_window_pipeline.sh."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2020 \
    --seq_dir data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP \
    --ckpt_path outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33/hazard_best.pt \
    --map_era prefix \
    --seed_label seed42 \
    --out_dir outputs/rolling/dec_window_cutoff_2020_control_seed42

echo "Done: Dec-2020 window scoring (control) seed=42 | $(date)"
