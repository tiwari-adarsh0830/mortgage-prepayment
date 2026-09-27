#!/bin/bash
#SBATCH --job-name=score_decwin_2020_s2026
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2020_s2026_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2020_s2026_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Dec-2020 window scoring (MATCHED CONTROL) | seed=2026 | frozen hazard (no --time_varying)"
echo "Checkpoint: outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33_seed2026_goldenbackup/hazard_best.pt"
echo "This is the GOLDEN_BACKUP retrain (job 18606772) that fixes the seed2026 POSTFIX_MISMATCH --"
echo "the earlier seed2026 checkpoint trained on post-fix data while seed42/7/123 trained pre-fix;"
echo "that mismatched checkpoint/outputs dir is renamed ..._seed2026_POSTFIX_MISMATCH and must never"
echo "be referenced. map_era=prefix to match seed42/7/123 (pre-fix property_type_enc codes {0,1,2,3})."
echo "DEPENDS on run_build_raw_cache_2020_control.sbatch -- reuses the seed-independent cached pass."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2020 \
    --seq_dir data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP \
    --ckpt_path outputs/rolling/cutoff_2020_multiobs_k5_h1_ipw_f0.2_L33_seed2026_goldenbackup/hazard_best.pt \
    --map_era prefix \
    --seed_label seed2026 \
    --out_dir outputs/rolling/dec_window_cutoff_2020_control_seed2026

echo "Done: Dec-2020 window scoring (control) seed=2026 | $(date)"
