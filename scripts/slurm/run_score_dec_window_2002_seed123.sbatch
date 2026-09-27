#!/bin/bash
#SBATCH --job-name=score_decwin_2002_s123
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2002_s123_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/score_decwin_2002_s123_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Dec-2002 window scoring | seed=123 | frozen hazard (no --time_varying)"
echo "Checkpoint: outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed123/hazard_best.pt"
echo "map_era=fixed (trained on the post-CP/U-fix rebuild)"
echo "Reuses the SAME cached combined pass as seed42/7 (population hash is seed-independent)"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year 2002 \
    --seq_dir data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist \
    --ckpt_path outputs/rolling/cutoff_2002_multiobs_k5_h1_ipw_f0.2_L33_hist_seed123/hazard_best.pt \
    --map_era fixed \
    --include_pre2013 \
    --seed_label seed123 \
    --out_dir outputs/rolling/dec_window_cutoff_2002_seed123

echo "Done: Dec-2002 window scoring seed=123 | $(date)"
