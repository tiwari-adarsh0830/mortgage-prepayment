#!/bin/bash
#SBATCH --job-name=train_f0.2_L33_2020_seed2026_gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2020_seed2026_gb_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2020_seed2026_gb_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Multiobs train | cutoff=2020 h=1 | fixed_fraction f=0.2 | max_seq_len=33 | 50 epochs | --use_ipw | seed=2026"
echo "RETRAIN, explicit --seq_dir: the plain cutoff_2020_zbc_multiobs_f0.2_h1 dir was silently"
echo "rewritten with post-fix CP/U encoding by job 18022825 on 2026-09-19, AFTER seed42/7/123"
echo "trained on it (Sep 11-13, pre-fix) and BEFORE this run -- decoded and confirmed 2026-09-26"
echo "(plain dir property_type_enc={0,1,2,3,4}, GOLDEN_BACKUP={0,1,2,3}). Pointed explicitly at"
echo "GOLDEN_BACKUP via --seq_dir so this run matches seed42/7/123's actual training data by content,"
echo "not by a directory name that turned out not to be stable. The prior (mismatched) seed2026 run"
echo "is renamed outputs/rolling/..._seed2026_POSTFIX_MISMATCH and must not be used."
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py \
    --cutoff_year 2020 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 33 \
    --label_horizon 1 \
    --n_epochs 50 \
    --use_ipw \
    --seed 2026 \
    --seq_dir /scratch/at7095/mortgage_prepayment/data/sequences_rolling/cutoff_2020_zbc_multiobs_f0.2_h1_GOLDEN_BACKUP \
    --run_tag _f0.2_L33_seed2026_goldenbackup

echo "Done: cutoff=2020 multiobs f0.2 h=1 L=33 --use_ipw seed=2026 (GOLDEN_BACKUP) | $(date)"
