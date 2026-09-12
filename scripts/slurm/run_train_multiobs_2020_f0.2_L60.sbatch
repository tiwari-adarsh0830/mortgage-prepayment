#!/bin/bash
#SBATCH --job-name=train_f0.2_L60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L60_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L60_%j.err

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Multiobs train | cutoff=2020 h=1 | fixed_fraction f=0.2 | max_seq_len=60 | 50 epochs | --use_ipw"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py \
    --cutoff_year 2020 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 60 \
    --label_horizon 1 \
    --n_epochs 50 \
    --use_ipw \
    --run_tag _f0.2_L60

echo "Done: cutoff=2020 multiobs f0.2 h=1 L=60 --use_ipw | $(date)"
