#!/bin/bash
#SBATCH --job-name=train_f0.2_L33_2002_hist_ep10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2002_hist_ep10_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/train_f0.2_L33_2002_hist_ep10_%j.err

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Multiobs train | cutoff=2002 h=1 | fixed_fraction f=0.2 | max_seq_len=33 | --include_pre2013 | 50 epochs | --use_ipw"
echo "10-epoch replicate of seed42 for epochs-vs-calibration check (distinct run_tag _f0.2_L33_hist_ep10)"
echo "Checkpointing on (default --ckpt_every 10) -- train_ckpt.pt resumes automatically if this run is killed and resubmitted"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/train_hazard_multiobs.py \
    --cutoff_year 2002 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 33 \
    --label_horizon 1 \
    --n_epochs 10 \
    --use_ipw \
    --include_pre2013 \
    --run_tag _f0.2_L33_hist_ep10

echo "Done: cutoff=2002 multiobs f0.2 h=1 L=33 --include_pre2013 --use_ipw | $(date)"
