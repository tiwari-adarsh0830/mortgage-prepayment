#!/bin/bash
#SBATCH --job-name=prepay_counts_30y
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cs
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/prepay_counts_30y_%j.log

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python3 -u scripts/count_prepay_events_pre2013.py \
    --ckpt outputs/prepay_event_counts_ckpt_30y.pkl \
    --final outputs/prepay_event_counts_pre2013_30y.csv
