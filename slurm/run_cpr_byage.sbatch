#!/bin/bash
#SBATCH --job-name=cpr_byage
#SBATCH --account=torch_pr_932_general
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/cpr_byage_%j.out
#SBATCH --error=logs/cpr_byage_%j.err

# NOT cpu_short: this reads all 41 raw files (409G). The prior full run of
# this scan used --time=20:00:00 --mem=96G. Pass 0 is skipped here (its
# checkpoint exists and does not depend on age), so 12h with per-file
# Pass 1 checkpointing. If it times out, resubmit -- it resumes.

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment
python -u scripts/realized_cpr_v6_upb_byage.py
