#!/bin/bash
#SBATCH --job-name=synthetic_paths
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/synthetic_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/synthetic_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
pip install scipy --quiet
python /scratch/at7095/mortgage_prepayment/scripts/synthetic_rate_paths.py
echo "Job finished: $(date)"
