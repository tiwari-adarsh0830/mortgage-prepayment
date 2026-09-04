#!/bin/bash
#SBATCH --job-name=diag_multiobs_1v
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/diag_multiobs_1v_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/diag_multiobs_1v_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
python /scratch/at7095/mortgage_prepayment/scripts/diag/diag_multiobs_onevintage.py
echo "Job finished: $(date)"
