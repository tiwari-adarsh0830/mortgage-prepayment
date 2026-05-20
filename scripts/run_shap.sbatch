#!/bin/bash
#SBATCH --job-name=shap_transformer
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/shap_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/shap_%j.err

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
pip install shap matplotlib --quiet
python /scratch/at7095/mortgage_prepayment/scripts/shap_transformer.py
echo "Job finished: $(date)"
