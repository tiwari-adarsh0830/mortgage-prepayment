#!/bin/bash
#SBATCH --job-name=diag_equity
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/diag_equity_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/diag_equity_%j.err

# This diagnostic runs on the EXISTING production model (outputs/hazard_best.pt).
# No retraining needed. Runtime ~5-15 minutes on CPU.
# Submit this NOW, before the rolling pipeline — results go to Gupta.

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

mkdir -p $BASE/logs

echo "Equity x Incentive Diagnostic | $(date)"

python $BASE/scripts/diag_equity_incentive.py \
    --model_path  $BASE/outputs/hazard_best.pt \
    --scaler_path $BASE/data/sequences/scaler.pkl \
    --out_dir     $BASE/outputs

echo "Done | $(date)"
echo "Output: $BASE/outputs/diag_equity_incentive.png"
