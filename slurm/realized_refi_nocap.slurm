#!/bin/bash
#SBATCH --job-name=refi_nocap
#SBATCH --account=torch_pr_932_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/slurm/realized_refi_nocap_%j.out

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env

cd /scratch/at7095/mortgage_prepayment
python3 scripts/realized_cpr_by_refi_v1.py --max-age 200 --cohorts pre2020 --tag _nocap
