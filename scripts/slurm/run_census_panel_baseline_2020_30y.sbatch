#!/bin/bash
#SBATCH --job-name=census_2020_30y
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/census_2020_30y_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/census_2020_30y_%j.err
set -euo pipefail

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

echo "cutoff_2020 census, under the current (30-year/post-mod-fixed) loader. No --include_pre2013"
echo "(modern vintages 2013Q1-2023Q1 only); --cell_sample does not apply (gates only PRE2013_VINTAGES)."

python -u scripts/diag/census_panel_baseline.py --cutoff_year 2020 --out_tag _30y

echo "Job finished: $(date)"
