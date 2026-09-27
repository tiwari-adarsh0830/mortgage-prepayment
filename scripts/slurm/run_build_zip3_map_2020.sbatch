#!/bin/bash
#SBATCH --job-name=zip3_map_2020
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/zip3_map_2020_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/zip3_map_2020_%j.err

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "CPU-only build: loan_id -> zip3 map for cutoff_2020 control test population"
echo "(mirrors build_zip3_map_2002.py; own cache_dir/schema tag, does not touch"
echo "the production dec-window cache)"
echo "Node: $SLURMD_NODENAME | $(date)"
echo "=========================================="

python -u scripts/build_zip3_map_2020.py

echo "Done: cutoff_2020 zip3 map build | $(date)"
