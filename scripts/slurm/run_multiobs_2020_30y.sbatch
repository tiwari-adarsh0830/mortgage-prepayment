#!/bin/bash
#SBATCH --job-name=multiobs_2020_30y
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/multiobs_2020_30y_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/multiobs_2020_30y_%j.err
set -euo pipefail

echo "Job started: $(date)"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd /scratch/at7095/mortgage_prepayment

echo "cutoff_2020 _30y build, under the current (30-year/post-mod-fixed) loader."
echo "FRESH --sample_frac 0.1 draw (NOT --reuse_from cutoff_2020_zbc_trail): the old trail build's"
echo "split was drawn before the term/post-mod fix existed, from the unfiltered population --"
echo "reusing it would carry over non-360/post-mod loans the new loader would otherwise exclude."
echo "--run_tag _30y avoids colliding with the existing (pre-fix) cutoff_2020_zbc_multiobs_f0.2_h1 dir."

python -u scripts/prepare_sequences_multiobs_zbc.py \
    --cutoff_year 2020 \
    --sampling_mode fixed_fraction \
    --frac_draws 0.2 \
    --max_seq_len 33 \
    --label_horizon 1 \
    --draw_scheme uniform \
    --sample_frac 0.1 \
    --run_tag _30y

echo "Job finished: $(date)"
