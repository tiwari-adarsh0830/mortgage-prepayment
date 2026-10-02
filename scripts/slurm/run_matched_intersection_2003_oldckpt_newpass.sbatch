#!/bin/bash
#SBATCH --job-name=mi2003_oldckpt_newpass
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/mi2003_oldckpt_newpass_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/mi2003_oldckpt_newpass_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

echo "=========================================="
echo "Matched-intersection 2003 decomposition | ckpts=old | pass_set=new"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | $(date)"
echo "=========================================="

python -u scripts/score_matched_intersection_2003.py \
    --ckpts old \
    --pass_set new \
    --out_dir outputs/rolling/matched_intersection_2003/oldckpt_newpass

echo "Done: matched_intersection_2003 oldckpt_newpass | $(date)"
