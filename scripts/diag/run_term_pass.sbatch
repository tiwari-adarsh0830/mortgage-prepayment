#!/bin/bash
#SBATCH --job-name=term_pass
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --array=1-93%30
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/at7095/mortgage_prepayment/.claude_tmp/term_pass/logs/tp_%A_%a.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/.claude_tmp/term_pass/logs/tp_%A_%a.err

set -euo pipefail
BASE=/scratch/at7095/mortgage_prepayment
cd "$BASE"

CSV=$(sed -n "${SLURM_ARRAY_TASK_ID}p" .claude_tmp/term_pass/manifest.txt)
VINT=$(basename "$CSV" .csv)
OUT=.claude_tmp/term_pass/parquet/${VINT}.parquet

echo "task ${SLURM_ARRAY_TASK_ID} ${VINT} start=$(date -Is)"
python3 scripts/diag/term_pass_worker.py "$CSV" "$OUT"
echo "done ${VINT} rc=$? end=$(date -Is)"
