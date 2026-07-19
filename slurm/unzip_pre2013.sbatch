#!/bin/bash
#SBATCH --job-name=unzip_pre2013
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/at7095/mortgage_prepayment/unzip_pre2013_%j.log

cd /scratch/at7095/mortgage_prepayment/data_pre2013_raw
for f in *.zip; do
  base="${f%.zip}"
  if ls "${base}"*.txt "${base}"*.csv >/dev/null 2>&1; then
    echo "skip $f (already unzipped)"
    continue
  fi
  echo "unzipping $f"
  unzip -o "$f"
done
