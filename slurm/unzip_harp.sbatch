#!/bin/bash
#SBATCH --job-name=unzip_harp
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/at7095/mortgage_prepayment/unzip_harp_%j.log

cd /scratch/at7095/mortgage_prepayment/data_harp_raw

# resume guard: skip if the expected uncompressed size is already there
expected_size=25887233311
if [ -f HARPLPPub.csv ]; then
  actual_size=$(stat -c%s HARPLPPub.csv)
  if [ "$actual_size" -eq "$expected_size" ]; then
    echo "HARPLPPub.csv already fully extracted, skipping"
  else
    echo "HARPLPPub.csv exists but wrong size ($actual_size vs $expected_size), re-extracting"
    unzip -o HARP_Files.zip
  fi
else
  unzip -o HARP_Files.zip
fi
