#!/bin/bash
# submit_dec_window_pipeline.sh -- submits the CPU build jobs, then the GPU
# scoring/alignment jobs with --dependency=afterok on the matching build job.
# sbatch --dependency needs a real job ID, which only exists at submission
# time -- that's why this is a small wrapper script rather than a hardcoded
# #SBATCH directive in each scoring sbatch file.
#
# Run once approved (dry-run + fixes 1-5 checked first).
set -euo pipefail
cd /scratch/at7095/mortgage_prepayment

BUILD_2002=$(sbatch --parsable scripts/slurm/run_build_raw_cache_2002.sbatch)
BUILD_2020=$(sbatch --parsable scripts/slurm/run_build_raw_cache_2020_control.sbatch)
echo "cutoff_2002 build job: $BUILD_2002"
echo "cutoff_2020 control build job: $BUILD_2020"

sbatch --dependency=afterok:$BUILD_2002 scripts/slurm/run_score_dec_window_2002_seed42.sbatch
sbatch --dependency=afterok:$BUILD_2002 scripts/slurm/run_score_dec_window_2002_seed7.sbatch

sbatch --dependency=afterok:$BUILD_2020 scripts/slurm/run_score_dec_window_2020_control_seed42.sbatch
sbatch --dependency=afterok:$BUILD_2020 scripts/slurm/run_score_dec_window_2020_control_seed7.sbatch
sbatch --dependency=afterok:$BUILD_2020 scripts/slurm/run_score_dec_window_2020_control_seed123.sbatch
sbatch --dependency=afterok:$BUILD_2020 scripts/slurm/run_measure_alignment_effect.sbatch
