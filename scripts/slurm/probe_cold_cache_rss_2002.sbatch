#!/bin/bash
#SBATCH --job-name=probe_cold_rss_2002
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --output=/scratch/at7095/mortgage_prepayment/logs/probe_cold_rss_2002_%j.out
#SBATCH --error=/scratch/at7095/mortgage_prepayment/logs/probe_cold_rss_2002_%j.err
set -euo pipefail

BASE=/scratch/at7095/mortgage_prepayment
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/at7095/conda_envs/mortgage_env
cd $BASE

# Cold 2002 combined-pass build into a SCRATCH cache dir (the real cache is not
# touched). ru_maxrss is the process's own peak resident set (no page cache),
# to compare with sacct MaxRSS, which sat exactly at the request on every cold run.
CACHE=$BASE/.claude_tmp/rss_probe_cache_2002
rm -rf "$CACHE"; mkdir -p "$CACHE"
python - <<'PY'
import resource, subprocess, sys
r = subprocess.run([sys.executable, '-u', 'scripts/score_multiobs_dec_window.py',
    '--cutoff_year', '2002', '--seq_dir', 'data/sequences_rolling/cutoff_2002_zbc_multiobs_f0.2_h1_hist_seq',
    '--map_era', 'fixed', '--include_pre2013', '--cell_sample', 'outputs/pre2013_cell_sample_30y_loans.csv',
    '--out_dir', '.claude_tmp/rss_probe_out_2002', '--cache_dir', '.claude_tmp/rss_probe_cache_2002',
    '--build_cache_only'])
print('child rc', r.returncode)
print('RUSAGE_CHILDREN ru_maxrss KB =', resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
sys.exit(r.returncode)
PY
