#!/bin/bash
# submit_cutoff_chain.sh -- submits the full census -> build -> gate -> 10 seeds
# (train -> dec-window -> one-step each) -> ensemble pipeline for ONE cutoff
# year, under the new schema (origination-row term rule, 10-feature
# harp_eligible column, --include_pre2013 + --sample_frac 0.1 for every
# cutoff). Part 3 of the Oct 5, 2026 prompt -- advisor's Oct 4 decisions:
# yearly sequence 2002->2024, ten seeds per cutoff, no recency weighting.
#
# Does NOT run anything itself beyond `sbatch` submissions -- every actual
# job runs under its own `set -euo pipefail` (via --wrap) and its own
# --account/--partition. This script just prints job ids and exits.
#
# Usage:
#   scripts/slurm/submit_cutoff_chain.sh <cutoff_year>
#
# Example:
#   scripts/slurm/submit_cutoff_chain.sh 2002
#
# Naming (the "_seq" suffix, chosen in Part 3 to avoid any collision with
# existing _30y/_hist/_GOLDEN_BACKUP/etc. directories -- confirmed via
# `find data/sequences_rolling outputs/rolling -iname '*_seq*'`, zero hits
# before this chain existed):
#   build:      data/sequences_rolling/cutoff_<year>_zbc_multiobs_f0.2_h1_hist_seq
#   train:      outputs/rolling/cutoff_<year>_multiobs_k5_h1_ipw_seq_s<seed>
#   dec-window: outputs/rolling/dec_window_cutoff_<year>_seq_seed<seed>
#   one-step:   outputs/rolling/rolling_onestep_cutoff_<year>_seq_seed<seed>
#   ensemble:   outputs/rolling/ensemble_onestep_cutoff_<year>_seq
#   census:     outputs/census_panel_baseline_cutoff_<year>_seq.json
#
# 23 cutoffs (2002..2024) x 10 seeds under this one "_seq" tag cannot collide
# with each other either: every path is keyed by both <year> and <seed>.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <cutoff_year>" >&2
    exit 1
fi
YEAR="$1"
if ! [[ "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "cutoff_year must be a 4-digit year, got: $YEAR" >&2
    exit 1
fi

BASE=/scratch/at7095/mortgage_prepayment
cd "$BASE"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

ACCOUNT=torch_pr_932_general
CONDA_INIT='source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh && conda activate /scratch/at7095/conda_envs/mortgage_env'
ENV_EXPORTS='export CLAUDE_CODE_TMPDIR=/scratch/at7095/mortgage_prepayment/.claude_tmp'

SEEDS=(42 7 123 1001 2026 3 11 77 314 999)   # advisor's Oct 4 decision: 10 seeds/cutoff
SEEDS_CSV=$(IFS=,; echo "${SEEDS[*]}")

CELL_SAMPLE="$BASE/outputs/pre2013_cell_sample_30y_loans.csv"
BUILD_DIR="$BASE/data/sequences_rolling/cutoff_${YEAR}_zbc_multiobs_f0.2_h1_hist_seq"
CENSUS_JSON="$BASE/outputs/census_panel_baseline_cutoff_${YEAR}_seq.json"

# cutoffs >= 2013 are the only ones where --sample_frac 0.1 actually fires
# (RELEVANT_VINTAGES excludes every modern vintage for cutoffs <= 2012 --
# see Part 3's inventory), so only those need the census_check_seq.py
# --divide_out_subsample correction.
if (( YEAR >= 2013 )); then
    DIVIDE_OUT_FLAG="--divide_out_subsample"
else
    DIVIDE_OUT_FLAG=""
fi

echo "=== submit_cutoff_chain.sh: cutoff_year=$YEAR seeds=[$SEEDS_CSV] ===" >&2

# ── 1. Census baseline ───────────────────────────────────────────────────────
CENSUS_JOBNAME="census_${YEAR}_seq"
CENSUS_JOBID=$(sbatch --parsable \
    --job-name="$CENSUS_JOBNAME" --account="$ACCOUNT" --partition=cs \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=8:00:00 \
    --output="$LOGDIR/${CENSUS_JOBNAME}_%j.out" --error="$LOGDIR/${CENSUS_JOBNAME}_%j.err" \
    --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/diag/census_panel_baseline.py \
    --cutoff_year $YEAR --include_pre2013 --cell_sample $CELL_SAMPLE --out_tag _seq")
echo "census:      job $CENSUS_JOBID"

# ── 2. Build sequences ───────────────────────────────────────────────────────
BUILD_JOBNAME="multiobs_${YEAR}_seq"
BUILD_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$CENSUS_JOBID" \
    --job-name="$BUILD_JOBNAME" --account="$ACCOUNT" --partition=cs \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=12:00:00 \
    --output="$LOGDIR/${BUILD_JOBNAME}_%j.out" --error="$LOGDIR/${BUILD_JOBNAME}_%j.err" \
    --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/prepare_sequences_multiobs_zbc.py \
    --cutoff_year $YEAR --sampling_mode fixed_fraction --frac_draws 0.2 \
    --max_seq_len 33 --label_horizon 1 --draw_scheme uniform \
    --include_pre2013 --cell_sample $CELL_SAMPLE --sample_frac 0.1 --run_tag _seq")
echo "build:       job $BUILD_JOBID"
BUILD_LOG="$LOGDIR/${BUILD_JOBNAME}_${BUILD_JOBID}.out"

# ── 3. Build gate ────────────────────────────────────────────────────────────
GATE_JOBNAME="gate_${YEAR}_seq"
GATE_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$BUILD_JOBID" \
    --job-name="$GATE_JOBNAME" --account="$ACCOUNT" --partition=cpu_short \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=0:30:00 \
    --output="$LOGDIR/${GATE_JOBNAME}_%j.out" --error="$LOGDIR/${GATE_JOBNAME}_%j.err" \
    --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/diag/check_build_seq.py \
    --build_dir $BUILD_DIR --build_log $BUILD_LOG")
echo "gate:        job $GATE_JOBID"

# ── 4. Ten seeds: train -> dec-window -> one-step, each seed's own chain ─────
ONESTEP_JOBIDS=()
for SEED in "${SEEDS[@]}"; do
    TRAIN_JOBNAME="train_${YEAR}_seq_s${SEED}"
    TRAIN_JOBID=$(sbatch --parsable \
        --dependency=afterok:"$GATE_JOBID" \
        --job-name="$TRAIN_JOBNAME" --account="$ACCOUNT" --partition=l40s_public \
        --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=8:00:00 \
        --output="$LOGDIR/${TRAIN_JOBNAME}_%j.out" --error="$LOGDIR/${TRAIN_JOBNAME}_%j.err" \
        --wrap="set -euo pipefail; export CUBLAS_WORKSPACE_CONFIG=:4096:8; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/train_hazard_multiobs.py \
    --cutoff_year $YEAR --sampling_mode fixed_fraction --frac_draws 0.2 \
    --max_seq_len 33 --label_horizon 1 --n_epochs 50 --use_ipw --include_pre2013 \
    --seed $SEED --seq_dir $BUILD_DIR --run_tag _seq_s${SEED} --ckpt_every 10")
    echo "train s$SEED:   job $TRAIN_JOBID"
    TRAIN_OUT_DIR="$BASE/outputs/rolling/cutoff_${YEAR}_multiobs_k5_h1_ipw_seq_s${SEED}"
    CKPT_PATH="$TRAIN_OUT_DIR/hazard_best.pt"

    DECWIN_JOBNAME="decwin_${YEAR}_seq_s${SEED}"
    DECWIN_OUT_DIR="$BASE/outputs/rolling/dec_window_cutoff_${YEAR}_seq_seed${SEED}"
    DECWIN_JOBID=$(sbatch --parsable \
        --dependency=afterok:"$TRAIN_JOBID" \
        --job-name="$DECWIN_JOBNAME" --account="$ACCOUNT" --partition=l40s_public \
        --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=4:00:00 \
        --output="$LOGDIR/${DECWIN_JOBNAME}_%j.out" --error="$LOGDIR/${DECWIN_JOBNAME}_%j.err" \
        --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; mkdir -p $DECWIN_OUT_DIR; \
python -u scripts/score_multiobs_dec_window.py \
    --cutoff_year $YEAR --seq_dir $BUILD_DIR --ckpt_path $CKPT_PATH \
    --map_era fixed --include_pre2013 --seed_label seed${SEED} \
    --out_dir $DECWIN_OUT_DIR --cell_sample $CELL_SAMPLE")
    echo "decwin s$SEED:  job $DECWIN_JOBID"
    FROZEN_SCORES="$DECWIN_OUT_DIR/dec_window_scores_seed${SEED}.csv"

    ONESTEP_JOBNAME="onestep_${YEAR}_seq_s${SEED}"
    ONESTEP_OUT_DIR="$BASE/outputs/rolling/rolling_onestep_cutoff_${YEAR}_seq_seed${SEED}"
    ONESTEP_JOBID=$(sbatch --parsable \
        --dependency=afterok:"$DECWIN_JOBID" \
        --job-name="$ONESTEP_JOBNAME" --account="$ACCOUNT" --partition=l40s_public \
        --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=4:00:00 \
        --output="$LOGDIR/${ONESTEP_JOBNAME}_%j.out" --error="$LOGDIR/${ONESTEP_JOBNAME}_%j.err" \
        --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; mkdir -p $ONESTEP_OUT_DIR; \
python -u scripts/score_rolling_one_step.py \
    --cutoff_year $YEAR --seq_dir $BUILD_DIR --ckpt_path $CKPT_PATH \
    --map_era fixed --include_pre2013 --seed_label seed${SEED} \
    --frozen_scores $FROZEN_SCORES --out_dir $ONESTEP_OUT_DIR --cell_sample $CELL_SAMPLE")
    echo "onestep s$SEED: job $ONESTEP_JOBID"
    ONESTEP_JOBIDS+=("$ONESTEP_JOBID")
done

# ── 5. Ensemble (depends on ALL 10 one-step jobs) ────────────────────────────
ONESTEP_DEP=$(IFS=:; echo "${ONESTEP_JOBIDS[*]}")
ENSEMBLE_JOBNAME="ensemble_${YEAR}_seq"
ENSEMBLE_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$ONESTEP_DEP" \
    --job-name="$ENSEMBLE_JOBNAME" --account="$ACCOUNT" --partition=cpu_short \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=0:30:00 \
    --output="$LOGDIR/${ENSEMBLE_JOBNAME}_%j.out" --error="$LOGDIR/${ENSEMBLE_JOBNAME}_%j.err" \
    --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/ensemble_onestep_seq.py \
    --cutoff_year $YEAR --seeds $SEEDS_CSV --run_tag _seq")
echo "ensemble:    job $ENSEMBLE_JOBID"

echo "" >&2
echo "=== submit_cutoff_chain.sh done: cutoff_year=$YEAR ===" >&2
echo "census=$CENSUS_JOBID build=$BUILD_JOBID gate=$GATE_JOBID onestep_jobs=[$ONESTEP_DEP] ensemble=$ENSEMBLE_JOBID" >&2
echo "" >&2
echo "NOTE: census_check_seq.py is NOT in this automated chain (it's informational," >&2
echo "not a pass/fail gate) -- run it manually after the build/census jobs finish:" >&2
echo "  python scripts/diag/census_check_seq.py --build_dir $BUILD_DIR --census_json $CENSUS_JSON $DIVIDE_OUT_FLAG" >&2
