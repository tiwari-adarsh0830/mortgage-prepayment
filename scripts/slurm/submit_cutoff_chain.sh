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
# CPU_PARTITION: override for the census/build submissions only (both long,
# CPU-only, memory-heavy jobs). Gate and ensemble already use cpu_short
# unconditionally -- they're short/light regardless of which partition the
# census/build ran on. GPU stages (train/dec-window/one-step) are untouched.
# Default "cs" preserves every existing invocation's behavior.
CPU_PARTITION="${CPU_PARTITION:-cs}"
# cpu_short carries QoS cpu_short, which caps MaxWall at 6:00:00 (confirmed via
# `sacctmgr show qos cpu_short` -- found 2026-10-05 when the plain partition
# swap alone made sbatch reject both jobs with "CPU job setup is not valid").
# The "cs" --time requests below (8h/12h) are safety margins, not real
# requirements -- actual runtimes are ~18min (census) and ~2:45 (build, see
# the cutoff_2020 _30y build, job 19067719 sacct record) -- so 5:45:00 still
# leaves comfortable headroom under the 6h cap.
if [[ "$CPU_PARTITION" == "cpu_short" ]]; then
    CENSUS_TIME=5:45:00
    BUILD_TIME=5:45:00
else
    CENSUS_TIME=8:00:00
    BUILD_TIME=12:00:00
fi
CONDA_INIT='source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh && conda activate /scratch/at7095/conda_envs/mortgage_env'
ENV_EXPORTS='export CLAUDE_CODE_TMPDIR=/scratch/at7095/mortgage_prepayment/.claude_tmp'

SEEDS=(42 7 123 1001 2026 3 11 77 314 999)   # advisor's Oct 4 decision: 10 seeds/cutoff
SEEDS_CSV=$(IFS=,; echo "${SEEDS[*]}")

# SCHEMA_SMOKE: 1-epoch seed-42 train + test_train_forecast_consistency.py
# (ad hoc mode) between gate and the 10 real trainings. Default ON. Added
# 2026-10-06 after all 10 decwin_2002_seq jobs failed on
# KeyError: "['harp_eligible'] not in index" -- the 10 real trainings (each
# ~2h) had already completed by the time the schema mismatch surfaced, at
# the dec-window stage. A 1-epoch smoke train (~minutes) run through the
# SAME consistency test catches a schema divergence between the train and
# forecast paths before burning 10 real GPU trainings on it. Set
# SCHEMA_SMOKE=0 to skip (not recommended; no known case where this is
# correct other than re-running an already-validated schema unchanged).
SCHEMA_SMOKE="${SCHEMA_SMOKE:-1}"

CELL_SAMPLE="$BASE/outputs/pre2013_cell_sample_30y_loans.csv"
BUILD_DIR="$BASE/data/sequences_rolling/cutoff_${YEAR}_zbc_multiobs_f0.2_h1_hist_seq"
CENSUS_JSON="$BASE/outputs/census_panel_baseline_cutoff_${YEAR}_seq.json"

# START_AT: "census" (default, full chain) or "smoketest" (resume: skip
# census/build/gate/smoke-train, which must already have completed, and submit
# the consistency test with no dependency plus everything after it). Added
# 2026-10-07 after the 2003-2005 chains died at the smoke test's old 0:30:00
# limit with census/build/gate/smoke-train already done.
START_AT="${START_AT:-census}"
if [[ "$START_AT" != "census" && "$START_AT" != "smoketest" ]]; then
    echo "START_AT must be census or smoketest, got: $START_AT" >&2
    exit 1
fi
if [[ "$START_AT" == "smoketest" ]]; then
    if [[ "${SCHEMA_SMOKE}" != "1" ]]; then
        echo "START_AT=smoketest requires SCHEMA_SMOKE=1" >&2
        exit 1
    fi
    if [[ ! -d "$BUILD_DIR" ]]; then
        echo "START_AT=smoketest: build dir missing: $BUILD_DIR" >&2
        exit 1
    fi
    GATE_LOG=$(ls -t "$LOGDIR"/gate_"${YEAR}"_seq_*.out 2>/dev/null | head -1 || true)
    if [[ -z "$GATE_LOG" ]] || ! grep -q "ALL CHECKS PASSED" "$GATE_LOG"; then
        echo "START_AT=smoketest: no gate log with ALL CHECKS PASSED for $YEAR (newest: ${GATE_LOG:-none})" >&2
        exit 1
    fi
    SMOKE_CKPT_CHECK="$BASE/outputs/rolling/cutoff_${YEAR}_multiobs_k5_h1_ipw_seq_smoke_s42/hazard_best.pt"
    if [[ ! -f "$SMOKE_CKPT_CHECK" ]]; then
        echo "START_AT=smoketest: smoke checkpoint missing: $SMOKE_CKPT_CHECK" >&2
        exit 1
    fi
    echo "START_AT=smoketest: build dir, gate log ($GATE_LOG) and smoke checkpoint verified" >&2
fi

# SMOKE_TEST_TIME: the consistency test builds the dec-window raw cache
# (build_combined_pass) cold when it runs first in the chain. The old 0:30:00
# was only ever validated against a hand-warmed cache (2002 _seq) and timed
# out on all three 2003-2005 chains. Memory 96G: both cold-path runs sat at
# their cap (prewarm 2002 64G, smoke tests 40G).
SMOKE_TEST_TIME="${SMOKE_TEST_TIME:-3:00:00}"
# SMOKE_TEST_MEM: added 2026-10-08. The 2004/2005 smoke tests (19388334,
# 19388410) peaked at MaxRSS 100,659,604K / 100,659,296K -- at their 96G cap.
# 2006+ caches cover more vintages; QoS cpu_short has no per-job memory limit
# but caps each user at mem=120G (MaxTRESPU), so 120G is the per-job ceiling.
SMOKE_TEST_MEM="${SMOKE_TEST_MEM:-96G}"
# SMOKE_PARTITION: added 2026-10-08. Default cpu_short preserves every existing
# invocation. cpu_short caps a user at mem=120G (MaxTRESPU), and the 2007
# smoke test (19413412) peaked at MaxRSS 125,824,956K against its 120G request,
# so a cutoff needing more than 120G must go to a partition under QoS cpu48
# (per-user mem=6000G, MaxWall 2-00:00:00), e.g. SMOKE_PARTITION=cs.
SMOKE_PARTITION="${SMOKE_PARTITION:-cpu_short}"
# BUILD_MEM: added 2026-10-08. Memory for the build job only (census stays
# 96G). Default 96G preserves every existing invocation. The 2003-2008 builds
# all peaked at their 96G cap (MaxRSS ~100,659,000K), so the true need is unobserved.
BUILD_MEM="${BUILD_MEM:-96G}"

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

if [[ "$START_AT" == "census" ]]; then
# ── 1. Census baseline ───────────────────────────────────────────────────────
CENSUS_JOBNAME="census_${YEAR}_seq"
CENSUS_JOBID=$(sbatch --parsable \
    --job-name="$CENSUS_JOBNAME" --account="$ACCOUNT" --partition="$CPU_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --time="$CENSUS_TIME" \
    --output="$LOGDIR/${CENSUS_JOBNAME}_%j.out" --error="$LOGDIR/${CENSUS_JOBNAME}_%j.err" \
    --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/diag/census_panel_baseline.py \
    --cutoff_year $YEAR --include_pre2013 --cell_sample $CELL_SAMPLE --out_tag _seq")
echo "census:      job $CENSUS_JOBID"

# ── 2. Build sequences ───────────────────────────────────────────────────────
BUILD_JOBNAME="multiobs_${YEAR}_seq"
BUILD_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$CENSUS_JOBID" \
    --job-name="$BUILD_JOBNAME" --account="$ACCOUNT" --partition="$CPU_PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem="$BUILD_MEM" --time="$BUILD_TIME" \
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
fi


# ── 3b. Schema smoke: 1-epoch seed-42 train + consistency test ──────────────
FIRST_TRAIN_DEP="${GATE_JOBID:-}"
if [[ "$SCHEMA_SMOKE" == "1" ]]; then
    SMOKE_TEST_DEP_ARGS=()
    if [[ "$START_AT" == "census" ]]; then
    SMOKE_TRAIN_JOBNAME="train_${YEAR}_seq_smoke_s42"
    SMOKE_TRAIN_JOBID=$(sbatch --parsable \
        --dependency=afterok:"$GATE_JOBID" \
        --job-name="$SMOKE_TRAIN_JOBNAME" --account="$ACCOUNT" --partition=l40s_public \
        --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=0:30:00 \
        --output="$LOGDIR/${SMOKE_TRAIN_JOBNAME}_%j.out" --error="$LOGDIR/${SMOKE_TRAIN_JOBNAME}_%j.err" \
        --wrap="set -euo pipefail; export CUBLAS_WORKSPACE_CONFIG=:4096:8; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/train_hazard_multiobs.py \
    --cutoff_year $YEAR --sampling_mode fixed_fraction --frac_draws 0.2 \
    --max_seq_len 33 --label_horizon 1 --n_epochs 1 --use_ipw --include_pre2013 \
    --seed 42 --seq_dir $BUILD_DIR --run_tag _seq_smoke_s42 --ckpt_every 10")
    echo "smoke train: job $SMOKE_TRAIN_JOBID"
    SMOKE_TEST_DEP_ARGS=(--dependency=afterok:"$SMOKE_TRAIN_JOBID")
    fi
    SMOKE_CKPT_PATH="$BASE/outputs/rolling/cutoff_${YEAR}_multiobs_k5_h1_ipw_seq_smoke_s42/hazard_best.pt"

    SMOKE_TEST_JOBNAME="smoketest_${YEAR}_seq"
    SMOKE_TEST_JOBID=$(sbatch --parsable \
        ${SMOKE_TEST_DEP_ARGS[@]+"${SMOKE_TEST_DEP_ARGS[@]}"} \
        --job-name="$SMOKE_TEST_JOBNAME" --account="$ACCOUNT" --partition="$SMOKE_PARTITION" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem="$SMOKE_TEST_MEM" --time="$SMOKE_TEST_TIME" \
        --output="$LOGDIR/${SMOKE_TEST_JOBNAME}_%j.out" --error="$LOGDIR/${SMOKE_TEST_JOBNAME}_%j.err" \
        --wrap="set -euo pipefail; $ENV_EXPORTS; $CONDA_INIT; cd $BASE; \
python -u scripts/tests/test_train_forecast_consistency.py \
    --seq_dir $BUILD_DIR --ckpt_path $SMOKE_CKPT_PATH --cutoff_year $YEAR \
    --map_era fixed --include_pre2013 --cell_sample $CELL_SAMPLE")
    echo "smoke test:  job $SMOKE_TEST_JOBID"
    FIRST_TRAIN_DEP="$SMOKE_TEST_JOBID"
fi

# ── 4. Ten seeds: train -> dec-window -> one-step, each seed's own chain ─────
ONESTEP_JOBIDS=()
for SEED in "${SEEDS[@]}"; do
    TRAIN_JOBNAME="train_${YEAR}_seq_s${SEED}"
    TRAIN_JOBID=$(sbatch --parsable \
        --dependency=afterok:"$FIRST_TRAIN_DEP" \
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
echo "census=${CENSUS_JOBID:-skipped} build=${BUILD_JOBID:-skipped} gate=${GATE_JOBID:-skipped} onestep_jobs=[$ONESTEP_DEP] ensemble=$ENSEMBLE_JOBID" >&2
echo "" >&2
echo "NOTE: census_check_seq.py is NOT in this automated chain (it's informational," >&2
echo "not a pass/fail gate) -- run it manually after the build/census jobs finish:" >&2
echo "  python scripts/diag/census_check_seq.py --build_dir $BUILD_DIR --census_json $CENSUS_JSON $DIVIDE_OUT_FLAG" >&2
