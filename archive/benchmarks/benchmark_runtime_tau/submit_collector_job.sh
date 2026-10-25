#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash benchmark_runtime_tau/submit_collector_job.sh <benchmark_root_path> [job_name] [time_limit] [mem]"
    echo "Example: bash benchmark_runtime_tau/submit_collector_job.sh <V6_DATA_ROOT>/benchmark_runtime_tau/runtime_tau_smoke3_28-03-2026 RTauCollectSmoke3 00:20:00 4GB"
    exit 1
fi

BENCH_ROOT="$1"
JOB_NAME="${2:-RTauCollect}"
TIME_LIMIT="${3:-00:20:00}"
MEMORY="${4:-4GB}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$HOME/.julia_curie}"
ACCOUNT="${SLURM_ACCOUNT:-dror-account}"
PARTITION="${SLURM_PARTITION:-drorq}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

mkdir -p "$BENCH_ROOT"

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --time="$TIME_LIMIT"
    --mem="$MEMORY"
    --output="$BENCH_ROOT/collect_%j.out"
    --error="$BENCH_ROOT/collect_%j.error"
)

if [ -n "$ACCOUNT" ]; then
    SBATCH_ARGS+=( -A "$ACCOUNT" )
fi

if [ -n "$PARTITION" ]; then
    SBATCH_ARGS+=( -p "$PARTITION" )
fi

if [ -n "$EXCLUDE_NODES" ]; then
    SBATCH_ARGS+=( --exclude="$EXCLUDE_NODES" )
fi

WRAP_CMD="cd '$PROJECT_DIR'; export JULIA_DEPOT_PATH='$JULIA_DEPOT'; '$JULIA_BIN' benchmark_runtime_tau/benchmark_runtime_tau_collect.jl '$BENCH_ROOT'"

echo "Submitting collector job..."
echo "  benchmark root: $BENCH_ROOT"
echo "  job name: $JOB_NAME"
echo "  account: $ACCOUNT"
echo "  partition: $PARTITION"
echo "  output log: $BENCH_ROOT/collect_%j.out"

sbatch "${SBATCH_ARGS[@]}" --wrap="$WRAP_CMD"
