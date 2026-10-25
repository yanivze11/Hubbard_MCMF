#!/bin/bash
# tests/test_scratch_job_local.sh — local (laptop) test of scratch_job.sh.
# No SLURM: julia starts directly (LAUNCH=""), scratch is a temp folder.
#
#   bash tests/test_scratch_job_local.sh <work_dir>
#
#   A  one unit runs start to finish             -> check
#   B  a second unit is stopped with SIGTERM once a checkpoint was copied back
#      (like a preemption), then started again    -> check --expect-resume
set -u
W=${1:?usage: test_scratch_job_local.sh <work_dir>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rm -rf "$W"; mkdir -p "$W/data" "$W/scr"
export JULIA_BIN=${JULIA_BIN:-$(command -v julia)} RUN_FILE="$REPO/fh_run_n.jl"
export SCRATCH_ROOT="$W/scr" LAUNCH="" SYNC_INTERVAL=15 POLL_SEC=2 RETRY_SEC=1 QUIET_SEC=3
fail=0

echo "=== A: full run ==="
julia "$REPO/tests/scratch_job_make_unit.jl" "$W/data/A" >/dev/null
DIR="$W/data/A" SLURM_JOB_ID=A bash "$REPO/scratch_job.sh"; echo "wrapper exit $?"
julia "$REPO/tests/scratch_job_check.jl" "$W/data/A" || fail=1
[ -z "$(ls -A "$W/scr")" ] || { echo "FAIL: scratch not cleaned: $(ls "$W/scr")"; fail=1; }

echo "=== B: SIGTERM after the first checkpoint copy, then restart ==="
julia "$REPO/tests/scratch_job_make_unit.jl" "$W/data/B" 4 3000 >/dev/null
DIR="$W/data/B" SLURM_JOB_ID=B bash "$REPO/scratch_job.sh" & WPID=$!
for i in $(seq 1 600); do
    [ -f "$W/data/B/seed_1/checkpoint.jld2" ] && [ -f "$W/data/B/seed_1/results.jld2" ] && break
    sleep 1
done
echo "copied back after ~${i}s: $(ls "$W/data/B" "$W/data/B/seed_1" | tr '\n' ' ')"
sleep 4
kill -TERM "$WPID"; wait "$WPID"; echo "wrapper exit $? (143 = stopped by SIGTERM)"
[ -f "$W/data/B/sim.scratch.jld2" ] || { echo "FAIL: no sim.scratch.jld2 after the stop"; fail=1; }
ls "$W/data/B"/.*.sync_tmp "$W/data/B"/*/.*.sync_tmp 2>/dev/null && { echo "FAIL: temp files left"; fail=1; }
echo "--- restart ---"
DIR="$W/data/B" SLURM_JOB_ID=B SLURM_RESTART_COUNT=1 bash "$REPO/scratch_job.sh"; echo "wrapper exit $?"
julia "$REPO/tests/scratch_job_check.jl" --expect-resume "$W/data/B" || fail=1
[ -z "$(ls -A "$W/scr")" ] || { echo "FAIL: scratch not cleaned: $(ls "$W/scr")"; fail=1; }
echo "--- B scratch.log ---"; cat "$W/data/B/scratch.log"

[ $fail = 0 ] && echo "LOCAL SCRATCH TEST: PASS" || echo "LOCAL SCRATCH TEST: FAIL"
exit $fail
