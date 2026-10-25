#!/bin/bash
# ============================================================================
# scratch_job.sh — run ONE MC unit in node-local scratch (JOB_MODE=scratch).
#
# Follows the FHRC cluster rules (2026-10-20): a running job writes only to
# node-local /home/scr, and /home/data (the master's disk) sees ONE copy pass
# per SYNC_INTERVAL instead of a stream of small writes.
#
#   1. stage-in   copy the unit folder <DIR> (sim.jld2, and on a requeue also
#                 seed_N/checkpoint.jld2, results.jld2, logs) to
#                 <SCRATCH_ROOT>/mcjob_<jobid>/<unit>/
#   2. run        fh_run_n.jl --sim_path=<scratch unit>; Julia's stdout and
#                 stderr go to stdout.log in scratch, not to the SLURM .out file
#   3. sync       every SYNC_INTERVAL s: ONE pass copying every changed file
#                 back. Each file goes to a hidden temp name first, then is
#                 renamed, so /home/data always holds a complete earlier copy.
#                 JLD2 files are copied only when Julia has not touched them
#                 for QUIET_SEC s and they did not change during the copy.
#   4. finish     a last pass, then scratch_fix_paths.jl rewrites the paths
#                 recorded in sim.jld2 (scratch -> /home/data), and scratch is
#                 deleted.
#   5. SIGTERM    (preemption: KillWait is 30 s, no grace time) one best-effort
#                 pass of the files that were complete before the signal, then
#                 exit. SLURM requeues the job; its stage-in resumes from the
#                 last copy on /home/data.
#
# While a unit runs, its sim.jld2 is synced as sim.scratch.jld2: the scratch
# copy records scratch paths, which would break num_control / slim if they
# saw it. The /home/data sim.jld2 stays the submitter's stub until the job
# ends and step 4 writes the fixed one. A requeue stages sim.scratch.jld2 in
# as sim.jld2 (it holds the mu-search state).
#
# Environment (set by the generated slurm.txt; see slurm_job_mode.jl):
#   DIR            unit folder on /home/data                       (required)
#   JULIA_BIN      julia executable                                (required)
#   RUN_FILE       fh_run_n.jl                                     (required)
#   SYNC_INTERVAL  seconds between copy passes        (default 3600)
#   QUIET_SEC      JLD2 must be untouched this long   (default 10)
#   POLL_SEC       loop tick / heartbeat, local only  (default 60)
#   RETRY_SEC      tick while a due pass waits for Julia to finish a write
#                  (default 5; local stat calls only)
#   STALE_MIN      leftover scratch of dead jobs on this node is removed when
#                  its heartbeat is older than this    (default 180)
#   SCRATCH_ROOT   set by the job script from cluster_config.jl;
#                  fallback for older job scripts: /home/scr/$USER
#   LAUNCH         "srun" (default) or "" to start julia directly (local tests)
#   FIX_SCRIPT     default: scratch_fix_paths.jl next to this file
#
# Fallback: if scratch cannot be created, the unit runs in place exactly like
# JOB_MODE=legacy and the SLURM .out file says so.
# ============================================================================
# (No `set -u`: bash 4.2 treats an empty "${arr[@]}" as unbound.)

: "${DIR:?DIR must be the unit folder on /home/data}"
: "${JULIA_BIN:?JULIA_BIN must be set}"
: "${RUN_FILE:?RUN_FILE must be set}"
SYNC_INTERVAL=${SYNC_INTERVAL:-3600}
QUIET_SEC=${QUIET_SEC:-10}
POLL_SEC=${POLL_SEC:-60}
RETRY_SEC=${RETRY_SEC:-5}
STALE_MIN=${STALE_MIN:-180}
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/scr/${USER:-$(id -un)}}
LAUNCH=${LAUNCH-srun}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FIX_SCRIPT=${FIX_SCRIPT:-$HERE/scratch_fix_paths.jl}

DATA=${DIR%/}
UNIT=$(basename "$DATA")
JOBID=${SLURM_JOB_ID:-local$$}
JOBROOT=$SCRATCH_ROOT/mcjob_$JOBID
SCR=$JOBROOT/$UNIT
HEARTBEAT=$JOBROOT/.heartbeat

declare -A SYNCED          # rel path -> "size mtime" of the copy on /home/data
JPID=""
SLEEPPID=""
SYNC_NOW=0

# To the SLURM .out on /home/data: start, end and warnings only.
out() { echo "[scratch $(date '+%F %T')] $*"; }
# To scratch.log in scratch; it travels with the sync passes.
say() { echo "[scratch $(date '+%F %T')] $*" >> "$SCR/scratch.log"; }

# Scratch-side name -> /home/data-side name.
data_name() { [ "$1" = "sim.jld2" ] && echo "sim.scratch.jld2" || echo "$1"; }

# Regular files of the scratch unit, relative, depth 1-2 (unit + seed_N/).
list_scr() {
    (cd "$SCR" && for f in * */*; do [ -f "$f" ] && printf '%s\n' "$f"; done)
}

fstat() { stat -c '%s %Y' "$1" 2>/dev/null; }
mtime() { stat -c '%Y' "$1" 2>/dev/null || echo 0; }
is_jld2() { case "$1" in *.jld2) return 0 ;; esac; return 1; }

# Rename order: results before checkpoint. If a kill lands between two
# renames, an older checkpoint with newer results is safe (resume counts what
# results.jld2 holds), the reverse would lose measurements.
rank() {
    case "$1" in
        */results.jld2|results.jld2)       echo 0 ;;
        */checkpoint.jld2|checkpoint.jld2) echo 3 ;;
        sim.jld2)                          echo 2 ;;
        *)                                 echo 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# One copy pass.  $1 = periodic | final | term
#                 $2 = cutoff epoch: JLD2 files modified after it are skipped
#                      (0 = no cutoff). For "periodic" the cutoff is
#                      now-QUIET_SEC and a busy file makes the whole pass wait.
# Returns 0 when done (or nothing to do), 2 when Julia was writing (retry).
# ---------------------------------------------------------------------------
sync_pass() {
    local mode=$1 cutoff=$2 rel st m
    local -a changed=() picked=() tmps=()
    local -A cur=()
    local t0; t0=$(date +%s)

    while IFS= read -r rel; do
        st=$(fstat "$SCR/$rel") || continue
        cur[$rel]=$st
        [ "${SYNCED[$rel]:-}" != "$st" ] && changed+=("$rel")
    done < <(list_scr)
    [ ${#changed[@]} -eq 0 ] && return 0

    [ "$mode" = periodic ] && cutoff=$(( t0 - QUIET_SEC ))
    for rel in "${changed[@]}"; do
        if is_jld2 "$rel" && [ "$cutoff" -gt 0 ]; then
            m=${cur[$rel]#* }
            if [ "$m" -gt "$cutoff" ]; then
                [ "$mode" = periodic ] && return 2      # Julia busy: try later
                say "$mode: skip $rel (written after the cutoff)"
                continue
            fi
        fi
        picked+=("$rel")
    done
    [ ${#picked[@]} -eq 0 ] && return 0

    # Copy to hidden temp names next to the targets.
    local ddir dname tmp bytes=0
    for rel in "${picked[@]}"; do
        dname=$(data_name "$rel")
        ddir=$(dirname "$DATA/$dname")
        [ -d "$ddir" ] || mkdir -p "$ddir"
        tmp="$ddir/.$(basename "$dname").sync_tmp"
        if ! cp -p "$SCR/$rel" "$tmp"; then
            say "$mode: copy of $rel FAILED; keeping the previous copy"
            rm -f "$tmp" "${tmps[@]}"
            return 2
        fi
        tmps+=("$tmp")
        bytes=$(( bytes + ${cur[$rel]%% *} ))
    done

    # A JLD2 file that changed while it was copied may be torn: drop the pass.
    for rel in "${picked[@]}"; do
        if is_jld2 "$rel" && [ "$(fstat "$SCR/$rel")" != "${cur[$rel]}" ]; then
            say "$mode: $rel changed during the copy; pass dropped"
            rm -f "${tmps[@]}"
            return 2
        fi
    done

    # Rename into place, results first, checkpoint last.
    local r
    for r in 0 1 2 3; do
        for rel in "${picked[@]}"; do
            [ "$(rank "$rel")" = "$r" ] || continue
            dname=$(data_name "$rel")
            ddir=$(dirname "$DATA/$dname")
            mv -f "$ddir/.$(basename "$dname").sync_tmp" "$DATA/$dname"
            SYNCED[$rel]=${cur[$rel]}
        done
    done
    say "$mode: copied ${#picked[@]} file(s), $(( bytes / 1048576 )) MB, $(( $(date +%s) - t0 )) s"
    return 0
}

# ---------------------------------------------------------------------------
# Preemption / scancel. SLURM sends SIGTERM to Julia too; KillWait is 30 s.
# ---------------------------------------------------------------------------
on_term() {
    trap '' TERM USR1
    local t_sig; t_sig=$(date +%s)
    [ -n "$SLEEPPID" ] && kill "$SLEEPPID" 2>/dev/null
    say "SIGTERM: stopping (preempted, requeued or cancelled)"
    [ -n "$JPID" ] && kill -TERM "$JPID" 2>/dev/null
    local i
    for i in 1 2 3 4 5; do
        { [ -z "$JPID" ] || ! kill -0 "$JPID" 2>/dev/null; } && break
        sleep 1
    done
    # Files complete 2 s before the signal; anything newer may be half-written.
    sync_pass term $(( t_sig - 2 ))
    out "SIGTERM: last copy done; scratch removed; exiting (a requeue resumes from $DATA)"
    rm -rf "$JOBROOT"
    exit 143
}

# ---------------------------------------------------------------------------
# 0. Remove scratch left on this node by jobs that were killed before their
#    cleanup ran (their heartbeat stopped). Local disk only.
# ---------------------------------------------------------------------------
sweep_stale() {
    local d hb
    for d in "$SCRATCH_ROOT"/mcjob_*; do
        [ -d "$d" ] && [ "$d" != "$JOBROOT" ] || continue
        hb="$d/.heartbeat"; [ -e "$hb" ] || hb="$d"
        if [ -n "$(find "$hb" -maxdepth 0 -mmin +"$STALE_MIN" 2>/dev/null)" ]; then
            rm -rf "$d" && out "removed stale scratch $d"
        fi
    done
}

# ---------------------------------------------------------------------------
# Fallback: no scratch on this node -> run in place exactly like legacy.
# ---------------------------------------------------------------------------
run_in_place() {
    out "WARNING: no scratch at $SCRATCH_ROOT on $(hostname): running IN PLACE like JOB_MODE=legacy"
    # A preempted scratch attempt left its live sim.jld2 as sim.scratch.jld2.
    [ -f "$DATA/sim.scratch.jld2" ] && mv -f "$DATA/sim.scratch.jld2" "$DATA/sim.jld2"
    cd "$DATA" || exit 1
    if [ -n "$LAUNCH" ]; then
        exec $LAUNCH --chdir="$DATA" "$JULIA_BIN" "$RUN_FILE" --sim_path="$DATA"
    else
        exec "$JULIA_BIN" "$RUN_FILE" --sim_path="$DATA"
    fi
}

# =========================== main ==========================================
[ -f "$DATA/sim.jld2" ] || [ -f "$DATA/sim.scratch.jld2" ] || { out "ERROR: no sim.jld2 in $DATA"; exit 1; }

mkdir -p "$SCRATCH_ROOT" 2>/dev/null && [ -w "$SCRATCH_ROOT" ] || run_in_place
sweep_stale
rm -rf "$JOBROOT"                      # same node after a requeue: start clean
mkdir -p "$SCR" || run_in_place
touch "$HEARTBEAT"

# Leftover temp files from a pass that was killed mid-way.
for f in "$DATA"/.*.sync_tmp "$DATA"/*/.*.sync_tmp; do [ -f "$f" ] && rm -f "$f"; done

# 1. stage-in
t0=$(date +%s); n_in=0
use_scratch_sim=0; [ -f "$DATA/sim.scratch.jld2" ] && use_scratch_sim=1
while IFS= read -r rel; do
    case "$rel" in
        *.out|*.error|slurm.txt) continue ;;
        sim.jld2)         [ $use_scratch_sim = 1 ] && continue; target=sim.jld2 ;;
        sim.scratch.jld2) target=sim.jld2 ;;
        *)                target=$rel ;;
    esac
    mkdir -p "$(dirname "$SCR/$target")"
    cp -p "$DATA/$rel" "$SCR/$target" || { out "ERROR: stage-in of $rel failed"; rm -rf "$JOBROOT"; exit 1; }
    n_in=$((n_in + 1))
    # What /home/data already holds needs no copy back. sim.jld2 always goes
    # back once, as sim.scratch.jld2.
    [ "$target" = sim.jld2 ] && [ $use_scratch_sim = 0 ] || SYNCED[$target]=$(fstat "$SCR/$target")
done < <(cd "$DATA" && for f in * */*; do [ -f "$f" ] && printf '%s\n' "$f"; done)

restart=${SLURM_RESTART_COUNT:-0}
out "start on $(hostname): scratch $SCR, staged in $n_in file(s) in $(( $(date +%s) - t0 )) s, restart #$restart, sync every ${SYNC_INTERVAL}s"
say "start on $(hostname), job $JOBID, restart #$restart, staged in $n_in file(s)"

trap on_term TERM
trap 'SYNC_NOW=1' USR1            # --signal=B:USR1@900 before a time limit

# 2. run
if [ -n "$LAUNCH" ]; then
    $LAUNCH --chdir="$SCR" "$JULIA_BIN" "$RUN_FILE" --sim_path="$SCR" >> "$SCR/stdout.log" 2>&1 &
else
    (cd "$SCR" && exec "$JULIA_BIN" "$RUN_FILE" --sim_path="$SCR") >> "$SCR/stdout.log" 2>&1 &
fi
JPID=$!

# 3. periodic sync
last_sync=$(date +%s)
tick=$POLL_SEC
while kill -0 "$JPID" 2>/dev/null; do
    sleep "$tick" & SLEEPPID=$!
    wait "$SLEEPPID" 2>/dev/null; SLEEPPID=""
    touch "$HEARTBEAT"
    kill -0 "$JPID" 2>/dev/null || break
    now=$(date +%s)
    tick=$POLL_SEC
    if [ "$SYNC_NOW" = 1 ] || [ $(( now - last_sync )) -ge "$SYNC_INTERVAL" ]; then
        if sync_pass periodic 0; then
            last_sync=$now; SYNC_NOW=0
        else
            tick=$RETRY_SEC                # Julia is writing: look again soon
        fi
    fi
done
wait "$JPID"; rc=$?
JPID=""
t_exit=$(date +%s)

# 4. finish. On a crash (rc != 0, e.g. an OOM kill) a JLD2 file may be torn,
#    so only files complete before the exit are copied.
trap on_term TERM
say "julia exited with code $rc"
if [ "$rc" = 0 ]; then sync_pass final 0; else sync_pass final $(( t_exit - 2 )); fi
if [ -f "$DATA/sim.scratch.jld2" ]; then
    if "$JULIA_BIN" "$FIX_SCRIPT" "$DATA" >> "$SCR/scratch.log" 2>&1; then
        say "sim.jld2 paths fixed"
    else
        say "WARNING: scratch_fix_paths.jl failed; sim.scratch.jld2 left in $DATA"
        out "WARNING: scratch_fix_paths.jl failed; run it on $DATA by hand"
    fi
fi
sync_pass final 0                          # the log lines written above
out "end: julia exit $rc; copied back to $DATA; scratch removed"
rm -rf "$JOBROOT"
exit "$rc"
