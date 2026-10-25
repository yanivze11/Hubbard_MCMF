# ============================================================================
# pre_run_num_control_V.jl — submit run_num_control_V.jl as a slurm job.
#
# The V7 counterpart of V6's pre_run_sim_num.jl + run_sim_num_control.jl pair.
# It is LIGHT (writes a slurm script and sbatches it), so it runs on the login
# node; the actual means computation runs inside the job, on a compute node.
#
# By default the job makes ONE pass over the campaign and exits: run it after
# the campaign has finished. LOOP_MODE=1 turns it into a MONITOR that recomputes
# the means every LOOP_INTERVAL_SEC while the simulations run (each pass touches
# only seeds with new data). On the FHRC clusters the monitor breaks the storage
# rules (no polling of /home/data), so keep the default there.
#
#   FOLDER_NAME=<campaign> julia pre_run_num_control_V.jl
#
# Optional environment (with defaults):
#   JULIA_DEPOT_PATH   this cluster's depot from cluster_config.jl
#                      (refused if it is the OTHER cluster's depot)
#   MEANS_DB_CLUSTER   landau | curie          (else detected from hostname)
#   V7_DATA_ROOT       /home/data/<user>/data/MC_Hubbard_V7  (cluster_config.jl)
#   FOLDER_NAME        Vdisorder_test__23-07-2026
#   NUMCTL_MEM         8GB          per-cpu memory — RAISE for large L
#                                   (a single L=20 results.jld2 is ~369 MB)
#   NUMCTL_TIME        7-00:00:00   slurm walltime
#   NUMCTL_CPUS        1 landau / 4 curie
#   LOOP_MODE          0            1 = monitor loop (not on the FHRC clusters)
#   LOOP_INTERVAL_SEC  3600         (1 hour)
#   LOOP_MAX_HOURS     168
#   DRY_RUN            1            print the slurm script without submitting
# ============================================================================
using Dates
include(joinpath(@__DIR__, "cluster_config.jl"))   # V7_DATA_ROOT, JULIA_BIN, accounts, cluster settings

const FOLDER_NAME  = get(ENV, "FOLDER_NAME", "Vdisorder_test__23-07-2026")

const RUN_FILE  = joinpath(@__DIR__, "run_num_control_V.jl")
const DRY_RUN   = get(ENV, "DRY_RUN", "0") == "1"

# --- detect cluster, then its resource defaults ------------------------------
const CLUSTER = detect_cluster_name()
haskey(CLUSTER_SETTINGS, CLUSTER) ||
    error("cannot tell which cluster this is (hostname=$(gethostname())); set MEANS_DB_CLUSTER")

# The depot is baked into the job. Default: this cluster's depot from
# cluster_config.jl; an exported JULIA_DEPOT_PATH wins, unless it is ANOTHER
# cluster's depot (packages compiled there target the other cluster's CPUs).
const DEPOT = get(ENV, "JULIA_DEPOT_PATH", CLUSTER_SETTINGS[CLUSTER].depot)
for (other, s) in CLUSTER_SETTINGS
    other == CLUSTER && continue
    occursin(basename(s.depot), DEPOT) && error(
        "JULIA_DEPOT_PATH=$DEPOT is the $other depot, but this is $CLUSTER. " *
        "Unset it or use $(CLUSTER_SETTINGS[CLUSTER].depot).")
end

const EXCLUDE   = cluster_setting(CLUSTER, :exclude_nodes)
# num_control has its own core count (env NUMCTL_CPUS), separate from the
# simulations' CPUS_PER_TASK.
const CPUS      = get(ENV, "NUMCTL_CPUS", CLUSTER == "curie" ? "4" : "1")
const MEM       = get(ENV, "NUMCTL_MEM", "8GB")
const WALL      = get(ENV, "NUMCTL_TIME", "7-00:00:00")
# The group queue is proven to accept jobs on both clusters. If it rejects a
# 7-day walltime, set NUMCTL_PARTITION to one that allows it (e.g. allq).
const PARTITION = get(ENV, "NUMCTL_PARTITION", PRIORITY_PARTITION)

# --- loop config passed through to the worker --------------------------------
const LOOP_MODE     = get(ENV, "LOOP_MODE", "0")
const LOOP_INTERVAL = get(ENV, "LOOP_INTERVAL_SEC", "3600") # 1 hour
const LOOP_MAX_HRS  = get(ENV, "LOOP_MAX_HOURS", "168")

isfile(JULIA_BIN) || error("JULIA_BIN not found: $JULIA_BIN")
isfile(RUN_FILE)  || error("run_num_control_V.jl not next to this script: $RUN_FILE")

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
log_dir   = joinpath(V7_DATA_ROOT, "sim_logs", "num_control_$(CLUSTER)_$timestamp")
mkpath(log_dir)

# --requeue lets the monitor survive preemption; --open-mode append preserves the
# log across a requeue.
slurm = """#!/bin/bash
#SBATCH --job-name=numctl_$(CLUSTER)_$(timestamp)
#SBATCH --output=$log_dir/num_control.out
#SBATCH --error=$log_dir/num_control.error
#SBATCH --open-mode=append
#SBATCH -c $CPUS
#SBATCH -n 1
#SBATCH --mem-per-cpu=$MEM
#SBATCH --time=$WALL
#SBATCH --requeue
#SBATCH --exclude=$EXCLUDE
#SBATCH -A $PRIORITY_ACCOUNT
#SBATCH -p $PARTITION

export JULIA_DEPOT_PATH=$DEPOT
export OPENBLAS_NUM_THREADS=1
export MEANS_DB_CLUSTER=$CLUSTER
export V7_DATA_ROOT=$V7_DATA_ROOT
export FOLDER_NAME=$FOLDER_NAME
export LOOP_MODE=$LOOP_MODE
export LOOP_INTERVAL_SEC=$LOOP_INTERVAL
export LOOP_MAX_HOURS=$LOOP_MAX_HRS

$JULIA_BIN $RUN_FILE
"""

println("cluster       : $CLUSTER")
println("campaign      : $(joinpath(V7_DATA_ROOT, "data", FOLDER_NAME))")
println("resources     : $CPUS cpu(s), $MEM/cpu, walltime $WALL, part $PARTITION, exclude $EXCLUDE")
println("mode          : LOOP_MODE=$LOOP_MODE  interval=$(LOOP_INTERVAL)s  max=$(LOOP_MAX_HRS)h")
println("log dir       : $log_dir")

submit_file = joinpath(log_dir, "submit_num_control.sh")
open(submit_file, "w") do f; write(f, slurm); end

if DRY_RUN
    println("\nDRY RUN — slurm script written but not submitted:\n")
    println(slurm)
else
    println("\nsubmitting ...")
    run(`sbatch $submit_file`)
    println("submitted. follow it with:  tail -f $log_dir/num_control.out")
end
