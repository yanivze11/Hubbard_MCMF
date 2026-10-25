# ============================================================================
# pre_pre_run.jl — run any cluster script as a small SLURM job ("pre_run job").
#
#   julia pre_pre_run.jl [--prerun-queue=drorq|general] <script.jl> [script args...]
#
# This is the ONE command you type on a cluster login node. It does no physics:
#   1. makes a log folder  <V7_DATA_ROOT>/sim_logs/<script>_<timestamp>/
#      with a copy of the script (config_snapshot.jl) and run_info.txt, so every
#      campaign keeps a record of exactly what was submitted;
#   2. detects the cluster (Landau / Curie) from the hostname;
#   3. writes submit_me.txt, a SLURM script that exports the cluster's settings
#      (depot, partitions, cores, excluded nodes, MEANS_DB_CLUSTER, JOB_MODE)
#      and then runs  `julia <script.jl> [script args...]`;
#   4. sbatches it.
#
# The script then runs on a compute node, not on the login node. For a submit
# driver such as fh_pre_run_V.jl that job writes one sim folder per simulation
# and sbatches one simulation job each. Its own output goes to
# <log folder>/pre_run.out and pre_run.error.
#
# --prerun-queue selects where the pre_run job itself runs:
#   drorq (default)  PRIORITY_ACCOUNT / PRIORITY_PARTITION from cluster_config.jl
#   general          JOB_ACCOUNT on the cluster's normal partitions
#                    (aliases: allq, infiq, infiniq)
#
# Environment read here (all optional): V7_DATA_ROOT, JOB_MODE (scratch|legacy,
# default scratch, see slurm_job_mode.jl), SYNC_INTERVAL, plus the overrides of
# cluster_config.jl. Settings are in cluster_config.jl.
# ============================================================================
using Dates
include(joinpath(@__DIR__, "cluster_config.jl"))

"""
    shell_quote(arg) -> String

Quote `arg` for a POSIX shell, so script arguments with spaces or quotes reach
the script unchanged.
"""
shell_quote(arg::AbstractString) = "'" * replace(arg, "'" => "'\"'\"'") * "'"

"""
    parse_cli(args) -> (prerun_queue, script, script_args)

Split the command line into the `--prerun-queue` choice (`"drorq"` or
`"general"`), the script to run and the arguments forwarded to it.
"""
function parse_cli(args)
    prerun_queue = "drorq"
    positional = String[]
    for arg in args
        if startswith(arg, "--prerun-queue=")
            val = lowercase(split(arg, "=", limit=2)[2])
            if val == "drorq"
                prerun_queue = "drorq"
            elseif val in ("general", "allq", "infiq", "infiniq", "allq,infiniq")
                prerun_queue = "general"
            else
                error("Unknown --prerun-queue value: '$val'. Use 'drorq' or 'general'.")
            end
        else
            push!(positional, arg)
        end
    end
    isempty(positional) && error("Usage: julia pre_pre_run.jl [--prerun-queue=drorq|general] " *
                                 "<script_to_run.jl> [script args...]")
    return prerun_queue, positional[1], positional[2:end]
end

"""
    write_run_record(log_dir, script, script_args, timestamp)

Copy the submitted script into `log_dir/config_snapshot.jl` and write
`log_dir/run_info.txt` (date, host, script, arguments).
"""
function write_run_record(log_dir, script, script_args, timestamp)
    cp(script, joinpath(log_dir, "config_snapshot.jl"))
    open(joinpath(log_dir, "run_info.txt"), "w") do f
        println(f, "Run Date: $timestamp")
        println(f, "Submitted from Host: $(gethostname())")
        println(f, "Original Script: $script")
        println(f, "Script Args: $(isempty(script_args) ? "(none)" : join(script_args, " "))")
    end
end

"""
    prerun_slurm_script(; ...) -> String

The text of `submit_me.txt`: the SLURM header of the pre_run job, the exported
cluster settings, and the `julia <script>` line.

The settings come straight from `CLUSTER_SETTINGS`, NOT from the login shell's
environment: this job is what exports them, and a stale `JULIA_DEPOT_PATH` in the
login shell (e.g. the other cluster's depot) must not leak into the jobs.
"""
function prerun_slurm_script(; timestamp, log_dir, account, partition, cluster,
                               job_mode, sync_line, script_cmd)
    s = CLUSTER_SETTINGS[cluster]
    depot, exclude_nodes = s.depot, s.exclude_nodes
    return """#!/bin/bash
#SBATCH --job-name=PreRun_$(timestamp)
#SBATCH --output=$log_dir/pre_run.out
#SBATCH --error=$log_dir/pre_run.error
#SBATCH --time=08:00:00
#SBATCH --mem=4GB
#SBATCH --exclude=$exclude_nodes
#SBATCH -A $account
#SBATCH -p $partition

export JULIA_DEPOT_PATH=$depot
# sbatch exports this environment to every job the script submits; without the
# unset they would inherit this job's --mem=4GB as SLURM_MEM_PER_NODE=4096 and
# their memory check would think they have 4 GB.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU
export EXCLUDE_NODES=$exclude_nodes
export CPUS_PER_TASK=$(s.cpus_per_task)
export PARTITIONS=$(s.partitions)
# Tells means_db which work units belong to this cluster, and labels every
# simulation it registers with where it was produced.
export MEANS_DB_CLUSTER=$cluster
export V7_DATA_ROOT=$V7_DATA_ROOT
export JOB_MODE=$job_mode
$sync_line

$script_cmd
"""
end

"""
    main(args)

Submit `<script.jl> [script args...]` as a pre_run SLURM job: make the log folder with
a copy of the script and run_info.txt, write submit_me.txt with this cluster's
environment, and `sbatch` it (see the header).
"""
function main(args)
    prerun_queue, script, script_args = parse_cli(args)
    isfile(script) || error("Script not found: $script")
    println("Will submit script: $script")

    # 1. log folder + record of what was submitted
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    log_dir = joinpath(V7_DATA_ROOT, "sim_logs", "$(replace(script, ".jl" => ""))_$timestamp")
    mkpath(log_dir)
    println("Created log directory: $log_dir")
    write_run_record(log_dir, script, script_args, timestamp)

    # 2. which cluster
    cluster = detect_cluster_name()
    haskey(CLUSTER_SETTINGS, cluster) || error("Unknown cluster! Hostname is: $(gethostname())")
    s = CLUSTER_SETTINGS[cluster]
    println("Detected cluster $cluster. Depot: $(s.depot), partitions: $(s.partitions)")

    # 3. where the pre_run job itself runs
    account, partition = prerun_queue == "drorq" ?
        (PRIORITY_ACCOUNT, PRIORITY_PARTITION) : (JOB_ACCOUNT, s.partitions)
    println("Pre_run job will use account: $account, partition: $partition")

    # JOB_MODE=scratch (default) | legacy (the way back), see slurm_job_mode.jl.
    # Written into submit_me.txt so the log folder records which mode was used.
    job_mode = lowercase(get(ENV, "JOB_MODE", "scratch"))
    job_mode in ("legacy", "scratch") || error("JOB_MODE=$job_mode: use legacy or scratch")
    sync_line = haskey(ENV, "SYNC_INTERVAL") ? "export SYNC_INTERVAL=$(ENV["SYNC_INTERVAL"])" : ""
    println("Job mode: $job_mode", isempty(sync_line) ? "" : " ($sync_line)")

    # 4. write and submit
    script_cmd = join(["julia", shell_quote(script), shell_quote.(script_args)...], " ")
    submit_file = joinpath(log_dir, "submit_me.txt")
    write(submit_file, prerun_slurm_script(; timestamp, log_dir, account, partition,
                                             cluster, job_mode, sync_line, script_cmd))
    println("Submitting job...")
    run(`sbatch $submit_file`)
    println("Done! Logs and documentation are in: $log_dir")
end

main(ARGS)
