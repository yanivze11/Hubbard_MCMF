# ============================================================================
# slurm_job_mode.jl — JOB_MODE switch for the submit drivers.
#
#   JOB_MODE=scratch (default since 2026-10-21) the job runs scratch_job.sh:
#                    node-local /home/scr, one copy pass back to /home/data every
#                    SYNC_INTERVAL s (default 3600), requeue-safe. See scratch_job.sh.
#                    Forced-requeue test passed on Curie and Landau 2026-10-21
#                    (tests/scratch_requeue_test.jl).
#   JOB_MODE=legacy  the driver writes its own templateSLURM exactly as before:
#                    the job runs in its /home/data unit folder.
#
# Going back = submit with JOB_MODE=legacy. The mode is baked into each
# unit's slurm.txt, so fh_resubmit_campaign.jl resubmits a unit in the mode it was
# submitted with. Do not switch a unit that already ran partly in one mode to the
# other mode by hand (a scratch unit holds its live state in sim.scratch.jld2).
#
# Base only, like slurm_memory.jl.
# ============================================================================
# JOB_ACCOUNT. The drivers include the config before this file, so load it only
# if it is not there yet (a second include would redefine its consts).
@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "cluster_config.jl"))

"The two job modes a submit driver can write into slurm.txt."
const JOB_MODES = ("legacy", "scratch")

"Job mode of this submission (env `JOB_MODE`, default `\"scratch\"`)."
const JOB_MODE = let m = lowercase(strip(get(ENV, "JOB_MODE", "scratch")))
    m in JOB_MODES || error("JOB_MODE=$m: must be one of $(JOB_MODES)")
    m
end

"The wrapper every scratch-mode job runs (it then runs fh_run_n.jl)."
const SCRATCH_JOB_SH = joinpath(@__DIR__, "scratch_job.sh")

"Seconds between two copy passes scratch -> /home/data (env `SYNC_INTERVAL`)."
const SYNC_INTERVAL  = parse(Int, get(ENV, "SYNC_INTERVAL", "3600"))

"""
    check_job_mode()

Print the job mode for the submitter's header, and fail early if scratch mode
cannot work (missing wrapper or path fixer next to this file).
"""
function check_job_mode()
    if JOB_MODE == "scratch"
        println("job mode       : scratch (node-local /home/scr, copy back every $(SYNC_INTERVAL) s)")
        isfile(SCRATCH_JOB_SH) || error("JOB_MODE=scratch but $SCRATCH_JOB_SH is missing")
        isfile(joinpath(@__DIR__, "scratch_fix_paths.jl")) ||
            error("JOB_MODE=scratch but scratch_fix_paths.jl is missing next to $SCRATCH_JOB_SH")
        SYNC_INTERVAL >= 60 || error("SYNC_INTERVAL=$(SYNC_INTERVAL) s is below 60 s")
    else
        println("job mode       : legacy (runs in the /home/data unit folder)")
    end
end

"""
    scratch_slurm_script(; job_name, dir, cpus, mem_per_cpu_gb, partitions,
                           exclude_nodes, depot, julia_bin, run_file,
                           sync_interval=SYNC_INTERVAL) -> String

The slurm.txt of one unit in JOB_MODE=scratch. The #SBATCH lines are the legacy
ones plus --open-mode=append (a requeue appends to the .out file instead of
truncating it) and --signal=B:USR1@900 (an extra copy pass 15 min before a time
limit, if the partition has one).
"""
function scratch_slurm_script(; job_name, dir, cpus, mem_per_cpu_gb, partitions,
                                exclude_nodes, depot, julia_bin, run_file,
                                sync_interval::Integer=SYNC_INTERVAL)
    return """#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --output=$dir/%A_%a.out
#SBATCH --error=$dir/%A_%a.error
#SBATCH --open-mode=append
#SBATCH --threads-per-core=1
#SBATCH -c $cpus
#SBATCH -n 1
#SBATCH --mem-per-cpu=$(mem_per_cpu_gb)GB
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

#SBATCH -A $JOB_ACCOUNT
#SBATCH -p $partitions
#SBATCH --exclude=$exclude_nodes
#SBATCH --chdir=$dir

#SBATCH --array=1-1

# JOB_MODE=scratch: runs in node-local $SCRATCH_ROOT and copies back to $dir
# once every $(sync_interval) s and at the end (see scratch_job.sh).
export DIR=$dir
export SCRATCH_ROOT=$SCRATCH_ROOT
export JULIA_DEPOT_PATH=$depot
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export JULIA_BIN=$julia_bin
export RUN_FILE=$run_file
export SYNC_INTERVAL=$sync_interval
exec bash "$SCRATCH_JOB_SH"

"""
end
