# ============================================================================
# tests_pre_run.jl — submits test files as SLURM jobs on THIS cluster, one job
# per (test file, Julia thread layout). Nothing runs on the login node.
#
#   Run on each cluster (after git pull), through the usual wrapper:
#       julia pre_pre_run.jl tests_pre_run.jl
#   Check first (loads nothing, writes and submits nothing):
#       DRY_RUN=1 JULIA_DEPOT_PATH=~/.julia_curie julia tests_pre_run.jl
#   When the jobs are done:
#       cat <the folder printed below>/summary.txt
#       tail -n 40 <the folder printed below>/*/*.out
#
#   Default: tests/test_measurement_kernels.jl (fast vs reference measurement
#   kernels) with three thread layouts:
#       1     -t 1
#       4     -t 4    -> Julia 1.12 starts 4 worker threads + 1 interactive thread,
#                        the layout whose thread ids crashed the prototypes
#       4+0   -t 4,0  -> 4 worker threads, no interactive thread (the campaign
#                        and benchmark job layout)
#   The test prints the actual layout ("threads N (+M interactive)") and every
#   number of the report: kernel and MeasureData differences, allocations, n_only,
#   memory estimate.
#
#   Optional environment:
#     TEST_FILES    test files in tests/, without .jl   default "test_measurement_kernels"
#     TEST_THREADS  thread layouts, "+" for the interactive count   default "1,4,4+0"
#     MEM_GB        memory per job                     default 8 (covers package compilation)
#     TIME_LIMIT    #SBATCH --time                     default "02:00:00"
#     FOLDER_NAME   output folder name                 default Tests_<cluster>__dd-mm-yyyy
#     DRY_RUN=1     print the jobs, write and submit nothing
#     V7_DATA_ROOT, JULIA_BIN, PARTITIONS, EXCLUDE_NODES (set by pre_pre_run.jl)
#
#   NOTE: older test files write output into the code checkout (local_smoke_runs/,
#   PNGs in tests/), which both clusters share. test_measurement_kernels.jl writes
#   nothing there.
#
#   Depot safety, as bench_profile_pre_run.jl: refuses to run with the OTHER
#   cluster's depot, and uses Base only (no package is loaded or compiled here).
# ============================================================================

include(joinpath(@__DIR__, "cluster_config.jl"))   # V7_DATA_ROOT, JULIA_BIN, cluster settings
const DRY_RUN      = get(ENV, "DRY_RUN", "0") == "1"

# ── which cluster (same detection as pre_pre_run.jl) ────────────────────────
const CLUSTER = detect_cluster_name()
haskey(CLUSTER_SETTINGS, CLUSTER) ||
    error("cannot tell which cluster this is (hostname=$(gethostname())); set MEANS_DB_CLUSTER=landau|curie")

depot = cluster_setting(CLUSTER, :depot)
# Never run (or bake into job scripts) another cluster's depot: packages
# compiled there target the other cluster's CPUs.
for (other, s) in CLUSTER_SETTINGS
    other == CLUSTER && continue
    occursin(basename(s.depot), depot) && error(
        "JULIA_DEPOT_PATH=$depot is the $other depot, but this is $CLUSTER. " *
        "Use $(CLUSTER_SETTINGS[CLUSTER].depot). Nothing was written or submitted.")
end
partitions    = cluster_setting(CLUSTER, :partitions)
exclude_nodes = cluster_setting(CLUSTER, :exclude_nodes)
julia_bin     = JULIA_BIN
const TESTS_DIR = joinpath(@__DIR__, "tests")

"""
    split_list(s) -> Vector{String}

The non-empty items of the comma list `s`.
"""
split_list(s) = [String(strip(x)) for x in split(s, ",") if !isempty(strip(x))]
const TEST_FILES = split_list(get(ENV, "TEST_FILES", "test_measurement_kernels"))
# "4+1" -> julia -t 4,1 ; cores requested = worker threads
const LAYOUTS    = [(spec=l, flag=replace(l, "+" => ","), cores=parse(Int, split(l, "+")[1]))
                    for l in split_list(get(ENV, "TEST_THREADS", "1,4,4+0"))]
const MEM_GB     = parse(Int, get(ENV, "MEM_GB", "8"))
const TIME_LIMIT = get(ENV, "TIME_LIMIT", "02:00:00")

"""
    stamp(fmt) -> String

The current time formatted with `strftime` format `fmt` (Base only, no Dates).
"""
stamp(fmt) = Libc.strftime(fmt, time())
folder_name = get(ENV, "FOLDER_NAME", "Tests_$(CLUSTER)__$(stamp("%d-%m-%Y"))")
path        = joinpath(V7_DATA_ROOT, "test_runs", folder_name)

"""
    slurm_script(job_name, dir, test_path, layout) -> String

SLURM script that runs one test file with one thread layout (`julia -t <layout>`) in
`dir`, records its exit code, and appends a line to the run's summary.txt.
"""
function slurm_script(job_name, dir, test_path, layout)
    excl_nodes = isempty(exclude_nodes) ? "" : "#SBATCH --exclude=$exclude_nodes\n"
    return """#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --output=$dir/%j.out
#SBATCH --error=$dir/%j.error
#SBATCH --threads-per-core=1
#SBATCH -c $(layout.cores)
#SBATCH -n 1
#SBATCH --mem=$(MEM_GB)G
#SBATCH --time=$TIME_LIMIT
#SBATCH --requeue

#SBATCH -A $JOB_ACCOUNT
#SBATCH -p $partitions
$(excl_nodes)#SBATCH --chdir=$dir

export DIR=$dir
export JULIA_DEPOT_PATH=$depot
# package compilation at job start uses only this job's cores (by default Julia
# starts workers for EVERY core of the node, and that blew the memory limit)
export JULIA_NUM_PRECOMPILE_TASKS=\$SLURM_CPUS_PER_TASK
# the code uses MKL (OPENBLAS_NUM_THREADS has no effect on it); the tests pin BLAS to 1 thread
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
echo "host \$(hostname)  job \$SLURM_JOB_ID  julia -t $(layout.flag)  $test_path"
srun --chdir=\$DIR $julia_bin -t $(layout.flag) $test_path
rc=\$?
echo \$rc > \$DIR/exit_code.txt
echo "$(basename(dir))  exit=\$rc  host=\$(hostname)  job=\$SLURM_JOB_ID  \$(date '+%Y-%m-%dT%H:%M:%S')" >> $path/summary.txt
exit \$rc
"""
end

"""
    main()

Submit one job per (test file, thread layout) into a new folder under
`<V7_DATA_ROOT>/test_runs`, never overwriting earlier runs, and write manifest.txt.
With `DRY_RUN=1` it only prints the jobs.
"""
function main()
    println("cluster      : ", CLUSTER)
    println("output       : ", path)
    println("julia binary : ", julia_bin, isfile(julia_bin) ? "" : "   *** NOT FOUND ***")
    println("julia depot  : ", depot)
    println("partitions   : ", partitions, "   exclude: ", exclude_nodes)
    println("tests        : ", join(TEST_FILES, ", "))
    println("threads      : ", join([l.spec for l in LAYOUTS], ", "))
    println("per job      : $(MEM_GB) GB, time limit $TIME_LIMIT\n")
    for t in TEST_FILES
        isfile(joinpath(TESTS_DIR, t * ".jl")) || error("test file not found: $(joinpath(TESTS_DIR, t * ".jl"))")
    end
    DRY_RUN || isfile(julia_bin) || error("JULIA_BIN does not exist: $julia_bin")
    DRY_RUN && println("DRY RUN — nothing is written or submitted\n")

    DRY_RUN || mkpath(path)
    manifest = String[]
    for t in TEST_FILES, layout in LAYOUTS
        name = "$(t)_t$(replace(layout.spec, "+" => "p"))"
        dir = joinpath(path, name)
        k = 2
        while isdir(dir)                                  # never overwrite an earlier run
            dir = joinpath(path, "$(name)_$k"); k += 1
        end
        test_path = joinpath(TESTS_DIR, t * ".jl")
        line = rpad(basename(dir), 40) * " julia -t " * rpad(layout.flag, 4) * " cores=$(layout.cores) mem=$(MEM_GB)G"
        if DRY_RUN
            println("  would submit  ", line)
            continue
        end
        mkpath(dir)
        slurm_file = joinpath(dir, "slurm.txt")
        write(slurm_file, slurm_script("test_$(CLUSTER)_$(basename(dir))", dir, test_path, layout))
        try
            out = read(`sbatch $slurm_file`, String)
            println("  submitted  ", line, "   ", strip(out))
            push!(manifest, line * "   " * strip(out))
        catch e
            @error "sbatch failed" dir exception=e
            push!(manifest, line * "   SBATCH FAILED")
        end
        flush(stdout)
    end
    if !DRY_RUN
        open(joinpath(path, "manifest.txt"), "a") do io
            println(io, "# ", stamp("%Y-%m-%dT%H:%M:%S"), "  cluster=", CLUSTER, "  host=", gethostname())
            foreach(l -> println(io, l), manifest)
        end
        println("\n$(length(manifest)) job(s). When they finish:")
        println("  cat $path/summary.txt")
        println("  tail -n 40 $path/*/*.out")
    end
end

main()
