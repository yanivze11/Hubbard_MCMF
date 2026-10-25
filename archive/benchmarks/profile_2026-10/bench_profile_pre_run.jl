@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
# ============================================================================
# bench_profile_pre_run.jl — submits the runtime-profile benchmark on THIS cluster.
#
#   Same tests as the 2026-10-06 local profile, one SLURM job per (L, cores):
#   MC sweep vs measurement breakdown, prototype kernels vs current ones,
#   occupation-phase measurement cost, BLAS-thread scaling. Plus one small job
#   for the Metropolis acceptance rates. Worker: bench_profile_run.jl.
#
#   Run on each cluster (after git pull), either through the usual wrapper
#       julia pre_pre_run.jl bench_profile_pre_run.jl
#   or directly on the login node
#       JULIA_DEPOT_PATH=~/.julia_curie julia bench_profile_pre_run.jl
#   Then, when the jobs are done,
#       julia bench_profile_collect.jl <the folder printed below>
#
#   Optional environment:
#     BENCH_LS        lattice sizes            default "12,16,20,24,28,32"
#     BENCH_CORES     cores per job            default "1,4"   (both clusters)
#     BENCH_SECTIONS  per-L sections           default "profile,proto,occ,threads"
#     BENCH_ACCEPT    1/0 add the acceptance job  default 1
#     FOLDER_NAME     output folder name       default Profile_<cluster>__dd-mm-yyyy
#     TIME_LIMIT      e.g. "12:00:00" adds #SBATCH --time (none by default, as the campaign)
#     EXCLUSIVE=1     whole node per job (cleaner timings, costs node time)
#     DRY_RUN=1       print the jobs, write and submit nothing
#     V7_DATA_ROOT, JULIA_BIN, PARTITIONS, EXCLUDE_NODES, CPUS_PER_TASK (ignored here)
#
#   Writes nothing to means_db: these are timing runs, not campaign data.
#
#   Depot safety. Both clusters run the same julia binary, so a package compiled
#   on one cluster gets the SAME cache filename in a depot as on the other, and
#   overwrites it. Hence:
#     * the script refuses to run with the OTHER cluster's depot (for example
#       ~/.julia_landau on Curie), before loading anything;
#     * it uses Base only. In Julia 1.12 even Printf/Dates are compiled into the
#       depot, so they are avoided. A DRY_RUN therefore loads no package and
#       writes nothing into any depot. Only a real run loads JLD2, to write
#       sim.jld2.
# ============================================================================

# V7_DATA_ROOT comes from cluster_config.jl
const DRY_RUN      = get(ENV, "DRY_RUN", "0") == "1"

# ── which cluster (same detection as pre_pre_run.jl) ────────────────────────
function detect_cluster()
    c = lowercase(get(ENV, "MEANS_DB_CLUSTER", ""))
    isempty(c) || return c
    h = lowercase(gethostname())
    occursin("landau", h) && return "landau"
    occursin("curie", h)  && return "curie"
    error("cannot tell which cluster this is (hostname=$(gethostname())); set MEANS_DB_CLUSTER=landau|curie")
end
const CLUSTER = detect_cluster()

const _DEPOT      = Dict("landau" => "~/.julia_landau",  "curie" => "~/.julia_curie")
const _PARTITIONS = Dict("landau" => "allq,infiniq",     "curie" => "allq")
const _EXCLUDE    = Dict("landau" => "landau59,landau61",         "curie" => "curie25,curie31")

depot         = get(ENV, "JULIA_DEPOT_PATH", _DEPOT[CLUSTER])
# Never load packages from (or bake into job scripts) the other cluster's depot:
# packages compiled there target the other cluster's CPUs.
let other = CLUSTER == "landau" ? "curie" : "landau"
    occursin("julia_$other", depot) && error(
        "JULIA_DEPOT_PATH=$depot is the $other depot, but this is $CLUSTER. " *
        "Use $(_DEPOT[CLUSTER]). Nothing was loaded or written.")
end
# JLD2 is needed only to write sim.jld2, so a dry run never touches the depot.
DRY_RUN || @eval using JLD2
partitions    = get(ENV, "PARTITIONS", _PARTITIONS[CLUSTER])
exclude_nodes = get(ENV, "EXCLUDE_NODES", _EXCLUDE[CLUSTER])
julia_bin     = JULIA_BIN
run_file      = joinpath(@__DIR__, "bench_profile_run.jl")

parse_ints(s) = [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
const LS       = parse_ints(get(ENV, "BENCH_LS", "12,16,20,24,28,32"))
const CORES    = parse_ints(get(ENV, "BENCH_CORES", "1,4"))
const SECTIONS = [String(strip(x)) for x in split(get(ENV, "BENCH_SECTIONS", "profile,proto,occ,threads"), ",")]
const ACCEPT   = get(ENV, "BENCH_ACCEPT", "1") == "1"

stamp(fmt) = Libc.strftime(fmt, time())  # Base only (no Dates)
folder_name = get(ENV, "FOLDER_NAME", "Profile_$(CLUSTER)__$(stamp("%d-%m-%Y"))")
path        = joinpath(V7_DATA_ROOT, "benchmarks", folder_name)

# ── what every job measures: the Phi_channel__17-09-2026 campaign numerics ───
const CAMPAIGN = Dict{String,Any}(
    "U" => 4.0, "beta" => 10.0, "mu" => -0.5,          # mu only sets n; timings do not care
    "TClustersize" => 8,
    "num_thermal" => 5000, "num_measure" => 20000, "measuring_interval" => 25,
    "measuring_interval_occ" => 5,
    "nsweep" => 3,                                      # timed sweeps (minimum is reported)
)
const ACCEPT_CFG = Dict{String,Any}(
    "accept_L" => 16, "accept_burn" => 30, "accept_sweeps" => 10,
    # (U, beta, beta_phi): campaign U range, low and high T, both schemes
    "accept_points" => [[2.0, 20.0, 0.0], [4.0, 10.0, 0.0], [4.0, 2.0, 0.0], [6.0, 5.0, 0.0],
                        [8.0, 10.0, 0.0], [8.0, 1.0, 0.0], [10.0, 10.0, 0.0],
                        [4.0, 10.0, 0.5], [8.0, 10.0, 0.5]],
)

# Memory: U_matrix copies + the 2(L+1) current-vertex matrices of the current
# code + (L+1) more for the prototype comparison, each (2N)^2 complex.
# The 8 GB floor covers Julia compiling packages at job start; with 4 GB, two jobs
# were OOM-killed on 2026-10-06 before reaching the benchmark.
function mem_gb(L, sections)
    mat = (2L^2)^2 * 16 / 2^30
    nmat = 2(L + 1) + ("proto" in sections ? (L + 1) : 0) + 12
    return max(8, ceil(Int, 1.25 * mat * nmat + 2.5))
end

# Optional JULIA_CPU_TARGET for the package caches. Default: unset (Julia's
# "native"). JULIA_CPU_TARGET=generic was tried on Landau on 2026-10-06 and FAILED:
# with this juliaup 1.12.1 install, packages that depend on Pkg / LazyArtifacts /
# OrderedCollections could not be precompiled ("Module ... is missing from the
# cache", "?"), because those modules were already loaded from images built for a
# different target. With the default, a job that lands on a CPU type other than
# the one that compiled the depot recompiles at start. That is slow but harmless
# (8 GB floor, JULIA_NUM_PRECOMPILE_TASKS below) and happens before anything is timed.
const CPU_TARGET = get(ENV, "BENCH_CPU_TARGET", "")

function slurm_script(job_name, dir, cores, mem)
    time_line = haskey(ENV, "TIME_LIMIT") ? "#SBATCH --time=$(ENV["TIME_LIMIT"])\n" : ""
    excl_line = get(ENV, "EXCLUSIVE", "0") == "1" ? "#SBATCH --exclusive\n" : ""
    excl_nodes = isempty(exclude_nodes) ? "" : "#SBATCH --exclude=$exclude_nodes\n"
    cpu_line = isempty(CPU_TARGET) ? "" : "export JULIA_CPU_TARGET=\"$CPU_TARGET\"\n"
    return """#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --output=$dir/%j.out
#SBATCH --error=$dir/%j.error
#SBATCH --threads-per-core=1
#SBATCH -c $cores
#SBATCH -n 1
#SBATCH --mem=$(mem)G
#SBATCH --requeue

#SBATCH -A all-account
#SBATCH -p $partitions
$(excl_nodes)#SBATCH --chdir=$dir
$(time_line)$(excl_line)
export DIR=$dir
export JULIA_DEPOT_PATH=$depot
# ",0": no interactive thread (Julia 1.12 adds one by default for -t > 1), so the
# worker threads are exactly 1..ncores
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK,0
# package compilation at job start uses only this job's cores (by default Julia
# starts workers for EVERY core of the node, and that blew the memory limit)
export JULIA_NUM_PRECOMPILE_TASKS=\$SLURM_CPUS_PER_TASK
$(cpu_line)# the code uses MKL (OPENBLAS_NUM_THREADS has no effect on it); pin it explicitly
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
srun --chdir=\$DIR $julia_bin $run_file --sim_path=\$DIR
"""
end

function main()
    println("cluster      : ", CLUSTER)
    println("output       : ", path)
    println("julia binary : ", julia_bin, isfile(julia_bin) ? "" : "   *** NOT FOUND ***")
    println("julia depot  : ", depot)
    println("cpu target   : ", isempty(CPU_TARGET) ? "(julia default)" : CPU_TARGET)
    println("worker       : ", run_file, isfile(run_file) ? "" : "   *** NOT FOUND ***")
    println("partitions   : ", partitions, "   exclude: ", exclude_nodes)
    println("L            : ", join(LS, ", "))
    println("cores        : ", join(CORES, ", "))
    println("sections     : ", join(SECTIONS, ","), ACCEPT ? "  (+ 1 acceptance job)" : "", "\n")
    isfile(run_file) || error("bench_profile_run.jl not found next to this script")
    DRY_RUN || isfile(julia_bin) || error("JULIA_BIN does not exist: $julia_bin")
    DRY_RUN && println("DRY RUN — nothing is written or submitted\n")

    jobs = Any[]
    for L in LS, c in CORES
        push!(jobs, (name="L$(L)_c$(c)", L=L, cores=c, sections=SECTIONS))
    end
    ACCEPT && push!(jobs, (name="accept_L$(ACCEPT_CFG["accept_L"])_c1", L=ACCEPT_CFG["accept_L"], cores=1,
                           sections=["accept"]))

    DRY_RUN || mkpath(path)
    manifest = String[]
    for job in jobs
        dir = joinpath(path, job.name)
        k = 2
        while isdir(dir)                                  # never overwrite an earlier run
            dir = joinpath(path, "$(job.name)_$k"); k += 1
        end
        mem = mem_gb(job.L, job.sections)
        cfg = merge(CAMPAIGN, ACCEPT_CFG, Dict{String,Any}(
            "cluster" => CLUSTER, "L" => job.L, "cores" => job.cores,
            "sections" => collect(job.sections), "mem_GB" => mem,
            "created" => stamp("%Y-%m-%dT%H:%M:%S"), "folder" => folder_name))
        line = rpad(basename(dir), 16) * " L=" * rpad(string(job.L), 3) * " cores=$(job.cores) mem=" *
               lpad(string(mem), 3) * "G sections=" * join(job.sections, ",")
        if DRY_RUN
            println("  would submit  ", line)
            continue
        end
        mkpath(dir)
        JLD2.save(joinpath(dir, "sim.jld2"), cfg)
        slurm_file = joinpath(dir, "slurm.txt")
        write(slurm_file, slurm_script("prof_$(CLUSTER)_$(basename(dir))", dir, job.cores, mem))
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
        println("\n$(length(manifest)) job(s). When they finish:\n  julia bench_profile_collect.jl $path")
    end
end

main()
