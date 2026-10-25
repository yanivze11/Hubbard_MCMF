# ============================================================================
# fh_pre_run_V.jl — the V7 submission driver: one SLURM job per work unit.
#
#   Edit the sweep block below (sizes, U, V, beta, filling, disorder seeds and
#   density-channel SCHEMES), then submit through pre_pre_run.jl:
#       julia pre_pre_run.jl fh_pre_run_V.jl
#   A work unit is one parameter point (+ disorder seed + scheme) with one MC
#   seed; each runs fh_run_n.jl (fixed filling, see occupation_convergence.jl).
#   Each scheme sets (alpha, beta_phi, Qx, Qy); "baseline" is the channel-off run.
#   See docs/WORKFLOW.md §4.
#
# How it avoids duplicate work:
#
#   1. THE SUBMITTING JOB REGISTERS THE SIMULATION. `register_simulation` writes
#      a parameters-only stub into means_db and updates the index immediately,
#      so the canonical parameter file lists the simulation before it has run.
#
#   2. DEDUP VIA IDENTITY HASH, not a hand-written field-by-field comparison
#      against a monolithic sim_info.jld2. One lookup in the index.
#
#   3. CROSS-CLUSTER SPLIT BY CONSTRUCTION. `assign_cluster(identity, mc_seed)`
#      is a pure function, so this script submits only the work units that
#      belong to the cluster it is running on. Landau and Curie therefore cannot
#      duplicate each other's work without exchanging any information, and
#      re-running this script submits nothing new.
#
#   4. `disorder_seed` is a real parameter, independent of the MC seed. Seeds
#      listed in MC_SEEDS sample MC noise WITHIN one disordered sample;
#      DISORDER_SEEDS selects which sample.
#
# Required environment:
#   MEANS_DB_CLUSTER  = landau | curie   which cluster this is
#   JULIA_DEPOT_PATH  = this cluster's depot (.julia_landau / .julia_curie) —
#                       it is baked into every SLURM script
# Optional: V7_DATA_ROOT, MEANS_DB_ROOT, OUT_DIR, FOLDER_NAME, JULIA_BIN,
#           DRY_RUN=1 to print what would be submitted without submitting.
#
# Data defaults to /home/data/<user>/data/MC_Hubbard_V7 (cluster_config.jl) — the SAME path string on
# both clusters, resolving to each one's own storage. /usr is shared between the
# clusters, so one code checkout there serves both; the depots are not shared.
# ============================================================================
using JLD2, Formatting, Printf, Base.Iterators

# V7_DATA_ROOT, JULIA_BIN, JOB_ACCOUNT and the per-cluster settings.
include(joinpath(@__DIR__, "cluster_config.jl"))
# Must be set BEFORE the include: MEANS_DB_ROOT is a const captured at load time.
haskey(ENV, "MEANS_DB_ROOT") || (ENV["MEANS_DB_ROOT"] = joinpath(V7_DATA_ROOT, "means_db"))

include("sim_num_control.jl")
include("slurm_memory.jl")   # per-L job memory (L=32 needs 8 GB)
include("slurm_job_mode.jl") # JOB_MODE=scratch (default) | legacy

run_file = joinpath(@__DIR__, "fh_run_n.jl")
julia_bin = JULIA_BIN
# The DEPOTS are per-cluster (.julia_landau / .julia_curie) and this value gets
# baked into every SLURM script, so an empty one silently breaks the jobs.
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")

# Which cluster is this? pre_pre_run.jl exports MEANS_DB_CLUSTER; otherwise the
# hostname decides, so a direct run also works.
const THIS_CLUSTER = detect_cluster_name()
const DRY_RUN      = get(ENV, "DRY_RUN", "0") == "1"
# PREVIEW=1 runs a pure, DB-free count of the sweep and its cross-cluster split
# (see preview()). It needs no cluster identity, no depot and no julia binary,
# so it runs on a laptop before submitting: `PREVIEW=1 julia fh_pre_run_V.jl`.
const PREVIEW      = get(ENV, "PREVIEW", "0") == "1"

if !PREVIEW && !(THIS_CLUSTER in DB_CLUSTERS)
    error("cannot tell which cluster this is (hostname=$(gethostname())). Set " *
          "MEANS_DB_CLUSTER to one of $(DB_CLUSTERS) — it decides which work " *
          "units belong to this machine.")
end

# Cluster-specific SLURM settings (cluster_config.jl; pre_pre_run.jl exports
# the same values). The fallbacks only matter for PREVIEW on a laptop.
partitions    = cluster_setting_or(THIS_CLUSTER, :partitions, "allq")
cpus_per_task = cluster_setting_or(THIS_CLUSTER, :cpus_per_task, 1)
exclude_nodes = cluster_setting_or(THIS_CLUSTER, :exclude_nodes, "")

######################### Generate slurm.txt #################################
templateSLURM = FormatExpr("""#!/bin/bash

#SBATCH --job-name={1}
#SBATCH --output={2}/%A_%a.out
#SBATCH --error={2}/%A_%a.error
#SBATCH --threads-per-core=1
#SBATCH -c {4}
#SBATCH -n 1
#SBATCH --mem-per-cpu={6}GB
#SBATCH --requeue

#SBATCH -A $JOB_ACCOUNT
#SBATCH -p $partitions
#SBATCH --exclude=$exclude_nodes
#SBATCH --chdir={2}

#SBATCH --array=1-{3}

export DIR={2}
export JULIA_DEPOT_PATH={5}
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
srun --chdir=\$DIR $julia_bin $run_file --sim_path=\$DIR

""")
###############################################################################

out_dir     = get(ENV, "OUT_DIR", joinpath(V7_DATA_ROOT, "data"))
folder_name = get(ENV, "FOLDER_NAME", "HighT__13-10-2026")
path        = joinpath(out_dir, folder_name)
sim_name    = "HighT"
# sim_name = "Disorder_V_sweep"

# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
begin
const Ls              = [(8,8), (10,10)]
const Us              = [2.0, 4.0, 6.0]
const Vs              = [0.0]
# const Vs              = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
# const BETAS           = [20.0, 10.0, 5.0] 
# const BETAS = unique(round.([0.02:0.02:0.2; 0.2:0.04:2;].^-1, digits=1)) # the usual
# High T: the usual grid keeps digits=1 so its betas hash identically to existing
# data. The T>2 extension needs 3 digits: with one decimal, beta<=0.5 only takes
# the values 0.5/0.4/0.3/0.2/0.1, which collapses these 43 temperatures onto 5.
const T_USUAL = [0.05:0.4:2;]
const T_HIGH  = [2:0.4:10;]
const BETAS = unique([round.(T_USUAL.^-1, digits=1); round.(T_HIGH.^-1, digits=2)])
# const WANTED_NS       = [0.875]
const WANTED_NS       = [0.6, 0.7, 0.8, 0.9]
# WHICH disordered samples to draw. This is where the statistics come from:
# the Monte-Carlo average is self-averaging, so extra MC chains on the SAME
# sample buy little, whereas each new sample is genuinely new information.
#
# Each (config, disorder_seed) is its own identity, and assign_cluster spreads
# them over the clusters — so both clusters run the SAME U/L/V/beta and differ
# only in which samples they own. Loading locally pools them into a
# disorder-average (with the sample-to-sample spread available).
const DISORDER_SEEDS = [1]

# Density-channel decoupling schemes. Each is its own identity (alpha/beta_phi/Qx/Qy
# enter the means_db hash when beta_phi != 0), so all four run as separate work
# units at every (U,L,V,beta,disorder_seed). "baseline" is the legacy channel-off
# run and hashes identically to pre-density-channel data.
const SCHEMES = [
    (name="baseline", alpha=1.0, beta_phi=0.0, Qx=0.0,           Qy=0.0),
    # (name="a",        alpha=2/3, beta_phi=1/3, Qx=0.0,           Qy=0.0),          # split, uniform Hartree
#     (name="b",        alpha=1.0, beta_phi=1.0, Qx=0.0,           Qy=0.0),          # BdG + full Hartree (BAD)
#     (name="c",        alpha=2/3, beta_phi=1/3, Qx=Float64(pi),   Qy=Float64(pi)),  # split, staggered CDW
    (name="d",        alpha=1.0, beta_phi=0.5, Qx=0.0,   Qy=0.0),  # BdG + half Hartree - Similar to Tarrat 2014
]

# The MC seed is fixed, not swept — see above.
const MC_SEED = 1
# const DIAG_METHOD     = "K0-TCA"
const DIAG_METHOD     = "K0"
const SC_L            = 1
const B               = 0.0
const T_HOP           = 1.0
const U_DISORDER      = "uniform"      # V is the disorder here, not U
const IMPURITY        = 0.0
const PAIRING         = "s-wave"
const GUESS           = "uniform"
const BOUNDARY        = "PBC"
const GAUGE           = "Mixed"

const num_measure    = 20000
const num_thermal    = 5000
const measuring_interval = 25
const num_thermal_occ = 200
const num_measure_occ = 100
const measuring_interval_occ = 5
const n_tol           = 0.001
const max_iter        = 100
const num_uncorrelated_measurments = 16
const smooth_U        = (false, 0.8)
const U_puddle_params = [0.0, 6.0, 0.7, 0.7]

# TClustersize is meaningless for K0 and is pinned out of the identity hash by
# CONDITIONAL_IDENTITY; recorded only so the file matches the usual schema.
const TClustersize = 8

# MC-field observables, accumulated over ALL production measurements (not just the
# last configuration). fh_run_n.jl treats a missing key as OFF, so they must be
# written here explicitly. Bookkeeping only: they are not part of the identity hash.
#   |Delta| axis: bin 0.01 -> ~36 bins per Delta_0 at U=2 (Delta_0~0.36), and the
#   range covers the U=10 tails (Delta_0~4.9); values past the range clamp into the
#   last bin. phi axis: bin 0.02.
const DELTA_HIST_ABS_RANGE = [0.0, 15.0]
const DELTA_HIST_ABS_NBINS = 1500
const DELTA_HIST_ARG_NBINS = 50          # phase bins over [0, 2pi]
const PHI_HIST_RANGE       = [-10.0, 10.0]
const PHI_HIST_NBINS       = 1000
# Full field configuration (phi, Delta) every SNAPSHOT_INTERVAL measurements, to
# field_snapshots.jld2: 40 snapshots per 20000-measurement run, ~56 KB each at L=32.
const SNAPSHOT_INTERVAL    = 500

# Size-dependent grids. 4/(Lx*Ly) is the mean BdG level spacing (bandwidth 8 over 2N levels).
"""
    w_list_of(Lx, Ly)

ω grid of the DOS, LDOS and spectral function: 101 points in [-5, 5].
"""
w_list_of(Lx, Ly)  = collect(range(-5.0, 5.0, length=101))

"""
    w_cond_of(Lx, Ly)

ω grid of the fixed-broadening conductivity: 7 points from 0.1 to 2 level spacings,
then 55 points from 3 level spacings to 16.
"""
w_cond_of(Lx, Ly)  = [collect(range(4/(Lx * Ly) * 0.1, 4/(Lx * Ly) * 2.0, length=7)); collect(range(4/(Lx * Ly) * 3.0, 16.0, length=55))]

"""
    delta_fixed_of(Lx, Ly)

Broadenings δ of the fixed-δ conductivity: 0.5, 1, 2, …, 10 level spacings (for the δ → 0 extrapolation).
"""
delta_fixed_of(Lx, Ly)     = 4 / (Lx * Ly) .* [0.5; 1.0:10.0;]

"""
    delta_stiffness_of(Lx, Ly)

Empty: `delta_values_stiffness` is a removed feature (kept for the file schema).
"""
delta_stiffness_of(Lx, Ly) = []

"""
    dw_of(Lx, Ly)

Δω windows of the DC conductivity / `L_D_dc` measurement; empty = not measured.
"""
dw_of(Lx, Ly)              = []

"""
    build_params(; Lx, Ly, U, V, beta, wanted_n, disorder_seed, mc_seed=MC_SEED,
                 alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0) -> Dict{String,Any}

The sim.jld2 parameter Dict of one work unit: the arguments plus the fixed settings of
the sweep block. Its identity hash (`params_hash`) decides dedup and the owning cluster.
fh_run_n.jl reads it back.
"""
function build_params(; Lx, Ly, U, V, beta, wanted_n, disorder_seed, mc_seed=MC_SEED,
                       alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0)
    Dict{String,Any}(
        "Lx" => Lx, "Ly" => Ly, "SC_L" => SC_L, "U" => U, "t" => T_HOP, "V" => V,
        "disorder_seed" => disorder_seed,
        # density-channel decoupling scheme (channel off by default)
        "alpha" => alpha, "beta_phi" => beta_phi, "Qx" => Qx, "Qy" => Qy,
        "seeds" => Int64[mc_seed],
        "wanted_n" => wanted_n, "beta" => beta, "B" => B,
        "num_measure" => num_measure, "num_thermal" => num_thermal,
        "measuring_interval" => measuring_interval,
        "w_list" => w_list_of(Lx, Ly), "guess" => GUESS,
        "U_disorder" => U_DISORDER, "diag_method" => DIAG_METHOD,
        "cores" => cpus_per_task,
        "w_conductivity_range" => w_cond_of(Lx, Ly),
        "U_puddle_params" => U_puddle_params, "impurity" => IMPURITY,
        "TClustersize" => TClustersize, "smooth_U" => smooth_U,
        "pairing_symmetry" => PAIRING,
        "num_thermal_occ" => num_thermal_occ, "num_measure_occ" => num_measure_occ,
        "measuring_interval_occ" => measuring_interval_occ,
        "n_tol" => n_tol, "max_iter" => max_iter,
        "num_uncorrelated_measurments" => num_uncorrelated_measurments,
        "boundary_condition" => BOUNDARY, "gauge_choice" => GAUGE,
        "delta_fixed_values" => delta_fixed_of(Lx, Ly),
        "delta_values_stiffness" => delta_stiffness_of(Lx, Ly),
        "dw_values" => dw_of(Lx, Ly),
        # field histograms + snapshots (see the consts above)
        "Delta_hist_abs_range" => DELTA_HIST_ABS_RANGE,
        "Delta_hist_abs_nbins" => DELTA_HIST_ABS_NBINS,
        "Delta_hist_arg_nbins" => DELTA_HIST_ARG_NBINS,
        "phi_hist_range" => PHI_HIST_RANGE, "phi_hist_nbins" => PHI_HIST_NBINS,
        "snapshot_interval" => SNAPSHOT_INTERVAL,
    )
end

"""
    disorder_seeds_for(V)

Disorder seeds to run at potential disorder `V`: only the first for V = 0 (the clean
sample is the same for every seed), all of `DISORDER_SEEDS` otherwise.
"""
disorder_seeds_for(V) = V == 0 ? DISORDER_SEEDS[1:1] : DISORDER_SEEDS

"""
    sweep_units() -> Vector

All work units of the sweep, `((Lx, Ly), U, V, beta, wanted_n, disorder_seed, scheme)`,
with the V = 0 seed collapse. `preview` and `main` both use it, so their counts agree.
"""
function sweep_units()
    us = Vector{Any}()
    for ((Lx, Ly), U, V, beta, wanted_n, scheme) in product(Ls, Us, Vs, BETAS, WANTED_NS, SCHEMES)
        for dseed in disorder_seeds_for(V)
            push!(us, ((Lx, Ly), U, V, beta, wanted_n, dseed, scheme))
        end
    end
    return us
end

# ---------------------------------------------------------------------------
# Local preview (PREVIEW=1)
#
# How many work units the sweep contains and how the PURE cross-cluster split
# (assign_cluster) routes them — computed with no database, no depot and no
# julia binary, so it runs on a laptop before you submit anywhere. The identity
# hash is built only from cluster-independent physics/numerics params (`cores`
# and friends are non-identity), so these per-cluster counts match exactly what
# Landau and Curie will each pick up.
#
# NOTE: this is the FULL sweep, an UPPER BOUND. It does NOT subtract units a
# cluster has already registered — that dedup needs each cluster's means_db, so
# the real "SUBMIT" count on a first run equals this, and less on re-runs.
# ---------------------------------------------------------------------------
"""
    preview()

Print the number of work units in the sweep and how many each cluster owns
(`assign_cluster`). Needs no database, depot or Julia binary (`PREVIEW=1`). An upper
bound: units already registered are not subtracted.
"""
function preview()
    units = sweep_units()
    per_cluster = Dict(c => 0 for c in DB_CLUSTERS)
    for ((Lx, Ly), U, V, beta, wanted_n, dseed, scheme) in units
        params = build_params(; Lx=Lx, Ly=Ly, U=U, V=V, beta=beta,
                                wanted_n=wanted_n, disorder_seed=dseed,
                                alpha=scheme.alpha, beta_phi=scheme.beta_phi,
                                Qx=scheme.Qx, Qy=scheme.Qy)
        per_cluster[assign_cluster(params_hash(params), MC_SEED)] += 1
    end
    @printf("PREVIEW — full sweep, before any dedup (upper bound)\n\n")
    @printf("  %d work unit(s) total  (%d configs x schemes, with V=0 using 1 seed, V!=0 using %d)\n",
            length(units), length(BETAS) * length(Us) * length(Vs) * length(Ls),
            length(DISORDER_SEEDS))
    @printf("    %d beta x %d U x %d V x %d L x %d wanted_n = %d configs\n",
            length(BETAS), length(Us), length(Vs), length(Ls), length(WANTED_NS),
            length(BETAS) * length(Us) * length(Vs) * length(Ls) * length(WANTED_NS))
    @printf("    disorder samples: 1 for V=0 (clean), %d for V!=0\n", length(DISORDER_SEEDS))
    @printf("    schemes: %s\n\n", join([s.name for s in SCHEMES], ", "))
    for c in DB_CLUSTERS
        @printf("  %-8s : %d unit(s)\n", c, per_cluster[c])
    end
end

end

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
"""
    main()

Submit this cluster's share of the sweep. For each work unit: skip it if another
cluster owns it or it is already registered in means_db; otherwise make the folder
`<path>/<i>`, write sim.jld2 and slurm.txt (scratch or legacy job mode), `sbatch` it,
and only then register it in means_db. The means_db index is updated once at the end.
With `DRY_RUN=1` it only prints what it would do.
"""
function main()
    @printf("cluster        : %s\n", THIS_CLUSTER)
    @printf("means_db root  : %s\n", MEANS_DB_ROOT)
    @printf("sim directory  : %s\n", path)
    @printf("julia binary   : %s%s\n", julia_bin, isfile(julia_bin) ? "" : "   *** NOT FOUND ***")
    @printf("julia depot    : %s%s\n", isempty(current_depot) ? "(unset)" : current_depot,
            isempty(current_depot) ? "   *** must be this cluster's depot ***" : "")
    @printf("run file       : %s%s\n", run_file, isfile(run_file) ? "" : "   *** NOT FOUND ***")
    isfile(julia_bin) || error("JULIA_BIN does not exist: $julia_bin")
    isfile(run_file)  || error("fh_run_n.jl not found next to this script: $run_file")
    check_job_mode()
    # The depot is per-cluster and is written into every SLURM script; an empty
    # one produces jobs that cannot find their packages.
    isempty(current_depot) && error(
        "JULIA_DEPOT_PATH is unset. Export this cluster's depot " *
        "(.julia_landau / .julia_curie) before submitting, or the jobs will " *
        "run against the wrong package environment.")
    DRY_RUN && println("\nDRY RUN — nothing will be submitted or written")
    println()
    DRY_RUN || mkpath(path)

    units = sweep_units()
    @printf("\n%d work unit(s) in the sweep (%d configs x schemes; V=0 uses 1 seed, V!=0 uses %d)\n\n",
            length(units), length(BETAS) * length(Us) * length(Vs) * length(Ls),
            length(DISORDER_SEEDS))

    # Dedup and registration never touch the index. Dedup is a direct file
    # check on the unit's identity path, and registration writes only the stub;
    # the index is updated ONCE at the end. Per-unit index re-reads/rewrites
    # made the 2026-09-17 Landau submitter take ~25 min per job, so it hit its
    # 8 h limit after 17 of ~600 units. A run killed midway is still safe: the
    # stubs it wrote exist, so the next run's file check sees them.
    registered = String[]
    t_start = time()

    n_submitted = 0; n_other = 0; n_known = 0
    i = 1
    for ((Lx, Ly), U, V, beta, wanted_n, dseed, scheme) in units
        mc_seed = MC_SEED
        params = build_params(; Lx=Lx, Ly=Ly, U=U, V=V, beta=beta,
                                wanted_n=wanted_n, disorder_seed=dseed,
                                alpha=scheme.alpha, beta_phi=scheme.beta_phi,
                                Qx=scheme.Qx, Qy=scheme.Qy)
        h     = params_hash(params)
        owner = assign_cluster(h, mc_seed)

        if owner != THIS_CLUSTER
            n_other += 1
            @printf("  skip  U=%s b=%-5s dseed=%d sch=%-8s mc=%d  ->  %s owns it\n",
                    string(U), string(beta), dseed, scheme.name, mc_seed, owner)
            continue
        end

        # Already registered (submitted, running or done) with this MC seed?
        # Exact identity match. (The old field-by-field index filter omitted
        # wanted_n, diag_method, TClustersize and mu_disorder, so the n=0.875
        # mu-disorder runs at L=20, U=4 made every L=20, U=4 unit look "known".)
        if is_registered(params, mc_seed)
            n_known += 1
            @printf("  known U=%s b=%-5s dseed=%d sch=%-8s mc=%d  hash=%s\n",
                    string(U), string(beta), dseed, scheme.name, mc_seed, h)
            continue
        end

        c_path = joinpath(path, "$i")
        while isdir(c_path)
            i += 1
            c_path = joinpath(path, "$i")
        end

        @printf("  SUBMIT U=%s b=%-5s dseed=%d sch=%-8s mc=%d  hash=%s  -> %s\n",
                string(U), string(beta), dseed, scheme.name, mc_seed, h, c_path)
        if DRY_RUN
            n_submitted += 1
            i += 1
            continue
        end

        mkpath(c_path)
        JLD2.save(joinpath(c_path, "sim.jld2"), params)

        slurm_file_name = joinpath(c_path, "slurm.txt")
        slurm_job_name  = joinpath(sim_name,
            "n$(wanted_n)_Lx$(Lx)Ly$(Ly)U$(U)V$(V)b$(beta)_d$(dseed)_$(scheme.name)_s$(mc_seed)_$i")
        open(slurm_file_name, "w") do slurm_file
            if JOB_MODE == "scratch"
                write(slurm_file, scratch_slurm_script(; job_name=slurm_job_name, dir=c_path,
                      cpus=cpus_per_task, mem_per_cpu_gb=mem_per_cpu_gb(Lx, Ly, cpus_per_task),
                      partitions=partitions, exclude_nodes=exclude_nodes, depot=current_depot,
                      julia_bin=julia_bin, run_file=run_file))
            else
                printfmt(slurm_file, templateSLURM, slurm_job_name, c_path, 1,
                         cpus_per_task, current_depot, mem_per_cpu_gb(Lx, Ly, cpus_per_task))
            end
        end

        # Submit FIRST, register only once the job is actually queued.
        #
        # Registering first would be worse than useless if sbatch then fails:
        # the database would claim the unit as "submitted", the dedup check
        # would skip it on every future run, and the work would silently never
        # happen. An unregistered-but-running job is recoverable — num_control
        # creates its entry later — so this is the safe order.
        try
            run(`sbatch $slurm_file_name`)
        catch e
            rm(c_path; recursive=true, force=true)   # leave no half-made sim dir
            @error "sbatch failed; simulation NOT registered and its directory removed" c_path
            rethrow(e)
        end

        push!(registered, register_simulation(params; cluster=THIS_CLUSTER, seeds=[mc_seed],
                                              source=folder_name, update_index=false))
        n_submitted += 1
        i += 1
        # stdout to a file is block-buffered; flush so a killed run still
        # shows how far it got and how fast it was going.
        @printf("        [%d submitted, %.0f s elapsed]\n", n_submitted, time() - t_start)
        flush(stdout)
    end

    @printf("\nsubmitted %d, already known %d, owned by the other cluster %d  (%.0f s)\n",
            n_submitted, n_known, n_other, time() - t_start)
    if !DRY_RUN && !isempty(registered)
        idx = append_index!(MEANS_DB_ROOT, registered)
        @printf("index now holds %d simulation(s)\n", idx["n"])
    end
end

# ── Script entry point ──────────────────────────────────────────────────────
# Runs ONLY when the file is executed directly: `julia fh_pre_run_V.jl` submits,
# `PREVIEW=1 julia fh_pre_run_V.jl` previews. include()-ing the file from the
# REPL or VS Code just loads the definitions and submits nothing — which is what
# makes the preview block below safe.
if abspath(PROGRAM_FILE) == @__FILE__
    PREVIEW ? preview() : main()
end

# ════════════════════════════════════════════════════════════════════════════
#  ▶ VS CODE PREVIEW  —  put the cursor inside the begin…end block below and
#    press Alt+Enter (Julia: Execute Code Block).  It prints how many
#    simulations the sweep will submit in total and how they split across
#    Landau / Curie.  Submits NOTHING; needs no cluster, depot or database.
#    The first run loads the file's definitions (a few seconds); reruns are
#    instant.  This block never runs during `julia fh_pre_run_V.jl`.
# ════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) != @__FILE__
    begin
        if @isdefined(preview)
            preview()                                       # already loaded
        else
            ENV["PREVIEW"] = "1"                            # skip the cluster check
            include(joinpath(@__DIR__, "fh_pre_run_V.jl"))  # load; prints once
        end
    end
end