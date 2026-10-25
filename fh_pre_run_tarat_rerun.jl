# ============================================================================
# fh_pre_run_tarat_rerun.jl — re-run the suspect points of the Tarat 2015
# benchmark with FRESH MC CHAINS.
#
# WHY THIS EXISTS
#   The U=6, beta_phi=0.5 benchmark (vault note "Tarat 2015 benchmark -
#   normalization study 17-09-2026") contains points that cannot be physical:
#
#     L=26, T=0.06 (beta=16.7): S/Delta0^2 = 0.034, while L=22 gives 0.528 and
#                               the SAME curve recovers to 0.286 at T=0.08.
#                               S must not RISE with temperature -> stuck chain.
#     L=14, T=0.10 (beta=10.0): 0.198 vs 0.322 at L=12 -- same suspicion, milder.
#
#   That single L=26 point drags its T_c estimate from ~0.073 down to 0.048 and
#   is the reason our T_c looks like it keeps sliding with L (Tarat's Fig. 7b
#   has T_c converged beyond L=16). Before concluding anything about size
#   dependence, these points need independent chains.
#
# WHAT IT SUBMITS
#   U=6, V=0, n=0.9, scheme "d" (alpha=1, beta_phi=0.5) ONLY, at
#   L = 14, 18, 22, 26 x beta = 25.0, 16.7, 12.5, 10.0, 8.3  (T = 0.04 .. 0.12),
#   for MC seeds 2, 3, 4.
#
#   The two flagged points sit in the middle of that window; their neighbours in
#   BOTH L and T come along so the re-run shows a curve, not an isolated number.
#
# THE KEY POINT — IDENTICAL IDENTITY, NEW SEEDS
#   Every numerical parameter matches the existing runs exactly (verified
#   against means_db: L=26/22/18 used K0-TCA with TClustersize=10, L=14 used
#   plain K0; all used num_thermal=5000, num_measure=20000). So these units
#   hash to the SAME means_db entries and simply ADD seeds 2,3,4 next to the
#   existing seed 1 — get_curve then averages them and the seed scatter becomes
#   visible. A parameter change would fork a separate entry instead and the new
#   data would never pool with the old.
#
#   PER-L DIAG METHOD IS NOT COSMETIC: L=14 was a full K0 diagonalization while
#   the larger sizes are K0-TCA(10x10). Submitting L=14 as TCA would create a
#   new entry and answer a different question.
#
# LONGER CHAINS (optional, separate identity)
#   LONG_RUN=1 quadruples the thermalization (20000) and doubles the measuring
#   (40000). num_thermal/num_measure ARE identity parameters, so this deliberately
#   creates its own entries and does NOT pool with the existing data -- it answers
#   "does a longer chain change the answer?" rather than "how noisy is the mean?".
#
# USAGE (same as fh_pre_run_V.jl — run it on BOTH clusters)
#   PREVIEW=1 julia fh_pre_run_tarat_rerun.jl     # laptop: count units, no DB needed
#   DRY_RUN=1 julia fh_pre_run_tarat_rerun.jl     # cluster: show submissions only
#   julia fh_pre_run_tarat_rerun.jl               # cluster: submit
#   LONG_RUN=1 julia fh_pre_run_tarat_rerun.jl    # the long-chain variant
#
# Required environment (identical to fh_pre_run_V.jl):
#   MEANS_DB_CLUSTER = landau | curie
#   JULIA_DEPOT_PATH = this cluster's depot (.julia_landau / .julia_curie)
# Optional: V7_DATA_ROOT, MEANS_DB_ROOT, OUT_DIR, FOLDER_NAME, JULIA_BIN,
#           PARTITIONS, CPUS_PER_TASK, EXCLUDE_NODES, DRY_RUN, PREVIEW, LONG_RUN
# ============================================================================
using JLD2, Formatting, Printf, Base.Iterators

include(joinpath(@__DIR__, "cluster_config.jl"))   # V7_DATA_ROOT, JULIA_BIN, cluster settings
haskey(ENV, "MEANS_DB_ROOT") || (ENV["MEANS_DB_ROOT"] = joinpath(V7_DATA_ROOT, "means_db"))

include("sim_num_control.jl")
include("slurm_job_mode.jl") # JOB_MODE=scratch (default) | legacy

run_file  = joinpath(@__DIR__, "fh_run_n.jl")
julia_bin = JULIA_BIN
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")

const THIS_CLUSTER = detect_cluster_name()
const DRY_RUN  = get(ENV, "DRY_RUN", "0") == "1"
const PREVIEW  = get(ENV, "PREVIEW", "0") == "1"
const LONG_RUN = get(ENV, "LONG_RUN", "0") == "1"

if !PREVIEW && !(THIS_CLUSTER in DB_CLUSTERS)
    error("cannot tell which cluster this is (hostname=$(gethostname())). Set " *
          "MEANS_DB_CLUSTER to one of $(DB_CLUSTERS).")
end

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
#SBATCH --mem-per-cpu=4GB
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
folder_name = get(ENV, "FOLDER_NAME", LONG_RUN ? "Tarat_rerun_long__17-09-2026" :
                                                 "Tarat_rerun__17-09-2026")
path        = joinpath(out_dir, folder_name)
sim_name    = "Tarat_rerun"

# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
begin

# Each lattice carries the diagonalization it was ORIGINALLY run with, read back
# from means_db. Change these and the re-run forks a new entry instead of adding
# seeds to the existing one.
const POINTS = [
    (L=14, diag_method="K0",     TClustersize=4),    # full diagonalization
    (L=18, diag_method="K0-TCA", TClustersize=10),
    (L=22, diag_method="K0-TCA", TClustersize=10),
    (L=26, diag_method="K0-TCA", TClustersize=10),   # holds the stuck T=0.06 point
]

# T = 0.04, 0.0599, 0.08, 0.10, 0.12 — the window that brackets both bad points.
# These are the exact beta values of the original sweep
# (unique(round.([0.02:0.02:0.2; 0.2:0.04:2;].^-1, digits=1))); a beta that does
# not match to the digit is a different identity and will not pool.
const BETAS = [25.0, 16.7, 12.5, 10.0, 8.3]

const Us        = [6.0]
const Vs        = [0.0]
const WANTED_NS = [0.9]
const DISORDER_SEEDS = [1]      # V=0 => clean, one seed is all there is

# Scheme "d" only: alpha=1, beta_phi=0.5 — Tarat's Eq. 1.
const SCHEMES = [
    (name="d", alpha=1.0, beta_phi=0.5, Qx=0.0, Qy=0.0),
]

# THE POINT OF THIS SCRIPT: independent MC chains at an unchanged identity.
# Seed 1 already exists in means_db for every unit below.
const MC_SEEDS = [2, 3, 4]

const SC_L       = 1
const B          = 0.0
const T_HOP      = 1.0
const U_DISORDER = "uniform"
const IMPURITY   = 0.0
const PAIRING    = "s-wave"
const GUESS      = "uniform"
const BOUNDARY   = "PBC"
const GAUGE      = "Mixed"

# Matched to the existing entries (5000 / 20000) unless LONG_RUN asks otherwise.
const num_measure = LONG_RUN ? 40000 : 20000
const num_thermal = LONG_RUN ? 20000 :  5000
const measuring_interval = 25
const num_thermal_occ = 200
const num_measure_occ = 100
const measuring_interval_occ = 5
const n_tol   = 0.001
const max_iter = 100
const num_uncorrelated_measurments = 16
const smooth_U = (false, 0.8)
const U_puddle_params = [0.0, 6.0, 0.7, 0.7]

# Size-dependent grids, identical to the original runs (they are identity parameters).
"""
    w_list_of(Lx, Ly)

ω grid of the DOS, LDOS and spectral function: 101 points in [-5, 5].
"""
w_list_of(Lx, Ly)  = collect(range(-5.0, 5.0, length=101))

"""
    w_cond_of(Lx, Ly)

ω grid of the fixed-broadening conductivity (same as fh_pre_run_V.jl).
"""
w_cond_of(Lx, Ly)  = [collect(range(4/(Lx * Ly) * 0.1, 4/(Lx * Ly) * 2.0, length=7)); collect(range(4/(Lx * Ly) * 3.0, 16.0, length=55))]

"""
    delta_fixed_of(Lx, Ly)

Broadenings δ of the fixed-δ conductivity: 0.5, 1, …, 10 level spacings.
"""
delta_fixed_of(Lx, Ly)     = 4 / (Lx * Ly) .* [0.5; 1.0:10.0;]

"""
    delta_stiffness_of(Lx, Ly)

Empty: `delta_values_stiffness` is a removed feature.
"""
delta_stiffness_of(Lx, Ly) = []

"""
    dw_of(Lx, Ly)

Δω windows of the DC conductivity measurement; empty = not measured.
"""
dw_of(Lx, Ly)              = []

"""
    build_params(; Lx, Ly, U, V, beta, wanted_n, disorder_seed, mc_seed, alpha=1.0,
                 beta_phi=0.0, Qx=0.0, Qy=0.0, diag_method, TClustersize) -> Dict{String,Any}

The sim.jld2 parameter Dict of one work unit, matching the original benchmark runs
exactly (per-L `diag_method` and `TClustersize`), so the new MC seeds pool with seed 1.
"""
function build_params(; Lx, Ly, U, V, beta, wanted_n, disorder_seed, mc_seed,
                       alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0,
                       diag_method, TClustersize)
    Dict{String,Any}(
        "Lx" => Lx, "Ly" => Ly, "SC_L" => SC_L, "U" => U, "t" => T_HOP, "V" => V,
        "disorder_seed" => disorder_seed,
        "alpha" => alpha, "beta_phi" => beta_phi, "Qx" => Qx, "Qy" => Qy,
        "seeds" => Int64[mc_seed],
        "wanted_n" => wanted_n, "beta" => beta, "B" => B,
        "num_measure" => num_measure, "num_thermal" => num_thermal,
        "measuring_interval" => measuring_interval,
        "w_list" => w_list_of(Lx, Ly), "guess" => GUESS,
        "U_disorder" => U_DISORDER, "diag_method" => diag_method,
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
    )
end

"""
    disorder_seeds_for(V)

Only the first disorder seed for V = 0 (clean sample), all of `DISORDER_SEEDS` otherwise.
"""
disorder_seeds_for(V) = V == 0 ? DISORDER_SEEDS[1:1] : DISORDER_SEEDS

"""
    sweep_units() -> Vector

One work unit per (point, U, V, beta, n, disorder_seed, scheme, MC seed).
"""
function sweep_units()
    us = Vector{Any}()
    for (pt, U, V, beta, wanted_n, scheme, mc_seed) in
            product(POINTS, Us, Vs, BETAS, WANTED_NS, SCHEMES, MC_SEEDS)
        for dseed in disorder_seeds_for(V)
            push!(us, (pt, U, V, beta, wanted_n, dseed, scheme, mc_seed))
        end
    end
    return us
end

"""
    preview()

Print the number of work units, each cluster's share, and whether they pool with the
existing seed-1 entries (`PREVIEW=1`, no database needed).
"""
function preview()
    units = sweep_units()
    per_cluster = Dict(c => 0 for c in DB_CLUSTERS)
    for (pt, U, V, beta, wanted_n, dseed, scheme, mc_seed) in units
        params = build_params(; Lx=pt.L, Ly=pt.L, U=U, V=V, beta=beta,
                                wanted_n=wanted_n, disorder_seed=dseed, mc_seed=mc_seed,
                                alpha=scheme.alpha, beta_phi=scheme.beta_phi,
                                Qx=scheme.Qx, Qy=scheme.Qy,
                                diag_method=pt.diag_method, TClustersize=pt.TClustersize)
        per_cluster[assign_cluster(params_hash(params), mc_seed)] += 1
    end
    @printf("PREVIEW — Tarat re-run%s, before dedup (upper bound)\n\n",
            LONG_RUN ? " [LONG_RUN: thermal=$(num_thermal), measure=$(num_measure)]" : "")
    @printf("  %d work unit(s) = %d L x %d beta x %d MC seed(s)\n\n",
            length(units), length(POINTS), length(BETAS), length(MC_SEEDS))
    @printf("  U=%s  V=%s  n=%s  scheme: %s (alpha=%g, beta_phi=%g)\n",
            string(Us), string(Vs), string(WANTED_NS), SCHEMES[1].name,
            SCHEMES[1].alpha, SCHEMES[1].beta_phi)
    @printf("  T values : %s\n", join([@sprintf("%.4f", 1/b) for b in BETAS], ", "))
    @printf("  MC seeds : %s   (seed 1 already exists -> these ADD to it)\n\n", string(MC_SEEDS))
    for pt in POINTS
        @printf("    L=%-3d %-8s TClustersize=%d\n", pt.L, pt.diag_method, pt.TClustersize)
    end
    println()
    for c in DB_CLUSTERS
        @printf("  %-8s : %d unit(s)\n", c, per_cluster[c])
    end
    if !LONG_RUN
        println("\n  identity matches the existing runs, so these pool with seed 1.")
    else
        println("\n  LONG_RUN changes num_thermal/num_measure — SEPARATE entries by design.")
    end
end

end

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
"""
    main()

Submit this cluster's share of the re-run, as `main` in fh_pre_run_V.jl. Dedup is per
MC seed, so the existing seed 1 does not block seeds 2–4 at the same identity.
"""
function main()
    @printf("cluster        : %s\n", THIS_CLUSTER)
    @printf("means_db root  : %s\n", MEANS_DB_ROOT)
    @printf("sim directory  : %s\n", path)
    @printf("julia binary   : %s%s\n", julia_bin, isfile(julia_bin) ? "" : "   *** NOT FOUND ***")
    @printf("julia depot    : %s%s\n", isempty(current_depot) ? "(unset)" : current_depot,
            isempty(current_depot) ? "   *** must be this cluster's depot ***" : "")
    @printf("run file       : %s%s\n", run_file, isfile(run_file) ? "" : "   *** NOT FOUND ***")
    @printf("chains         : num_thermal=%d num_measure=%d%s\n",
            num_thermal, num_measure, LONG_RUN ? "   [LONG_RUN — separate entries]" : "")
    isfile(julia_bin) || error("JULIA_BIN does not exist: $julia_bin")
    isfile(run_file)  || error("fh_run_n.jl not found next to this script: $run_file")
    check_job_mode()
    isempty(current_depot) && error(
        "JULIA_DEPOT_PATH is unset. Export this cluster's depot " *
        "(.julia_landau / .julia_curie) before submitting.")
    DRY_RUN && println("\nDRY RUN — nothing will be submitted or written")
    println()
    DRY_RUN || mkpath(path)

    units = sweep_units()
    @printf("\n%d work unit(s): %d L x %d beta x %d MC seed(s)\n\n",
            length(units), length(POINTS), length(BETAS), length(MC_SEEDS))

    n_submitted = 0; n_other = 0; n_known = 0
    i = 1
    for (pt, U, V, beta, wanted_n, dseed, scheme, mc_seed) in units
        params = build_params(; Lx=pt.L, Ly=pt.L, U=U, V=V, beta=beta,
                                wanted_n=wanted_n, disorder_seed=dseed, mc_seed=mc_seed,
                                alpha=scheme.alpha, beta_phi=scheme.beta_phi,
                                Qx=scheme.Qx, Qy=scheme.Qy,
                                diag_method=pt.diag_method, TClustersize=pt.TClustersize)
        h     = params_hash(params)
        owner = assign_cluster(h, mc_seed)

        if owner != THIS_CLUSTER
            n_other += 1
            @printf("  skip  L=%-3d b=%-5s mc=%d  ->  %s owns it\n",
                    pt.L, string(beta), mc_seed, owner)
            continue
        end

        # Dedup on THIS MC seed only: seed 1 is present everywhere and must not
        # make us skip seeds 2/3/4 at the same identity.
        # Exact identity match (see is_registered in means_db.jl).
        if is_registered(params, mc_seed)
            n_known += 1
            @printf("  known L=%-3d b=%-5s mc=%d  hash=%s\n", pt.L, string(beta), mc_seed, h)
            continue
        end

        c_path = joinpath(path, "$i")
        while isdir(c_path)
            i += 1
            c_path = joinpath(path, "$i")
        end

        @printf("  SUBMIT L=%-3d b=%-5s T=%.4f mc=%d %-8s  hash=%s  -> %s\n",
                pt.L, string(beta), 1/beta, mc_seed, pt.diag_method, h, c_path)
        if DRY_RUN
            n_submitted += 1
            i += 1
            continue
        end

        mkpath(c_path)
        JLD2.save(joinpath(c_path, "sim.jld2"), params)

        slurm_file_name = joinpath(c_path, "slurm.txt")
        slurm_job_name  = joinpath(sim_name,
            "n$(wanted_n)_Lx$(pt.L)Ly$(pt.L)U$(U)V$(V)b$(beta)_d$(dseed)_$(scheme.name)_s$(mc_seed)_$i")
        open(slurm_file_name, "w") do slurm_file
            if JOB_MODE == "scratch"
                # 4 GB per CPU, as in templateSLURM above
                write(slurm_file, scratch_slurm_script(; job_name=slurm_job_name, dir=c_path,
                      cpus=cpus_per_task, mem_per_cpu_gb=4, partitions=partitions,
                      exclude_nodes=exclude_nodes, depot=current_depot,
                      julia_bin=julia_bin, run_file=run_file))
            else
                printfmt(slurm_file, templateSLURM, slurm_job_name, c_path, 1,
                         cpus_per_task, current_depot)
            end
        end

        # Submit FIRST, register only once the job is queued (see fh_pre_run_V.jl).
        try
            run(`sbatch $slurm_file_name`)
        catch e
            rm(c_path; recursive=true, force=true)
            @error "sbatch failed; simulation NOT registered and its directory removed" c_path
            rethrow(e)
        end

        register_simulation(params; cluster=THIS_CLUSTER, seeds=[mc_seed],
                            source=folder_name)
        n_submitted += 1
        i += 1
    end

    @printf("\nsubmitted %d, already known %d, owned by the other cluster %d\n",
            n_submitted, n_known, n_other)
    if !DRY_RUN
        idx = load_index(; refresh=true)
        @printf("index now holds %d simulation(s)\n", idx["n"])
    end
end

# ── Script entry point ──────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    PREVIEW ? preview() : main()
end

# ════════════════════════════════════════════════════════════════════════════
#  ▶ VS CODE PREVIEW — cursor inside the begin…end block, Alt+Enter.
#    Prints the unit count and the Landau/Curie split. Submits NOTHING.
# ════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) != @__FILE__
    begin
        if @isdefined(preview)
            preview()
        else
            ENV["PREVIEW"] = "1"
            include(joinpath(@__DIR__, "fh_pre_run_tarat_rerun.jl"))
        end
    end
end
