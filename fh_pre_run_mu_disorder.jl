# ============================================================================
# fh_pre_run_mu_disorder.jl — submission driver for the mu-U disorder campaign.
#
#   Sweeps L/beta x f x 3 U-mu COUPLING SCHEMES x 2 density-channel SCHEMES at fixed
#   U=4. The three U-mu couplings (the "isolation trio") share one site partition at
#   a given (f, disorder_seed):
#
#     interaction-only : U_i in {0,U} (interaction on/off), mu NOT shifted.
#     coupled          : U_i in {0,U} AND mu_i = mu - U_i/2.
#     potential-only   : interaction stays uniform U, only mu shifted (mu-U/2 on (1-f)).
#
#   Each is a distinct means_db identity (mu_disorder + mu_disorder_f + disorder_seed
#   + the scheme scalars enter the hash), so all run as separate work units and
#   none dedups against another.
#
#   Field observables (phi 1D + Delta_s 2D joint histograms, and periodic raw-field
#   snapshots) are ON for this campaign; they are recorded in sim.jld2 but are NOT
#   identity params, so they never split a run into two entries.
#
#   This reuses the SAME infrastructure as fh_pre_run_V.jl (means_db registration,
#   identity-hash dedup, pure cross-cluster split). See that file's header for the
#   environment variables and the mechanics; everything below only changes the sweep.
#
# Required environment:
#   MEANS_DB_CLUSTER  = landau | curie
#   JULIA_DEPOT_PATH  = this cluster's depot (.julia_landau / .julia_curie)
# Optional: V7_DATA_ROOT, MEANS_DB_ROOT, OUT_DIR, FOLDER_NAME, JULIA_BIN,
#           DRY_RUN=1, PREVIEW=1 (laptop-safe count + cross-cluster split).
# ============================================================================
using JLD2, Formatting, Printf, Base.Iterators

include(joinpath(@__DIR__, "cluster_config.jl"))   # V7_DATA_ROOT, JULIA_BIN, cluster settings
haskey(ENV, "MEANS_DB_ROOT") || (ENV["MEANS_DB_ROOT"] = joinpath(V7_DATA_ROOT, "means_db"))

include("sim_num_control.jl")
include("slurm_job_mode.jl") # JOB_MODE=scratch (default) | legacy

run_file = joinpath(@__DIR__, "fh_run_n.jl")
julia_bin = JULIA_BIN
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")

const THIS_CLUSTER = detect_cluster_name()
const DRY_RUN      = get(ENV, "DRY_RUN", "0") == "1"
const PREVIEW      = get(ENV, "PREVIEW", "0") == "1"

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
folder_name = get(ENV, "FOLDER_NAME", "Mu_disorder__30-08-2026")
path        = joinpath(out_dir, folder_name)
sim_name    = "Mu_disorder_sweep"

# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
begin
const Ls              = [(12, 12), (20, 20)]
const Us              = [4.0]
const Vs              = [0.0]                    # no potential-disorder V here; the disorder is mu_disorder
const BETAS           = unique(round.([0.02:0.02:0.2; 0.2:0.04:2;].^-1, digits=1))  # "usual" temperatures
const WANTED_NS       = [0.875]

# mu-U disorder: the empty-site fractions and the isolation trio of couplings.
const FS              = [0.05, 0.1, 0.2, 0.4]
# How the interaction disorder (U on/off) couples to the chemical potential. Each
# value is a SimData.mu_disorder mode -- see MC_Hubburd.jl (the mu_disorder field
# doc + build_mu_disorder!) for exactly what each does:
#   "interaction-only" : U_i in {0,U}, mu NOT shifted.
#   "coupled"          : U_i in {0,U} AND mu_i = mu - U_i/2.
#   "potential-only"   : uniform U, only mu shifted (mu - U/2 on the (1-f) sites).
const U_MU_COUPLING_SCHEMES = ["interaction-only", "coupled", "potential-only"]

# Which disordered samples to draw. The mu-U partition is drawn from disorder_seed,
# so each seed is a genuinely different sample. Bump this list for disorder-averaging.
const DISORDER_SEEDS  = [1,2 ,3]

# Density-channel decoupling schemes: beta_phi = 0 (baseline, channel off) and
# beta_phi = 0.5 (BdG + half Hartree, alpha=1, Q=0 -- as Tarat 2014 / item-3).
const SCHEMES = [
    (name="baseline", alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0),
    (name="d",        alpha=1.0, beta_phi=0.5, Qx=0.0, Qy=0.0),
]

const MC_SEED         = 1
const DIAG_METHOD     = "K0-TCA"
const SC_L            = 1
const B               = 0.0
const T_HOP           = 1.0
const U_DISORDER      = "uniform"   # the interaction on/off is applied by mu_disorder, NOT U_disorder
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

# K0-TCA with a fixed cluster size of 8 (user choice). For L=8 the run auto-falls
# back to exact K0 since TClustersize >= L (see running_simulation_wanted_occupation).
const TClustersize = 8

# --- field observables (ON for this campaign) ------------------------------
# NOT identity params: recorded in sim.jld2, but they never split a run.
const SNAPSHOT_INTERVAL     = 500                 # raw MC-field snapshot cadence (measurements)
const PHI_HIST_RANGE        = [-10.0, 10.0]
const PHI_HIST_NBINS        = 100
const DELTA_HIST_ABS_RANGE  = [0.0, 10.0]
const DELTA_HIST_ABS_NBINS  = 100
const DELTA_HIST_ARG_NBINS  = 50

# Size-dependent grids, as in fh_pre_run_V.jl (4/(Lx*Ly) = mean BdG level spacing).
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
    build_params(; Lx, Ly, U, V, beta, wanted_n, disorder_seed, mu_disorder, mu_disorder_f,
                 mc_seed=MC_SEED, alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0) -> Dict{String,Any}

The sim.jld2 parameter Dict of one work unit, as in fh_pre_run_V.jl plus the μ–U
disorder mode and fraction, with the field observables switched on.
"""
function build_params(; Lx, Ly, U, V, beta, wanted_n, disorder_seed, mu_disorder, mu_disorder_f,
                       mc_seed=MC_SEED, alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0)
    Dict{String,Any}(
        "Lx" => Lx, "Ly" => Ly, "SC_L" => SC_L, "U" => U, "t" => T_HOP, "V" => V,
        "disorder_seed" => disorder_seed,
        # density-channel decoupling scheme
        "alpha" => alpha, "beta_phi" => beta_phi, "Qx" => Qx, "Qy" => Qy,
        # coupled / potential mu-U disorder (identity params)
        "mu_disorder" => mu_disorder, "mu_disorder_f" => mu_disorder_f,
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
        # field observables (non-identity; recorded so fh_run_n turns them on)
        "snapshot_interval" => SNAPSHOT_INTERVAL,
        "phi_hist_range" => PHI_HIST_RANGE, "phi_hist_nbins" => PHI_HIST_NBINS,
        "Delta_hist_abs_range" => DELTA_HIST_ABS_RANGE,
        "Delta_hist_abs_nbins" => DELTA_HIST_ABS_NBINS,
        "Delta_hist_arg_nbins" => DELTA_HIST_ARG_NBINS,
    )
end

"""
    sweep_units() -> Vector

All work units, `((Lx, Ly), U, V, beta, wanted_n, f, u_mu_coupling, disorder_seed, scheme)`.
The μ–U disorder is on for every coupling, so every disorder seed is a different
sample (no V = 0 seed collapse).
"""
function sweep_units()
    us = Vector{Any}()
    for ((Lx, Ly), U, V, beta, wanted_n, f, u_mu_coupling, scheme) in
            product(Ls, Us, Vs, BETAS, WANTED_NS, FS, U_MU_COUPLING_SCHEMES, SCHEMES)
        for dseed in DISORDER_SEEDS
            push!(us, ((Lx, Ly), U, V, beta, wanted_n, f, u_mu_coupling, dseed, scheme))
        end
    end
    return us
end

# ---------------------------------------------------------------------------
# Local preview (PREVIEW=1): work-unit count + cross-cluster split, no DB.
# ---------------------------------------------------------------------------
"""
    preview()

Print the number of work units and each cluster's share, without a database
(`PREVIEW=1`). An upper bound: already registered units are not subtracted.
"""
function preview()
    units = sweep_units()
    per_cluster = Dict(c => 0 for c in DB_CLUSTERS)
    for ((Lx, Ly), U, V, beta, wanted_n, f, u_mu_coupling, dseed, scheme) in units
        params = build_params(; Lx=Lx, Ly=Ly, U=U, V=V, beta=beta, wanted_n=wanted_n,
                                disorder_seed=dseed, mu_disorder=u_mu_coupling, mu_disorder_f=f,
                                alpha=scheme.alpha, beta_phi=scheme.beta_phi,
                                Qx=scheme.Qx, Qy=scheme.Qy)
        per_cluster[assign_cluster(params_hash(params), MC_SEED)] += 1
    end
    @printf("PREVIEW — full sweep, before any dedup (upper bound)\n\n")
    @printf("  %d work unit(s) total\n", length(units))
    @printf("    %d L x %d U x %d beta x %d f x %d U-mu couplings x %d schemes x %d disorder_seed\n",
            length(Ls), length(Us), length(BETAS), length(FS), length(U_MU_COUPLING_SCHEMES),
            length(SCHEMES), length(DISORDER_SEEDS))
    @printf("    U-mu couplings : %s\n", join(U_MU_COUPLING_SCHEMES, ", "))
    @printf("    f        : %s\n", join(string.(FS), ", "))
    @printf("    schemes  : %s\n", join([@sprintf("%s(bphi=%g)", s.name, s.beta_phi) for s in SCHEMES], ", "))
    @printf("    betas    : %d values in [%g, %g]\n\n", length(BETAS), minimum(BETAS), maximum(BETAS))
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

Submit this cluster's share of the sweep, as `main` in fh_pre_run_V.jl (skip units
owned by the other cluster or already registered; write sim.jld2 + slurm.txt;
`sbatch`; then register). Note: this older driver updates the means_db index after
every unit, which is slow for large sweeps; fh_pre_run_V.jl updates it once at the end.
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
    isempty(current_depot) && error(
        "JULIA_DEPOT_PATH is unset. Export this cluster's depot before submitting.")
    DRY_RUN && println("\nDRY RUN — nothing will be submitted or written")
    println()
    DRY_RUN || mkpath(path)

    units = sweep_units()
    @printf("\n%d work unit(s) in the sweep\n\n", length(units))

    n_submitted = 0; n_other = 0; n_known = 0
    i = 1
    for ((Lx, Ly), U, V, beta, wanted_n, f, u_mu_coupling, dseed, scheme) in units
        mc_seed = MC_SEED
        params = build_params(; Lx=Lx, Ly=Ly, U=U, V=V, beta=beta, wanted_n=wanted_n,
                                disorder_seed=dseed, mu_disorder=u_mu_coupling, mu_disorder_f=f,
                                alpha=scheme.alpha, beta_phi=scheme.beta_phi,
                                Qx=scheme.Qx, Qy=scheme.Qy)
        h     = params_hash(params)
        owner = assign_cluster(h, mc_seed)

        if owner != THIS_CLUSTER
            n_other += 1
            continue
        end

        # Already registered with this MC seed? Exact identity match — the old
        # field-by-field filter silently skipped 422 L=12 disorder_seed=1 units
        # of the 2026-08-30 campaign.
        if is_registered(params, mc_seed)
            n_known += 1
            @printf("  known U=%s b=%-5s f=%-4s U-mu=%-16s sch=%-8s mc=%d  hash=%s\n",
                    string(U), string(beta), string(f), u_mu_coupling, scheme.name, mc_seed, h)
            continue
        end

        c_path = joinpath(path, "$i")
        while isdir(c_path)
            i += 1
            c_path = joinpath(path, "$i")
        end

        @printf("  SUBMIT U=%s b=%-5s f=%-4s U-mu=%-16s sch=%-8s mc=%d  hash=%s  -> %s\n",
                string(U), string(beta), string(f), u_mu_coupling, scheme.name, mc_seed, h, c_path)
        if DRY_RUN
            n_submitted += 1
            i += 1
            continue
        end

        mkpath(c_path)
        JLD2.save(joinpath(c_path, "sim.jld2"), params)

        slurm_file_name = joinpath(c_path, "slurm.txt")
        slurm_job_name  = joinpath(sim_name,
            "n$(wanted_n)_Lx$(Lx)Ly$(Ly)U$(U)b$(beta)_f$(f)_$(u_mu_coupling)_$(scheme.name)_d$(dseed)_s$(mc_seed)_$i")
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
#  ▶ VS CODE PREVIEW — put the cursor inside the begin…end sweep block above and
#    press Alt+Enter, or run `PREVIEW=1 julia fh_pre_run_mu_disorder.jl`. Prints
#    the work-unit count and the Landau/Curie split. Submits NOTHING.
# ════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) != @__FILE__
    begin
        if @isdefined(preview)
            preview()
        else
            ENV["PREVIEW"] = "1"
            include(joinpath(@__DIR__, "fh_pre_run_mu_disorder.jl"))
        end
    end
end
