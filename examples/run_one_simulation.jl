# ============================================================================
# examples/run_one_simulation.jl — one small simulation on your own computer.
#
#   julia --project=. examples/run_one_simulation.jl [output folder]
#
# It does exactly what one cluster job does, at a size that takes about a minute:
#   1. writes the parameters into <output>/sim.jld2 (what a submit driver writes);
#   2. runs fh_run_n.jl on that folder (what the SLURM job runs): μ search for the
#      wanted density, thermalization, measurements, saved in <output>/seed_1/;
#   3. loads <output>/seed_1/results.jld2 and computes every mean and error bar
#      with get_means (what num_control does for each seed).
#
# Change the parameters below to explore. Keep L small on a laptop (L = 4 takes
# about two minutes, most of it compilation; with diag_method = "K0" one sweep is L^2 diagonalizations of a
# 2L^2 x 2L^2 matrix, so the cost grows like L^8).
# ============================================================================
using JLD2, Printf

const REPO = normpath(joinpath(@__DIR__, ".."))
const OUT  = abspath(get(ARGS, 1, joinpath(REPO, "examples", "output", "one_simulation")))

# ---------------------------------------------------------------------------
# 1. Parameters (same keys as build_params in fh_pre_run_V.jl; Appendix A of
#    docs/WORKFLOW.md explains each one)
# ---------------------------------------------------------------------------
L = 4
run_params = Dict{String,Any}(
    # model
    "Lx" => L, "Ly" => L, "U" => 4.0, "t" => 1.0, "wanted_n" => 0.875, "beta" => 5.0,
    "V" => 0.0, "disorder_seed" => 0, "B" => 0.0,
    "alpha" => 1.0, "beta_phi" => 0.0, "Qx" => 0.0, "Qy" => 0.0,   # density channel off
    "pairing_symmetry" => "s-wave", "U_disorder" => "uniform", "impurity" => 0.0,
    "U_puddle_params" => [0.0, 6.0, 0.7, 0.7], "smooth_U" => (false, 0.8),
    "mu_disorder" => "none", "mu_disorder_f" => 0.0,
    # method
    "diag_method" => "K0", "TClustersize" => 4, "SC_L" => 1,
    "boundary_condition" => "PBC", "gauge_choice" => "Mixed", "guess" => "uniform",
    "seeds" => [1], "cores" => 1,
    # Monte Carlo lengths: thermalization as in production (fh_pre_run_V.jl uses
    # 5000 / 20000 / 25); fewer measurements, so the error bars are larger
    "num_thermal" => 5000, "num_measure" => 2000, "measuring_interval" => 5,
    # μ search, as in production: at each trial μ, 200 thermalization sweeps and
    # 100 measurements of ⟨n⟩, extended until 16 of them are uncorrelated
    "num_thermal_occ" => 200, "num_measure_occ" => 100, "measuring_interval_occ" => 5,
    "n_tol" => 0.005, "max_iter" => 100, "num_uncorrelated_measurments" => 16,
    # measurement grids
    "w_list" => collect(range(-5.0, 5.0, length=41)),
    "w_conductivity_range" => collect(range(0.1, 4.0, length=10)),
    "delta_fixed_values" => 4 / L^2 .* [1.0, 2.0, 4.0],
    "delta_values_stiffness" => Float64[], "dw_values" => Float64[],
)

mkpath(OUT)
isdir(joinpath(OUT, "seed_1")) &&
    error("$OUT already holds a run (seed_1/). Delete it or pass another folder: " *
          "a run in an existing folder RESUMES from its checkpoint.")
JLD2.save(joinpath(OUT, "sim.jld2"), run_params)
println("parameters written to ", joinpath(OUT, "sim.jld2"))

# ---------------------------------------------------------------------------
# 2. The simulation itself: the same command a cluster job runs
# ---------------------------------------------------------------------------
# The child julia uses the same package environment as this one.
project = dirname(Base.active_project())
cmd = `$(Base.julia_cmd()) --project=$project $(joinpath(REPO, "fh_run_n.jl")) --sim_path=$OUT`
println("running: ", cmd)
withenv("MC_SKIP_MEMORY_CHECK" => "1") do   # no SLURM memory limit on a laptop
    run(cmd)
end

# ---------------------------------------------------------------------------
# 3. Means and error bars of this one seed
# ---------------------------------------------------------------------------
include(joinpath(REPO, "sim_num_control.jl"))
data  = load_results(joinpath(OUT, "seed_1"); to_load_checkpoint=false)
means = get_means(data)

println("\nconverged mu = ", data.sim_data.mu, "   measurements = ",
        data.current_state.num_current_measurments)
for key in ("n", "S_Delta_Q0", "Ps", "stiffness", "drude_direct_xx")
    haskey(means, key) && means[key] isa AbstractDict || continue
    m = means[key]
    @printf("  %-16s = %10.5f ± %.5f   (τ = %.1f measurements)\n",
            key, real(m["value"]), real(m["error"]), m["correlation_time"])
end
println("\nall observables: ", join(sort(collect(keys(means))), ", "))
