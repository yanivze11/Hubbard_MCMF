# Local NIL (U = 0) run at a specific target occupation `wanted_n`.
#
# Mirrors the cluster NIL setup in fh_pre_run_NIL_n.jl but runs a single seed
# on this machine. Edit the parameters at the top and execute:
#
#     julia --project run_local_NIL_n.jl
#
# Results land under <working folder>/data/NIL_local/<sim_num>\seed_<seed>\
# (numbered automatically by setup_simulation_paths, with seed subfolder).

using Dates, Statistics

include(joinpath(@__DIR__, "..", "..", "occupation_convergence.jl"))
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

# ============================================================================
# Parameters (edit these)
# ============================================================================

# Target filling
wanted_n = 0.875

# Lattice
Lx, Ly = 10, 11
SC_L   = 1

# NIL: no interaction, no chemical-potential disorder, no impurity, no field
U          = 0.0
t          = 1.0
V          = 0.0
U_impurity = 0.0
B          = 0.0

# Temperature and seed
beta = 30.0
seed = 12345

# NIL main-simulation knobs: a single measurement is enough since there is no MC
num_measure        = 1
num_thermal        = 0
measuring_interval = 1

# Frequency grids
w_list               = collect(-4.5:0.1:4.5)
w_conductivity_range = [10.0^a for a in -10:0.05:0.5]

# Diagonalization / pairing / disorder model
diag_method      = "K0"
pairing_symmetry = "s-wave"
initial_guess    = "uniform"
U_disorder       = "uniform"
U_puddle_params  = (0.0, 4.0, 1.0, 0.7)
U_smooth         = (false, 0.8)
TClustersize     = 4

# Geometry / gauge
boundary_conditions = "PBC"
gauge_choice        = "Mixed"

# Occupation-convergence knobs (NIL: deterministic, so single-shot probes suffice)
num_thermal_occ              = 0
num_measure_occ              = 1
measuring_interval_occ       = 1
n_tol                        = 1e-4
max_iter                     = 50
num_uncorrelated_measurments = 1

# Output
directory = joinpath(pwd(), "data", "NIL_local")
mkpath(directory)

# ============================================================================
# Run
# ============================================================================

println("=" ^ 70)
println("Local NIL run  -  wanted_n = $wanted_n,  L = $(Lx)x$(Ly),  beta = $beta,  seed = $seed")
println("Saving under: $directory")
println("=" ^ 70)

data = running_simulation_wanted_occupation(
    Lx, Ly, SC_L,
    U, t, V,
    seed,
    wanted_n, beta, B,
    num_measure, num_thermal, measuring_interval,
    w_list,
    initial_guess,
    U_disorder,
    diag_method,
    pairing_symmetry,
    w_conductivity_range,
    U_impurity;
    directory                    = directory,
    resume_path                  = directory,
    num_cores                    = 1,
    savedata                     = true,
    saving_interval              = 500,
    saveplots                    = false,
    date_folder                  = false,
    numbered                     = true,
    seed_folder                  = true,
    dump_time                    = Dates.Hour(1),
    U_puddle_params              = U_puddle_params,
    TClustersize_x               = TClustersize,
    TClustersize_y               = TClustersize,
    U_smooth                     = U_smooth,
    boundary_conditions          = boundary_conditions,
    gauge_choice                 = gauge_choice,
    num_thermal_occ              = num_thermal_occ,
    num_measure_occ              = num_measure_occ,
    measuring_interval_occ       = measuring_interval_occ,
    n_tol                        = n_tol,
    max_iter                     = max_iter,
    num_uncorrelated_measurments = num_uncorrelated_measurments,
)

n_meas    = data.current_state.num_current_measurments
achieved  = n_meas > 0 ? mean(data.measurement_data.n[1:n_meas]) : NaN
final_mu  = data.sim_data.mu
println("\nDone. measurements = $n_meas,  converged mu = $final_mu,  achieved n = $achieved")
