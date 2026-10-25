# local_test_running_once_impurity_gaussian.jl — smoke test: a short 6×6 K0-TCA run
# of running_simulation_once with the "impurity-gaussian-kernel" U disorder
# (p = 0.5, σ = 1); checks it finishes and prints the U-map statistics.
# Writes into test_code/tmp_local_running_once/.
using Dates
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "MC_Hubburd.jl"))

function run_local_test()
    println(repeat("=", 72))
    println("Local smoke test: running_simulation_once with impurity-gaussian-kernel")
    println(repeat("=", 72))

    Lx, Ly = 6, 6
    L_SC = 1
    U = 4.0
    t = 1.0
    V = 0.0
    seed = 12345
    mu = -0.9
    beta = 2.0
    B = 0.0

    # Keep this run short so it can be used as a local smoke test.
    num_measure = 100
    num_thermal = 100
    measuring_interval = 1
    w_list = collect(-2.0:0.5:2.0)
    initial_guess = "uniform"
    U_disorder = "impurity-gaussian-kernel"
    diag_method = "K0-TCA"
    pairing_symmetry = "s-wave"
    w_conductivity_range = collect(logrange(1e-3, 1.0, 6))

    impurity_fraction = 0.5
    sigma = 1.0

    # Temporary output directory under test_code to avoid cluttering main data paths.
    out_dir = joinpath(PROJECT_ROOT, "test_code", "tmp_local_running_once")
    mkpath(out_dir)

    start_t = now()
    data = running_simulation_once(
        Lx,
        Ly,
        L_SC,
        U,
        t,
        V,
        seed,
        mu,
        beta,
        B,
        num_measure,
        num_thermal,
        measuring_interval,
        w_list,
        initial_guess,
        U_disorder,
        diag_method,
        pairing_symmetry,
        w_conductivity_range,
        impurity_fraction;
        directory=out_dir,
        num_cores=1,
        savedata=false,
        saveplots=false,
        date_folder=false,
        numbered=false,
        seed_folder=false,
        U_puddle_params=(0.0, 10.0, sigma, 0.7),
        TClustersize_x=4,
        TClustersize_y=4,
        U_smooth=(false, 0.8),
        gauge_choice="Mixed",
        boundary_conditions="PBC",
    )
    elapsed = now() - start_t

    @assert data.sim_data.Lx == Lx
    @assert data.sim_data.Ly == Ly
    @assert data.sim_data.U_disorder == U_disorder

    u_vec = data.const_struct.U_params

    println("Run completed in: ", elapsed)
    println("Lattice: ", data.sim_data.Lx, "x", data.sim_data.Ly)
    println("Measurements collected: ", data.current_state.num_current_measurments)
    println("U stats from generator: mean=", round(mean(u_vec), digits=6), ", std=", round(std(u_vec), digits=6))
    println("Smoke test passed.")
    return data
end

dat = run_local_test();
