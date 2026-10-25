# test_k0_tca_resume_smoke.jl — checkpoint/resume with K0-TCA: a 4×4 run is stopped
# early, then resumed from its checkpoint; checks the files, that the measurement count
# continues to the target, and that the joined n stream is complete and fluctuates.
using Test
using Dates
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

function build_small_k0_tca_sim(run_dir::String; num_measure::Int64=24, num_thermal::Int64=30, saving_interval::Int64=8)
    return SimData(
        Lx=4,
        Ly=4,
        SC_L=1,
        U=4.0,
        t=1.0,
        V=0.0,
        seed=123456,
        mu=-1.2,
        beta=8.0,
        num_measure=num_measure,
        num_thermal=num_thermal,
        measuring_interval=1,
        saving_interval=saving_interval,
        w_list=collect(-1.0:0.5:1.0),
        guess="rand",
        U_disorder="uniform",
        diag_method="K0-TCA",
        pairing_symmetry="s-wave",
        directory=run_dir,
        num_cores=1,
        num_threads=1,
        start_time=Dates.now(),
        w_conductivity_range=collect(-0.2:0.2:0.2),
        delta_values_fixed=[0.05],
        delta_values_stiffness=[0.05],
        U_disorder_params=[0.0, 10.0, 0.7, 0.7],
        U_impurity=1.0,
        TCluster_size_x=2,
        TCluster_size_y=2,
        smooth_U=(false, 0.8),
        loop_random=true,
        B=0.0,
        gauge_choice="LandauX",
        boundary_conditions="PBC",
        dw_values=[0.05]
    )
end

function summarize_n_trace(n_trace::Vector{Float64})
    n_total = length(n_trace)
    chunk = max(3, n_total ÷ 4)
    early = n_trace[1:chunk]
    mid = n_trace[max(1, n_total - 2 * chunk + 1):(n_total - chunk)]
    late = n_trace[(n_total - chunk + 1):n_total]

    fluct = std(n_trace)
    early_mean = mean(early)
    mid_mean = mean(mid)
    late_mean = mean(late)
    drift_mid_to_late = abs(late_mean - mid_mean)

    println("Occupation diagnostics:")
    @printf("  n measurements: %d\n", n_total)
    @printf("  std(n): %.6f\n", fluct)
    @printf("  early mean: %.6f\n", early_mean)
    @printf("  mid mean: %.6f\n", mid_mean)
    @printf("  late mean: %.6f\n", late_mean)
    @printf("  |late-mid| drift: %.6f\n", drift_mid_to_late)

    return fluct, drift_mid_to_late
end

function run_k0_tca_resume_smoke()
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    run_dir = joinpath(@__DIR__, "tmp_k0_tca_resume_" * stamp)
    mkpath(run_dir)

    sim_info_path = joinpath(run_dir, "sim.jld2")
    log_filename = joinpath(run_dir, "log.txt")
    results_path = joinpath(run_dir, "results.jld2")

    total_measure = Int64(24)
    partial_measure = Int64(8)

    println("="^70)
    println("K0-TCA checkpoint/resume smoke test")
    println("Run directory: ", run_dir)
    println("="^70)

    sim_data_partial = build_small_k0_tca_sim(run_dir; num_measure=partial_measure, num_thermal=30, saving_interval=partial_measure)

    println("\nPhase 1: start simulation and stop early")
    partial = MonteCarloSimulation(sim_data_partial, run_dir;
        saveplots=false,
        savedata=true,
        saving_interval=partial_measure,
        dump_time=Dates.Millisecond(1),
        sim_info_path=sim_info_path,
        log_filename=log_filename,
    )

    @test partial.current_state.num_current_measurments == partial_measure
    @test isfile(joinpath(run_dir, "checkpoint.jld2"))
    @test isfile(results_path)

    dumped_after_phase1 = count_dumped_measurements(results_path)
    println("  dumped measurements after phase 1: ", dumped_after_phase1)
    @test dumped_after_phase1 >= partial_measure

    println("\nPhase 2: resume unfinished simulation from checkpoint")
    resumed = load_checkpoint(run_dir)
    @test resumed !== nothing

    resumed.sim_data.num_measure = total_measure
    resumed.sim_data.saving_interval = partial_measure

    resumed = MonteCarloSimulation(resumed, run_dir;
        saveplots=false,
        savedata=true,
        saving_interval=partial_measure,
        dump_time=Dates.Millisecond(1),
        sim_info_path=sim_info_path,
        log_filename=log_filename,
    )

    @test resumed.current_state.num_current_measurments == total_measure

    merged = load_results(run_dir; to_load_checkpoint=true)
    @test merged !== nothing

    n_trace = merged.measurement_data.n
    @test length(n_trace) == total_measure

    fluct, drift_mid_to_late = summarize_n_trace(n_trace)

    # Hard checks: simulation ran, measurements exist, and occupation fluctuates.
    @test isfinite(fluct)
    @test fluct > 1e-8

    # Soft equilibrium indicator: should usually stabilize in late windows.
    # Keep non-failing to avoid flaky CI behavior in stochastic runs.
    if drift_mid_to_late > 0.08
        @warn "Late-window occupation drift is relatively large; increase thermalization for cleaner equilibrium." drift_mid_to_late
    end

    println("\nTest finished successfully.")
    println("Inspect results in: ", run_dir)
end

run_k0_tca_resume_smoke()
