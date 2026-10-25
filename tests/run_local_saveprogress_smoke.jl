# run_local_saveprogress_smoke.jl — smoke test of saving: a tiny 2×2 run for s-wave
# and for d-wave, then checks that results.jld2 and checkpoint.jld2 were written and
# that load_results reads the measurements back. Writes into local_smoke_runs/.
using Test
using Dates

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

function run_mode_smoke(pairing_symmetry::String, seed::Int)
    ts = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    out_dir = joinpath(@__DIR__, "..", "local_smoke_runs", "saveprogress_$(pairing_symmetry)_$(ts)")
    mkpath(out_dir)

    sim = SimData(
        Lx=2,
        Ly=2,
        SC_L=1,
        U=2.0,
        t=1.0,
        V=0.0,
        seed=seed,
        mu=0.0,
        beta=2.0,
        num_measure=3,
        num_thermal=3,
        measuring_interval=1,
        saving_interval=1,
        w_list=[-1.0, 0.0, 1.0],
        guess="uniform",
        U_disorder="uniform",
        diag_method="K0",
        pairing_symmetry=pairing_symmetry,
        directory=out_dir,
        num_cores=1,
        num_threads=1,
        w_conductivity_range=[0.1],
        delta_values_fixed=[0.1],
        delta_values_stiffness=[0.1],
        U_disorder_params=[0.0, 10.0, 0.7, 0.7],
        U_impurity=0.0,
        TCluster_size_x=1,
        TCluster_size_y=1,
        smooth_U=(false, 0.8),
        loop_random=false,
        B=0.0,
        gauge_choice="LandauX",
        boundary_conditions="PBC",
        dw_values=[0.1],
    )

    data = Lattice_Data(sim)
    sim_info_path = joinpath(out_dir, "sim.jld2")
    log_path = joinpath(out_dir, "simulation_log.txt")

    out = MonteCarloSimulation(
        data,
        out_dir;
        saveplots=false,
        savedata=true,
        saving_interval=1,
        dump_time=Dates.Millisecond(1),
        sim_info_path=sim_info_path,
        log_filename=log_path,
    )

    results_path = joinpath(out_dir, "results.jld2")
    checkpoint_path = joinpath(out_dir, "checkpoint.jld2")

    @test isfile(results_path)
    @test isfile(checkpoint_path)

    loaded = load_results(out_dir; to_load_checkpoint=false)
    @test loaded !== nothing
    @test loaded.current_state.num_current_measurments >= 1

    println(
        "OK mode=", pairing_symmetry,
        " dumped_meas=", loaded.current_state.num_current_measurments,
        " out_dir=", out_dir,
        " final_step=", out.current_state.step_count,
    )

    return out_dir
end

println("Running local save_progress smoke runs (s-wave + d-wave)")
out_s = run_mode_smoke("s-wave", 101)
out_d = run_mode_smoke("d-wave", 202)
println("All local save_progress smoke runs passed.")
println("s-wave path: ", out_s)
println("d-wave path: ", out_d)
