# test_field_snapshots.jl — tests of the raw field snapshots (details below).
using Test
using Dates
using JLD2

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

# Smoke + correctness test for periodic raw-field snapshots (item 2).
#
# Verifies:
#   * field_snapshots.jld2 is created in a SEPARATE file,
#   * snapshots land exactly at the expected measurement indices,
#   * each snapshot stores phi + Delta_s of the right length and matching metadata,
#   * snapshot_interval = 0 disables the feature entirely.

function make_sim(out_dir::String; snapshot_interval::Int64, num_measure::Int64)
    return SimData(
        Lx=3,
        Ly=3,
        SC_L=1,
        U=2.0,
        t=1.0,
        V=0.0,
        seed=4321,
        mu=0.0,
        beta=2.0,
        num_measure=num_measure,
        num_thermal=2,
        measuring_interval=1,
        saving_interval=3,
        w_list=[-1.0, 0.0, 1.0],
        guess="uniform",
        U_disorder="uniform",
        diag_method="K0",
        pairing_symmetry="s-wave",
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
        snapshot_interval=snapshot_interval,
    )
end

function run_sim(sim::SimData, out_dir::String)
    data = Lattice_Data(sim)
    return MonteCarloSimulation(
        data, out_dir;
        saveplots=false, savedata=true, saving_interval=sim.saving_interval,
        dump_time=Dates.Hour(1),
        sim_info_path=joinpath(out_dir, "sim.jld2"),
        log_filename=joinpath(out_dir, "simulation_log.txt"),
    )
end

@testset "field snapshots" begin
    ts = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

    # ---- default constructor picks up the new field with the on-by-default value
    @test SimData(Lx=4, U=2.0, beta=2.0).snapshot_interval == 500

    # ---- snapshot_interval = 2 over 6 measurements -> snapshots at 2, 4, 6 ----
    out_on = joinpath(@__DIR__, "..", "local_smoke_runs", "snapshots_on_$(ts)")
    mkpath(out_on)
    sim = make_sim(out_on; snapshot_interval=2, num_measure=6)
    run_sim(sim, out_on)

    snap_path = joinpath(out_on, "field_snapshots.jld2")
    @test isfile(snap_path)

    jldopen(snap_path, "r") do file
        ks = keys(file)
        @test "sim_data" in ks
        snap_keys = sort(filter(k -> startswith(k, "snapshot_"), collect(ks)))
        @test snap_keys == ["snapshot_2", "snapshot_4", "snapshot_6"]

        N = 3 * 3
        for (k, expected_n) in (("snapshot_2", 2), ("snapshot_4", 4), ("snapshot_6", 6))
            snap = file[k]
            @test snap["num_current_measurments"] == expected_n
            @test length(snap["phi"]) == N
            @test length(snap["Delta_s"]) == N
            @test eltype(snap["Delta_s"]) <: Complex
            @test haskey(snap, "S")
            @test haskey(snap, "step_count")
        end
    end

    # ---- separation: snapshots file must not disturb results / checkpoint ----
    @test isfile(joinpath(out_on, "results.jld2"))
    @test isfile(joinpath(out_on, "checkpoint.jld2"))

    # ---- snapshot_interval = 0 disables snapshots entirely ----
    out_off = joinpath(@__DIR__, "..", "local_smoke_runs", "snapshots_off_$(ts)")
    mkpath(out_off)
    sim_off = make_sim(out_off; snapshot_interval=0, num_measure=4)
    run_sim(sim_off, out_off)
    @test !isfile(joinpath(out_off, "field_snapshots.jld2"))

    println("OK field-snapshot test: on-dir=", out_on, " off-dir=", out_off)
end
