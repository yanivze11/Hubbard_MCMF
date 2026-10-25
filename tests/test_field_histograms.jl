# test_field_histograms.jl — tests of the φ and Δ field histograms (details below).
using Test
using Dates
using JLD2

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

# Tests for lifetime MC-field histograms (item 1).
#
# Covers: clamp-into-edge-bin binning, the exact per-measurement click invariant
# (sum == measurements * N, i.e. no value is ever dropped), accumulation across a
# resume (constant-memory accumulator that survives checkpoint/reload), and the
# nbins == 0 off switch.

function make_sim(out_dir::String; num_measure::Int64,
                  phi_nbins::Int64=100, abs_nbins::Int64=100, arg_nbins::Int64=50)
    return SimData(
        Lx=3, Ly=3, SC_L=1, U=2.0, t=1.0, V=0.0, seed=2024, mu=0.0, beta=2.0,
        num_measure=num_measure, num_thermal=2, measuring_interval=1, saving_interval=3,
        w_list=[-1.0, 0.0, 1.0], guess="uniform", U_disorder="uniform",
        diag_method="K0", pairing_symmetry="s-wave", directory=out_dir,
        num_cores=1, num_threads=1, w_conductivity_range=[0.1],
        delta_values_fixed=[0.1], delta_values_stiffness=[0.1],
        U_disorder_params=[0.0, 10.0, 0.7, 0.7], U_impurity=0.0,
        TCluster_size_x=1, TCluster_size_y=1, smooth_U=(false, 0.8),
        loop_random=false, B=0.0, gauge_choice="LandauX", boundary_conditions="PBC",
        dw_values=[0.1], snapshot_interval=0,
        phi_hist_nbins=phi_nbins, Delta_hist_abs_nbins=abs_nbins,
        Delta_hist_arg_nbins=arg_nbins,
    )
end

function run_sim(sim::SimData, out_dir::String)
    data = Lattice_Data(sim)
    return MonteCarloSimulation(
        data, out_dir; saveplots=false, savedata=true, saving_interval=sim.saving_interval,
        dump_time=Dates.Hour(1), sim_info_path=joinpath(out_dir, "sim.jld2"),
        log_filename=joinpath(out_dir, "simulation_log.txt"),
    )
end

@testset "field histograms" begin
    N = 3 * 3

    # ---------- clamp-into-edge-bin binning (unit) ----------
    @testset "hist_bin_clamped" begin
        # 100 bins over [-10, 10]: width 0.2
        @test hist_bin_clamped(-10.0, -10.0, 10.0, 100) == 1     # lower edge -> bin 1
        @test hist_bin_clamped(-999.0, -10.0, 10.0, 100) == 1    # far below  -> bin 1
        @test hist_bin_clamped(10.0, -10.0, 10.0, 100) == 100    # upper edge -> last bin (clamped)
        @test hist_bin_clamped(999.0, -10.0, 10.0, 100) == 100   # far above  -> last bin
        @test hist_bin_clamped(0.0, -10.0, 10.0, 100) == 51      # just above midpoint
        @test hist_bin_clamped(-0.0001, -10.0, 10.0, 100) == 50  # just below midpoint
        # arg over [0, 2pi], 50 bins
        @test hist_bin_clamped(0.0, 0.0, 2pi, 50) == 1
        @test hist_bin_clamped(2pi - 1e-9, 0.0, 2pi, 50) == 50
        @test hist_bin_clamped(Float64(pi), 0.0, 2pi, 50) == 26
    end

    # ---------- defaults ----------
    sd_def = SimData(Lx=4, U=2.0, beta=2.0)
    @test sd_def.phi_hist_nbins == 100
    @test sd_def.phi_hist_range == [-10.0, 10.0]
    @test sd_def.Delta_hist_abs_nbins == 100
    @test sd_def.Delta_hist_abs_range == [0.0, 10.0]
    @test sd_def.Delta_hist_arg_nbins == 50
    # fresh state: phi 1D, Delta 2D (amplitude x phase), zero-initialized
    cs0 = Lattice_Data(sd_def).current_state
    @test length(cs0.phi_hist) == 100
    @test size(cs0.Delta_hist) == (100, 50)
    @test ndims(cs0.Delta_hist) == 2
    @test all(==(0), cs0.phi_hist)
    @test all(==(0), cs0.Delta_hist)

    # ---------- the memory estimate counts the accumulators ----------
    sd_off = SimData(Lx=4, U=2.0, beta=2.0, phi_hist_nbins=0, Delta_hist_abs_nbins=0, Delta_hist_arg_nbins=0)
    @test field_hist_bytes(sd_def) == sizeof(cs0.phi_hist) + sizeof(cs0.Delta_hist)
    @test field_hist_bytes(sd_off) == 0
    @test estimate_memory_bytes(sd_def).lattice - estimate_memory_bytes(sd_off).lattice == (100 + 100 * 50) * 8

    ts = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

    # ---------- click invariant: sum == measurements * N (no drops) ----------
    out = joinpath(@__DIR__, "..", "local_smoke_runs", "hist_$(ts)")
    mkpath(out)
    M = 6
    res = run_sim(make_sim(out; num_measure=M), out)
    cs = res.current_state
    @test cs.num_current_measurments == M
    @test length(cs.phi_hist) == 100
    @test size(cs.Delta_hist) == (100, 50)
    @test sum(cs.phi_hist) == M * N
    # each site contributes ONE click to the 2D joint histogram per measurement
    @test sum(cs.Delta_hist) == M * N

    # histograms live in the checkpoint (checkpoint-only, per design)
    loaded = load_checkpoint(out)
    @test loaded !== nothing
    @test sum(loaded.current_state.phi_hist) == M * N
    @test loaded.current_state.phi_hist == cs.phi_hist
    @test loaded.current_state.Delta_hist == cs.Delta_hist
    @test size(loaded.current_state.Delta_hist) == (100, 50)

    # ---------- resume accumulates (does NOT reset) ----------
    resumed = load_checkpoint(out)
    resumed.sim_data.num_measure = 2 * M      # ask for M more measurements
    res2 = MonteCarloSimulation(
        resumed, out; saveplots=false, savedata=true, saving_interval=resumed.sim_data.saving_interval,
        dump_time=Dates.Hour(1), sim_info_path=joinpath(out, "sim.jld2"),
        log_filename=joinpath(out, "simulation_log.txt"),
    )
    @test res2.current_state.num_current_measurments == 2 * M
    @test sum(res2.current_state.phi_hist) == 2 * M * N        # counts continued, not wiped
    @test sum(res2.current_state.Delta_hist) == 2 * M * N

    # ---------- nbins = 0 => histogram OFF ----------
    out_off = joinpath(@__DIR__, "..", "local_smoke_runs", "hist_off_$(ts)")
    mkpath(out_off)
    res_off = run_sim(make_sim(out_off; num_measure=4, phi_nbins=0, abs_nbins=0, arg_nbins=0), out_off)
    @test isempty(res_off.current_state.phi_hist)
    @test isempty(res_off.current_state.Delta_hist)

    # ---------- Delta joint histogram is OFF unless BOTH axes are on ----------
    out_half = joinpath(@__DIR__, "..", "local_smoke_runs", "hist_half_$(ts)")
    mkpath(out_half)
    res_half = run_sim(make_sim(out_half; num_measure=3, phi_nbins=20, abs_nbins=100, arg_nbins=0), out_half)
    @test length(res_half.current_state.phi_hist) == 20
    @test sum(res_half.current_state.phi_hist) == 3 * N
    @test isempty(res_half.current_state.Delta_hist)   # arg axis off => whole joint hist off

    println("OK field-histogram test: dir=", out)
end
