# Local smoke/thermalization test for the density-channel (phi) decoupling.
#
# Exercises the three schemes the project cares about, plus the legacy baseline:
#   baseline : alpha=1,   beta_phi=0,   Q=(0,0)      -> density channel OFF
#   option a : alpha=2/3, beta_phi=1/3, Q=(0,0)      -> split, uniform (Hartree)
#   option b : alpha=1,   beta_phi=1,   Q=(0,0)      -> BdG + full Hartree
#   option c : alpha=2/3, beta_phi=1/3, Q=(pi,pi)    -> split, staggered CDW
#
# Checks: the run completes, the action stays finite, phi stays exactly zero for
# the baseline (legacy behaviour preserved) and evolves for the active schemes,
# and the action stabilizes (thermalizes) rather than drifting/blowing up.

using Test
using Dates
using Statistics

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

function build_sim(; alpha, beta_phi, Q, method="exact", seed=2024, num_thermal=100, num_measure=10)
    SimData(
        Lx=4, Ly=4, SC_L=1,
        U=3.0, t=1.0, V=0.0, seed=seed, mu=-1.5, beta=20.0,
        num_measure=num_measure, num_thermal=num_thermal,
        measuring_interval=1, saving_interval=num_measure,
        guess="uniform", U_disorder="uniform",
        diag_method=method, pairing_symmetry="s-wave",
        num_cores=1, num_threads=1,
        U_impurity=0.0, TCluster_size_x=2, TCluster_size_y=2,
        loop_random=true,
        alpha=alpha, beta_phi=beta_phi, Q_density=Q,
    )
end

# Run thermalization while recording the action trajectory (no disk IO).
function thermalize_trace(sim)
    data = Lattice_Data(sim)
    method = sim.diag_method
    action_of(d) = method == "K0" ? d.current_state.S_K0 : d.current_state.S
    trace = Float64[action_of(data)]
    while data.current_state.step_count <= sim.num_thermal
        MC_stepV2!(data; method=method, pairing=sim.pairing_symmetry, loop_random=sim.loop_random)
        push!(trace, action_of(data))
    end
    return data, trace
end

# A crude "thermalized" check: the action is finite throughout and the second
# half of the trajectory is not systematically worse (higher) than the first.
function looks_thermalized(trace)
    all(isfinite, trace) || return false
    n = length(trace)
    n < 8 && return true
    h = div(n, 2)
    mean(trace[h+1:end]) <= mean(trace[1:h]) + 3*std(trace[1:h]) + 1.0
end

@testset "density channel (phi) decoupling" begin

    @testset "baseline: density channel OFF keeps phi at zero" begin
        data, trace = thermalize_trace(build_sim(alpha=1.0, beta_phi=0.0, Q=[0.0, 0.0]))
        @test all(iszero, data.current_state.phi)        # phi never touched
        @test all(isfinite, trace)
        @test looks_thermalized(trace)
    end

    @testset "option a: split, uniform (Q=0)" begin
        data, trace = thermalize_trace(build_sim(alpha=2/3, beta_phi=1/3, Q=[0.0, 0.0]))
        @test any(!iszero, data.current_state.phi)        # phi moved
        @test all(isfinite, data.current_state.phi)
        @test looks_thermalized(trace)
        @test all(==(1.0), data.const_struct.density_sign) # Q=0 -> uniform +1
    end

    @testset "option b: BdG + full Hartree (alpha=beta=1, Q=0)" begin
        data, trace = thermalize_trace(build_sim(alpha=1.0, beta_phi=1.0, Q=[0.0, 0.0]))
        @test any(!iszero, data.current_state.phi)
        @test all(isfinite, data.current_state.phi)
        @test looks_thermalized(trace)
    end

    @testset "option c: split, staggered CDW (Q=(pi,pi))" begin
        sim = build_sim(alpha=2/3, beta_phi=1/3, Q=Float64[pi, pi])
        data, trace = thermalize_trace(sim)
        @test any(!iszero, data.current_state.phi)
        @test all(isfinite, data.current_state.phi)
        @test looks_thermalized(trace)
        # staggered sign: neighbours differ, |sign| == 1 everywhere
        ds = data.const_struct.density_sign
        @test all(x -> isapprox(abs(x), 1.0; atol=1e-10), ds)
        @test any(x -> x < 0, ds) && any(x -> x > 0, ds)
    end

    @testset "TCluster path runs with phi (option c)" begin
        sim = build_sim(alpha=2/3, beta_phi=1/3, Q=Float64[pi, pi], method="K0-TCA")
        data, trace = thermalize_trace(sim)
        @test all(isfinite, trace)
        @test all(isfinite, data.current_state.phi)
    end

    @testset "K0 path runs with phi (option a)" begin
        sim = build_sim(alpha=2/3, beta_phi=1/3, Q=Float64[0.0, 0.0], method="K0")
        data, trace = thermalize_trace(sim)
        @test all(isfinite, trace)
        @test any(!iszero, data.current_state.phi)
    end

    @testset "phi observables recorded in measurement stream" begin
        # Full sim (thermalize + measure + dump) exercises make_measurement_K!.
        # NOTE: the in-memory measurement_data buffer is flushed to zeros on dump,
        # so measurements must be read back with load_results (as the real
        # analysis pipeline does), not from out.measurement_data.
        function run_short(; alpha, beta_phi, Q)
            sim = SimData(
                Lx=4, Ly=4, SC_L=1, U=3.0, t=1.0, V=0.0, seed=7, mu=-0.6, beta=5.0,
                num_measure=6, num_thermal=40, measuring_interval=1, saving_interval=1000,
                guess="uniform", U_disorder="uniform", diag_method="K0",
                pairing_symmetry="s-wave", num_cores=1, num_threads=1, U_impurity=0.0,
                TCluster_size_x=2, TCluster_size_y=2, loop_random=true,
                alpha=alpha, beta_phi=beta_phi, Q_density=Q,
            )
            dir = mktempdir()
            MonteCarloSimulation(Lattice_Data(sim), dir; saveplots=false,
                savedata=true, saving_interval=1000, dump_time=Dates.Hour(1),
                sim_info_path=joinpath(dir, "sim.jld2"),
                log_filename=joinpath(dir, "log.txt"))
            loaded = load_results(dir; to_load_checkpoint=false)
            nm = loaded.current_state.num_current_measurments
            return loaded.measurement_data, nm
        end

        # option a: uniform Hartree (Q=0) away from half-filling -> phi nonzero.
        md, nm = run_short(alpha=2/3, beta_phi=1/3, Q=Float64[0.0, 0.0])
        @test hasfield(typeof(md), :phi_Q)
        @test nm >= 1
        @test all(isfinite, md.phi_mean[1:nm])
        @test all(isfinite, md.phi_abs_mean[1:nm])
        @test all(isfinite, md.phi_Q[1:nm])
        @test any(>(0.0), md.phi_abs_mean[1:nm])          # phi field is nonzero
        @test all(md.phi_abs_mean[1:nm] .>= abs.(md.phi_mean[1:nm]) .- 1e-12)

        # baseline: density channel off — phi observables identically zero.
        md0, nm0 = run_short(alpha=1.0, beta_phi=0.0, Q=Float64[0.0, 0.0])
        @test all(iszero, md0.phi_mean[1:nm0])
        @test all(iszero, md0.phi_abs_mean[1:nm0])
        @test all(iszero, md0.phi_Q[1:nm0])
    end
end

println("density-channel test complete.")
