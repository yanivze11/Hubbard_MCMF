# End-to-end test of running_simulation_wanted_occupation with the k-space
# mean-field wiring active (mf_starting_mu seeds the mu bracket; Initial_state
# seeds Delta and phi from the coupled MF saddle).
#
# Exercises the WHOLE flow for a Q=0 run: MF starting mu -> occupation
# convergence (MC on Delta AND phi) -> main Monte-Carlo simulation. Checks that
# it converges near the target density, that phi is active for the density
# channel and stays exactly zero for the baseline, and that the returned state /
# measurements are finite and sane. Small system, no disk IO (savedata=false).
#
# NOTE: MF is only the seed; the reported n is the MC result, so tolerances are
# loose (small system, few stochastic measurements).
#
# REGRESSION: num_uncorrelated_measurments (24) > num_measure_occ (20) on purpose,
# so a single measurement batch (<= the 20-slot buffer) can never satisfy the
# request and n_uncorrelated_measurments MUST run its batch loop. That loop used
# to bump num_measure to a hardcoded floor of 32 and overflow the 20-slot,
# savedata=false buffer -> BoundsError at n[1:num_current]. Keep this shape so
# the buffer-size clamp stays covered.

using Test, Statistics, Dates

include(joinpath(@__DIR__, "..", "occupation_convergence.jl"))

# Small but adequately-thermalized wanted-occupation run (slower integration
# test, not a unit test). Enough thermal steps that n is a real signal, not
# noise the bracket logic would chase. Returns (data, converged_struct).
function run_wanted(; alpha=2/3, beta_phi=1/3, Q=[0.0, 0.0], target_n=0.85, seed=12345)
    dir = mktempdir()
    return running_simulation_wanted_occupation(
        4, 4, 1, 3.0, 1.0, 0.0, seed, target_n, 5.0, 0.0,    # L.., U, t, V, seed, wanted_n, beta, B
        1000, 2000, 1,                                        # num_measure, num_thermal, measuring_interval
        [0.0, 0.1],                                          # w_list
        "uniform", "uniform", "K0", "s-wave",
        [0.1], 0.0;                                          # w_conductivity_range, U_impurity
        directory=dir, savedata=false, seed_folder=false, resume_path="",
        num_thermal_occ=2000, num_measure_occ=20, measuring_interval_occ=1,
        n_tol=0.02, max_iter=10, num_uncorrelated_measurments=24,   # 24 > 20 forces the batch loop
        alpha=alpha, beta_phi=beta_phi, Q_density=Q,
        return_converegede_state=true,
    )
end

@testset "running_simulation_wanted_occupation end-to-end (MF-wired, Q=0)" begin

    @testset "scheme a (alpha=2/3, beta_phi=1/3): converges, phi active" begin
        target_n = 0.85
        data, conv = run_wanted(alpha=2/3, beta_phi=1/3, Q=[0.0, 0.0], target_n=target_n)

        # scheme flowed through into the actual simulation
        @test data isa LatticeData
        @test data.sim_data.alpha == 2/3
        @test data.sim_data.beta_phi == 1/3
        @test data.sim_data.Q_density == [0.0, 0.0]

        # a mu was chosen and the achieved density is in the right neighbourhood
        @test isfinite(conv.best_mu)
        @test isfinite(conv.best_n)
        @test abs(conv.best_n - target_n) < 0.12

        # density channel is genuinely active and finite
        @test any(!iszero, data.current_state.phi)
        @test all(isfinite, data.current_state.phi)
        @test all(isfinite, abs.(data.current_state.Delta_s))

        # the main simulation actually ran and measured a sane density
        @test data.current_state.step_count > 0
        n_meas = data.measurement_data.n[1:max(data.current_state.num_current_measurments, 1)]
        @test all(isfinite, n_meas)
        @test 0.0 < mean(n_meas) < 2.0
    end

    @testset "baseline (beta_phi=0): converges, phi stays exactly zero" begin
        target_n = 0.85
        data, conv = run_wanted(alpha=1.0, beta_phi=0.0, Q=[0.0, 0.0], target_n=target_n)

        @test data.sim_data.beta_phi == 0.0
        @test isfinite(conv.best_mu)
        @test abs(conv.best_n - target_n) < 0.12
        @test all(iszero, data.current_state.phi)          # channel OFF everywhere
        @test data.current_state.step_count > 0
    end
end

println("wanted-occupation end-to-end test complete.")
