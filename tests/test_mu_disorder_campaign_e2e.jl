# test_mu_disorder_campaign_e2e.jl — end-to-end run of the μ–U disorder campaign path
# (details below). Writes into local_smoke_runs/.
using Test
using Dates
using JLD2

include(joinpath(@__DIR__, "..", "occupation_convergence.jl"))

# End-to-end test of the mu-U disorder campaign path (as fh_run_n.jl drives it):
# running_simulation_wanted_occupation with coupled disorder + field observables ON.
# The key correctness property is that histograms accumulate ONLY in production
# (never during the mu-finding convergence phase), which the click invariant
# sum(hist) == production_measurements * N pins down exactly.

@testset "mu-disorder campaign e2e" begin
    out = joinpath(@__DIR__, "..", "local_smoke_runs", "mu_campaign_" * Dates.format(Dates.now(), "HHMMSS"))
    mkpath(out)
    seed = 3
    N = 4 * 4
    num_measure = 8

    data = running_simulation_wanted_occupation(
        4, 4, 1, 4.0, 1.0, 0.0, seed, 0.9, 6.0, 0.0,
        num_measure, 4, 1,                       # num_measure, num_thermal, measuring_interval
        [0.0], "uniform", "uniform", "K0", "s-wave", [0.0], 0.0;
        directory=out, resume_path=out, savedata=true, saving_interval=num_measure,
        seed_folder=true, numbered=false, date_folder=false,
        TClustersize_x=1, TClustersize_y=1,
        num_thermal_occ=15, num_measure_occ=15, measuring_interval_occ=1,
        n_tol=0.05, max_iter=4, num_uncorrelated_measurments=8,
        # campaign settings: coupled mu-U disorder + field observables ON
        mu_disorder="coupled", mu_disorder_f=0.25,
        snapshot_interval=2,
        phi_hist_nbins=100, Delta_hist_abs_nbins=100, Delta_hist_arg_nbins=50,
    )

    cs = data.current_state
    @test cs.num_current_measurments == num_measure

    # coupled disorder actually took effect: floor(0.25*16)=4 empty sites
    @test count(data.const_struct.U_params .== 0.0) == 4
    @test count(data.const_struct.mu_shift .== 0.0) == 4     # unshifted on empty sites

    # histograms are sized and accumulated ONLY over production measurements
    @test length(cs.phi_hist) == 100
    @test size(cs.Delta_hist) == (100, 50)
    @test sum(cs.phi_hist) == num_measure * N               # no convergence-phase pollution
    @test sum(cs.Delta_hist) == num_measure * N

    # the production checkpoint holds the same histograms, and snapshots were written
    seed_dir = joinpath(out, "seed_$seed")
    loaded = load_checkpoint(seed_dir)
    @test loaded !== nothing
    @test sum(loaded.current_state.phi_hist) == num_measure * N
    @test isfile(joinpath(seed_dir, "field_snapshots.jld2"))
    jldopen(joinpath(seed_dir, "field_snapshots.jld2"), "r") do f
        snaps = filter(k -> startswith(k, "snapshot_"), keys(f))
        @test !isempty(snaps)                                # snapshots at 2,4,6,8
    end

    rm(out; recursive=true, force=true)
    println("OK mu-disorder campaign e2e")
end
