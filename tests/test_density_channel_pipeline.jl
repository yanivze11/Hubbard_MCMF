# End-to-end cluster-pipeline tests for the density-channel scheme wiring:
#   * a legacy means_db entry (no scheme fields) stays loadable and becomes
#     queryable by the new params, which the index fills with their registered
#     defaults (baseline) WITHOUT rewriting the data file.
#   * running_simulation_wanted_occupation with beta_phi != 0 finds mu while
#     doing MC on BOTH Delta and phi, and phi is active in the result.
#
# A fresh temp means_db root is used so nothing touches real data.

using Test, JLD2, Printf, SHA, Dates, Statistics

# MEANS_DB_ROOT is captured as a const at include time, so set it FIRST.
const TMP_DB = mktempdir()
ENV["MEANS_DB_ROOT"] = TMP_DB
ENV["MEANS_DB_CLUSTER"] = "local"

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
include(joinpath(@__DIR__, "..", "occupation_convergence.jl"))

@testset "density-channel pipeline" begin

    @testset "legacy entry: loads + queryable by new params (index defaults)" begin
        # A legacy-style completed entry: identity-complete params, no scheme fields.
        legacy = Dict{String,Any}(
            "U" => 3.0, "wanted_n" => 0.875, "Lx" => 8, "Ly" => 8,
            "beta" => 5.0, "diag_method" => "K0", "V" => 1.0, "disorder_seed" => 1,
            "means" => Dict{String,Any}("n" => Dict("value" => 0.87, "error" => 0.01)),
        )
        rp = entry_relpath(legacy)
        path = joinpath(TMP_DB, rp)
        save_entry(legacy; root=TMP_DB, source="test")
        @test isfile(path)

        # The data file itself is NOT rewritten with the new params...
        stored = _open_jld2(f -> f["params"], path)
        @test !haskey(stored, "beta_phi")

        # ...but the (rebuildable) index fills the registered defaults, so old
        # data is queryable by the density-channel params as baseline.
        idx = rebuild_index(; root=TMP_DB, verbose=false)
        @test all(c -> c in idx["columns"], ["alpha", "beta_phi", "Qx", "Qy"])

        @test length(find_entries(; root=TMP_DB, index=idx, beta_phi=0.0)) == 1
        @test isempty(find_entries(; root=TMP_DB, index=idx, beta_phi=1/3))
        @test length(find_entries(; root=TMP_DB, index=idx, alpha=1.0, Qx=0.0, Qy=0.0)) == 1

        # And it still loads without error.
        si = load_means(; root=TMP_DB, verbose=false)
        @test si isa AbstractDict && !isempty(si)
    end

    @testset "mu-finding runs MC on Delta AND phi (scheme c)" begin
        dir = mktempdir()
        data, conv = running_simulation_wanted_occupation(
            4, 4, 1, 3.0, 1.0, 1.0, 12345, 0.85, 5.0, 0.0,
            30, 15, 1,                          # num_measure, num_thermal, measuring_interval
            collect(range(-3.0, 3.0, length=41)),
            "uniform", "uniform", "K0", "s-wave",
            [0.1], 0.0;                         # w_conductivity_range, U_impurity
            directory=dir, savedata=false, seed_folder=false, resume_path="",
            num_thermal_occ=20, num_measure_occ=20, measuring_interval_occ=1,
            n_tol=0.05, max_iter=4, num_uncorrelated_measurments=4,
            alpha=2/3, beta_phi=1/3, Q_density=Float64[pi, pi],   # scheme c
            return_converegede_state=true,
        )
        @test data.sim_data.beta_phi == 1/3
        @test isfinite(conv.best_mu)                       # a mu was chosen
        @test isfinite(conv.best_n)
        @test any(!iszero, data.current_state.phi)         # phi was actually sampled
        @test all(isfinite, data.current_state.phi)
    end
end

println("density-channel pipeline test complete.")
