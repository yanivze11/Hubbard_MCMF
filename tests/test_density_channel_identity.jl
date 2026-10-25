# Identity/hash tests for the density-channel scheme params in means_db.
# Verifies: baseline (channel off) hashes identically to a legacy dict that
# never had the fields; the three active schemes get distinct identities; and
# alpha/Qx/Qy are ignored when beta_phi == 0 (CONDITIONAL_IDENTITY).

using Test, JLD2, Printf, SHA, Dates, Statistics

include(joinpath(@__DIR__, "..", "means_db.jl"))

# Minimal identity-complete params dict (required: U, wanted_n, Lx, Ly, beta, diag_method).
base(; kw...) = merge(Dict{String,Any}(
    "U" => 3.0, "wanted_n" => 0.875, "Lx" => 8, "Ly" => 8,
    "beta" => 5.0, "diag_method" => "K0", "V" => 1.0, "disorder_seed" => 1,
), Dict{String,Any}(string(k) => v for (k, v) in kw))

@testset "density-channel identity hashing" begin
    legacy   = base()                                              # no scheme fields at all
    baseline = base(alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0)       # channel off, fields present
    a = base(alpha=2/3, beta_phi=1/3, Qx=0.0, Qy=0.0)
    b = base(alpha=1.0, beta_phi=1.0, Qx=0.0, Qy=0.0)
    c = base(alpha=2/3, beta_phi=1/3, Qx=Float64(pi), Qy=Float64(pi))

    hL, hBase = params_hash(legacy), params_hash(baseline)
    hA, hB, hC = params_hash(a), params_hash(b), params_hash(c)

    @testset "baseline == legacy (old data unaffected)" begin
        @test hBase == hL
    end

    @testset "active schemes are pairwise distinct and != baseline" begin
        hs = [hL, hA, hB, hC]
        @test length(unique(hs)) == 4
    end

    @testset "alpha/Qx/Qy irrelevant when beta_phi == 0" begin
        # non-default alpha and Q but channel OFF -> must still equal baseline/legacy
        off_weird = base(alpha=2/3, beta_phi=0.0, Qx=Float64(pi), Qy=Float64(pi))
        @test params_hash(off_weird) == hL
    end

    @testset "beta_phi alone splits identity" begin
        # same alpha/Q as baseline but channel ON must differ from baseline
        @test params_hash(base(alpha=1.0, beta_phi=1/3, Qx=0.0, Qy=0.0)) != hL
    end
end

println("density-channel identity test complete.")
