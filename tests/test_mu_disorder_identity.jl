# Identity/hash tests for the mu-U disorder params in means_db.
# Verifies: mu_disorder="none" hashes identically to a legacy dict that never had
# the fields; the three variants are distinct; f and disorder_seed split identity
# only when the disorder is ON (CONDITIONAL_IDENTITY).

using Test, JLD2, Printf, SHA, Dates, Statistics

include(joinpath(@__DIR__, "..", "means_db.jl"))

# Clean base: V=0, uniform U, no impurity -> the ONLY disorder source is mu_disorder,
# so disorder_seed is identity-relevant iff mu_disorder != "none".
base(; kw...) = merge(Dict{String,Any}(
    "U" => 4.0, "wanted_n" => 0.9, "Lx" => 8, "Ly" => 8,
    "beta" => 5.0, "diag_method" => "K0-TCA", "V" => 0.0,
), Dict{String,Any}(string(k) => v for (k, v) in kw))

@testset "mu-U disorder identity hashing" begin
    legacy = base()                                                    # no mu fields at all
    none   = base(mu_disorder="none", mu_disorder_f=0.2, disorder_seed=3)  # off, fields present

    @testset "none == legacy (old data unaffected)" begin
        @test params_hash(none) == params_hash(legacy)
        # f and disorder_seed are irrelevant while the disorder is off
        @test params_hash(base(mu_disorder="none", mu_disorder_f=0.4)) == params_hash(legacy)
        @test params_hash(base(disorder_seed=1)) == params_hash(base(disorder_seed=2))
    end

    @testset "three variants are distinct and != none" begin
        io = base(mu_disorder="interaction-only", mu_disorder_f=0.2, disorder_seed=1)
        cp = base(mu_disorder="coupled",          mu_disorder_f=0.2, disorder_seed=1)
        po = base(mu_disorder="potential-only",   mu_disorder_f=0.2, disorder_seed=1)
        hs = params_hash.([legacy, io, cp, po])
        @test length(unique(hs)) == 4
    end

    @testset "f splits identity when the disorder is on" begin
        h1 = params_hash(base(mu_disorder="coupled", mu_disorder_f=0.1, disorder_seed=1))
        h2 = params_hash(base(mu_disorder="coupled", mu_disorder_f=0.2, disorder_seed=1))
        @test h1 != h2
    end

    @testset "disorder_seed splits identity when the disorder is on" begin
        h1 = params_hash(base(mu_disorder="coupled", mu_disorder_f=0.2, disorder_seed=1))
        h2 = params_hash(base(mu_disorder="coupled", mu_disorder_f=0.2, disorder_seed=2))
        @test h1 != h2
    end
end

println("mu-U disorder identity test complete.")
