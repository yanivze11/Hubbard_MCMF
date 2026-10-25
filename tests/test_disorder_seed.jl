# ============================================================================
# tests/test_disorder_seed.jl — the disorder realization must depend on
# `disorder_seed` alone, not on the Monte-Carlo seed.
#
#   julia -O1 tests/test_disorder_seed.jl
# ============================================================================
using Test, Random

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

@testset "disorder_seed" begin

@testset "make_V_matrix draws from its own seed" begin
    a = make_V_matrix(4, 4, 1.0; seed=1)
    b = make_V_matrix(4, 4, 1.0; seed=1)
    c = make_V_matrix(4, 4, 1.0; seed=2)
    @test a == b                     # reproducible
    @test a != c                     # a different sample
    @test length(a) == 16
    @test all(-0.5 .<= a .<= 0.5)    # V_i ~ U[-V/2, V/2] for V=1

    # V=0 => no disorder, whatever the seed
    @test all(make_V_matrix(4, 4, 0.0; seed=7) .== 0.0)
    @test make_V_matrix(4, 4, 0.0; seed=7) == make_V_matrix(4, 4, 0.0; seed=99)
end

@testset "it does NOT disturb the global RNG" begin
    # The old implementation called Random.seed!(seed+2), silently reseeding the
    # global stream. Nothing outside should be able to tell that we drew.
    Random.seed!(4242)
    expected = [rand() for _ in 1:5]
    Random.seed!(4242)
    make_V_matrix(6, 6, 1.0; seed=17)
    @test [rand() for _ in 1:5] == expected
end

@testset "SimData carries disorder_seed, defaulting to legacy behaviour" begin
    sd = SimData(Lx=4, U=3.0, V=1.0, seed=1001, beta=20.0)
    @test sd.disorder_seed == 0
    @test effective_disorder_seed(sd) == 1001      # 0 => derive from MC seed

    sd2 = SimData(Lx=4, U=3.0, V=1.0, seed=1001, beta=20.0, disorder_seed=7)
    @test effective_disorder_seed(sd2) == 7
end

@testset "one sample, several MC seeds" begin
    # This is the property the whole change exists for: two different MC seeds
    # on the same disordered sample must see the SAME V_i.
    s1 = SimData(Lx=4, U=3.0, V=1.0, seed=1001, beta=20.0, disorder_seed=1)
    s2 = SimData(Lx=4, U=3.0, V=1.0, seed=2002, beta=20.0, disorder_seed=1)
    v1 = make_V_matrix(s1.Lx, s1.Ly, s1.V; seed=effective_disorder_seed(s1))
    v2 = make_V_matrix(s2.Lx, s2.Ly, s2.V; seed=effective_disorder_seed(s2))
    @test v1 == v2

    # ...and two samples must differ even at the same MC seed
    s3 = SimData(Lx=4, U=3.0, V=1.0, seed=1001, beta=20.0, disorder_seed=2)
    v3 = make_V_matrix(s3.Lx, s3.Ly, s3.V; seed=effective_disorder_seed(s3))
    @test v1 != v3
end

@testset "old checkpoints still reconstruct" begin
    # dict2struct fills absent fields with defaults; a pre-disorder_seed
    # checkpoint must therefore still load, landing on the legacy sentinel.
    sd = SimData(Lx=4, U=3.0, V=0.0, seed=55, beta=1.0)
    d = struct2dict(sd)
    delete!(d, "disorder_seed")            # simulate an old file
    back = dict2struct(d)
    @test back isa SimData
    @test back.seed == 55
    @test back.disorder_seed == 0
    @test effective_disorder_seed(back) == 55
end

end

println("\nAll disorder_seed tests passed.")
