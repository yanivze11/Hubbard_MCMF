# ============================================================================
# tests/test_spectrum_merge.jl — read-time ω-stitching (get_spectrum) and the
# physics-variant enumeration (variant_groups / get_curve / get_observable).
#
#   julia -O1 tests/test_spectrum_merge.jl
# ============================================================================
using Test, JLD2, Printf, Dates

include(joinpath(@__DIR__, "..", "means_db.jl"))

const ROOT = joinpath(tempdir(), "means_db_spec_" * string(hash(time_ns()) % 100000))
rm(ROOT; recursive=true, force=true)

cobs(v; err) = Dict{String,Any}("value" => v, "error" => err,
                                "correlation_time" => 1.0,
                                "num_uncorrelated_measurements" => 100,
                                "num_measurements" => 20000)

# A conductivity entry on a chosen ω window, one δ value. value/err are [δ][ω].
function cond_entry(; U=3.0, beta=50.0, V=1.0, dseed=0, alpha=1.0, beta_phi=0.0,
                     Qx=0.0, Qy=0.0, w, deltas=[0.05], vals, errs)
    Dict{String,Any}(
        "U"=>U, "wanted_n"=>0.875, "Lx"=>8, "Ly"=>8, "beta"=>beta,
        "diag_method"=>"K0", "V"=>V, "disorder_seed"=>dseed,
        "alpha"=>alpha, "beta_phi"=>beta_phi, "Qx"=>Qx, "Qy"=>Qy,
        "t"=>1.0, "SC_L"=>1, "impurity"=>0.0, "U_disorder"=>"uniform",
        "pairing_symmetry"=>"s-wave", "boundary_condition"=>"PBC",
        "gauge_choice"=>"Mixed", "guess"=>"uniform", "max_iter"=>100,
        "n_tol"=>0.001, "num_thermal"=>5000, "num_measure"=>20000,
        "w_conductivity_range"=>w, "delta_fixed_values"=>deltas,
        "sim_num"=>1, "seeds"=>[1], "start_time"=>Dates.now(),
        "convergence"=>Dict{String,Any}("best_mu"=>-1.2, "converged"=>true),
        "means"=>Dict{String,Any}("conductivity_regular_xx_fixed"=>cobs(vals; err=errs)),
    )
end

@testset "spectrum merge + variants" begin

@testset "ω-window stitching (single seed, disjoint + overlap)" begin
    # window A: ω = 0,1,2 ; window B: ω = 2,3,4 (they SHARE ω=2).
    a = cond_entry(w=[0.0,1.0,2.0], vals=[[10.0,11.0,12.0]], errs=[[1.0,1.0,1.0]])
    b = cond_entry(w=[2.0,3.0,4.0], vals=[[99.0,13.0,14.0]], errs=[[1.0,1.0,1.0]])
    save_entry(a; root=ROOT, source="t"); save_entry(b; root=ROOT, source="t")

    s = get_spectrum(; U=3.0, Lx=8, V=1.0, beta=50.0, wanted_n=0.875,
                       obs=:conductivity_regular_xx_fixed, roots=[ROOT], warn_varied=false)
    @test s.w == [0.0,1.0,2.0,3.0,4.0]                 # union grid, sorted
    @test length(s.deltas) == 1 && s.deltas[1] ≈ 0.05
    row = s.value[1]
    @test row[1] ≈ 10.0 && row[2] ≈ 11.0               # only in A
    @test row[4] ≈ 13.0 && row[5] ≈ 14.0               # only in B
    # ω=2 is in BOTH (12 err 1, 99 err 1) -> inverse-variance = simple mean here
    @test row[3] ≈ (12.0 + 99.0)/2
end

@testset "disorder average over union coverage" begin
    r2 = joinpath(ROOT, "avg"); rm(r2; recursive=true, force=true)
    # seed 0 covers ω=0,1 ; seed 1 covers ω=0,1,2. ω=2 only seed 1 (partial avg).
    save_entry(cond_entry(dseed=0, w=[0.0,1.0],     vals=[[4.0,4.0]],     errs=[[1.0,1.0]]); root=r2, source="t")
    save_entry(cond_entry(dseed=1, w=[0.0,1.0,2.0], vals=[[6.0,6.0,7.0]], errs=[[1.0,1.0,1.0]]); root=r2, source="t")
    s = get_spectrum(; U=3.0, Lx=8, V=1.0, beta=50.0, wanted_n=0.875,
                       obs=:conductivity_regular_xx_fixed, roots=[r2], warn_varied=false)
    @test s.w == [0.0,1.0,2.0]
    @test s.n_samples[1] == [2,2,1]                    # ω=2 covered by one seed
    @test s.value[1][1] ≈ 5.0                          # mean(4,6)
    @test s.value[1][3] ≈ 7.0                          # only seed 1
end

@testset "variant_groups enumerates distinct physics" begin
    r3 = joinpath(ROOT, "var"); rm(r3; recursive=true, force=true)
    save_entry(cond_entry(dseed=0, beta_phi=0.0, w=[0.0,1.0], vals=[[1.0,1.0]], errs=[[1.0,1.0]]); root=r3, source="t")
    save_entry(cond_entry(dseed=1, beta_phi=0.0, w=[0.0,1.0], vals=[[1.0,1.0]], errs=[[1.0,1.0]]); root=r3, source="t")
    save_entry(cond_entry(dseed=0, beta_phi=1/3, alpha=0.5, w=[0.0,1.0], vals=[[2.0,2.0]], errs=[[1.0,1.0]]); root=r3, source="t")

    hits = find_all(; roots=[r3], U=3.0, Lx=8, V=1.0, beta=50.0, wanted_n=0.875)
    g = variant_groups(hits; ignore=("disorder_seed",))
    @test length(g) == 2                               # channel off (2 seeds) + channel on (1)
    @test g[1].n == 2                                  # sorted by count desc
    @test any(gr -> get(gr.params, "beta_phi", 0.0) != 0.0, g)
    # strict raises on multiple variants
    @test_throws ErrorException get_observable(; roots=[r3], U=3.0, Lx=8, V=1.0,
        beta=50.0, wanted_n=0.875, obs=:conductivity_regular_xx_fixed, strict=true)
end

end

println("\nAll spectrum-merge tests passed.")
