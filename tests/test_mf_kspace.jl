# Unit tests for the standalone homogeneous k-space mean-field solver
# (mf_kspace_density_channel.jl): coupled pairing (Delta) + density (phi)
# channels at Q = (0,0).
#
# Validation is INTERNAL CONSISTENCY only (this is the MF minimizer / an initial
# guess for the MC, NOT the MC equilibrium):
#   * baseline (beta_phi=0, alpha=1): phi=0 and Delta == saddle_point_delta
#   * coupled fixed point satisfies the note's saddle relations exactly
#   * phi sits at a MINIMUM of the free energy (perturbing it raises S)
#   * the target_n solver reproduces the requested density for EVERY scheme,
#     including the over-complete scheme b where n(mu) at fixed mu is
#     discontinuous (this is the regression test for that fix)
#   * Q != 0 is rejected

using Test
using Statistics

# saddle_point_delta lives in MC_Hubburd.jl; the solver is its own module.
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
if !isdefined(@__MODULE__, :MFKSpaceDensityChannel)
    include(joinpath(@__DIR__, "..", "mf_kspace_density_channel.jl"))
end
using .MFKSpaceDensityChannel
const MF = MFKSpaceDensityChannel

@testset "MF k-space density channel (Q=0)" begin
    t = 1.0; U = 4.0; beta = 10.0; Lx = 8; Ly = 8

    @testset "baseline reduces to BCS saddle_point_delta" begin
        mu = 0.0
        res = solve_mf(; t=t, U=U, alpha=1.0, beta_phi=0.0, beta=beta,
                         Lx=Lx, Ly=Ly, mu=mu, Delta0=0.5, tol=1e-12, mixing=0.6)
        sp = saddle_point_delta(U, mu, beta, Lx, Ly; d_min=1e-4, t=t)[1]
        @test res.phi == 0.0
        @test isapprox(res.Delta, sp; atol=5e-3)
        @test abs(res.gap_residual) < 1e-8
        @test res.converged
    end

    @testset "coupled saddle relations (scheme a, alpha=2/3, beta_phi=1/3)" begin
        a = 2/3; bp = 1/3; mu = -1.0
        res = solve_mf(; t=t, U=U, alpha=a, beta_phi=bp, beta=beta,
                         Lx=Lx, Ly=Ly, mu=mu, Delta0=0.5, tol=1e-12)
        @test res.phi > 0.0                              # positive Hartree shift
        @test isapprox(res.phi, bp*U*res.n; atol=1e-8)   # phi = beta_phi U n
        @test res.Delta > 1e-6                           # nontrivial SC solution
        @test abs(res.gap_residual) < 1e-8               # gap equation holds
    end

    @testset "phi sits at a free-energy minimum" begin
        a = 2/3; bp = 1/3; mu = -1.0
        res = solve_mf(; t=t, U=U, alpha=a, beta_phi=bp, beta=beta,
                         Lx=Lx, Ly=Ly, mu=mu, Delta0=0.5, tol=1e-12)
        eps = MF._epsilon_grid(t, Lx, Ly)
        _, f0 = mf_free_energy(eps, mu, res.phi,        res.Delta, U, a, bp, beta; N=Lx*Ly)
        _, fp = mf_free_energy(eps, mu, res.phi + 0.02, res.Delta, U, a, bp, beta; N=Lx*Ly)
        _, fm = mf_free_energy(eps, mu, res.phi - 0.02, res.Delta, U, a, bp, beta; N=Lx*Ly)
        @test fp > f0
        @test fm > f0
    end

    @testset "target_n reproduces density for every scheme" begin
        schemes = [(:baseline, 1.0, 0.0),
                   (:a,        2/3, 1/3),
                   (:b,        1.0, 1.0)]   # scheme b: previously broken
        for (name, a, bp) in schemes, target in (0.3, 0.5, 0.9, 1.0, 1.3)
            r = solve_mf(; t=t, U=U, alpha=a, beta_phi=bp, beta=beta,
                           Lx=Lx, Ly=Ly, target_n=target, Delta0=0.5, tol=1e-12)
            @test isapprox(r.n, target; atol=1e-6)       # density hit
            @test r.mu_converged                          # honest flag
            @test isapprox(r.phi, bp*U*r.n; atol=1e-6)   # phi = beta_phi U n
            if bp == 0.0
                @test r.phi == 0.0
            end
        end
    end

    @testset "fixed-mu scheme b runaway is a valid fixed point" begin
        # With strong density coupling at fixed mu the uniform saddle is unstable
        # and n runs to the full band; that limit is still self-consistent.
        r = solve_mf(; t=t, U=U, alpha=1.0, beta_phi=1.0, beta=beta,
                       Lx=Lx, Ly=Ly, mu=-0.5, Delta0=0.5, tol=1e-12)
        @test isapprox(r.n, 2.0; atol=1e-3)
        @test isapprox(r.phi, 1.0*U*r.n; atol=1e-6)      # phi = beta_phi U n still holds
    end

    @testset "Q != 0 is rejected" begin
        @test_throws ErrorException solve_mf(; t=t, U=U, alpha=1.0, beta_phi=1.0,
            beta=beta, Lx=Lx, Ly=Ly, mu=0.0, Q=(pi, pi))
    end

    @testset "grid_factor gives a self-consistent solution" begin
        r = solve_mf(; t=t, U=U, alpha=2/3, beta_phi=1/3, beta=beta,
                       Lx=Lx, Ly=Ly, target_n=0.9, grid_factor=3, tol=1e-12)
        @test isapprox(r.n, 0.9; atol=1e-6)
        @test abs(r.gap_residual) < 1e-8
    end

    @testset "input guards" begin
        # must supply exactly one of mu / target_n
        @test_throws ErrorException solve_mf(; t=t, U=U, alpha=1.0, beta_phi=0.0,
            beta=beta, Lx=Lx, Ly=Ly)
        @test_throws ErrorException solve_mf(; t=t, U=U, alpha=1.0, beta_phi=0.0,
            beta=beta, Lx=Lx, Ly=Ly, mu=0.0, target_n=1.0)
        # target_n out of (0,2)
        @test_throws ErrorException solve_mf(; t=t, U=U, alpha=1.0, beta_phi=0.0,
            beta=beta, Lx=Lx, Ly=Ly, target_n=2.5)
    end
end
