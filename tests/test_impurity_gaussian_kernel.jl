# test_impurity_gaussian_kernel.jl — tests of the "impurity-gaussian-kernel" U map:
# reproducible for a seed, p = 0 gives 0 and p = 1 gives uniform U, mean(U) follows
# U·round(pN)/N, and in make_U_params the mean is kept (or rescaled with assert_mean).
println("=" ^ 64)
println("Testing impurity-gaussian-kernel disorder")
println("=" ^ 64)

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

using Test
using Statistics

function run_tests()
    @testset "make_U_impurity_gaussian_kernel" begin
        @testset "Reproducibility and shape" begin
            Lx, Ly = 6, 4
            N = Lx * Ly
            U_val = 3.0
            sigma = 1.1

            U_kernel = make_U_impurity_gaussian_kernel(Lx, Ly, U_val; impurity_fraction=0.4, sigma=sigma, seed=123)
            U_kernel2 = make_U_impurity_gaussian_kernel(Lx, Ly, U_val; impurity_fraction=0.4, sigma=sigma, seed=123)
            U_kernel3 = make_U_impurity_gaussian_kernel(Lx, Ly, U_val; impurity_fraction=0.4, sigma=sigma, seed=456)

            @test length(U_kernel) == N
            @test all(U_kernel .>= 0.0)
            @test U_kernel == U_kernel2
            @test U_kernel != U_kernel3
        end

        @testset "p=0 and p=1 limits" begin
            Lx, Ly = 12, 10
            U_val = 4.0
            sigma = 1.0

            U_zero = make_U_impurity_gaussian_kernel(Lx, Ly, U_val; impurity_fraction=0.0, sigma=sigma, seed=123)
            U_full = make_U_impurity_gaussian_kernel(Lx, Ly, U_val; impurity_fraction=1.0, sigma=sigma, seed=123)

            @test all(U_zero .== 0.0)
            @test maximum(abs.(U_full .- U_val)) < 1e-12
            @test std(U_full) < 1e-12
        end

        @testset "Mean follows finite-size expectation" begin
            Lx, Ly = 10, 8
            N = Lx * Ly
            U_val = 4.0
            sigma = 1.0
            p_vals = [0.0, 0.2, 0.5, 0.8, 1.0]

            means = Float64[]
            for p in p_vals
                U_vec = make_U_impurity_gaussian_kernel(Lx, Ly, U_val; impurity_fraction=p, sigma=sigma, seed=123)
                m = mean(U_vec)
                expected = U_val * round(Int, p * N) / N
                push!(means, m)
                @test isapprox(m, expected; atol=1e-12)
            end

            @test all(diff(means) .>= -1e-12)
        end
    end

    @testset "make_U_params integration" begin
        Lx, Ly = 6, 4
        N = Lx * Ly
        U = 2.0
        sigma = 1.0
        p = 0.3

        @testset "Default keeps natural mean(U) vs p" begin
            U_params_default, _, _ = make_U_params(U, Lx, Ly, "impurity-gaussian-kernel";
                                                   U_puddle_params=(0.0, 10.0, sigma, 0.7),
                                                   impurity=p, seed=123)
            expected = U * round(Int, p * N) / N
            @test isapprox(mean(U_params_default), expected; atol=1e-12)
        end

        @testset "Optional assert_mean rescales to U" begin
            U_params_scaled, _, _ = make_U_params(U, Lx, Ly, "impurity-gaussian-kernel";
                                                  U_puddle_params=(0.0, 10.0, sigma, 0.7),
                                                  impurity=p, seed=123, assert_mean=true)
            @test isapprox(mean(U_params_scaled), U; atol=1e-10)
        end

        @testset "p=1 is uniform in make_U_params" begin
            U_params_full, _, _ = make_U_params(U, Lx, Ly, "impurity-gaussian-kernel";
                                                U_puddle_params=(0.0, 10.0, sigma, 0.7),
                                                impurity=1.0, seed=123)
            @test maximum(abs.(U_params_full .- U)) < 1e-12
        end
    end

    println("\nAll impurity-gaussian-kernel tests passed.")
end

Base.invokelatest(run_tests)
