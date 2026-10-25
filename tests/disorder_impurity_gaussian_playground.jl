# disorder_impurity_gaussian_playground.jl — checks of the "impurity-gaussian-kernel"
# U map (no MC): at p = 1 the map is exactly uniform U for every kernel width σ, and
# mean(U) follows U·round(pN)/N for one seed and averaged over many seeds. Prints tables.
using Statistics
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "MC_Hubburd.jl"))

function make_u_kernel_vec(U::Float64, p::Float64; Lx::Int=12, Ly::Int=12, sigma::Float64=1.0, seed::Int=123, assert_mean=nothing)
    return make_U_params(U, Lx, Ly, "impurity-gaussian-kernel";
        impurity=p,
        seed=seed,
        U_puddle_params=(0.0, 10.0, sigma, 0.7),
        assert_positive=true,
        assert_mean=assert_mean
    )[1]
end

function print_p1_normalization(U::Float64; Lx::Int=12, Ly::Int=12, sigma_list=[0.6, 1.0, 1.8, 2.5], seed::Int=123)
    println("\n=== p=1 normalization checks ===")
    println("Lattice: $(Lx)x$(Ly), target U=$(U)")
    for sigma in sigma_list
        U_vec = make_u_kernel_vec(U, 1.0; Lx=Lx, Ly=Ly, sigma=sigma, seed=seed)
        max_dev = maximum(abs.(U_vec .- U))
        @printf("sigma=%4.2f | mean(U)=% .10f | std(U)=%.3e | max|U-U0|=%.3e\n", sigma, mean(U_vec), std(U_vec), max_dev)
    end
end

function print_mean_vs_p(U::Float64; Lx::Int=12, Ly::Int=12, sigma::Float64=1.0, seed::Int=123)
    println("\n=== mean(U) vs p (single seed) ===")
    N = Lx * Ly
    @printf("%-8s %-14s %-14s %-14s\n", "p", "mean(U)", "expected", "difference")
    for p in 0.0:0.1:1.0
        U_vec = make_u_kernel_vec(U, p; Lx=Lx, Ly=Ly, sigma=sigma, seed=seed)
        m = mean(U_vec)
        expected = U * round(Int, p * N) / N
        @printf("%-8.2f %-14.8f %-14.8f %-14.3e\n", p, m, expected, m - expected)
    end
end

function print_mean_vs_p_multiseed(U::Float64; Lx::Int=12, Ly::Int=12, sigma::Float64=1.0, seed0::Int=123, num_seeds::Int=40)
    println("\n=== mean(U) vs p (multi-seed) ===")
    N = Lx * Ly
    seed_list = collect(seed0:(seed0 + num_seeds - 1))
    @printf("%-8s %-14s %-14s %-14s\n", "p", "ensemble mean", "SEM", "expected")
    for p in 0.0:0.1:1.0
        means = [mean(make_u_kernel_vec(U, p; Lx=Lx, Ly=Ly, sigma=sigma, seed=s)) for s in seed_list]
        m = mean(means)
        sem = std(means) / sqrt(length(means))
        expected = U * round(Int, p * N) / N
        @printf("%-8.2f %-14.8f %-14.3e %-14.8f\n", p, m, sem, expected)
    end
end

function main()
    cd(PROJECT_ROOT)

    U = 4.0
    Lx, Ly = 12, 12
    sigma = 1.0

    println("Running impurity-gaussian-kernel playground in: $(pwd())")
    print_p1_normalization(U; Lx=Lx, Ly=Ly, seed=123)
    print_mean_vs_p(U; Lx=Lx, Ly=Ly, sigma=sigma, seed=123)
    print_mean_vs_p_multiseed(U; Lx=Lx, Ly=Ly, sigma=sigma, seed0=123, num_seeds=40)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
