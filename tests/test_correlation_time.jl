# Test file for find_correlation_time and the new "correlation" scheme in obs_std
# Tests that autocorrelation time is correctly estimated for synthetic correlated data

using Statistics
using Random

include(joinpath(@__DIR__, "..", "statistics.jl"))

println("="^60)
println("Testing find_correlation_time and correlation-based error estimation")
println("="^60)

# ============================================================================
# Helper: Generate AR(1) correlated data with known autocorrelation time
# ============================================================================
"""
Generate AR(1) process: x_{n+1} = ρ * x_n + √(1-ρ²) * ε_n
where ε_n ~ N(0, σ²)

The integrated autocorrelation time for AR(1) (using τ = 1 + 2Σρ^k) is:
τ_int = (1 + ρ) / (1 - ρ)

So for a given τ, we need: ρ = (τ - 1) / (τ + 1)
"""
function generate_ar1(n::Int, τ_target::Float64; σ::Float64=1.0, μ::Float64=0.0, seed::Int=42)
    Random.seed!(seed)
    
    # Calculate ρ from target τ: τ = (1+ρ)/(1-ρ) => ρ = (τ-1)/(τ+1)
    ρ = (τ_target - 1) / (τ_target + 1)
    
    # Generate AR(1) process
    x = zeros(n)
    x[1] = randn() * σ
    noise_scale = sqrt(1 - ρ^2) * σ
    
    for i in 2:n
        x[i] = ρ * x[i-1] + noise_scale * randn()
    end
    
    return x .+ μ, ρ
end

# ============================================================================
# Test 1: Uncorrelated data (τ ≈ 1)
# ============================================================================
println("\n" * "-"^60)
println("Test 1: Uncorrelated data (expected τ ≈ 0.5-1)")
println("-"^60)

Random.seed!(123)
uncorr_data = randn(10000)
τ_uncorr = find_correlation_time(uncorr_data)
println("Estimated τ for uncorrelated data: $(round(τ_uncorr, digits=2))")
println("Expected: ~0.5-1.0 (pure noise)")

if τ_uncorr < 3.0
    println("✓ PASS: τ is small for uncorrelated data")
else
    println("✗ FAIL: τ should be small for uncorrelated data")
end

# ============================================================================
# Test 2: Strongly correlated data (τ = 10)
# ============================================================================
println("\n" * "-"^60)
println("Test 2: AR(1) correlated data (target τ = 10)")
println("-"^60)

τ_target = 10.0
corr_data, ρ = generate_ar1(50000, τ_target; σ=1.0, μ=5.0)
τ_estimated = find_correlation_time(corr_data)

println("AR(1) coefficient ρ = $(round(ρ, digits=4))")
println("Target τ_int = $τ_target")
println("Estimated τ = $(round(τ_estimated, digits=2))")
println("Relative error: $(round(abs(τ_estimated - τ_target)/τ_target * 100, digits=1))%")

if abs(τ_estimated - τ_target) / τ_target < 0.3  # Within 30%
    println("✓ PASS: τ estimate within 30% of target")
else
    println("✗ FAIL: τ estimate too far from target")
end

# ============================================================================
# Test 3: Moderate correlation (τ = 5)
# ============================================================================
println("\n" * "-"^60)
println("Test 3: AR(1) correlated data (target τ = 5)")
println("-"^60)

τ_target = 5.0
corr_data2, ρ2 = generate_ar1(20000, τ_target; σ=2.0, μ=0.0)
τ_estimated2 = find_correlation_time(corr_data2)

println("AR(1) coefficient ρ = $(round(ρ2, digits=4))")
println("Target τ_int = $τ_target")
println("Estimated τ = $(round(τ_estimated2, digits=2))")
println("Relative error: $(round(abs(τ_estimated2 - τ_target)/τ_target * 100, digits=1))%")

if abs(τ_estimated2 - τ_target) / τ_target < 0.3
    println("✓ PASS: τ estimate within 30% of target")
else
    println("✗ FAIL: τ estimate too far from target")
end

# ============================================================================
# Test 4: Compare error estimation schemes
# ============================================================================
println("\n" * "-"^60)
println("Test 4: Compare error estimation schemes on correlated data")
println("-"^60)

# For AR(1), true SEM = σ / √N × √(2τ) where σ is the process std dev
n = 10000
τ_target = 8.0
test_data, ρ_test = generate_ar1(n, τ_target; σ=1.0, μ=10.0)
τ_est = find_correlation_time(test_data)

# True standard error for AR(1)
process_std = std(test_data)
raw_sem = process_std / sqrt(n)
theoretical_sem = raw_sem * sqrt(2 * τ_target)

# Estimated errors
err_raw = obs_std(test_data; scheme="raw")
err_corr = obs_std(test_data; scheme="correlation")
err_blocking = obs_std(test_data; scheme="blocking")

println("Sample size: $n")
println("Estimated τ: $(round(τ_est, digits=2)) (target: $τ_target)")
println()
println("Error estimates:")
println("  Raw SEM (ignoring correlations): $(round(err_raw, digits=5))")
println("  Theoretical SEM (√(2τ) corrected): $(round(theoretical_sem, digits=5))")
println("  Correlation scheme: $(round(err_corr, digits=5))")
println("  Old blocking scheme: $(round(err_blocking, digits=5))")
println()
println("Ratios to raw:")
println("  Theoretical: $(round(theoretical_sem/err_raw, digits=2))×")
println("  Correlation: $(round(err_corr/err_raw, digits=2))×")
println("  Blocking: $(round(err_blocking/err_raw, digits=2))×")

# The correlated schemes should give errors significantly larger than raw
if err_corr > 1.5 * err_raw
    println("\n✓ PASS: Correlation scheme correctly inflates error for correlated data")
else
    println("\n✗ FAIL: Correlation scheme should give larger error than raw for correlated data")
end

# ============================================================================
# Test 5: Verify correlation scheme gives sensible block size
# ============================================================================
println("\n" * "-"^60)
println("Test 5: Block size selection")
println("-"^60)

τ_est = find_correlation_time(test_data)
block_size = max(1, round(Int, 2 * τ_est))
num_blocks = div(length(test_data), block_size)

println("Estimated τ: $(round(τ_est, digits=2))")
println("Block size (2τ): $block_size")
println("Number of blocks: $num_blocks")
println("Expected independent samples: ~$(round(n / (2*τ_target), digits=0))")

if num_blocks > 10
    println("✓ PASS: Enough blocks for reliable statistics")
else
    println("⚠ WARNING: Few blocks may lead to unreliable error estimate")
end

# ============================================================================
# Test 6: Performance comparison
# ============================================================================
println("\n" * "-"^60)
println("Test 6: Performance comparison (correlation vs blocking)")
println("-"^60)

# Generate test data
perf_data, _ = generate_ar1(5000, 5.0)

# Time the correlation scheme
t_corr = @elapsed begin
    for _ in 1:100
        obs_std(perf_data; scheme="correlation")
    end
end

# Time the old blocking scheme
t_block = @elapsed begin
    for _ in 1:100
        obs_std(perf_data; scheme="blocking")
    end
end

println("Time for 100 calls on 5000 samples:")
println("  Correlation scheme: $(round(t_corr * 1000, digits=1)) ms")
println("  Blocking (fit) scheme: $(round(t_block * 1000, digits=1)) ms")
println("  Speedup: $(round(t_block/t_corr, digits=1))×")

if t_corr < t_block
    println("✓ PASS: Correlation scheme is faster")
else
    println("Note: Blocking scheme was faster (unexpected but not critical)")
end

# ============================================================================
# Test 7: Edge cases
# ============================================================================
println("\n" * "-"^60)
println("Test 7: Edge cases")
println("-"^60)

# Very short data
short_data = randn(20)
τ_short = find_correlation_time(short_data)
err_short = obs_std(short_data; scheme="correlation")
println("Short data (n=20): τ=$(round(τ_short, digits=2)), error=$(round(err_short, digits=4))")

# Constant data (no variation)
const_data = fill(5.0, 100)
τ_const = find_correlation_time(const_data)
err_const = obs_std(const_data; scheme="correlation")
println("Constant data: τ=$(τ_const), error=$(err_const)")

# Data with NaN
nan_data = [1.0, 2.0, NaN, 4.0]
err_nan = obs_std(nan_data; scheme="correlation")
println("Data with NaN: error=$(err_nan) (should be NaN)")

println("\n" * "="^60)
println("TESTS COMPLETE")
println("="^60)
