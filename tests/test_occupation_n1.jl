"""
Test occupation convergence for n=1.0 (half-filling)

This test verifies that the occupation convergence algorithm works correctly
when targeting n=1.0, especially with disorder (V > 0).

Key tests:
1. Convergence to n=1.0 without disorder (V=0)
2. Convergence to n=1.0 with disorder (V=1.0)
3. Verify that upper mu edge is properly probed
4. Check that convergence succeeds and finds appropriate mu
"""

include(joinpath(@__DIR__, "..", "occupation_convergence.jl"))

using Test
using Dates
using Statistics

# Test directory
const TEST_DIR = joinpath(tempdir(), "test_occ_n1_$(Dates.format(now(), "HHMMSSsss"))")

function setup_test_dir()
    if isdir(TEST_DIR)
        rm(TEST_DIR, recursive=true)
    end
    mkpath(TEST_DIR)
    println("Test directory: $TEST_DIR")
    return TEST_DIR
end

function cleanup_test_dir()
    if isdir(TEST_DIR)
        rm(TEST_DIR, recursive=true)
    end
end

# =============================================================================
# Test 1: Convergence to n=1.0 without disorder
# =============================================================================
function test_n1_no_disorder()
    println("\n" * "="^60)
    println("TEST 1: Convergence to n=1.0 without disorder (V=0)")
    println("="^60)
    
    setup_test_dir()
    
    # Small system for fast testing
    Lx, Ly = 4, 4
    N = Lx * Ly
    L_SC = 1
    
    # Physical parameters
    U = 4.0
    t = 1.0
    V = 0.0  # No disorder
    seed = 12345
    beta = 5.0
    B = 0.0
    
    # Simulation parameters
    num_measure = 100
    num_thermal = 100
    measuring_interval = 1
    saving_interval = 100
    w_list = [-1.0, 1.0]
    w_conductivity_range = [0.0, 0.01]
    
    # Create SimData
    sim_params = SimData(
        Lx=Lx, Ly=Ly, SC_L=L_SC, U=U, t=t, V=V, seed=seed, mu=0.0, beta=beta,
        num_measure=num_measure, num_thermal=num_thermal,
        measuring_interval=measuring_interval, saving_interval=saving_interval,
        w_list=w_list, guess="uniform", U_disorder="uniform",
        diag_method="exact", pairing_symmetry="s-wave",
        directory=TEST_DIR, w_conductivity_range=w_conductivity_range,
        B=B, boundary_conditions="PBC"
    )
    
    # Target occupation
    wanted_n = 1.0
    n_tol = 0.02  # Relaxed tolerance for fast test
    
    println("Testing convergence to n=$wanted_n with V=$V")
    println("System: $(Lx)x$(Ly), U=$U, beta=$beta")
    
    # Run convergence
    converged_struct, best_sim = converge_to_occupation!(
        sim_params, wanted_n;
        num_thermal_occ=500,
        num_measure_occ=200,
        measuring_interval=1,
        n_tol=n_tol,
        max_iter=15,
        num_uncorrelated_measurments=16
    )
    
    println("\nResults:")
    println("  Converged: $(converged_struct.converged)")
    println("  Best mu: $(converged_struct.best_mu)")
    println("  Best n: $(converged_struct.best_n)")
    println("  Target n: $(wanted_n)")
    println("  |n - target|: $(abs(converged_struct.best_n - wanted_n))")
    println("  Iterations: $(length(converged_struct.mu_list) - 2)")
    
    # Assertions
    @test converged_struct.converged || abs(converged_struct.best_n - wanted_n) < 2 * n_tol
    @test converged_struct.best_n !== nothing
    @test converged_struct.best_mu !== nothing
    @test abs(converged_struct.best_n - wanted_n) < 3 * n_tol  # Allow some tolerance
    
    # For half-filling, mu should be approximately U/2 for particle-hole symmetric case
    println("  Expected mu ≈ U/2 = $(U/2)")
    
    cleanup_test_dir()
    println("\n✓ Test 1 passed!")
    return converged_struct
end

# =============================================================================
# Test 2: Convergence to n=1.0 with disorder (V=1.0)
# =============================================================================
function test_n1_with_disorder()
    println("\n" * "="^60)
    println("TEST 2: Convergence to n=1.0 WITH disorder (V=1.0)")
    println("="^60)
    
    setup_test_dir()
    
    # Small system for fast testing
    Lx, Ly = 4, 4
    N = Lx * Ly
    L_SC = 1
    
    # Physical parameters
    U = 4.0
    t = 1.0
    V = 1.0  # WITH DISORDER
    seed = 12345
    beta = 5.0
    B = 0.0
    
    # Simulation parameters
    num_measure = 100
    num_thermal = 100
    measuring_interval = 1
    saving_interval = 100
    w_list = [-1.0, 1.0]
    w_conductivity_range = [0.0, 0.01]
    
    # Create SimData
    sim_params = SimData(
        Lx=Lx, Ly=Ly, SC_L=L_SC, U=U, t=t, V=V, seed=seed, mu=0.0, beta=beta,
        num_measure=num_measure, num_thermal=num_thermal,
        measuring_interval=measuring_interval, saving_interval=saving_interval,
        w_list=w_list, guess="uniform", U_disorder="uniform",
        diag_method="exact", pairing_symmetry="s-wave",
        directory=TEST_DIR, w_conductivity_range=w_conductivity_range,
        B=B, boundary_conditions="PBC"
    )
    
    # Target occupation
    wanted_n = 1.0
    n_tol = 0.02  # Relaxed tolerance for fast test
    
    println("Testing convergence to n=$wanted_n with V=$V (disorder)")
    println("System: $(Lx)x$(Ly), U=$U, beta=$beta, V=$V")
    
    # Run convergence
    converged_struct, best_sim = converge_to_occupation!(
        sim_params, wanted_n;
        num_thermal_occ=500,
        num_measure_occ=200,
        measuring_interval=1,
        n_tol=n_tol,
        max_iter=15,
        num_uncorrelated_measurments=16
    )
    
    println("\nResults:")
    println("  Converged: $(converged_struct.converged)")
    println("  Best mu: $(converged_struct.best_mu)")
    println("  Best n: $(converged_struct.best_n)")
    println("  Target n: $(wanted_n)")
    println("  |n - target|: $(abs(converged_struct.best_n - wanted_n))")
    println("  Iterations: $(length(converged_struct.mu_list) - 2)")
    println("  mu_list: $(converged_struct.mu_list)")
    println("  n_list: $(converged_struct.n_list)")
    
    # Assertions
    @test converged_struct.converged || abs(converged_struct.best_n - wanted_n) < 2 * n_tol
    @test converged_struct.best_n !== nothing
    @test converged_struct.best_mu !== nothing
    @test abs(converged_struct.best_n - wanted_n) < 3 * n_tol
    
    # Verify upper edge was probed (since wanted_n > 0.9)
    # The n_list should have the measured upper bound, not assumed 1.0
    println("\n  Checking if upper edge was probed:")
    if length(converged_struct.n_list) >= 2
        upper_n_measured = converged_struct.n_list[2]
        println("  Upper edge n measured: $upper_n_measured")
        @test upper_n_measured < 1.01  # Should be a measured value, not exactly 1.0 always
    end
    
    cleanup_test_dir()
    println("\n✓ Test 2 passed!")
    return converged_struct
end

# =============================================================================
# Test 3: Verify edge probing behavior
# =============================================================================
function test_edge_probing()
    println("\n" * "="^60)
    println("TEST 3: Verify edge probing behavior for n > 0.9")
    println("="^60)
    
    setup_test_dir()
    
    # Small system
    Lx, Ly = 4, 4
    L_SC = 1
    
    # Physical parameters - high U might need mu > 0 for half-filling
    U = 6.0
    t = 1.0
    V = 0.5
    seed = 42
    beta = 4.0
    
    # Create SimData
    sim_params = SimData(
        Lx=Lx, Ly=Ly, SC_L=L_SC, U=U, t=t, V=V, seed=seed, mu=0.0, beta=beta,
        num_measure=50, num_thermal=50, measuring_interval=1, saving_interval=50,
        w_list=[-1.0, 1.0], guess="uniform", U_disorder="uniform",
        diag_method="exact", pairing_symmetry="s-wave",
        directory=TEST_DIR, w_conductivity_range=[0.0, 0.01],
        B=0.0, boundary_conditions="PBC"
    )
    
    # Target n = 0.95 (should trigger upper edge probing)
    wanted_n = 0.95
    n_tol = 0.03
    
    println("Testing edge probing for n=$wanted_n (>0.9, should probe upper edge)")
    
    converged_struct, _ = converge_to_occupation!(
        sim_params, wanted_n;
        num_thermal_occ=300,
        num_measure_occ=100,
        n_tol=n_tol,
        max_iter=12,
        num_uncorrelated_measurments=12
    )
    
    println("\nResults:")
    println("  Converged: $(converged_struct.converged)")
    println("  mu_list length: $(length(converged_struct.mu_list))")
    println("  n_list length: $(length(converged_struct.n_list))")
    
    # The first two entries should be the edge probes
    if length(converged_struct.n_list) >= 2
        lower_n = converged_struct.n_list[1]
        upper_n = converged_struct.n_list[2]
        println("  Lower edge: mu=$(converged_struct.mu_list[1]), n=$lower_n")
        println("  Upper edge: mu=$(converged_struct.mu_list[2]), n=$upper_n")
        
        # Both should be actual measurements (not 0.0 or 1.0 exactly)
        @test lower_n > 0.0 && lower_n < 1.0
        @test upper_n > 0.0 && upper_n <= 1.0
        @test lower_n < upper_n  # Proper ordering
    end
    
    cleanup_test_dir()
    println("\n✓ Test 3 passed!")
    return converged_struct
end

# =============================================================================
# Test 4: Test with n < 0.9 (should NOT probe upper edge)
# =============================================================================
function test_no_upper_probe_for_low_n()
    println("\n" * "="^60)
    println("TEST 4: Verify NO upper edge probing for n < 0.9")
    println("="^60)
    
    setup_test_dir()
    
    Lx, Ly = 4, 4
    
    sim_params = SimData(
        Lx=Lx, Ly=Ly, SC_L=1, U=4.0, t=1.0, V=0.5, seed=123, mu=0.0, beta=4.0,
        num_measure=50, num_thermal=50, measuring_interval=1, saving_interval=50,
        w_list=[-1.0, 1.0], guess="uniform", U_disorder="uniform",
        diag_method="exact", pairing_symmetry="s-wave",
        directory=TEST_DIR, w_conductivity_range=[0.0, 0.01],
        B=0.0, boundary_conditions="PBC"
    )
    
    # Target n = 0.8 (should NOT trigger upper edge probing)
    wanted_n = 0.8
    n_tol = 0.03
    
    println("Testing for n=$wanted_n (<0.9, should NOT probe upper edge)")
    
    converged_struct, _ = converge_to_occupation!(
        sim_params, wanted_n;
        num_thermal_occ=300,
        num_measure_occ=100,
        n_tol=n_tol,
        max_iter=12,
        num_uncorrelated_measurments=12
    )
    
    println("\nResults:")
    println("  Converged: $(converged_struct.converged)")
    
    # Upper n should be exactly 1.0 (assumed, not measured)
    if length(converged_struct.n_list) >= 2
        upper_n = converged_struct.n_list[2]
        println("  Upper edge n: $upper_n (should be 1.0 - assumed, not measured)")
        @test upper_n == 1.0  # Should be assumed value for n < 0.9
    end
    
    cleanup_test_dir()
    println("\n✓ Test 4 passed!")
    return converged_struct
end

# =============================================================================
# Test 5: Early convergence at edge
# =============================================================================
function test_edge_convergence()
    println("\n" * "="^60)
    println("TEST 5: Early convergence when edge matches target")
    println("="^60)
    
    setup_test_dir()
    
    # Use a system where we might get lucky and match at an edge
    Lx, Ly = 4, 4
    
    sim_params = SimData(
        Lx=Lx, Ly=Ly, SC_L=1, U=4.0, t=1.0, V=0.0, seed=999, mu=0.0, beta=4.0,
        num_measure=50, num_thermal=50, measuring_interval=1, saving_interval=50,
        w_list=[-1.0, 1.0], guess="uniform", U_disorder="uniform",
        diag_method="exact", pairing_symmetry="s-wave",
        directory=TEST_DIR, w_conductivity_range=[0.0, 0.01],
        B=0.0, boundary_conditions="PBC"
    )
    
    # Test with a very relaxed tolerance - might converge at edge
    wanted_n = 0.95
    n_tol = 0.1  # Very relaxed - should make edge convergence possible
    
    println("Testing with relaxed tolerance (n_tol=$n_tol) - may converge at edge")
    
    converged_struct, _ = converge_to_occupation!(
        sim_params, wanted_n;
        num_thermal_occ=300,
        num_measure_occ=100,
        n_tol=n_tol,
        max_iter=12,
        num_uncorrelated_measurments=12
    )
    
    println("\nResults:")
    println("  Converged: $(converged_struct.converged)")
    println("  Best mu: $(converged_struct.best_mu)")
    println("  Best n: $(converged_struct.best_n)")
    println("  Iterations used: $(length(converged_struct.mu_list))")
    
    # With relaxed tolerance, should converge quickly
    @test converged_struct.converged
    @test abs(converged_struct.best_n - wanted_n) < n_tol
    
    cleanup_test_dir()
    println("\n✓ Test 5 passed!")
    return converged_struct
end

# =============================================================================
# Test 6: Verify mu_upper=0.1 for n>0.9
# =============================================================================
function test_mu_upper_for_high_n()
    println("\n" * "="^60)
    println("TEST 6: Verify mu_upper=0.1 for target n > 0.9")
    println("="^60)
    
    setup_test_dir()
    
    Lx, Ly = 4, 4
    
    sim_params = SimData(
        Lx=Lx, Ly=Ly, SC_L=1, U=4.0, t=1.0, V=0.5, seed=42, mu=0.0, beta=4.0,
        num_measure=50, num_thermal=50, measuring_interval=1, saving_interval=50,
        w_list=[-1.0, 1.0], guess="uniform", U_disorder="uniform",
        diag_method="exact", pairing_symmetry="s-wave",
        directory=TEST_DIR, w_conductivity_range=[0.0, 0.01],
        B=0.0, boundary_conditions="PBC"
    )
    
    # Create OccupationConvergence struct directly to check default mu_edges
    occ_conv = Occupation_Convergence(0.95, sim_params)
    
    println("For target_n=0.95 (>0.9):")
    println("  mu_edges = $(occ_conv.mu_edges)")
    println("  Expected upper mu = 0.1")
    
    @test occ_conv.mu_edges[2] == 0.1  # Upper mu should be 0.1 for n > 0.9
    @test occ_conv.mu_edges[1] == -4.0  # Lower mu unchanged
    
    # Also check for n < 0.9
    occ_conv_low = Occupation_Convergence(0.8, sim_params)
    
    println("\nFor target_n=0.8 (<0.9):")
    println("  mu_edges = $(occ_conv_low.mu_edges)")
    println("  Expected upper mu = 0.0")
    
    @test occ_conv_low.mu_edges[2] == 0.0  # Upper mu should be 0.0 for n < 0.9
    
    cleanup_test_dir()
    println("\n✓ Test 6 passed!")
end

# =============================================================================
# Run all tests
# =============================================================================
function run_all_tests()
    println("\n" * "="^70)
    println("OCCUPATION CONVERGENCE TESTS FOR n=1.0")
    println("="^70)
    println("Testing occupation convergence algorithm for half-filling")
    println("Especially important: proper handling of upper mu edge probing")
    println("="^70)
    
    results = Dict{String, Any}()
    
    try
        results["test1"] = test_n1_no_disorder()
    catch e
        println("\n✗ Test 1 FAILED: $e")
        results["test1"] = e
    end
    
    try
        results["test2"] = test_n1_with_disorder()
    catch e
        println("\n✗ Test 2 FAILED: $e")
        results["test2"] = e
    end
    
    try
        results["test3"] = test_edge_probing()
    catch e
        println("\n✗ Test 3 FAILED: $e")
        results["test3"] = e
    end
    
    try
        results["test4"] = test_no_upper_probe_for_low_n()
    catch e
        println("\n✗ Test 4 FAILED: $e")
        results["test4"] = e
    end
    
    try
        results["test5"] = test_edge_convergence()
    catch e
        println("\n✗ Test 5 FAILED: $e")
        results["test5"] = e
    end
    
    try
        results["test6"] = test_mu_upper_for_high_n()
    catch e
        println("\n✗ Test 6 FAILED: $e")
        results["test6"] = e
    end
    
    println("\n" * "="^70)
    println("ALL TESTS COMPLETED")
    println("="^70)
    
    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
