"""
Tests for mu_convergence information collection in sim_num_control.jl

Tests:
1. Seeds in same folder (from running_simulation_seeds_wanted_occupation)
2. Seeds in different folders (handled by merge_sims)
"""

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))

using Test
using Dates

# Test directory
const TEST_DIR = joinpath(tempdir(), "test_mu_convergence_$(Dates.format(now(), "HHMMSSsss"))")

function setup_test_dir()
    if isdir(TEST_DIR)
        rm(TEST_DIR, recursive=true)
    end
    mkpath(TEST_DIR)
    mkpath(joinpath(TEST_DIR, "num_control"))  # Required for save_sim_info
    println("Test directory: $TEST_DIR")
end

function cleanup_test_dir()
    if isdir(TEST_DIR)
        rm(TEST_DIR, recursive=true)
    end
end

# =============================================================================
# Helper: Create mock simulation data with convergence info
# =============================================================================
function create_mock_sim_data(; seed::Int64, mu::Float64, target_n::Float64, best_n::Float64, 
                               converged::Bool=true, n_tol::Float64=1e-3,
                               Lx::Int64=10, Ly::Int64=10, beta::Float64=10.0)
    return Dict{String, Any}(
        "seed" => seed,
        "Lx" => Lx,
        "Ly" => Ly,
        "beta" => beta,
        "mu" => mu,
        "num_current_measurments" => 100,
        "convergence" => Dict{String, Any}(
            "target_n" => target_n,
            "best_n" => best_n,
            "best_mu" => mu,
            "converged" => converged,
            "n_tol" => n_tol,
            "mu_list" => [-4.0, 0.0, mu],
            "n_list" => [0.0, 1.0, best_n]
        ),
        "means" => Dict{String, Any}(
            "n" => Dict("value" => best_n, "error" => 0.01),
            "Delta_mean" => Dict("value" => 0.1, "error" => 0.01)
        ),
        "status" => "finished"
    )
end

# =============================================================================
# Test 1: Seeds in same folder (running_simulation_seeds structure)
# =============================================================================
function test_seeds_same_folder()
    println("\n" * "="^60)
    println("TEST 1: Seeds in same folder (running_simulation_seeds structure)")
    println("="^60)
    
    setup_test_dir()
    
    try
        # Create a simulation folder structure like running_simulation_seeds creates
        sim_folder = joinpath(TEST_DIR, "1")
        mkpath(sim_folder)
        
        # Create the main sim.jld2 with multiple seeds in "simulations" dict
        target_n = 0.875
        seeds_data = Dict{Any, Any}()
        
        # Seed 1: mu = -1.50, best_n = 0.874
        seeds_data[111] = create_mock_sim_data(
            seed=111, mu=-1.50, target_n=target_n, best_n=0.874, converged=true
        )
        seeds_data[111]["dir_path"] = joinpath(sim_folder, "seed_111")
        
        # Seed 2: mu = -1.52, best_n = 0.876
        seeds_data[222] = create_mock_sim_data(
            seed=222, mu=-1.52, target_n=target_n, best_n=0.876, converged=true
        )
        seeds_data[222]["dir_path"] = joinpath(sim_folder, "seed_222")
        
        # Seed 3: mu = -1.48, best_n = 0.873 (not converged)
        seeds_data[333] = create_mock_sim_data(
            seed=333, mu=-1.48, target_n=target_n, best_n=0.873, converged=false
        )
        seeds_data[333]["dir_path"] = joinpath(sim_folder, "seed_333")
        
        # Create main sim.jld2
        sim_data = Dict{String, Any}(
            "Lx" => 10,
            "Ly" => 10,
            "beta" => 10.0,
            "seeds" => [111, 222, 333],
            "simulations" => seeds_data
        )
        
        JLD2.save(joinpath(sim_folder, "sim.jld2"), sim_data)
        
        # Load and process using sim_num_control functions
        sim_info = get_sim_info(TEST_DIR)
        sim_info = get_sim_info_means(sim_info)
        
        # Check the results
        @test haskey(sim_info, "1")
        @test haskey(sim_info["1"], "means")
        means = sim_info["1"]["means"]
        
        @test haskey(means, "mu_convergence")
        mu_conv = means["mu_convergence"]
        
        println("mu_convergence contents:")
        for (k, v) in mu_conv
            println("  $k: $v")
        end
        
        # Check mu tracking
        @test haskey(mu_conv, "mu_list")
        @test length(mu_conv["mu_list"]) == 3
        @test haskey(mu_conv, "mu_mean")
        @test isapprox(mu_conv["mu_mean"], mean([-1.50, -1.52, -1.48]), atol=1e-6)
        @test haskey(mu_conv, "mu_std")
        @test haskey(mu_conv, "mu_range")
        
        # Check n convergence tracking
        @test haskey(mu_conv, "target_n")
        @test mu_conv["target_n"] == target_n
        @test haskey(mu_conv, "best_n_list")
        @test length(mu_conv["best_n_list"]) == 3
        @test haskey(mu_conv, "best_n_mean")
        @test isapprox(mu_conv["best_n_mean"], mean([0.874, 0.876, 0.873]), atol=1e-6)
        
        # Check convergence flags
        @test haskey(mu_conv, "num_converged")
        @test mu_conv["num_converged"] == 2  # Only 2 out of 3 converged
        @test haskey(mu_conv, "num_seeds")
        @test mu_conv["num_seeds"] == 3
        @test haskey(mu_conv, "all_converged")
        @test mu_conv["all_converged"] == false  # Not all converged
        
        @test haskey(mu_conv, "n_tol")
        
        println("✓ Test passed: Seeds in same folder")
        return true
    catch e
        println("✗ Test failed: $e")
        rethrow(e)
    finally
        cleanup_test_dir()
    end
end

# =============================================================================
# Test 2: Seeds in different folders (merge_sims scenario)
# =============================================================================
function test_seeds_different_folders_merge()
    println("\n" * "="^60)
    println("TEST 2: Seeds in different folders (merge_sims scenario)")
    println("="^60)
    
    setup_test_dir()
    
    try
        target_n = 0.5
        beta = 20.0
        
        # Create 3 separate simulation folders with same parameters but different seeds
        for (folder_num, seed, mu, best_n, converged) in [
            (1, 111, -2.00, 0.501, true),
            (2, 222, -2.02, 0.499, true),
            (3, 333, -1.98, 0.502, true)
        ]
            sim_folder = joinpath(TEST_DIR, "$folder_num")
            mkpath(sim_folder)
            
            seed_data = create_mock_sim_data(
                seed=seed, mu=mu, target_n=target_n, best_n=best_n, 
                converged=converged, beta=beta
            )
            seed_data["dir_path"] = sim_folder
            
            sim_data = Dict{String, Any}(
                "Lx" => 10,
                "Ly" => 10,
                "beta" => beta,
                "V" => 0.0,
                "U" => 4.0,
                "seeds" => [seed],
                "simulations" => Dict{Any, Any}(seed => seed_data)
            )
            
            JLD2.save(joinpath(sim_folder, "sim.jld2"), sim_data)
        end
        
        # Load, merge, and process
        sim_info = get_sim_info(TEST_DIR)
        println("Before merge: $(length(keys(sim_info)) - 1) simulations")  # -1 for updated_at
        
        sim_info = merge_sims(sim_info)
        println("After merge: $(length(keys(sim_info)) - 1) simulations")
        
        sim_info = get_sim_info_means(sim_info)
        
        # After merging, there should be only 1 simulation (parameters are same)
        sim_keys = filter(k -> k != "updated_at", collect(keys(sim_info)))
        @test length(sim_keys) == 1
        
        merged_sim = sim_info[first(sim_keys)]
        @test haskey(merged_sim, "means")
        means = merged_sim["means"]
        
        @test haskey(means, "mu_convergence")
        mu_conv = means["mu_convergence"]
        
        println("mu_convergence contents after merge:")
        for (k, v) in mu_conv
            println("  $k: $v")
        end
        
        # Check mu tracking - should have 3 different mu values
        @test haskey(mu_conv, "mu_list")
        @test length(mu_conv["mu_list"]) == 3
        @test haskey(mu_conv, "mu_mean")
        @test isapprox(mu_conv["mu_mean"], mean([-2.00, -2.02, -1.98]), atol=1e-6)
        
        # Check n convergence tracking
        @test haskey(mu_conv, "target_n")
        @test mu_conv["target_n"] == target_n
        @test haskey(mu_conv, "best_n_list")
        @test length(mu_conv["best_n_list"]) == 3
        @test isapprox(mu_conv["best_n_mean"], mean([0.501, 0.499, 0.502]), atol=1e-6)
        
        # All should have converged
        @test mu_conv["num_converged"] == 3
        @test mu_conv["all_converged"] == true
        
        println("✓ Test passed: Seeds in different folders (merge_sims)")
        return true
    catch e
        println("✗ Test failed: $e")
        rethrow(e)
    finally
        cleanup_test_dir()
    end
end

# =============================================================================
# Test 3: Mixed scenario - some with convergence info, some without
# =============================================================================
function test_mixed_convergence_info()
    println("\n" * "="^60)
    println("TEST 3: Mixed scenario - some seeds with/without convergence info")
    println("="^60)
    
    setup_test_dir()
    
    try
        sim_folder = joinpath(TEST_DIR, "1")
        mkpath(sim_folder)
        
        seeds_data = Dict{Any, Any}()
        
        # Seed 1: Has convergence info
        seeds_data[111] = create_mock_sim_data(
            seed=111, mu=-1.50, target_n=0.875, best_n=0.874, converged=true
        )
        seeds_data[111]["dir_path"] = joinpath(sim_folder, "seed_111")
        
        # Seed 2: No convergence info (old simulation or fixed mu)
        seeds_data[222] = Dict{String, Any}(
            "seed" => 222,
            "mu" => -1.50,  # Same mu (maybe fixed)
            "num_current_measurments" => 100,
            "means" => Dict{String, Any}(
                "n" => Dict("value" => 0.876, "error" => 0.01)
            ),
            "dir_path" => joinpath(sim_folder, "seed_222"),
            "status" => "finished"
        )
        # Note: No "convergence" key
        
        sim_data = Dict{String, Any}(
            "Lx" => 10,
            "Ly" => 10,
            "beta" => 10.0,
            "seeds" => [111, 222],
            "simulations" => seeds_data
        )
        
        JLD2.save(joinpath(sim_folder, "sim.jld2"), sim_data)
        
        sim_info = get_sim_info(TEST_DIR)
        sim_info = get_sim_info_means(sim_info)
        
        means = sim_info["1"]["means"]
        @test haskey(means, "mu_convergence")
        mu_conv = means["mu_convergence"]
        
        println("mu_convergence with mixed data:")
        for (k, v) in mu_conv
            println("  $k: $v")
        end
        
        # Should have 2 mu values
        @test length(mu_conv["mu_list"]) == 2
        
        # But only 1 convergence info entry
        @test haskey(mu_conv, "best_n_list")
        @test length(mu_conv["best_n_list"]) == 1  # Only seed 111 has convergence
        @test mu_conv["num_seeds"] == 1  # Only 1 has convergence info
        
        println("✓ Test passed: Mixed convergence info scenario")
        return true
    catch e
        println("✗ Test failed: $e")
        rethrow(e)
    finally
        cleanup_test_dir()
    end
end

# =============================================================================
# Run all tests
# =============================================================================
function run_all_tests()
    println("\n" * "="^60)
    println("RUNNING MU_CONVERGENCE INFO TESTS")
    println("="^60)
    
    results = Dict{String, Bool}()
    
    try
        results["seeds_same_folder"] = test_seeds_same_folder()
    catch e
        results["seeds_same_folder"] = false
        println("Test failed with error: $e")
    end
    
    try
        results["seeds_different_folders_merge"] = test_seeds_different_folders_merge()
    catch e
        results["seeds_different_folders_merge"] = false
        println("Test failed with error: $e")
    end
    
    try
        results["mixed_convergence_info"] = test_mixed_convergence_info()
    catch e
        results["mixed_convergence_info"] = false
        println("Test failed with error: $e")
    end
    
    # Summary
    println("\n" * "="^60)
    println("TEST SUMMARY")
    println("="^60)
    
    passed = 0
    failed = 0
    for (name, result) in results
        status = result ? "✓ PASSED" : "✗ FAILED"
        println("  $name: $status")
        result ? (passed += 1) : (failed += 1)
    end
    
    println("-"^60)
    println("Total: $passed passed, $failed failed out of $(length(results))")
    println("="^60)
    
    return all(values(results))
end

# Run tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
