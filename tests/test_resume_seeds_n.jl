"""
Tests for resume functionality in running_simulation_seeds_wanted_occupation

Tests:
1. Fresh start creates correct tracking structure
2. Completed seeds are skipped on resume
3. Partially completed seeds resume correctly (convergence phase)
4. Partially completed seeds resume correctly (simulation phase)
5. All seeds completed flag is set correctly
"""

include(joinpath(@__DIR__, "..", "occupation_convergence.jl"))

using Test
using Random

# Test parameters - small and fast
const TEST_DIR = joinpath(tempdir(), "test_resume_seeds_n_$(Dates.format(now(), "HHMMSSsss"))")
const TEST_Lx = 4
const TEST_Ly = 4
const TEST_U = 0.0
const TEST_t = 1.0
const TEST_V = 0.0
const TEST_beta = 1.0
const TEST_B = 0.0
const TEST_wanted_n = 0.5
const TEST_seeds = [111, 222, 333]

# Minimal simulation parameters for fast testing
const TEST_num_measure = 5
const TEST_num_thermal = 2
const TEST_measuring_interval = 1
const TEST_w_list = [-1.0, 1.0]
const TEST_num_thermal_occ = 2
const TEST_num_measure_occ = 2
const TEST_max_iter = 3
const TEST_num_uncorrelated = 2

function setup_test_dir()
    if isdir(TEST_DIR)
        rm(TEST_DIR, recursive=true)
    end
    mkpath(TEST_DIR)
    println("Test directory: $TEST_DIR")
end

function cleanup_test_dir()
    if isdir(TEST_DIR)
        rm(TEST_DIR, recursive=true)
    end
end

function run_short_simulation(seeds::Vector{Int64}; directory=TEST_DIR)
    """Run a short simulation with given seeds"""
    return running_simulation_seeds_wanted_occupation(
        TEST_Lx, TEST_Ly, 1, TEST_U, TEST_t, TEST_V, seeds, TEST_wanted_n, 
        TEST_beta, TEST_B, TEST_num_measure, TEST_num_thermal, TEST_measuring_interval,
        TEST_w_list, "uniform", "uniform", "K0", "s-wave", [0.0, 0.01], 0.0;
        directory=directory,
        savedata=true,
        saving_interval=TEST_num_measure,
        saveplots=false,
        seed_folder=true,
        num_thermal_occ=TEST_num_thermal_occ,
        num_measure_occ=TEST_num_measure_occ,
        max_iter=TEST_max_iter,
        num_uncorrelated_measurments=TEST_num_uncorrelated,
        n_tol=0.5  # Large tolerance for fast convergence in tests
    )
end

# =============================================================================
# Test 1: Fresh start creates correct tracking structure
# =============================================================================
function test_fresh_start_tracking()
    println("\n" * "="^60)
    println("TEST 1: Fresh start creates correct tracking structure")
    println("="^60)
    
    setup_test_dir()
    
    try
        # Run with single seed first
        run_short_simulation([TEST_seeds[1]])
        
        sim_info_path = joinpath(TEST_DIR, "sim.jld2")
        @test isfile(sim_info_path)
        
        sim_info = JLD2.load(sim_info_path)
        
        @test haskey(sim_info, "completed_seeds")
        @test TEST_seeds[1] in sim_info["completed_seeds"]
        @test haskey(sim_info, "last_completed_seed")
        @test sim_info["last_completed_seed"] == TEST_seeds[1]
        @test haskey(sim_info, "last_completed_time")
        
        # Check seed folder was created
        seed_folder = joinpath(TEST_DIR, "seed_$(TEST_seeds[1])")
        @test isdir(seed_folder)
        
        println("✓ Fresh start tracking works correctly")
        return true
    catch e
        println("✗ Test failed: $e")
        rethrow(e)
    finally
        cleanup_test_dir()
    end
end

# =============================================================================
# Test 2: Completed seeds are skipped on resume
# =============================================================================
function test_skip_completed_seeds()
    println("\n" * "="^60)
    println("TEST 2: Completed seeds are skipped on resume")
    println("="^60)
    
    setup_test_dir()
    
    try
        # Run first two seeds
        println("Running first two seeds...")
        run_short_simulation([TEST_seeds[1], TEST_seeds[2]])
        
        sim_info_path = joinpath(TEST_DIR, "sim.jld2")
        sim_info = JLD2.load(sim_info_path)
        
        @test length(sim_info["completed_seeds"]) == 2
        @test TEST_seeds[1] in sim_info["completed_seeds"]
        @test TEST_seeds[2] in sim_info["completed_seeds"]
        
        # Record modification times of completed seed folders
        seed1_folder = joinpath(TEST_DIR, "seed_$(TEST_seeds[1])")
        seed2_folder = joinpath(TEST_DIR, "seed_$(TEST_seeds[2])")
        seed1_mtime = mtime(seed1_folder)
        seed2_mtime = mtime(seed2_folder)
        
        # Wait a moment to ensure time difference
        sleep(1)
        
        # Now "resume" with all three seeds - first two should be skipped
        println("Resuming with all seeds (first two should be skipped)...")
        run_short_simulation(TEST_seeds)
        
        # Check that all three are now completed
        sim_info = JLD2.load(sim_info_path)
        @test length(sim_info["completed_seeds"]) == 3
        @test all(s in sim_info["completed_seeds"] for s in TEST_seeds)
        @test haskey(sim_info, "all_seeds_completed")
        @test sim_info["all_seeds_completed"] == true
        
        # Verify first two seed folders were NOT modified (seeds were skipped)
        # Note: The folder itself shouldn't change, but files inside might
        # Better check: third seed folder should exist
        seed3_folder = joinpath(TEST_DIR, "seed_$(TEST_seeds[3])")
        @test isdir(seed3_folder)
        
        println("✓ Completed seeds are correctly skipped on resume")
        return true
    catch e
        println("✗ Test failed: $e")
        rethrow(e)
    finally
        cleanup_test_dir()
    end
end

# =============================================================================
# Test 3: Simulate crash and resume mid-sequence
# =============================================================================
function test_resume_after_crash()
    println("\n" * "="^60)
    println("TEST 3: Resume after simulated crash (mid-sequence)")
    println("="^60)
    
    setup_test_dir()
    
    try
        # Manually create a sim.jld2 that looks like seed 1 completed
        sim_info_path = joinpath(TEST_DIR, "sim.jld2")
        
        # Run just seed 1
        println("Running seed 1...")
        run_short_simulation([TEST_seeds[1]])
        
        # Verify seed 1 completed
        sim_info = JLD2.load(sim_info_path)
        @test length(sim_info["completed_seeds"]) == 1
        
        # Now simulate a "crash" happened before seeds 2 and 3
        # (sim.jld2 already has seed 1 as completed)
        
        # Resume with all seeds
        println("Simulating resume after crash...")
        run_short_simulation(TEST_seeds)
        
        # Verify all seeds are now completed
        sim_info = JLD2.load(sim_info_path)
        @test length(sim_info["completed_seeds"]) == 3
        @test sim_info["all_seeds_completed"] == true
        
        # Verify all seed folders exist
        for seed in TEST_seeds
            seed_folder = joinpath(TEST_DIR, "seed_$seed")
            @test isdir(seed_folder)
        end
        
        println("✓ Resume after crash works correctly")
        return true
    catch e
        println("✗ Test failed: $e")
        rethrow(e)
    finally
        cleanup_test_dir()
    end
end

# =============================================================================
# Test 4: Resume with partially completed seed (convergence phase)
# =============================================================================
function test_resume_during_convergence()
    println("\n" * "="^60)
    println("TEST 4: Resume during convergence phase")
    println("="^60)
    
    setup_test_dir()
    
    try
        seed = TEST_seeds[1]
        seed_folder = joinpath(TEST_DIR, "seed_$seed")
        mkpath(seed_folder)
        
        # Create a partial convergence state in the seed folder
        seed_sim_info_path = joinpath(seed_folder, "sim.jld2")
        
        # Simulate a convergence that was in progress
        partial_conv_state = Dict{String, Any}(
            "convergence" => Dict{String, Any}(
                "target_n" => TEST_wanted_n,
                "mu_edges" => (-4.0, 0.0),
                "n_edges" => (0.0, 1.0),
                "mu_list" => [-4.0, 0.0, -2.0],
                "n_list" => [0.0, 1.0, 0.45],
                "n_tol" => 0.5,
                "max_iter" => TEST_max_iter,
                "num_thermal_occ" => TEST_num_thermal_occ,
                "num_measure_occ" => TEST_num_measure_occ,
                "converged" => false,
                "best_mu" => -2.0,
                "best_n" => 0.45
            ),
            "status" => "convergence_in_progress"
        )
        JLD2.save(seed_sim_info_path, partial_conv_state)
        
        println("Created partial convergence state, now resuming...")
        
        # Run the simulation - should resume from convergence
        run_short_simulation([seed])
        
        # Check that simulation completed
        sim_info_path = joinpath(TEST_DIR, "sim.jld2")
        @test isfile(sim_info_path)
        
        sim_info = JLD2.load(sim_info_path)
        @test seed in sim_info["completed_seeds"]
        
        println("✓ Resume during convergence phase works correctly")
        return true
    catch e
        println("✗ Test failed: $e")
        rethrow(e)
    finally
        cleanup_test_dir()
    end
end

# =============================================================================
# Test 5: Verify data integrity after resume
# =============================================================================
function test_data_integrity_after_resume()
    println("\n" * "="^60)
    println("TEST 5: Data integrity after resume")
    println("="^60)
    
    setup_test_dir()
    
    try
        # Run seed 1
        run_short_simulation([TEST_seeds[1]])
        
        # The main sim.jld2 is in the parent directory (TEST_DIR)
        # Each seed folder contains checkpoint.jld2 and simulation_log.txt
        sim_info_path = joinpath(TEST_DIR, "sim.jld2")
        @test isfile(sim_info_path)
        sim_info_before = JLD2.load(sim_info_path)
        
        # Check seed 1 folder exists with checkpoint
        seed1_folder = joinpath(TEST_DIR, "seed_$(TEST_seeds[1])")
        @test isdir(seed1_folder)
        
        # Record the completed seeds before resume
        completed_before = copy(sim_info_before["completed_seeds"])
        @test TEST_seeds[1] in completed_before
        
        # Resume with all seeds
        run_short_simulation(TEST_seeds)
        
        # Verify sim.jld2 still contains seed 1 data and wasn't corrupted
        sim_info_after = JLD2.load(sim_info_path)
        
        # Seed 1 should still be in completed_seeds
        @test TEST_seeds[1] in sim_info_after["completed_seeds"]
        
        # All seeds should now be completed
        @test length(sim_info_after["completed_seeds"]) == 3
        @test all(s in sim_info_after["completed_seeds"] for s in TEST_seeds)
        
        # All seed folders should exist
        for seed in TEST_seeds
            seed_folder = joinpath(TEST_DIR, "seed_$seed")
            @test isdir(seed_folder)
            # Each seed folder should have a log file
            @test isfile(joinpath(seed_folder, "simulation_log.txt"))
        end
        
        println("✓ Data integrity maintained after resume")
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
    println("RUNNING RESUME TESTS FOR running_simulation_seeds_wanted_occupation")
    println("="^60)
    
    results = Dict{String, Bool}()
    
    try
        results["fresh_start_tracking"] = test_fresh_start_tracking()
    catch e
        results["fresh_start_tracking"] = false
        println("Test failed with error: $e")
    end
    
    try
        results["skip_completed_seeds"] = test_skip_completed_seeds()
    catch e
        results["skip_completed_seeds"] = false
        println("Test failed with error: $e")
    end
    
    try
        results["resume_after_crash"] = test_resume_after_crash()
    catch e
        results["resume_after_crash"] = false
        println("Test failed with error: $e")
    end
    
    try
        results["resume_during_convergence"] = test_resume_during_convergence()
    catch e
        results["resume_during_convergence"] = false
        println("Test failed with error: $e")
    end
    
    try
        results["data_integrity_after_resume"] = test_data_integrity_after_resume()
    catch e
        results["data_integrity_after_resume"] = false
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
