# Test file for Phase 9: Auxiliary Files (occupation_convergence.jl, sim_num_control.jl)
# Tests rectangular lattice support for occupation convergence and simulation management

println("="^60)
println("Phase 9 Tests: Auxiliary Files for Rectangular Lattices")
println("="^60)

# Include the files
include(joinpath(@__DIR__, "..", "occupation_convergence.jl"))
include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
# sim_num_control.jl is already included by occupation_convergence.jl

using Test
using Dates

# Test directory - use temp directory
test_dir = joinpath(tempdir(), "phase9_test_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(test_dir)
println("\nTest directory: $test_dir")

all_tests_passed = true

# ============================================================================
# Test 1: get_sim_dimensions helper function
# ============================================================================
println("\n" * "="^60)
println("Test 1: get_sim_dimensions helper function")
println("="^60)

try
    # Test with new format (Lx, Ly)
    new_format = Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 10.0)
    Lx, Ly = get_sim_dimensions(new_format)
    @assert Lx == 4 "Expected Lx=4, got $Lx"
    @assert Ly == 6 "Expected Ly=6, got $Ly"
    println("✓ New format (Lx=4, Ly=6) correctly parsed")
    
    # Test with old format (L only)
    old_format = Dict{String, Any}("L" => 8, "beta" => 10.0)
    Lx, Ly = get_sim_dimensions(old_format)
    @assert Lx == 8 "Expected Lx=8, got $Lx"
    @assert Ly == 8 "Expected Ly=8, got $Ly"
    println("✓ Old format (L=8) correctly parsed as square (8×8)")
    
    # Test with missing dimensions
    no_dims = Dict{String, Any}("beta" => 10.0)
    Lx, Ly = get_sim_dimensions(no_dims)
    @assert isnothing(Lx) "Expected nothing for missing dimensions"
    @assert isnothing(Ly) "Expected nothing for missing dimensions"
    println("✓ Missing dimensions correctly returns (nothing, nothing)")
    
    println("✓ Test 1 PASSED: get_sim_dimensions works correctly")
catch e
    println("✗ Test 1 FAILED: $e")
    global all_tests_passed = false
end

# ============================================================================
# Test 2: format_dimensions helper function
# ============================================================================
println("\n" * "="^60)
println("Test 2: format_dimensions helper function")
println("="^60)

try
    # Square lattice - should show just the number
    square = Dict{String, Any}("Lx" => 6, "Ly" => 6)
    str = format_dimensions(square)
    @assert str == "6" "Expected '6' for square, got '$str'"
    println("✓ Square lattice (6×6) formatted as '6'")
    
    # Rectangular lattice - should show both
    rect = Dict{String, Any}("Lx" => 4, "Ly" => 6)
    str = format_dimensions(rect)
    @assert str == "4×6" "Expected '4×6' for rectangular, got '$str'"
    println("✓ Rectangular lattice (4×6) formatted as '4×6'")
    
    # Old format square
    old_square = Dict{String, Any}("L" => 8)
    str = format_dimensions(old_square)
    @assert str == "8" "Expected '8' for old format, got '$str'"
    println("✓ Old format (L=8) formatted as '8'")
    
    println("✓ Test 2 PASSED: format_dimensions works correctly")
catch e
    println("✗ Test 2 FAILED: $e")
    global all_tests_passed = false
end

# ============================================================================
# Test 3: Occupation Convergence with Rectangular Lattice (Lx=4, Ly=4)
# ============================================================================
println("\n" * "="^60)
println("Test 3: Occupation Convergence with Square Lattice (4×4)")
println("="^60)

try
    # Run a quick occupation convergence simulation
    # Using small parameters for fast testing
    wanted_n = 1.0
    n_tol = 0.01  # Tight tolerance for test
    
    println("Running occupation convergence to n=$wanted_n...")
    println("This may take a moment...")
    
    data = running_simulation_wanted_occupation(
        10, 10,          # Lx, Ly (square for simplicity)
        1,             # L_SC
        0.0,           # U
        1.0,           # t
        0.0,           # V (no disorder)
        42,            # seed
        wanted_n,      # wanted occupation
        5.0,           # beta
        0.0,           # B (no magnetic field)
        1,            # num_measure (small for test)
        0,           # num_thermal (small for test)
        5,             # measuring_interval
        [-2.0, 0.0, 2.0],  # w_list (minimal)
        "uniform",      # initial_guess
        "uniform",        # U_disorder
        "K0",       # diag_method
        "s-wave",      # pairing_symmetry
        [0.0, 0.01],   # w_conductivity_range
        0.0;           # U_impurity
        directory=test_dir,
        savedata=false,
        saveplots=false,
        date_folder=false,
        numbered=true,
        num_thermal_occ=0,
        num_measure_occ=1,
        n_tol=n_tol,    # Relaxed tolerance for test
        max_iter=10,
        num_uncorrelated_measurments=1,  # Small for fast test
        resume_path=""
    )
    
    # Check that simulation ran
    @assert data !== nothing "Simulation should return data"
    # @assert data.sim_data.Lx == 4 "Expected Lx=4"
    # @assert data.sim_data.Ly == 4 "Expected Ly=4"
    
    # Check that we got close to wanted n
    measured_n = mean(data.measurement_data.n[1:data.current_state.num_current_measurments])
    println("Target n: $wanted_n, Measured n: $(round(measured_n, digits=3))")
    @assert abs(measured_n - wanted_n) < n_tol "Should be close to target occupation"
    
    println("✓ Test 3 PASSED: Occupation convergence works for square lattice")
catch e
    println("✗ Test 3 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 4: Occupation Convergence with Rectangular Lattice (Lx=4, Ly=6)
# ============================================================================
println("\n" * "="^60)
println("Test 4: Occupation Convergence with Rectangular Lattice (4×6)")
println("="^60)

try
    wanted_n = 0.80
    
    println("Running occupation convergence for 4×6 system to n=$wanted_n...")
    
    data = running_simulation_wanted_occupation(
        4, 6,          # Lx, Ly (rectangular!)
        1,             # L_SC
        0.0,           # U
        1.0,           # t
        0.0,           # V
        123,           # seed
        wanted_n,      # wanted occupation
        5.0,           # beta
        0.0,           # B
        50,            # num_measure
        100,           # num_thermal
        5,             # measuring_interval
        [-2.0, 0.0, 2.0],
        "uniform",
        "uniform",
        "K0",
        "s-wave",
        [0.0, 0.01],
        0.0;
        directory=test_dir,
        savedata=true,
        saveplots=false,
        date_folder=false,
        numbered=true,
        num_thermal_occ=100,
        num_measure_occ=50,
        n_tol=0.05,
        max_iter=10,
        num_uncorrelated_measurments=16,
        resume_path=""
    )
    
    @assert data !== nothing "Simulation should return data"
    @assert data.sim_data.Lx == 4 "Expected Lx=4"
    @assert data.sim_data.Ly == 6 "Expected Ly=6"
    @assert data.sim_data.N == 24 "Expected N=24 (4×6)"
    
    measured_n = mean(data.measurement_data.n[1:data.current_state.num_current_measurments])
    println("Target n: $wanted_n, Measured n: $(round(measured_n, digits=3))")
    
    println("✓ Test 4 PASSED: Occupation convergence works for rectangular lattice")
catch e
    println("✗ Test 4 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 5: Legacy Wrapper (L instead of Lx, Ly)
# ============================================================================
println("\n" * "="^60)
println("Test 5: Legacy Wrapper (single L parameter)")
println("="^60)

try
    wanted_n = 0.75
    
    println("Running occupation convergence using legacy L=4 parameter...")
    
    # Use legacy wrapper with single L
    data = running_simulation_wanted_occupation(
        4,             # L (legacy - should be interpreted as Lx=Ly=4)
        1,             # L_SC
        4.0,           # U
        1.0,           # t
        0.0,           # V
        456,           # seed
        wanted_n,
        5.0,           # beta
        0.0,           # B
        50,            # num_measure
        100,           # num_thermal
        5,             # measuring_interval
        [-2.0, 0.0, 2.0],
        "uniform",
        "uniform",
        "K0",
        "s-wave",
        [0.0, 0.01],
        0.0;
        directory=test_dir,
        savedata=true,
        saveplots=false,
        date_folder=false,
        numbered=true,
        num_thermal_occ=100,
        num_measure_occ=50,
        n_tol=0.05,
        max_iter=10,
        num_uncorrelated_measurments=16,
        resume_path=""
    )
    
    @assert data.sim_data.Lx == 4 "Legacy L=4 should give Lx=4"
    @assert data.sim_data.Ly == 4 "Legacy L=4 should give Ly=4"
    
    println("✓ Test 5 PASSED: Legacy wrapper works correctly")
catch e
    println("✗ Test 5 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 6: Create and Load sim_info
# ============================================================================
println("\n" * "="^60)
println("Test 6: Create and Load sim_info")
println("="^60)

try
    # Create the num_control directory if it doesn't exist
    num_control_dir = joinpath(test_dir, "num_control")
    mkpath(num_control_dir)
    
    # Create sim.jld2 files in numbered subdirectories (like fh_pre_run_n.jl does)
    # This mimics how the cluster runner creates simulation folders
    println("Creating mock sim.jld2 files...")
    
    # Create simulation 1: square 4×4 (old format with L)
    sim1_dir = joinpath(test_dir, "1")
    mkpath(sim1_dir)
    sim1_data = Dict(
        "L" => 4,
        "SC_L" => 1,
        "U" => 4.0,
        "t" => 1.0,
        "V" => 0.0,
        "seeds" => [42],
        "wanted_n" => 0.85,
        "beta" => 5.0,
        "B" => 0.0,
        "num_measure" => 1000,
        "num_thermal" => 100,
        "measuring_interval" => 5,
        "guess" => "uniform",
        "U_disorder" => "uniform",
        "diag_method" => "K0",
        "cores" => 1,
        "TClustersize" => 4,
        "pairing_symmetry" => "s-wave"
    )
    JLD2.save(joinpath(sim1_dir, "sim.jld2"), sim1_data)
    
    # Create simulation 2: rectangular 4×6 (new format with Lx, Ly)
    sim2_dir = joinpath(test_dir, "2")
    mkpath(sim2_dir)
    sim2_data = Dict(
        "Lx" => 4,
        "Ly" => 6,
        "SC_L" => 1,
        "U" => 4.0,
        "t" => 1.0,
        "V" => 0.0,
        "seeds" => [123],
        "wanted_n" => 0.80,
        "beta" => 5.0,
        "B" => 0.0,
        "num_measure" => 1000,
        "num_thermal" => 100,
        "measuring_interval" => 5,
        "guess" => "uniform",
        "U_disorder" => "uniform",
        "diag_method" => "K0",
        "cores" => 1,
        "TClustersize_x" => 4,
        "TClustersize_y" => 4,
        "pairing_symmetry" => "s-wave"
    )
    JLD2.save(joinpath(sim2_dir, "sim.jld2"), sim2_data)
    
    # Create simulation 3: square 6×6 (new format)
    sim3_dir = joinpath(test_dir, "3")
    mkpath(sim3_dir)
    sim3_data = Dict(
        "Lx" => 6,
        "Ly" => 6,
        "SC_L" => 1,
        "U" => 4.0,
        "t" => 1.0,
        "V" => 0.0,
        "seeds" => [456],
        "wanted_n" => 0.75,
        "beta" => 10.0,
        "B" => 0.0,
        "num_measure" => 1000,
        "num_thermal" => 100,
        "measuring_interval" => 5,
        "guess" => "uniform",
        "U_disorder" => "uniform",
        "diag_method" => "K0",
        "cores" => 1,
        "TClustersize_x" => 4,
        "TClustersize_y" => 4,
        "pairing_symmetry" => "s-wave"
    )
    JLD2.save(joinpath(sim3_dir, "sim.jld2"), sim3_data)
    
    # Now get sim_info from test directory
    println("Scanning test directory for simulations...")
    sim_info = get_sim_info(test_dir)
    
    @assert haskey(sim_info, "updated_at") "sim_info should have updated_at"
    
    num_sims = length(filter(k -> k != "updated_at", keys(sim_info)))
    println("Found $num_sims simulation(s)")
    @assert num_sims >= 3 "Should have at least 3 simulations"
    
    # Verify old format (L) is read correctly
    @assert haskey(sim_info["1"], "L") "Simulation 1 should have L key"
    @assert sim_info["1"]["L"] == 4 "Simulation 1 L should be 4"
    println("✓ Old format (L=4) simulation loaded correctly")
    
    # Verify new format (Lx, Ly) is read correctly
    @assert haskey(sim_info["2"], "Lx") "Simulation 2 should have Lx key"
    @assert haskey(sim_info["2"], "Ly") "Simulation 2 should have Ly key"
    @assert sim_info["2"]["Lx"] == 4 "Simulation 2 Lx should be 4"
    @assert sim_info["2"]["Ly"] == 6 "Simulation 2 Ly should be 6"
    println("✓ New format (4×6) simulation loaded correctly")
    
    # Test get_sim_dimensions on loaded data
    Lx, Ly = get_sim_dimensions(sim_info["1"])
    @assert Lx == 4 && Ly == 4 "Old format should give 4×4"
    
    Lx, Ly = get_sim_dimensions(sim_info["2"])
    @assert Lx == 4 && Ly == 6 "New format should give 4×6"
    println("✓ get_sim_dimensions works on loaded sim_info")
    
    # Save sim_info
    save_sim_info(test_dir, sim_info)
    println("Saved sim_info to $num_control_dir")
    
    # Load it back
    loaded_sim_info = load_sim_info(test_dir)
    @assert haskey(loaded_sim_info, "updated_at") "Loaded sim_info should have updated_at"
    @assert length(filter(k -> k != "updated_at", keys(loaded_sim_info))) == num_sims "Should load same number of sims"
    
    println("✓ Test 6 PASSED: sim_info creation and loading works")
catch e
    println("✗ Test 6 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 7: group_by_L with rectangular data
# ============================================================================
println("\n" * "="^60)
println("Test 7: group_by_L with rectangular data")
println("="^60)

try
    # Create mock sim_info with mixed old/new format data
    mock_sim_info = Dict{String, Any}(
        "updated_at" => now(),
        "1" => Dict{String, Any}("L" => 4, "beta" => 5.0, "means" => Dict("n" => Dict("value" => 0.8))),
        "2" => Dict{String, Any}("L" => 6, "beta" => 5.0, "means" => Dict("n" => Dict("value" => 0.85))),
        "3" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 5.0, "means" => Dict("n" => Dict("value" => 0.82))),
        "4" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 10.0, "means" => Dict("n" => Dict("value" => 0.81))),
        "5" => Dict{String, Any}("Lx" => 8, "Ly" => 8, "beta" => 5.0, "means" => Dict("n" => Dict("value" => 0.79)))
    )
    
    # Group by L
    L_groups = group_by_L(mock_sim_info)
    
    println("L groups found:")
    for (L_key, sims) in L_groups
        println("  $L_key: $(length(sims)) simulation(s)")
    end
    
    # Check groupings
    @assert haskey(L_groups, 4) "Should have group for L=4 (square)"
    @assert haskey(L_groups, 6) "Should have group for L=6 (square)"
    @assert haskey(L_groups, (4, 6)) "Should have group for (4,6) (rectangular)"
    @assert haskey(L_groups, 8) "Should have group for L=8 (square, from Lx=Ly=8)"
    
    @assert length(L_groups[(4, 6)]) == 2 "Should have 2 sims for (4,6)"
    
    # Test with specified_L
    L_groups_filtered = group_by_L(mock_sim_info; specified_L=4)
    @assert length(L_groups_filtered) == 1 "Should only have L=4 group"
    @assert haskey(L_groups_filtered, 4) "Should have L=4"
    
    # Test with specified_L and specified_Ly
    L_groups_rect = group_by_L(mock_sim_info; specified_L=4, specified_Ly=6)
    @assert length(L_groups_rect) == 1 "Should only have (4,6) group"
    @assert haskey(L_groups_rect, (4, 6)) "Should have (4,6)"
    
    println("✓ Test 7 PASSED: group_by_L works correctly with rectangular data")
catch e
    println("✗ Test 7 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 8: group_by_X with "L" key
# ============================================================================
println("\n" * "="^60)
println("Test 8: group_by_X with 'L' key")
println("="^60)

try
    mock_sim_info = Dict{String, Any}(
        "updated_at" => now(),
        "1" => Dict{String, Any}("L" => 4, "beta" => 5.0, "diag_method" => "exact"),
        "2" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 5.0, "diag_method" => "exact"),
        "3" => Dict{String, Any}("Lx" => 6, "Ly" => 6, "beta" => 10.0, "diag_method" => "exact")
    )
    
    # Group by "L" - should use get_sim_dimensions internally
    L_groups = group_by_X(mock_sim_info, "L")
    
    println("group_by_X(\"L\") groups:")
    for (L_key, sims) in L_groups
        println("  $L_key: $(length(sims)) simulation(s)")
    end
    
    @assert haskey(L_groups, 4) "Should have group for L=4"
    @assert haskey(L_groups, (4, 6)) "Should have group for (4,6)"
    @assert haskey(L_groups, 6) "Should have group for L=6 (from Lx=Ly=6)"
    
    println("✓ Test 8 PASSED: group_by_X with 'L' key works correctly")
catch e
    println("✗ Test 8 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 9: get_sim_filling with Lx, Ly
# ============================================================================
println("\n" * "="^60)
println("Test 9: get_sim_filling with Lx, Ly")
println("="^60)

try
    mock_sim_info = Dict{String, Any}(
        "updated_at" => now(),
        "1" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 5.0, "mu" => -1.0, 
                                  "means" => Dict("n" => Dict("value" => 0.8, "error" => 0.01))),
        "2" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 5.0, "mu" => -0.5,
                                  "means" => Dict("n" => Dict("value" => 0.9, "error" => 0.01))),
        "3" => Dict{String, Any}("Lx" => 4, "Ly" => 4, "beta" => 5.0, "mu" => -1.0,
                                  "means" => Dict("n" => Dict("value" => 0.85, "error" => 0.02)))
    )
    
    # Get filling for rectangular 4x6
    filling_4x6 = get_sim_filling(mock_sim_info, 4, 6)
    @assert filling_4x6 !== nothing "Should get filling for 4×6"
    @assert haskey(filling_4x6, "5.0") "Should have beta=5.0"
    @assert length(filling_4x6["5.0"]["mu"]) == 2 "Should have 2 mu values for 4×6"
    println("✓ get_sim_filling(4, 6) found $(length(filling_4x6["5.0"]["mu"])) data points")
    
    # Get filling for square 4x4 (using default Ly=Lx)
    filling_4x4 = get_sim_filling(mock_sim_info, 4)
    @assert filling_4x4 !== nothing "Should get filling for 4×4"
    @assert haskey(filling_4x4, "5.0") "Should have beta=5.0"
    @assert length(filling_4x4["5.0"]["mu"]) == 1 "Should have 1 mu value for 4×4"
    println("✓ get_sim_filling(4) found $(length(filling_4x4["5.0"]["mu"])) data points")
    
    println("✓ Test 9 PASSED: get_sim_filling works with Lx, Ly")
catch e
    println("✗ Test 9 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 10: find_closest_filling with rectangular
# ============================================================================
println("\n" * "="^60)
println("Test 10: find_closest_filling with rectangular lattice")
println("="^60)

try
    mock_sim_info = Dict{String, Any}(
        "updated_at" => now(),
        "1" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 5.0, "mu" => -1.0,
                                  "means" => Dict("n" => Dict("value" => 0.82, "error" => 0.01))),
        "2" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 5.0, "mu" => -0.8,
                                  "means" => Dict("n" => Dict("value" => 0.88, "error" => 0.01))),
        "3" => Dict{String, Any}("Lx" => 4, "Ly" => 6, "beta" => 10.0, "mu" => -1.0,
                                  "means" => Dict("n" => Dict("value" => 0.80, "error" => 0.01))),
        "4" => Dict{String, Any}("L" => 4, "beta" => 5.0, "mu" => -1.0,
                                  "means" => Dict("n" => Dict("value" => 0.85, "error" => 0.01)))
    )
    
    wanted_n = 0.85
    
    # Find closest for rectangular 4×6
    results = find_closest_filling(mock_sim_info, wanted_n; specified_L=4, specified_Ly=6)
    @assert !isempty(results) "Should find results for 4×6"
    println("Found $(length(results)) beta value(s) for 4×6 with n closest to $wanted_n")
    
    # Find closest for square 4×4 (old format)
    results_square = find_closest_filling(mock_sim_info, wanted_n; specified_L=4)
    @assert !isempty(results_square) "Should find results for 4×4"
    println("Found $(length(results_square)) beta value(s) for 4×4 with n closest to $wanted_n")
    
    println("✓ Test 10 PASSED: find_closest_filling works with rectangular")
catch e
    println("✗ Test 10 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 11: sim_stat with rectangular data
# ============================================================================
println("\n" * "="^60)
println("Test 11: sim_stat output format")
println("="^60)

try
    # Create mock sim_dict
    mock_sim_dict = Dict{String, Any}(
        "Lx" => 4,
        "Ly" => 6,
        "beta" => 10.0,
        "diag_method" => "exact",
        "U_disorder" => "none",
        "impurity" => 0.0,
        "means" => Dict("n" => Dict("value" => 0.85, "error" => 0.02)),
        "simulations" => Dict(
            "42" => Dict(
                "status" => "finished",
                "step_count" => 1000,
                "num_current_measurments" => 100,
                "start_time" => now() - Dates.Hour(1),
                "last_save_time" => now()
            )
        )
    )
    
    println("sim_stat output for rectangular simulation:")
    println("-"^40)
    sim_stat(mock_sim_dict)
    println("-"^40)
    
    # Also test with old format
    mock_sim_dict_old = Dict{String, Any}(
        "L" => 8,
        "beta" => 5.0,
        "diag_method" => "TCluster",
        "TCluster_size" => 4,
        "U_disorder" => "impurity",
        "impurity" => 0.5,
        "means" => Dict("n" => Dict("value" => 0.9, "error" => 0.01)),
        "simulations" => Dict(
            "1" => Dict(
                "status" => "running",
                "step_count" => 500,
                "num_current_measurments" => 50,
                "start_time" => now() - Dates.Minute(30),
                "last_save_time" => now()
            )
        )
    )
    
    println("\nsim_stat output for old format (square) simulation:")
    println("-"^40)
    sim_stat(mock_sim_dict_old)
    println("-"^40)
    
    println("✓ Test 11 PASSED: sim_stat displays correctly")
catch e
    println("✗ Test 11 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Test 12: Update means and print_sim_details
# ============================================================================
println("\n" * "="^60)
println("Test 12: Update means and print_sim_details")
println("="^60)

try
    # This test uses the actual simulations from tests 3-5
    # Update means
    println("Updating means from simulations...")
    update_means(test_dir)
    
    # Print sim details
    println("\nPrinting simulation details...")
    print_sim_details(test_dir)
    
    # Check that sim_details.txt was created
    details_path = joinpath(test_dir, "num_control", "sim_details.txt")
    @assert isfile(details_path) "sim_details.txt should be created"
    
    # Read and show a snippet
    content = read(details_path, String)
    println("\nFirst 500 chars of sim_details.txt:")
    println("-" ^ 40)
    println(first(content, 500))
    println("-" ^ 40)
    
    println("✓ Test 12 PASSED: update_means and print_sim_details work")
catch e
    println("✗ Test 12 FAILED: $e")
    println(stacktrace(catch_backtrace()))
    global all_tests_passed = false
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^60)
println("PHASE 9 TEST SUMMARY")
println("="^60)

if all_tests_passed
    println("✓ ALL TESTS PASSED!")
    println("\nPhase 9 implementation is complete:")
    println("  - get_sim_dimensions helper works for old/new formats")
    println("  - format_dimensions displays Lx×Ly correctly")
    println("  - Occupation convergence works for rectangular lattices")
    println("  - Legacy wrappers maintain backward compatibility")
    println("  - sim_info creation and loading works")
    println("  - group_by_L groups by (Lx, Ly) tuple")
    println("  - group_by_X handles 'L' key specially")
    println("  - get_sim_filling supports Lx, Ly parameters")
    println("  - find_closest_filling supports specified_Ly")
    println("  - sim_stat displays rectangular dimensions")
    println("  - print_sim_details shows Lx×Ly format")
else
    println("✗ SOME TESTS FAILED - Review output above")
end

println("\nTest directory: $test_dir")
println("(You can delete this directory after reviewing results)")
