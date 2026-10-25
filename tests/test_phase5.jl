# Test file for Phase 5: Spectral, Current Functions, I/O & Plotting
# Tests rectangular lattice support for plotting, file naming, and logging functions

using Test
using Dates

println("Loading MC_Hubburd.jl...")
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
println("Module loaded successfully!")

# ============================================================================
# Helper: Create a SimData with positional arguments
# ============================================================================
function create_test_sim_data(; Lx=4, Ly=4, SC_L=1, pairing_symmetry="s-wave", 
                               diag_method="K0", TCluster_size_x=2, TCluster_size_y=2)
    N = Lx * Ly
    SimData(
        Lx,                          # Lx
        Ly,                          # Ly
        N,                           # N
        SC_L,                        # SC_L (supercell size)
        4.0,                         # U
        1.0,                         # t
        0.0,                         # V (disorder)
        12345,                       # seed
        0.0,                         # mu
        5.0,                         # beta
        5,                           # num_measure
        2,                           # num_thermal
        1,                           # measuring_interval
        5,                           # saving_interval
        collect(-3.0:0.5:3.0),       # w_list
        "uniform",                   # guess
        "uniform",                   # U_disorder
        diag_method,                 # diag_method
        pairing_symmetry,            # pairing_symmetry
        "test_dir",                  # directory
        1,                           # num_cores
        1,                           # num_threads
        Dates.now(),                 # start_time
        collect(-2.0:0.5:2.0),       # w_conductivity_range
        [0.0, 10.0, 0.7, 0.7],       # U_disorder_params
        1.0,                         # U_impurity
        TCluster_size_x,             # TCluster_size_x
        TCluster_size_y,             # TCluster_size_y
        (false, 0.8),                # smooth_U
        true,                        # loop_random
        0.0,                         # B
        "LandauX",                   # gauge_choice
        "PBC"                        # boundary_conditions
    )
end

@testset "Phase 5: Spectral, Current, I/O & Plotting" begin
    
    @testset "Square lattice baseline (4x4)" begin
        println("\n--- Test 1: Square lattice 4x4 ---")
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Run a few MC steps
        for _ in 1:3
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Test reduced_spectral_function returns correct size
        reduced_A = reduced_spectral_function(lattice_data, 0.0)
        println("  reduced_A length: $(length(reduced_A))")
        @test length(reduced_A) > 0
        
        # Test file name generation
        filename = choose_file_name(".", sim_data; date_folder=false, numbered=false)
        println("  Generated filename: $filename")
        @test occursin("4x4", filename)
        @test occursin("beta=5.0", filename)
        
        println("  ✓ Square lattice tests passed")
    end
    
    @testset "Rectangular lattice (6x4)" begin
        println("\n--- Test 2: Rectangular lattice 6x4 ---")
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Run a few MC steps
        for _ in 1:3
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Test reduced_spectral_function returns correct size
        reduced_A = reduced_spectral_function(lattice_data, 0.0)
        println("  reduced_A length: $(length(reduced_A))")
        @test length(reduced_A) > 0
        
        # Test file name generation with rectangular dimensions
        filename = choose_file_name(".", sim_data; date_folder=false, numbered=false)
        println("  Generated filename: $filename")
        @test occursin("6x4", filename)
        
        # Test BZ reshape works correctly (no longer square)
        BZ = lattice_data.measurement_data.BZ
        @test length(BZ) == 6 * 4  # Lx * Ly
        BZ_reshaped = reshape(BZ, 6, 4)
        @test size(BZ_reshaped) == (6, 4)
        
        println("  ✓ Rectangular lattice tests passed")
    end
    
    @testset "Log simulation start" begin
        println("\n--- Test 3: Log simulation start ---")
        sim_data = create_test_sim_data(Lx=6, Ly=4, diag_method="K0-TCA", 
                                        pairing_symmetry="d-wave",
                                        TCluster_size_x=3, TCluster_size_y=2)
        
        # Create a temporary log file
        test_log = "test_log_phase5.txt"
        start_time = now()
        
        # Test logging function
        log_simulation_start(test_log, sim_data, start_time)
        
        # Read and verify log content
        log_content = read(test_log, String)
        println("  Log content snippet: $(first(log_content, 200))...")
        
        @test occursin("Lx=6", log_content)
        @test occursin("Ly=4", log_content)
        @test occursin("TClustersize=3x2", log_content)
        
        # Clean up
        rm(test_log)
        
        println("  ✓ Log simulation start tests passed")
    end
    
    @testset "Saddle point plot function" begin
        println("\n--- Test 4: Saddle point plot (uses N) ---")
        sim_data = create_test_sim_data(Lx=4, Ly=6)
        
        # Test saddle_point_delta (which is called by saddle_point_plot)
        delta_guess, eqs, min_ix = saddle_point_delta(sim_data)
        println("  delta_guess: $delta_guess, min_ix: $min_ix")
        @test delta_guess >= 0
        @test length(eqs) > 0
        
        println("  ✓ Saddle point tests passed")
    end
    
    @testset "Rectangular with SC_L > 1" begin
        println("\n--- Test 5: Rectangular 4x6 with SC_L=2 ---")
        sim_data = create_test_sim_data(Lx=4, Ly=6, SC_L=2)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Run MC step and measurement
        for _ in 1:2
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Test BZ has correct generalized size
        BZ = lattice_data.measurement_data.BZ
        Lx_gen = 4 * 2  # Lx * SC_L
        Ly_gen = 6 * 2  # Ly * SC_L
        @test length(BZ) == Lx_gen * Ly_gen
        
        # Test spectral function path length
        # Correct formula: 2 * (Lx_gen - div(Lx_gen,2)) + (Ly_gen - div(Ly_gen,2))
        A = lattice_data.measurement_data.A
        expected_path_len = 2 * (Lx_gen - div(Lx_gen, 2)) + (Ly_gen - div(Ly_gen, 2))
        println("  A[1] length: $(length(A[1])), expected path_len: $expected_path_len")
        @test length(A[1]) == expected_path_len
        
        println("  ✓ SC_L > 1 rectangular tests passed")
    end
    
    @testset "D-wave rectangular plotting data" begin
        println("\n--- Test 6: D-wave rectangular 6x4 ---")
        sim_data = create_test_sim_data(Lx=6, Ly=4, pairing_symmetry="d-wave")
        
        lattice_data = Lattice_Data(sim_data)
        
        # Run MC step and measurement
        for _ in 1:2
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Verify measurement data structures are correct sizes
        @test size(lattice_data.measurement_data.BZ) == (6 * 4,)
        
        # Test phase correlation arrays have correct size
        # ls_list_reduced_size = L_min/2 + L_max/2 + 1 = 4/2 + 6/2 + 1 = 6
        expected_ls_size = Int(min(6,4)/2 + max(6,4)/2 + 1)
        @test length(lattice_data.measurement_data.Phase_Corr[1]) == expected_ls_size
        @test length(lattice_data.measurement_data.Phase_Corr_dwave_x[1]) == expected_ls_size
        @test length(lattice_data.measurement_data.Phase_Corr_dwave_y[1]) == expected_ls_size
        
        println("  ✓ D-wave rectangular tests passed")
    end

    @testset "Odd L support (5x5 square)" begin
        println("\n--- Test 7: Odd L square lattice 5x5 ---")
        sim_data = create_test_sim_data(Lx=5, Ly=5)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Verify k-lists have correct count for odd L
        kx_list = get_kx_list(5)
        ky_list = get_ky_list(5)
        @test length(kx_list) == 5
        @test length(ky_list) == 5
        println("  kx_list: $kx_list")
        
        # Run MC steps
        for _ in 1:3
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Test reduced_spectral_function works with odd L
        reduced_A = reduced_spectral_function(lattice_data, 0.0)
        println("  reduced_A length: $(length(reduced_A))")
        @test length(reduced_A) > 0
        
        # Test file name generation
        filename = choose_file_name(".", sim_data; date_folder=false, numbered=false)
        println("  Generated filename: $filename")
        @test occursin("5x5", filename)
        
        println("  ✓ Odd L square tests passed")
    end
    
    @testset "Odd L support (5x7 rectangular)" begin
        println("\n--- Test 8: Odd L rectangular 5x7 ---")
        sim_data = create_test_sim_data(Lx=5, Ly=7)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Verify k-lists have correct count for odd L
        kx_list = get_kx_list(5)
        ky_list = get_ky_list(7)
        @test length(kx_list) == 5
        @test length(ky_list) == 7
        
        # Check N = Lx * Ly
        @test lattice_data.sim_data.N == 35
        @test length(lattice_data.current_state.Delta_s) == 35
        
        # Run MC steps
        for _ in 1:3
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Test measurements work
        @test isfinite(lattice_data.measurement_data.n[1])
        @test isfinite(lattice_data.measurement_data.Kx[1])
        
        # Test BZ size
        @test length(lattice_data.measurement_data.BZ) == 5 * 7
        
        println("  ✓ Odd L rectangular tests passed")
    end
    
    @testset "Mixed even/odd L (4x5)" begin
        println("\n--- Test 9: Mixed even/odd L 4x5 ---")
        sim_data = create_test_sim_data(Lx=4, Ly=5)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Run MC steps
        for _ in 1:3
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Check Lxx_qy and Lxx_qx array sizes
        # Lxx_qy should have div(Ly,2) = div(5,2) = 2 elements
        # Lxx_qx should have div(Lx,2) = div(4,2) = 2 elements
        @test length(lattice_data.measurement_data.Lxx_qy[1]) == div(5, 2)  # 2
        @test length(lattice_data.measurement_data.Lxx_qx[1]) == div(4, 2)  # 2
        
        println("  Lxx_qy size: $(length(lattice_data.measurement_data.Lxx_qy[1])) (expected: $(div(5,2)))")
        println("  Lxx_qx size: $(length(lattice_data.measurement_data.Lxx_qx[1])) (expected: $(div(4,2)))")
        
        # Test reduced_spectral_function with mixed even/odd
        reduced_A = reduced_spectral_function(lattice_data, 0.0)
        @test length(reduced_A) > 0
        
        println("  ✓ Mixed even/odd L tests passed")
    end
    
    @testset "Odd L with SC_L > 1" begin
        println("\n--- Test 10: Odd L 5x5 with SC_L=2 ---")
        sim_data = create_test_sim_data(Lx=5, Ly=5, SC_L=2)
        
        lattice_data = Lattice_Data(sim_data)
        
        # L_gen = L * SC_L = 5 * 2 = 10 (even)
        Lx_gen = 5 * 2
        Ly_gen = 5 * 2
        
        # Run MC steps
        for _ in 1:2
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Test BZ has correct generalized size
        @test length(lattice_data.measurement_data.BZ) == Lx_gen * Ly_gen
        
        # Test Lxx_qy/Lxx_qx sizes with generalized dimensions
        @test length(lattice_data.measurement_data.Lxx_qy[1]) == div(Ly_gen, 2)  # 5
        @test length(lattice_data.measurement_data.Lxx_qx[1]) == div(Lx_gen, 2)  # 5
        
        println("  ✓ Odd L with SC_L > 1 tests passed")
    end
end

println("\n" * "="^50)
println("Phase 5 Tests Complete!")
println("="^50)
