# Test file for Phase 4: Measurements
# Tests measurement functions with rectangular lattice support

println("=" ^ 60)
println("Testing Phase 4: Measurements")
println("=" ^ 60)

# Include the main file
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

using Test

# Wrap all tests in a function to avoid world age issues
function run_phase4_tests()

# ============================================================================
# Helper: Create a minimal SimData for testing
# ============================================================================
function create_test_sim_data(; Lx=6, Ly=4, pairing_symmetry="s-wave")
    N = Lx * Ly
    SimData(
        Lx,                          # Lx
        Ly,                          # Ly
        N,                           # N
        1,                           # SC_L (supercell size)
        1.0,                         # U
        1.0,                         # t
        0.0,                         # V (disorder)
        12345,                       # seed
        0.0,                         # mu
        10.0,                        # beta
        10,                          # num_measure
        100,                         # num_thermal
        10,                          # measuring_interval
        10,                          # saving_interval
        collect(-3.0:0.1:3.0),       # w_list
        "uniform",                   # guess
        "uniform",                   # U_disorder
        "exact",                     # diag_method
        pairing_symmetry,            # pairing_symmetry
        "test_dir",                  # directory
        1,                           # num_cores
        1,                           # num_threads
        Dates.now(),                 # start_time
        collect(-2.0:0.2:2.0),       # w_conductivity_range
        [0.0, 10.0, 0.7, 0.7],       # U_disorder_params
        1.0,                         # U_impurity
        2,                           # TCluster_size_x
        2,                           # TCluster_size_y
        (false, 0.8),                # smooth_U
        false,                       # loop_random
        0.0,                         # B (magnetic field)
        "LandauX",                   # gauge_choice
        "PBC"                        # boundary_conditions
    )
end

# ============================================================================
# Test 1: avrage_n functions
# ============================================================================
println("\n--- Test 1: Average Occupation (n) ---")

@testset "Average Occupation" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        n = avrage_n_K(data)
        @test isfinite(n)
        @test 0.0 <= n <= 2.0  # Occupation should be between 0 and 2 (for spin)
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        n = avrage_n_K(data)
        @test isfinite(n)
        @test 0.0 <= n <= 2.0
    end
end

# ============================================================================
# Test 2: compute_Kx_operator
# ============================================================================
println("\n--- Test 2: Kinetic Energy Operator (Kx) ---")

@testset "Kinetic Energy Kx" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        Kx = compute_Kx_operator(data)
        @test isfinite(Kx)
        @test isreal(Kx)
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        Kx = compute_Kx_operator(data)
        @test isfinite(Kx)
        @test isreal(Kx)
    end
end

# ============================================================================
# Test 3: compute_D_dagger_D functions
# ============================================================================
println("\n--- Test 3: D†D Computation ---")

@testset "D dagger D" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        DdD = compute_D_dagger_D_K(data)
        @test isfinite(DdD)
        @test DdD >= 0.0  # D†D should be non-negative
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        DdD = compute_D_dagger_D_K(data)
        @test isfinite(DdD)
        @test DdD >= 0.0
    end
end

# ============================================================================
# Test 4: compute_Ps_l_K
# ============================================================================
println("\n--- Test 4: Pairing Correlation Ps_l ---")

@testset "Pairing Correlation Ps_l" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        # Test at r=[0,0] - should be same as DdD in some limit
        Ps_l_00 = compute_Ps_l_K(data, [0, 0])
        @test isfinite(Ps_l_00)
        
        # Test at various displacements
        for l in [[1,0], [0,1], [1,1], [2,2]]
            Ps_l = compute_Ps_l_K(data, l)
            @test isfinite(Ps_l)
        end
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        # Test at various displacements within bounds
        for l in [[0,0], [1,0], [0,1], [2,1], [3,2]]
            Ps_l = compute_Ps_l_K(data, l)
            @test isfinite(Ps_l)
        end
    end
end

# ============================================================================
# Test 5: Phase Correlation
# ============================================================================
println("\n--- Test 5: Phase Correlation ---")

@testset "Phase Correlation" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        # Phase correlation at r=[0,0] should be 1
        pc_00 = compute_phase_correlation(data, [0, 0])
        @test isfinite(pc_00)
        @test pc_00 ≈ 1.0 atol=1e-10
        
        # Other displacements
        for r in [[1,0], [0,1], [1,1]]
            pc = compute_phase_correlation(data, r)
            @test isfinite(pc)
            @test -1.0 <= pc <= 1.0  # Cosine-like correlation
        end
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        pc_00 = compute_phase_correlation(data, [0, 0])
        @test pc_00 ≈ 1.0 atol=1e-10
        
        for r in [[1,0], [0,1], [2,1]]
            pc = compute_phase_correlation(data, r)
            @test isfinite(pc)
            @test -1.0 <= pc <= 1.0
        end
    end
end

# ============================================================================
# Test 6: current_correlation_q_vec
# ============================================================================
println("\n--- Test 6: Current Correlation ---")

@testset "Current Correlation" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        # Test at a few q points
        q_vecs = [[0.0, 0.5], [0.5, 0.0], [0.3, 0.3]]
        Lxx_vec = current_correlation_q_vec(data, q_vecs)
        
        @test length(Lxx_vec) == 3
        @test all(isfinite.(Lxx_vec))
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        # q vectors using rectangular lattice spacing
        Lx, Ly = 6, 4
        q_vecs = [[0.0, 2π/Ly], [2π/Lx, 0.0], [π/Lx, π/Ly]]
        Lxx_vec = current_correlation_q_vec(data, q_vecs)
        
        @test length(Lxx_vec) == 3
        @test all(isfinite.(Lxx_vec))
    end
end

# ============================================================================
# Test 7: compute_full_LDOS
# ============================================================================
println("\n--- Test 7: LDOS ---")

@testset "Full LDOS" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        LDOS = compute_full_LDOS(data, 0.0)
        N = 16
        @test length(LDOS) == N
        @test all(isfinite.(LDOS))
        @test all(LDOS .>= 0.0)  # LDOS should be non-negative
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        LDOS = compute_full_LDOS(data, 0.0)
        N = 24
        @test length(LDOS) == N
        @test all(isfinite.(LDOS))
        @test all(LDOS .>= 0.0)
    end
end

# ============================================================================
# Test 8: make_measurement_K! (full measurement)
# ============================================================================
println("\n--- Test 8: Full Measurement ---")

@testset "make_measurement_K!" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        # Run one measurement
        measurements = make_measurement_K!(data, 1)
        
        @test isfinite(measurements.Ss[1])
        @test isfinite(measurements.n[1])
        @test isfinite(measurements.Kx[1])
        @test isfinite(measurements.stiffness[1])
        @test isfinite(measurements.Ps[1])
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        measurements = make_measurement_K!(data, 1)
        
        @test isfinite(measurements.Ss[1])
        @test isfinite(measurements.n[1])
        @test isfinite(measurements.Kx[1])
        @test isfinite(measurements.stiffness[1])
        @test isfinite(measurements.Ps[1])
        @test isfinite(measurements.DdD[1])
    end
    
    @testset "D-wave rectangular" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4, pairing_symmetry="d-wave")
        data = Lattice_Data(sim_data)
        
        measurements = make_measurement_K!(data, 1)
        
        @test isfinite(measurements.Ss[1])
        @test isfinite(measurements.n[1])
        @test isfinite(measurements.Kx[1])
    end
end

# ============================================================================
# Test 9: M-matrix functions
# ============================================================================
println("\n--- Test 9: M-matrix Functions ---")

@testset "M-matrix Functions" begin
    sim_data = create_test_sim_data(Lx=6, Ly=4)
    data = Lattice_Data(sim_data)
    N = 24
    
    M = make_M_matrix(data; w=0.0)
    @test size(M) == (2*N, 2*N)
    @test all(isfinite.(M))
    
    M_q = make_M_matrix_q(data)
    @test size(M_q) == (2*N, 2*N)
    @test all(isfinite.(M_q))
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Phase 4 Tests Complete!")
println("=" ^ 60)
println("\nIf all tests passed, Phase 4 (Measurements) is correct for rectangular lattices.")
println("You can now proceed to Phase 5: Spectral & Current Functions")

end  # end of run_phase4_tests function

# Run tests using invokelatest to avoid world age issues
Base.invokelatest(run_phase4_tests)
