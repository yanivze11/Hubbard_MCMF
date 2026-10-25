# Test file for Phase 3: Hamiltonian & Action Functions
# Tests make_H_matrix_K, MCMF_action_K, MCMF_action_K0, MC_stepV2!

println("=" ^ 60)
println("Testing Phase 3: Hamiltonian & Action Functions")
println("=" ^ 60)

# Include the main file
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

using Test

# Wrap all tests in a function to avoid world age issues
function run_phase3_tests()

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
# Test 1: make_H_matrix_K dimensions
# ============================================================================
println("\n--- Test 1: make_H_matrix_K Dimensions ---")

@testset "make_H_matrix_K Dimensions" begin
    
    @testset "Square lattice (Lx=Ly=4)" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 16
        # Hamiltonian should be 2N x 2N (particle-hole space)
        @test size(data.temporary_data.K) == (2*N, 2*N)
        
        # Test with explicit N parameter
        Delta_s = ones(ComplexF64, N) * 0.1
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        H = make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; N=N)
        @test size(H) == (2*N, 2*N)
    end
    
    @testset "Rectangular lattice (Lx=6, Ly=4)" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 24
        @test size(data.temporary_data.K) == (2*N, 2*N)
        
        Delta_s = ones(ComplexF64, N) * 0.1
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        H = make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; N=N)
        @test size(H) == (2*N, 2*N)
    end
end

# ============================================================================
# Test 2: Hamiltonian Hermiticity
# ============================================================================
println("\n--- Test 2: Hamiltonian Hermiticity ---")

@testset "Hamiltonian Hermiticity" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 16
        Delta_s = (rand(ComplexF64, N) .- 0.5) * 0.2
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        
        H = make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; N=N)
        
        # Check Hermiticity: H = H†
        @test isapprox(H, H', atol=1e-12)
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 24
        Delta_s = (rand(ComplexF64, N) .- 0.5) * 0.2
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        
        H = make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; N=N)
        
        @test isapprox(H, H', atol=1e-12)
    end
    
    @testset "D-wave rectangular" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4, pairing_symmetry="d-wave")
        data = Lattice_Data(sim_data)
        
        N = 24
        Delta_s = zeros(ComplexF64, N)
        Delta_dx = (rand(ComplexF64, N) .- 0.5) * 0.2
        Delta_dy = (rand(ComplexF64, N) .- 0.5) * 0.2
        
        H = make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; N=N)
        
        @test isapprox(H, H', atol=1e-12)
    end
end

# ============================================================================
# Test 3: MCMF_action_K
# ============================================================================
println("\n--- Test 3: MCMF_action_K ---")

@testset "MCMF_action_K" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 16
        Delta_s = ones(ComplexF64, N) * 0.1
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        
        S, Energies, U_matrix = MCMF_action_K(data, Delta_s, Delta_dx, Delta_dy)
        
        # Action should be real
        @test isreal(S)
        
        # Energies should have correct dimensions
        @test size(Energies) == (2*N, sim_data.SC_L^2)
        
        # U_matrix should have correct dimensions
        @test size(U_matrix) == (2*N, 2*N, sim_data.SC_L^2)
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 24
        Delta_s = ones(ComplexF64, N) * 0.1
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        
        S, Energies, U_matrix = MCMF_action_K(data, Delta_s, Delta_dx, Delta_dy)
        
        @test isreal(S)
        @test size(Energies) == (2*N, sim_data.SC_L^2)
        @test size(U_matrix) == (2*N, 2*N, sim_data.SC_L^2)
    end
end

# ============================================================================
# Test 4: MCMF_action_K0
# ============================================================================
println("\n--- Test 4: MCMF_action_K0 ---")

@testset "MCMF_action_K0" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 16
        Delta_s = ones(ComplexF64, N) * 0.1
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        
        S, Energies, U_matrix = MCMF_action_K0(data, Delta_s, Delta_dx, Delta_dy)
        
        @test isreal(S)
        @test length(Energies) == 2*N
        @test size(U_matrix) == (2*N, 2*N)
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = 24
        Delta_s = ones(ComplexF64, N) * 0.1
        Delta_dx = zeros(ComplexF64, N)
        Delta_dy = zeros(ComplexF64, N)
        
        S, Energies, U_matrix = MCMF_action_K0(data, Delta_s, Delta_dx, Delta_dy)
        
        @test isreal(S)
        @test length(Energies) == 2*N
        @test size(U_matrix) == (2*N, 2*N)
    end
end

# ============================================================================
# Test 5: MC_stepV2! (Monte Carlo step)
# ============================================================================
println("\n--- Test 5: MC_stepV2! ---")

@testset "MC_stepV2!" begin
    
    @testset "Square lattice s-wave" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        data = Lattice_Data(sim_data)
        
        initial_step = data.current_state.step_count
        
        # Run one MC step
        MC_stepV2!(data; method="exact", loop_random=false, pairing="s-wave")
        
        # Step count should increase
        @test data.current_state.step_count == initial_step + 1
    end
    
    @testset "Rectangular lattice s-wave" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        initial_step = data.current_state.step_count
        initial_S = data.current_state.S
        
        # Run one MC step
        MC_stepV2!(data; method="exact", loop_random=false, pairing="s-wave")
        
        @test data.current_state.step_count == initial_step + 1
        # Action should be finite
        @test isfinite(data.current_state.S)
    end
    
    @testset "Rectangular lattice d-wave" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4, pairing_symmetry="d-wave")
        data = Lattice_Data(sim_data)
        
        initial_step = data.current_state.step_count
        
        MC_stepV2!(data; method="exact", loop_random=false, pairing="d-wave")
        
        @test data.current_state.step_count == initial_step + 1
        @test isfinite(data.current_state.S)
    end
end

# ============================================================================
# Test 6: Consistency - Square should give same structure as old code
# ============================================================================
println("\n--- Test 6: Consistency Check ---")

@testset "Square vs Rectangular Consistency" begin
    # For Lx=Ly, results should have same structure
    sim_sq = create_test_sim_data(Lx=4, Ly=4)
    data_sq = Lattice_Data(sim_sq)
    
    N = 16
    @test data_sq.sim_data.N == N
    @test data_sq.sim_data.Lx == data_sq.sim_data.Ly
    
    # Action should be computable
    S, E, U = MCMF_action_K(data_sq, data_sq.current_state.Delta_s, 
                            data_sq.current_state.Delta_dx, data_sq.current_state.Delta_dy)
    @test isfinite(S)
    
    # Same for rectangular
    sim_rect = create_test_sim_data(Lx=6, Ly=4)
    data_rect = Lattice_Data(sim_rect)
    
    S_rect, _, _ = MCMF_action_K(data_rect, data_rect.current_state.Delta_s,
                                  data_rect.current_state.Delta_dx, data_rect.current_state.Delta_dy)
    @test isfinite(S_rect)
end

# ============================================================================
# Test 7: Eigenvalue spectrum
# ============================================================================
println("\n--- Test 7: Eigenvalue Spectrum ---")

@testset "Eigenvalue Spectrum" begin
    sim_data = create_test_sim_data(Lx=6, Ly=4)
    data = Lattice_Data(sim_data)
    
    N = 24
    
    # Eigenvalues should be real for Hermitian Hamiltonian
    Energies = data.current_state.Energies
    @test all(isreal.(Energies))
    
    # For BdG Hamiltonian with particle-hole symmetry, 
    # eigenvalues should come in ±E pairs (approximately)
    E_flat = vec(Energies)
    sorted_E = sort(E_flat)
    
    # Check that there are both positive and negative eigenvalues
    @test any(sorted_E .< 0)
    @test any(sorted_E .> 0)
end

# ============================================================================
# Test 8: Phase 3b - Rectangular TCluster (Traveling Cluster Approximation)
# ============================================================================
println("\n--- Test 8: Phase 3b - Rectangular TCluster ---")

@testset "Rectangular TCluster" begin
    
    @testset "TCluster data structures" begin
        # Test with square TCluster (2x2)
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        TCluster_N = sim_data.TCluster_size_x * sim_data.TCluster_size_y
        N = sim_data.N
        
        # Check TCluster dimensions in SimData
        @test sim_data.TCluster_size_x == 2
        @test sim_data.TCluster_size_y == 2
        @test TCluster_N == 4
        
        # Legacy accessor should return TCluster_size_x
        @test sim_data.TCluster_size == sim_data.TCluster_size_x
        
        # Check TCluster assignments array
        @test length(data.const_struct.TCluster_assignments) == N
        @test length(data.const_struct.TCluster_assignments[1]) == TCluster_N
        
        # Check TCluster NN matrix
        @test length(data.const_struct.TCluster_NN_mats) >= 1
        @test size(data.const_struct.TCluster_NN_mats[1]) == (TCluster_N, TCluster_N)
        
        # Check TCluster temporary arrays
        @test length(data.temporary_data.TCluster_Delta_s) == TCluster_N
        @test length(data.temporary_data.TCluster_Delta_dx) == TCluster_N
        @test length(data.temporary_data.TCluster_Delta_dy) == TCluster_N
        @test length(data.temporary_data.TCluster_V) == TCluster_N
        @test size(data.temporary_data.TCluster_K) == (2*TCluster_N, 2*TCluster_N)
    end
    
    @testset "TCluster assignments correctness" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = sim_data.N
        TCluster_N = sim_data.TCluster_size_x * sim_data.TCluster_size_y
        
        # Each assignment should have TCluster_N unique valid indices
        for i in 1:N
            assignments = data.const_struct.TCluster_assignments[i]
            @test length(assignments) == TCluster_N
            @test all(1 .<= assignments .<= N)
            # Indices should be unique within each cluster
            @test length(unique(assignments)) == TCluster_N
        end
    end
    
    @testset "MCMF_action_TCluster_K0 square cluster" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        TCluster_N = sim_data.TCluster_size_x * sim_data.TCluster_size_y
        
        # Call TCluster action for site 1
        S, energies, U_matrix = MCMF_action_TCluster_K0(data, 1)
        
        # Action should be real and finite
        @test isreal(S)
        @test isfinite(S)
        
        # Energies should have correct dimension (2*TCluster_N for BdG)
        @test length(energies) == 2 * TCluster_N
        @test all(isreal.(energies))
        
        # U_matrix should have correct dimensions
        @test size(U_matrix) == (2*TCluster_N, 2*TCluster_N)
    end
    
    @testset "MCMF_action_TCluster_K0 all sites" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        data = Lattice_Data(sim_data)
        
        N = sim_data.N
        
        # Action should be computable for all sites
        for i in 1:N
            S, energies, _ = MCMF_action_TCluster_K0(data, i)
            @test isfinite(S)
            @test all(isfinite.(energies))
        end
    end
    
    @testset "Rectangular TCluster (3x2)" begin
        # Create SimData with non-square TCluster
        N = 6 * 4
        sim_data_rect_tc = SimData(
            6, 4, N, 1,              # Lx, Ly, N, SC_L
            1.0, 1.0, 0.0, 12345,    # U, t, V, seed
            0.0, 10.0,               # mu, beta
            10, 100, 10, 10,         # num_measure, num_thermal, measuring_interval, saving_interval
            collect(-3.0:0.1:3.0),   # w_list
            "uniform", "uniform", "exact", "s-wave",  # guess, U_disorder, diag_method, pairing
            "test_dir", 1, 1,        # directory, num_cores, num_threads
            Dates.now(),             # start_time
            collect(-2.0:0.2:2.0),   # w_conductivity_range
            [0.0, 10.0, 0.7, 0.7],   # U_disorder_params
            1.0,                     # U_impurity
            3, 2,                    # TCluster_size_x=3, TCluster_size_y=2 (rectangular!)
            (false, 0.8), false,     # smooth_U, loop_random
            0.0, "LandauX", "PBC"    # B, gauge_choice, boundary_conditions
        )
        
        data_rect_tc = Lattice_Data(sim_data_rect_tc)
        
        TCluster_N = 3 * 2  # 6 sites in cluster
        N = 24
        
        # Verify TCluster dimensions
        @test sim_data_rect_tc.TCluster_size_x == 3
        @test sim_data_rect_tc.TCluster_size_y == 2
        
        # Check TCluster structures have correct sizes
        @test length(data_rect_tc.const_struct.TCluster_assignments[1]) == TCluster_N
        @test size(data_rect_tc.const_struct.TCluster_NN_mats[1]) == (TCluster_N, TCluster_N)
        @test length(data_rect_tc.temporary_data.TCluster_Delta_s) == TCluster_N
        @test size(data_rect_tc.temporary_data.TCluster_K) == (2*TCluster_N, 2*TCluster_N)
        
        # Action should work
        S, energies, U_matrix = MCMF_action_TCluster_K0(data_rect_tc, 1)
        @test isfinite(S)
        @test length(energies) == 2 * TCluster_N
        @test size(U_matrix) == (2*TCluster_N, 2*TCluster_N)
    end
    
    @testset "TCluster with larger lattice" begin
        # 8x6 lattice with 4x3 TCluster
        N = 8 * 6
        sim_data_large = SimData(
            8, 6, N, 1,              # Lx, Ly, N, SC_L
            1.0, 1.0, 0.0, 12345,    # U, t, V, seed
            0.0, 10.0,               # mu, beta
            10, 100, 10, 10,         # num_measure, num_thermal, measuring_interval, saving_interval
            collect(-3.0:0.1:3.0),   # w_list
            "uniform", "uniform", "exact", "s-wave",  # guess, U_disorder, diag_method, pairing
            "test_dir", 1, 1,        # directory, num_cores, num_threads
            Dates.now(),             # start_time
            collect(-2.0:0.2:2.0),   # w_conductivity_range
            [0.0, 10.0, 0.7, 0.7],   # U_disorder_params
            1.0,                     # U_impurity
            4, 3,                    # TCluster_size_x=4, TCluster_size_y=3
            (false, 0.8), false,     # smooth_U, loop_random
            0.0, "LandauX", "PBC"    # B, gauge_choice, boundary_conditions
        )
        
        data_large = Lattice_Data(sim_data_large)
        
        TCluster_N = 4 * 3  # 12 sites in cluster
        
        # Verify structure
        @test length(data_large.const_struct.TCluster_assignments[1]) == TCluster_N
        @test size(data_large.temporary_data.TCluster_K) == (2*TCluster_N, 2*TCluster_N)
        
        # Test action at multiple sites
        for i in [1, 10, 24, 48]
            S, energies, _ = MCMF_action_TCluster_K0(data_large, i)
            @test isfinite(S)
            @test length(energies) == 2 * TCluster_N
        end
    end
    
    @testset "TCluster d-wave pairing" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4, pairing_symmetry="d-wave")
        data = Lattice_Data(sim_data)
        
        # TCluster should work with d-wave pairing
        S, energies, _ = MCMF_action_TCluster_K0(data, 1)
        @test isfinite(S)
        @test all(isfinite.(energies))
    end
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Phase 3 Tests Complete!")
println("=" ^ 60)
println("\nIf all tests passed, Phase 3 implementation is correct.")
println("You can now proceed to Phase 4: Measurements")

end  # end of run_phase3_tests function

# Run tests using invokelatest to avoid world age issues
Base.invokelatest(run_phase3_tests)
