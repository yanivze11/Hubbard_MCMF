# Test file for Phase 2: Data Structures
# Tests ConstStruct, TemporaryData, MeasureData, and CurrentState initialization

println("=" ^ 60)
println("Testing Phase 2: Data Structures")
println("=" ^ 60)

# Include the main file
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

using Test

# ============================================================================
# Helper: Create a minimal SimData for testing
# ============================================================================
function create_test_sim_data(; Lx=6, Ly=4)
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
        "uniform",                      # guess - use "rand" to avoid saddle_point_delta call
        "uniform",                   # U_disorder
        "exact",                     # diag_method
        "s-wave",                    # pairing_symmetry
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

# Wrap all tests in a function to avoid world age issues
function run_phase2_tests()

# ============================================================================
# Test 1: SimData Legacy Accessor
# ============================================================================
println("\n--- Test 1: SimData Legacy Accessor ---")

@testset "SimData Legacy Accessor" begin
    sim_data = create_test_sim_data(Lx=6, Ly=4)
    
    @test sim_data.Lx == 6
    @test sim_data.Ly == 4
    @test sim_data.N == 24
    
    # Legacy accessor: .L should return Lx
    @test sim_data.L == 6
    @test sim_data.L == sim_data.Lx
end

# ============================================================================
# Test 2: ConstStruct Initialization
# ============================================================================
println("\n--- Test 2: ConstStruct Initialization ---")

@testset "ConstStruct Initialization" begin
    
    @testset "Square lattice (Lx=Ly=4)" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        const_struct = Const_Struct(sim_data)
        
        @test const_struct.Lx == 4
        @test const_struct.Ly == 4
        @test const_struct.N == 16
        
        # Check CI and LI are created correctly
        @test size(const_struct.CI) == (4, 4)
        @test size(const_struct.LI) == (4, 4)
        
        # Check arrays have correct sizes
        @test length(const_struct.x) == 16
        @test length(const_struct.y) == 16
        @test length(const_struct.V_matrix) == 16
        @test length(const_struct.U_params) == 16
        @test length(const_struct.new_ix_list) == 16
        @test length(const_struct.new_iy_list) == 16
        
        # Check k-lists
        @test length(const_struct.kx_list) == 4
        @test length(const_struct.ky_list) == 4
        
        # Legacy accessor
        @test const_struct.ki_list == const_struct.kx_list
        
        # Check neighbor matrices
        @test size(const_struct.NN_mat[1]) == (16, 16)
        @test size(const_struct.NN_x) == (16, 16)
        @test size(const_struct.NN_y) == (16, 16)
    end
    
    @testset "Rectangular lattice (Lx=6, Ly=4)" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        const_struct = Const_Struct(sim_data)
        
        @test const_struct.Lx == 6
        @test const_struct.Ly == 4
        @test const_struct.N == 24
        
        # Check CI and LI for rectangular
        @test size(const_struct.CI) == (6, 4)
        @test size(const_struct.LI) == (6, 4)
        
        # Check arrays have correct sizes
        @test length(const_struct.x) == 24
        @test length(const_struct.y) == 24
        @test length(const_struct.V_matrix) == 24
        @test length(const_struct.U_params) == 24
        
        # Check k-lists are different
        @test length(const_struct.kx_list) == 6
        @test length(const_struct.ky_list) == 4
        
        # Check neighbor matrices
        @test size(const_struct.NN_mat[1]) == (24, 24)
        @test size(const_struct.NN_x) == (24, 24)
        @test size(const_struct.NN_y) == (24, 24)
        
        # k_vectors should have N elements
        @test length(const_struct.k_vectors) == 24
    end
end

# ============================================================================
# Test 3: TemporaryData Initialization
# ============================================================================
println("\n--- Test 3: TemporaryData Initialization ---")

@testset "TemporaryData Initialization" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        temp_data = Temporary_Data(sim_data)
        
        N = 16
        @test size(temp_data.Energies_Bckup, 1) == 2*N
        @test size(temp_data.U_matrix_Bckup, 1) == 2*N
        @test size(temp_data.K) == (2*N, 2*N)
        @test length(temp_data.Energies_K) == 2*N
        @test length(temp_data.LDOS_vec) == N
        @test size(temp_data.S_r) == (4, 4)
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        temp_data = Temporary_Data(sim_data)
        
        N = 24
        @test size(temp_data.Energies_Bckup, 1) == 2*N
        @test size(temp_data.U_matrix_Bckup, 1) == 2*N
        @test size(temp_data.K) == (2*N, 2*N)
        @test length(temp_data.Energies_K) == 2*N
        @test length(temp_data.LDOS_vec) == N
        @test size(temp_data.S_r) == (6, 4)  # Now Lx x Ly
    end
end

# ============================================================================
# Test 4: MeasureData Initialization
# ============================================================================
println("\n--- Test 4: MeasureData Initialization ---")

@testset "MeasureData Initialization" begin
    
    @testset "Square lattice" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        measure_data = Initial_measurments(sim_data)
        
        N = 16
        saving_interval = sim_data.saving_interval
        
        @test length(measure_data.Ss) == saving_interval
        @test length(measure_data.n) == saving_interval
        @test length(measure_data.LDOS) == length(sim_data.w_list)
        @test length(measure_data.LDOS[1]) == N  # Now uses N
    end
    
    @testset "Rectangular lattice" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        measure_data = Initial_measurments(sim_data)
        
        N = 24
        saving_interval = sim_data.saving_interval
        
        @test length(measure_data.Ss) == saving_interval
        @test length(measure_data.LDOS) == length(sim_data.w_list)
        @test length(measure_data.LDOS[1]) == N  # N = 24 for 6x4
        
        # BZ should be Lx*SC_L * Ly*SC_L = 6*4 = 24 for SC_L=1
        @test length(measure_data.BZ) == 24
    end
end

# ============================================================================
# Test 5: Initial_state (CurrentState) Initialization
# ============================================================================
println("\n--- Test 5: Initial_state Initialization ---")

@testset "Initial_state (CurrentState)" begin
    
    @testset "Square lattice s-wave" begin
        sim_data = create_test_sim_data(Lx=4, Ly=4)
        const_struct = Const_Struct(sim_data)
        current_state = Initial_state(sim_data; const_data=const_struct)
        
        N = 16
        @test length(current_state.Delta_s) == N
        @test length(current_state.Delta_dx) == N
        @test length(current_state.Delta_dy) == N
    end
    
    @testset "Rectangular lattice s-wave" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        const_struct = Const_Struct(sim_data)
        current_state = Initial_state(sim_data; const_data=const_struct)
        
        N = 24
        @test length(current_state.Delta_s) == N
        @test length(current_state.Delta_dx) == N
        @test length(current_state.Delta_dy) == N
    end
    
    @testset "Rectangular lattice d-wave" begin
        sim_data = create_test_sim_data(Lx=6, Ly=4)
        # Change to d-wave - create new SimData with d-wave pairing
        sim_data_dwave = SimData(
            sim_data.Lx, sim_data.Ly, sim_data.N, sim_data.SC_L,
            sim_data.U, sim_data.t, sim_data.V, sim_data.seed,
            sim_data.mu, sim_data.beta, sim_data.num_measure,
            sim_data.num_thermal, sim_data.measuring_interval,
            sim_data.saving_interval, sim_data.w_list, 
            sim_data.guess,  
            sim_data.U_disorder, sim_data.diag_method,
            "d-wave",  # pairing_symmetry
            sim_data.directory, sim_data.num_cores, sim_data.num_threads,
            sim_data.start_time, sim_data.w_conductivity_range,
            sim_data.U_disorder_params, sim_data.U_impurity,
            sim_data.TCluster_size_x, sim_data.TCluster_size_y,  # Both TCluster dimensions
            sim_data.smooth_U, sim_data.loop_random,
            sim_data.B, sim_data.gauge_choice, sim_data.boundary_conditions
        )
        
        const_struct = Const_Struct(sim_data_dwave)
        current_state = Base.invokelatest(Initial_state, sim_data_dwave; const_data=const_struct)
        
        N = 24
        @test length(current_state.Delta_s) == N
        @test length(current_state.Delta_dx) == N
        @test length(current_state.Delta_dy) == N
        
        # For d-wave, Delta_s should be zeros
        @test all(current_state.Delta_s .== 0.0)
    end
end

# ============================================================================
# Test 6: make_V_matrix and make_U_params
# ============================================================================
println("\n--- Test 6: make_V_matrix and make_U_params ---")

@testset "make_V_matrix" begin
    # Rectangular
    V_rect = make_V_matrix(6, 4, 1.0; seed=123)
    @test length(V_rect) == 24
    
    # Square (legacy)
    V_sq = make_V_matrix(4, 1.0; seed=123)
    @test length(V_sq) == 16
end

@testset "make_U_params" begin
    # Rectangular
    U, Ux, Uy = make_U_params(1.0, 6, 4, "uniform")
    @test length(U) == 24
    @test length(Ux) == 24
    @test length(Uy) == 24
    
    # For s-wave with uniform, U should be all ones
    @test all(U .≈ 1.0)
    
    # Square (legacy)
    U_sq, _, _ = make_U_params(1.0, 4, "uniform")
    @test length(U_sq) == 16
end

# ============================================================================
# Test 7: Peierls Phases
# ============================================================================
println("\n--- Test 7: Peierls Phases ---")

@testset "make_peierls_phases" begin
    sim_data = create_test_sim_data(Lx=6, Ly=4)
    phases = make_peierls_phases(sim_data)
    
    N = 24
    @test size(phases) == (N, 2)
    
    # For B=0, all phases should be 1
    @test all(phases .== 1.0)
end

# ============================================================================
# Test 8: Consistency Checks
# ============================================================================
println("\n--- Test 8: Consistency Checks ---")

@testset "Square vs Rectangular Consistency" begin
    # For square lattice, rectangular code should give same results
    sim_sq = create_test_sim_data(Lx=4, Ly=4)
    const_sq = Const_Struct(sim_sq)
    
    # Check basic dimensions
    @test const_sq.Lx == const_sq.Ly
    @test const_sq.N == const_sq.Lx^2
    
    # kx_list and ky_list should be equal for square
    @test const_sq.kx_list == const_sq.ky_list
end

@testset "Index consistency in ConstStruct" begin
    sim_data = create_test_sim_data(Lx=6, Ly=4)
    const_struct = Const_Struct(sim_data)
    
    N = 24
    CI = const_struct.CI
    LI = const_struct.LI
    
    # Check that stored x, y arrays match CI
    for ix in 1:N
        @test const_struct.x[ix] == get_x(ix, CI)
        @test const_struct.y[ix] == get_y(ix, CI)
    end
    
    # Check new_ix_list and new_iy_list
    for ix in 1:N
        x_coord = const_struct.x[ix]
        y_coord = const_struct.y[ix]
        
        # new_ix should be neighbor in +x direction
        expected_new_ix = get_ix(add_xy_ix(x_coord, 1, 6), y_coord, LI)
        @test const_struct.new_ix_list[ix] == expected_new_ix
        
        # new_iy should be neighbor in +y direction
        expected_new_iy = get_ix(x_coord, add_xy_ix(y_coord, 1, 4), LI)
        @test const_struct.new_iy_list[ix] == expected_new_iy
    end
end

# ============================================================================
# Test 9: Disorder Functions
# ============================================================================
println("\n--- Test 9: Disorder Functions ---")

@testset "make_U_gaussian" begin
    @testset "Rectangular lattice" begin
        Lx, Ly = 6, 4
        N = Lx * Ly
        U_mean = 2.0
        U_std = 0.5
        
        U_gaussian = make_U_gaussian(Lx, Ly, U_mean, U_std; seed=123)
        
        # Check size - returns 1D array of length N
        @test length(U_gaussian) == N
        
        # With seed, should be reproducible
        U_gaussian2 = make_U_gaussian(Lx, Ly, U_mean, U_std; seed=123)
        @test U_gaussian == U_gaussian2
        
        # Different seed gives different result
        U_gaussian3 = make_U_gaussian(Lx, Ly, U_mean, U_std; seed=456)
        @test U_gaussian != U_gaussian3
    end
    
    @testset "Square lattice (legacy)" begin
        L = 4
        U_gaussian_sq = make_U_gaussian(L, 2.0, 0.5; seed=123)
        @test length(U_gaussian_sq) == L^2
    end
    
    @testset "Smoothing option" begin
        Lx, Ly = 8, 6
        U_no_smooth = make_U_gaussian(Lx, Ly, 2.0, 0.5; seed=123, smooth=false)
        U_smooth = make_U_gaussian(Lx, Ly, 2.0, 0.5; seed=123, smooth=true, smooth_sigma=1.0)
        
        # Both should have same length
        @test length(U_no_smooth) == length(U_smooth)
        
        # Smoothed should be different
        @test U_no_smooth != U_smooth
    end
end

@testset "make_U_puddles" begin
    @testset "Rectangular lattice" begin
        Lx, Ly = 6, 4
        N = Lx * Ly
        
        U_puddles = make_U_puddles(Lx, Ly; V0=0.0, VI=1.0, λ=1.0, rz=0.5, 
                                   impurity_ratio=0.5, seed=123)
        
        # Check size - returns 1D array of length N
        @test length(U_puddles) == N
        
        # With seed, should be reproducible
        U_puddles2 = make_U_puddles(Lx, Ly; V0=0.0, VI=1.0, λ=1.0, rz=0.5, 
                                    impurity_ratio=0.5, seed=123)
        @test U_puddles == U_puddles2
    end
    
    @testset "Square lattice (legacy)" begin
        L = 4
        U_puddles_sq = make_U_puddles(L; V0=0.0, VI=1.0, λ=1.0, rz=0.5, 
                                       impurity_ratio=0.5, seed=123)
        @test length(U_puddles_sq) == L^2
    end
    
    @testset "Impurity ratio affects number of impurities" begin
        Lx, Ly = 10, 8
        
        # Low impurity ratio
        U_low = make_U_puddles(Lx, Ly; V0=0.0, VI=10.0, λ=0.5, rz=0.5, 
                               impurity_ratio=0.1, seed=123)
        # High impurity ratio
        U_high = make_U_puddles(Lx, Ly; V0=0.0, VI=10.0, λ=0.5, rz=0.5, 
                                impurity_ratio=0.9, seed=123)
        
        # Should have different distributions (different number of impurities)
        @test sum(abs.(U_low) .> 0.1) != sum(abs.(U_high) .> 0.1)
    end
end

@testset "make_U_on_off_puddles" begin
    @testset "Rectangular lattice" begin
        Lx, Ly = 6, 4
        N = Lx * Ly
        U_val = 3.0
        
        U_onoff = make_U_on_off_puddles(Lx, Ly, U_val; num_puddles=5, 
                                        puddle_size=2.0, seed=123)
        
        # Check size - returns 1D array of length N
        @test length(U_onoff) == N
        
        # Values should be multiples of U_val (0, U_val, 2*U_val, etc. due to overlapping puddles)
        @test all(x -> x >= 0.0 && isapprox(x % U_val, 0.0, atol=1e-10) || isapprox(x, 0.0, atol=1e-10), U_onoff)
        
        # With seed, should be reproducible
        U_onoff2 = make_U_on_off_puddles(Lx, Ly, U_val; num_puddles=5, 
                                         puddle_size=2.0, seed=123)
        @test U_onoff == U_onoff2
    end
    
    @testset "Square lattice (legacy)" begin
        L = 4
        U_onoff_sq = make_U_on_off_puddles(L, 3.0; num_puddles=3, 
                                           puddle_size=2.0, seed=123)
        @test length(U_onoff_sq) == L^2
    end
    
    @testset "Puddle coverage" begin
        Lx, Ly = 8, 6
        U_val = 2.0
        
        # More puddles = more coverage
        U_few = make_U_on_off_puddles(Lx, Ly, U_val; num_puddles=1, 
                                      puddle_size=2.0, seed=123)
        U_many = make_U_on_off_puddles(Lx, Ly, U_val; num_puddles=10, 
                                       puddle_size=2.0, seed=123)
        
        # Many puddles should have more non-zero sites
        @test sum(U_many .!= 0) >= sum(U_few .!= 0)
    end
end

@testset "make_U_params with different disorder types" begin
    Lx, Ly = 6, 4
    N = Lx * Ly
    U = 2.0
    
    @testset "uniform disorder" begin
        U_params, Ux, Uy = make_U_params(U, Lx, Ly, "uniform")
        @test length(U_params) == N
        @test all(U_params .≈ U)
        @test all(Ux .== 0.0)  # s-wave default
        @test all(Uy .== 0.0)
    end
    
    @testset "rand disorder" begin
        U_params, _, _ = make_U_params(U, Lx, Ly, "rand")
        @test length(U_params) == N
        # Mean should be approximately U after assert_mean
        @test isapprox(mean(U_params), U, atol=1e-10)
    end
    
    @testset "puddles disorder" begin
        U_params, _, _ = make_U_params(U, Lx, Ly, "puddles"; 
                                       U_puddle_params=(0.0, 10.0, 0.7, 0.7),
                                       impurity=0.5, seed=123)
        @test length(U_params) == N
    end
    
    @testset "gaussian disorder" begin
        U_params, _, _ = make_U_params(U, Lx, Ly, "gaussian"; 
                                       impurity=0.3, seed=123)
        @test length(U_params) == N
        # Mean should be approximately U after assert_mean
        @test isapprox(mean(U_params), U, atol=1e-10)
    end
    
    @testset "on-off-puddles disorder" begin
        U_params, _, _ = make_U_params(U, Lx, Ly, "on-off-puddles"; 
                                       U_puddle_params=(0.0, 10.0, 2.0, 0.7),
                                       impurity=0.3, seed=123,
                                       assert_mean=false)  # Don't rescale
        @test length(U_params) == N
        # All values should be >= 0 (after assert_positive)
        @test all(U_params .>= 0.0)
    end

    @testset "d-wave pairing symmetry" begin
        U_params, Ux, Uy = make_U_params(U, Lx, Ly, "uniform"; 
                                         pairing_symmetry="d-wave")
        @test all(U_params .== 0.0)  # d-wave has no s-wave component
        @test length(Ux) == N
        @test length(Uy) == N
    end
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Phase 2 Tests Complete!")
println("=" ^ 60)
println("\nIf all tests passed, Phase 2 implementation is correct.")
println("You can now proceed to Phase 3: Hamiltonian")

end  # end of run_phase2_tests function

# Run tests using invokelatest to avoid world age issues
Base.invokelatest(run_phase2_tests)
