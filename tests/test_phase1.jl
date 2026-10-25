# Test file for Phase 1: Core Infrastructure
# Tests indexing functions, neighbor matrices, and SimData struct changes

println("=" ^ 60)
println("Testing Phase 1: Core Infrastructure")
println("=" ^ 60)

# Include the main file
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

using Test

# ============================================================================
# Test 1: Indexing Functions
# ============================================================================
println("\n--- Test 1: Indexing Functions ---")

@testset "Indexing Functions" begin
    
    # Test add_xy_ix (periodic boundary wrapping with mod1)
    @testset "add_xy_ix (periodic wrapping)" begin
        @test add_xy_ix(1, 0, 6) == 1
        @test add_xy_ix(6, 1, 6) == 1  # wrap around
        @test add_xy_ix(1, -1, 6) == 6  # wrap around negative
        @test add_xy_ix(3, 2, 6) == 5
        @test add_xy_ix(5, 3, 6) == 2  # 5+3=8, mod1(8,6)=2
        
        # Different L values
        @test add_xy_ix(4, 1, 4) == 1
        @test add_xy_ix(1, -1, 4) == 4
    end
    
    # Test idx function (x, y -> linear index)
    @testset "idx (x,y -> linear)" begin
        # For Lx = 6
        @test idx(1, 1, 6) == 1
        @test idx(6, 1, 6) == 6
        @test idx(1, 2, 6) == 7
        @test idx(6, 2, 6) == 12
        @test idx(3, 4, 6) == 21  # (4-1)*6 + 3 = 21
        
        # For Lx = 4 (rectangular case)
        @test idx(1, 1, 4) == 1
        @test idx(4, 1, 4) == 4
        @test idx(1, 2, 4) == 5
        @test idx(2, 3, 4) == 10  # (3-1)*4 + 2 = 10
    end
    
    # Test get_x and get_y fallback versions (using Lx)
    @testset "get_x, get_y fallback (Lx only)" begin
        # For Lx = 6
        @test get_x(1, 6) == 1
        @test get_x(6, 6) == 6
        @test get_x(7, 6) == 1
        @test get_x(12, 6) == 6
        
        @test get_y(1, 6) == 1
        @test get_y(6, 6) == 1
        @test get_y(7, 6) == 2
        @test get_y(12, 6) == 2
        
        # For Lx = 4 (rectangular)
        @test get_x(5, 4) == 1
        @test get_x(8, 4) == 4
        @test get_y(5, 4) == 2
        @test get_y(8, 4) == 2
    end
    
    # Test CartesianIndices and LinearIndices
    @testset "CartesianIndices / LinearIndices" begin
        # Square case: 6x6
        CI_6x6 = make_CartesianIndices(6, 6)
        LI_6x6 = make_LinearIndices(6, 6)
        
        @test get_x(1, CI_6x6) == 1
        @test get_y(1, CI_6x6) == 1
        @test get_x(7, CI_6x6) == 1
        @test get_y(7, CI_6x6) == 2
        @test get_xy(7, CI_6x6) == (1, 2)
        @test get_ix(1, 2, LI_6x6) == 7
        
        # Rectangular case: 6x4
        CI_6x4 = make_CartesianIndices(6, 4)
        LI_6x4 = make_LinearIndices(6, 4)
        
        @test get_x(1, CI_6x4) == 1
        @test get_y(1, CI_6x4) == 1
        @test get_x(7, CI_6x4) == 1
        @test get_y(7, CI_6x4) == 2
        @test get_xy(7, CI_6x4) == (1, 2)
        @test get_ix(1, 2, LI_6x4) == 7
        
        # Verify consistency: idx should give same result as LinearIndices
        for ix in 1:24
            x_coord = get_x(ix, CI_6x4)
            y_coord = get_y(ix, CI_6x4)
            @test idx(x_coord, y_coord, 6) == ix
            @test get_ix(x_coord, y_coord, LI_6x4) == ix
        end
        
        # Rectangular case: 4x6 (opposite)
        CI_4x6 = make_CartesianIndices(4, 6)
        LI_4x6 = make_LinearIndices(4, 6)
        
        @test get_x(5, CI_4x6) == 1
        @test get_y(5, CI_4x6) == 2
        @test get_ix(1, 2, LI_4x6) == 5
    end
    
    # Consistency check: fallback vs CartesianIndices should match for square
    @testset "Fallback vs CartesianIndices consistency" begin
        L = 8
        CI = make_CartesianIndices(L, L)
        for ix in 1:L^2
            @test get_x(ix, L) == get_x(ix, CI)
            @test get_y(ix, L) == get_y(ix, CI)
        end
    end
end

# ============================================================================
# Test 2: Neighbor Matrix Functions
# ============================================================================
println("\n--- Test 2: Neighbor Matrix Functions ---")

@testset "Neighbor Matrix Functions" begin
    
    # Test basic neighbor_matrix for square lattice
    @testset "neighbor_matrix square (L=4)" begin
        A = neighbor_matrix(4, 4)
        N = 16
        @test size(A) == (N, N)
        
        # Each site should have exactly 4 neighbors
        for i in 1:N
            @test sum(A[i, :]) == 4
        end
        
        # Check specific neighbors for corner site (1,1) = ix=1
        # Neighbors: (2,1)=2, (4,1)=4 (wrap), (1,2)=5, (1,4)=13 (wrap)
        @test A[1, 2] == 1  # right neighbor
        @test A[1, 4] == 1  # left neighbor (wrapped)
        @test A[1, 5] == 1  # up neighbor
        @test A[1, 13] == 1 # down neighbor (wrapped)
    end
    
    # Test neighbor_matrix for rectangular lattice
    @testset "neighbor_matrix rectangular (Lx=6, Ly=4)" begin
        A = neighbor_matrix(6, 4)
        N = 24
        @test size(A) == (N, N)
        
        # Each site should have exactly 4 neighbors
        for i in 1:N
            @test sum(A[i, :]) == 4
        end
        
        # Check wrapping in x-direction: site (6,1)=6 should wrap to (1,1)=1
        @test A[6, 1] == 1  # right neighbor wraps
        @test A[6, 5] == 1  # left neighbor is (5,1)=5
        
        # Check wrapping in y-direction: site (1,4)=19 should wrap to (1,1)=1
        # For Lx=6: (1,4) = (4-1)*6 + 1 = 19
        @test A[19, 1] == 1  # up neighbor wraps
    end
    
    # Test legacy wrapper
    @testset "Legacy wrapper neighbor_matrix(L)" begin
        A_legacy = neighbor_matrix(4)
        A_new = neighbor_matrix(4, 4)
        @test A_legacy == A_new
    end
    
    # Test neighbor_matrix_x
    @testset "neighbor_matrix_x" begin
        # Square case
        Ax = neighbor_matrix_x(4, 4, "x")
        Ay = neighbor_matrix_x(4, 4, "y")
        
        @test size(Ax) == (16, 16)
        @test size(Ay) == (16, 16)
        
        # Each site should have exactly 1 neighbor in x or y direction
        for i in 1:16
            @test sum(abs.(Ax[i, :])) == 1
            @test sum(abs.(Ay[i, :])) == 1
        end
        
        # Rectangular case
        Ax_rect = neighbor_matrix_x(6, 4, "x")
        Ay_rect = neighbor_matrix_x(6, 4, "y")
        
        @test size(Ax_rect) == (24, 24)
        @test size(Ay_rect) == (24, 24)
    end
    
    # Test neighbor_matrix_K_vector
    @testset "neighbor_matrix_K_vector" begin
        K_vectors = [[0.0, 0.0], [π/2, 0.0], [0.0, π/2]]
        
        # Square case
        NN_sq = neighbor_matrix_K_vector(4, 4, K_vectors)
        @test length(NN_sq) == 3
        @test all(size(M) == (16, 16) for M in NN_sq)
        
        # Rectangular case
        NN_rect = neighbor_matrix_K_vector(6, 4, K_vectors)
        @test length(NN_rect) == 3
        @test all(size(M) == (24, 24) for M in NN_rect)
        
        # K=0 should give real values (no phase)
        @test all(imag.(NN_sq[1]) .≈ 0.0)
    end
end

# ============================================================================
# Test 3: K-space Lists
# ============================================================================
println("\n--- Test 3: K-space Lists ---")

@testset "K-space Lists" begin
    
    @testset "get_kx_list / get_ky_list" begin
        kx = get_kx_list(6)
        ky = get_ky_list(4)
        
        @test length(kx) == 6
        @test length(ky) == 4
        
        # Check first and last values
        @test kx[1] ≈ -π
        @test kx[end] ≈ π - 2π/6
        @test ky[1] ≈ -π
        @test ky[end] ≈ π - 2π/4
        
        # Legacy compatibility
        @test get_ki_list(6) == get_kx_list(6)
    end
end

# ============================================================================
# Test 4: SimData Structure
# ============================================================================
println("\n--- Test 4: SimData Structure ---")

@testset "SimData Structure" begin
    
    @testset "SimData has Lx, Ly, N fields" begin
        # Check that the struct has the right field names
        @test :Lx in fieldnames(SimData)
        @test :Ly in fieldnames(SimData)
        @test :N in fieldnames(SimData)
        @test !(:L in fieldnames(SimData))  # Old L should not exist
    end
end

# ============================================================================
# Test 5: Rectangular vs Square Consistency
# ============================================================================
println("\n--- Test 5: Rectangular vs Square Consistency ---")

@testset "Rectangular vs Square Consistency" begin
    
    @testset "Square lattice: both methods should match" begin
        L = 6
        
        # Using Lx=Ly=L
        A_rect = neighbor_matrix(L, L)
        
        # Using legacy L
        A_sq = neighbor_matrix(L)
        
        @test A_rect == A_sq
    end
    
    @testset "Index roundtrip for rectangular" begin
        Lx, Ly = 6, 4
        N = Lx * Ly
        CI = make_CartesianIndices(Lx, Ly)
        LI = make_LinearIndices(Lx, Ly)
        
        for ix in 1:N
            x_c, y_c = get_xy(ix, CI)
            ix_back = get_ix(x_c, y_c, LI)
            @test ix == ix_back
            
            # Also test with fallback
            x_f = get_x(ix, Lx)
            y_f = get_y(ix, Lx)
            ix_back_f = idx(x_f, y_f, Lx)
            @test ix == ix_back_f
            
            # Both methods should give same coordinates
            @test x_c == x_f
            @test y_c == y_f
        end
    end
    
    @testset "Periodic wrapping for rectangular" begin
        Lx, Ly = 6, 4
        
        # x-direction wrapping uses Lx
        @test add_xy_ix(Lx, 1, Lx) == 1
        @test add_xy_ix(1, -1, Lx) == Lx
        
        # y-direction wrapping uses Ly
        @test add_xy_ix(Ly, 1, Ly) == 1
        @test add_xy_ix(1, -1, Ly) == Ly
    end
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Phase 1 Tests Complete!")
println("=" ^ 60)
println("\nIf all tests passed, Phase 1 implementation is correct.")
println("You can now proceed to Phase 2: Data Structures")
