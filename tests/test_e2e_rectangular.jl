# End-to-End Test: Full rectangular lattice simulation (4x5) and odd L support (5x7)
# This test runs a complete Monte Carlo simulation on a rectangular lattice
# and verifies all components work together correctly, including odd dimensions.

using Test
using Dates
using Statistics

println("=" ^ 60)
println("End-to-End Rectangular Lattice Simulation Test")
println("Systems: 4x5 (even x odd), 5x7 (odd x odd)")
println("=" ^ 60)

println("\nLoading MC_Hubburd.jl...")
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
println("Module loaded successfully!")

@testset "End-to-End Rectangular Simulation (4x5)" begin
    
    @testset "Test 1: SimData keyword constructor" begin
        println("\n--- Test 1: SimData keyword constructor ---")
        
        # Test the new keyword constructor
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, num_measure=10, num_thermal=5)
        
        @test sim_data.Lx == 4
        @test sim_data.Ly == 5
        @test sim_data.N == 20
        @test sim_data.U == 4.0
        @test sim_data.beta == 5.0
        
        # Test default values
        @test sim_data.SC_L == 1
        @test sim_data.t == 1.0
        @test sim_data.pairing_symmetry == "s-wave"
        
        println("  ✓ SimData keyword constructor works correctly")
    end
    
    @testset "Test 2: LatticeData initialization (4x5)" begin
        println("\n--- Test 2: LatticeData initialization ---")
        
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, 
                          num_measure=10, num_thermal=5,
                          diag_method="K0", pairing_symmetry="s-wave")
        
        lattice_data = Lattice_Data(sim_data)
        
        # Check dimensions
        @test lattice_data.sim_data.Lx == 4
        @test lattice_data.sim_data.Ly == 5
        @test lattice_data.sim_data.N == 20
        
        # Check data structure sizes
        @test length(lattice_data.current_state.Delta_s) == 20
        @test size(lattice_data.const_struct.NN_mat[1]) == (20, 20)  # N x N
        
        println("  ✓ LatticeData initialized correctly for 4x5 lattice")
    end
    
    @testset "Test 3: MC steps on rectangular lattice" begin
        println("\n--- Test 3: Monte Carlo steps ---")
        
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, 
                          num_measure=10, num_thermal=5,
                          diag_method="K0", pairing_symmetry="s-wave",
                          seed=42)
        
        lattice_data = Lattice_Data(sim_data)
        
        initial_S = lattice_data.current_state.S
        println("  Initial action S = $initial_S")
        
        # Run several MC steps
        for i in 1:10
            MC_stepV2!(lattice_data)
        end
        
        final_S = lattice_data.current_state.S
        println("  Final action S = $final_S after 10 steps")
        
        @test lattice_data.current_state.step_count == 10
        @test isfinite(final_S)
        
        println("  ✓ MC steps completed successfully")
    end
    
    @testset "Test 4: Measurements on rectangular lattice" begin
        println("\n--- Test 4: Measurements ---")
        
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, 
                          num_measure=10, num_thermal=5,
                          diag_method="K0", pairing_symmetry="s-wave",
                          w_list=collect(-2.0:0.5:2.0),
                          saving_interval=10)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Thermalization
        println("  Thermalizing (5 steps)...")
        for _ in 1:5
            MC_stepV2!(lattice_data)
        end
        
        # Measurement
        println("  Making measurement...")
        make_measurement_K!(lattice_data, 1)
        
        # Check measurements
        n_avg = lattice_data.measurement_data.n[1]
        println("  Average occupation <n> = $n_avg")
        @test 0 <= n_avg <= 2  # Physical bounds
        
        Kx = lattice_data.measurement_data.Kx[1]
        println("  Kinetic energy Kx = $Kx")
        @test isfinite(Kx)
        
        # Check BZ size for rectangular
        @test length(lattice_data.measurement_data.BZ) == 4 * 5
        
        println("  ✓ Measurements completed successfully")
    end
    
    @testset "Test 5: D-wave pairing on rectangular" begin
        println("\n--- Test 5: D-wave pairing symmetry ---")
        
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, 
                          num_measure=10, num_thermal=5,
                          diag_method="K0", pairing_symmetry="d-wave",
                          saving_interval=10)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Run a few steps
        for _ in 1:5
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        @test isfinite(lattice_data.measurement_data.n[1])
        @test isfinite(lattice_data.measurement_data.Kx[1])
        
        # Check phase correlation array sizes
        # ls_list_reduced_size = div(L_min,2) + div(L_max,2) + 1 = 2 + 2 + 1 = 5
        expected_ls_size = div(min(4,5), 2) + div(max(4,5), 2) + 1
        @test length(lattice_data.measurement_data.Phase_Corr[1]) == expected_ls_size
        
        println("  ✓ D-wave pairing on rectangular lattice works")
    end
    
    @testset "Test 6: TCluster method on rectangular" begin
        println("\n--- Test 6: TCluster diagonalization (3x2 cluster) ---")
        
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, 
                          num_measure=5, num_thermal=3,
                          diag_method="K0-TCA", pairing_symmetry="s-wave",
                          TCluster_size_x=3, TCluster_size_y=2,
                          saving_interval=5)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Check TCluster dimensions
        @test sim_data.TCluster_size_x == 3
        @test sim_data.TCluster_size_y == 2
        
        # Run a few steps with TCluster (use "K0-TCA" method)
        for _ in 1:3
            MC_stepV2!(lattice_data; method="K0-TCA")
        end
        make_measurement_K!(lattice_data, 1)
        
        @test isfinite(lattice_data.measurement_data.n[1])
        
        println("  ✓ TCluster method works on rectangular lattice")
    end
    
    @testset "Test 7: Full mini-simulation (4x5)" begin
        println("\n--- Test 7: Full mini Monte Carlo simulation ---")
        println("  Parameters: 4x5 lattice, U=4.0, beta=5.0")
        println("  5 thermal + 10 measurement steps")
        
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, 
                          num_measure=10, num_thermal=5,
                          diag_method="K0", pairing_symmetry="s-wave",
                          w_list=collect(-2.0:0.5:2.0),
                          saving_interval=10,
                          seed=12345)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Thermalization phase
        println("  Thermalizing...")
        for i in 1:5
            MC_stepV2!(lattice_data)
        end
        println("    Action after thermalization: $(lattice_data.current_state.S)")
        
        # Measurement phase
        println("  Measuring...")
        n_values = Float64[]
        S_values = Float64[]
        
        for i in 1:10
            MC_stepV2!(lattice_data)
            make_measurement_K!(lattice_data, i)
            push!(n_values, lattice_data.measurement_data.n[i])
            push!(S_values, lattice_data.measurement_data.Ss[i])
        end
        
        # Statistics
        n_mean = mean(n_values)
        n_std = std(n_values)
        S_mean = mean(S_values)
        
        println("\n  Results:")
        println("    <n> = $(round(n_mean, digits=4)) ± $(round(n_std, digits=4))")
        println("    <S> = $(round(S_mean, digits=4))")
        println("    Steps completed: $(lattice_data.current_state.step_count)")
        
        @test 0 <= n_mean <= 2
        @test isfinite(S_mean)
        @test lattice_data.current_state.step_count == 15
        
        println("  ✓ Full mini-simulation completed successfully!")
    end
    
    @testset "Test 8: File naming for rectangular" begin
        println("\n--- Test 8: File naming and logging ---")
        
        sim_data = SimData(Lx=4, Ly=5, U=4.0, beta=5.0, 
                          num_measure=10, num_thermal=5,
                          diag_method="K0", pairing_symmetry="s-wave")
        
        # Test file name generation
        filename = choose_file_name(".", sim_data; date_folder=false, numbered=false)
        println("  Generated filename: $filename")
        
        @test occursin("4x5", filename)
        @test occursin("beta=5.0", filename)
        
        println("  ✓ File naming works correctly for rectangular lattice")
    end
    
    @testset "Test 9: running_simulation_once (no save)" begin
        println("\n--- Test 9: running_simulation_once function ---")
        println("  Running short simulation without saving...")
        
        # Use a temporary directory
        temp_dir = mktempdir()
        
        try
            data = running_simulation_once(
                4, 5,           # Lx, Ly
                1,              # SC_L
                4.0,            # U
                1.0,            # t
                0.0,            # V
                42,             # seed
                0.0,            # mu
                5.0,            # beta
                0.0,            # B
                5,              # num_measure
                3,              # num_thermal
                1,              # measuring_interval
                collect(-2.0:0.5:2.0),  # w_list
                "uniform",      # initial_guess
                "uniform",      # U_disorder
                "K0",           # diag_method
                "s-wave",       # pairing_symmetry
                collect(-1.0:0.5:1.0),  # w_conductivity_range
                1.0;            # U_impurity
                directory=temp_dir,
                savedata=false,
                date_folder=false,
                numbered=false
            )
            
            @test data.sim_data.Lx == 4
            @test data.sim_data.Ly == 5
            @test data.sim_data.N == 20
            @test data.current_state.step_count > 0
            
            n_mean = mean(data.measurement_data.n[1:5])
            println("  Simulation completed, <n> = $(round(n_mean, digits=4))")
            
            println("  ✓ running_simulation_once works for rectangular lattice")
        finally
            rm(temp_dir; recursive=true, force=true)
        end
    end
    
    @testset "Test 10: Odd L k-point list generation" begin
        println("\n--- Test 10: k-point lists for odd L ---")
        
        # Test get_kx_list and get_ky_list with odd values
        for L in [3, 5, 7, 9]
            kx_list = get_kx_list(L)
            @test length(kx_list) == L
            println("  L=$L: kx_list has $(length(kx_list)) points ✓")
        end
        
        # Verify k-points are in correct range [-π, π)
        kx_5 = get_kx_list(5)
        @test all(-π <= k < π for k in kx_5)
        println("  All k-points in [-π, π) ✓")
        
        println("  ✓ k-point list generation works for odd L")
    end
    
    @testset "Test 11: Full simulation on odd system (5x7)" begin
        println("\n--- Test 11: Full mini simulation on 5x7 odd lattice ---")
        println("  Parameters: 5x7 lattice (both odd), U=4.0, beta=5.0")
        
        sim_data = SimData(Lx=5, Ly=7, U=4.0, beta=5.0, 
                          num_measure=10, num_thermal=5,
                          diag_method="K0", pairing_symmetry="s-wave",
                          w_list=collect(-2.0:0.5:2.0),
                          saving_interval=10,
                          seed=54321)
        
        @test sim_data.N == 35  # 5 * 7
        
        lattice_data = Lattice_Data(sim_data)
        
        # Verify k-list sizes
        @test length(lattice_data.const_struct.kx_list) == 5
        @test length(lattice_data.const_struct.ky_list) == 7
        
        # Thermalization phase
        println("  Thermalizing...")
        for i in 1:5
            MC_stepV2!(lattice_data)
        end
        
        # Measurement phase
        println("  Measuring...")
        n_values = Float64[]
        
        for i in 1:10
            MC_stepV2!(lattice_data)
            make_measurement_K!(lattice_data, i)
            push!(n_values, lattice_data.measurement_data.n[i])
        end
        
        n_mean = mean(n_values)
        println("  <n> = $(round(n_mean, digits=4))")
        
        @test 0 <= n_mean <= 2
        @test lattice_data.current_state.step_count == 15
        
        # Check array sizes with odd dimensions
        @test length(lattice_data.measurement_data.BZ) == 5 * 7
        @test length(lattice_data.measurement_data.Lxx_qy[1]) == div(7, 2)  # 3
        @test length(lattice_data.measurement_data.Lxx_qx[1]) == div(5, 2)  # 2
        
        println("  ✓ Full simulation on 5x7 odd lattice completed!")
    end
    
    @testset "Test 12: D-wave on odd system (5x5)" begin
        println("\n--- Test 12: D-wave pairing on 5x5 odd square ---")
        
        sim_data = SimData(Lx=5, Ly=5, U=4.0, beta=5.0, 
                          num_measure=5, num_thermal=3,
                          diag_method="K0", pairing_symmetry="d-wave",
                          saving_interval=5)
        
        lattice_data = Lattice_Data(sim_data)
        
        # Run steps
        for _ in 1:5
            MC_stepV2!(lattice_data)
        end
        make_measurement_K!(lattice_data, 1)
        
        # Check phase correlation array sizes for odd L
        # ls_list_reduced_size = L_min/2 + L_max/2 + 1 = 5/2 + 5/2 + 1 = 2 + 2 + 1 = 5
        expected_ls_size = div(5, 2) + div(5, 2) + 1
        @test length(lattice_data.measurement_data.Phase_Corr[1]) == expected_ls_size
        @test length(lattice_data.measurement_data.Phase_Corr_dwave_x[1]) == expected_ls_size
        
        @test isfinite(lattice_data.measurement_data.n[1])
        
        println("  ✓ D-wave on odd square lattice works")
    end
    
    @testset "Test 13: TCluster on odd system (5x7 with 3x3 cluster)" begin
        println("\n--- Test 13: TCluster on 5x7 odd lattice ---")
        
        sim_data = SimData(Lx=5, Ly=7, U=4.0, beta=5.0, 
                          num_measure=3, num_thermal=2,
                          diag_method="K0-TCA", pairing_symmetry="s-wave",
                          TCluster_size_x=3, TCluster_size_y=3,
                          saving_interval=3)
        
        lattice_data = Lattice_Data(sim_data)
        
        # TCluster assignments should exist
        @test length(lattice_data.const_struct.TCluster_assignments) == 35  # N = 5*7
        
        # Run a few TCluster steps (use "K0-TCA" method, not "TCluster")
        for _ in 1:2
            MC_stepV2!(lattice_data; method="K0-TCA")
        end
        make_measurement_K!(lattice_data, 1)
        
        @test isfinite(lattice_data.measurement_data.n[1])
        
        println("  ✓ TCluster on odd lattice works")
    end
    
    @testset "Test 14: File naming for odd dimensions" begin
        println("\n--- Test 14: File naming for odd L ---")
        
        sim_data = SimData(Lx=5, Ly=7, U=4.0, beta=5.0)
        
        filename = choose_file_name(".", sim_data; date_folder=false, numbered=false)
        println("  Generated filename: $filename")
        
        @test occursin("5x7", filename)
        
        println("  ✓ File naming works for odd dimensions")
    end
end

println("\n" * "=" ^ 60)
println("🎉 All End-to-End Tests Passed!")
println("The rectangular lattice generalization (including odd L) is fully functional.")
println("=" ^ 60)
