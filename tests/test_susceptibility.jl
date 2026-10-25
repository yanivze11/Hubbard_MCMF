# Test for susceptibility_q_vec
using Test
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))


# Real simulation test for susceptibility_q_vec
# @testset "susceptibility_q_vec real simulation" begin
    # Minimal simulation parameters for a quick test
    Lx = 4
    Ly = 4
    L_SC = 1
    U = 4.0
    t = 1.0
    V = 0.0
    seed = 1234
    mu = -0.65
    beta = 20.0
    B = 0.0
    num_measure = 10
    num_thermal = 0
    measuring_interval = 1
    w_list = [-3.0:0.1:3.0;]
    initial_guess = "uniform"
    U_disorder = "uniform"
    diag_method = "K0"
    pairing_symmetry = "s-wave"
    w_conductivity_range = [0.0:0.01:1.0;]
    U_impurity = 0.0
# @profview 
    data = running_simulation_once(
        Lx, Ly, L_SC, U, t, V, seed, mu, beta, B, num_measure, num_thermal, measuring_interval, w_list,
        initial_guess, U_disorder, diag_method, pairing_symmetry, w_conductivity_range, U_impurity;
        directory="", num_cores=1, savedata=false, saving_interval=100, dump_time=Dates.Hour(1),
        saveplots=false, date_folder=true, numbered=true, seed_folder=false, U_puddle_params=[0.0, 6.0, 0.7, 0.7],
         U_smooth=(false, 0.8), loop_random=false, boundary_conditions="PBC", gauge_choice="LandauX"
    );

    # qs_y = (2*pi/Ly) * (0:(div(Ly,2)))
    # qs_x = (2*pi/Lx) * (0:(div(Lx,2)))
    # q_list = [[0.0, q] for q in qs_y]  # qx=0, qy varying
    # append!(q_list, [[q, 0.0] for q in qs_x])  # qx varying, qy=0
    # chi_spin = susceptibility_q_vec(data, q_list; type=:spin)
    # chi_charge = susceptibility_q_vec(data, q_list; type=:charge)
    # @test length(chi_spin) == length(q_list)
    # @test length(chi_charge) == length(q_list)
    # @test isfinite(chi_spin[1])
    # @test isfinite(chi_charge[1])
# end
