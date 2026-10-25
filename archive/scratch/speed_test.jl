include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))

mutable struct SpeedResults
    prepare_time::Float64
    step_times::Array{Float64,1}
    measurement_times::Array{Float64,1}


end


function speed_test(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, mu::Float64, beta::Float64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, diag_method::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
		 							directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=500, dump_time=Dates.Hour(1), saveplots=false, date_folder=true, numbered=true, seed_folder=false, U_puddle_params=(0.0, 6.0, 0.7, 0.7), TClustersize=4, U_smooth=(false, 0.8))
    # this function runs a short simulation to test the speed of the simulation for given parameters
    num_measure = 10
    num_thermal = 10
    #prepare the SimData and Lattice_Data objects
    now_time = Dates.now()
    #make SimData file
    sim_data = SimData(L, L_SC, U, t, V, seed, mu, beta, num_measure, num_thermal, 1, saving_interval, w_list, initial_guess, U_disorder, diag_method, directory, num_cores, Threads.nthreads(), Dates.now(), w_conductivity_range, U_puddle_params, U_impurity, TClustersize, U_smooth)
    # make_Lattice_data
    Lattice_data = Lattice_Data(sim_data)
    prepare_time = Dates.value(Dates.now() - now_time) / 1e9 # in seconds

    # diagonalization test
    mc_step_time = []
    for _ in 1:10
        now_time = Dates.now()
        MC_step!(Lattice_data; method=diag_method)
        diag_time = Dates.value(Dates.now() - now_time) / 1e9 # in seconds
        push!(mc_step_time, diag_time)
    end

    # measurement test
    measurements_time = []
    for _ in 1:10
        now_time = Dates.now()
        make_measurement_K!(Lattice_data, 1)
        meas_time = Dates.value(Dates.now() - now_time) / 1e9 # in seconds
        push!(measurements_time, meas_time)
    end

    speed_results = SpeedResults(prepare_time, mc_step_time, measurements_time)
    println("Speed Test Results for L=$L, V=$V, beta=$beta, diag_method=$diag_method, U_disorder=$U_disorder, TCA_size=$TClustersize")
    println("Preparation time: $(speed_results.prepare_time) seconds")
    println("Average MC step time: $(mean(speed_results.step_times)) seconds")
    println("Average measurement time: $(mean(speed_results.measurement_times)) seconds")
    return speed_results

end