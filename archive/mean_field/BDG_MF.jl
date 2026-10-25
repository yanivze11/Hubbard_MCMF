include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))


begin  # data structures

    mutable struct MFParams
        max_iter::Int64
        tol::Float64
        a::Float64
    end

    mutable struct MFState
        equations::Array{ComplexF64,1}
        max_eq_vec::Array{Float64,1}
        iter::Int64
        converged::Bool
    end

    mutable struct MFSim
        lattice_data::LatticeData
        mf_params::MFParams
        mf_state::MFState
    end


    function MF_Sim(lattice_data::LatticeData, mf_params::MFParams)
        # The # of equations depends on the pairing symmetry
        N = lattice_data.sim_data.N
        pairing_symmetry = lattice_data.sim_data.pairing_symmetry
        num_equations = 0
        if pairing_symmetry == "s-wave"
            num_equations = N
        elseif pairing_symmetry in ["d-wave", "mayr-d-wave", "d-p-wave"]
            num_equations = 2N
        else
            throw(ErrorException("Pairing symmetry $pairing_symmetry not yet implemented in MF_Sim"))
        end
        # initialize the MF simulation state
        mf_state = MFState(zeros(ComplexF64, num_equations), zeros(Float64, mf_params.max_iter), 0, false)

        return MFSim(lattice_data, mf_params, mf_state)
    end
end



begin
    function running_MF(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, mu::Float64, beta::Float64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
		 							directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=1, dump_time=Dates.Hour(1), saveplots=false, date_folder=true, numbered=true, seed_folder=false, U_puddle_params=[0.0, 6.0, 0.7, 0.7], U_smooth=(true, 0.8),
                                    max_iter::Int64=5000, tol::Float64=1e-8, a::Float64=0.6)
		start_time = Dates.now()

        sim_data = SimData(
            Lx=L,
            Ly=L,
            SC_L=L_SC,
            U=U,
            t=t,
            V=V,
            seed=seed,
            mu=mu,
            beta=beta,
            num_measure=1,
            num_thermal=0,
            measuring_interval=1,
            saving_interval=saving_interval,
            w_list=w_list,
            guess=initial_guess,
            U_disorder=U_disorder,
            diag_method="K0",
            pairing_symmetry=pairing_symmetry,
            directory=directory,
            num_cores=num_cores,
            num_threads=Threads.nthreads(),
            start_time=start_time,
            w_conductivity_range=w_conductivity_range,
            U_disorder_params=U_puddle_params,
            U_impurity=U_impurity,
            TCluster_size_x=2,
            TCluster_size_y=2,
            smooth_U=U_smooth,
            loop_random=false
        )
        mf_params = MFParams(max_iter, tol, a)
		out_Lattice_data = nothing
        dir_path = choose_path(directory, sim_data; date_folder=date_folder, numbered=numbered, seed_folder=seed_folder)
		sim_info_path = joinpath(directory, "sim.jld2")
		log_filename = joinpath(dir_path, "simulation_log.txt")
		checkpoint_path = joinpath(dir_path, "checkpoint.jld2")

		if !isfile(checkpoint_path)
			if savedata
				# same start time in the sim.jld2 file
				if isfile(sim_info_path)
					sim_info = JLD2.load(sim_info_path)
					if !haskey(sim_info, "simulations")
						sim_info["sim_num"] = tryparse(Int, basename(directory))
						sim_info["start_time"] = start_time
						sim_info["info_path"] = sim_info_path
						sim_info["simulations"] = Dict{Any, Any}()
					end
					seed_dict = Dict{String, Any}()
					seed_dict["status"] = "running"
					seed_dict["start_time"] = start_time
					seed_dict["checkpoint_path"] = checkpoint_path
					seed_dict["dir_path"] = dir_path
					seed_dict["info_path"] = sim_info_path
					sim_info["simulations"][seed] = seed_dict
					JLD2.save(sim_info_path, sim_info)
				end

				# log file
				open(log_filename, "a") do io
					println(io, "----------------------------------------")
					println(io, "Simulation parameters: L=$(L), L_s=$(L_SC), U=$(U), t=$(t), mu=$(mu), beta=$(beta), initial_guess=$(initial_guess), diag_method=MF, num_cores=$(num_cores),  U_disorder=$(U_disorder), $(U_disorder=="puddles" ? "U_puddle_params=$(U_puddle_params)," : ""), U_impurity=$(U_impurity)")
					println(io, "start time: $(start_time)")
				end
			end

			out_Lattice_data = MeanFieldSimulation(sim_data, mf_params, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename);
		else
			# log file
			open(log_filename, "a") do io
				println(io, "continuing simulation with existing file: $(checkpoint_path)")
				println(io, "start time: $(start_time)")
			end

			# Try to load existing data, if fails, start from scratch
			try
				# data = JLD2.load(filename, "data")
                data = load_checkpoint_MF(dir_path; mf_params=mf_params)
				out_Lattice_data = MeanFieldSimulation(data, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename)
			catch e
				open(log_filename, "a") do io
					println(io, "Failed to load existing data: $(e). Starting simulation from scratch.")
				end
				out_Lattice_data = MeanFieldSimulation(sim_data, mf_params, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename)
			end
		end


		if savedata
			endtime = Dates.now()
			# save the end time in the sim.jld2 file
			if isfile(sim_info_path)
				sim_info = JLD2.load(sim_info_path)
				sim_info["simulations"][seed]["end_time"] = endtime
				sim_info["simulations"][seed]["status"] = "finished"
				JLD2.save(sim_info_path, sim_info)
			end

			# log file
			open(log_filename, "a") do io
				println(io, "simulation finished!")
				println(io, "end time: $(endtime)")
				println(io, "Elapsed time: $(endtime - start_time)")
				println(io, "----------------------------------------")
			end
		end

        return out_Lattice_data
    end

    function run_MF_wanted_n(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, mu_range::Array{Float64,1}, wanted_n::Float64, beta::Float64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
		 							directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=1, dump_time=Dates.Hour(1), saveplots=false, date_folder=true, numbered=true, seed_folder=false, U_puddle_params=[0.0, 6.0, 0.7, 0.7], U_smooth=(true, 0.8),
                                    max_iter::Int64=5000, tol::Float64=1e-8, a::Float64=0.6)
        
        n_list = zeros(Float64, length(mu_range))
        n_min_dist = Inf
        closest_sim_data = nothing
        closest_mu = nothing
        closest_n = nothing
        for (i, mu) in enumerate(mu_range)
            println("Trying mu = $mu")
            sim_data = running_MF(L, L_SC, U, t, V, seed, mu, beta, w_list, initial_guess, U_disorder, pairing_symmetry, w_conductivity_range, U_impurity;
                                  directory=directory, num_cores=num_cores, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, saveplots=saveplots, date_folder=date_folder, numbered=numbered, seed_folder=seed_folder, U_puddle_params=U_puddle_params, U_smooth=U_smooth,
                                  max_iter=max_iter, tol=tol, a=a)
            n = sim_data.lattice_data.measurement_data.n[1]
            println("Resulting n = $n")
            n_list[i] = n
            n_dist = abs(n - wanted_n)
            if n_dist < n_min_dist
                n_min_dist = n_dist
                closest_sim_data = sim_data
                closest_mu = mu
                closest_n = n
            end
        end
        println("Closest mu = $closest_mu with n = $(closest_sim_data.lattice_data.measurement_data.n[1]) (distance = $n_min_dist)")

        return closest_sim_data, closest_mu, closest_n

    end

    function run_MF_lion_hunting_n(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, mu_lims::Array{Float64,1}, wanted_n::Float64, n_resolution::Float64, beta::Float64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
		 							directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=1, dump_time=Dates.Hour(1), saveplots=false, date_folder=true, numbered=true, seed_folder=false, U_puddle_params=[0.0, 6.0, 0.7, 0.7], U_smooth=(true, 0.8),
                                    max_iter::Int64=5000, tol::Float64=1e-8, a::Float64=0.6)

        # find n of the two edges
        n_edges = zeros(Float64, 2)
        for (i, mu) in enumerate(mu_lims)
            println("Finding edge n at mu = $mu")
            if mu== 0.0
                n_edges[i] = 1.0
                continue
            end
            sim, mu_found, n = run_MF_wanted_n(L, L_SC, U, t, V, seed, [mu], wanted_n, beta, w_list, initial_guess, U_disorder, pairing_symmetry, w_conductivity_range, U_impurity;
                                        directory=directory, num_cores=num_cores, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, saveplots=saveplots, date_folder=date_folder, numbered=numbered, seed_folder=seed_folder, U_puddle_params=U_puddle_params, U_smooth=U_smooth,
                                        max_iter=max_iter, tol=tol, a=a)
            n_edges[i] = n
            println("Edge mu: $mu with n = $n")
            # check if wanted_n is within the range
            # if not, return the closest edge
            if wanted_n < n && i == 1
                println("Wanted n is below the range. Returning lower edge.")
                return sim, mu_found, n
            elseif wanted_n > n && i == 2
                println("Wanted n is above the range. Returning upper edge.")
                return sim, mu_found, n
            end
        end


        # lion hunting
        n_dist = Inf
        best_sim = nothing
        best_mu = nothing
        best_n = nothing
        mu_low = mu_lims[1]
        mu_high = mu_lims[2]
        while n_dist > n_resolution
            m = (n_edges[2] - n_edges[1])/(mu_high - mu_low) 
            c = n_edges[1] - m * mu_low
            mu_mid = (wanted_n - c) / m
            # mu_mid = (mu_low + mu_high) / 2
            
            println("Trying mu = $mu_mid")
            sim, mu_found, n = run_MF_wanted_n(L, L_SC, U, t, V, seed, [mu_mid], wanted_n, beta, w_list, initial_guess, U_disorder, pairing_symmetry, w_conductivity_range, U_impurity;
                                        directory=directory, num_cores=num_cores, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, saveplots=saveplots, date_folder=date_folder, numbered=numbered, seed_folder=seed_folder, U_puddle_params=U_puddle_params, U_smooth=U_smooth,
                                        max_iter=max_iter, tol=tol, a=a)
            n_dist = abs(n - wanted_n)
            println("Resulting n = $n (distance = $n_dist)")
            if n < wanted_n
                mu_low = mu_mid
            else
                mu_high = mu_mid
            end
            if n_dist < n_resolution
                best_sim = sim
                best_mu = mu_found
                best_n = n
            end
        end
        println("Found mu = $best_mu with n = $best_n (distance = $(abs(best_n - wanted_n)))")
        
        return best_sim, best_mu, best_n
        
    end

end

begin # main simulation function
    function MeanFieldSimulation(sim_data::SimData, mf_params::MFParams, dir_path::String; saveplots::Bool=false, savedata::Bool=false, saving_interval::Int=10000, dump_time=Dates.Hour(1), sim_info_path=nothing, log_filename=nothing)
        lattice_data = Lattice_Data(sim_data)
        sim = MF_Sim(lattice_data, mf_params)
        sim = MeanFieldSimulation(sim, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename)
        return sim
    end


    function MeanFieldSimulation(sim::MFSim, dir_path::String; saveplots=false, savedata=false, saving_interval=10000, dump_time=Dates.Hour(1), sim_info_path=nothing, log_filename=nothing)
        # reach the MF solution
        sim = solve_MF!(sim)
        # compute observables
        sim.lattice_data.measurement_data = make_measurement_K!(sim.lattice_data, 1)
        # save data
        if savedata
            save_simulation_data_MF(sim, dir_path; sim_info_path=sim_info_path, log_filename=log_filename)
        end
        # return the lattice data with measurements
        return sim
    end
end


    function solve_MF!(sim::MFSim)
        max_iter = sim.mf_params.max_iter
        # converged = sim.mf_state.converged
        
        while sim.mf_state.iter < max_iter
            sim.mf_state.iter += 1
            # perform one iteration of the self-consistent loop
            sim = mf_iteration!(sim)
            # check for convergence
            if sim.mf_state.converged
                println("Mean-field converged in $(sim.mf_state.iter) iterations.")
                println("Max equation value: $(sim.mf_state.max_eq_vec[sim.mf_state.iter])")
                println("Final Delta mean: ")
                println("|<Δs>| = ", round((abs(mean(sim.lattice_data.current_state.Delta_s))), digits=3), " , <|Δs|> = ", round((mean(abs.(sim.lattice_data.current_state.Delta_s))), digits=3))
                println("|<Δx>| = ", (abs(mean(sim.lattice_data.current_state.Delta_dx))), " , <|Δx|> = ", (mean(abs.(sim.lattice_data.current_state.Delta_dx))))
                println("|<Δy>| = ", (abs(mean(sim.lattice_data.current_state.Delta_dy))), " , <|Δy|> = ", (mean(abs.(sim.lattice_data.current_state.Delta_dy))))
                sim.mf_state.max_eq_vec = sim.mf_state.max_eq_vec[1:sim.mf_state.iter]
                break
            end
        end
        if !sim.mf_state.converged
            println("Mean-field did not converge in $(max_iter) iterations. Max equation value: $(sim.mf_state.max_eq_vec[sim.mf_state.iter])")
        end

        return sim
    end

    function mf_iteration!(sim::MFSim)
        # one iteration of the self-consistent loop
        # extract parameters
        lattice_data = sim.lattice_data

        # make the new Hamiltonian and diagonalize
        lattice_data.temporary_data.K = make_H_matrix_K(lattice_data, lattice_data.current_state.Delta_s, lattice_data.current_state.Delta_dx, lattice_data.current_state.Delta_dy; H=lattice_data.temporary_data.K, NN_mat=lattice_data.const_struct.NN_mat[1], V_matrix=lattice_data.const_struct.V_matrix, N=lattice_data.sim_data.N)
		Energies_current, U_matrix_current = FastLapackInterface.syevr!(lattice_data.temporary_data.ws, 'V', 'A', 'U', lattice_data.temporary_data.K, 1.0, 1.0, 1, 1, 10e-10; resize=false)
        lattice_data.current_state.Energies .= Energies_current
        lattice_data.current_state.U_matrix .= U_matrix_current

        # check the self-consistent equations
        sim.mf_state.equations = self_consistent_equations!(sim)
        max_eq = maximum(abs.(sim.mf_state.equations))
        # println("Iteration $iter: max(|equations|) = $norm_eq ; |<Δx>| = $(abs(mean(Delta_Dx))) ; |<Δy>| = $(abs(mean(Delta_Dy))); |<Δs>| = $(abs(mean(Delta_S)))")
        sim.mf_state.max_eq_vec[sim.mf_state.iter] = max_eq
        if max_eq < sim.mf_params.tol
            sim.mf_state.converged = true
            return sim
        end

        # Update Delta (simple fixed-point iteration)
        sim = update_delta_MF!(sim)

        return sim
    end

    function update_delta_MF!(sim::MFSim)
        N = sim.lattice_data.sim_data.N
        lattice_data = sim.lattice_data
        a = sim.mf_params.a
        Delta_s = lattice_data.current_state.Delta_s
        Delta_x = lattice_data.current_state.Delta_dx
        Delta_y = lattice_data.current_state.Delta_dy
        pairing_symmetry = lattice_data.sim_data.pairing_symmetry
        if pairing_symmetry == "s-wave"
            for i in 1:N
                new_Delta = Delta_s[i] + sim.mf_state.equations[i]
                lattice_data.current_state.Delta_s[i] = a * lattice_data.current_state.Delta_s[i] + (1 - a) * new_Delta
            end
        elseif pairing_symmetry in ["d-wave", "mayr-d-wave", "d-p-wave"]
            for i in 1:N
                # x-direction
                new_Dx = Delta_x[i] - sim.mf_state.equations[i]
                lattice_data.current_state.Delta_dx[i] =  a* lattice_data.current_state.Delta_dx[i] + (1 - a) * new_Dx
                # y-direction
                new_Dy = Delta_y[i] - sim.mf_state.equations[i + N]
                lattice_data.current_state.Delta_dy[i] =  a* lattice_data.current_state.Delta_dy[i] + (1 - a) * new_Dy
            end
        else
            throw(ErrorException("Pairing symmetry $pairing_symmetry not yet implemented in update_delta_MF!"))
        end
        return sim
    end


    function self_consistent_equations!(sim::MFSim)
        equations = sim.mf_state.equations
        N = sim.lattice_data.sim_data.N
        Delta_s = sim.lattice_data.current_state.Delta_s
        Delta_x = sim.lattice_data.current_state.Delta_dx
        Delta_y = sim.lattice_data.current_state.Delta_dy
        U_potential_s = sim.lattice_data.const_struct.U_params
        U_potential_x = sim.lattice_data.const_struct.U_params_x
        U_potential_y = sim.lattice_data.const_struct.U_params_y
        pairing_symmetry = sim.lattice_data.sim_data.pairing_symmetry
        new_ix_list = sim.lattice_data.const_struct.new_ix_list
        new_iy_list = sim.lattice_data.const_struct.new_iy_list
        if pairing_symmetry =="s-wave"
            for i in 1:N
            # S-wave - on site pairing
            cc_expectation = compute_cc_expectation(sim, i, i)
            # cc_expectation = compute_cc_expectation2(sim, i, i)
            equations[i] = -Delta_s[i] - cc_expectation* U_potential_s[i]
            end
        elseif pairing_symmetry in ["d-wave", "mayr-d-wave", "d-p-wave"]
            for i in 1:N
                # D-wave - nearest neighbor pairing
                # x-direction
                right_neighbor = new_ix_list[i]
                # cc_expectation_x = compute_cc_expectation2(sim, i, right_neighbor) + compute_cc_expectation2(sim, right_neighbor, i)
                cc_expectation_x = 1*compute_cc_expectation_dwave(sim, i, right_neighbor)
                equations[i] = Delta_x[i] - cc_expectation_x * U_potential_x[i]
                # y-direction
                up_neighbor = new_iy_list[i]
                # cc_expectation_y = compute_cc_expectation2(sim, i, up_neighbor) + compute_cc_expectation2(sim, up_neighbor, i)
                cc_expectation_y =  1*compute_cc_expectation_dwave(sim, i, up_neighbor)
                equations[i + N] = Delta_y[i] - cc_expectation_y * U_potential_y[i]
            end
        else
            throw(ErrorException("Pairing symmetry $pairing_symmetry not yet implemented in self_consistent_equations!"))

        end
        return equations
    end

    function compute_cc_expectation(sim::MFSim, i::Int64, j::Int64)
        N = sim.lattice_data.sim_data.N
        Energies = sim.lattice_data.current_state.Energies
        U_matrix = sim.lattice_data.current_state.U_matrix
        beta = sim.lattice_data.sim_data.beta
        delta_ij = 0.0 + 0.0im
        for n in 1:2N
            En = Energies[n, 1]
            fn = (1 + exp(-beta*En))^(-1) #f(-En)
            U_up = U_matrix[i, n, 1]
            U_down = U_matrix[j + N, n, 1]
            delta_ij += U_up * conj(U_down) * fn
            # delta_ij += U_up * conj(U_down) * tanh(beta*En/2)
        end
        return delta_ij
    end

    function compute_cc_expectation2(sim::MFSim, i::Int64, j::Int64)
        # <c_i,down c_j,up>
        N = sim.lattice_data.sim_data.N
        Energies = sim.lattice_data.current_state.Energies
        U_matrix = sim.lattice_data.current_state.U_matrix
        beta = sim.lattice_data.sim_data.beta
        delta_ij = 0.0 + 0.0im
        for n in 1:2N
            En = Energies[n, 1]
            fn = (1 + exp(beta*En))^(-1) #f(En)
            U_idown = U_matrix[i + N, n, 1]
            U_jup = U_matrix[j, n, 1]
            
            delta_ij += U_jup * conj(U_idown) * fn
        end
        return delta_ij
    end

    function compute_cc_expectation_dwave(sim::MFSim, i::Int64, j::Int64)
        # <c_i,down c_j,up + c_j,down c_i,up>
        N = sim.lattice_data.sim_data.N
        Energies = sim.lattice_data.current_state.Energies
        U_matrix = sim.lattice_data.current_state.U_matrix
        beta = sim.lattice_data.sim_data.beta
        delta_ij = 0.0 + 0.0im
        for n in 1:2N
            En = Energies[n, 1]
            U_idown = U_matrix[i + N, n, 1]
            U_jup = U_matrix[j, n, 1]
            
            delta_ij += U_jup * conj(U_idown) * tanh(beta*En/2)
        end
        delta_ij *= -1.0
        return delta_ij
    end



begin # loading and saving
    function load_checkpoint_MF(dir_path::String; mf_params::MFParams=MFParams(5000, 1e-8, 0.6))
        lattice_data = load_checkpoint(dir_path)
        if isnothing(lattice_data)
            throw(ErrorException("Failed to load MF checkpoint from $(dir_path)"))
        end
        return MF_Sim(lattice_data, mf_params)
    end

    function save_simulation_data_MF(sim::MFSim, dir_path::String; sim_info_path=nothing, log_filename=nothing)
        dump_measurements(sim.lattice_data, dir_path)
        return nothing
    end
end

