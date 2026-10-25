using Statistics, Dates, Base.Iterators, JLD2, ArgParse

include(joinpath(@__DIR__, "..", "..", "occupation_convergence.jl"))

function run_me(sim_path)
    sim_data_file=joinpath(sim_path,"sim.jld2")
    sim_data_dict = JLD2.load(sim_data_file)

    delta_fixed_values = get(sim_data_dict, "delta_fixed_values", Float64[])
    delta_values_stiffness = get(sim_data_dict, "delta_values_stiffness", Float64[])
    dw_values = get(sim_data_dict, "dw_values", Float64[])

    # Use running_simulation_seeds_wanted_occupation for multiple seeds
    running_simulation_seeds_wanted_occupation(
                            sim_data_dict["Lx"],
                            sim_data_dict["Ly"],
                            sim_data_dict["SC_L"],
                            sim_data_dict["U"],
                            sim_data_dict["t"],
                            sim_data_dict["V"],
                            sim_data_dict["seeds"],  # Full seeds vector (not just seeds[1])
                            sim_data_dict["wanted_n"],
                            sim_data_dict["beta"],
                            sim_data_dict["B"],
                            sim_data_dict["num_measure"],
                            sim_data_dict["num_thermal"],
                            sim_data_dict["measuring_interval"],
                            sim_data_dict["w_list"],
                            sim_data_dict["guess"],
                            sim_data_dict["U_disorder"],
                            sim_data_dict["diag_method"],
                            sim_data_dict["pairing_symmetry"],
                            sim_data_dict["w_conductivity_range"],
                            sim_data_dict["impurity"];

                            directory=sim_path,
                            num_cores=sim_data_dict["cores"],
                            savedata=true,
                            saving_interval=500,
                            saveplots=false,
                            date_folder=false,
                            seed_folder=true,  # Each seed gets its own folder
                            dump_time=Dates.Hour(1),
                            numbered=false,
                            U_puddle_params=sim_data_dict["U_puddle_params"],
                            TClustersize_x=sim_data_dict["TClustersize"],
                            TClustersize_y=sim_data_dict["TClustersize"],
                            U_smooth=sim_data_dict["smooth_U"],
                            # Occupation convergence parameters
                            num_thermal_occ=sim_data_dict["num_thermal_occ"],
                            num_measure_occ=sim_data_dict["num_measure_occ"],
                            measuring_interval_occ=sim_data_dict["measuring_interval_occ"],
                            n_tol=sim_data_dict["n_tol"],
                            max_iter=sim_data_dict["max_iter"],
                            num_uncorrelated_measurments=sim_data_dict["num_uncorrelated_measurments"],
                            boundary_conditions=sim_data_dict["boundary_condition"],
                            gauge_choice=sim_data_dict["gauge_choice"],
                            delta_fixed_values=delta_fixed_values,
                            delta_values_stiffness=delta_values_stiffness,
                            dw_values=dw_values
                            )
end

s = ArgParseSettings()
@add_arg_table s begin
     "--sim_path"
         help = "sim_path"
end

parsed_args = parse_args(ARGS, s)
sim_path = parsed_args["sim_path"]

run_me(sim_path)
