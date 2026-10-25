# simulation_obs_inspection.jl — look at the saved statistics-report plots of one
# simulation from a notebook: find the simulation in a `get_sims` result, map its
# path to this machine, and display the PNGs that simulation_statistics_report.jl
# wrote into <simulation>/results/statistics_report (dashboard, per-index plots,
# observable-vs-index plots).
#
#   inspect_sim(sims, [4.0, 0.0, 10.0], "stiffness"; cluster="curie", dashboard=true)
#   (the keys of `sims` level by level, as for find_sim)

include("MC_Hubburd.jl")
include("sim_num_control.jl")

using ImageShow


"""
    sim_dir(sim; seed=nothing) -> Union{String, Nothing}

The seed folder (`dir_path`) of a `get_sims` leaf: of seed `seed`, or of the first
seed. Prints a message and returns `nothing` if the entry has no seeds.
"""
function sim_dir(sim::Dict{String, Any}; seed=nothing)

    # To Do :
    #   1. handle multiple seeds better

    # Check if 'sim_dict' and 'simulations' keys exist
    if !haskey(sim, "sim_dict")
        println("❌ 'sim_dict' key not found in the simulation entry.")
        return
    end
    if !haskey(sim["sim_dict"], "simulations")
        println("❌ 'simulations' key not found in the sim_dict.")
        return
    end
    # get the simulation seeds
    simulation_seeds = keys(sim["sim_dict"]["simulations"])
    sim_seed = first(simulation_seeds)
    
    # if multiple seeds, inform the user
    if length(simulation_seeds) > 1
        println("   seeds: ", collect(simulation_seeds))
        if seed !== nothing && seed in simulation_seeds
            sim_seed = seed
            println("   (Using specified seed: $(seed) for inspection.)")
        else
            println("   (No seed specified. Using the first seed for inspection: $(sim_seed)).)")
        end
    end
    
    # get the directory path for the selected seed
    dir_path = sim["sim_dict"]["simulations"][sim_seed]["dir_path"]
    return dir_path
end

"""
Shortcut to find a simulation and print the inspection command.
Usage: find_sim(4.0, 0.0, 10.0)
"""
function find_sim(sims::Dict{Any, Any}, sort_keys::Union{Vector{Float64}, Vector{Any}})
    num_sort_keys = length(sort_keys)
    try
        # Direct lookup in your dictionary structure
        # sim_entry = sims_Us[U_val][B_val][beta_val]
        c_entry = sims
        for i in 1:num_sort_keys
            key = sort_keys[i]
            if !haskey(c_entry, key)
                println("❌ Key $(key) not found at level $(i).")
                return
            end
            c_entry = c_entry[key]
        end
        sim_entry = c_entry
        dir_path = sim_dir(sim_entry)
        
        println("✅ Simulation Found!")
        
        println("   path: ", dir_path)
    
        
        return sim_entry
    catch e
        println("❌ Not found: keys=$(sort_keys)")
        # println("Error: ", e)
    end
end


"""
    inspect_observable_dir(obs_dir; dashboard=true, obs_name=basename(obs_dir))

Display the plots saved for one observable: only its 00_dashboard.png (`dashboard=true`),
or every other PNG in `obs_dir`.
"""
function inspect_observable_dir(obs_dir::String; dashboard::Bool=true, obs_name::String=basename(obs_dir))
    if dashboard
        # load a png file called "00_dashboard.png" if exists
        dashboard_path = joinpath(obs_dir, "00_dashboard.png")
        if isfile(dashboard_path)
            img = load(dashboard_path)
            display(img)
            println("✅ Displayed dashboard for observable $(obs_name) from ", dashboard_path)
        else
            println("❌ Dashboard image not found: ", dashboard_path)
        end
    else
        # plot all png files in the obs_dir except "00_dashboard.png"
        png_files = filter(f -> endswith(f, ".png"), readdir(obs_dir))
        if length(png_files) == 0
            println("❌ No PNG files found in observable directory: ", obs_dir)
            return
        end
        for png_file in png_files
            if png_file == "00_dashboard.png"
                continue
            end
            png_path = joinpath(obs_dir, png_file)
            img = load(png_path)
            display(img)
            println("✅ Displayed image: ", png_path)
        end
    end
end


"""
    inspect_vector_observable(obs_dir; dashboard=false, plot_ix=[], obs_vs_ix_path="", obs_name)

For a vector observable (one `index_<i>` folder per component): list the available
indices, show the observable-vs-index plot(s) `obs_vs_ix_path` if given, and display
the plots of the components in `plot_ix`.
"""
function inspect_vector_observable(obs_dir::String; dashboard=false, plot_ix=[], obs_vs_ix_path::Union{String, Vector{String}}="", obs_name::String=basename(obs_dir))

    if !isdir(obs_dir)
        println("❌ Observable directory not found: ", obs_dir)
        return
    end

    # write the available indicies to the user
    # find available indicies from the directory
    available_ix = Int[]
    dir_entries = readdir(obs_dir)
    for entry in dir_entries
        if isdir(joinpath(obs_dir, entry)) && startswith(entry, "index_")
            push!(available_ix, parse(Int, split(entry, "_")[2]))
        end
    end
    sort!(available_ix)
    if length(available_ix) > 0
            println("Available indicies in observable '$(obs_name)': 1-$(last(available_ix))")
        if length(plot_ix) == 0
            # select first, middle, last indicies
            mid_ix = available_ix[cld(length(available_ix), 2)]
            plot_ix = [first(available_ix), mid_ix, last(available_ix)]
            println("No plot_ix provided. Will plot representative indicies: ", plot_ix)
        end
    else
        println("❌ No subdirectories with indicies found in observable directory: ", obs_dir)
    end



    # show plots the obs vs its index from the obs_vs_ix_path if given
    if obs_vs_ix_path != "" 
        if isa(obs_vs_ix_path, String)
            obs_vs_ix_path = [obs_vs_ix_path]
        end
        for c_obs_vs_ix_path in obs_vs_ix_path
            if isfile(c_obs_vs_ix_path)
               # load and display the image
                img = load(c_obs_vs_ix_path)
                display(img)
                println("✅ Displayed obs_vs_ix image from file: ", c_obs_vs_ix_path)
            else
                println("❌ obs_vs_ix_path is not a directory: ", obs_vs_ix_path)
                return
            end
        end

    end

    if isnothing(plot_ix) || length(plot_ix) == 0
        println("❌ No available plot_ix indices to plot.")
        return
    end

    # plot 3 representative indicies if plot_ix is empty (first, middle, last)
    if !dashboard
        plot_names = filter(f -> !startswith(f, "00"), readdir(joinpath(obs_dir, "index_1")))
        for c_plot in plot_names
            imgs = []
            for ix in plot_ix
                ix_dir = joinpath(obs_dir, "index_$(ix)")
                if !isdir(ix_dir)
                    println("❌ Index directory not found for index $(ix): ", ix_dir)
                    return
                end
                png_path = joinpath(ix_dir, c_plot)
                if isfile(png_path)
                    img = load(png_path)
                    append!(imgs, [img])
                else
                    println("❌ PNG file not found for observable '$(obs_name)' at index $(ix): ", png_path)
                end
                # display the images side by side

                
            end
            if length(imgs) > 0
                combined_img = hcat(imgs...)
                display(combined_img)
                println("✅ Displayed combined image for observable '$(obs_name)' for plot '$(c_plot)' at indices ", plot_ix)
            end
        end
    else
        for ix in plot_ix
        ix_dir = joinpath(obs_dir, "index_$(ix)")
        if !isdir(ix_dir)
            println("❌ Index directory not found for index $(ix): ", ix_dir)
            continue
        end
        println("Inspecting observable '$(obs_name)' at index $(ix):")
        inspect_observable_dir(ix_dir; dashboard=dashboard, obs_name="$(obs_name) index $(ix)")
    end
    end
end

"""
    path_to_obs_vs_index(dir_path, obs_name)

Path(s) of the saved plot of a vector observable against its index (conductivity vs ω,
Λ vs q, χ vs q, …) in the statistics report of the simulation of seed folder `dir_path`;
"" for observables without one.
"""
function path_to_obs_vs_index(dir_path, obs_name)
    stats_dir = joinpath(dirname(dir_path), "results\\statistics_report")

    if startswith(obs_name, "conductivity") || obs_name == "Drude_weight"
        return joinpath(stats_dir, "fixed_delta_observables\\conductivity_vs_omega.png")

    elseif startswith(obs_name, "L") && occursin("_q", obs_name)
        # obs like Lxx_qx, Lyy_qy, L_xy_qx etc
        obs_dir = joinpath(stats_dir, "vector_observables\\Lxx_vs_q\\")
        return joinpath(obs_dir, "$(split(obs_name,"_")[1])_vs_q.png")

    elseif startswith(obs_name, "chi")
        # obs like chi_charge, chi_spin etc
        # for now not plotted as a function of index
        return ""

    elseif obs_name == "Ps_l"
        return joinpath(stats_dir, "vector_observables\\Ps_l\\Ps_l.png")
    elseif obs_name == "Phase_Corr"
        return joinpath(stats_dir, "vector_observables\\phase_correlation\\phase_correlation.png")
    elseif obs_name == "DOS"
        return joinpath(stats_dir, "spectral_plots\\03_dos.png")

    elseif obs_name == "spectral_function"
        return joinpath(stats_dir, "spectral_plots\\01_spectral_function.png")

    elseif obs_name == "BZ"
        return joinpath(stats_dir, "spectral_plots\\02_brillouin_zone.png")

    elseif obs_name == "Delta_complex"
        delta_comp_dir = joinpath(stats_dir, "complex_field")
        fig_names = ["01_trajectory.png", "02_phase_distribution.png", "03_amplitude_distribution.png", "04_amplitude_vs_step.png", "05_phase_vs_step.png"]
        return [joinpath(delta_comp_dir, fig_name) for fig_name in fig_names]

    elseif obs_name == "stiffness" || obs_name == "stiffness_lowest_q" || obs_name == "Lxx_anamoly"
        obs_dir = joinpath(stats_dir, "vector_observables\\Lxx_vs_q\\")
        joint_dist = joinpath(stats_dir, "joint_distributions")
        return [joinpath(obs_dir, "Lxx_vs_q.png"), joinpath(joint_dist, "Lxx_qy_1_vs_Kx.png"), joinpath(joint_dist, "Lxx_qy_1_vs_Lxx_qy_2.png")]

    elseif obs_name == "stiffness_y" || obs_name == "stiffness_y_lowest_q"
        obs_dir = joinpath(stats_dir, "vector_observables\\Lyy_vs_q\\")
        return joinpath(obs_dir, "Lyy_vs_q.png")

    elseif obs_name == "Ps"
        joint_dist = joinpath(stats_dir, "joint_distributions")
        return joinpath(joint_dist, "n_vs_DdD.png")


    else
        println("❌ Unknown or not yet implemented observable name for obs_vs_index plot: ", obs_name)
        return ""
    end
end


"""
    inspect_observable(dir_path, obs_name; dashboard=false, plot_ix=nothing)

Display the statistics-report plots of observable `obs_name` for the simulation whose
seed folder is `dir_path` (scalar, vector, or spectral observables).
"""
function inspect_observable(dir_path, obs_name; dashboard::Bool=false, plot_ix=nothing)
    # To DO:
    #   1. Add exception for Bootstrap observables - show the relevant obs and the multi-correlation
    #   2. Add the value of the q,w,l and also the index.
    #   3. add exception for delta_fixed observables
    #   4. Add a plotting of the comples HS field
    spectral_obs = ["spectral_function", "BZ"]
    # dir_patth is the path to the simulation seed
    results_dir = joinpath(dirname(dir_path), "results")
    if !isdir(results_dir)
        println("❌ results Directory not found: ", results_dir)
        return
    end
    all_obs_dir = joinpath(results_dir, "statistics_report", "observables")
    all_obs_names = readdir(all_obs_dir)
    if obs_name in all_obs_names
        obs_dir = joinpath(all_obs_dir, obs_name)
        obs_dir_content = readdir(obs_dir)
        # if has folder starting with index_
        if any(startswith.(obs_dir_content, "index_"))
            # vector obs
            plot_ix = isnothing(plot_ix) ? Int[] : plot_ix isa Int ? [plot_ix] : plot_ix
            obs_vs_ix_path = path_to_obs_vs_index(dir_path, obs_name)
            inspect_vector_observable(obs_dir; dashboard=dashboard, plot_ix=plot_ix, obs_name=obs_name, obs_vs_ix_path=obs_vs_ix_path)
        else
            # scalar obs
            inspect_observable_dir(obs_dir; dashboard=dashboard, obs_name=obs_name)
        end
    elseif obs_name in spectral_obs
        # spectral obs
        obs_dir = joinpath(dirname(dir_path), "results", "statistics_report", "spectral_plots")
        obs_vs_ix_path = path_to_obs_vs_index(dir_path, obs_name)
        inspect_vector_observable(obs_dir; dashboard=dashboard, plot_ix=nothing, obs_name=obs_name, obs_vs_ix_path=obs_vs_ix_path)
    else
        println("❌ Observable $(obs_name) not found in ", all_obs_dir)
        # plot obs_vs_index if possible
        obs_vs_ix_path = path_to_obs_vs_index(dir_path, obs_name)
        if obs_vs_ix_path != ""
            obs_vs_ix_path_dirname = isa(obs_vs_ix_path, String) ? dirname(obs_vs_ix_path) : dirname(first(obs_vs_ix_path))
            inspect_vector_observable(obs_vs_ix_path_dirname; dashboard=dashboard, plot_ix=nothing, obs_name=obs_name, obs_vs_ix_path=obs_vs_ix_path)
        end
    end
end


"""
    inspect_sim(sims, sort_keys, obs_name; cluster="curie", dashboard=false, plot_ix=nothing, seed=nothing)

Find the simulation `sort_keys` in a nested `get_sims` result, map its seed folder to
this machine for `cluster` (`fix_cluster_path`), and show the plots of `obs_name`
(`inspect_observable`).
"""
function inspect_sim(sims, sort_keys, obs_name::String; cluster="curie", dashboard=false, plot_ix=nothing, seed=nothing)
    # Find the simulation entry in the sims dictionary
    sim_entry = find_sim(sims, sort_keys)
    if isnothing(sim_entry)
        return
    end
    # Get the simulation directory path form the sim dir
    dir_path = sim_dir(sim_entry; seed=seed)
    if isnothing(dir_path)
        return
    end

    # Fix the path for the given cluster
    dir_path_fixed = fix_fluster_path(dir_path, cluster)
    println("inspecting sim on cluster $(cluster) at path: $(dir_path_fixed)")
    println("for observable: $(obs_name)")

    # Inspect the specified observable
    inspect_observable(dir_path_fixed, obs_name; dashboard=dashboard, plot_ix=plot_ix)
end

"""
    fix_fluster_path(path, cluster) -> String

Old name, kept for the notebooks: `fix_cluster_path(path; cluster)` (MC_Hubburd.jl),
which maps a cluster path to that cluster's mount or the backup drive.
`cluster = "local"` returns `path` unchanged.
"""
function fix_fluster_path(original_path::String, cluster::String)
    cluster == "local" && return original_path
    if !(cluster in ("curie", "landau"))
        println("Unknown cluster: $(cluster). Using original path.")
        return original_path
    end
    return fix_cluster_path(original_path; cluster=cluster)
end