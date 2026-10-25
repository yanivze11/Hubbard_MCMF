# sim_num_control.jl — from raw simulation folders to means with error bars ("num_control").
#
# A campaign folder holds one sub-folder per parameter point (its sim.jld2 lists the
# MC seeds under "simulations") and one seed_<s> folder per seed with results.jld2.
# This file
#   * reads every sim.jld2 into one nested Dict `sim_info`         get_sim_info, merge_sims
#   * computes mean ± error of every observable for one seed        get_means
#   * combines the seeds of one parameter point                      average_over_seeds
#   * does all of the above for a campaign and writes means_db      update_means
# It also holds the bootstrap estimators of derived quantities (BT_*: superfluid
# stiffness, resistivity tensor), the δ→0 extrapolation of fixed-broadening
# observables, lookup helpers on `sim_info` (get_sims, group_by_*, get_obs_vs_T),
# LDOS binning by local spectral gap, and per-seed progress diagnostics.
# Cluster entry point: run_num_control_V.jl. See docs/WORKFLOW.md §6.

using JLD2
using Dates
using CurveFit
include("MC_Hubburd.jl")
include("statistics.jl")
# Guarded: several scripts include both this file and means_db.jl, and a bare
# re-include would warn on every `const` (Set/String consts are never ===).
isdefined(Main, :MEANS_DB_SCHEMA_VERSION) || include("means_db.jl")

"""
    get_sim_dimensions(sim_dict) -> (Lx, Ly)

Lattice size of a parameter point. Old entries store a single `"L"` (square lattice);
returns `(nothing, nothing)` if neither form is present.
"""
function get_sim_dimensions(sim_dict::Dict)
    if haskey(sim_dict, "Lx") && haskey(sim_dict, "Ly")
        return sim_dict["Lx"], sim_dict["Ly"]
    elseif haskey(sim_dict, "L")
        L = sim_dict["L"]
        return L, L  # Legacy: square lattice
    else
        return nothing, nothing
    end
end

"""
    format_dimensions(sim_dict) -> String

Lattice size for printing: `"12"` for a square lattice, `"12×8"` otherwise, `"N/A"` if unknown.
"""
function format_dimensions(sim_dict::Dict)
    Lx, Ly = get_sim_dimensions(sim_dict)
    if isnothing(Lx)
        return "N/A"
    elseif Lx == Ly
        return "$(Lx)"  # Square: just show single number for compactness
    else
        return "$(Lx)×$(Ly)"  # Rectangular: show both
    end
end

"""
    get_sim_info(path) -> Dict{String,Any}

Read the `sim.jld2` of every sub-folder of the campaign folder `path`.

Returns `sim_info`: folder name (the "sim_num") => the contents of its sim.jld2
(parameters, plus `"simulations"`: seed => per-seed tracking info), and
`"updated_at"` => the time of the read. Folders whose sim.jld2 cannot be read are
skipped with an error message.
"""
function get_sim_info(path::String)
    simulations = filter(f -> isdir(joinpath(path, f)), readdir(path))
    sim_info = Dict{String, Any}()
    sim_info["updated_at"] = Dates.now()  # Store the update time

    for simulation in simulations
        sim_path = joinpath(path, simulation)
        param_path = joinpath(sim_path, "sim.jld2")
        sim_num = basename(simulation)  # Get the simulation number from the directory name
        if isfile(param_path)
            try
                jldopen(param_path, "r") do file
                    sim_data = Dict{String, Any}()
                    for key in keys(file)
                        sim_data[key] = file[key]
                    end
                    # Add the simulation data to the dictionary with the simulation number as the key
                    sim_info[sim_num] = sim_data
                end
            catch e
                @error "Failed to load simulation data from $param_path: $e"
                continue  # Skip this simulation if loading fails
            end
        end
    end
    return sim_info
end

"""
    get_sim_info_seeds(path) -> Dict{String,Any}

Like `get_sim_info`, but also read the `sim.jld2` inside every sub-sub-folder (one per seed).

Returns folder name => Dict(`"no seed"` => the folder's own sim.jld2, `"<seed>"` =>
each seed folder's sim.jld2), plus `"updated_at"`.
"""
function get_sim_info_seeds(path::String)
    simulations = filter(f -> isdir(joinpath(path, f)), readdir(path))
    sim_info = Dict{String, Any}()
    sim_info["updated_at"] = Dates.now()  # Store the update time

    for simulation_folder in simulations
        sim_info["$(simulation_folder)"] = Dict{String, Any}()  # Initialize a dictionary for each simulation folder

        # check for files in this folder
        param_path = joinpath(path, simulation_folder, "sim.jld2")
        if isfile(param_path)
            try
                param_data = JLD2.load(param_path)
                sim_info["$simulation_folder"]["no seed"] = param_data
            catch e
                @error "Failed to load seed data from $param_path: $e"
            end
        end

        # check for seed folders in this folder
        seed_folders = filter(f -> isdir(joinpath(path, simulation_folder, f)), readdir(joinpath(path, simulation_folder)))
        for seed_folder in seed_folders
            seed_param_path = joinpath(path, simulation_folder, seed_folder, "sim.jld2")
            if isfile(seed_param_path)
                try
                    seed_data = JLD2.load(seed_param_path)
                    sim_info["$simulation_folder"]["$(seed_data["seed"])"] = seed_data
                catch e
                    @error "Failed to load seed data from $seed_param_path: $e"
                    continue  # Skip this seed if loading fails
                end
            end
        end
    end

    return sim_info
end

"""
    merge_sims(sim_info) -> Dict{String,Any}

Merge folders that hold the same parameter point (same `sim_parameters`), e.g. when a
point was resubmitted into a new folder. The seeds (`"simulations"`, `"seeds"`) of the
duplicates are added to the first folder found, and the duplicates are dropped.
"""
function merge_sims(sim_info::Dict{String, Any})
    new_sim_info = Dict{String, Any}("updated_at" => sim_info["updated_at"])
    params_list = []
    for (sim_num, sim) in sim_info
        if sim_num == "updated_at"
            continue
        end
        params = sim_parameters(sim)
        if params in params_list
            for (new_sim_num, new_sim) in new_sim_info
                if new_sim_num == "updated_at"
                    continue
                end
                if params == sim_parameters(new_sim)
                    for s in [sim, new_sim]
                        if !haskey(s, "simulations")
                            s["simulations"] = Dict()
                        end
                    end
                    new_sim_info[new_sim_num]["simulations"] = merge(new_sim["simulations"], sim["simulations"])
                    new_sim_info[new_sim_num]["seeds"] = vcat(new_sim_info[new_sim_num]["seeds"], sim["seeds"])
                    continue
                end
            end
        else
            new_sim_info["$sim_num"] = sim
            param_list = push!(params_list, params)
        end
    end
    return new_sim_info
end

"""
    sim_parameters(sim) -> Dict{String,Any}

The physical/run parameters of a parameter point: its sim.jld2 entries without
bookkeeping keys (seeds, paths, times, means, convergence state, `measuring_interval`).
Two folders with equal `sim_parameters` are the same point (see `merge_sims`).
"""
function sim_parameters(sim::Dict{String, Any})
    params = collect(keys(sim))
    field_not_matter =  ["simulations", "dir_path", "sim_num", "info_path", "start_time", "seeds", "means", "measuring_interval", "convergence", "num_uncorrelated_measurments", "all_seeds_completed", "last_completed_seed", "last_completed_time", "endtime", "completed_seeds"]
    params = filter(p -> !(p in field_not_matter), params)
    out = Dict{String, Any}()
    for p in params
        out[p] = get(sim, p, "N/A")
    end
    return out
end

"""
    save_sim_info(path, sim_info; filters=nothing)

Write `sim_info` to `path/num_control/sim_info.jld2` (the old single-file format), or to
`sim_info_<k=v_...>.jld2` when `filters` is given.
"""
function save_sim_info(path::String, sim_info::Dict{String, Any}; filters=nothing)
    sim_info_path = joinpath(path, "num_control", "sim_info.jld2")
    if filters !== nothing
        filter_name = join([string(k, "=", v) for (k, v) in filters], "_")
        sim_info_path = joinpath(path, "num_control", "sim_info_$filter_name.jld2")
    end
    JLD2.save(sim_info_path, sim_info)
end

"""
    print_sim_details(path)

Write a text report of a campaign to `path/num_control/sim_details.txt`, from its
saved `num_control/sim_info.jld2`.
"""
function print_sim_details(path::String)
    sim_info = load_sim_info(path)
    sim_details_path = joinpath(path, "num_control", "sim_details.txt")
    print_sim_details(sim_info, sim_details_path)
    
end

"""
    print_sim_details(sim_info, sim_details_path)

Write a text report of `sim_info` to `sim_details_path`: one summary line per parameter
point (size, method, μ, β, V, <n>, MC steps, finished/total seeds, progress of a running
seed), then all parameters of each point.
"""
function print_sim_details(sim_info::Dict{String, Any}, sim_details_path::String)

    open(sim_details_path, "w") do io
        println(io, "Simulation Details Report")
        println(io, "Generated at: ", get(sim_info, "updated_at", "Unknown"))
        println(io, "----------------------------------------")

        # table that summarizes the simulations
        println(io, "Simulation Summary:")
        println(io, "sim num | Lx×Ly | L_SC | diag_method | TCA size | mu | beta | V | <n> | #Thermal | #Measurements | measuring interval | cores | seed done")
        # Sort sim_num as integers
        sorted_sim_nums = sort(filter(x -> !isnothing(x), tryparse.(Int, keys(sim_info))))
        for sim_num in sorted_sim_nums
            sim_data = sim_info["$sim_num"]
            means = get(sim_data, "means", Dict{String, Any}())
            
            L_str = format_dimensions(sim_data)
            L_SC = get(sim_data, "SC_L", "N/A")
            diag_method = get(sim_data, "diag_method", "N/A")
            TCA_size = diag_method in ["TCluster", "K0-TCA"] ? get(sim_data, "TClusersize", "N/A") : "--"
            mu = get(sim_data, "mu", "N/A")
            beta = get(sim_data, "beta", "N/A")
            V = get(sim_data, "V", "N/A")
            n = (isempty(means) || typeof(means)==String) ? "N/A" : round(get(get(means, "n", Dict{String, Any}()), "value", NaN), digits=3)
            num_thermal = get(sim_data, "num_thermal", "N/A")
            num_measure = get(sim_data, "num_measure", "N/A")
            measuring_interval = get(sim_data, "measuring_interval", "N/A")
            cores = get(sim_data, "cores", "N/A")
            seeds = length(get(sim_data, "seeds", []))
            seeds_finished = 0
            running_seed = nothing
            if haskey(sim_data, "simulations")
                for (seed, seed_data) in sim_data["simulations"]
                    if seed_data["status"] == "finished"
                        seeds_finished += 1
                    elseif seed_data["status"] == "running"
                        running_seed = seed
                    end
                end
            end
            seeds_print = "$seeds_finished/$seeds" 

            

            println(io, "$sim_num | $L_str | $L_SC | $diag_method | $TCA_size | $mu | $beta | $V | $n | $num_thermal | $num_measure | $measuring_interval | $cores | $seeds_print")
            if running_seed !== nothing
                
                step_count = get(sim_data["simulations"][running_seed], "step_count", 0)
                c_meas = get(sim_data["simulations"][running_seed], "num_current_measurments", 0)
                Thermalization = step_count <= num_thermal ? "$(step_count)/$(num_thermal) = $(step_count/num_thermal *100)%" : "$(num_thermal)/$(num_thermal) = 100%"
                Measurements = "$(c_meas)/$(num_measure) = $(c_meas/num_measure *100)%"
                println(io, "  Running seed: $running_seed (Thermalization: $Thermalization, Measurements: $Measurements, Total steps: $step_count)")
            end
        end
        println(io, "----------------------------------------")


        # detail each simulation
        for sim_num in sorted_sim_nums
            sim_data = sim_info["$sim_num"]
            println(io, "Simulation: ", sim_num)
            if isa(sim_data, Dict)
                for (k, v) in sim_data
                    if k=="simulations"
                        continue
                    elseif k=="means" && isa(v, Dict)
                        continue
                    else
                        println(io, "  $k: $v")
                    end
                end

            end
            println(io, "----------------------------------------")
        end
    end
end


"""
    load_sim_info(path) -> Dict{String,Any}

Load `path/num_control/sim_info.jld2` (the old single-file means of a campaign, still
used for V6 data). Returns an empty Dict if the file is missing or unreadable.
"""
function load_sim_info(path::String)
    sim_info_path = joinpath(path, "num_control", "sim_info.jld2")
    if isfile(sim_info_path)
        try
            return JLD2.load(sim_info_path)
        catch e
            @error "Failed to load sim_info.jld2: $e"
            println("Returning an empty dictionary.")
            return Dict{String, Any}()
        end
    else
        return Dict{String, Any}()
    end
end


"""
    update_means(path; export_db=true, db_root=MEANS_DB_ROOT, reuse_db=true,
                 save_monolith=true, write_means_cache=false)

Compute the means of every parameter point of the campaign folder `path` and store them.

Steps: `get_sim_info` → `merge_sims` → `get_sim_info_means` (per-seed `get_means`,
then `average_over_seeds`) → write.
- `export_db`: write one small file per parameter point to means_db under `db_root`.
- `reuse_db`: seeds whose means_db entry is up to date (same number of measurements)
  are read back from means_db instead of loading their results.jld2.
- `save_monolith`: also write the old single file `num_control/sim_info.jld2`.
  The V7 driver run_num_control_V.jl passes `false`; the default `true` is kept for
  the archived V6 pre-run scripts, which read that file to skip finished jobs.
- `write_means_cache`: also write `means_cache.jld2` into each seed folder (old cache,
  superseded by means_db; it writes into the data folders, so it is off).
"""
function update_means(path::String; export_db::Bool=true, db_root::String=MEANS_DB_ROOT,
                      reuse_db::Bool=true, save_monolith::Bool=true,
                      write_means_cache::Bool=false)
    # It writes into the campaign folder and db_root: never on the backup drive.
    _assert_writable(path, "update_means")
    export_db && _assert_writable(db_root, "update_means (means_db export)")

    # make the sim_info from folders
    println("Loading simulation info from path: ", path)
    sim_info = get_sim_info(path)
    # merge similar parameter simulations that are in different folders
    println("Merging simulations with identical parameters...")
    sim_info = merge_sims(sim_info)
    # get the means for each simulation
    println("Calculating means for each simulation...")
    sim_info = get_sim_info_means(sim_info; reuse_db=reuse_db, db_root=db_root,
                                  write_means_cache=write_means_cache)

    # Legacy monolith — analysis no longer needs it; the pre-run scripts still do.
    if save_monolith
        save_sim_info(path, sim_info)
    end

    # Fan out to means_db: one light file per simulation, so analysis never has
    # to open the monolith again and there is no store to rebuild.
    if export_db
        println("Exporting to means_db at $db_root ...")
        export_sim_info!(sim_info; root=db_root, source=basename(rstrip(path, ['/', '\\'])))
    end
end

"""
    update_means_filtered(path; filters::Dict{String,Any})

Like `update_means`, but only for the parameter points whose sim.jld2 matches ALL
`filters` (e.g. `Dict("Lx" => 12, "beta" => 3.0)`). The result is merged into and saved
as `num_control/sim_info.jld2`; means_db is not written.

```julia
update_means_filtered(path; filters=Dict("Lx" => 12, "mu" => 2.0))
```
"""
function update_means_filtered(path::String; filters::Dict{String, Any}=Dict{String, Any}())
    if isempty(filters)
        println("Warning: No filters provided. Use update_means() to process all simulations.")
        return
    end
    
    # make the sim_info from folders
    println("Loading simulation info from path: ", path)
    sim_info = get_sim_info(path)
    
    # Filter simulations based on provided parameters
    println("Filtering simulations based on criteria: $filters")
    filtered_sim_info = Dict{String, Any}("updated_at" => sim_info["updated_at"])
    
    for (sim_num, sim_data) in sim_info
        if sim_num == "updated_at"
            continue
        end
        
        # Check if all filter criteria are met
        all_match = true
        for (filter_key, filter_value) in filters
            if !haskey(sim_data, filter_key)
                all_match = false
                break
            end
            if sim_data[filter_key] != filter_value
                all_match = false
                break
            end
        end
        
        if all_match
            filtered_sim_info[sim_num] = sim_data
        end
    end
    
    num_filtered = length(filtered_sim_info) - 1  # subtract 1 for "updated_at"
    println("Found $num_filtered simulation(s) matching filters")
    
    if num_filtered == 0
        println("No simulations match the specified filters.")
        return
    end
    
    # merge similar parameter simulations that are in different folders
    println("Merging simulations with identical parameters...")
    filtered_sim_info = merge_sims(filtered_sim_info)
    
    # get the means for each simulation
    println("Calculating means for filtered simulations...")
    filtered_sim_info = get_sim_info_means(filtered_sim_info)

    # Merge back into main sim_info and save
    for (sim_num, sim_data) in filtered_sim_info
        if sim_num != "updated_at"
            sim_info[sim_num] = sim_data
        end
    end
    
    # Save the updated simulation info with means
    save_sim_info(path, sim_info)
    println("Successfully updated means for filtered simulations!")
end

"""
    get_sim_info_means(sim_info; reuse_db=true, db_root=MEANS_DB_ROOT, write_means_cache=false)

For every parameter point of `sim_info`: compute each seed's means (`get_seeds_means`)
and store the seed average (`average_over_seeds`) as `sim_info[sim_num]["means"]`.
Points without seeds get a message string instead. Modifies and returns `sim_info`.
"""
function get_sim_info_means(sim_info::Dict{String, Any};
                            reuse_db::Bool=true, db_root::String=MEANS_DB_ROOT,
                            write_means_cache::Bool=false)
    for sim_num in keys(sim_info)
        if sim_num == "updated_at"
            continue
        end
        if !haskey(sim_info[sim_num], "simulations")
            sim_info[sim_num]["means"] = "Simulation not yet run or no data available"
            continue  # Skip if no simulations are available
        end

        # get means for each seed in this simulation
        get_seeds_means(sim_info, sim_num; reuse_db=reuse_db, db_root=db_root,
                        write_means_cache=write_means_cache)

        # average over all seeds
        total_mean_dict = average_over_seeds(sim_info, sim_num)
        sim_info[sim_num]["means"] = total_mean_dict
    end
    return sim_info
end

# ============================================================================
# Means Cache: save/load functions
# ============================================================================

"""
    is_dwave_pairing(pairing_symmetry) -> Bool

True for the (experimental) pairing symmetries with a d-wave part, which have extra
d-wave phase-correlation observables.
"""
function is_dwave_pairing(pairing_symmetry::String)
    pairing_symmetry in ["d-wave", "d-p-wave", "mayr-d-wave", "d+s-wave"]
end

"""
    first_component_or_self(value)

`value[1]` for a vector (`nothing` if empty), else `value` itself.
"""
function first_component_or_self(value)
    if value isa AbstractVector
        return isempty(value) ? nothing : value[1]
    end
    return value
end

"""
    add_legacy_means_aliases!(means_dict, measurments, num_current_measurment; std_scheme="correlation")

Add the old names of renamed means so old notebooks keep working:
`"U4"` ⇄ `"K0_action"` (copy whichever is missing), and `"Delta_mean"` computed from the
first component of the `Delta_MC` stream if it is not there yet.
"""
function add_legacy_means_aliases!(means_dict::Dict{String, Any}, measurments::MeasureData, num_current_measurment::Int; std_scheme::String="correlation")
    if haskey(means_dict, "K0_action") && !haskey(means_dict, "U4")
        means_dict["U4"] = deepcopy(means_dict["K0_action"])
    elseif haskey(means_dict, "U4") && !haskey(means_dict, "K0_action")
        means_dict["K0_action"] = deepcopy(means_dict["U4"])
    end

    if haskey(means_dict, "Delta_mean")
        return
    end
    if !hasfield(typeof(measurments), :Delta_MC)
        return
    end

    delta_mc_data = getfield(measurments, :Delta_MC)
    if isempty(delta_mc_data) || length(delta_mc_data) < num_current_measurment
        return
    end

    delta_s_series = ComplexF64[]
    for delta_vec in delta_mc_data[1:num_current_measurment]
        if !isempty(delta_vec)
            push!(delta_s_series, delta_vec[1])
        end
    end
    if length(delta_s_series) != num_current_measurment
        return
    end

    mean_val, err = observable_mean(delta_s_series, num_current_measurment; std_scheme=std_scheme)
    τ = find_correlation_time(delta_s_series)

    if isa(τ, Vector) && !isempty(τ) && isa(τ[1], Vector)
        num_uncorr = [Int.(floor.(num_current_measurment ./ (maximum(tau_array) == 0 ? 1 : maximum(tau_array)))) for tau_array in τ]
    elseif isa(τ, AbstractArray)
        num_uncorr = Int.(floor.(num_current_measurment ./ ifelse.(τ .== 0, 1, τ)))
    else
        num_uncorr = Int(floor(num_current_measurment / (τ == 0 ? 1 : τ)))
    end

    means_dict["Delta_mean"] = Dict{String, Any}(
        "value" => mean_val,
        "error" => err,
        "correlation_time" => τ,
        "num_uncorrelated_measurements" => num_uncorr,
        "num_measurements" => num_current_measurment,
    )
end

"""
    save_means_cache(seed_dir, means_dict, num_measurements; std_scheme="correlation", fit_stiffness=false)

Save the output of `get_means` to `seed_dir/means_cache.jld2` (old per-seed cache, now
superseded by means_db), so the statistics report can skip the recomputation.
"""
function save_means_cache(seed_dir::String, means_dict::Dict, num_measurements::Int;
                          std_scheme::String="correlation", fit_stiffness::Bool=false)
    cache_path = joinpath(seed_dir, "means_cache.jld2")
    cache_data = Dict(
        "means" => means_dict,
        "num_measurements" => num_measurements,
        "computed_at" => Dates.now(),
        "std_scheme" => std_scheme,
        "fit_stiffness" => fit_stiffness,
        "schema_version" => 2,
        "schema_notes" => [
            "K0_action is canonical alias for legacy U4",
            "Delta_mean is legacy alias derived from Delta_MC[1]",
        ],
    )
    JLD2.save(cache_path, cache_data)
    println("  Saved means cache to: $cache_path (N=$num_measurements)")
end

"""
    load_means_cache(seed_dir) -> Union{Dict, Nothing}

Load `seed_dir/means_cache.jld2` and add the legacy aliases (`"U4"`/`"K0_action"`,
`"Delta_mean"`). Returns the whole cache Dict, or `nothing` if missing or unreadable.
"""
function load_means_cache(seed_dir::String)
    cache_path = joinpath(seed_dir, "means_cache.jld2")
    if !isfile(cache_path)
        return nothing
    end
    try
        cache_data = JLD2.load(cache_path)
        cache_data["schema_version"] = get(cache_data, "schema_version", 1)
        if haskey(cache_data, "means") && cache_data["means"] isa Dict
            means = cache_data["means"]

            if haskey(means, "K0_action") && !haskey(means, "U4")
                means["U4"] = deepcopy(means["K0_action"])
            elseif haskey(means, "U4") && !haskey(means, "K0_action")
                means["K0_action"] = deepcopy(means["U4"])
            end

            if !haskey(means, "Delta_mean") && haskey(means, "Delta_MC")
                delta_mc = means["Delta_MC"]
                if delta_mc isa Dict
                    delta_val = first_component_or_self(get(delta_mc, "value", nothing))
                    if !isnothing(delta_val)
                        means["Delta_mean"] = Dict{String, Any}(
                            "value" => delta_val,
                            "error" => first_component_or_self(get(delta_mc, "error", 0.0)),
                            "correlation_time" => first_component_or_self(get(delta_mc, "correlation_time", 0.0)),
                            "num_uncorrelated_measurements" => first_component_or_self(get(delta_mc, "num_uncorrelated_measurements", 0)),
                            "num_measurements" => get(delta_mc, "num_measurements", get(cache_data, "num_measurements", 0)),
                        )
                    end
                end
            end

            cache_data["means"] = means
        end

        return cache_data
    catch e
        @warn "Failed to load means cache from $cache_path: $e"
        return nothing
    end
end

"""
    means_db_average(seeds::AbstractDict)

Seed average for means_db: wrap means_db's `seed => Dict("info"=>..., "means"=>...)`
mapping as one parameter point and call `average_over_seeds`. means_db uses it (through
`MEANS_DB_AVERAGE_FN`) to re-average after merging seeds, without depending on this file.
"""
function means_db_average(seeds::AbstractDict)
    sims = Dict{Any, Any}()
    for (s, d) in seeds
        sd = Dict{String, Any}()
        haskey(d, "info")  && merge!(sd, d["info"])
        haskey(d, "means") && (sd["means"] = d["means"])
        sims[something(tryparse(Int, string(s)), string(s))] = sd
    end
    fake = Dict{String, Any}("1" => Dict{String, Any}("simulations" => sims))
    return average_over_seeds(fake, "1")
end
MEANS_DB_AVERAGE_FN[] = means_db_average

"""
    _slimmed_path(dir) -> Union{String, Nothing}

Path of `dir/results_slim.jld2` if the seed folder was slimmed (slim_results.jl replaced
results.jld2 by results_slim.jld2: exact means + binned streams + sim_data), else `nothing`.
"""
function _slimmed_path(dir::AbstractString)
    slim = joinpath(dir, "results_slim.jld2")
    return (!isfile(joinpath(dir, "results.jld2")) && isfile(slim)) ? slim : nothing
end

"""
    _slim_means(dir) -> Union{Dict, Nothing}

The per-seed means stored in a slimmed seed folder, or `nothing` if not slimmed.
"""
function _slim_means(dir::AbstractString)
    slim = _slimmed_path(dir)
    slim === nothing && return nothing
    return _open_jld2(slim) do f
        haskey(f, "means") || return nothing
        Dict{String, Any}(string(k) => f["means/$k"] for k in keys(f["means"]))
    end
end

"""
    get_seeds_means(sim_info, sim_num; reuse_db=true, db_root=MEANS_DB_ROOT, write_means_cache=false)

Fill `"means"` for every seed of parameter point `sim_num` (in place) and return its
`"simulations"` Dict. For each seed, the first source that works is used:
1. means_db, if the stored entry has the same number of measurements (`reuse_db`);
2. results_slim.jld2, if the seed was slimmed;
3. `get_means` on the loaded results.jld2.
A seed that cannot be loaded gets the string `"Failed to load data"`.
"""
function get_seeds_means(sim_info::Dict{String, Any}, sim_num::String;
                         reuse_db::Bool=true, db_root::String=MEANS_DB_ROOT,
                         write_means_cache::Bool=false)
    simulations = sim_info[sim_num]["simulations"]
    # Where this simulation lives in means_db (nothing if it lacks required
    # identity parameters, e.g. very old entries).
    db_path = reuse_db ? (try entry_path(sim_info[sim_num]; root=db_root) catch; nothing end) : nothing

    for seed in keys(simulations)
        # Already-computed seeds are read back from means_db instead of being
        # re-derived from the ~369 MB results.jld2. This is what makes the
        # per-seed means_cache.jld2 redundant.
        if db_path !== nothing
            nmeas = get(simulations[seed], "num_current_measurments", 0)
            if nmeas > 0 && seed_is_current(db_path, seed, nmeas)
                stored = read_seed_means(db_path, seed)
                if stored !== nothing
                    simulations[seed]["means"] = stored
                    continue
                end
            end
        end

        # The cluster this pass runs for (MEANS_DB_CLUSTER), so a mount path is
        # never mapped to the other cluster's disk.
        data_dir = fix_cluster_path(simulations[seed]["dir_path"];
                                    cluster=MEANS_DB_CLUSTER in DB_CLUSTERS ? MEANS_DB_CLUSTER : "")
        # A slimmed seed has no raw stream left; its exact means were copied
        # into results_slim.jld2 (slim_results.jl), so read them from there.
        slim_means = _slim_means(data_dir)
        if slim_means !== nothing
            simulations[seed]["means"] = slim_means
            continue
        end
        data = nothing
        try
            data = load_results(data_dir, to_load_checkpoint=false)
        catch e
            @warn "Failed to load data from $data_dir: $e"
            simulations[seed]["means"] = "Failed to load data"
            continue
        end
        if data !== nothing
            means_dict = get_means(data; no_avg_obs=true, fit_stiffness=false)
            simulations[seed]["means"] = means_dict

            # Legacy per-seed cache. Superseded by means_db, which stores these
            # same per-seed means (see seed_is_current above) — and this writes
            # INTO the simulation's own directory, i.e. onto the cluster, so it
            # is off by default.
            if write_means_cache
                num_meas = data.current_state.num_current_measurments
                save_means_cache(data_dir, means_dict, num_meas;
                                std_scheme="correlation", fit_stiffness=false)
            end
        end
    end
    return simulations
end

# Fields `load_results` collapses to a single mean across chunks (mean_obs) — no
# per-measurement series survives on disk, so `get_stream` must refuse them
# rather than hand back the mean array dressed up as a time series.
const _STREAMLESS_MEAN_OBS = Set([:BZ, :A, :LDOS])

"""
    _means_db_load_stream(dir_path, obs, vec_index; cluster="") -> (stream, n_meas, dir) or nothing

Heavy half of means_db's `get_stream`: load ONE seed's results.jld2 (~370 MB) and return
the per-measurement series of the raw `MeasureData` field `obs` (component `vec_index`
for vector observables). Errors with an explanation for derived observables, for
fields stored only as a mean (`:BZ`, `:A`, `:LDOS`), for 3-layer fixed-δ fields and for
slimmed seeds. Returns `nothing` if the file cannot be loaded.
Injected into means_db through `MEANS_DB_STREAM_FN` (means_db does not depend on the engine).
"""
function _means_db_load_stream(dir_path::AbstractString, obs, vec_index::Union{Int,Nothing};
                               cluster::AbstractString="")
    field = Symbol(obs)
    if !(field in fieldnames(MeasureData))
        valid = join(sort(string.(collect(fieldnames(MeasureData)))), ", ")
        error("get_stream: `$obs` is not a raw MeasureData field, so it has no " *
              "per-measurement stream. Derived observables (stiffness, " *
              "resistivity_tensor, conductivity_*, drude_*) exist only as means — " *
              "use get_observable/get_curve for those. Valid stream fields:\n  $valid")
    end
    if field in _STREAMLESS_MEAN_OBS
        error("get_stream: `$obs` is stored already averaged over measurements " *
              "(mean_obs in load_results), so there is no per-measurement stream — " *
              "only its mean. Use get_observable(; obs=:$field, ...).")
    end

    data_dir = fix_cluster_path(String(dir_path); cluster=cluster)
    if _slimmed_path(data_dir) !== nothing
        error("get_stream: $data_dir was slimmed, so its raw per-measurement stream is " *
              "gone. results_slim.jld2 there keeps a binned version under binned/$obs " *
              "(see meta/bin_size); its means are unchanged in means_db.")
    end
    data = nothing
    try
        data = load_results(data_dir, to_load_checkpoint=false)
    catch e
        @warn "get_stream: failed to load $data_dir" exception=e
        return nothing
    end
    data === nothing && return nothing

    col = getfield(data.measurement_data, field)
    nm  = min(data.current_state.num_current_measurments, length(col))
    nm < 1 && return nothing

    stream = if eltype(col) <: AbstractVector
        if col[1] isa AbstractVector && !isempty(col[1]) && col[1][1] isa AbstractVector
            error("get_stream: `$obs` is a 3-layer (fixed-delta) observable; " *
                  "not supported as a scalar stream.")
        end
        idx = vec_index === nothing ? 1 : vec_index
        (1 <= idx <= length(col[1])) ||
            error("get_stream: vec_index=$idx out of range for `$obs` " *
                  "($(length(col[1])) components).")
        [v[idx] for v in col[1:nm]]
    else
        col[1:nm]
    end
    return (stream, nm, data_dir)
end
MEANS_DB_STREAM_FN[] = _means_db_load_stream

"""
    _means_db_read_simdata(dir_path, key; cluster="")

Read ONE field `key` of a seed's stored `sim_data` (from results.jld2, or
results_slim.jld2 if slimmed) without loading the measurements; `nothing` if absent.
Used by means_db's `get_mu`/`get_param` through `MEANS_DB_SIMDATA_FN`.
"""
function _means_db_read_simdata(dir_path::AbstractString, key; cluster::AbstractString="")
    dir = fix_cluster_path(String(dir_path); cluster=cluster)
    # a slimmed seed keeps sim_data verbatim in results_slim.jld2
    results = something(_slimmed_path(dir), joinpath(dir, "results.jld2"))
    isfile(results) || return nothing
    return _open_jld2(results) do f
        haskey(f, "sim_data") || return nothing
        sd = f["sim_data"]
        (sd isa AbstractDict && haskey(sd, string(key))) ? sd[string(key)] : nothing
    end
end
MEANS_DB_SIMDATA_FN[] = _means_db_read_simdata

"""
    get_means(data::LatticeData; std_scheme="correlation", no_avg_obs=false,
              fit_stiffness=false, complex_corr_mode="reim_max") -> Dict{String,Any}

Mean and error of every observable of ONE seed, from its measurement streams.

Each entry is a Dict with `"value"`, `"error"`, `"correlation_time"`,
`"num_uncorrelated_measurements"` and `"num_measurements"`. Errors of single streams come
from `observable_mean` (blocking with `std_scheme`, see statistics.jl); `complex_corr_mode`
says how the correlation time of a complex stream is taken.

Entries:
- every raw stream of `MeasureData` in the list below, under its own name
  (`U4` is stored as `"K0_action"`, with `"U4"` as an alias);
- `"stiffness"` = -¼(<Kx> + Λxx(qx=0, qy→0)): Λxx at the two smallest qy extrapolated
  linearly to qy = 0 (quadratic fit in q if `fit_stiffness`), bootstrap error (`BT_stiffness`);
  `"stiffness_lowest_q"` uses the smallest qy only, `"stiffness_q0"` the q = 0 stream
  `Lxx_q0`, and `"stiffness_y"`, `"stiffness_y_lowest_q"` the same along y;
- `"Lxx_anamoly"` = -(<Kx> + Λxx(qx→0, qy=0)), the longitudinal limit, which should vanish
  (gauge invariance); a check of the current measurement;
- `"Ps"` = 2<DdD> - <n> + 1, the pairing structure factor, with the DdD–n covariance in its error;
- `"S_Delta_Q0"` = <|Δ_mean|²>, the q = 0 structure factor of the pairing field;
- fixed-broadening (δ) conductivity at each δ in `delta_values_fixed`:
  `"conductivity_regular_ab_fixed"` = Im Λab(ω)/ω, `"Lxx_D_fixed"` = Re Λxx(ω),
  `"resistivity_tensor_fixed"` (inverse of the 2×2 conductivity tensor, bootstrap) and
  `"Drude_weight_fixed"` = -π(<Kx> + Re Λxx(ω₁)); with ≥ 3 δ values each is also
  extrapolated to δ → 0 (`extrapolate_observable_delta_to_zero`);
- `"drude_dc_ab"` (one value per Δω window) = -π(<K> + L_D_dc) on the diagonal, -π L_D_dc off it;
- `"drude_direct_ab"` = the same from the direct ω→0, δ→0 observable `L_D_direct`;
- `"resistivity_dc_tensor"` (one per Δω window) from `conductivity_dc_ab`.

With `no_avg_obs=true`, `"LDOS"` is copied as stored (already averaged in the run; error 0).
If the seed has no measurements yet, returns a message string instead of a Dict.
"""
function get_means(data::LatticeData; std_scheme::String="correlation", no_avg_obs=false, fit_stiffness=false, complex_corr_mode::String="reim_max")
    measurments = data.measurement_data
    num_current_measurment = data.current_state.num_current_measurments
    if num_current_measurment < 1
        return "No measurements available before thermalization."
    end
    means_dict = Dict{String, Any}()
    measure_dwave_observables = is_dwave_pairing(data.sim_data.pairing_symmetry)

    # normal obs
    observable_list = [:n, :Lxx_qx, :Lxx_qy, :Kx, :Ky, :DOS, :DdD, :Ps_l, :Phase_Corr, :U4, :Delta_MC_abs, :Delta_MC, :Delta_operator, :chi_spin, :chi_charge,
                        :energy_spacing_mean, :delta_mean_diff_low,
                        :Lxx_w_fixed, :Lxy_w_fixed, :Lyx_w_fixed, :Lyy_w_fixed,
                        :Lxy_qx, :Lxy_qy, :Lyx_qx, :Lyx_qy, :Lyy_qx, :Lyy_qy,
                        :Lxx_q0, :Lxy_q0, :Lyx_q0, :Lyy_q0,
                        # DC conductivities
                        :conductivity_dc_xx, :conductivity_dc_xy, :conductivity_dc_yx, :conductivity_dc_yy,
                        # L_D DC (current-current correlator, integrated formula)
                        :L_D_dc_xx, :L_D_dc_xy, :L_D_dc_yx, :L_D_dc_yy,
                        # L_D direct (current-current correlator, ω→0 δ→0 closed form)
                        :L_D_direct_xx, :L_D_direct_xy, :L_D_direct_yx, :L_D_direct_yy,
                        # density-channel (phi) observables
                        :phi_mean, :phi_abs_mean, :phi_Q, :m_cdw]
    if measure_dwave_observables
        append!(observable_list, [:Phase_Corr_dwave_x, :Phase_Corr_dwave_y, :Phase_Corr_mixed])
    end

    for observable in observable_list
        output_key = observable == :U4 ? "K0_action" : "$observable"
        if hasfield(typeof(measurments), observable)
            obs_data = getfield(measurments, observable)
            if isa(obs_data, AbstractVector) && isempty(obs_data)
                continue  # skip fields that exist but have no data (e.g. old simulations)
            end
            mean_val, err = observable_mean(obs_data, num_current_measurment; std_scheme=std_scheme)
            means_dict[output_key] = Dict{String, Any}()
            means_dict[output_key]["value"] = mean_val
            means_dict[output_key]["error"] = err
            # Add correlation time and related info
            τ = find_correlation_time(obs_data[1:num_current_measurment]; complex_mode=complex_corr_mode)
            means_dict[output_key]["correlation_time"] = τ

            # Handle num_uncorr for different correlation time types (scalar, array, or vector of vectors)
            if isa(τ, Vector) && !isempty(τ) && isa(τ[1], Vector)
                # For 3D fixed_delta observables: vector of vectors (one per delta)
                num_uncorr = [Int.(floor.(num_current_measurment ./ (maximum(tau_array) == 0 ? 1 : maximum(tau_array)))) for tau_array in τ]
            elseif isa(τ, AbstractArray)
                # For vector observables: array of correlation times
                num_uncorr = Int.(floor.(num_current_measurment ./ ifelse.(τ .== 0, 1, τ)))
            else
                # For scalar observables: single correlation time
                num_uncorr = Int(floor(num_current_measurment / (τ == 0 ? 1 : τ)))
            end

            means_dict[output_key]["num_uncorrelated_measurements"] = num_uncorr

            
            means_dict[output_key]["num_measurements"] = num_current_measurment
            if observable == :U4
                means_dict["U4"] = deepcopy(means_dict[output_key])
            end
        else
            @warn "Observable $observable not found in measurements."
            means_dict[output_key] = "No data for observable $observable"
            if observable == :U4
                means_dict["U4"] = means_dict[output_key]
            end
        end
    end

    add_legacy_means_aliases!(means_dict, measurments, num_current_measurment; std_scheme=std_scheme)

    # stiffness calculation
    if hasfield(typeof(measurments), :Kx) && hasfield(typeof(measurments), :Lxx_qy)
        kx = getfield(measurments, :Kx)
        lxx_qy = getfield(measurments, :Lxx_qy)
        # Standard stiffness (fit/extrapolate)
        stiff_val, stiff_err = BT_stiffness(kx, lxx_qy; fit=fit_stiffness)
        means_dict["stiffness"] = Dict{String, Any}()
        means_dict["stiffness"]["value"] = stiff_val
        means_dict["stiffness"]["error"] = stiff_err
        means_dict["stiffness"]["correlation_time"] = maximum([means_dict["Kx"]["correlation_time"], maximum(means_dict["Lxx_qy"]["correlation_time"])] )
        means_dict["stiffness"]["num_uncorrelated_measurements"] = minimum([means_dict["Kx"]["num_uncorrelated_measurements"], minimum(means_dict["Lxx_qy"]["num_uncorrelated_measurements"])] )
        means_dict["stiffness"]["num_measurements"] = num_current_measurment

        # Stiffness using only lowest q
        stiff_val_lowq, stiff_err_lowq = BT_stiffness(kx, lxx_qy; fit=fit_stiffness, lowest_q_only=true)
        means_dict["stiffness_lowest_q"] = Dict{String, Any}()
        means_dict["stiffness_lowest_q"]["value"] = stiff_val_lowq
        means_dict["stiffness_lowest_q"]["error"] = stiff_err_lowq
        means_dict["stiffness_lowest_q"]["correlation_time"] = means_dict["stiffness"]["correlation_time"]
        means_dict["stiffness_lowest_q"]["num_uncorrelated_measurements"] = means_dict["stiffness"]["num_uncorrelated_measurements"]
        means_dict["stiffness_lowest_q"]["num_measurements"] = num_current_measurment
    end

    # stiffness from the strict q=0, w=0 channel: -0.25*(Kx + Λxx(q=0))
    if hasfield(typeof(measurments), :Lxx_q0) && hasfield(typeof(measurments), :Kx)
        lxx_q0 = getfield(measurments, :Lxx_q0)
        if !(isa(lxx_q0, AbstractVector) && isempty(lxx_q0))
            kx = getfield(measurments, :Kx)
            stiff_val_q0, stiff_err_q0 = BT_stiffness_scalar(kx, lxx_q0)
            means_dict["stiffness_q0"] = Dict{String, Any}()
            means_dict["stiffness_q0"]["value"] = stiff_val_q0
            means_dict["stiffness_q0"]["error"] = stiff_err_q0
            means_dict["stiffness_q0"]["correlation_time"] = maximum([means_dict["Kx"]["correlation_time"], means_dict["Lxx_q0"]["correlation_time"]])
            means_dict["stiffness_q0"]["num_uncorrelated_measurements"] = minimum([means_dict["Kx"]["num_uncorrelated_measurements"], means_dict["Lxx_q0"]["num_uncorrelated_measurements"]])
            means_dict["stiffness_q0"]["num_measurements"] = num_current_measurment
        end
    end

    # stiffness calculation in y direction
    if hasfield(typeof(measurments), :Ky) && hasfield(typeof(measurments), :Lyy_qx)
        ky = getfield(measurments, :Ky)
        lyy_qx = getfield(measurments, :Lyy_qx)
        stiff_val_y, stiff_err_y = BT_stiffness(ky, lyy_qx; fit=fit_stiffness)
        means_dict["stiffness_y"] = Dict{String, Any}()
        means_dict["stiffness_y"]["value"] = stiff_val_y
        means_dict["stiffness_y"]["error"] = stiff_err_y
        means_dict["stiffness_y"]["correlation_time"] = maximum([means_dict["Ky"]["correlation_time"], maximum(means_dict["Lyy_qx"]["correlation_time"])] )
        means_dict["stiffness_y"]["num_uncorrelated_measurements"] = minimum([means_dict["Ky"]["num_uncorrelated_measurements"], minimum(means_dict["Lyy_qx"]["num_uncorrelated_measurements"])] )
        means_dict["stiffness_y"]["num_measurements"] = num_current_measurment

        # Stiffness_y using only lowest q
        stiff_val_y_lowq, stiff_err_y_lowq = BT_stiffness(ky, lyy_qx; fit=fit_stiffness, lowest_q_only=true)
        means_dict["stiffness_y_lowest_q"] = Dict{String, Any}()
        means_dict["stiffness_y_lowest_q"]["value"] = stiff_val_y_lowq
        means_dict["stiffness_y_lowest_q"]["error"] = stiff_err_y_lowq
        means_dict["stiffness_y_lowest_q"]["correlation_time"] = means_dict["stiffness_y"]["correlation_time"]
        means_dict["stiffness_y_lowest_q"]["num_uncorrelated_measurements"] = means_dict["stiffness_y"]["num_uncorrelated_measurements"]
        means_dict["stiffness_y_lowest_q"]["num_measurements"] = num_current_measurment
    end

    # Longitude Current Correlation Anamoly
    if hasfield(typeof(measurments), :Lxx_qx) && hasfield(typeof(measurments), :Kx)
        lxx_qx = getfield(measurments, :Lxx_qx)
        kx = getfield(measurments, :Kx)
        lxx_anamoly, lxx_anamoly_err = BT_stiffness(kx, lxx_qx; fit=fit_stiffness)
        means_dict["Lxx_anamoly"] = Dict{String, Any}()
        means_dict["Lxx_anamoly"]["value"] = 4*lxx_anamoly
        means_dict["Lxx_anamoly"]["error"] = 4*lxx_anamoly_err
        means_dict["Lxx_anamoly"]["correlation_time"] = maximum([means_dict["Kx"]["correlation_time"], maximum(means_dict["Lxx_qx"]["correlation_time"])])
        means_dict["Lxx_anamoly"]["num_uncorrelated_measurements"] = minimum([means_dict["Kx"]["num_uncorrelated_measurements"], minimum(means_dict["Lxx_qx"]["num_uncorrelated_measurements"])])
        means_dict["Lxx_anamoly"]["num_measurements"] = num_current_measurment
    end

    # Ps calculation
    if hasfield(typeof(measurments), :DdD) && hasfield(typeof(measurments), :n)
        (DdD_mean, DdD_var), (n_mean, n_var), obs_cov = get_obs_cov(measurments, :DdD, :n)
        means_dict["Ps"] = Dict{String, Any}()
        means_dict["Ps"]["value"] =  2*DdD_mean - n_mean + 1
        err2 = 4*DdD_var + n_var - 4*obs_cov
        err2 < 0 && (err2 = 0.0)  # prevent negative error squared due to numerical issues
        means_dict["Ps"]["error"] =  sqrt(err2)
        means_dict["Ps"]["correlation_time"] = maximum([means_dict["DdD"]["correlation_time"], means_dict["n"]["correlation_time"]])
        means_dict["Ps"]["num_uncorrelated_measurements"] = minimum([means_dict["DdD"]["num_uncorrelated_measurements"], means_dict["n"]["num_uncorrelated_measurements"]])
        means_dict["Ps"]["num_measurements"] = num_current_measurment
    end

    # S(Q=0) structure factor of the HS pairing (Delta) field.
    #   S(q) = (1/N^2) Σ_ij <Δ_i* Δ_j> e^{iq·(r_i-r_j)}   (Tarat & Majumdar 2015)
    # At q=0 this collapses to <|(1/N) Σ_i Δ_i|^2> = <|Δ_mean|^2>, i.e. the second
    # moment of the already-stored per-measurement Delta_mean stream. This is the
    # HS-field analogue of Ps (which uses the quantum operator c↑c↓ instead of the
    # auxiliary field), and carries Tarat's 1/N^2 normalization, so it is directly
    # comparable in scale to means_dict["Ps"].
    if hasfield(typeof(measurments), :Delta_mean)
        dm = getfield(measurments, :Delta_mean)
        if !(isa(dm, AbstractVector) && isempty(dm))
            # per-measurement |Δ_mean|^2 stream (Float64); observable_mean slices to num_current_measurment
            s0_stream = abs2.(dm)
            mean_val, err = observable_mean(s0_stream, num_current_measurment; std_scheme=std_scheme)
            τ = find_correlation_time(s0_stream[1:num_current_measurment]; complex_mode=complex_corr_mode)
            means_dict["S_Delta_Q0"] = Dict{String, Any}()
            means_dict["S_Delta_Q0"]["value"] = mean_val
            means_dict["S_Delta_Q0"]["error"] = err
            means_dict["S_Delta_Q0"]["correlation_time"] = τ
            means_dict["S_Delta_Q0"]["num_uncorrelated_measurements"] =
                Int(floor(num_current_measurment / (τ == 0 ? 1 : τ)))
            means_dict["S_Delta_Q0"]["num_measurements"] = num_current_measurment
        end
    end

    # Derive conductivity-related quantities from L_w_fixed
    if hasfield(typeof(measurments), :Lxx_w_fixed) && hasfield(typeof(measurments), :Lxy_w_fixed) && hasfield(typeof(measurments), :Lyx_w_fixed) && hasfield(typeof(measurments), :Lyy_w_fixed)
        w_range = data.sim_data.w_conductivity_range
        lxx_w_fixed = getfield(measurments, :Lxx_w_fixed)
        lxy_w_fixed = getfield(measurments, :Lxy_w_fixed)
        lyx_w_fixed = getfield(measurments, :Lyx_w_fixed)
        lyy_w_fixed = getfield(measurments, :Lyy_w_fixed)

        # Derive conductivity_regular = imag(L_w)/w for all measurements
        cond_xx_fixed = [[imag.(m[d]) ./ w_range for d in eachindex(m)] for m in lxx_w_fixed]
        cond_xy_fixed = [[imag.(m[d]) ./ w_range for d in eachindex(m)] for m in lxy_w_fixed]
        cond_yx_fixed = [[imag.(m[d]) ./ w_range for d in eachindex(m)] for m in lyx_w_fixed]
        cond_yy_fixed = [[imag.(m[d]) ./ w_range for d in eachindex(m)] for m in lyy_w_fixed]
        lxx_d_fixed = [[real.(m[d]) for d in eachindex(m)] for m in lxx_w_fixed]

        # Store derived conductivity_regular_*_fixed means
        for (cond_name, cond_data) in [("conductivity_regular_xx_fixed", cond_xx_fixed),
                                        ("conductivity_regular_xy_fixed", cond_xy_fixed),
                                        ("conductivity_regular_yx_fixed", cond_yx_fixed),
                                        ("conductivity_regular_yy_fixed", cond_yy_fixed)]
            mean_val, err = observable_mean(cond_data, num_current_measurment; std_scheme=std_scheme)
            τ = find_correlation_time(cond_data[1:num_current_measurment]; complex_mode=complex_corr_mode)
            if isa(τ, Vector) && !isempty(τ) && isa(τ[1], Vector)
                num_uncorr = [Int.(floor.(num_current_measurment ./ (maximum(tau_array) == 0 ? 1 : maximum(tau_array)))) for tau_array in τ]
            elseif isa(τ, AbstractArray)
                num_uncorr = Int.(floor.(num_current_measurment ./ ifelse.(τ .== 0, 1, τ)))
            else
                num_uncorr = Int(floor(num_current_measurment / (τ == 0 ? 1 : τ)))
            end
            means_dict[cond_name] = Dict{String, Any}(
                "value" => mean_val, "error" => err,
                "correlation_time" => τ, "num_measurements" => num_current_measurment,
                "num_uncorrelated_measurements" => num_uncorr
            )
        end

        # Store Lxx_D_fixed(omega) = Re[Lxx_w_fixed(omega)] for each fixed delta.
        lxx_d_mean, lxx_d_err = observable_mean(lxx_d_fixed, num_current_measurment; std_scheme=std_scheme)
        tau_lxx_d = find_correlation_time(lxx_d_fixed[1:num_current_measurment]; complex_mode=complex_corr_mode)
        if isa(tau_lxx_d, Vector) && !isempty(tau_lxx_d) && isa(tau_lxx_d[1], Vector)
            num_uncorr_lxx_d = [Int.(floor.(num_current_measurment ./ (maximum(tau_array) == 0 ? 1 : maximum(tau_array)))) for tau_array in tau_lxx_d]
        elseif isa(tau_lxx_d, AbstractArray)
            num_uncorr_lxx_d = Int.(floor.(num_current_measurment ./ ifelse.(tau_lxx_d .== 0, 1, tau_lxx_d)))
        else
            num_uncorr_lxx_d = Int(floor(num_current_measurment / (tau_lxx_d == 0 ? 1 : tau_lxx_d)))
        end
        means_dict["Lxx_D_fixed"] = Dict{String, Any}(
            "value" => lxx_d_mean,
            "error" => lxx_d_err,
            "correlation_time" => tau_lxx_d,
            "num_measurements" => num_current_measurment,
            "num_uncorrelated_measurements" => num_uncorr_lxx_d,
        )

        # Resistivity tensor for fixed_delta versions
        if !isempty(cond_xx_fixed) && !isempty(cond_xx_fixed[1])
            num_deltas = length(cond_xx_fixed[1])
            means_dict["resistivity_tensor_fixed"] = Dict{String, Any}()
            means_dict["resistivity_tensor_fixed"]["value"] = Vector{Any}(undef, num_deltas)
            means_dict["resistivity_tensor_fixed"]["error"] = Vector{Any}(undef, num_deltas)
            means_dict["resistivity_tensor_fixed"]["correlation_time"] = Vector{Any}(undef, num_deltas)
            means_dict["resistivity_tensor_fixed"]["num_uncorrelated_measurements"] = Vector{Any}(undef, num_deltas)
            means_dict["resistivity_tensor_fixed"]["warnings"] = Dict{Int, String}()

            for delta_idx in 1:num_deltas
                cond_xx_delta = [measurement[delta_idx] for measurement in cond_xx_fixed if delta_idx <= length(measurement)]
                cond_xy_delta = [measurement[delta_idx] for measurement in cond_xy_fixed if delta_idx <= length(measurement)]
                cond_yx_delta = [measurement[delta_idx] for measurement in cond_yx_fixed if delta_idx <= length(measurement)]
                cond_yy_delta = [measurement[delta_idx] for measurement in cond_yy_fixed if delta_idx <= length(measurement)]

                if !isempty(cond_xx_delta)
                    try
                        (res_tensor, res_tensor_err, correlation_time, num_blocks) = BT_resistivity_tensor(cond_xx_delta, cond_xy_delta, cond_yx_delta, cond_yy_delta; B=1000)
                        means_dict["resistivity_tensor_fixed"]["value"][delta_idx] = res_tensor
                        means_dict["resistivity_tensor_fixed"]["error"][delta_idx] = res_tensor_err
                        means_dict["resistivity_tensor_fixed"]["correlation_time"][delta_idx] = correlation_time
                        means_dict["resistivity_tensor_fixed"]["num_uncorrelated_measurements"][delta_idx] = num_blocks
                    catch e
                        means_dict["resistivity_tensor_fixed"]["value"][delta_idx] = nothing
                        means_dict["resistivity_tensor_fixed"]["error"][delta_idx] = nothing
                        means_dict["resistivity_tensor_fixed"]["correlation_time"][delta_idx] = nothing
                        means_dict["resistivity_tensor_fixed"]["num_uncorrelated_measurements"][delta_idx] = nothing
                        means_dict["resistivity_tensor_fixed"]["warnings"][delta_idx] =
                            "failed at delta_idx=$(delta_idx): $(sprint(showerror, e))"
                    end
                else
                    means_dict["resistivity_tensor_fixed"]["value"][delta_idx] = nothing
                    means_dict["resistivity_tensor_fixed"]["error"][delta_idx] = nothing
                    means_dict["resistivity_tensor_fixed"]["correlation_time"][delta_idx] = nothing
                    means_dict["resistivity_tensor_fixed"]["num_uncorrelated_measurements"][delta_idx] = nothing
                    means_dict["resistivity_tensor_fixed"]["warnings"][delta_idx] =
                        "skipped at delta_idx=$(delta_idx): missing conductivity data"
                end
            end
            means_dict["resistivity_tensor_fixed"]["num_measurements"] = num_current_measurment
            means_dict["resistivity_tensor_fixed"]["num_failed"] =
                length(means_dict["resistivity_tensor_fixed"]["warnings"])
        end

        # Drude weight for fixed_delta versions (derived from L_w_fixed)
        kx = getfield(measurments, :Kx)
        if !isempty(kx) && !isempty(lxx_w_fixed) && !isempty(lxx_w_fixed[1])
            num_deltas = length(lxx_w_fixed[1])
            means_dict["Drude_weight_fixed"] = Dict{String, Any}()
            means_dict["Drude_weight_fixed"]["value"] = Vector{Any}(undef, num_deltas)
            means_dict["Drude_weight_fixed"]["error"] = Vector{Any}(undef, num_deltas)
            means_dict["Drude_weight_fixed"]["correlation_time"] = Vector{Any}(undef, num_deltas)
            means_dict["Drude_weight_fixed"]["num_uncorrelated_measurements"] = Vector{Any}(undef, num_deltas)

            for delta_idx in 1:num_deltas
                # Derive L_D_fixed = real(L_w_fixed[i][d][1])
                lxx_d_delta = [real(m[delta_idx][1]) for m in lxx_w_fixed if delta_idx <= length(m)]

                if !isempty(lxx_d_delta)
                    (kx_mean, kx_var), (lxx_d_mean, lxx_d_var), obs_cov = get_obs_cov(kx, lxx_d_delta)
                    means_dict["Drude_weight_fixed"]["value"][delta_idx] = -pi * (kx_mean + lxx_d_mean)
                    err2 = kx_var + lxx_d_var + 2*obs_cov
                    err2 < 0 && (err2 = 0.0)
                    means_dict["Drude_weight_fixed"]["error"][delta_idx] = pi * sqrt(err2)
                    tau_kx = find_correlation_time(kx; complex_mode=complex_corr_mode)
                    tau_lxx = find_correlation_time(lxx_d_delta; complex_mode=complex_corr_mode)
                    means_dict["Drude_weight_fixed"]["correlation_time"][delta_idx] = maximum([tau_kx, tau_lxx])
                    means_dict["Drude_weight_fixed"]["num_uncorrelated_measurements"][delta_idx] = Int(floor(min(num_current_measurment / (tau_kx == 0 ? 1 : tau_kx), num_current_measurment / (tau_lxx == 0 ? 1 : tau_lxx))))
                else
                    means_dict["Drude_weight_fixed"]["value"][delta_idx] = nothing
                    means_dict["Drude_weight_fixed"]["error"][delta_idx] = nothing
                    means_dict["Drude_weight_fixed"]["correlation_time"][delta_idx] = nothing
                    means_dict["Drude_weight_fixed"]["num_uncorrelated_measurements"][delta_idx] = nothing
                end
            end
            means_dict["Drude_weight_fixed"]["num_measurements"] = num_current_measurment
        end
    end

    # Drude weight DC computation from L_D_dc_* observables
    # D_dc_xx[dw] = -π(Kx + L_D_dc_xx[dw]),  D_dc_yy[dw] = -π(Ky + L_D_dc_yy[dw])
    # D_dc_xy[dw] = -π * L_D_dc_xy[dw],       D_dc_yx[dw] = -π * L_D_dc_yx[dw]
    if hasfield(typeof(measurments), :L_D_dc_xx) && hasfield(typeof(measurments), :Kx)
        kx = getfield(measurments, :Kx)
        ky = hasfield(typeof(measurments), :Ky) ? getfield(measurments, :Ky) : nothing

        for (ld_sym, drude_key, kinetic_data, is_diagonal) in [
            (:L_D_dc_xx, "drude_dc_xx", kx, true),
            (:L_D_dc_yy, "drude_dc_yy", ky, true),
            (:L_D_dc_xy, "drude_dc_xy", nothing, false),
            (:L_D_dc_yx, "drude_dc_yx", nothing, false),
        ]
            if !hasfield(typeof(measurments), ld_sym)
                continue
            end
            ld_data = getfield(measurments, ld_sym)
            if isempty(ld_data) || length(ld_data) < num_current_measurment
                continue
            end
            num_dw = length(ld_data[1])
            means_dict[drude_key] = Dict{String, Any}()
            means_dict[drude_key]["value"] = Vector{Any}(undef, num_dw)
            means_dict[drude_key]["error"] = Vector{Any}(undef, num_dw)
            means_dict[drude_key]["correlation_time"] = Vector{Any}(undef, num_dw)
            means_dict[drude_key]["num_uncorrelated_measurements"] = Vector{Any}(undef, num_dw)

            for dw_idx in 1:num_dw
                ld_dw = [m[dw_idx] for m in ld_data[1:num_current_measurment]]

                if is_diagonal && kinetic_data !== nothing
                    # D = -π(K + L_D)
                    (k_mean, k_var), (ld_mean, ld_var), obs_cov = get_obs_cov(kinetic_data[1:num_current_measurment], ld_dw)
                    means_dict[drude_key]["value"][dw_idx] = -pi * (k_mean + ld_mean)
                    err2 = k_var + ld_var + 2*obs_cov
                    err2 < 0 && (err2 = 0.0)
                    means_dict[drude_key]["error"][dw_idx] = pi * sqrt(err2)
                    tau_k = find_correlation_time(kinetic_data[1:num_current_measurment]; complex_mode=complex_corr_mode)
                    tau_ld = find_correlation_time(ld_dw; complex_mode=complex_corr_mode)
                    means_dict[drude_key]["correlation_time"][dw_idx] = maximum([tau_k, tau_ld])
                    means_dict[drude_key]["num_uncorrelated_measurements"][dw_idx] = Int(floor(min(
                        num_current_measurment / (tau_k == 0 ? 1 : tau_k),
                        num_current_measurment / (tau_ld == 0 ? 1 : tau_ld))))
                else
                    # D = -π * L_D (off-diagonal)
                    ld_mean, ld_err = observable_mean(ld_dw, num_current_measurment; std_scheme=std_scheme)
                    means_dict[drude_key]["value"][dw_idx] = -pi * ld_mean
                    means_dict[drude_key]["error"][dw_idx] = pi * ld_err
                    tau_ld = find_correlation_time(ld_dw; complex_mode=complex_corr_mode)
                    means_dict[drude_key]["correlation_time"][dw_idx] = tau_ld
                    means_dict[drude_key]["num_uncorrelated_measurements"][dw_idx] = Int(floor(
                        num_current_measurment / (tau_ld == 0 ? 1 : tau_ld)))
                end
            end
            means_dict[drude_key]["num_measurements"] = num_current_measurment
        end
    end

    # Drude weight from the direct ω→0, δ→0 observable L_D_direct_*
    # D_direct_xx = -π(Kx + L_D_direct_xx),  D_direct_yy = -π(Ky + L_D_direct_yy)
    # D_direct_xy = -π * L_D_direct_xy,       D_direct_yx = -π * L_D_direct_yx
    # (each L_D_direct_* is a single real scalar per measurement, no Δω dimension)
    if hasfield(typeof(measurments), :L_D_direct_xx) && hasfield(typeof(measurments), :Kx)
        kx = getfield(measurments, :Kx)
        ky = hasfield(typeof(measurments), :Ky) ? getfield(measurments, :Ky) : nothing

        for (ld_sym, drude_key, kinetic_data, is_diagonal) in [
            (:L_D_direct_xx, "drude_direct_xx", kx, true),
            (:L_D_direct_yy, "drude_direct_yy", ky, true),
            (:L_D_direct_xy, "drude_direct_xy", nothing, false),
            (:L_D_direct_yx, "drude_direct_yx", nothing, false),
        ]
            if !hasfield(typeof(measurments), ld_sym)
                continue
            end
            ld_data = getfield(measurments, ld_sym)
            if isempty(ld_data) || length(ld_data) < num_current_measurment
                continue
            end
            ld_vals = ld_data[1:num_current_measurment]
            means_dict[drude_key] = Dict{String, Any}()

            if is_diagonal && kinetic_data !== nothing
                # D = -π(K + L_D_direct)
                (k_mean, k_var), (ld_mean, ld_var), obs_cov = get_obs_cov(kinetic_data[1:num_current_measurment], ld_vals)
                means_dict[drude_key]["value"] = -pi * (k_mean + ld_mean)
                err2 = k_var + ld_var + 2*obs_cov
                err2 < 0 && (err2 = 0.0)
                means_dict[drude_key]["error"] = pi * sqrt(err2)
                tau_k = find_correlation_time(kinetic_data[1:num_current_measurment]; complex_mode=complex_corr_mode)
                tau_ld = find_correlation_time(ld_vals; complex_mode=complex_corr_mode)
                means_dict[drude_key]["correlation_time"] = maximum([tau_k, tau_ld])
                means_dict[drude_key]["num_uncorrelated_measurements"] = Int(floor(min(
                    num_current_measurment / (tau_k == 0 ? 1 : tau_k),
                    num_current_measurment / (tau_ld == 0 ? 1 : tau_ld))))
            else
                # D = -π * L_D_direct (off-diagonal)
                ld_mean, ld_err = observable_mean(ld_vals, num_current_measurment; std_scheme=std_scheme)
                means_dict[drude_key]["value"] = -pi * ld_mean
                means_dict[drude_key]["error"] = pi * ld_err
                tau_ld = find_correlation_time(ld_vals; complex_mode=complex_corr_mode)
                means_dict[drude_key]["correlation_time"] = tau_ld
                means_dict[drude_key]["num_uncorrelated_measurements"] = Int(floor(
                    num_current_measurment / (tau_ld == 0 ? 1 : tau_ld)))
            end
            means_dict[drude_key]["num_measurements"] = num_current_measurment
        end
    end

    # DC Resistivity tensor calculation (from DC conductivity components)
    # Compute one resistivity tensor per Δω window (dw index).
    dc_cond_fields = [:conductivity_dc_xx, :conductivity_dc_xy, :conductivity_dc_yx, :conductivity_dc_yy]
    if all(f -> hasfield(typeof(measurments), f), dc_cond_fields)
        cond_dc_xx = getfield(measurments, :conductivity_dc_xx)
        cond_dc_xy = getfield(measurments, :conductivity_dc_xy)
        cond_dc_yx = getfield(measurments, :conductivity_dc_yx)
        cond_dc_yy = getfield(measurments, :conductivity_dc_yy)

        if !isempty(cond_dc_xx) && all(length(arr) >= num_current_measurment for arr in (cond_dc_xx, cond_dc_xy, cond_dc_yx, cond_dc_yy))
            dc_xx = cond_dc_xx[1:num_current_measurment]
            dc_xy = cond_dc_xy[1:num_current_measurment]
            dc_yx = cond_dc_yx[1:num_current_measurment]
            dc_yy = cond_dc_yy[1:num_current_measurment]

            common_len = minimum(vcat(length.(dc_xx), length.(dc_xy), length.(dc_yx), length.(dc_yy)))
            if common_len <= 0
                # No DC windows were measured (e.g., empty dw_values); skip gracefully.
            else
                dc_xx_common = [v[1:common_len] for v in dc_xx]
                dc_xy_common = [v[1:common_len] for v in dc_xy]
                dc_yx_common = [v[1:common_len] for v in dc_yx]
                dc_yy_common = [v[1:common_len] for v in dc_yy]

                means_dict["resistivity_dc_tensor"] = Dict{String, Any}(
                    "value" => Vector{Any}(undef, common_len),
                    "error" => Vector{Any}(undef, common_len),
                    "correlation_time" => Vector{Any}(undef, common_len),
                    "num_uncorrelated_measurements" => Vector{Any}(undef, common_len),
                    "num_measurements" => num_current_measurment,
                    "warnings" => Dict{Int, String}()
                )

                for dw_idx in 1:common_len
                    try
                        (res_tensor, res_tensor_err, correlation_time, num_blocks_dc) = BT_resistivity_tensor(
                            dc_xx_common, dc_xy_common,
                            dc_yx_common, dc_yy_common; B=1000, w_ix=dw_idx)

                        means_dict["resistivity_dc_tensor"]["value"][dw_idx] = res_tensor
                        means_dict["resistivity_dc_tensor"]["error"][dw_idx] = res_tensor_err
                        means_dict["resistivity_dc_tensor"]["correlation_time"][dw_idx] = correlation_time
                        means_dict["resistivity_dc_tensor"]["num_uncorrelated_measurements"][dw_idx] = num_blocks_dc
                    catch e
                        means_dict["resistivity_dc_tensor"]["value"][dw_idx] = nothing
                        means_dict["resistivity_dc_tensor"]["error"][dw_idx] = nothing
                        means_dict["resistivity_dc_tensor"]["correlation_time"][dw_idx] = nothing
                        means_dict["resistivity_dc_tensor"]["num_uncorrelated_measurements"][dw_idx] = nothing
                        means_dict["resistivity_dc_tensor"]["warnings"][dw_idx] =
                            "failed at dw_idx=$(dw_idx): $(sprint(showerror, e))"
                    end
                end
                means_dict["resistivity_dc_tensor"]["num_failed"] =
                    length(means_dict["resistivity_dc_tensor"]["warnings"])
            end
        end
    end

    # No average observables
    if no_avg_obs
        No_avg_obs_list = [:LDOS]
        for obs in No_avg_obs_list
            if hasfield(typeof(measurments), obs)
                means_dict["$obs"] = Dict{String, Any}()
                means_dict["$(obs)"]["value"] = getfield(measurments, obs)
                means_dict["$(obs)"]["error"] = 0.0  # Placeholder for error
            end
        end
    end

    # Delta→0 extrapolation of all delta-dependent observables
    sim_data = data.sim_data
    Lx, Ly = sim_data.Lx, sim_data.Ly
    MLS = 4.0 / (Lx * Ly)

    # Fixed-delta conductivity & Drude weight extrapolation
    delta_fixed = sim_data.delta_values_fixed
    if !isempty(delta_fixed) && length(delta_fixed) >= 3
        fixed_delta_obs = [
            "conductivity_regular_xx_fixed",
            "conductivity_regular_xy_fixed",
            "conductivity_regular_yx_fixed",
            "conductivity_regular_yy_fixed",
            "Lxx_D_fixed",
            "Drude_weight_fixed",
            "resistivity_tensor_fixed",
        ]
        for obs_key in fixed_delta_obs
            if haskey(means_dict, obs_key)
                try
                    extrapolate_observable_delta_to_zero(means_dict, obs_key, delta_fixed; MLS=MLS)
                catch e
                    @warn "Delta→0 extrapolation failed for $obs_key: $e"
                end
            end
        end
    end


    return means_dict
end

"""
    get_obs_cov(a_data, b_data) -> ((mean_a, var_a), (mean_b, var_b), cov_ab)
    get_obs_cov(measurments::MeasureData, a_field::Symbol, b_field::Symbol)

Block both streams with the same block size (the larger of the two correlation times,
and at least length/32) and return the block means, the variances of the means
(from `get_block_means`) and the covariance of the two means. Used to propagate errors of
combinations such as Ps = 2 DdD - n + 1.
"""
function get_obs_cov(measurments::MeasureData, a_field::Symbol, b_field::Symbol)
    a_data = getfield(measurments, a_field)
    b_data = getfield(measurments, b_field)
    return get_obs_cov(a_data, b_data)
end


function get_obs_cov(a_data::AbstractArray, b_data::AbstractArray)
    block_size = round(Int, maximum([BSfromNB(length(a_data), 32), find_correlation_time(a_data), find_correlation_time(b_data)]))
    a_blocks, a_var = get_block_means(a_data, block_size; return_nothing=false)
    b_blocks, b_var = get_block_means(b_data, block_size; return_nothing=false)
    obs_cov = cov(a_blocks, b_blocks)/length(a_blocks)
    return (mean(a_blocks), a_var), (mean(b_blocks), b_var), obs_cov
end

"""
    get_obs_dict(sims) -> Dict{String,Any}

Collect the per-seed results of one parameter point into lists, for averaging.

`sims` is the point's `"simulations"` (seed => seed info with `"means"`). Seeds with no
measurements are skipped. Returns observable => Dict("value" => [...], "error" => [...]),
plus `"num_measures"`, `"mu_list"`, `"seed_list"` and `"convergence_info"` (the μ-search
result of each seed).
"""
function get_obs_dict(sims::Dict{Any, Any})
    list_Dict = Dict{String, Any}(
        "num_measures" => [], 
        "mu_list" => [], 
        "seed_list" => [],
        "convergence_info" => []  # Store convergence data per seed
    )
    for (seed, sim_data) in sims
        if isa(sim_data, String)
            continue  # Skip if the data is a string (e.g., "Failed to load data")
        end
        c_meas = get(sim_data, "num_current_measurments", 0)
        if c_meas == 0
            continue  # Skip if no measurements have been made
        end
        # Collect mu and seed for each simulation
        push!(list_Dict["mu_list"], get(sim_data, "mu", NaN))
        push!(list_Dict["seed_list"], seed)
        
        # Collect convergence info if available
        if haskey(sim_data, "convergence") && isa(sim_data["convergence"], Dict)
            conv = sim_data["convergence"]
            push!(list_Dict["convergence_info"], Dict(
                "seed" => seed,
                "target_n" => get(conv, "target_n", NaN),
                "best_n" => get(conv, "best_n", NaN),
                "best_mu" => get(conv, "best_mu", NaN),
                "converged" => get(conv, "converged", false),
                "n_tol" => get(conv, "n_tol", NaN)
            ))
        end
        
        means = get(sim_data, "means", Dict{String, Any}())
        if !isa(means, String)
            push!(list_Dict["num_measures"], c_meas)
            for (observable, value) in means
                if !haskey(list_Dict, observable)
                    list_Dict[observable] = Dict("value" => [], "error" => [])
                end
                if value === nothing || !isa(value, AbstractDict) ||
                   value["value"] === nothing || value["error"] === nothing
                    @warn "Skipping seed=$seed obs=$observable: missing/nothing value or error"
                    continue
                end
                push!(list_Dict[observable]["value"], value["value"])
                push!(list_Dict[observable]["error"], value["error"])
            end
        end
    end
    return list_Dict
end

"""
    average_over_seeds(sim_info, sim_num) -> Dict{String,Any}

Combine the per-seed means of parameter point `sim_num` into one value ± error per observable.

Each scalar, and each component of a vector or matrix observable, is combined with
`obs_seed_mean2(...; method="DL")`: the DerSimonian–Laird random-effects mean, which
includes the seed-to-seed (e.g. disorder-realization) spread in the error.
With a single seed its means are returned unchanged. `"LDOS"` is not averaged.
Also adds `"mu_convergence"`: the μ of each seed (mean, std, range) and the μ-search
results (target n, reached n, how many seeds converged).
"""
function average_over_seeds(sim_info::Dict{String, Any}, sim_num::String)
    if sim_num == "updated_at"
        return nothing
    end
    sims = sim_info[sim_num]["simulations"]
    if length(sims) == 0
        return "No simulation data available."
    elseif length(sims) == 1
        if haskey(sims[first(keys(sims))], "means")
            return sims[first(keys(sims))]["means"]
        else
            return "No means data available."
        end
    end
    Out_Dict = Dict{String, Any}()

    # Get the observables from all of the simulation seeds
    list_Dict = get_obs_dict(sims)

    # Store mu information from all seeds in a sub-dictionary
    Out_Dict["mu_convergence"] = Dict{String, Any}()
    if haskey(list_Dict, "mu_list") && !isempty(list_Dict["mu_list"])
        mu_values = filter(!isnan, list_Dict["mu_list"])
        Out_Dict["mu_convergence"]["mu_list"] = list_Dict["mu_list"]  # All mu values per seed
        Out_Dict["mu_convergence"]["seed_list"] = list_Dict["seed_list"]  # Corresponding seeds
        if !isempty(mu_values)
            Out_Dict["mu_convergence"]["mu_mean"] = mean(mu_values)
            Out_Dict["mu_convergence"]["mu_std"] = length(mu_values) > 1 ? std(mu_values) : 0.0
            Out_Dict["mu_convergence"]["mu_range"] = (minimum(mu_values), maximum(mu_values))
        end
    end
    
    # Store n convergence information in the sub-dictionary
    if haskey(list_Dict, "convergence_info") && !isempty(list_Dict["convergence_info"])
        conv_list = list_Dict["convergence_info"]
        
        # Get target_n (should be same for all seeds)
        target_ns = [c["target_n"] for c in conv_list if !isnan(c["target_n"])]
        if !isempty(target_ns)
            Out_Dict["mu_convergence"]["target_n"] = first(target_ns)  # wanted_n
        end
        
        # Collect best_n achieved per seed
        best_ns = [c["best_n"] for c in conv_list if !isnan(c["best_n"])]
        if !isempty(best_ns)
            Out_Dict["mu_convergence"]["best_n_list"] = best_ns
            Out_Dict["mu_convergence"]["best_n_mean"] = mean(best_ns)
            Out_Dict["mu_convergence"]["best_n_std"] = length(best_ns) > 1 ? std(best_ns) : 0.0
        end
        
        # Check how many converged
        converged_flags = [c["converged"] for c in conv_list]
        Out_Dict["mu_convergence"]["num_converged"] = sum(converged_flags)
        Out_Dict["mu_convergence"]["num_seeds"] = length(converged_flags)
        Out_Dict["mu_convergence"]["all_converged"] = all(converged_flags)
        
        # Get n_tol used
        n_tols = [c["n_tol"] for c in conv_list if !isnan(c["n_tol"])]
        if !isempty(n_tols)
            Out_Dict["mu_convergence"]["n_tol"] = first(n_tols)
        end
    end

    # average over all of the simulations

    for (observable, value) in list_Dict
        if observable in ["num_measures", "mu_list", "seed_list", "convergence_info"]
            continue  # Skip tracking keys (already handled above)
        end
        if !haskey(Out_Dict, observable)
            Out_Dict[observable] = Dict()
        end

        if observable == "LDOS"
            continue  # Skip LDOS for now
        end


        if length(value["value"]) > 0
            total_meas = sum(list_Dict["num_measures"])
            element_type = typeof(value["value"][1])
            eleltype = typeof(value["value"][1][1])
            if element_type <: Number
                # Scalar observable case
                mean_value, error_value = obs_seed_mean2(value["value"], value["error"]; method="DL")
            elseif element_type <: AbstractVector{<:Number} || element_type <: AbstractMatrix{<:Number} || eleltype <: Number
                # 2D observable case
                obs_length = length(value["value"][1])
                err_eleltype = typeof(value["error"][1][1])
                mean_value = zeros(eleltype, obs_length)
                error_value = zeros(err_eleltype, obs_length)
                for ix in 1:obs_length
                    try
                        vals = Any[]
                        errs = Any[]
                        for k in eachindex(value["value"])
                            v = value["value"][k]
                            e_v = value["error"][k]
                            (v === nothing || e_v === nothing) && continue
                            vk = v[ix]; ek = e_v[ix]
                            (vk === nothing || ek === nothing) && continue
                            push!(vals, vk); push!(errs, ek)
                        end
                        mean_value[ix], error_value[ix] = obs_seed_mean2(vals, errs; method="DL")
                    catch e
                        @error "[2D] Failed averaging sim_num=$sim_num obs=$observable ix=$ix" exception=(e, catch_backtrace())
                        for (sx, v) in enumerate(value["value"])
                            @error "  seed_idx=$sx typeof(v)=$(typeof(v)) length=$(applicable(length, v) ? length(v) : "N/A") v[ix]=$(try v[ix] catch err; "ERR: $err" end)"
                        end
                        rethrow(e)
                    end
                end
            elseif eleltype <: AbstractVector{<:Number} || eleltype <: AbstractMatrix{<:Number}
                # 3D observable case
                obs_length = length(value["value"][1])
                obs_sub_length = length(value["value"][1][1])
                err_eleleltype = typeof(value["error"][1][1][1])
                eleltype = typeof(value["value"][1][1][1])
                mean_value = zeros(eleltype, obs_length, obs_sub_length)
                error_value = zeros(err_eleleltype, obs_length, obs_sub_length)
                for ix in 1:obs_length
                    for jx in 1:obs_sub_length
                        try
                            vals = Any[]
                            errs = Any[]
                            for k in eachindex(value["value"])
                                v = value["value"][k]
                                e_v = value["error"][k]
                                (v === nothing || e_v === nothing) && continue
                                vix = v[ix]; eix = e_v[ix]
                                (vix === nothing || eix === nothing) && continue
                                vkj = vix[jx]; ekj = eix[jx]
                                (vkj === nothing || ekj === nothing) && continue
                                push!(vals, vkj); push!(errs, ekj)
                            end
                            mean_value[ix, jx], error_value[ix, jx] = obs_seed_mean2(vals, errs; method="DL")
                        catch e
                            @error "[3D] Failed averaging sim_num=$sim_num obs=$observable ix=$ix jx=$jx" exception=(e, catch_backtrace())
                            for (sx, v) in enumerate(value["value"])
                                vix_repr = try
                                    vix = v[ix]
                                    "typeof(v[ix])=$(typeof(vix)) length=$(applicable(length, vix) ? length(vix) : "N/A")"
                                catch err
                                    "ERR indexing v[ix]: $err"
                                end
                                @error "  seed_idx=$sx typeof(v)=$(typeof(v)) length=$(applicable(length, v) ? length(v) : "N/A") $vix_repr"
                            end
                            rethrow(e)
                        end
                    end
                end
            else 
                # Unknown type
                println("Unknown observable type for $observable: $element_type | $eleltype")
                mean_value = NaN
                error_value = NaN
            end

            Out_Dict[observable]["value"] = mean_value
            Out_Dict[observable]["error"] = error_value
        else
            Out_Dict[observable]["value"] = NaN
            Out_Dict[observable]["error"] = NaN
        end


    end
    return Out_Dict
end

"""
    obs_seed_mean2(values, errors; method="DL", num_measures=nothing) -> (mean, error)

Combine independent estimates `values ± errors` (one per seed) into one.

`method`:
- `"DL"` (default): DerSimonian–Laird random-effects mean (`get_x_DL` in statistics.jl);
- `"simple"`: plain mean, error = std/√N;
- `"IV weighted"`: inverse-variance weighted mean;
- `"num_measures weighted"`: mean weighted by `num_measures` (error returned as 0).
Falls back to `"simple"` if all errors are NaN/Inf, any error is 0, or DL gives NaN/Inf.
One value is returned as is; no values give `(NaN, NaN)`.
"""
function obs_seed_mean2(values::AbstractVector, errors::AbstractVector; method="DL", num_measures=nothing)

    if isempty(values) || length(values) == 0
        return (NaN, NaN)
    end
    
    # Single value case - just return it
    if length(values) == 1
        return (values[1], errors[1])
    end

    if isnan.(errors) |> all || isinf.(errors) |> all
        method = "simple"
    end
    
    # Check for zero errors which cause DL method to fail
    if any(iszero, errors)
        method = "simple"
    end

    if method == "simple"
        mean_value = mean(values)
        error_value = std(values) / sqrt(length(values))
        return (mean_value, error_value)

    elseif method == "IV weighted"
        # inverse variance weighted mean; it returns a VARIANCE, so take the root
        (mean_value, variance_value) = inverse_variance_weighted_mean(values, errors)
        return (mean_value, sqrt(variance_value))

    elseif method == "DL"
        mean_value, error_value = get_x_DL(values, errors)
        # Fall back to simple method if DL produces NaN
        if isnan(mean_value) || isnan(error_value) || isinf(mean_value) || isinf(error_value)
            mean_value = mean(values)
            error_value = std(values) / sqrt(length(values))
        end
        return (mean_value, error_value)
        
    elseif method == "num_measures weighted"
        if num_measures === nothing
            return (NaN, NaN)
        end
        total_meas = sum(num_measures)
        mean_value = 0.0
        error_value = 0.0
        for ix in eachindex(values)
            mean_value += values[ix] * num_measures[ix] / total_meas
        end
        return (mean_value, error_value)
    else
        println("Unknown method: $method")
        return (NaN, NaN)
    end
end



"""
    get_sim_filling(sim_info, Lx, Ly=Lx) -> Dict{String,Any}

Filling n(μ) of every `Lx×Ly` point with means, grouped by β:
`"<beta>"` => Dict("mu" => [...], "n" => [...], "n_err" => [...]), sorted by μ.
"""
function get_sim_filling(sim_info::Dict{String, Any}, Lx::Int64, Ly::Int64=Lx)
    if length(sim_info) == 0
        return nothing # Return nothing if no simulations are available
    end
    sim_filling = Dict{String, Any}()
    for (sim_num, sim_data) in sim_info
        if sim_num == "updated_at"
            continue
        end
        sim_Lx, sim_Ly = get_sim_dimensions(sim_data)
        if sim_Lx != Lx || sim_Ly != Ly
            continue  # Skip simulations that do not match the desired lattice size
        end

        if !haskey(sim_data, "means") || typeof(sim_data["means"]) == String
            continue  # Skip if means not available or no measurement data
        end

        # Extract the filling data from the simulation data
        beta = sim_data["beta"]
        if !haskey(sim_filling, "$beta")
            sim_filling["$beta"] = Dict{String, Any}("mu" => [], "n" => [], "n_err" => [])
        end
        beta_dict = sim_filling["$beta"]
        mu = sim_data["mu"]
        means = sim_data["means"]
        n = means["n"]["value"]
        n_err = means["n"]["error"]
        append!(beta_dict["mu"], mu)
        append!(beta_dict["n"], n)
        append!(beta_dict["n_err"], n_err)
        sim_filling["$beta"] = beta_dict
    end

    for (beta, beta_dict) in sim_filling
        mu_list = beta_dict["mu"]
        n_list = beta_dict["n"]
        n_err_list = beta_dict["n_err"]
        sorted_indices = sortperm(mu_list)
        beta_dict["mu"] = mu_list[sorted_indices]
        beta_dict["n"] = n_list[sorted_indices]
        beta_dict["n_err"] = n_err_list[sorted_indices]
        sim_filling[beta] = beta_dict
    end
    return sim_filling
end
"""
    BSfromNB(data_L, num_blocks) -> Int

Block size that splits `data_L` measurements into `num_blocks` blocks. If that gives
blocks shorter than 2, `num_blocks` is halved until it does not (minimum block size 1).
"""
function  BSfromNB(data_L::Int, num_blocks::Int)
    block_size = data_L ÷ num_blocks
    while block_size < 2
        num_blocks ÷= 2
        if num_blocks == 0 
            block_size = 1
            break
        end
        block_size = data_L ÷ num_blocks
    end
    return block_size
end

"""
    BT_resistivity(cond_xx, cond_xy; num_blocks=32, B=1000) -> ((ρxx, err), (ρxy, err))

Bootstrap ρxx and ρxy from conductivity streams, assuming σyy = σxx and σyx = -σxy.
Uses frequency index 2 of each measurement. Superseded by `BT_resistivity_tensor`,
which makes no symmetry assumption.
"""
function BT_resistivity(cond_xx::Vector{Vector{Float64}}, cond_xy::Vector{Vector{Float64}}; num_blocks::Int64=32, B=1000)
    # assuming cond_xx and cond_xy have the same length and are both up to num_current_measurements
    # block the data
    w_ix = 2
    cond_xx_data = hcat(cond_xx...)[w_ix,:]
    cond_xy_data = hcat(cond_xy...)[w_ix,:]
    block_size = Int(round(maximum([BSfromNB(length(cond_xx_data), num_blocks), find_correlation_time(cond_xx_data), find_correlation_time(cond_xy_data)])))
    cond_xx_blocks, _ = get_block_means(cond_xx_data, block_size)
    cond_xy_blocks, _ = get_block_means(cond_xy_data, block_size)
    num_blocks = length(cond_xx_blocks)

    # Bootstrapp the data
    res_xx_BT = zeros(Float64, B)
    res_xy_BT = zeros(Float64, B)
    for b in 1:B
        #pick B times a random sample from the blocked data
        BT_ix = rand(1:num_blocks, num_blocks)
        cond_xx_BT = mean(cond_xx_blocks[BT_ix])
        cond_xy_BT = mean(cond_xy_blocks[BT_ix])
        #calculate BT means
        res_xx_BT[b] = cond_xx_BT / (cond_xx_BT^2 + cond_xy_BT^2)
        res_xy_BT[b] = -cond_xy_BT / (cond_xx_BT^2 + cond_xy_BT^2)
    end

    # calculate the mean of the conductivity without BT
    res_xx_mean = mean(res_xx_BT)
    res_xy_mean = mean(res_xy_BT)

    # get the variance of the BT conductivity
    res_xx_err = sqrt(mean((res_xx_BT .- res_xx_mean).^2) * (num_blocks /(num_blocks-1)))
    res_xy_err = sqrt(mean((res_xy_BT .- res_xy_mean).^2) * (num_blocks /(num_blocks-1)))
    return (res_xx_mean, res_xx_err), (res_xy_mean, res_xy_err)
end


"""
    BT_resistivity_tensor(cond_xx, cond_xy, cond_yx, cond_yy; num_blocks=32, B=1000, w_ix=1,
                          det_eps=1e-12) -> (ρ, ρ_err, correlation_time, num_blocks)

Resistivity tensor ρ = σ⁻¹ with a block bootstrap error.

Each `cond_ab` is a per-measurement vector over frequencies (or Δω windows); entry `w_ix`
is used. The four streams are blocked together (block size: the largest correlation time,
at least length/`num_blocks`), and `B` resamples of the block means are inverted.
`ρ` and `ρ_err` are `[ρxx, ρxy, ρyx, ρyy]`. Resamples with |det σ| ≤ `det_eps` are dropped.
With fewer than 2 blocks, the plain means are inverted and the errors are NaN.
"""
function BT_resistivity_tensor(cond_xx::Vector{Vector{Float64}}, cond_xy::Vector{Vector{Float64}}, cond_yx::Vector{Vector{Float64}}, cond_yy::Vector{Vector{Float64}}; num_blocks::Int64=32, B=1000, w_ix::Int=1, det_eps::Float64=1e-12)
    # block the data
    isempty(cond_xx) && error("cond_xx is empty")
    isempty(cond_xy) && error("cond_xy is empty")
    isempty(cond_yx) && error("cond_yx is empty")
    isempty(cond_yy) && error("cond_yy is empty")

    max_w = minimum([length(cond_xx[1]), length(cond_xy[1]), length(cond_yx[1]), length(cond_yy[1])])
    (w_ix < 1 || w_ix > max_w) && error("w_ix=$w_ix out of bounds (available 1:$max_w)")
    
    cond_xx_data = hcat(cond_xx...)[w_ix,:]
    cond_xy_data = hcat(cond_xy...)[w_ix,:]
    cond_yx_data = hcat(cond_yx...)[w_ix,:]
    cond_yy_data = hcat(cond_yy...)[w_ix,:]
    correlation_time = maximum([find_correlation_time(cond_xx_data), find_correlation_time(cond_xy_data), find_correlation_time(cond_yx_data), find_correlation_time(cond_yy_data)])
    block_size = Int(round(maximum([BSfromNB(length(cond_xx_data), num_blocks), correlation_time])))
    block_size = max(block_size, 1)
    cond_xx_blocks, _ = get_block_means(cond_xx_data, block_size)
    cond_xy_blocks, _ = get_block_means(cond_xy_data, block_size)
    cond_yx_blocks, _ = get_block_means(cond_yx_data, block_size)
    cond_yy_blocks, _ = get_block_means(cond_yy_data, block_size)
    num_blocks = length(cond_xx_blocks)

    if num_blocks < 2
        cond_det = mean(cond_xx_data) * mean(cond_yy_data) - mean(cond_xy_data) * mean(cond_yx_data)
        if abs(cond_det) <= det_eps
            error("Conductivity tensor is near-singular (|det| <= $det_eps) with insufficient blocks")
        end
        res_tensor_mean = [
            mean(cond_yy_data) / cond_det,
            -mean(cond_xy_data) / cond_det,
            -mean(cond_yx_data) / cond_det,
            mean(cond_xx_data) / cond_det,
        ]
        res_tensor_err = fill(NaN, 4)
        return res_tensor_mean, res_tensor_err, correlation_time, num_blocks
    end

    # Bootstrapp the data
    res_tensor_BT = zeros(Float64, B, 4)
    valid_bt = falses(B)
    for b in 1:B
        #pick B times a random sample from the blocked data
        BT_ix = rand(1:num_blocks, num_blocks)
        cond_xx_BT = mean(cond_xx_blocks[BT_ix])
        cond_xy_BT = mean(cond_xy_blocks[BT_ix])
        cond_yx_BT = mean(cond_yx_blocks[BT_ix])
        cond_yy_BT = mean(cond_yy_blocks[BT_ix])
        #calculate BT means
        cond_det = cond_xx_BT * cond_yy_BT - cond_xy_BT * cond_yx_BT
        if abs(cond_det) <= det_eps
            continue
        end
        res_tensor_BT[b, 1] = cond_yy_BT / cond_det
        res_tensor_BT[b, 2] = -cond_xy_BT / cond_det
        res_tensor_BT[b, 3] = -cond_yx_BT / cond_det
        res_tensor_BT[b, 4] = cond_xx_BT / cond_det
        valid_bt[b] = true
    end

    num_valid = count(valid_bt)
    if num_valid < 2
        error("Too few valid bootstrap samples after determinant filtering (num_valid=$num_valid)")
    end

    valid_bt_data = res_tensor_BT[valid_bt, :]
    # calculate mean/error and store as flat 4-component vectors
    res_tensor_mean = vec(mean(valid_bt_data, dims=1))

    # get the variance of the BT conductivity
    mult = num_blocks /(num_blocks-1)
    res_tensor_err = vec(sqrt.(mean((valid_bt_data .- reshape(res_tensor_mean, 1, :)).^2, dims=1) .* mult))
    return res_tensor_mean, res_tensor_err, correlation_time, num_blocks
end


"""
    BT_stiffness_scalar(Kx, Lxx_q0; num_blocks=32, B=1000) -> (ρs, err)

Superfluid stiffness ρs = -¼(<Kx> + <Λxx(q=0, ω=0)>) from the scalar q = 0 stream,
with a block bootstrap error (block size set by the slower of the two streams).
"""
function BT_stiffness_scalar(Kx::Array{Float64}, Lxx_q0::Array{Float64}; num_blocks::Int64=32, B::Int=1000)
    @assert length(Kx) == length(Lxx_q0) "Kx and Lxx_q0 must have the same length"
    block_size = Int(round(maximum([BSfromNB(length(Kx), num_blocks),
                                    find_correlation_time(Kx),
                                    find_correlation_time(Lxx_q0)])))
    Kx_blocks, _      = get_block_means(Kx,      block_size; return_nothing=false)
    Lxx_q0_blocks, _  = get_block_means(Lxx_q0,  block_size; return_nothing=false)
    nb = length(Kx_blocks)

    stiffness_BT = zeros(Float64, B)
    for b in 1:B
        idx = rand(1:nb, nb)
        stiffness_BT[b] = -0.25 * (mean(Kx_blocks[idx]) + mean(Lxx_q0_blocks[idx]))
    end
    stiffness_mean = -0.25 * (mean(Kx_blocks) + mean(Lxx_q0_blocks))
    stiffness_err  = sqrt(mean((stiffness_BT .- stiffness_mean).^2) * (nb / (nb - 1)))
    return stiffness_mean, stiffness_err
end

"""
    BT_stiffness(Kx, Lxx_qy; num_blocks=32, fit=true, B=1000, lowest_q_only=false) -> (ρs, err)

Superfluid stiffness ρs = -¼(<Kx> + Λxx(qx=0, qy→0)) with a block bootstrap error.

`Lxx_qy` is the per-measurement vector Λxx(qy), smallest qy first. qy → 0 is taken by
`extrapolate_Lxx0` (linear from the two smallest qy, or a quadratic fit if `fit` and
there are more than 3 qy values); with `lowest_q_only` the smallest qy is used as is.
Also used with `Lxx_qx` to get the longitudinal limit (`"Lxx_anamoly"` in `get_means`).
"""
function BT_stiffness(Kx::Array{Float64}, Lxx_qy::Vector{Vector{Float64}}; num_blocks::Int64=32, fit=true, B=1000, lowest_q_only::Bool=false)
    # assuming Kx and Lxx_qy have the same length and are both up to num_current_measurements
    # with 3 or fewer q values (L <= 4) there is too little data to fit
    if length(Lxx_qy[1]) <= 3
        fit = false
    end
    # block the data
    block_size = Int(round(maximum([BSfromNB(length(Kx), num_blocks), find_correlation_time(Kx), maximum([find_correlation_time(lxx) for lxx in Lxx_qy])])))

    Kx_blocks, _ = get_block_means(Kx, block_size; return_nothing=false)
    Lxx_qy_blocks = [get_block_means(hcat(Lxx_qy...)[i,:], block_size; return_nothing=false)[1] for i in 1:length(Lxx_qy[1])]
    num_blocks = length(Kx_blocks)

    # Bootstrapp the data
    stiffness_BT = zeros(Float64, B)
    for b in 1:B
        BT_ix = rand(1:num_blocks, num_blocks)
        Kx_b = mean(Kx_blocks[BT_ix])
        if lowest_q_only
            Lxx_qy_0 = mean(Lxx_qy_blocks[1][BT_ix])
        else
            Lxx_qy_0 = extrapolate_Lxx0(Lxx_qy_blocks; fit=fit, BT_ix=BT_ix)
        end
        stiffness_BT[b] = -0.25 * (Kx_b + Lxx_qy_0)
    end

    Kx_mean = mean(Kx_blocks)
    if lowest_q_only
        Lxx_qy_0 = mean(Lxx_qy_blocks[1])
    else
        Lxx_qy_0 = extrapolate_Lxx0(Lxx_qy_blocks; fit=fit)
    end
    stiffness_mean = -0.25 * (Kx_mean + Lxx_qy_0)

    # get the variance of the BT stiffness
    stiffness_err = sqrt(mean((stiffness_BT .- stiffness_mean).^2) * (num_blocks /(num_blocks-1)))
    return stiffness_mean, stiffness_err
end

"""
    extrapolate_Lxx0(Lxx; x_vals=1:length(Lxx), fit=false, BT_ix=1:length(Lxx[1])) -> Float64

Extrapolate Λ(q) to q = 0. `Lxx[k]` are the block means at the k-th smallest q; only the
blocks `BT_ix` (a bootstrap resample) are averaged.
- `fit`: quadratic fit in `x_vals` (q index, ∝ q), value at x = 0;
- otherwise, with ≥ 2 q values: linear extrapolation 2Λ(q₁) - Λ(q₂);
- otherwise Λ(q₁).
"""
function extrapolate_Lxx0(Lxx::Vector{Vector{Float64}}; x_vals=1:length(Lxx), fit=false, BT_ix=1:length(Lxx[1]))
    if fit
        # fit Lxx_qy to a quadratic function and take Lxx_qy(0)
        data_b = [mean(lxx[BT_ix]) for lxx in Lxx]
        x_float = Float64.(collect(x_vals))
        prob = CurveFit.CurveFitProblem(x_float, data_b)
        sol = CurveFit.solve(prob, CurveFit.PolynomialFitAlgorithm(degree=2))
        params = sol.u  # [c0, c1, c2] for c0 + c1*x + c2*x^2
        Lxx_qy_0 = params[1]
    elseif length(Lxx) >= 2
        # linear extrapolation from the two smallest q
        data_b = [mean(lxx[BT_ix]) for lxx in Lxx]
        Lxx_qy_0 = 2*data_b[1] - data_b[2]
    else
        Lxx_qy_0 = mean(Lxx[1][BT_ix])
    end
    return Lxx_qy_0
end


"""
    extrapolate_delta_to_zero(delta_values, obs_values, obs_errors; 
                              min_points=3, MLS=nothing, min_delta_MLS=1.0, 
                              max_delta_MLS=nothing, chi2_max_ratio=20.0, verbose=false)

Extrapolate an observable measured at several broadening δ values to the δ→0 limit.

The function identifies the linear regime in obs(δ) by searching over contiguous 
windows of δ values, trimming from **both** the low-δ side (noisy / sub-MLS) and 
the high-δ side (non-linear / saturating), and fitting a weighted linear function. 
The final extrapolated value is a DerSimonian–Laird (DL) average over all "quality" 
windows (those whose reduced χ²/dof lies within [1/chi2_max_ratio, chi2_max_ratio]). 
This combines statistical uncertainty from each fit with model uncertainty from the 
choice of fitting window. If fewer than 2 quality windows exist, falls back to the 
single best-scoring window.

# Algorithm
1. Sort data by δ (ascending).
2. Determine allowed index range:
   - Low-δ cutoff: if `MLS` is provided, exclude δ < `min_delta_MLS * MLS`.
   - High-δ cutoff: if `max_delta_MLS` is provided, exclude δ > `max_delta_MLS * MLS`.
3. Search over all contiguous windows `(start_idx, end_idx)` within the allowed range
   with at least `min_points` points.
4. For each window compute a weighted linear fit, score by |log(χ²/dof)| with a
   small penalty for fewer points, and store the intercept and its error.
5. Identify the best-scoring window (minimum score).
6. Collect all "quality" windows: those with χ²/dof ∈ [1/chi2_max_ratio, chi2_max_ratio].
7. DL-average the intercepts of quality windows → final extrapolated value and error.
   (Falls back to best single window if < 2 quality windows.)

# Arguments
- `delta_values::Vector{<:Real}`: The δ values at which the observable was measured.
- `obs_values::Vector{<:Real}`: The observable values at each δ.
- `obs_errors::Vector{<:Real}`: The standard errors at each δ.

# Keyword Arguments
- `min_points::Int=3`: Minimum number of points required for a valid linear fit.
- `MLS::Union{Nothing,Float64}=nothing`: Mean Level Spacing. If provided, used to set 
  δ cutoffs via `min_delta_MLS` and `max_delta_MLS`.
- `min_delta_MLS::Float64=1.0`: Minimum δ in units of MLS to include in the fit.
- `max_delta_MLS::Union{Nothing,Float64}=nothing`: Maximum δ in units of MLS to include 
  in the fit. If `nothing`, no hard upper cutoff — the best window is found 
  automatically by also scanning the end index.
- `chi2_max_ratio::Float64=20.0`: A window is accepted as "quality" if its χ²/dof is 
  between `1/chi2_max_ratio` and `chi2_max_ratio`. Wider ratios include more (possibly 
  worse) windows in the DL average; narrower ratios are more selective.
- `force_full_range::Bool=false`: If `true`, skip the window search entirely and fit 
  over the full allowed δ range (`global_start:global_end`). The DL averaging is 
  bypassed (single window). Useful for a quick/reference fit without any trimming.
- `verbose::Bool=false`: If true, print diagnostic information about the fit.

# Returns
`(extrapolated_value, extrapolated_error, fit_info)` where `fit_info` is a Dict with:
- `"slope"`, `"slope_error"`: the linear coefficient and its error (best window)
- `"intercept"`, `"intercept_error"`: the DL-averaged δ→0 extrapolated value and error  
- `"chi2_per_dof"`: reduced χ² of the best-window fit
- `"R2"`: coefficient of determination of the best-window fit
- `"n_points"`: number of points in the best-window fit
- `"delta_range"`: (δ_min, δ_max) of the best-window fit
- `"start_index"`, `"end_index"`: best-window indices in the sorted array
- `"used_indices"`: data indices used in the best-window fit
- `"n_windows_DL"`: number of quality windows used in the DL average (1 = fallback)
- `"DL_intercepts"`: intercepts of all quality windows
- `"DL_errors"`: errors of all quality windows
- `"force_full_range"`: whether the full allowed δ range was forced (skipping window search)
"""
function extrapolate_delta_to_zero(delta_values::Vector{<:Real}, obs_values::Vector{<:Real}, 
                                   obs_errors::Vector{<:Real};
                                   min_points::Int=3, MLS::Union{Nothing,Float64}=nothing,
                                   min_delta_MLS::Float64=1.0, 
                                   max_delta_MLS::Union{Nothing,Float64}=nothing,
                                   chi2_max_ratio::Float64=20.0,
                                   force_full_range::Bool=false,
                                   verbose::Bool=false)
    
    n = length(delta_values)
    @assert n == length(obs_values) == length(obs_errors) "All input arrays must have the same length"
    
    # Sort by delta
    sort_idx = sortperm(delta_values)
    δ_sorted = Float64.(delta_values[sort_idx])
    y_sorted = Float64.(obs_values[sort_idx])
    σ_sorted = Float64.(obs_errors[sort_idx])
    
    # Check if errors are usable for weighting
    # If any error is zero, negative or NaN, fall back to unweighted fit
    use_weights = all(σ -> isfinite(σ) && σ > 0, σ_sorted)
    if !use_weights
        # Uniform weights: set all errors to 1 (unweighted least squares)
        σ_sorted .= 1.0
    end
    
    # --- Determine the allowed index range from MLS cutoffs ---
    global_start = 1
    global_end = n
    
    if MLS !== nothing
        # Low-δ cutoff
        min_delta = min_delta_MLS * MLS
        low_idx = findall(δ_sorted .>= min_delta)
        if !isempty(low_idx)
            global_start = first(low_idx)
        end
        
        # High-δ cutoff (new)
        if max_delta_MLS !== nothing
            max_delta = max_delta_MLS * MLS
            high_idx = findall(δ_sorted .<= max_delta)
            if !isempty(high_idx)
                global_end = last(high_idx)
            end
        end
        
        # Relax if not enough points in the allowed range
        n_available = global_end - global_start + 1
        if n_available < min_points
            # Expand range to get at least min_points, preferring the center
            global_start = max(1, global_start - (min_points - n_available))
            n_available = global_end - global_start + 1
            if n_available < min_points
                global_end = min(n, global_start + min_points - 1)
            end
            if verbose
                @warn "MLS cutoff too strict, relaxed to indices $global_start:$global_end"
            end
        end
    end
    
    # --- Search over all contiguous windows (start, end) for best linear regime ---
    best_start = global_start
    best_end = global_end
    best_score = Inf

    # Storage for DL averaging over quality windows
    dl_intercepts = Float64[]
    dl_errors     = Float64[]
    dl_chi2_reds  = Float64[]
    dl_ranges     = Tuple{Int,Int}[]

    if force_full_range
        # Skip window search — use the full allowed range directly.
        # DL pool will contain exactly this one window (no averaging).
        best_start = global_start
        best_end   = global_end
    else
    for start_idx in global_start:global_end
        for end_idx in start_idx:global_end
            n_pts = end_idx - start_idx + 1
            if n_pts < min_points
                continue
            end
            
            idx_range = start_idx:end_idx
            δ_fit_w = δ_sorted[idx_range]
            y_fit_w = y_sorted[idx_range]
            σ_fit_w = σ_sorted[idx_range]
            
            # Linear fit (weighted if errors are usable, unweighted otherwise)
            a_w, b_w, cov_w = _weighted_linear_fit(δ_fit_w, y_fit_w, σ_fit_w)
            
            # Intercept error for this window
            if use_weights
                a_w_err = sqrt(max(cov_w[1, 1], 0.0))
            else
                residuals_w = y_fit_w .- (a_w .+ b_w .* δ_fit_w)
                s2_w = n_pts > 2 ? sum(residuals_w.^2) / (n_pts - 2) : sum(residuals_w.^2)
                a_w_err = sqrt(max(cov_w[1, 1] * s2_w, 0.0))
            end

            # Compute reduced χ²
            residuals_w2 = y_fit_w .- (a_w .+ b_w .* δ_fit_w)
            if use_weights
                chi2_w = sum((residuals_w2 ./ σ_fit_w).^2)
            else
                chi2_w = sum(residuals_w2.^2) / (var(y_fit_w) > 0 ? var(y_fit_w) : 1.0)
            end
            dof_w = n_pts - 2
            chi2_red_w = dof_w > 0 ? chi2_w / dof_w : chi2_w
            
            # Score: |log(χ²/dof)| penalised for fewer points.
            # The penalty is small (0.1 per missing point relative to the full range)
            # so that fit quality dominates, but wider windows win ties.
            n_max = global_end - global_start + 1
            point_penalty = 0.1 * (n_max - n_pts) / max(n_max, 1)
            score = abs(log(max(chi2_red_w, 1e-10))) + point_penalty
            
            if score < best_score
                best_score = score
                best_start = start_idx
                best_end = end_idx
            end

            # Collect window for DL averaging if it is a "quality" fit:
            # χ²/dof must lie within [1/chi2_max_ratio, chi2_max_ratio]
            # and the intercept error must be positive and finite.
            if isfinite(chi2_red_w) && chi2_red_w > 0 &&
               chi2_red_w >= 1.0 / chi2_max_ratio && chi2_red_w <= chi2_max_ratio &&
               isfinite(a_w_err) && a_w_err > 0
                push!(dl_intercepts, a_w)
                push!(dl_errors,     a_w_err)
                push!(dl_chi2_reds,  chi2_red_w)
                push!(dl_ranges,     (start_idx, end_idx))
            end
        end
    end
    end # force_full_range branch
    
    # --- Best-window fit (used for slope, R², and fallback) ---
    fit_idx = best_start:best_end
    δ_fit = δ_sorted[fit_idx]
    y_fit = y_sorted[fit_idx]
    σ_fit = σ_sorted[fit_idx]
    n_fit = length(fit_idx)
    
    a_best, b_best, cov_best = _weighted_linear_fit(δ_fit, y_fit, σ_fit)
    
    if use_weights
        a_best_err = sqrt(max(cov_best[1, 1], 0.0))
        b_err      = sqrt(max(cov_best[2, 2], 0.0))
    else
        residuals_prelim = y_fit .- (a_best .+ b_best .* δ_fit)
        s2 = n_fit > 2 ? sum(residuals_prelim.^2) / (n_fit - 2) : sum(residuals_prelim.^2)
        cov_best   = cov_best .* s2
        a_best_err = sqrt(max(cov_best[1, 1], 0.0))
        b_err      = sqrt(max(cov_best[2, 2], 0.0))
    end
    
    # Goodness of fit for best window
    residuals = y_fit .- (a_best .+ b_best .* δ_fit)
    if use_weights
        chi2 = sum((residuals ./ σ_fit).^2)
    else
        chi2 = n_fit > 2 ? sum(residuals.^2) / (sum(residuals.^2) / (n_fit - 2)) : NaN
    end
    dof = n_fit - 2
    chi2_red = dof > 0 ? chi2 / dof : NaN
    
    ss_res = sum(residuals.^2)
    ss_tot = sum((y_fit .- mean(y_fit)).^2)
    R2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN

    # --- DL average over quality windows ---
    # Ensure the best window itself is in the DL pool (it may have been excluded
    # only because a_best_err was 0 in a degenerate case; add it if missing).
    if isfinite(a_best_err) && a_best_err > 0 && !any(
            r -> r == (best_start, best_end), dl_ranges)
        # best window passed score but was filtered by chi2_max_ratio – still use it
        push!(dl_intercepts, a_best)
        push!(dl_errors,     a_best_err)
    end

    n_dl = length(dl_intercepts)
    if n_dl >= 2
        a_dl, a_dl_err = obs_seed_mean2(dl_intercepts, dl_errors; method="DL")
        # If DL fails (NaN/Inf), fall back to best window
        if !isfinite(a_dl) || !isfinite(a_dl_err)
            a_dl, a_dl_err = a_best, a_best_err
            n_dl = 1
        end
    else
        # Not enough quality windows — use best single window
        a_dl, a_dl_err = a_best, a_best_err
        n_dl = max(n_dl, 1)
    end
    
    fit_info = Dict{String, Any}(
        "slope"                => b_best,
        "slope_error"          => b_err,
        "intercept"            => a_dl,
        "intercept_error"      => a_dl_err,
        "intercept_error_best" => a_best_err,  # best-window intercept error (for fit ribbon)
        "cov_ab"               => cov_best[1, 2],   # covariance between intercept and slope
        "chi2_per_dof"   => chi2_red,
        "R2"             => R2,
        "n_points"       => n_fit,
        "delta_range"    => (δ_fit[1], δ_fit[end]),
        "start_index"    => best_start,
        "end_index"      => best_end,
        "used_indices"   => sort_idx[fit_idx],
        "weighted"       => use_weights,
        "n_windows_DL"   => n_dl,
        "DL_intercepts"  => dl_intercepts,
        "DL_errors"      => dl_errors,
        "force_full_range" => force_full_range,
    )
    
    if verbose
        println("  δ→0 extrapolation: best window $(n_fit)/$(n) points, δ ∈ [$(round(δ_fit[1], sigdigits=3)), $(round(δ_fit[end], sigdigits=3))]")
        if MLS !== nothing
            println("    δ range in MLS: [$(round(δ_fit[1]/MLS, sigdigits=3)), $(round(δ_fit[end]/MLS, sigdigits=3))]")
        end
        excluded_low = best_start - 1
        excluded_high = n - best_end
        println("    Best window: excluded $excluded_low low-δ and $excluded_high high-δ points")
        println("    Best-window fit: obs = $(round(a_best, sigdigits=5)) + $(round(b_best, sigdigits=5)) · δ")
        println("    χ²/dof = $(round(chi2_red, sigdigits=3)), R² = $(round(R2, sigdigits=4))")
        println("    DL average over $n_dl quality window(s): $(round(a_dl, sigdigits=5)) ± $(round(a_dl_err, sigdigits=3))")
    end
    
    return a_dl, a_dl_err, fit_info
end

"""
    _weighted_linear_fit(x, y, σ)

Perform a weighted linear least squares fit y = a + b*x with weights w = 1/σ².
Returns (a, b, covariance_matrix).
"""
function _weighted_linear_fit(x::Vector{Float64}, y::Vector{Float64}, σ::Vector{Float64})
    w = 1.0 ./ (σ.^2)
    W = sum(w)
    Wx = sum(w .* x)
    Wy = sum(w .* y)
    Wxx = sum(w .* x.^2)
    Wxy = sum(w .* x .* y)
    
    det = W * Wxx - Wx^2
    if abs(det) < 1e-30
        # Degenerate case: return unweighted mean
        a = mean(y)
        return a, 0.0, [var(y)/length(y) 0.0; 0.0 0.0]
    end
    
    a = (Wxx * Wy - Wx * Wxy) / det
    b = (W * Wxy - Wx * Wy) / det
    
    # Covariance matrix of (a, b)
    cov_aa = Wxx / det
    cov_bb = W / det
    cov_ab = -Wx / det
    cov_matrix = [cov_aa cov_ab; cov_ab cov_bb]
    
    return a, b, cov_matrix
end


"""
    fit_finite_size_power_law(L, y, σ; fixed_a=nothing)

Fit the power-law finite-size scaling `y ≈ A · L^a` by a weighted linear fit in
log-log space: `log(y) = log(A) + a · log(L)`, with log-space uncertainty
`σ_logy = σ_y / |y|`.

Points with `y ≤ 0`, non-finite `L/y/σ`, or `L ≤ 0` are dropped with a warning
(log-log requires `y > 0`).

# Arguments
- `L`, `y`, `σ`: equal-length vectors of system sizes, observable values, and 1-σ errors.

# Keywords
- `fixed_a`: if a `Real`, fix the exponent and only fit `A` (useful for testing a known
  exponent). If `nothing` (default), fit both `A` and `a`.

# Returns
A NamedTuple with fields:
- `A`, `a`            — best-fit amplitude and exponent
- `A_err`, `a_err`    — 1-σ errors (`a_err == 0` when `fixed_a` is used)
- `chi2`, `chi2_red`, `dof` — log-space goodness of fit
- `log_cov`           — covariance matrix of `(log A, a)` from the linearized fit
- `L_used`, `y_used`, `σ_used` — the points actually used (after filtering)
- `fixed_a::Bool`     — whether the exponent was held fixed
"""
function fit_finite_size_power_law(
    L::AbstractVector, y::AbstractVector, σ::AbstractVector;
    fixed_a::Union{Nothing, Real}=nothing,
)
    length(L) == length(y) == length(σ) ||
        error("fit_finite_size_power_law: L, y, σ must have equal length (got $(length(L)), $(length(y)), $(length(σ))).")

    keep = findall(i -> isfinite(L[i]) && L[i] > 0 &&
                         isfinite(y[i]) && y[i] > 0 &&
                         isfinite(σ[i]) && σ[i] >= 0,
                   eachindex(L))
    n_drop = length(L) - length(keep)
    n_drop > 0 && @warn "fit_finite_size_power_law: dropped $n_drop point(s) with non-positive y or non-finite entries"

    n_params = fixed_a === nothing ? 2 : 1
    length(keep) >= n_params ||
        error("fit_finite_size_power_law: not enough valid points ($(length(keep))) to fit $n_params parameter(s).")

    Lu = Float64.(L[keep])
    yu = Float64.(y[keep])
    σu = Float64.(σ[keep])

    xlog = log.(Lu)
    ylog = log.(yu)
    σlog = @. max(σu / abs(yu), eps())

    if fixed_a === nothing
        logA, a_fit, cov_m = _weighted_linear_fit(xlog, ylog, σlog)
        a_est = a_fit
        A = exp(logA)
        logA_err = sqrt(max(cov_m[1, 1], 0.0))
        a_err = sqrt(max(cov_m[2, 2], 0.0))
        A_err = A * logA_err
        cov_out = cov_m
    else
        a_est = Float64(fixed_a)
        w = 1.0 ./ σlog .^ 2
        logA = sum(w .* (ylog .- a_est .* xlog)) / sum(w)
        A = exp(logA)
        logA_err = 1.0 / sqrt(sum(w))
        A_err = A * logA_err
        a_err = 0.0
        cov_out = [logA_err^2 0.0; 0.0 0.0]
    end

    ylog_pred = log(A) .+ a_est .* xlog
    resid = (ylog .- ylog_pred) ./ σlog
    chi2 = sum(resid .^ 2)
    dof = max(length(keep) - n_params, 1)
    chi2_red = chi2 / dof

    return (; A=A, a=a_est, A_err=A_err, a_err=a_err,
              chi2=chi2, chi2_red=chi2_red, dof=dof,
              log_cov=cov_out,
              L_used=Lu, y_used=yu, σ_used=σu,
              fixed_a=(fixed_a !== nothing))
end


"""
    extrapolate_observable_delta_to_zero(means_dict, obs_key, delta_values;
                                         MLS=nothing, min_delta_MLS=1.0,
                                         min_points=3, verbose=false)

Extrapolate a delta-indexed observable from `means_dict` to δ→0.

Handles both:
- **Scalar** observables (one value per δ), e.g. "Drude_weight_fixed", "stiffness_delta"
- **Vector** observables (one vector per δ, e.g. per-ω conductivity), 
  e.g. "conductivity_regular_xx_fixed" — extrapolates each component independently.

The result is stored back into `means_dict` under the key `obs_key * "_extrap"`.

# Arguments
- `means_dict::Dict`: The means dictionary (modified in place).
- `obs_key::String`: Key of the delta-indexed observable in `means_dict`.
- `delta_values::Vector{<:Real}`: The δ values corresponding to each index.

# Keyword Arguments  
- `MLS`, `min_delta_MLS`, `max_delta_MLS`, `min_points`, `force_full_range`, `verbose`: passed to `extrapolate_delta_to_zero`.

# Returns
The extrapolation result Dict stored in `means_dict[obs_key * "_extrap"]`.
"""
function extrapolate_observable_delta_to_zero(means_dict::Dict, obs_key::String, 
                                              delta_values::Vector{<:Real};
                                              MLS::Union{Nothing,Float64}=nothing,
                                              min_delta_MLS::Float64=1.0,
                                              max_delta_MLS::Union{Nothing,Float64}=nothing,
                                              min_points::Int=3, force_full_range::Bool=false,
                                              verbose::Bool=false)
    if !haskey(means_dict, obs_key)
        verbose && println("  Key '$obs_key' not found in means_dict, skipping.")
        return nothing
    end
    
    obs_data = means_dict[obs_key]
    values_raw = obs_data["value"]
    errors_raw = obs_data["error"]

    # Normalize storage so indexing is always by delta point first.
    # This supports both Vector-per-delta storage and Matrix storage (n_delta x n_components)
    # that can appear after averaging over seeds.
    values = if isa(values_raw, AbstractMatrix)
        [view(values_raw, i, :) for i in axes(values_raw, 1)]
    elseif isa(values_raw, AbstractVector)
        collect(values_raw)
    else
        verbose && println("  Unsupported value container for '$obs_key': $(typeof(values_raw)), skipping.")
        return nothing
    end

    errors = if isa(errors_raw, AbstractMatrix)
        [view(errors_raw, i, :) for i in axes(errors_raw, 1)]
    elseif isa(errors_raw, AbstractVector)
        collect(errors_raw)
    else
        verbose && println("  Unsupported error container for '$obs_key': $(typeof(errors_raw)), skipping.")
        return nothing
    end

    if length(values) != length(errors)
        verbose && println("  Mismatch in value/error lengths for '$obs_key' ($(length(values)) vs $(length(errors))), skipping.")
        return nothing
    end

    if length(delta_values) < length(values)
        verbose && println("  Not enough delta values for '$obs_key' ($(length(delta_values)) < $(length(values))), skipping.")
        return nothing
    end
    
    # Filter out Nothing entries
    valid_mask = [v !== nothing && e !== nothing for (v, e) in zip(values, errors)]
    valid_idx = findall(valid_mask)
    
    if length(valid_idx) < min_points
        verbose && println("  Not enough valid delta points for '$obs_key' ($(length(valid_idx)) < $min_points)")
        return nothing
    end
    
    δ_valid = Float64.(delta_values[valid_idx])
    
    extrap_key = obs_key * "_extrap"
    
    # Detect if observable is scalar-per-delta or vector-per-delta
    first_val = values[valid_idx[1]]
    
    if isa(first_val, AbstractArray)
        # Vector observable (e.g., conductivity per ω for each δ)
        n_components = length(vec(first_val))
        if obs_key == "resistivity_tensor_fixed" && n_components != 4
            verbose && println("  Expected 4 components for '$obs_key', got $n_components")
        end
        extrap_values = Vector{Float64}(undef, n_components)
        extrap_errors = Vector{Float64}(undef, n_components)
        fit_infos = Vector{Dict{String,Any}}(undef, n_components)
        
        for j in 1:n_components
            y_j = Float64.([vec(values[i])[j] for i in valid_idx])
            σ_j = Float64.([vec(errors[i])[j] for i in valid_idx])
            
            val, err, info = extrapolate_delta_to_zero(δ_valid, y_j, σ_j;
                min_points=min_points, MLS=MLS, min_delta_MLS=min_delta_MLS, 
                max_delta_MLS=max_delta_MLS, force_full_range=force_full_range,
                verbose=verbose)
            
            extrap_values[j] = val
            extrap_errors[j] = err
            fit_infos[j] = info
        end
        
        means_dict[extrap_key] = Dict{String, Any}(
            "value" => extrap_values,
            "error" => extrap_errors,
            "fit_info" => fit_infos,
            "source" => obs_key,
            "delta_values_used" => δ_valid,
            "method" => "linear_extrapolation_delta_to_zero"
        )
    else
        # Scalar observable per delta
        y_valid = Float64.([values[i] for i in valid_idx])
        σ_valid = Float64.([errors[i] for i in valid_idx])
        
        val, err, info = extrapolate_delta_to_zero(δ_valid, y_valid, σ_valid;
            min_points=min_points, MLS=MLS, min_delta_MLS=min_delta_MLS, 
            max_delta_MLS=max_delta_MLS, force_full_range=force_full_range,
            verbose=verbose)
        
        means_dict[extrap_key] = Dict{String, Any}(
            "value" => val,
            "error" => err,
            "fit_info" => info,
            "source" => obs_key,
            "delta_values_used" => δ_valid,
            "method" => "linear_extrapolation_delta_to_zero"
        )
    end
    
    if verbose
        println("  → Stored extrapolation in '$extrap_key'")
    end
    
    return means_dict[extrap_key]
end


"""
    extrapolate_all_delta_observables!(means_dict, sim_dict; 
                                       min_delta_MLS=1.0, max_delta_MLS=nothing,
                                       min_points=3, verbose=false)

Extrapolate all delta-dependent observables in `means_dict` to δ→0.

Automatically finds observables indexed by delta (fixed-delta conductivity, 
Drude weight, stiffness, etc.) and calls `extrapolate_observable_delta_to_zero` 
on each.

# Observables handled
- `"conductivity_regular_xx_fixed"`, `"..._xy_fixed"`, `"..._yx_fixed"`, `"..._yy_fixed"` (vector per δ)
- `"Lxx_D_fixed"` (vector per δ)
- `"Drude_weight_fixed"` (scalar per δ)
- `"stiffness_delta"`, `"stiffness_y_delta"` (scalar per δ)
- `"resistivity_tensor_fixed"` (4-component vector per δ)

Also handles DC-integral observables (`conductivity_dc_xx`, etc.) if present, 
treating dw_values as the broadening parameter.

# Arguments
- `means_dict::Dict`: Modified in place with new `"*_extrap"` keys.
- `sim_dict::Dict`: Simulation info containing `"delta_fixed_values"` and `"dw_values"`.

# Keyword Arguments
- `min_delta_MLS::Float64=1.0`: Minimum δ in MLS units (low-δ cutoff).
- `max_delta_MLS::Union{Nothing,Float64}=nothing`: Maximum δ in MLS units (high-δ cutoff). 
  If `nothing`, the best window end is found automatically.
- `min_points::Int=3`: Minimum points in fit window.
- `verbose::Bool=false`: Print diagnostics.

# Returns
A Dict mapping observable name → extrapolation result (or nothing if skipped).
"""
function extrapolate_all_delta_observables!(means_dict::Dict, sim_dict::Dict;
                                            min_delta_MLS::Float64=1.0,
                                            max_delta_MLS::Union{Nothing,Float64}=nothing,
                                            min_points::Int=3, force_full_range::Bool=false,
                                            verbose::Bool=false)
    Lx, Ly = get_sim_dimensions(sim_dict)
    MLS = 4.0 / (Lx * Ly)
    
    results = Dict{String, Any}()
    
    # --- Fixed-delta observables ---
    delta_fixed = get(sim_dict, "delta_fixed_values", Float64[])
    if !isempty(delta_fixed)
        fixed_delta_obs = [
            "conductivity_regular_xx_fixed",
            "conductivity_regular_xy_fixed",
            "conductivity_regular_yx_fixed",
            "conductivity_regular_yy_fixed",
            "Lxx_D_fixed",
            "Drude_weight_fixed",
            "resistivity_tensor_fixed",
        ]
        
        for obs_key in fixed_delta_obs
            if haskey(means_dict, obs_key)
                verbose && println("Extrapolating '$obs_key' ($(length(delta_fixed)) δ values)...")
                result = extrapolate_observable_delta_to_zero(
                    means_dict, obs_key, delta_fixed;
                    MLS=MLS, min_delta_MLS=min_delta_MLS,
                    max_delta_MLS=max_delta_MLS,
                    min_points=min_points, force_full_range=force_full_range,
                    verbose=verbose
                )
                results[obs_key] = result
            end
        end
    end

    # ----- stiffness vs delta observables (different delta values than conductivity) -----
    stiffness_delta_values = get(sim_dict, "delta_values_stiffness", Float64[])
    if !isempty(stiffness_delta_values)
        stiffness_obs = [
            "stiffness_delta",
            "stiffness_y_delta",
        ]
        
        for obs_key in stiffness_obs
            if haskey(means_dict, obs_key)
                verbose && println("Extrapolating '$obs_key' ($(length(stiffness_delta_values)) δ values)...")
                result = extrapolate_observable_delta_to_zero(
                    means_dict, obs_key, stiffness_delta_values;
                    MLS=MLS, min_delta_MLS=min_delta_MLS,
                    max_delta_MLS=max_delta_MLS,
                    min_points=min_points, force_full_range=force_full_range,
                    verbose=verbose
                )
                results[obs_key] = result
            end
        end
    end
    
    # --- DC integral observables (dw broadening) ---
    dw_values = get(sim_dict, "dw_values", Float64[])
    if !isempty(dw_values)
        dc_obs = [
            "conductivity_dc_xx",
            "conductivity_dc_yy",
        ]
        
        for obs_key in dc_obs
            if haskey(means_dict, obs_key)
                verbose && println("Extrapolating '$obs_key' ($(length(dw_values)) Δω values)...")
                result = extrapolate_observable_delta_to_zero(
                    means_dict, obs_key, dw_values;
                    MLS=MLS, min_delta_MLS=min_delta_MLS,
                    max_delta_MLS=max_delta_MLS,
                    min_points=min_points, force_full_range=force_full_range,
                    verbose=verbose
                )
                results[obs_key] = result
            end
        end
    end
    
    return results
end


# a wrapper function for a sim_info dictionary 
function extrapolate_all_delta_observables!(sim_info::Dict{String, Any};
                                            min_delta_MLS::Float64=1.0,
                                            max_delta_MLS::Union{Nothing,Float64}=nothing,
                                            min_points::Int=3, force_full_range::Bool=false,
                                            verbose::Bool=false)
    # go over all sims in sim_info and call extrapolate_all_delta_observables! on each means_dict
    for (sim_key, sim_data) in sim_info
        if sim_key == "updated_at" || !isa(sim_data, Dict)
            continue
        end
        verbose && println("Processing simulation '$sim_key'...")
        means_dict = sim_data["means"]
        if !isa(means_dict, Dict)
            @warn "Simulation '$sim_key' does not have a valid 'means' dictionary, skipping."
            continue
        end
        extrapolate_all_delta_observables!(means_dict, sim_data; 
            min_delta_MLS=min_delta_MLS, max_delta_MLS=max_delta_MLS,
            min_points=min_points, force_full_range=force_full_range,
            verbose=verbose)
    end

end


# a wrapper to extrapolate for all similations in a sims dictionary



"""
    plot_delta_extrapolation(means_dict, obs_key, delta_values;
                             MLS=nothing, component_idx=nothing, 
                             units_MLS=true, title_prefix="")

Plot an observable vs δ together with the linear fit used for extrapolation.

Data points excluded from the fit are shown in gray; points used in the fit are 
shown in blue with error bars. The fit line is extended to δ=0, and the 
extrapolated value is marked with a star.

# Arguments
- `means_dict::Dict`: The means dictionary containing both `obs_key` and `obs_key*"_extrap"`.
- `obs_key::String`: Key of the delta-indexed observable.
- `delta_values::Vector{<:Real}`: The δ values corresponding to each index.

# Keyword Arguments
- `component_idx::Union{Nothing,Int}=nothing`: For vector observables (e.g. conductivity 
  per ω), which component to plot. If `nothing` and the observable is a vector, plots 
  component 1.
- `units_MLS::Bool=true`: Whether to label the x-axis in MLS units. MLS is read from 
  `means_dict["energy_spacing_mean"]["value"]`.
- `title_prefix::String=""`: Optional prefix for the plot title.

# Returns
A `Plots.Plot` object.
"""
function plot_delta_extrapolation(means_dict::Dict, obs_key::String, 
                                  delta_values::Vector{<:Real};
                                  MLS=nothing,
                                  component_idx::Union{Nothing,Int}=nothing,
                                  units_MLS::Bool=true,
                                  title_prefix::String="",
                                  show_dc::Bool=false,
                                  dw_values::Union{Nothing, Vector{<:Real}}=nothing)
    extrap_key = obs_key * "_extrap"
    if !haskey(means_dict, obs_key) || !haskey(means_dict, extrap_key)
        error("Missing '$obs_key' or '$extrap_key' in means_dict. Run extrapolation first.")
    end

    # Get MLS from energy_spacing_mean
    if MLS === nothing
        if haskey(means_dict, "energy_spacing_mean")
            MLS = Float64(means_dict["energy_spacing_mean"]["value"])
        end
    else
        MLS = Float64(MLS)
    end

    obs_data = means_dict[obs_key]
    extrap_data = means_dict[extrap_key]
    values = obs_data["value"]
    errors = obs_data["error"]

    # Determine if scalar or vector per delta
    first_valid = findfirst(v -> v !== nothing, values)
    is_vector = isa(values[first_valid], AbstractArray)

    # Select component for vector observables
    cidx = something(component_idx, 1)
    if is_vector
        fit_info = extrap_data["fit_info"][cidx]
        extrap_val = extrap_data["value"][cidx]
        extrap_err = extrap_data["error"][cidx]
    else
        fit_info = extrap_data["fit_info"]
        extrap_val = extrap_data["value"]
        extrap_err = extrap_data["error"]
    end

    # Extract data points
    n = length(delta_values)
    δ_all = Float64.(delta_values)
    y_all = Vector{Union{Float64,Nothing}}(undef, n)
    σ_all = Vector{Union{Float64,Nothing}}(undef, n)
    for i in 1:n
        if values[i] === nothing
            y_all[i] = nothing
            σ_all[i] = nothing
        elseif is_vector
            y_all[i] = Float64(values[i][cidx])
            σ_all[i] = Float64(errors[i][cidx])
        else
            y_all[i] = Float64(values[i])
            σ_all[i] = Float64(errors[i])
        end
    end

    # Determine which points were used in the fit
    used_idx = fit_info["used_indices"]
    all_valid = findall(v -> v !== nothing, y_all)
    excluded_idx = setdiff(all_valid, used_idx)

    # X-axis scaling
    x_scale = (units_MLS && MLS !== nothing) ? 1.0 / MLS : 1.0
    x_label = (units_MLS && MLS !== nothing) ? "δ [MLS]" : "δ"

    # Fit parameters (best window)
    a = fit_info["intercept"]   # DL-averaged intercept
    b = fit_info["slope"]       # best-window slope
    R2 = fit_info["R2"]
    chi2_dof = fit_info["chi2_per_dof"]
    n_pts = fit_info["n_points"]

    # DL window info
    n_dl         = get(fit_info, "n_windows_DL", 1)
    dl_intercepts = get(fit_info, "DL_intercepts", Float64[])
    dl_errors     = get(fit_info, "DL_errors",     Float64[])

    # Build plot
    component_label = is_vector ? " [component $cidx]" : ""
    title_str = title_prefix * obs_key * component_label * " — δ→0 extrapolation (DL over $n_dl windows)"

    fig = plot(
        xlabel=x_label,
        ylabel=obs_key * component_label,
        title=title_str,
        size=(800, 500),
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
    )

    # Plot excluded points (gray)
    if !isempty(excluded_idx)
        scatter!(fig, δ_all[excluded_idx] .* x_scale, 
                 Float64.([y_all[i] for i in excluded_idx]),
                 yerr=Float64.([σ_all[i] for i in excluded_idx]),
                 label="excluded from fit", m=:circle, ms=5, mc=:gray, msc=:gray,
                 lc=:gray)
    end

    # Plot used points (blue) — best window
    scatter!(fig, δ_all[used_idx] .* x_scale, 
             Float64.([y_all[i] for i in used_idx]),
             yerr=Float64.([σ_all[i] for i in used_idx]),
             label="best-window pts ($n_pts pts, χ²/dof=$(round(chi2_dof, sigdigits=3)))",
             m=:circle, ms=6, mc=:blue, msc=:blue, lc=:blue)

    # Plot best-window fit line with propagated error ribbon
    # σ_fit(δ)² = C_aa + 2·C_ab·δ + C_bb·δ²
    cov_aa = get(fit_info, "intercept_error_best", get(fit_info, "intercept_error", 0.0))^2
    cov_bb = get(fit_info, "slope_error", 0.0)^2
    cov_ab = get(fit_info, "cov_ab", 0.0)
    δ_max = maximum(δ_all)
    δ_line = collect(range(0.0, δ_max, length=100))
    y_line = a .+ b .* δ_line
    σ_line = sqrt.(max.(cov_aa .+ 2 .* cov_ab .* δ_line .+ cov_bb .* δ_line.^2, 0.0))
    plot!(fig, δ_line .* x_scale, y_line;
          ribbon=σ_line,
          fillalpha=0.2, fillcolor=:royalblue,
          label="best-window fit (slope=$(round(b, sigdigits=3)))",
          lw=2, ls=:dash, lc=:royalblue)

    # Show spread of DL window intercepts at δ=0 (if more than one window)
    if n_dl > 1 && !isempty(dl_intercepts)
        # Horizontal band: min–max range of DL intercepts
        dl_min = minimum(dl_intercepts)
        dl_max = maximum(dl_intercepts)
        δ_band_half = δ_max * 0.04   # narrow band around δ=0
        hspan_x = [-δ_band_half * x_scale, δ_band_half * x_scale]
        plot!(fig, hspan_x, [dl_min, dl_min],
              ribbon=(zeros(2), fill(dl_max - dl_min, 2)),
              fillalpha=0.15, fillcolor=:orange, lw=0,
              label="DL windows range ($n_dl windows)")

        # Individual DL window intercepts as tiny jittered dots
        jitter_step = δ_band_half * 0.6 / max(n_dl - 1, 1) * x_scale
        jitter_x = [(i - (n_dl + 1) / 2) * jitter_step for i in 1:n_dl]
        scatter!(fig, jitter_x, dl_intercepts,
                 yerr=dl_errors,
                 label=nothing, m=:circle, ms=4,
                 mc=:orange, msc=:orange, lc=:orange, alpha=0.6)
    end

    # Mark DL-averaged extrapolated value at δ=0
    scatter!(fig, [0.0], [extrap_val],
             yerr=[extrap_err],
             label="DL extrap ($n_dl win): $(round(extrap_val, sigdigits=4)) ± $(round(extrap_err, sigdigits=2))",
             m=:star5, ms=10, mc=:red, msc=:red, lc=:red)

    # add a vertical line for the abrava MLS from the simulation
    mls = get(means_dict, "energy_spacing_mean", Dict{String, Any}())["value"]
    if mls !== nothing
        if units_MLS
            mls_label = "<ϵ> = $(round(mls, sigdigits=3))"
        else
            mls_label = "<ϵ> = $(round(mls * x_scale, sigdigits=3)) MLS"
        end
        vline!(fig, [mls * x_scale], label=mls_label, lc=:green, ls=:dot)
    end

    # Overlay conductivity_dc data if requested
    if show_dc
        # Map conductivity_regular obs_key → corresponding conductivity_dc key
        dc_map = Dict(
            "conductivity_regular_xx_fixed" => "conductivity_dc_xx",
            "conductivity_regular_yy_fixed" => "conductivity_dc_yy",
            "conductivity_regular_xy_fixed" => "conductivity_dc_xy",
            "conductivity_regular_yx_fixed" => "conductivity_dc_yx",
        )
        dc_key = get(dc_map, obs_key, nothing)
        if dc_key !== nothing && haskey(means_dict, dc_key) && dw_values !== nothing
            dc_vals = means_dict[dc_key]["value"]
            dc_errs = means_dict[dc_key]["error"]
            dw_x = Float64.(dw_values) .* x_scale
            # For vector-valued DC obs, extract the right component
            if !isempty(dc_vals) && isa(dc_vals[1], AbstractArray)
                dc_y   = Float64[v[cidx] for v in dc_vals]
                dc_yerr = Float64[e[cidx] for e in dc_errs]
            else
                dc_y   = Float64.(real.(dc_vals))
                dc_yerr = Float64.(abs.(dc_errs))
            end
            scatter!(fig, dw_x, dc_y;
                     yerr=dc_yerr,
                     label="DC $(dc_key) vs Δω",
                     m=:square, ms=6, mc=:orange, msc=:orange, lc=:orange, lw=0)
        elseif dc_key === nothing
            @warn "show_dc=true but no DC mapping for obs_key='$obs_key'"
        elseif dw_values === nothing
            @warn "show_dc=true but dw_values not provided"
        end
    end

    # Annotate fit quality
    # annotate!(fig, [(δ_max * x_scale * 0.95, minimum(y_line) + 0.85*(maximum(y_line)-minimum(y_line)),
    #                  Plots.text("R²=$(round(R2, sigdigits=3))\nχ²/dof=$(round(chi2_dof, sigdigits=3))",
    #                             8, :right, :gray))])

    return fig
end


function plot_delta_extrapolation(sim_dict::Dict{String, Any}, obs_key::String;
                                  component_idx::Union{Nothing,Int}=nothing,
                                  units_MLS::Bool=true,
                                  title_prefix::String="",
                                  show_dc::Bool=false
        )
    # find the appropriate delta kind for this observable
    if obs_key in ["conductivity_regular_xx_fixed", "conductivity_regular_xy_fixed",
                   "conductivity_regular_yx_fixed", "conductivity_regular_yy_fixed",
                   "Lxx_D_fixed", "Drude_weight_fixed", "resistivity_tensor_fixed"]
        delta_values_name = "delta_fixed_values"
    elseif obs_key in ["stiffness_delta", "stiffness_y_delta"]
        delta_values_name = "delta_values_stiffness"
    elseif obs_key in ["conductivity_dc_xx", "conductivity_dc_yy"]
        delta_values_name = "dw_values"
    else
        error("Unknown observable key '$obs_key' for delta extrapolation")
    end

    if haskey(sim_dict, "means")
        L = get_sim_dimensions(sim_dict)
        N = L[1] * L[2]
        means_dict = sim_dict["means"]
        dw_vals = show_dc ? get(sim_dict, "dw_values", nothing) : nothing
        return plot_delta_extrapolation(means_dict, obs_key, sim_dict[delta_values_name];
                                      MLS = 4/N, component_idx=component_idx, units_MLS=units_MLS,
                                      title_prefix=title_prefix, show_dc=show_dc, dw_values=dw_vals)
    elseif haskey(sim_dict, "sim_dict") && haskey(sim_dict["sim_dict"], "means")
        L = get_sim_dimensions(sim_dict["sim_dict"])
        N = L[1] * L[2]
        means_dict = sim_dict["sim_dict"]["means"]
        inner = sim_dict["sim_dict"]
        dw_vals = show_dc ? get(inner, "dw_values", nothing) : nothing
        return plot_delta_extrapolation(means_dict, obs_key, inner[delta_values_name];
                                      MLS = 4/N, component_idx=component_idx, units_MLS=units_MLS,
                                      title_prefix=title_prefix, show_dc=show_dc, dw_values=dw_vals)
    else
        error("sim_dict missing 'means' key")
    end
end


"""
    plot_all_delta_extrapolations(means_dict, sim_dict; component_idx=1)

Plot δ→0 extrapolation diagnostics for all delta-dependent observables that 
have been extrapolated (keys ending in `"_extrap"`).

Returns a vector of (obs_key, figure) tuples.
"""
function plot_all_delta_extrapolations(means_dict::Dict, sim_dict::Dict;
                                       component_idx::Int=1)
    delta_fixed = get(sim_dict, "delta_fixed_values", Float64[])
    delta_stiffness = get(sim_dict, "delta_stiffness_values", delta_fixed)
    dw_values = get(sim_dict, "dw_values", Float64[])

    # Map observable key → appropriate delta array
    delta_map = Dict{String, Vector{Float64}}()
    for key in ["conductivity_regular_xx_fixed", "conductivity_regular_xy_fixed",
                "conductivity_regular_yx_fixed", "conductivity_regular_yy_fixed",
                "Lxx_D_fixed", "Drude_weight_fixed", "resistivity_tensor_fixed"]
        delta_map[key] = delta_fixed
    end
    for key in ["stiffness_delta", "stiffness_y_delta"]
        delta_map[key] = delta_stiffness
    end
    for key in ["conductivity_dc_xx", "conductivity_dc_yy"]
        delta_map[key] = dw_values
    end

    figs = Tuple{String, Any}[]
    for (obs_key, δ_vals) in delta_map
        extrap_key = obs_key * "_extrap"
        if haskey(means_dict, extrap_key) && !isempty(δ_vals)
            try
                fig = plot_delta_extrapolation(means_dict, obs_key, δ_vals;
                    component_idx=component_idx)
                push!(figs, (obs_key, fig))
            catch e
                @warn "Failed to plot extrapolation for $obs_key: $e"
            end
        end
    end

    return figs
end

# ============================================================================
# Multi-seed diagnostics for get_sims output
# ============================================================================

"""
    get_from_any_dict(d, key::AbstractString, default=nothing)

`d[key]`, else `d[Symbol(key)]`, else `default`.
"""
function get_from_any_dict(d::AbstractDict, key::AbstractString, default=nothing)
    if haskey(d, key)
        return d[key]
    end
    skey = Symbol(key)
    if haskey(d, skey)
        return d[skey]
    end
    return default
end

"""
    _is_get_sims_leaf(node) -> Bool

True if `node` is one parameter point of a `get_sims` result (has `"sim_num"` and a `"sim_dict"`).
"""
function _is_get_sims_leaf(node::AbstractDict)
    haskey(node, "sim_num") && haskey(node, "sim_dict") && (node["sim_dict"] isa AbstractDict)
end

"""
    _try_parse_int(v) -> Union{Int, Nothing}

`v` as an Int (rounded if a finite float, parsed if a string), else `nothing`.
"""
function _try_parse_int(v)
    if v isa Integer
        return Int(v)
    elseif v isa AbstractFloat
        return isfinite(v) ? Int(round(v)) : nothing
    elseif v isa AbstractString
        return tryparse(Int, strip(v))
    end
    return nothing
end

"""
    _max_finite_number(x) -> Float64

Largest finite number in `x` (a number, or nested arrays/Dicts of numbers); `-Inf` if none.
"""
function _max_finite_number(x)
    if x isa Number
        xf = Float64(x)
        return isfinite(xf) ? xf : -Inf
    elseif x isa AbstractArray
        isempty(x) && return -Inf
        out = -Inf
        for v in x
            out = max(out, _max_finite_number(v))
        end
        return out
    elseif x isa AbstractDict
        isempty(x) && return -Inf
        out = -Inf
        for v in values(x)
            out = max(out, _max_finite_number(v))
        end
        return out
    end
    return -Inf
end

"""
    _normalize_lookup_path(path) -> String

`path` normalized, lower-case, with `/` separators, so paths from logs and from sim.jld2 compare equal.
"""
function _normalize_lookup_path(path::String)
    replace(lowercase(normpath(path)), "\\" => "/")
end

"""
    _extract_worst_tau_from_means(means_dict) -> (tau, observable) or (nothing, nothing)

The largest correlation time among all observables of one seed's means, and its observable.
"""
function _extract_worst_tau_from_means(means_dict::AbstractDict)
    worst_tau = -Inf
    worst_observable = nothing

    for (k, v) in means_dict
        if !(v isa AbstractDict)
            continue
        end
        corr_time = get_from_any_dict(v, "correlation_time", nothing)
        if isnothing(corr_time)
            continue
        end
        tau_val = _max_finite_number(corr_time)
        if isfinite(tau_val) && tau_val > 0 && tau_val > worst_tau
            worst_tau = tau_val
            worst_observable = string(k)
        end
    end

    if !isfinite(worst_tau) || worst_tau <= 0
        return nothing, nothing
    end
    return worst_tau, worst_observable
end

"""
    build_worker_log_index(log_dir; max_lines_per_file=200000) -> Dict

Index the SLURM logs (`<job>.out`, `<job>_<task>.error`, ...) in `log_dir` by the
simulation folders they ran: every `Processing: <path>` line gives
normalized path => [Dict("job_id", "task_id", "log_file"), ...].
"""
function build_worker_log_index(log_dir::String; max_lines_per_file::Int=200000)
    index = Dict{String, Vector{Dict{String, Any}}}()
    if !isdir(log_dir)
        return index
    end

    for filename in readdir(log_dir)
        if !(endswith(filename, ".out") || endswith(filename, ".error"))
            continue
        end
        m = match(r"^(\d+)(?:_(\d+))?\.(?:out|error)$", filename)
        if isnothing(m)
            continue
        end

        job_id = parse(Int, m.captures[1])
        task_id = isnothing(m.captures[2]) ? nothing : parse(Int, m.captures[2])
        file_path = joinpath(log_dir, filename)

        try
            open(file_path, "r") do io
                line_count = 0
                for line in eachline(io)
                    line_count += 1
                    if line_count > max_lines_per_file
                        break
                    end
                    if !occursin("Processing:", line)
                        continue
                    end
                    parts = split(line, "Processing:", limit=2)
                    length(parts) == 2 || continue
                    sim_path = strip(parts[2])
                    isempty(sim_path) && continue

                    key = _normalize_lookup_path(sim_path)
                    entry = Dict{String, Any}(
                        "job_id" => job_id,
                        "task_id" => task_id,
                        "log_file" => file_path,
                    )
                    if !haskey(index, key)
                        index[key] = Vector{Dict{String, Any}}()
                    end
                    push!(index[key], entry)
                end
            end
        catch e
            @warn "Failed scanning worker log file $file_path: $e"
        end
    end

    return index
end

"""
    _resolve_seed_job_trace(seed_data; log_index=Dict()) -> Dict

Which SLURM job ran a seed: a job id stored in its tracking info if there is one,
else the first match of its `dir_path` in `log_index` (`build_worker_log_index`).
`"job_id_source"` says where the id came from (`"stored:<key>"`, `"worker_log"`, `"missing"`).
"""
function _resolve_seed_job_trace(seed_data::AbstractDict; log_index::Dict{String, Vector{Dict{String, Any}}}=Dict{String, Vector{Dict{String, Any}}}())
    dir_path = get_from_any_dict(seed_data, "dir_path", nothing)
    info_path = get_from_any_dict(seed_data, "info_path", nothing)

    trace = Dict{String, Any}(
        "job_id" => nothing,
        "task_id" => nothing,
        "job_id_source" => "missing",
        "dir_path" => dir_path,
        "info_path" => info_path,
        "log_file" => nothing,
    )

    # Priority 1: explicitly stored IDs in metadata.
    for key in ["submitted_job_id", "job_id", "slurm_job_id", "slurm_array_job_id"]
        v = get_from_any_dict(seed_data, key, nothing)
        jid = _try_parse_int(v)
        if !isnothing(jid)
            trace["job_id"] = jid
            trace["job_id_source"] = "stored:$key"
            return trace
        end
    end

    # Priority 2: infer from worker logs if available.
    if dir_path isa String && !isempty(log_index)
        lookup_key = _normalize_lookup_path(dir_path)
        if haskey(log_index, lookup_key) && !isempty(log_index[lookup_key])
            match_entry = first(log_index[lookup_key])
            trace["job_id"] = get(match_entry, "job_id", nothing)
            trace["task_id"] = get(match_entry, "task_id", nothing)
            trace["log_file"] = get(match_entry, "log_file", nothing)
            trace["job_id_source"] = "worker_log"
        end
    end

    return trace
end

"""
    _get_seed_progress_details(sim_data; log_index=Dict()) -> (seed_details, target_num_measurements)

Progress of every seed of one parameter point: status, measurements done vs.
`num_measure`, worst correlation time and its observable, an estimate of the number of
independent measurements (measurements / worst τ), last save time and job trace.
A point without seeds gets a single `"no_seed"` entry.
"""
function _get_seed_progress_details(sim_data::AbstractDict; log_index::Dict{String, Vector{Dict{String, Any}}}=Dict{String, Vector{Dict{String, Any}}}())
    target_num_measurements = _try_parse_int(get_from_any_dict(sim_data, "num_measure", nothing))
    seed_details = Dict{String, Any}()

    simulations = get_from_any_dict(sim_data, "simulations", nothing)
    if simulations isa AbstractDict && !isempty(simulations)
        for (seed_key, seed_data_any) in simulations
            if !(seed_data_any isa AbstractDict)
                continue
            end
            seed_data = seed_data_any
            seed_str = string(seed_key)
            num_current = something(_try_parse_int(get_from_any_dict(seed_data, "num_current_measurments", 0)), 0)
            status = string(get_from_any_dict(seed_data, "status", "unknown"))

            means = get_from_any_dict(seed_data, "means", nothing)
            worst_tau, worst_observable = means isa AbstractDict ? _extract_worst_tau_from_means(means) : (nothing, nothing)

            completion_fraction = if isnothing(target_num_measurements) || target_num_measurements <= 0
                nothing
            else
                num_current / target_num_measurements
            end

            effective_measurements_estimate = if isnothing(worst_tau) || worst_tau <= 0
                Float64(num_current)
            else
                num_current / worst_tau
            end

            seed_details[seed_str] = Dict{String, Any}(
                "status" => status,
                "num_current_measurements" => num_current,
                "target_num_measurements" => target_num_measurements,
                "completion_fraction" => completion_fraction,
                "worst_tau" => worst_tau,
                "worst_observable" => worst_observable,
                "effective_measurements_estimate" => effective_measurements_estimate,
                "step_count" => get_from_any_dict(seed_data, "step_count", nothing),
                "last_save_time" => get_from_any_dict(seed_data, "last_save_time", nothing),
                "job_trace" => _resolve_seed_job_trace(seed_data; log_index=log_index),
            )
        end
    else
        # Fallback for non-seeded points.
        means = get_from_any_dict(sim_data, "means", nothing)
        worst_tau, worst_observable = means isa AbstractDict ? _extract_worst_tau_from_means(means) : (nothing, nothing)
        seed_details["no_seed"] = Dict{String, Any}(
            "status" => "unknown",
            "num_current_measurements" => target_num_measurements,
            "target_num_measurements" => target_num_measurements,
            "completion_fraction" => 1.0,
            "worst_tau" => worst_tau,
            "worst_observable" => worst_observable,
            "effective_measurements_estimate" => isnothing(target_num_measurements) ? nothing : Float64(target_num_measurements),
            "step_count" => nothing,
            "last_save_time" => get_from_any_dict(sim_data, "last_save_time", nothing),
            "job_trace" => _resolve_seed_job_trace(sim_data; log_index=log_index),
        )
    end

    return seed_details, target_num_measurements
end

"""
    _summarize_seed_progress(seed_details) -> Dict

Totals over the seeds of one point: finished/running/other seeds, total and effective
measurements, the worst τ (with its seed and observable), and the job id of each seed.
"""
function _summarize_seed_progress(seed_details::Dict{String, Any})
    seeds_total = length(seed_details)
    seeds_finished = 0
    seeds_running = 0
    seeds_other = 0
    total_measurements = 0
    total_effective_measurements = 0.0

    point_worst_tau = nothing
    point_worst_seed = nothing
    point_worst_observable = nothing

    job_ids_by_seed = Dict{String, Any}()

    for (seed_key, entry_any) in seed_details
        if !(entry_any isa AbstractDict)
            continue
        end
        entry = entry_any

        status = string(get(entry, "status", "unknown"))
        if status == "finished"
            seeds_finished += 1
        elseif status == "running"
            seeds_running += 1
        else
            seeds_other += 1
        end

        c_meas = get(entry, "num_current_measurements", 0)
        if c_meas isa Number
            total_measurements += Int(round(c_meas))
        end

        eff_meas = get(entry, "effective_measurements_estimate", 0.0)
        if eff_meas isa Number && isfinite(Float64(eff_meas))
            total_effective_measurements += Float64(eff_meas)
        end

        tau = get(entry, "worst_tau", nothing)
        if tau isa Number && isfinite(Float64(tau))
            tau_f = Float64(tau)
            if isnothing(point_worst_tau) || tau_f > point_worst_tau
                point_worst_tau = tau_f
                point_worst_seed = seed_key
                point_worst_observable = get(entry, "worst_observable", nothing)
            end
        end

        job_trace = get(entry, "job_trace", nothing)
        if job_trace isa AbstractDict
            jid = get(job_trace, "job_id", nothing)
            if !isnothing(jid)
                job_ids_by_seed[seed_key] = jid
            end
        end
    end

    return Dict{String, Any}(
        "seeds_total" => seeds_total,
        "seeds_finished" => seeds_finished,
        "seeds_running" => seeds_running,
        "seeds_other" => seeds_other,
        "total_measurements" => total_measurements,
        "total_effective_measurements_estimate" => total_effective_measurements,
        "point_worst_tau" => point_worst_tau,
        "point_worst_seed" => point_worst_seed,
        "point_worst_observable" => point_worst_observable,
        "job_ids_by_seed" => job_ids_by_seed,
    )
end

"""
    evaluate_seed_point_readiness(point_diag; min_completed_seeds=1, min_measurements_per_seed=0,
                                  min_effective_measurements_per_seed=0.0) -> Dict

Decide whether a parameter point can be declared finished: enough finished seeds, and
every seed with enough measurements and enough independent measurements.
Returns `"ready_to_declare_finished"`, the criteria, and the reasons.
"""
function evaluate_seed_point_readiness(point_diag::Dict{String, Any};
                                       min_completed_seeds::Int=1,
                                       min_measurements_per_seed::Int=0,
                                       min_effective_measurements_per_seed::Float64=0.0)
    ready = true
    reasons = String[]

    summary = get(point_diag, "summary", Dict{String, Any}())
    seed_details = get(point_diag, "seed_details", Dict{String, Any}())

    seeds_finished = get(summary, "seeds_finished", 0)
    if seeds_finished < min_completed_seeds
        ready = false
        push!(reasons, "finished seeds $(seeds_finished) < required $(min_completed_seeds)")
    end

    for (seed_key, entry_any) in seed_details
        if !(entry_any isa AbstractDict)
            continue
        end
        entry = entry_any
        c_meas = get(entry, "num_current_measurements", 0)
        if c_meas isa Number && c_meas < min_measurements_per_seed
            ready = false
            push!(reasons, "seed $(seed_key): measurements $(c_meas) < $(min_measurements_per_seed)")
        end

        eff_meas = get(entry, "effective_measurements_estimate", 0.0)
        if eff_meas isa Number && eff_meas < min_effective_measurements_per_seed
            ready = false
            push!(reasons, "seed $(seed_key): effective measurements $(round(eff_meas, digits=2)) < $(min_effective_measurements_per_seed)")
        end
    end

    if ready
        push!(reasons, "all readiness criteria passed")
    end

    return Dict{String, Any}(
        "ready_to_declare_finished" => ready,
        "criteria" => Dict(
            "min_completed_seeds" => min_completed_seeds,
            "min_measurements_per_seed" => min_measurements_per_seed,
            "min_effective_measurements_per_seed" => min_effective_measurements_per_seed,
        ),
        "reasons" => reasons,
    )
end

"""
    _build_leaf_seed_diagnostics(leaf; key_path=[], log_index=Dict(), include_readiness=true, ...) -> Dict

Seed diagnostics of one `get_sims` leaf: its position in the nested result (`key_path`),
β, μ, size, per-seed details, summary and (optionally) the readiness decision.
"""
function _build_leaf_seed_diagnostics(leaf::AbstractDict;
                                      key_path::Vector{Any}=Any[],
                                      log_index::Dict{String, Vector{Dict{String, Any}}}=Dict{String, Vector{Dict{String, Any}}}(),
                                      include_readiness::Bool=true,
                                      min_completed_seeds::Int=1,
                                      min_measurements_per_seed::Int=0,
                                      min_effective_measurements_per_seed::Float64=0.0)
    sim_data_any = get_from_any_dict(leaf, "sim_dict", nothing)
    if !(sim_data_any isa AbstractDict)
        return Dict{String, Any}("error" => "Leaf does not include a valid sim_dict")
    end

    sim_data = sim_data_any
    seed_details, target_num_measurements = _get_seed_progress_details(sim_data; log_index=log_index)
    summary = _summarize_seed_progress(seed_details)
    Lx, Ly = get_sim_dimensions(sim_data)

    point_diag = Dict{String, Any}(
        "group_path" => copy(key_path),
        "sim_num" => get_from_any_dict(leaf, "sim_num", "N/A"),
        "beta" => get_from_any_dict(leaf, "beta", get_from_any_dict(sim_data, "beta", nothing)),
        "mu" => get_from_any_dict(leaf, "mu", get_from_any_dict(sim_data, "mu", nothing)),
        "Lx" => Lx,
        "Ly" => Ly,
        "target_num_measurements_per_seed" => target_num_measurements,
        "seed_details" => seed_details,
        "summary" => summary,
    )

    if include_readiness
        point_diag["readiness"] = evaluate_seed_point_readiness(point_diag;
            min_completed_seeds=min_completed_seeds,
            min_measurements_per_seed=min_measurements_per_seed,
            min_effective_measurements_per_seed=min_effective_measurements_per_seed)
    end

    return point_diag
end

"""
    _diagnose_sims_node(node; key_path=[], ...) -> Dict

Walk a nested `get_sims` result and add `"seed_diagnostics"` to a copy of every leaf.
"""
function _diagnose_sims_node(node::AbstractDict;
                             key_path::Vector{Any}=Any[],
                             log_index::Dict{String, Vector{Dict{String, Any}}}=Dict{String, Vector{Dict{String, Any}}}(),
                             include_readiness::Bool=true,
                             min_completed_seeds::Int=1,
                             min_measurements_per_seed::Int=0,
                             min_effective_measurements_per_seed::Float64=0.0)
    if _is_get_sims_leaf(node)
        leaf_out = copy(node)
        leaf_out["seed_diagnostics"] = _build_leaf_seed_diagnostics(node;
            key_path=key_path,
            log_index=log_index,
            include_readiness=include_readiness,
            min_completed_seeds=min_completed_seeds,
            min_measurements_per_seed=min_measurements_per_seed,
            min_effective_measurements_per_seed=min_effective_measurements_per_seed)
        return leaf_out
    end

    out = Dict{Any, Any}()
    for (k, v) in node
        if v isa AbstractDict
            out[k] = _diagnose_sims_node(v;
                key_path=[key_path; k],
                log_index=log_index,
                include_readiness=include_readiness,
                min_completed_seeds=min_completed_seeds,
                min_measurements_per_seed=min_measurements_per_seed,
                min_effective_measurements_per_seed=min_effective_measurements_per_seed)
        else
            out[k] = v
        end
    end
    return out
end

"""
    diagnose_sims_seed_progress(sims; kwargs...)

Build diagnostics for each get_sims data point (leaf), preserving the same nested shape.

For each leaf adds key "seed_diagnostics" with:
- Per-seed measurements and progress
- Per-seed worst autocorrelation time and observable
- Point-level summary (including worst seed/observable)
- Job trace info (stored job id or inferred from worker logs)
- Optional readiness decision
"""
function diagnose_sims_seed_progress(sims::Dict{Any, Any};
                                     log_dir::Union{Nothing, String}=nothing,
                                     infer_job_from_worker_logs::Bool=true,
                                     include_readiness::Bool=true,
                                     min_completed_seeds::Int=1,
                                     min_measurements_per_seed::Int=0,
                                     min_effective_measurements_per_seed::Float64=0.0)
    log_index = Dict{String, Vector{Dict{String, Any}}}()
    if infer_job_from_worker_logs && !isnothing(log_dir)
        log_index = build_worker_log_index(log_dir)
    end

    return _diagnose_sims_node(sims;
        key_path=Any[],
        log_index=log_index,
        include_readiness=include_readiness,
        min_completed_seeds=min_completed_seeds,
        min_measurements_per_seed=min_measurements_per_seed,
        min_effective_measurements_per_seed=min_effective_measurements_per_seed)
end



"""
    find_closest_filling(sim_info, wanted_n; top_k=1, max_n_difference=0.05,
                         specified_L=nothing, specified_Ly=nothing) -> Dict

For every lattice size and every β, pick the parameter point whose measured <n> is
closest to `wanted_n` (only if |<n> - wanted_n| ≤ `max_n_difference`).

Returns L => β => leaf, where a leaf is Dict(`"sim_num"`, `"n_val"`, `"n_err"`, `"n_dist"`,
`"mu"`, `"beta"`, `"sim_dict"`). L is an Int for square lattices and `(Lx, Ly)` otherwise.
With `specified_L` the L level is dropped (β => leaf for that size only).
"""
function find_closest_filling(sim_info::Dict, wanted_n::Float64; top_k::Int=1, max_n_difference::Float64=0.05, specified_L=nothing, specified_Ly=nothing)
    #Group by L (if L is specified - give only this L)
    L_groups = group_by_L(sim_info; specified_L=specified_L, specified_Ly=specified_Ly)
    # Group simulations by beta for all Ls
    for (L_key, sim_info_L) in L_groups
        L_groups[L_key] = group_by_beta(sim_info_L, wanted_n)
    end


    # For each beta, find the closest filling simulations
    results = Dict{Any, Any}()
    for (L_key, beta_groups) in L_groups
        results[L_key] = Dict{Any, Any}()  # Initialize results for this L
        for (beta, sims) in beta_groups
            sorted_sims = sort(sims, by = x -> x["n_dist"])
            for sim in first(sorted_sims, min(top_k, length(sorted_sims)))
                sim["beta"] = beta

                # Only include if the closest n_dist is below a threshold (e.g., 0.05)
                if sim["n_dist"] <= max_n_difference
                    results[L_key][beta] = sim
                end
            end
        end
    end

    if specified_L !== nothing
        # If a specific L was requested, filter results to only include that L
        # For rectangular: use (Lx, Ly) tuple as key
        L_key = isnothing(specified_Ly) ? specified_L : (specified_L, specified_Ly)
        results = get(results, L_key, Dict{Any, Any}())
    end

    if isempty(results)
        println("No simulations found for the desired filling.")
    end
    return results
end




"""
    group_by_beta(sim_info, wanted_n; L=nothing, Ly=nothing) -> Dict

β => list of the points at that β that have a measured <n>, each as
Dict(`"sim_num"`, `"n_val"`, `"n_err"`, `"n_dist"` = |<n> - wanted_n|, `"mu"`, `"sim_dict"`).
`L`/`Ly` keep only one lattice size.
"""
function group_by_beta(sim_info::Dict{String, Any}, wanted_n::Float64; L=nothing, Ly=nothing)
    # Group simulations by beta
    beta_groups = Dict{Any, Any}()
    for (sim_num, sim_dict) in sim_info
        if sim_num == "updated_at"
            continue
        end

        if L!==nothing
            sim_Lx, sim_Ly = get_sim_dimensions(sim_dict)
            target_Ly = isnothing(Ly) ? L : Ly
            if sim_Lx != L || sim_Ly != target_Ly
                continue  # Skip simulations that do not match the desired lattice size
            end
        end

        beta = get(sim_dict, "beta", nothing)
        if !haskey(beta_groups, beta)
            beta_groups[beta] = []
        end

        means = get(sim_dict, "means", nothing)
        if isa(means, Dict) && haskey(means, "n")
            n_val = means["n"]["value"]
            n_err = means["n"]["error"]
            mu = get(sim_dict, "mu", nothing)
            if !isnothing(n_val) && !isnothing(beta)
                push!(beta_groups[beta], Dict("sim_num"=>sim_num, "n_val"=>n_val, "n_dist"=>abs(n_val - wanted_n), "n_err"=>n_err, "mu"=>mu, "sim_dict"=>sim_dict))
            end
        end
    end
    return beta_groups
end


"""
    group_by_L(sim_info; specified_L=nothing, specified_Ly=nothing) -> Dict

L => (sim_num => sim_dict) for the points that have means. L is an Int for square
lattices and `(Lx, Ly)` otherwise. `specified_L` alone means an L×L lattice.
"""
function group_by_L(sim_info::Dict{String, Any}; specified_L=nothing, specified_Ly=nothing)
    # Group simulations by (Lx, Ly) tuple
    L_groups = Dict{Any, Any}()
    for (sim_num, sim_dict) in sim_info
        if sim_num == "updated_at"
            continue
        end

        sim_Lx, sim_Ly = get_sim_dimensions(sim_dict)
        if isnothing(sim_Lx)
            continue  # Skip if dimensions not available
        end
        
        # Handle specified_L filtering
        if specified_L !== nothing
            if isnothing(specified_Ly)
                # Legacy: specified_L means square, so check Lx == Ly == specified_L
                if sim_Lx != specified_L || sim_Ly != specified_L
                    continue
                end
            else
                # Rectangular: check both dimensions
                if sim_Lx != specified_L || sim_Ly != specified_Ly
                    continue
                end
            end
        end
        
        # Use (Lx, Ly) tuple as key for rectangular, or just L for square
        L_key = (sim_Lx == sim_Ly) ? sim_Lx : (sim_Lx, sim_Ly)

        if !haskey(L_groups, L_key)
            L_groups[L_key] = Dict{String, Any}()
        end

        means = get(sim_dict, "means", nothing)
        if isa(means, Dict)
            L_groups[L_key][sim_num] = sim_dict
        end
    end
    return L_groups
end

"""
    group_by_X(sim_info, x_key; specified_L=nothing, specified_Ly=nothing,
               diag_method="exact", disorder_type=nothing) -> Dict

Group points by the value of parameter `x_key`: value => (sim_num => sim_dict).

Only points with `diag_method` (note the default `"exact"`) and, if given, with
`U_disorder == disorder_type` and size `specified_L`×`specified_Ly` are kept.
`x_key = "L"` groups by lattice size (Int for square, `(Lx, Ly)` otherwise).
"""
function group_by_X(sim_info::Dict{String, Any}, x_key::String; specified_L=nothing, specified_Ly=nothing, diag_method="exact", disorder_type=nothing)
    # Group simulations by a specified key
    X_groups = Dict{Any, Any}()
    for (sim_num, sim_dict) in sim_info
        if contains(sim_num, "updated_at")
            continue
        end

        # Handle L filtering with rectangular support
        if specified_L !== nothing
            sim_Lx, sim_Ly = get_sim_dimensions(sim_dict)
            target_Ly = isnothing(specified_Ly) ? specified_L : specified_Ly
            if sim_Lx != specified_L || sim_Ly != target_Ly
                continue  # Skip simulations that do not match the specified L
            end
        end

        if get(sim_dict, "diag_method", nothing) != diag_method
            continue  # Skip simulations that do not match the specified diagonalization method
        end

        if disorder_type !== nothing && get(sim_dict, "U_disorder", nothing) != disorder_type
            continue  # Skip simulations that do not match the specified disorder type
        end

        # Special handling for "L" key - use get_sim_dimensions for both old and new data
        if x_key == "L"
            sim_Lx, sim_Ly = get_sim_dimensions(sim_dict)
            # Use (Lx, Ly) tuple as key for rectangular, or just L for square
            x_val = isnothing(sim_Lx) ? nothing : ((sim_Lx == sim_Ly) ? sim_Lx : (sim_Lx, sim_Ly))
        else
            x_val = get(sim_dict, x_key, nothing)
        end

        if !haskey(X_groups, x_val)
            X_groups[x_val] = Dict{String, Any}()
        end

        X_groups[x_val][sim_num] = sim_dict
    end
    return X_groups
end

"""
    group_disorder(sim_info; wanted_n=nothing, keep_sims=true, specific_L=nothing,
                   closest_filling=false, diag_method="exact") -> Dict

Older grouping for potential-disorder (`V`) campaigns run at fixed μ:
V => L => β => Dict("mu" => μ => {sims, means}, "avraged" => [...]).

At each (V, L, β, μ) the seeds are averaged with a plain mean (error set to 0).
With `closest_filling` and `wanted_n`, `"closest_filling"` holds the μ whose <n> is
closest to `wanted_n`; with `keep_sims=false` only the averaged entries are kept.
"""
function group_disorder(sim_info::Dict{String, Any}; wanted_n=nothing, keep_sims=true, specific_L=nothing, closest_filling=false, diag_method="exact")
    # Group simulations by V, L, beta, and mu in this order
    grouped = Dict{Any, Any}()
    for (V, sims_V) in group_by_X(sim_info, "V", diag_method=diag_method)
        grouped[V] = Dict{Any, Any}()
        for (L, sims_L) in group_by_X(sims_V, "L")
            if specific_L !== nothing && L != specific_L
                continue  # Skip simulations that do not match the specified L
            end
            L_dict = Dict{Any, Any}()
            for (beta, sims_beta) in group_by_X(sims_L, "beta")
               L_dict[beta] = Dict{Any, Any}()
               L_dict[beta]["mu"] = Dict{Any, Any}()
               L_dict[beta]["avraged"] = []
               for (mu, sims_mu) in group_by_X(sims_beta, "mu")
                    seeds=[]
                    sim_nums = []
                    filenames = []
                    if !haskey(L_dict[beta]["mu"], mu)
                        L_dict[beta]["mu"][mu] = Dict{String, Any}()
                        L_dict[beta]["mu"][mu]["sims"] = []
                        L_dict[beta]["mu"][mu]["sims_means"] = Dict{String, Any}()
                    end

                    for (_, sim) in sims_mu
                        push!(L_dict[beta]["mu"][mu]["sims"], sim)
                        push!(seeds, sim["seed"])
                        push!(sim_nums, sim["sim_num"])
                        if haskey(sim, "filename")
                            push!(filenames, sim["filename"])
                        end
                        if haskey(sim, "means") && isa(sim["means"], Dict)
                            for (k, v) in sim["means"]
                                if !haskey(L_dict[beta]["mu"][mu]["sims_means"], k)
                                    L_dict[beta]["mu"][mu]["sims_means"][k] = Dict{String, Any}()
                                    L_dict[beta]["mu"][mu]["sims_means"][k]["values"] = []
                                    L_dict[beta]["mu"][mu]["sims_means"][k]["errors"] = []
                                end
                                push!(L_dict[beta]["mu"][mu]["sims_means"][k]["values"], v["value"])
                                push!(L_dict[beta]["mu"][mu]["sims_means"][k]["errors"], v["error"])
                            end
                        end
                    end
                    L_dict[beta]["mu"][mu]["means"] = Dict{String, Any}()
                    for (k, v) in L_dict[beta]["mu"][mu]["sims_means"]
                        L_dict[beta]["mu"][mu]["means"][k] = Dict{String, Any}()
                        L_dict[beta]["mu"][mu]["means"][k]["value"] = mean(v["values"])
                        L_dict[beta]["mu"][mu]["means"][k]["error"] = zeros(length(v["values"])) # for now, just set error to zero

                    end

                    avraged_dict = Dict{String, Any}("L"=> L, "beta" => beta, "mu" => mu, "V" => V, "means"=>L_dict[beta]["mu"][mu]["means"],
                            "seeds"=>seeds, "filenames"=>filenames, "sim_nums"=>sim_nums)

                    if haskey(L_dict[beta]["mu"][mu]["means"], "n") && typeof(wanted_n)== Float64
                        n = L_dict[beta]["mu"][mu]["means"]["n"]["value"]
                        n_dist = abs(n - wanted_n)
                        avraged_dict["n_dist"] = n_dist
                        avraged_dict["n"] = n
                    end

                    push!(L_dict[beta]["avraged"], avraged_dict)
                end

                if closest_filling && wanted_n !== nothing
                    # Sort the simulations by n_dist and keep only the closest filling
                    L_dict[beta]["avraged"] = sort(L_dict[beta]["avraged"], by = x -> x["n_dist"])
                    if length(L_dict[beta]["avraged"]) > 0
                        L_dict[beta]["closest_filling"] = first(L_dict[beta]["avraged"])
                    else
                        L_dict[beta]["closest_filling"] = nothing
                    end
                end

                if !keep_sims
                    # Remove the sims list if not needed
                    if closest_filling
                        avraged = copy(L_dict[beta]["closest_filling"])
                    else
                        avraged = copy(L_dict[beta]["avraged"])
                    end
                    
                    L_dict[beta] = avraged
                end
            end
            if specific_L !== nothing && L == specific_L
                grouped[V] = L_dict
            else
                grouped[V][L] = L_dict
            end
        end
    end
    return grouped
end

"""
    get_sims_disorder(sim_info, wanted_n, Lx, Ly=Lx; disorder_type="uniform", diag_method="K0") -> Dict

value => β => leaf (see `find_closest_filling`) for one lattice size.

Here `disorder_type` is the NAME of the parameter to group by (e.g. `"impurity"` groups by
the disorder strength p, `"V"` by the potential-disorder strength); a name that is not a
parameter puts every point under `nothing`.
"""
function get_sims_disorder(sim_info::Dict{String, Any}, wanted_n::Float64, Lx::Int64, Ly::Int64=Lx; disorder_type="uniform", diag_method="K0")
    specified_Ly = (Lx == Ly) ? nothing : Ly
    V_groups = group_by_X(sim_info, disorder_type; specified_L=Lx, specified_Ly=specified_Ly, diag_method=diag_method)
    sims = Dict{Any, Any}()
    for (V, V_sims) in V_groups
        sims[V] = find_closest_filling(V_sims, wanted_n, specified_L=Lx, specified_Ly=specified_Ly)
    end
    return sims
end



"""
    get_sims(sim_info, wanted_n, x_sort; specified_L=nothing, specified_Ly=nothing,
             disorder_type="uniform", diag_method="K0", max_n_difference=0.05) -> Dict
    get_sims(sim_info, wanted_n, x_sort, y_sort; ...)
    get_sims(sim_info, wanted_n, sort_keys::Vector{String}; ...)

Nested lookup of the points closest to filling `wanted_n`: group by the parameters
`x_sort` (then `y_sort`, or each of `sort_keys` in turn), then by β
(`find_closest_filling`). E.g. `get_sims(sim_info, 0.875, ["U", "L"])` gives
U => L => β => leaf. Only points with `U_disorder == disorder_type` and `diag_method` are used.
"""
function get_sims(sim_info::Dict{String, Any}, wanted_n::Float64, x_sort::String; specified_L=nothing, specified_Ly=nothing, disorder_type="uniform", diag_method="K0", max_n_difference::Float64=0.05)
    x_groups = group_by_X(sim_info, x_sort; specified_L=specified_L, specified_Ly=specified_Ly, diag_method=diag_method, disorder_type=disorder_type)
    xisL = x_sort=="L" ? true : false
    sims = Dict{Any, Any}()
    for (x, x_sims) in x_groups
        sims[x] = find_closest_filling(x_sims, wanted_n, specified_L=xisL ? x : specified_L, max_n_difference=max_n_difference)
    end
    return sims
end

function get_sims(sim_info::Dict{String, Any}, wanted_n::Float64, x_sort::String, y_sort::String; specified_L=nothing, specified_Ly=nothing, disorder_type="uniform", diag_method="K0", max_n_difference::Float64=0.05)
    x_groups = group_by_X(sim_info, x_sort; specified_L=specified_L, specified_Ly=specified_Ly, diag_method=diag_method, disorder_type=disorder_type)
    sims = Dict{Any, Any}()
    for (x, x_sims) in x_groups
        simsx = get_sims(x_sims, wanted_n, y_sort; specified_L=x_sort=="L" ? x : specified_L, specified_Ly=x_sort=="Ly" ? x : specified_Ly, disorder_type=disorder_type, diag_method=diag_method, max_n_difference=max_n_difference)
        sims[x] = simsx
    end
    return sims
end

function get_sims(sim_info::Dict{String, Any}, wanted_n::Float64, sort_keys::Vector{String}; specified_L=nothing, specified_Ly=nothing, disorder_type="uniform", diag_method="K0", max_n_difference::Float64=0.05)
    if isempty(sort_keys)
        # Base case: no more sorting keys, find closest filling
        return find_closest_filling(sim_info, wanted_n; specified_L=specified_L, specified_Ly=specified_Ly, max_n_difference=max_n_difference)
    end
    
    x_sort = first(sort_keys)
    remaining_keys = sort_keys[2:end]
    
    x_groups = group_by_X(sim_info, x_sort; specified_L=specified_L, specified_Ly=specified_Ly, diag_method=diag_method, disorder_type=disorder_type)
    sims = Dict{Any, Any}()
    
    for (x, x_sims) in x_groups
        new_specified_L = x_sort == "Lx" ? x : specified_L
        new_specified_Ly = x_sort == "Ly" ? x : specified_Ly
        new_specified_L, new_specified_Ly = (x_sort == "L") ? ( (isa(x, Tuple) ? x[1] : x), (isa(x, Tuple) ? x[2] : x) ) : (new_specified_L, new_specified_Ly)

        sims[x] = get_sims(x_sims, wanted_n, remaining_keys; specified_L=new_specified_L, specified_Ly=new_specified_Ly, disorder_type=disorder_type, diag_method=diag_method, max_n_difference=max_n_difference)
    end
    
    return sims
end

"""
    get_obs_vs_T2(sim_info, wanted_n, Lx, obs; Ly=Lx, V=0.0, diag_method="exact", disorder_type="impurity")

Same as the `sim_info` method of `get_obs_vs_T` (older copy).
"""
function get_obs_vs_T2(sim_info::Dict{String, Any}, wanted_n::Float64, Lx::Int64, obs::String; Ly::Int64=Lx, V=0.0, diag_method="exact", disorder_type="impurity")
    sims = get_sims_disorder(sim_info, wanted_n, Lx, Ly; diag_method=diag_method, disorder_type=disorder_type)
    if !haskey(sims, V)
        return nothing  # No simulations for this V
    end
    sim_V = sims[V]
    Ts = Float64[]
    obs_vals = []
    obs_errs = []
    for (beta, sim) in sort(collect(sim_V); by=x->x[1])
        if haskey(sim["sim_dict"], "means")
            push!(Ts, 1 / beta)
            push!(obs_vals, sim["sim_dict"]["means"][obs]["value"])
            push!(obs_errs, sim["sim_dict"]["means"][obs]["error"])
        end
    end
    return (Ts, obs_vals, obs_errs)
end

"""
    get_obs_vs_T(sim_info, wanted_n, Lx, obs; Ly=Lx, V=0.0, diag_method="exact", disorder_type="impurity")
    get_obs_vs_T(sims, obs)
    get_obs_vs_T(sims, obs, ix)
        -> (Ts, values, errors)

Seed-averaged observable `obs` against temperature T = 1/β, sorted by β.

`sims` is a β => leaf Dict (one curve of `get_sims`); with `ix`, component `ix` of a
vector observable is taken. The `sim_info` method first calls `get_sims_disorder` with
`disorder_type` (the parameter to group by) and takes the group whose value is `V`;
it returns `nothing` if there is none.
"""
function get_obs_vs_T(sim_info::Dict{String, Any}, wanted_n::Float64, Lx::Int64, obs::String; Ly::Int64=Lx, V=0.0, diag_method="exact", disorder_type="impurity")
    sims = get_sims_disorder(sim_info, wanted_n, Lx, Ly; diag_method=diag_method, disorder_type=disorder_type)
    if !haskey(sims, V)
        return nothing  # No simulations for this V
    end
    sim_V = sims[V]
    return get_obs_vs_T(sim_V, obs)
end

function get_obs_vs_T(sims::Dict{Any, Any},  obs::String)
    Ts = Float64[]
    obs_vals = []
    obs_errs = []
    for (beta, sim) in sort(collect(sims); by=x->x[1])
        if haskey(sim["sim_dict"], "means")
            push!(Ts, 1 / beta)
            push!(obs_vals, sim["sim_dict"]["means"][obs]["value"])
            push!(obs_errs, sim["sim_dict"]["means"][obs]["error"])
        end
    end
    return (Ts, obs_vals, obs_errs)
end

function get_obs_vs_T(sims::Dict{Any, Any},  obs::String, ix::Int)
    Ts = Float64[]
    obs_vals = []
    obs_errs = []
    for (beta, sim) in sort(collect(sims); by=x->x[1])
        if haskey(sim["sim_dict"], "means")
            push!(Ts, 1 / beta)
            push!(obs_vals, sim["sim_dict"]["means"][obs]["value"][ix])
            push!(obs_errs, sim["sim_dict"]["means"][obs]["error"][ix])
        end
    end
    return (Ts, obs_vals, obs_errs)
end

"""
    get_mu_vs_T(sims) -> (Ts, mus)
    get_mu_vs_T(sim_info, wanted_n, Lx; Ly=Lx, V=0.0, diag_method="K0-TCA", disorder_type="uniform")

Chemical potential against T = 1/β for a β => leaf Dict: the μ found by the μ search
(`convergence.best_mu`) where available, else the point's `mu`. The `sim_info` method
selects the curve as in `get_obs_vs_T`.
"""
function get_mu_vs_T(sims::Dict{Any, Any})
    Ts = Float64[]
    mus = Float64[]
    for (beta, sim) in sort(collect(sims); by=x->x[1])
        sd = get(sim, "sim_dict", nothing)
        convg = isa(sd, Dict) ? get(sd, "convergence", nothing) : nothing
        mu = isa(convg, Dict) ? get(convg, "best_mu", nothing) : nothing
        if isnothing(mu)
            mu = get(sim, "mu", nothing)
        end
        if !isnothing(beta) && !isnothing(mu)
            push!(Ts, 1 / beta)
            push!(mus, mu)
        end
    end
    return (Ts, mus)
end

function get_mu_vs_T(sim_info::Dict{String, Any}, wanted_n::Float64, Lx::Int64;
                     Ly::Int64=Lx, V=0.0, diag_method="K0-TCA", disorder_type="uniform")
    sims = get_sims_disorder(sim_info, wanted_n, Lx, Ly;
                             diag_method=diag_method, disorder_type=disorder_type)
    if !haskey(sims, V)
        return nothing
    end
    return get_mu_vs_T(sims[V])
end


"""
    get_obs_vs_X(sims, obs, beta) -> (Xs, values, errors)

Observable `obs` at fixed `beta` against the outer key X of a nested X => β => leaf Dict
(e.g. from `get_sims(sim_info, n, "U")`), sorted by X.
"""
function get_obs_vs_X(sims::Dict{Any, Any}, obs::String, beta::Float64)
    Xs = Float64[]
    obs_vals = []
    obs_errs = []
    for (x, sim) in sort(collect(sims); by=x->x[1])
        if haskey(sim, beta) && haskey(sim[beta]["sim_dict"], "means")
            push!(Xs, x)
            push!(obs_vals, sim[beta]["sim_dict"]["means"][obs]["value"])
            push!(obs_errs, sim[beta]["sim_dict"]["means"][obs]["error"])
        end
    end
    return (Xs, obs_vals, obs_errs)
end



begin ##### LDOS binning by local spectral gap (as in Sulangi et al. 2025)
    """
        get_SG(LDOSi, w_list) -> (ω_SG, LDOS(ω_SG))

    Local spectral gap of one site: the energy ω ≥ 0 of the largest LDOS peak, and its height.
    """
    function get_SG(LDOSi::Array{Float64,1}, w_list::Array{Float64,1})
        mask = w_list .>= 0.0
        max_ix = argmax(LDOSi[mask])
        w_max = w_list[mask][max_ix]
        return (w_max, LDOSi[mask][max_ix])
    end

    """
        get_all_SG(LDOS, w_list) -> Vector{Float64}

    Local spectral gap of every site. `LDOS[k]` is the LDOS of all sites at energy `w_list[k]`.
    """
    function get_all_SG(LDOS::Array{Array{Float64,1},1}, w_list::Array{Float64,1})
        N = length(LDOS[1])
        LDOS_data = hcat(LDOS...)
        w_max_list = zeros(N)
        for ix in 1:N
            w_max, _ = get_SG(LDOS_data[ix, :], w_list)
            w_max_list[ix] = w_max
        end
        return w_max_list
    end

    """
        assign_bins(x, edges) -> Vector{Int}

    Bin index of each value of `x`: the first `k` with `x ≤ edges[k]`.
    """
    function assign_bins(x, edges)
        [searchsortedfirst(edges, v)  for v in x]
    end


    """
        get_bin_assignments(LDOS, w_list; num_bins=10) -> Vector{Int}

    Put every site in one of `num_bins` bins by its local spectral gap
    (edges evenly spaced from 0 to the largest gap).
    """
    function get_bin_assignments(LDOS::Array{Array{Float64,1},1}, w_list::Array{Float64,1}; num_bins::Int64=10)
        w_max_list = get_all_SG(LDOS, w_list)
        edges = range(0.0, maximum(w_max_list), length=num_bins)
        assignments = assign_bins(w_max_list, edges)
        return assignments
    end


    """
        get_binned_LDOS(LDOS, w_list; num_bins=10, assignments=nothing) -> (binned_LDOS, assignments)

    Average the LDOS over the sites of each spectral-gap bin. `binned_LDOS[b]` is the mean
    LDOS(ω) of bin `b` (NaN if the bin is empty). Pass `assignments` to reuse the bins of
    another temperature.
    """
    function get_binned_LDOS(LDOS::Array{Array{Float64,1},1}, w_list::Array{Float64,1}; num_bins::Int64=10, assignments=nothing)
        if assignments === nothing
            assignments = get_bin_assignments(LDOS, w_list, num_bins=num_bins)
        end

        binned_LDOS = [zeros(length(w_list)) for _ in 1:num_bins]
        bin_count = zeros(Int, num_bins)
        LDOS_data = hcat(LDOS...)
        for ix in 1:length(LDOS[1])
            bin = assignments[ix]
            binned_LDOS[bin] .+= LDOS_data[ix,:]
            bin_count[bin] += 1
        end

        for b in 1:num_bins
            if bin_count[b] > 0
                binned_LDOS[b] .= binned_LDOS[b] ./ bin_count[b]
            else
                binned_LDOS[b] .= NaN
            end
        end

        return binned_LDOS, assignments
    end


    """
        get_binned_LDOS_all_temps(sims) -> Dict(β => binned_LDOS)
        get_binned_LDOS_all_temps(sim_info, L, wanted_n, V)

    Binned LDOS at every temperature, with the sites binned ONCE by their spectral gap at
    the lowest temperature (largest β), so each bin follows the same sites as T rises.
    Done per seed (disorder realization) and averaged over the seeds of the lowest-T point.
    The `sim_info` method selects the curve as `get_sims_disorder(sim_info, wanted_n, L)[V]`.
    """
    function get_binned_LDOS_all_temps(sim_info::Dict{String, Any}, L::Int64, wanted_n::Float64, V::Float64)
        sims = get_sims_disorder(sim_info, wanted_n, L)[V]
        betas = sort(collect(keys(sims)))
        max_beta = maximum(betas)
        optional_seeds = collect(keys(sims[max_beta]["sim_dict"]["simulations"]))
        binned_LDOS = Dict()

        for seed_ix in eachindex(optional_seeds)
            binned_LDOS_min_T, bin_assignments = get_binned_LDOS(sims[max_beta]["sim_dict"]["simulations"][optional_seeds[seed_ix]]["means"]["LDOS"]["value"], sims[max_beta]["sim_dict"]["w_list"]; num_bins=10)
            if !haskey(binned_LDOS, max_beta)
                binned_LDOS[max_beta] = binned_LDOS_min_T
            else
                binned_LDOS[max_beta] += binned_LDOS_min_T
            end
            
            for beta in betas
                if beta != max_beta && haskey(sims[beta]["sim_dict"]["simulations"], optional_seeds[seed_ix])
                    binned_LDOS_current = get_binned_LDOS(sims[beta]["sim_dict"]["simulations"][optional_seeds[seed_ix]]["means"]["LDOS"]["value"], sims[beta]["sim_dict"]["w_list"]; assignments=bin_assignments, num_bins=10)[1]
                    if !haskey(binned_LDOS, beta)
                        binned_LDOS[beta] = binned_LDOS_current
                    else
                        binned_LDOS[beta] .+= binned_LDOS_current
                    end
                end
            end
        end
        for (k, v) in binned_LDOS
            binned_LDOS[k] .= v ./ length(optional_seeds)
        end
        return binned_LDOS
    end

    function get_binned_LDOS_all_temps(sims::Dict{Any, Any})
        betas = sort(collect(keys(sims)))
        max_beta = maximum(betas)
        optional_seeds = collect(keys(sims[max_beta]["sim_dict"]["simulations"]))
        binned_LDOS = Dict()

        for seed_ix in eachindex(optional_seeds)
            binned_LDOS_min_T, bin_assignments = get_binned_LDOS(sims[max_beta]["sim_dict"]["simulations"][optional_seeds[seed_ix]]["means"]["LDOS"]["value"], sims[max_beta]["sim_dict"]["w_list"]; num_bins=10)
            if !haskey(binned_LDOS, max_beta)
                binned_LDOS[max_beta] = binned_LDOS_min_T
            else
                binned_LDOS[max_beta] += binned_LDOS_min_T
            end
            
            for beta in betas
                if beta != max_beta && haskey(sims[beta]["sim_dict"]["simulations"], optional_seeds[seed_ix])
                    binned_LDOS_current = get_binned_LDOS(sims[beta]["sim_dict"]["simulations"][optional_seeds[seed_ix]]["means"]["LDOS"]["value"], sims[beta]["sim_dict"]["w_list"]; assignments=bin_assignments, num_bins=10)[1]
                    if !haskey(binned_LDOS, beta)
                        binned_LDOS[beta] = binned_LDOS_current
                    else
                        binned_LDOS[beta] .+= binned_LDOS_current
                    end
                end
            end
        end
        for (k, v) in binned_LDOS
            binned_LDOS[k] .= v ./ length(optional_seeds)
        end
        return binned_LDOS
    end


end


"""
    sim_stat(sim_dict)

Print a short summary of one parameter point: size, β, <n>, method, disorder, and for
each seed its status, MC steps, measurements and run time.
"""
function sim_stat(sim_dict::Dict{String, Any})
    n = haskey(sim_dict["means"], "n") ? sim_dict["means"]["n"]["value"] : "NA"
    n_err = haskey(sim_dict["means"], "n") ? sim_dict["means"]["n"]["error"] : nothing
    L_str = format_dimensions(sim_dict)
    println("Lx×Ly = $(L_str), beta=$(sim_dict["beta"]), n=$(round(n, sigdigits=2))+-$(round(n_err, sigdigits=1))")
    println("diagonalization method: $(sim_dict["diag_method"]) $(sim_dict["diag_method"]=="TCluster" ? "with cluster size=$(get(sim_dict, "TCluster_size_x", get(sim_dict, "TCluster_size", "N/A")))×$(get(sim_dict, "TCluster_size_y", get(sim_dict, "TCluster_size", "N/A")))" : "")")
    println("disorder type: $(sim_dict["U_disorder"]) with impurity strength=$(sim_dict["impurity"])")
    println("Number of disorder realizations: $(length(keys(sim_dict["simulations"])))")
    for (key, val) in sim_dict["simulations"]
        elapsed_sec = Dates.canonicalize(val["last_save_time"] - val["start_time"])
        println("  seed=$key: status=$(val["status"]), MC steps=$(val["step_count"]), measurements=$(val["num_current_measurments"]), time=$((elapsed_sec))")
    end
end

"""
    print_uncorrelated_measurements(sim_dict::Dict{String, Any}, observable::String)

Print the number of uncorrelated measurements for a given observable in a simulation.
Shows correlation time, total measurements, and uncorrelated measurements for each seed,
plus the total across all seeds.

# Arguments
- `sim_dict`: Simulation dictionary containing the simulation data
- `observable`: Name of the observable (e.g., "n", "Delta_mean", "stiffness")
"""
function print_uncorrelated_measurements(sim_dict::Dict{String, Any}, observable::String)
    L_str = format_dimensions(sim_dict)
    println("="^60)
    println("Uncorrelated Measurements for: $observable")
    println("Simulation: Lx×Ly=$(L_str), beta=$(get(sim_dict, "beta", "N/A")), mu=$(get(sim_dict, "mu", "N/A"))")
    println("="^60)
    
    if !haskey(sim_dict, "simulations")
        println("No simulations found in this sim_dict.")
        return
    end
    
    total_uncorr = 0
    total_meas = 0
    seed_count = 0
    
    println("\nPer-seed breakdown:")
    println("-"^60)
    println("Seed | Correlation Time | Total Meas | Uncorrelated Meas")
    println("-"^60)
    
    for (seed, seed_data) in sim_dict["simulations"]
        if !haskey(seed_data, "means") || typeof(seed_data["means"]) == String
            println("$seed | No means data available")
            continue
        end
        
        means = seed_data["means"]
        if !haskey(means, observable)
            println("$seed | Observable '$observable' not found")
            continue
        end
        
        obs_data = means[observable]
        if typeof(obs_data) == String
            println("$seed | $obs_data")
            continue
        end
        
        τ = get(obs_data, "correlation_time", NaN)
        num_meas = get(obs_data, "num_measurements", 0)
        num_uncorr = get(obs_data, "num_uncorrelated_measurements", 0)
        
        # Handle vector outputs (for q-dependent observables)
        if typeof(τ) <: AbstractVector
            τ_str = "$(round(minimum(τ), digits=2))-$(round(maximum(τ), digits=2))"
        else
            τ_str = "$(round(τ, digits=2))"
        end
        
        if typeof(num_uncorr) <: AbstractVector
            num_uncorr_val = minimum(num_uncorr)
            num_uncorr_str = "$(minimum(num_uncorr))-$(maximum(num_uncorr))"
        else
            num_uncorr_val = num_uncorr
            num_uncorr_str = "$num_uncorr"
        end
        
        println("$seed | τ = $τ_str | $num_meas | $num_uncorr_str")
        
        total_uncorr += num_uncorr_val
        total_meas += num_meas
        seed_count += 1
    end
    
    println("-"^60)
    println("\nSummary:")
    println("  Number of seeds: $seed_count")
    println("  Total measurements: $total_meas")
    println("  Total uncorrelated measurements: $total_uncorr")
    println("="^60)
    
    return total_uncorr
end

"""
    print_uncorrelated_measurements(path::String, sim_num::Union{Int, String}, observable::String)

Print the number of uncorrelated measurements for a given observable in a simulation,
loading the simulation info from the given path.

# Arguments
- `path`: Path to the simulation results directory
- `sim_num`: Simulation number (can be Int or String)
- `observable`: Name of the observable (e.g., "n", "Delta_mean", "stiffness")
"""
function print_uncorrelated_measurements(path::String, sim_num::Union{Int, String}, observable::String)
    sim_info = load_sim_info(path)
    sim_key = "$sim_num"
    
    if !haskey(sim_info, sim_key)
        println("Simulation #$sim_num not found in sim_info.")
        return
    end
    
    print_uncorrelated_measurements(sim_info[sim_key], observable)
end



"""
    _poly_peak(T, y, deg; ngrid=1001) -> (T_peak, y_peak, ok)

Fit a degree-`deg` polynomial to (T, y) and find its maximum on a fine grid inside the
T range. `ok = false` when the maximum is on an edge of the range (no interior peak).
T is centered before the fit to keep the Vandermonde matrix well conditioned.
"""
function _poly_peak(T::AbstractVector, y::AbstractVector, deg::Int; ngrid::Int=1001)
    T0 = sum(T) / length(T)
    t  = collect(Float64, T) .- T0
    V  = reduce(hcat, (t .^ k for k in 0:deg))   # columns 1, t, t^2, ...
    coeffs = V \ collect(Float64, y)             # coeffs[k+1] = c_k
    tlo, thi = extrema(t)
    grid = range(tlo, thi; length=ngrid)
    best_j = 1; best_v = -Inf
    @inbounds for (j, tt) in enumerate(grid)
        v = coeffs[deg + 1]                      # Horner from the top coefficient
        for k in deg:-1:1
            v = v * tt + coeffs[k]
        end
        if v > best_v; best_v = v; best_j = j; end
    end
    ok = !(best_j == 1 || best_j == ngrid)
    return (grid[best_j] + T0, best_v, ok)
end

"""
    find_Tstar(sim; len=3, deg=min(4, len-1), nboot=2000, ngrid=1001, rng=...)

Locate T* as the maximum of χ_spin(T). A degree-`deg` polynomial is fit to a
window of `len` temperature points centered on the grid peak, and its vertex is
taken on a fine grid (`deg=2`, `len=3` reproduces the old 3-point parabola).

Returns `(T_star, chi_star, worked, T_star_err)`:
- `worked` is `false` when the peak sits on the T-range boundary or there are too
  few points to fit (no reliable interior peak).
- `T_star_err` is the bootstrap 1σ uncertainty on T*: each χ point is resampled
  from `N(value, MC error)` `nboot` times, the polynomial is refit, and the std
  of the resulting peak locations is returned. It is `nothing` when the fit did
  not work or χ has no error bars. (The extra 4th return value is ignored by
  callers that destructure only the first three.)
"""
function find_Tstar(sim::Dict{Any, Any}; len::Int=3, deg::Int=min(4, len - 1),
                    nboot::Int=2000, ngrid::Int=1001,
                    rng::AbstractRNG=Random.MersenneTwister(0))
    obs = get_obs_vs_T(sim, "chi_spin", 1)
    Ts = obs[1]
    if isempty(Ts)
        return (nothing, nothing, false, nothing)
    end
    chi_vals = collect(Float64, obs[2])
    chi_errs = collect(Float64, obs[3])
    n = length(chi_vals)
    max_ix = argmax(chi_vals)

    # Adapt the window and degree to however many points are actually available.
    win = min(len, n)
    d   = clamp(deg, 1, win - 1)

    # Too few points to fit, or the peak is on the T-range edge: return the bare
    # grid maximum with no error bar (unreliable / no bracketed peak).
    if win < 3 || d < 2 || max_ix == 1 || max_ix == n
        return (Ts[max_ix], chi_vals[max_ix], false, nothing)
    end

    # Window of `win` points centered on the grid peak (clamped into the array).
    half = win ÷ 2
    lo   = clamp(max_ix - half, 1, n - win + 1)
    idxs = lo:(lo + win - 1)
    T_fit   = collect(Float64, Ts[idxs])
    chi_fit = chi_vals[idxs]
    err_fit = chi_errs[idxs]

    # Central estimate from the measured data.
    T_star, chi_star, ok = _poly_peak(T_fit, chi_fit, d; ngrid=ngrid)
    if !ok
        return (T_star, chi_star, false, nothing)
    end

    # Bootstrap the peak location through the χ Monte-Carlo errors.
    boot = Float64[]
    for _ in 1:nboot
        chi_b = chi_fit .+ err_fit .* randn(rng, win)
        Tb, _, okb = _poly_peak(T_fit, chi_b, d; ngrid=ngrid)
        okb && push!(boot, Tb)
    end
    T_star_err = length(boot) >= 2 ? std(boot) : nothing
    return (T_star, chi_star, true, T_star_err)
end

"""
    get_rho_xx_slope_heatmap(sims::Dict{Any, Any}; obs_name="resistivity_tensor", rho_index=1)

Compute the local slope (derivative) of ρ_xx as a function of T for each magnetic field B.
Returns data suitable for creating a heatmap with B on x-axis and T on y-axis.

# Arguments
- `sims`: Dictionary keyed by B values, each containing simulations keyed by β (inverse temperature)
- `obs_name`: Name of the observable to use (default: "resistivity_tensor")
- `rho_index`: Index of the resistivity component to use (1 for ρ_xx, default)
- `B_range`: Tuple (B_min, B_max) to filter B values. Use `nothing` for no filtering (default)
- `T_range`: Tuple (T_min, T_max) to filter T values. Use `nothing` for no filtering (default)

# Returns
- `B_values`: Sorted array of magnetic field values (filtered if B_range specified)
- `T_values`: Sorted array of temperature values (filtered if T_range specified)
- `slope_matrix`: Matrix of slopes, where slope_matrix[i, j] is dρ_xx/dT at (T_values[i], B_values[j])
                  NaN is used for missing data points
"""
function get_rho_xx_slope_heatmap(sims::Dict{Any, Any}; obs_name::String="resistivity_tensor", rho_index::Int=1, 
                                  B_range::Union{Nothing, Tuple{Real, Real}}=nothing, 
                                  T_range::Union{Nothing, Tuple{Real, Real}}=nothing,
                                  forward_difference::Bool=false,
                                  logscale::Bool=false,
                                  e2_over_h_units::Bool=true)
    B_values_all = sort(collect(keys(sims)))
    
    # Filter B values if range specified
    if B_range !== nothing
        B_values = filter(b -> B_range[1] <= b <= B_range[2], B_values_all)
    else
        B_values = B_values_all
    end
    
    # Collect all unique T values across all B values
    all_Ts = Set{Float64}()
    rho_data = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()  # B => (Ts, rho_xx)
    
    for B in B_values
        res_tensor = get_obs_vs_T(sims[B], obs_name)
        if res_tensor === nothing || isempty(res_tensor[1])
            continue
        end
        Ts = res_tensor[1]
        rho_xx = [res_tensor[2][j][rho_index] for j in 1:length(res_tensor[2])]
        if e2_over_h_units
            rho_xx .= rho_xx ./ (2pi)
        end
        if logscale
            rho_xx[rho_xx .<= 0.0] .= NaN  # Avoid log of non-positive values
            rho_xx = log.(rho_xx)
        end
        rho_data[B] = (Ts, rho_xx)
        for T in Ts
            push!(all_Ts, T)
        end
    end
    
    T_values_all = sort(collect(all_Ts))
    
    # Filter T values if range specified
    if T_range !== nothing
        T_values = filter(t -> T_range[1] <= t <= T_range[2], T_values_all)
    else
        T_values = T_values_all
    end
    
    # For each B, compute slope at each T using finite differences
    # Slope at T_i is estimated as (rho(T_{i+1}) - rho(T_{i-1})) / (T_{i+1} - T_{i-1}) (central difference)
    # At boundaries, use forward/backward difference
    slope_matrix = fill(NaN, length(T_values), length(B_values))
    
    for (j, B) in enumerate(B_values)
        if !haskey(rho_data, B)
            continue
        end
        Ts, rho_xx = rho_data[B]
        n = length(Ts)
        
        if n < 2
            continue  # Need at least 2 points to compute slope
        end
        
        # Sort by T
        perm = sortperm(Ts)
        Ts_sorted = Ts[perm]
        rho_sorted = rho_xx[perm]
        if logscale
            Ts_sorted_calc = log.(Ts_sorted)
        else
            Ts_sorted_calc = Ts_sorted
        end
        
        # Compute slopes at each T point for this B
        for k in 1:n
            T_k = Ts_sorted[k]
            
            # Skip if T is outside the requested range
            if T_range !== nothing && (T_k < T_range[1] || T_k > T_range[2])
                continue
            end
            
            # Find the index of this T in the filtered T_values array
            i = findfirst(t -> t ≈ T_k, T_values)
            if i === nothing
                continue
            end
            
            # Compute slope using finite differences
            if k == 1
                # Forward difference
                slope = (rho_sorted[k+1] - rho_sorted[k]) / (Ts_sorted_calc[k+1] - Ts_sorted_calc[k])
            elseif k == n
                # Backward difference
                slope = (rho_sorted[k] - rho_sorted[k-1]) / (Ts_sorted_calc[k] - Ts_sorted_calc[k-1])
            else
                if forward_difference
                    # Forward difference
                    slope = (rho_sorted[k+1] - rho_sorted[k]) / (Ts_sorted_calc[k+1] - Ts_sorted_calc[k])
                else
                    # Central difference
                    slope = (rho_sorted[k+1] - rho_sorted[k-1]) / (Ts_sorted_calc[k+1] - Ts_sorted_calc[k-1])
                end
            end

            slope_matrix[i, j] = slope
        end
    end
    
    return (B_values, T_values, slope_matrix)
end

"""
    get_rho_xx_slope_heatmap(sims_Us::Dict{Any, Any}, U::Float64; kwargs...)

Convenience method: extract the simulations for a specific U value and compute the slope heatmap.
"""
function get_rho_xx_slope_heatmap(sims_Us::Dict{Any, Any}, U::Float64; kwargs...)
    if !haskey(sims_Us, U)
        error("U=$U not found in simulations dictionary")
    end
    return get_rho_xx_slope_heatmap(sims_Us[U]; kwargs...)
end