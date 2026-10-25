#=
Comprehensive Statistical Analysis Report Generator for Monte Carlo Simulations
===============================================================================
This script generates a complete statistical analysis report for a single simulation,
including:
1. Per-observable statistics (time series, histogram, cumulative mean, blocking, autocorrelation)
2. Vector observable plots (Lxx_qx/qy, conductivity vs ω, Phase_Corr, Ps_l)
3. Bootstrap-derived quantities (stiffness, Drude weight, resistivity)
4. Joint distributions (correlations between observables)
5. Complex mean field trajectory in the complex plane

Usage:
    include("simulation_statistics_report.jl")
    generate_report("path/to/simulation")
    # or with options:
    generate_report("path/to/simulation"; save_plots=true, output_dir="./report")
=#

# Force GR backend to render headless (no display/Qt needed) — essential for cluster jobs
ENV["GKSwstype"] = "nul"

using Plots, Statistics, JLD2, Dates, Measures, Printf
include("MC_Hubburd.jl")
include("statistics.jl")
include("sim_num_control.jl")

# ============================================================================
# SECTION 1: Data Loading
# ============================================================================

"""
Load simulation data from path, handling both single-seed and multi-seed structures.
"""
function load_simulation_data(path::String; seed_index::Int=1)
    path = fix_cluster_path(path)
    
    # Check if results.jld2 exists directly in path
    results_path = joinpath(path, "results.jld2")
    if isfile(results_path)
        println("Found results.jld2 in: $path")
        return load_results(path, to_load_checkpoint=false), path
    end
    
    # Check for seed folders
    seed_folders = filter(f -> startswith(f, "seed_") && isdir(joinpath(path, f)), readdir(path))
    if !isempty(seed_folders)
        println("Found $(length(seed_folders)) seed folder(s): $(seed_folders)")
        if seed_index > length(seed_folders)
            @warn "Requested seed_index=$seed_index but only $(length(seed_folders)) seeds available. Using first seed."
            seed_index = 1
        end
        selected_seed = seed_folders[seed_index]
        seed_path = joinpath(path, selected_seed)
        println("Loading data from: $seed_path")
        return load_results(seed_path, to_load_checkpoint=false), seed_path
    end
    
    error("No results.jld2 found in path: $path")
end

"""
    report_get_sim_field(sim_data, field, default=nothing)

`sim_data.field` if the struct has that field (old files may not), else `default`.
"""
function report_get_sim_field(sim_data, field::Symbol, default=nothing)
    if hasfield(typeof(sim_data), field)
        return getfield(sim_data, field)
    end
    return default
end

"""
    report_to_float_or_nothing(v)

`v` as a finite Float64, else `nothing`.
"""
function report_to_float_or_nothing(v)
    if v isa Number
        vf = Float64(v)
        return isfinite(vf) ? vf : nothing
    end
    return nothing
end

"""
    report_first_numeric(x)

The first finite number found in `x` (searching nested arrays and Dicts; real part of
complex numbers), else `nothing`.
"""
function report_first_numeric(x)
    if x isa Number
        xf = Float64(real(x))
        return isfinite(xf) ? xf : nothing
    elseif x isa AbstractArray
        for item in x
            out = report_first_numeric(item)
            if !isnothing(out)
                return out
            end
        end
    elseif x isa AbstractDict
        for (_, item) in x
            out = report_first_numeric(item)
            if !isnothing(out)
                return out
            end
        end
    end
    return nothing
end

"""
    report_resolve_simulation_root(actual_path) -> (simulation_root, seed_folder_name)

For a path ending in `seed_N`, its parent folder and "seed_N"; otherwise the path and `nothing`.
"""
function report_resolve_simulation_root(actual_path::String)
    path_norm = normpath(actual_path)
    base = basename(path_norm)
    if startswith(base, "seed_")
        return dirname(path_norm), base
    end
    return path_norm, nothing
end

"""
    report_load_sim_metadata(seed_dir) -> Dict{String,Any}

The contents of `seed_dir/sim.jld2`, or an empty Dict if missing or unreadable.
"""
function report_load_sim_metadata(seed_dir::String)
    sim_path = joinpath(seed_dir, "sim.jld2")
    if !isfile(sim_path)
        return Dict{String, Any}()
    end
    try
        raw = JLD2.load(sim_path)
        if raw isa AbstractDict
            return Dict{String, Any}(string(k) => v for (k, v) in raw)
        end
    catch e
        @warn "Failed to load seed sim metadata from $sim_path: $e"
    end
    return Dict{String, Any}()
end

"""
    report_build_seed_diagnostics(simulation_root; include_readiness=true, min_completed_seeds=1,
                                  min_measurements_per_seed=0, min_effective_measurements_per_seed=0.0)

Per-seed progress of one simulation (from each `seed_*` folder) and, optionally, the
readiness decision, for the report. `"available" => false` with a reason if there are
no seed folders.
"""
function report_build_seed_diagnostics(simulation_root::String;
                                       include_readiness::Bool=true,
                                       min_completed_seeds::Int=1,
                                       min_measurements_per_seed::Int=0,
                                       min_effective_measurements_per_seed::Float64=0.0)
    if !isdir(simulation_root)
        return Dict{String, Any}(
            "available" => false,
            "reason" => "Simulation root does not exist: $simulation_root",
        )
    end

    seed_names = sort(filter(f -> startswith(f, "seed_") && isdir(joinpath(simulation_root, f)), readdir(simulation_root)))
    if isempty(seed_names)
        return Dict{String, Any}(
            "available" => false,
            "reason" => "No seed_* folders found",
        )
    end

    seed_entries = Dict{String, Any}()
    target_candidates = Int[]

    for seed_name in seed_names
        seed_dir = joinpath(simulation_root, seed_name)
        seed_meta = report_load_sim_metadata(seed_dir)
        cache = load_means_cache(seed_dir)

        num_current = something(_try_parse_int(get_from_any_dict(seed_meta, "num_current_measurments", nothing)), 0)
        target_num = _try_parse_int(get_from_any_dict(seed_meta, "num_measure", nothing))
        if !isnothing(target_num)
            push!(target_candidates, target_num)
        end

        if cache !== nothing
            cache_n = _try_parse_int(get(cache, "num_measurements", nothing))
            if num_current <= 0 && !isnothing(cache_n)
                num_current = cache_n
            end
        end

        status = string(get_from_any_dict(seed_meta, "status", "unknown"))
        if lowercase(status) == "unknown"
            if !isnothing(target_num) && target_num > 0 && num_current >= target_num
                status = "finished"
            elseif num_current > 0
                status = "running"
            end
        end

        means_dict = if cache !== nothing && haskey(cache, "means") && cache["means"] isa AbstractDict
            cache["means"]
        else
            Dict{String, Any}()
        end

        seed_entry = Dict{String, Any}(
            "status" => status,
            "num_current_measurments" => num_current,
            "target_num_measurements" => target_num,
            "means" => means_dict,
            "dir_path" => seed_dir,
            "info_path" => joinpath(seed_dir, "sim.jld2"),
        )

        # Preserve useful metadata for later convergence and trace summaries.
        for key in [
            "mu", "target_n", "n_tol", "best_n", "best_density", "convergence",
            "submitted_job_id", "job_id", "slurm_job_id", "slurm_array_job_id", "last_save_time"
        ]
            value = get_from_any_dict(seed_meta, key, nothing)
            if !isnothing(value)
                seed_entry[key] = value
            end
        end

        seed_entries[seed_name] = seed_entry
    end

    point_target = isempty(target_candidates) ? nothing : maximum(target_candidates)
    sim_data_for_diag = Dict{String, Any}("simulations" => seed_entries)
    if !isnothing(point_target)
        sim_data_for_diag["num_measure"] = point_target
    end

    seed_details, target_num_measurements = _get_seed_progress_details(sim_data_for_diag)
    summary = _summarize_seed_progress(seed_details)

    point_diag = Dict{String, Any}(
        "seed_details" => seed_details,
        "summary" => summary,
    )

    readiness = include_readiness ? evaluate_seed_point_readiness(point_diag;
        min_completed_seeds=min_completed_seeds,
        min_measurements_per_seed=min_measurements_per_seed,
        min_effective_measurements_per_seed=min_effective_measurements_per_seed,
    ) : nothing

    return Dict{String, Any}(
        "available" => true,
        "simulation_root" => simulation_root,
        "seed_count" => length(seed_entries),
        "target_num_measurements_per_seed" => target_num_measurements,
        "seed_entries" => seed_entries,
        "seed_details" => seed_details,
        "summary" => summary,
        "readiness" => readiness,
        "criteria" => Dict(
            "min_completed_seeds" => min_completed_seeds,
            "min_measurements_per_seed" => min_measurements_per_seed,
            "min_effective_measurements_per_seed" => min_effective_measurements_per_seed,
        ),
    )
end

"""
    report_build_mu_convergence(sim_data, seed_diag, cached_means) -> Dict

Summary of the μ search across seeds (μ per seed, reached n, how many converged). Taken
from the cached means' `"mu_convergence"` if present, else from the seed diagnostics.
"""
function report_build_mu_convergence(sim_data, seed_diag::Dict{String, Any}, cached_means)
    if !isnothing(cached_means) && haskey(cached_means, "mu_convergence") && cached_means["mu_convergence"] isa AbstractDict
        out = Dict{String, Any}(string(k) => v for (k, v) in cached_means["mu_convergence"])
        out["source"] = "cached_means.mu_convergence"
        out["available"] = true
        return out
    end

    if !get(seed_diag, "available", false)
        return Dict{String, Any}(
            "available" => false,
            "reason" => "Seed diagnostics unavailable",
        )
    end

    seed_entries = get(seed_diag, "seed_entries", Dict{String, Any}())
    mu_list = Float64[]
    best_n_list = Float64[]
    seed_list = String[]
    converged_flags = Bool[]
    target_n_candidates = Float64[]
    n_tol_candidates = Float64[]

    for (seed_key, entry_any) in seed_entries
        if !(entry_any isa AbstractDict)
            continue
        end
        entry = entry_any
        push!(seed_list, string(seed_key))

        mu_val = report_to_float_or_nothing(get_from_any_dict(entry, "mu", nothing))
        if !isnothing(mu_val)
            push!(mu_list, mu_val)
        end

        target_n = report_to_float_or_nothing(get_from_any_dict(entry, "target_n", nothing))
        if !isnothing(target_n)
            push!(target_n_candidates, target_n)
        end

        n_tol = report_to_float_or_nothing(get_from_any_dict(entry, "n_tol", nothing))
        if !isnothing(n_tol)
            push!(n_tol_candidates, n_tol)
        end

        best_n = report_to_float_or_nothing(get_from_any_dict(entry, "best_n", nothing))
        if isnothing(best_n)
            best_n = report_to_float_or_nothing(get_from_any_dict(entry, "best_density", nothing))
        end

        means_dict = get_from_any_dict(entry, "means", nothing)
        if isnothing(best_n) && means_dict isa AbstractDict && haskey(means_dict, "n") && means_dict["n"] isa AbstractDict
            best_n = report_to_float_or_nothing(get(means_dict["n"], "value", nothing))
        end
        if !isnothing(best_n)
            push!(best_n_list, best_n)
        end

        converged = nothing
        conv_info = get_from_any_dict(entry, "convergence", nothing)
        if conv_info isa AbstractDict
            conv_val = get_from_any_dict(conv_info, "converged", nothing)
            if conv_val isa Bool
                converged = conv_val
            end
        end
        if isnothing(converged)
            if !isempty(target_n_candidates) && !isempty(best_n_list)
                target_n_local = last(target_n_candidates)
                best_n_local = last(best_n_list)
                tol_local = !isempty(n_tol_candidates) ? last(n_tol_candidates) : 0.02
                converged = abs(best_n_local - target_n_local) <= tol_local
            else
                converged = false
            end
        end
        push!(converged_flags, converged)
    end

    target_n = isempty(target_n_candidates) ? report_to_float_or_nothing(report_get_sim_field(sim_data, :target_n, nothing)) : mean(target_n_candidates)
    n_tol = isempty(n_tol_candidates) ? 0.02 : mean(n_tol_candidates)
    num_seeds = length(seed_list)
    num_converged = count(identity, converged_flags)

    return Dict{String, Any}(
        "available" => true,
        "source" => "derived_from_seed_metadata",
        "seed_list" => seed_list,
        "mu_list" => mu_list,
        "mu_mean" => isempty(mu_list) ? nothing : mean(mu_list),
        "mu_std" => length(mu_list) <= 1 ? 0.0 : std(mu_list),
        "target_n" => target_n,
        "best_n_list" => best_n_list,
        "best_n_mean" => isempty(best_n_list) ? nothing : mean(best_n_list),
        "best_n_std" => length(best_n_list) <= 1 ? 0.0 : std(best_n_list),
        "n_tol" => n_tol,
        "num_converged" => num_converged,
        "num_seeds" => num_seeds,
        "all_converged" => num_seeds > 0 && num_converged == num_seeds,
    )
end

"""
    report_build_transport_tensor_summary(means_dict, sim_data) -> Dict

Summary of the transport observables present in `means_dict` (fixed-δ and DC
conductivities, Drude weights, resistivity tensors), including the Δω window closest to
the mean level spacing. `"available" => false` if there are no means.
"""
function report_build_transport_tensor_summary(means_dict, sim_data)
    if isnothing(means_dict) || !(means_dict isa AbstractDict)
        return Dict{String, Any}(
            "available" => false,
            "reason" => "No means dictionary available",
        )
    end

    transport_keys = [
        "conductivity_regular_xx_fixed", "conductivity_regular_xy_fixed", "conductivity_regular_yx_fixed", "conductivity_regular_yy_fixed",
        "Lxx_D_fixed", "Lxy_D_fixed", "Lyx_D_fixed", "Lyy_D_fixed",
        "Drude_weight_fixed", "drude_dc_xx", "drude_dc_xy", "drude_dc_yx", "drude_dc_yy",
        "conductivity_dc_xx", "conductivity_dc_xy", "conductivity_dc_yx", "conductivity_dc_yy",
        "resistivity_tensor_fixed", "resistivity_dc_tensor", "resistivity_tensor",
    ]

    entries = Dict{String, Any}()
    for key in transport_keys
        if !haskey(means_dict, key)
            continue
        end
        entry = means_dict[key]
        if !(entry isa AbstractDict)
            continue
        end
        value = get(entry, "value", nothing)
        err = get(entry, "error", nothing)
        entries[key] = Dict(
            "value_shape" => string(typeof(value)),
            "error_shape" => string(typeof(err)),
            "sample_value" => report_first_numeric(value),
            "sample_error" => report_first_numeric(err),
            "num_measurements" => get(entry, "num_measurements", nothing),
            "num_uncorrelated_measurements" => get(entry, "num_uncorrelated_measurements", nothing),
            "correlation_time" => get(entry, "correlation_time", nothing),
        )
    end

    mls = if haskey(means_dict, "energy_spacing_mean") && means_dict["energy_spacing_mean"] isa AbstractDict
        report_to_float_or_nothing(get(means_dict["energy_spacing_mean"], "value", nothing))
    else
        nothing
    end

    delta_fixed = report_get_sim_field(sim_data, :delta_values_fixed, Float64[])
    dw_values = report_get_sim_field(sim_data, :dw_values, Float64[])
    delta_ix_mls = (!isnothing(mls) && delta_fixed isa AbstractVector && !isempty(delta_fixed)) ? argmin(abs.(Float64.(delta_fixed) .- 2.0 * mls)) : nothing
    dw_ix_mls = (!isnothing(mls) && dw_values isa AbstractVector && !isempty(dw_values)) ? argmin(abs.(Float64.(dw_values) .- mls)) : nothing

    return Dict{String, Any}(
        "available" => !isempty(entries),
        "num_entries" => length(entries),
        "entries" => entries,
        "energy_spacing_mean" => mls,
        "closest_delta_to_2MLS_index" => delta_ix_mls,
        "closest_delta_to_2MLS_value" => isnothing(delta_ix_mls) ? nothing : Float64(delta_fixed[delta_ix_mls]),
        "closest_dw_to_MLS_index" => dw_ix_mls,
        "closest_dw_to_MLS_value" => isnothing(dw_ix_mls) ? nothing : Float64(dw_values[dw_ix_mls]),
    )
end

"""
    report_build_sim_dict_for_extrapolation(sim_data, means_dict) -> Dict

A minimal sim_dict (means, Lx, Ly, δ and Δω grids) in the shape the δ → 0 extrapolation
functions of sim_num_control.jl expect.
"""
function report_build_sim_dict_for_extrapolation(sim_data, means_dict)
    out = Dict{String, Any}(
        "means" => means_dict,
        "Lx" => report_get_sim_field(sim_data, :Lx, nothing),
        "Ly" => report_get_sim_field(sim_data, :Ly, report_get_sim_field(sim_data, :Lx, nothing)),
    )

    delta_fixed = report_get_sim_field(sim_data, :delta_values_fixed, Float64[])
    delta_stiff = report_get_sim_field(sim_data, :delta_values_stiffness, Float64[])
    dw_values = report_get_sim_field(sim_data, :dw_values, Float64[])

    out["delta_fixed_values"] = delta_fixed isa AbstractVector ? Float64.(delta_fixed) : Float64[]
    out["delta_values_stiffness"] = delta_stiff isa AbstractVector ? Float64.(delta_stiff) : Float64[]
    out["dw_values"] = dw_values isa AbstractVector ? Float64.(dw_values) : Float64[]
    return out
end

"""
    report_generate_delta_extrapolation!(data, means_dict; save_plots=false, output_dir=nothing,
                                         min_delta_MLS=1.0, max_delta_MLS=nothing, min_points=3,
                                         force_full_range=false) -> Dict

Run the δ → 0 extrapolation of the fixed-δ observables for the report (on a copy of the
means), optionally saving the plots, and return the results and warnings.
`"available" => false` if there are no means.
"""
function report_generate_delta_extrapolation!(data::LatticeData, means_dict;
                                              save_plots::Bool=false,
                                              output_dir::Union{Nothing, String}=nothing,
                                              min_delta_MLS::Float64=1.0,
                                              max_delta_MLS::Union{Nothing, Float64}=nothing,
                                              min_points::Int=3,
                                              force_full_range::Bool=false)
    if isnothing(means_dict) || !(means_dict isa AbstractDict)
        return Dict{String, Any}(
            "available" => false,
            "reason" => "No means dictionary available for extrapolation",
        )
    end

    sim_dict = report_build_sim_dict_for_extrapolation(data.sim_data, deepcopy(means_dict))
    means_work = sim_dict["means"]
    warnings = String[]

    extrap_results = Dict{String, Any}()
    try
        extrap_results = extrapolate_all_delta_observables!(means_work, sim_dict;
            min_delta_MLS=min_delta_MLS,
            max_delta_MLS=max_delta_MLS,
            min_points=min_points,
            force_full_range=force_full_range,
            verbose=false,
        )
    catch e
        push!(warnings, "Extrapolation computation failed: $e")
        return Dict{String, Any}(
            "available" => false,
            "reason" => "Extrapolation computation failed",
            "warnings" => warnings,
        )
    end

    extrap_keys = sort([k for (k, v) in extrap_results if !isnothing(v)])
    plots_saved = String[]

    if save_plots && !isnothing(output_dir)
        extrap_dir = joinpath(output_dir, "delta_extrapolation")
        mkpath(extrap_dir)

        for obs_key in extrap_keys
            result_entry = extrap_results[obs_key]
            value = get(result_entry, "value", nothing)
            n_components = value isa AbstractVector ? length(value) : 1
            selected_components = if n_components <= 1
                Int[]
            elseif n_components <= 3
                collect(1:n_components)
            else
                unique([1, cld(n_components, 2), n_components])
            end

            if isempty(selected_components)
                try
                    fig = plot_delta_extrapolation(sim_dict, obs_key)
                    fig_path = joinpath(extrap_dir, "$(obs_key).png")
                    savefig(fig, fig_path)
                    push!(plots_saved, fig_path)
                catch e
                    push!(warnings, "Plot failed for $(obs_key): $e")
                end
            else
                for comp_ix in selected_components
                    try
                        fig = plot_delta_extrapolation(sim_dict, obs_key; component_idx=comp_ix)
                        fig_path = joinpath(extrap_dir, "$(obs_key)_component_$(comp_ix).png")
                        savefig(fig, fig_path)
                        push!(plots_saved, fig_path)
                    catch e
                        push!(warnings, "Plot failed for $(obs_key) component $(comp_ix): $e")
                    end
                end
            end
        end

        summary_path = joinpath(extrap_dir, "extrapolation_summary.txt")
        open(summary_path, "w") do io
            println(io, "="^90)
            println(io, "DELTA-TO-ZERO EXTRAPOLATION SUMMARY")
            println(io, "="^90)
            println(io, "Generated: $(Dates.now())")
            println(io, "Extrapolated observables: $(length(extrap_keys))")
            println(io, "")

            for obs_key in extrap_keys
                entry = extrap_results[obs_key]
                println(io, "-"^90)
                println(io, obs_key)
                println(io, "-"^90)
                println(io, "  value: $(get(entry, "value", "N/A"))")
                println(io, "  error: $(get(entry, "error", "N/A"))")
                println(io, "  fit_info: $(get(entry, "fit_info", "N/A"))")
                println(io, "  delta_values_used: $(get(entry, "delta_values_used", "N/A"))")
                println(io, "")
            end

            if !isempty(warnings)
                println(io, "WARNINGS")
                println(io, "-"^90)
                for w in warnings
                    println(io, "  - $w")
                end
            end
        end
    end

    return Dict{String, Any}(
        "available" => true,
        "num_extrapolated_observables" => length(extrap_keys),
        "extrapolated_observables" => extrap_keys,
        "results" => extrap_results,
        "means_with_extrapolation" => means_work,
        "plots_saved" => plots_saved,
        "warnings" => warnings,
    )
end

# ============================================================================
# SECTION 2: Per-Observable Statistical Analysis Functions
# ============================================================================

"""
Get observable data from measurements, handling both scalar and vector observables.
"""
function get_observable_data(measurements, obs_name::Symbol, num_meas::Int; vec_index::Union{Int, Nothing}=nothing)
    if !hasfield(typeof(measurements), obs_name)
        error("Observable $obs_name not found in measurements.")
    end
    
    obs_data = getfield(measurements, obs_name)
    
    if obs_data isa Vector{<:Number}
        return obs_data[1:num_meas]
    elseif obs_data isa Vector{<:AbstractVector}
        vec_len = length(obs_data[1])
        idx = isnothing(vec_index) ? 1 : vec_index
        if idx < 1 || idx > vec_len
            error("Vector index $idx out of bounds. Observable has $vec_len components.")
        end
        return [x[idx] for x in obs_data[1:num_meas]]
    else
        error("Unsupported observable data type: $(typeof(obs_data))")
    end
end

"""
Check if observable is vector-valued and return its length.
Skips 3-layer (fixed-delta) observables.
"""
function is_vector_observable(measurements, obs_name::Symbol)
    if !hasfield(typeof(measurements), obs_name)
        return false, 0
    end
    obs_data = getfield(measurements, obs_name)
    if obs_data isa Vector{<:AbstractVector} && !isempty(obs_data)
        # Detect 3-layer observables (e.g., _fixed/_delta): Vector{Vector{Vector{T}}}
        if obs_data[1] isa AbstractVector && !isempty(obs_data[1]) && obs_data[1][1] isa AbstractVector
            return false, 0  # Skip 3-layer observables
        end
        return true, length(obs_data[1])
    end
    return false, 0
end

"""
List all available observables in measurements.
Classifies into scalar, vector (2-layer), and skips 3-layer (fixed-delta) observables.
"""
function list_observables(measurements)
    scalar_obs = Symbol[]
    vector_obs = Dict{Symbol, Int}()
    for field in fieldnames(typeof(measurements))
        data = getfield(measurements, field)
        if data isa Vector{<:Number} && !isempty(data)
            push!(scalar_obs, field)
        elseif data isa Vector{<:AbstractVector} && !isempty(data)
            # Detect 3-layer observables (e.g., _fixed/_delta): Vector{Vector{Vector{T}}}
            if data[1] isa AbstractVector && !isempty(data[1]) && data[1][1] isa AbstractVector
                continue  # Skip 3-layer observables — handled separately
            end
            vector_obs[field] = length(data[1])
        end
    end
    return scalar_obs, vector_obs
end

"""
    plot_time_series(obs_data, obs_name; mc_start=1)

Plot 1: the measured series against the measurement index, with its mean (|x| for
complex data).
"""
function plot_time_series(obs_data::AbstractVector, obs_name::String; mc_start=1)
    n = length(obs_data)
    mc_steps = mc_start:(mc_start + n - 1)
    
    # Handle complex data
    plot_data = eltype(obs_data) <: Complex ? abs.(obs_data) : obs_data
    
    fig = plot(mc_steps, plot_data, 
        xlabel="MC Step", 
        ylabel=obs_name,
        title="$obs_name vs MC Steps",
        legend=:topright,
        linewidth=0.5,
        alpha=0.7,
        size=(900, 400),
        leftmargin=10mm,
        bottommargin=10mm)
    
    mean_val = mean(plot_data)
    hline!(fig, [mean_val], color=:red, linewidth=2, linestyle=:dash, 
           label="Mean = $(round(mean_val, sigdigits=4))")
    
    return fig
end

"""
    plot_histogram(obs_data, obs_name; nbins=50, tau_mult=2.0)

Plot 2: histograms of the raw values and of block means with block size τ and
`tau_mult`·τ (blocks narrow the distribution as they decorrelate).
"""
function plot_histogram(obs_data::AbstractVector, obs_name::String; nbins=50, tau_mult=2.0)
    # Handle complex data
    plot_data = eltype(obs_data) <: Complex ? abs.(obs_data) : obs_data
    n = length(plot_data)
    
    tau = find_correlation_time(plot_data)
    
    # Block sizes to show
    block_sizes = [1]
    tau_block = max(1, round(Int, tau))
    if tau_block > 1 && tau_block < n ÷ 4
        push!(block_sizes, tau_block)
    end
    large_block = max(1, round(Int, tau_mult * tau))
    if large_block > tau_block && large_block < n ÷ 4
        push!(block_sizes, large_block)
    end
    
    n_plots = length(block_sizes)
    fig = plot(layout=(1, n_plots), size=(400*n_plots, 400), leftmargin=10mm, bottommargin=10mm)
    colors = [:blue, :green, :purple, :orange]
    
    for (i, bs) in enumerate(block_sizes)
        if bs == 1
            data_to_plot = plot_data
            title_str = "Raw Data (bs=1)"
        else
            result = get_block_means(plot_data, bs)
            if isnothing(result) || isnothing(result[1])
                continue
            end
            data_to_plot = result[1]
            title_str = "Block Size = $bs (n=$(length(data_to_plot)))"
        end
        
        block_mean = mean(data_to_plot)
        block_std = std(data_to_plot)
        block_se = block_std / sqrt(length(data_to_plot))
        
        histogram!(fig[i], data_to_plot,
            xlabel=obs_name, ylabel="Count", title=title_str,
            bins=min(nbins, max(10, length(data_to_plot) ÷ 5)),
            alpha=0.7, label="Data", color=colors[mod1(i, length(colors))])
        
        vline!(fig[i], [block_mean], color=:red, linewidth=2, label="μ=$(round(block_mean, sigdigits=4))")
        vline!(fig[i], [block_mean - block_std, block_mean + block_std], 
               color=:orange, linewidth=2, linestyle=:dash, label="±σ")
        scatter!(fig[i], [NaN], [NaN], label="SE=$(round(block_se, sigdigits=3))")
    end
    
    return fig
end

"""
    plot_cumulative_mean(obs_data, obs_name; mc_start=1)

Plot 3: the running mean with a naive standard-error ribbon (shows whether the mean
has settled).
"""
function plot_cumulative_mean(obs_data::AbstractVector, obs_name::String; mc_start=1)
    plot_data = eltype(obs_data) <: Complex ? abs.(obs_data) : obs_data
    n = length(plot_data)
    mc_steps = mc_start:(mc_start + n - 1)
    
    cum_mean = cumsum(plot_data) ./ (1:n)
    cum_var = zeros(n)
    for i in 2:n
        cum_var[i] = var(plot_data[1:i]) / i
    end
    cum_err = sqrt.(cum_var)
    
    fig = plot(mc_steps, cum_mean,
        xlabel="MC Step", ylabel="Cumulative Mean of $obs_name",
        title="Cumulative Mean vs MC Steps",
        linewidth=2, ribbon=cum_err, fillalpha=0.3,
        label="Cumulative mean ± SE", legend=:topright,
        size=(900, 400), leftmargin=10mm, bottommargin=10mm)
    
    final_mean = mean(plot_data)
    hline!(fig, [final_mean], color=:red, linewidth=2, linestyle=:dash, 
           label="Final mean = $(round(final_mean, sigdigits=4))")
    
    return fig
end

"""
    plot_error_vs_binsize(obs_data, obs_name; min_blocks=4, tau_mult=2.0)

Plot 4, blocking analysis: the error of the mean against block size (powers of 2, at
least `min_blocks` blocks). It should rise and level off once blocks exceed τ.
"""
function plot_error_vs_binsize(obs_data::AbstractVector, obs_name::String; min_blocks=4, tau_mult=2.0)
    plot_data = eltype(obs_data) <: Complex ? abs.(obs_data) : obs_data
    n = length(plot_data)
    max_block_size = div(n, min_blocks)
    
    max_exp = Int(floor(log2(max_block_size)))
    block_sizes = [2^i for i in 0:max_exp]
    
    block_errors = Float64[]
    for bs in block_sizes
        result = get_block_means(plot_data, bs)
        if result !== nothing && !isnan(result[2])
            push!(block_errors, sqrt(result[2]))
        else
            push!(block_errors, NaN)
        end
    end
    
    valid = .!isnan.(block_errors)
    valid_bs = block_sizes[valid]
    valid_err = block_errors[valid]
    
    if isempty(valid_bs)
        return plot(title="Not enough data for blocking analysis", size=(800, 400))
    end
    
    tau = find_correlation_time(plot_data)
    block_size_used = max(1, round(Int, tau_mult * tau))
    result_at_tau = get_block_means(plot_data, block_size_used)
    blocked_err_at_tau = isnothing(result_at_tau) ? NaN : sqrt(result_at_tau[2])
    raw_err = get_raw_std(plot_data)
    correlation_scheme_err = max(raw_err, isnan(blocked_err_at_tau) ? 0.0 : blocked_err_at_tau)
    
    fig = plot(valid_bs, valid_err,
        xlabel="Block Size", ylabel="Standard Error",
        title="Error vs Block Size for $obs_name\n(bs = $(tau_mult)τ = $(block_size_used))",
        xscale=:log10, marker=:circle, markersize=6, linewidth=2,
        label="Block error", size=(900, 400), legend=:bottomright,
        leftmargin=10mm, bottommargin=10mm)
    
    hline!(fig, [raw_err], color=:red, linewidth=2, linestyle=:dash, 
           label="Raw SE = $(round(raw_err, sigdigits=4))")
    
    if block_size_used >= minimum(valid_bs) && block_size_used <= maximum(valid_bs)
        vline!(fig, [block_size_used], color=:blue, linewidth=2, linestyle=:dot, 
               label="$(tau_mult)τ = $(block_size_used)")
    end
    
    if !isnan(blocked_err_at_tau)
        scatter!(fig, [block_size_used], [blocked_err_at_tau], 
                 color=:blue, markersize=10, markershape=:star5, 
                 label="Blocked SE @ $(tau_mult)τ = $(round(blocked_err_at_tau, sigdigits=4))")
    end
    
    hline!(fig, [correlation_scheme_err], color=:green, linewidth=3, linestyle=:solid, 
           label="USED ERROR = $(round(correlation_scheme_err, sigdigits=4))")
    
    # Try saturation fit
    try
        params, _, _, _ = fit_to_saturation(plot_data, min_blocks)
        if params[2] > 0 && params[3] > 0
            x_fit = range(minimum(valid_bs), maximum(valid_bs), length=100)
            y_fit = saturation_model(collect(x_fit), params)
            plot!(fig, x_fit, y_fit, color=:orange, linewidth=2, linestyle=:dashdot, 
                  alpha=0.7, label="Saturation fit")
        end
    catch
    end
    
    return fig
end

"""
    plot_autocorrelation(obs_data, obs_name; max_lag=nothing, C=5) -> (fig1, fig2)

Plots 5 and 6: the normalized autocorrelation C(k)/C(0), and the integrated τ_int(M)
against the cutoff M with the self-consistent window τ ≤ M/`C`.
"""
function plot_autocorrelation(obs_data::AbstractVector, obs_name::String; max_lag=nothing, C=5)
    plot_data = eltype(obs_data) <: Complex ? abs.(obs_data) : obs_data
    n = length(plot_data)
    
    if isnothing(max_lag)
        max_lag = min(div(n, 4), 1000)
    end
    
    chi_0 = chi_k(plot_data, 0)
    if chi_0 == 0
        fig1 = plot(
            xlabel="Lag k (MC steps)", ylabel="C(k)/C(0)",
            title="Autocorrelation Function - $obs_name \n Zero variance in data; cannot compute autocorrelation.",
            linewidth=2, label="C(k)/C(0)",
            size=(600, 500), leftmargin=10mm, bottommargin=10mm)
        fig2 = plot(xlabel="Cutoff M", ylabel="τ_int(M)",
            title="Integrated τ (self-consistent: τ ≤ M/$(C)) - $obs_name \n Zero variance in data; cannot compute autocorrelation.",
            linewidth=2, label="τ_int(M)",
            size=(600, 500), leftmargin=10mm, bottommargin=10mm)
        return fig1, fig2
    end
    lags = 0:max_lag
    autocorr = [chi_k(plot_data, k) / chi_0 for k in lags]
    tau = find_correlation_time(plot_data; C=C)
    
    tau_integrated = zeros(max_lag + 1)
    tau_integrated[1] = 1.0
    for k in 1:max_lag
        tau_integrated[k+1] = 1.0 + 2*sum(autocorr[2:k+1])
    end
    
    num_uncorr = Int(floor(n / tau))
    
    # Plot 1: Autocorrelation function
    fig1 = plot(lags, autocorr,
        xlabel="Lag k (MC steps)", ylabel="C(k)/C(0)",
        title="Autocorrelation Function - $obs_name",
        linewidth=2, label="C(k)/C(0)",
        size=(600, 500), leftmargin=10mm, bottommargin=10mm)
    
    hline!(fig1, [0], color=:black, linewidth=1, linestyle=:dash, label=nothing)
    
    if tau > 1 && tau < max_lag
        vline!(fig1, [tau], color=:red, linewidth=2, linestyle=:dash, 
               label="τ = $(round(tau, sigdigits=3))")
        exp_decay = exp.(-lags ./ tau)
        plot!(fig1, lags, exp_decay, color=:orange, linewidth=2, linestyle=:dot, 
              alpha=0.7, label="exp(-k/τ)")
    end
    
    # Plot 2: Integrated autocorrelation time
    max_tau_int = maximum(tau_integrated)
    y_upper = max(2*tau, max_tau_int * 1.2, 5.0)
    
    fig2 = plot(lags, tau_integrated,
        xlabel="Cutoff M", ylabel="τ_int(M)",
        title="Integrated τ (self-consistent: τ ≤ M/$(C)) - $obs_name",
        linewidth=2, label="τ_int(M)",
        size=(600, 500), ylims=(0, y_upper),
        leftmargin=10mm, bottommargin=10mm)
    
    m_c_cutoff = min(max_lag, Int(floor(y_upper * C)))
    m_c_lags = 0:m_c_cutoff
    plot!(fig2, m_c_lags, m_c_lags ./ C, color=:gray, linewidth=1, linestyle=:dash, label="M/$(C)")
    
    if tau > 1 && tau < max_lag
        hline!(fig2, [tau], color=:red, linewidth=1, linestyle=:dot, label="τ = $(round(tau, sigdigits=3))")
    end
    
    annotate!(fig2, [(max_lag*0.7, y_upper*0.85, text("N_eff = N/τ ≈ $(num_uncorr)", 10, :right))])
    
    return fig1, fig2
end

"""
    analyze_single_observable(measurements, obs_name, num_meas; vec_index=nothing, tau_mult=2.0,
                              show_plots=true, analyze_all_ixes=false) -> (stats, plots)

Statistics (mean, errors, τ, …) and the six diagnostic plots of one observable, or of
one component `vec_index` of a vector observable. With `analyze_all_ixes`, every
component is analysed and a Dict `index => (stats, plots)` is returned.
"""
function analyze_single_observable(measurements, obs_name::Symbol, num_meas::Int; 
                                   vec_index::Union{Int, Nothing}=nothing, tau_mult::Float64=2.0,
                                   show_plots::Bool=true, analyze_all_ixes::Bool=false)
    is_vec, vec_len = is_vector_observable(measurements, obs_name)
    if is_vec && analyze_all_ixes
        results = Dict{Int, Any}()
        for ix in 1:vec_len
            stats, plots = analyze_single_observable(measurements, obs_name, num_meas; 
                                                    vec_index=ix, tau_mult=tau_mult,
                                                    show_plots=show_plots, analyze_all_ixes=false)
            results[ix] = (stats=stats, plots=plots)
        end
        return results
    end
    idx_str = is_vec ? "[$(isnothing(vec_index) ? 1 : vec_index)/$vec_len]" : ""
    
    obs_data = get_observable_data(measurements, obs_name, num_meas; vec_index=vec_index)
    obs_str = "$(obs_name)$(idx_str)"
    
    # Handle complex data for statistics
    stats_data = eltype(obs_data) <: Complex ? abs.(obs_data) : obs_data
    
    # Calculate statistics
    mean_val = mean(stats_data)
    raw_err = get_raw_std(stats_data)
    tau = find_correlation_time(stats_data)
    num_uncorr = Int(floor(num_meas / max(1, tau)))
    block_size_used = max(1, round(Int, tau_mult * tau))
    result_blocked = get_block_means(stats_data, block_size_used)
    blocked_err = isnothing(result_blocked) ? raw_err : sqrt(result_blocked[2])
    if isnan(blocked_err)
        blocked_err = raw_err
    end
    final_err = max(raw_err, blocked_err)
    
    stats_dict = Dict(
        "name" => obs_str,
        "mean" => mean_val,
        "raw_error" => raw_err,
        "blocked_error" => blocked_err,
        "final_error" => final_err,
        "tau" => tau,
        "block_size" => block_size_used,
        "num_measurements" => num_meas,
        "num_uncorrelated" => num_uncorr
    )
    
    if !show_plots
        return stats_dict, nothing
    end
    
    # Generate plots
    p1 = plot_time_series(obs_data, obs_str)
    p2 = plot_histogram(stats_data, obs_str; tau_mult=tau_mult)
    p3 = plot_cumulative_mean(obs_data, obs_str)
    p4 = plot_error_vs_binsize(stats_data, obs_str; tau_mult=tau_mult)
    p5, p6 = plot_autocorrelation(stats_data, obs_str)
    
    plots = (p1, p2, p3, p4, p5, p6)
    return stats_dict, plots
end

# ============================================================================
# SECTION 2B: Fixed-Delta Observable Analysis
# ============================================================================

"""
New Extractor: Extract time series for a specific delta (and omega) index.
Handles:
  - 3-layer: [MC_step][Delta_idx][Omega_idx] (e.g., conductivity, Lxx_w)
  - 2-layer: [MC_step][Delta_idx] (e.g., Drude weight)
"""
function get_observable_data_fixed_delta(measurements, obs_name::Symbol, num_meas::Int, 
                                       delta_idx::Int; w_idx::Union{Int, Nothing}=nothing)
    if !hasfield(typeof(measurements), obs_name)
        error("Observable $obs_name not found.")
    end
    
    # Get the full raw data: Vector{Array}
    raw_data = getfield(measurements, obs_name)
    
    # Validation
    if num_meas > length(raw_data)
        num_meas = length(raw_data)
    end
    
    # Check structure of the first measurement
    first_meas = raw_data[1]
    
    # CASE A: 2-Layer [Delta_idx] (e.g., Drude Weight)
    # The inner array is just a Vector{Number} where length = num_deltas
    if first_meas isa Vector{<:Number}
        if w_idx !== nothing
            @warn "w_idx provided for scalar-per-delta observable $obs_name. Ignoring w_idx."
        end
        return [raw_data[i][delta_idx] for i in 1:num_meas]
        
    # CASE B: 3-Layer [Delta_idx][Omega_idx] (e.g., Conductivity)
    # The inner array is Vector{Vector{Number}}
    elseif first_meas isa Vector{<:AbstractVector}
        if w_idx === nothing
            error("w_idx is required for frequency-dependent observable $obs_name")
        end
        return [raw_data[i][delta_idx][w_idx] for i in 1:num_meas]
    else
        error("Unknown structure for fixed delta observable: $(typeof(first_meas))")
    end
end

"""
Generate comparative plots for Fixed-Delta observables.
1. Spectrum evolution: Value vs Omega (colored by Delta)
2. Delta sensitivity: Value vs Delta (at specific frequencies)
"""
function plot_fixed_delta_comparisons(data::LatticeData, obs_name::Symbol, 
                                    delta_values::Vector{Float64}, w_list::Vector{Float64};
                                    cached_means=nothing)
    measurements = data.measurement_data
    num_meas = data.current_state.num_current_measurments
    
    # Determine dimensionality (2-layer or 3-layer)
    is_freq_dep = false
    raw_sample = getfield(measurements, obs_name)[1]
    if raw_sample isa Vector{<:AbstractVector}
        is_freq_dep = true
    end
    
    plots_dict = Dict{String, Any}()
    
    # --- Calculation Phase: Get Means ---
    num_deltas = length(delta_values)
    num_ws = is_freq_dep ? length(w_list) : 1
    
    means_matrix = zeros(Float64, num_deltas, num_ws)
    errs_matrix = zeros(Float64, num_deltas, num_ws)
    
    obs_key = string(obs_name)
    if !isnothing(cached_means) && haskey(cached_means, obs_key) && isa(cached_means[obs_key], Dict)
        # Use cached values
        cm = cached_means[obs_key]
        cv = cm["value"]
        ce = cm["error"]
        if is_freq_dep
            # cv is Vector{Vector} indexed [delta_idx][omega_idx]
            for d_i in 1:min(num_deltas, length(cv))
                for w_i in 1:min(num_ws, length(cv[d_i]))
                    means_matrix[d_i, w_i] = real(cv[d_i][w_i])
                    errs_matrix[d_i, w_i] = abs(real(ce[d_i][w_i]))
                end
            end
        else
            # cv is Vector indexed [delta_idx]
            for d_i in 1:min(num_deltas, length(cv))
                means_matrix[d_i, 1] = real(isa(cv[d_i], AbstractVector) ? cv[d_i][1] : cv[d_i])
                errs_matrix[d_i, 1] = abs(real(isa(ce[d_i], AbstractVector) ? ce[d_i][1] : ce[d_i]))
            end
        end
    else
        # Recompute from raw data
        for d_i in 1:num_deltas
            if is_freq_dep
                for w_i in 1:num_ws
                    ts = get_observable_data_fixed_delta(measurements, obs_name, num_meas, d_i; w_idx=w_i)
                    m = mean(ts)
                    e = std(ts) / sqrt(length(ts)) 
                    means_matrix[d_i, w_i] = m
                    errs_matrix[d_i, w_i] = abs(e)
                end
            else
                ts = get_observable_data_fixed_delta(measurements, obs_name, num_meas, d_i)
                m = mean(ts)
                e = std(ts) / sqrt(length(ts))
                means_matrix[d_i, 1] = m
                errs_matrix[d_i, 1] = abs(e)
            end
        end
    end
    
    # Handle Real/Complex for plotting
    plot_vals = eltype(means_matrix) <: Complex ? real.(means_matrix) : means_matrix
    ylabel_str = eltype(means_matrix) <: Complex ? "Re[$obs_name]" : string(obs_name)

    
    # --- Plot 1: Value vs Delta (Sensitivity) ---
    # For frequency dependent, we pick 3 representative frequencies: DC (index 1), Low, High
    indices_to_plot = is_freq_dep ? unique([1, min(2, num_ws), div(num_ws, 2), num_ws]) : [1]
    
    p1 = plot(xlabel="Broadening Γ (Delta)", ylabel=ylabel_str, 
              title="Sensitivity: $obs_name vs Γ", legend=:best, leftmargin=10mm, bottommargin=10mm)
    
    for w_i in indices_to_plot
        label_str = is_freq_dep ? "ω=$(round(w_list[w_i], digits=3))" : "Integrated/Static"
        plot!(p1, delta_values, plot_vals[:, w_i], yerr=errs_matrix[:, w_i], 
              label=label_str, marker=:circle, linewidth=2)
    end
    # plot mean_spacing line
    mean_spacing = mean(measurements.energy_spacing_mean)
    vline!(p1, [mean_spacing], color=:red, linestyle=:dash, linewidth=2, 
           label="Mean Spacing = $(round(mean_spacing, sigdigits=3))")
    plots_dict["vs_delta"] = p1
    
    # --- Plot 2: Value vs Omega (Spectrum Evolution) ---
    # Only if frequency dependent
    if is_freq_dep
        p2 = plot(xlabel="Frequency ω", ylabel=ylabel_str, 
                  title="Spectrum: $obs_name (varying Γ) \n mean_spacing=$(round(mean_spacing, sigdigits=3))", legend=:topright,
                  leftmargin=10mm, bottommargin=10mm)
        
        # Plot a subset of deltas to avoid clutter (e.g., min, mid, max)
        d_indices = unique([1, div(num_deltas, 2), num_deltas])
        if length(delta_values) <= 5; d_indices = 1:length(delta_values); end
        
        for d_i in d_indices
            plot!(p2, w_list, plot_vals[d_i, :], 
                  label="Γ=$(round(delta_values[d_i], digits=3))", linewidth=2, m=:circle)
        end
        plots_dict["vs_omega"] = p2
    end
    
    return plots_dict
end

"""
Comprehensive analysis for Fixed-Delta derived observables.
Instead of analyzing the raw complex L*_w_fixed, we derive two real quantities per component:
  - Drude weight:          L_D(δ) = real(L*_w_fixed[i][d][1])
  - Conductivity regular:  σ(δ)   = imag(L*_w_fixed[i][d][1]) / ω₁
and run the standard 6-plot statistical analysis on each per-delta time series.
Also generates summary comparison plots (vs delta, vs omega) and derived conductivity/Drude plots.
"""
function analyze_fixed_delta_group(data::LatticeData, obs_list::Vector{Symbol}, 
                                   output_base_dir::String; tau_mult=10.0, cached_means=nothing)
    
    sim_data = data.sim_data
    measurements = data.measurement_data
    num_meas = data.current_state.num_current_measurments
    
    # Get parameters
    deltas = sim_data.delta_values_fixed
    w_list = sim_data.w_conductivity_range
    w1 = w_list[1]  # lowest frequency
    
    println("\n[Fixed-Delta Analysis] Processing $(length(obs_list)) observables (derived quantities only)...")
    
    for obs in obs_list
        if !hasfield(typeof(measurements), obs)
            println("  Skipping $obs (not found)")
            continue
        end
        
        # Check that this is a 3-layer (freq-dependent) observable
        raw_sample = getfield(measurements, obs)[1]
        if !(raw_sample isa Vector{<:AbstractVector})
            println("  Skipping $obs (not frequency-dependent)")
            continue
        end
        
        obs_data = getfield(measurements, obs)
        comp_label = replace(string(obs), "_w_fixed" => "")  # e.g. "Lxx"
        
        println("  > Analyzing derived quantities from $obs...")
        
        # Create top-level folders for derived quantities
        drude_base = joinpath(output_base_dir, "$(comp_label)_D_fixed")
        cond_base  = joinpath(output_base_dir, "conductivity_regular_$(comp_label)_fixed")
        mkpath(drude_base)
        mkpath(cond_base)
        
        # --- Per-delta 6-plot analysis of DERIVED real quantities ---
        for (d_i, delta_val) in enumerate(deltas)
            delta_str = @sprintf("delta_%.3f", delta_val)
            
            # --- Drude weight: real(L[i][d][1]) ---
            drude_dir = joinpath(drude_base, delta_str)
            mkpath(drude_dir)
            
            ts_drude = Float64[real(obs_data[i][d_i][1]) for i in 1:num_meas]
            drude_label = "$(comp_label)_D_fixed_d$(d_i)"
            
            p1 = plot_time_series(ts_drude, drude_label)
            p2 = plot_histogram(ts_drude, drude_label; tau_mult=tau_mult)
            p3 = plot_cumulative_mean(ts_drude, drude_label)
            p4 = plot_error_vs_binsize(ts_drude, drude_label; tau_mult=tau_mult)
            p5, p6 = plot_autocorrelation(ts_drude, drude_label)
            
            savefig(p1, joinpath(drude_dir, "01_timeseries.png"))
            savefig(p2, joinpath(drude_dir, "02_histogram.png"))
            savefig(p3, joinpath(drude_dir, "03_cum_mean.png"))
            savefig(p4, joinpath(drude_dir, "04_blocking.png"))
            savefig(p5, joinpath(drude_dir, "05_autocorr.png"))
            savefig(p6, joinpath(drude_dir, "06_tau_int.png"))
            
            # --- Conductivity regular at ω₁: imag(L[i][d][1]) / ω₁ ---
            cond_dir = joinpath(cond_base, delta_str)
            mkpath(cond_dir)
            
            ts_cond = Float64[imag(obs_data[i][d_i][1]) / w1 for i in 1:num_meas]
            cond_label = "σ_$(comp_label)_fixed_d$(d_i)_w1"
            
            p1 = plot_time_series(ts_cond, cond_label)
            p2 = plot_histogram(ts_cond, cond_label; tau_mult=tau_mult)
            p3 = plot_cumulative_mean(ts_cond, cond_label)
            p4 = plot_error_vs_binsize(ts_cond, cond_label; tau_mult=tau_mult)
            p5, p6 = plot_autocorrelation(ts_cond, cond_label)
            
            savefig(p1, joinpath(cond_dir, "01_timeseries.png"))
            savefig(p2, joinpath(cond_dir, "02_histogram.png"))
            savefig(p3, joinpath(cond_dir, "03_cum_mean.png"))
            savefig(p4, joinpath(cond_dir, "04_blocking.png"))
            savefig(p5, joinpath(cond_dir, "05_autocorr.png"))
            savefig(p6, joinpath(cond_dir, "06_tau_int.png"))
        end
    end
end

# ============================================================================
# SECTION 3B: Spectral & Reciprocal Space Plots
# ============================================================================

"""
Generate derived conductivity plots in the fixed_delta_observables folder.
Plots `conductivity_regular_αβ(ω) = imag(Lαβ_w_fixed) / ω` and `Lαβ_D(δ) = real(Lαβ_w_fixed[:][d][1])` (Drude weight).
"""
function plot_derived_conductivity(data::LatticeData, output_dir::String; cached_means=nothing)
    measurements = data.measurement_data
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    deltas = sim_data.delta_values_fixed
    w_list = sim_data.w_conductivity_range
    num_deltas = length(deltas)
    num_ws = length(w_list)
    
    mean_spacing = mean(measurements.energy_spacing_mean[1:num_meas])
    
    components = [(:Lxx_w_fixed, "xx"), (:Lxy_w_fixed, "xy"), (:Lyx_w_fixed, "yx"), (:Lyy_w_fixed, "yy")]
    cond_cache_keys = Dict("xx" => "conductivity_regular_xx_fixed", "xy" => "conductivity_regular_xy_fixed",
                           "yx" => "conductivity_regular_yx_fixed", "yy" => "conductivity_regular_yy_fixed")
    
    # Pre-compute or load from cache: cond_reg_all[label] = (means_matrix, errs_matrix)
    cond_reg_all = Dict{String, Tuple{Matrix{Float64}, Matrix{Float64}}}()
    ld_all = Dict{String, Tuple{Matrix{Float64}, Matrix{Float64}}}()  # full L_D[delta, omega] matrix
    
    for (obs_sym, label) in components
        cache_key = cond_cache_keys[label]
        
        # --- Conductivity Regular: σ(δ,ω) = imag(L)/ω ---
        if !isnothing(cached_means) && haskey(cached_means, cache_key) && isa(cached_means[cache_key], Dict)
            cm = cached_means[cache_key]
            cv = cm["value"]
            ce = cm["error"]
            cond_reg_means = zeros(Float64, num_deltas, num_ws)
            cond_reg_errs = zeros(Float64, num_deltas, num_ws)
            for d_i in 1:min(num_deltas, length(cv))
                for w_i in 1:min(num_ws, length(cv[d_i]))
                    cond_reg_means[d_i, w_i] = cv[d_i][w_i]
                    cond_reg_errs[d_i, w_i] = abs(ce[d_i][w_i])
                end
            end
        else
            obs_data = getfield(measurements, obs_sym)
            cond_reg_means = zeros(Float64, num_deltas, num_ws)
            cond_reg_errs = zeros(Float64, num_deltas, num_ws)
            for d_i in 1:num_deltas
                for w_i in 1:num_ws
                    vals = [imag(obs_data[i][d_i][w_i]) / w_list[w_i] for i in 1:num_meas]
                    cond_reg_means[d_i, w_i] = mean(vals)
                    cond_reg_errs[d_i, w_i] = std(vals) / sqrt(length(vals))
                end
            end
        end
        cond_reg_all[label] = (cond_reg_means, cond_reg_errs)
        
        # --- L_D(δ,ω) = Re[L_w_fixed(δ,ω)] — full matrix over delta and omega ---
        lw_cache_key = string(obs_sym)
        if !isnothing(cached_means) && haskey(cached_means, lw_cache_key) && isa(cached_means[lw_cache_key], Dict)
            cm = cached_means[lw_cache_key]
            cv = cm["value"]
            ce = cm["error"]
            ld_means = zeros(Float64, num_deltas, num_ws)
            ld_errs = zeros(Float64, num_deltas, num_ws)
            for d_i in 1:min(num_deltas, length(cv))
                for w_i in 1:min(num_ws, length(cv[d_i]))
                    ld_means[d_i, w_i] = real(cv[d_i][w_i])
                    ld_errs[d_i, w_i] = abs(real(ce[d_i][w_i]))
                end
            end
        else
            obs_data = getfield(measurements, obs_sym)
            ld_means = zeros(Float64, num_deltas, num_ws)
            ld_errs = zeros(Float64, num_deltas, num_ws)
            for d_i in 1:num_deltas
                for w_i in 1:num_ws
                    vals = [real(obs_data[i][d_i][w_i]) for i in 1:num_meas]
                    ld_means[d_i, w_i] = mean(vals)
                    ld_errs[d_i, w_i] = std(vals) / sqrt(length(vals))
                end
            end
        end
        ld_all[label] = (ld_means, ld_errs)
        
        # --- Save per-component plots into the matching subfolder ---
        comp_name = "L$(label)"
        
        # ── conductivity_regular folder ──
        cond_subdir = joinpath(output_dir, "conductivity_regular_$(comp_name)_fixed")
        mkpath(cond_subdir)
        
        # σ vs ω (colored by delta)
        p_cond = plot(xlabel="Frequency ω", ylabel="σ_$(label)(ω)", 
                      title="Regular Conductivity σ_$(label) vs ω\nmean_spacing=$(round(mean_spacing, sigdigits=3))", 
                      legend=:topright, leftmargin=10mm, bottommargin=10mm)
        d_indices = unique([1, div(num_deltas, 2), num_deltas])
        if num_deltas <= 5; d_indices = 1:num_deltas; end
        for d_i in d_indices
            plot!(p_cond, w_list, cond_reg_means[d_i, :], 
                  label="Γ=$(round(deltas[d_i], digits=3))", linewidth=2, m=:circle)
        end
        vline!(p_cond, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
               label="MLS=$(round(mean_spacing, sigdigits=3))")
        savefig(p_cond, joinpath(cond_subdir, "conductivity_regular_$(label)_vs_omega.png"))
        
        # σ vs Γ (at lowest ω)
        p_cond_delta = plot(xlabel="Broadening Γ (Delta)", ylabel="σ_$(label)(ω₁)",
                            title="Regular Conductivity σ_$(label) vs Γ (at ω₁=$(round(w_list[1], sigdigits=3)))\nmean_spacing=$(round(mean_spacing, sigdigits=3))",
                            legend=:best, leftmargin=10mm, bottommargin=10mm)
        plot!(p_cond_delta, deltas, cond_reg_means[:, 1], yerr=cond_reg_errs[:, 1],
              label="σ_$(label)(ω₁)", marker=:circle, linewidth=2)
        vline!(p_cond_delta, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
               label="MLS=$(round(mean_spacing, sigdigits=3))")
        savefig(p_cond_delta, joinpath(cond_subdir, "conductivity_regular_$(label)_vs_delta.png"))
        
        # ── L_D folder ──
        drude_subdir = joinpath(output_dir, "$(comp_name)_D_fixed")
        mkpath(drude_subdir)
        
        # Re[L] vs Γ (at lowest ω)
        p_ld_delta = plot(xlabel="Broadening Γ (Delta)", ylabel="Re[L_$(label)(ω₁)]",
                       title="Re[L_$(label)] vs Γ (current-current correlation, ω₁=$(round(w_list[1], sigdigits=3)))",
                       legend=:best, leftmargin=10mm, bottommargin=10mm)
        plot!(p_ld_delta, deltas, ld_means[:, 1], yerr=ld_errs[:, 1], 
              label="Re[L_$(label)(ω₁)]", marker=:circle, linewidth=2)
        vline!(p_ld_delta, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
               label="MLS=$(round(mean_spacing, sigdigits=3))")
        savefig(p_ld_delta, joinpath(drude_subdir, "L$(label)_D_vs_delta.png"))
        
        # Re[L] vs ω (colored by delta)
        p_ld_omega = plot(xlabel="Frequency ω", ylabel="Re[L_$(label)(ω)]",
                          title="Re[L_$(label)] vs ω (current-current correlation)\nmean_spacing=$(round(mean_spacing, sigdigits=3))",
                          legend=:topright, leftmargin=10mm, bottommargin=10mm)
        for d_i in d_indices
            plot!(p_ld_omega, w_list, ld_means[d_i, :],
                  label="Γ=$(round(deltas[d_i], digits=3))", linewidth=2, m=:circle)
        end
        vline!(p_ld_omega, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
               label="MLS=$(round(mean_spacing, sigdigits=3))")
        savefig(p_ld_omega, joinpath(drude_subdir, "L$(label)_D_vs_omega.png"))
    end
    
    # --- Combined outer plots (all components together) ---
    d_idx = argmin(abs.(deltas .- 2.0 * mean_spacing))
    
    # Combined: σ vs ω (all components)
    p_all_cond_w = plot(xlabel="Frequency ω", ylabel="σ(ω)", 
                 title="Regular Conductivity (all components) at Γ≈2MLS\nmean_spacing=$(round(mean_spacing, sigdigits=3))",
                 legend=:topright, leftmargin=10mm, bottommargin=10mm)
    for (obs_sym, label) in components
        cond_reg_means, _ = cond_reg_all[label]
        plot!(p_all_cond_w, w_list, cond_reg_means[d_idx, :], label="σ_$(label)", linewidth=2, m=:circle)
    end
    vline!(p_all_cond_w, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
           label="MLS=$(round(mean_spacing, sigdigits=3))")
    savefig(p_all_cond_w, joinpath(output_dir, "conductivity_vs_omega.png"))
    
    # Combined: σ vs Γ (all components, at lowest ω)
    p_all_cond_d = plot(xlabel="Broadening Γ (Delta)", ylabel="σ(ω₁)",
                   title="Regular Conductivity (all components) vs Γ (at ω₁=$(round(w_list[1], sigdigits=3)))\nmean_spacing=$(round(mean_spacing, sigdigits=3))",
                   legend=:best, leftmargin=10mm, bottommargin=10mm)
    for (obs_sym, label) in components
        cond_reg_means, cond_reg_errs = cond_reg_all[label]
        plot!(p_all_cond_d, deltas, cond_reg_means[:, 1], label="σ_$(label)(ω₁)", marker=:circle, linewidth=2)
    end
    vline!(p_all_cond_d, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
           label="MLS=$(round(mean_spacing, sigdigits=3))")
    savefig(p_all_cond_d, joinpath(output_dir, "conductivity_vs_delta.png"))
    
    # Combined: Re[L] vs Γ (all components, at lowest ω)
    p_all_ld_d = plot(xlabel="Broadening Γ (Delta)", ylabel="Re[L(ω₁)]",
                   title="Re[L] (all components) vs Γ\ncurrent-current correlation at ω₁=$(round(w_list[1], sigdigits=3))",
                   legend=:best, leftmargin=10mm, bottommargin=10mm)
    for (obs_sym, label) in components
        ld_means, _ = ld_all[label]
        plot!(p_all_ld_d, deltas, ld_means[:, 1], label="Re[L_$(label)]", marker=:circle, linewidth=2)
    end
    vline!(p_all_ld_d, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
           label="MLS=$(round(mean_spacing, sigdigits=3))")
    savefig(p_all_ld_d, joinpath(output_dir, "L_D_vs_delta.png"))
    
    # Combined: Re[L] vs ω (all components, at Γ≈2MLS)
    p_all_ld_w = plot(xlabel="Frequency ω", ylabel="Re[L(ω)]",
                   title="Re[L] (all components) vs ω at Γ≈2MLS\nmean_spacing=$(round(mean_spacing, sigdigits=3))",
                   legend=:topright, leftmargin=10mm, bottommargin=10mm)
    for (obs_sym, label) in components
        ld_means, _ = ld_all[label]
        plot!(p_all_ld_w, w_list, ld_means[d_idx, :], label="Re[L_$(label)]", linewidth=2, m=:circle)
    end
    vline!(p_all_ld_w, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
           label="MLS=$(round(mean_spacing, sigdigits=3))")
    savefig(p_all_ld_w, joinpath(output_dir, "L_D_vs_omega.png"))
    
    println("  Derived conductivity & Re[L] plots saved to $output_dir")
end

"""
Plot spectral function A(k,ω) - heatmap along k-path with frequency axis.
"""
function plot_spectral_function(lattice_data::LatticeData)
    try
        sim_data = lattice_data.sim_data
        Lx = sim_data.Lx
        Ly = sim_data.Ly
        L_S = sim_data.SC_L
        Lx_gen = Lx * L_S
        Ly_gen = Ly * L_S
        w_list = lattice_data.sim_data.w_list
        
        # Spectral function k-path length: seg1 + seg2 + seg3 + 1 (final (0,0) point)
        path_len = 2 * (Lx_gen - div(Lx_gen, 2)) + (Ly_gen - div(Ly_gen, 2)) + 1
        
        # A is Array{Array{Float64,1},1} with A[w_idx][k_path_idx]
        # hcat creates matrix of shape (path_len, num_w)
        # Transpose to get (num_w, path_len) for heatmap with k on x-axis and w on y-axis
        A = Matrix(hcat(lattice_data.measurement_data.A...)')
        
        # Create k-path labels for rectangular lattice
        # Always use index 1 for π (both even and odd L)
        # Path: (0,0) -> (π,0) -> (π,π) -> (0,0)
        seg1_len = Lx_gen - div(Lx_gen, 2)  # (0,0) to (π,0)
        seg2_len = Ly_gen - div(Ly_gen, 2)  # (π,0) to (π,π)
        seg3_len = Lx_gen - div(Lx_gen, 2)  # (π,π) to (0,0)
        
        # Triangular path: (0,0) → (π,0) → (π,π) → (0,0)
        k_names = fill("", path_len)
        k_names[1] = "(0,0)"
        k_names[seg1_len + 1] = "(π,0)"
        k_names[seg1_len + seg2_len + 1] = "(π,π)"
        k_names[path_len] = "(0,0)"
        
        # Plot heatmap
        Plots.heatmap(1:path_len, w_list, reverse(A, dims=2), xlabel="(kx,ky)", ylabel="ω", 
                     title="<A(ω,k)>, β=$(sim_data.beta), $(Lx)x$(Ly)", 
                     xticks=(1:path_len, k_names),
                     left_margin=15mm, bottommargin=15mm)
    catch e
        println("    Warning: Could not plot spectral function: $e")
        return nothing
    end
end

"""
Plot Brillouin zone occupation - heatmap of k-space population.
"""
function plot_brillouin_zone(data::LatticeData)
    try
        Lx = data.sim_data.Lx
        Ly = data.sim_data.Ly
        L_sc = data.sim_data.SC_L
        Lx_gen = Lx * L_sc
        Ly_gen = Ly * L_sc
        
        BZ = data.measurement_data.BZ
        # Reshape to (Ly_gen, Lx_gen) - ky varies along rows, kx along columns
        # This matches heatmap convention: first dim = y-axis, second dim = x-axis
        BZ_matrix = reshape(BZ, Ly_gen, Lx_gen)
        
        # Create proper axis labels (k values from 0 to 2π)
        kx_vals = (0:Lx_gen-1) .* (2π / Lx_gen)
        ky_vals = (0:Ly_gen-1) .* (2π / Ly_gen)
        
        Plots.heatmap(kx_vals, ky_vals, BZ_matrix, xlabel="kₓ", ylabel="kᵧ", 
                     title="Brillouin Zone $(Lx)x$(Ly), n=$(round(mean(data.measurement_data.n), digits=2)), β=$(data.sim_data.beta)", left_margin=15mm, bottommargin=15mm)
    catch e
        println("    Warning: Could not plot Brillouin zone: $e")
        return nothing
    end
end

"""
Plot density of states - line plot from averaged spectral function.
"""
function plot_dos(lattice_data::LatticeData; cached_means=nothing)
    try
        sim_data = lattice_data.sim_data
        num_current_measurment = length(lattice_data.measurement_data.DOS)
        
        w_list = sim_data.w_list
        beta = sim_data.beta
        
        # DOS is mean of A over k-points for each frequency — use cache if available
        if !isnothing(cached_means) && haskey(cached_means, "DOS") && isa(cached_means["DOS"], Dict)
            DOS = cached_means["DOS"]["value"]
            DOS_err = cached_means["DOS"]["error"]
        else
            DOS = mean(lattice_data.measurement_data.DOS)
            DOS_err = std(lattice_data.measurement_data.DOS) / sqrt(num_current_measurment)
        end
        title_str = "<DOS>, β=$(beta), $(sim_data.Lx)x$(sim_data.Ly)"
        
        Plots.plot(w_list, DOS, ribbon=DOS_err, legend=false, xlabel="ω", ylabel="n(ω)", 
                  title=title_str, linewidth=2, left_margin=10mm, bottommargin=10mm)
    catch e
        println("    Warning: Could not plot DOS: $e")
        return nothing
    end
end

"""
Plot histograms of vector observables at different q or ω indices with proper labels.
"""
function plot_vector_observable_histograms(data, obs_name::Symbol; num_indices::Int=3, q_values=nothing, w_values=nothing, l_values=nothing)
    try
        if !hasfield(typeof(data.measurement_data), obs_name)
            return nothing
        end
        
        obs_data = getfield(data.measurement_data, obs_name)
        
        if !isa(obs_data, AbstractArray) || length(obs_data[1]) < 2
            return nothing
        end
        
        # Get number of indices (q or ω points)
        num_w = size(obs_data[1], 1)
        
        # Select which indices to plot
        if num_w <= num_indices
            indices = 1:num_w
        else
            # Always include first one
            step = div(num_w - 1, num_indices - 1)
            indices = [1; collect(1+step:step:num_w)]
            indices = indices[1:min(num_indices, length(indices))]
        end
        
        # Create subplots with proper labels
        plots = []
        for (plot_idx, i) in enumerate(indices)
            hist_data = hcat(obs_data...)[i, :]
            if eltype(hist_data) <: Complex
                hist_data = abs.(hist_data)
            end
            
            # Create label based on available values
            if !isnothing(q_values) && i <= length(q_values)
                label_str = "q=$(round(q_values[i], digits=3))"
            elseif !isnothing(w_values) && i <= length(w_values)
                label_str = "ω=$(round(w_values[i], sigdigits=3))"
            elseif !isnothing(l_values) && i <= length(l_values)
                label_str = "l=$(l_values[i])"
            else
                label_str = "Index $i"
            end
            
            p = histogram(hist_data, bins=30, label="", 
                         title=label_str, xlabel="Value", ylabel="Frequency",
                         legend=false)
            push!(plots, p)
        end
        
        # Combine plots
        if length(plots) == 1
            return plots[1]
        else
            combined = plot(plots..., layout=(1, length(plots)), size=(400*length(plots), 300), leftmargin=10mm, bottommargin=10mm)
            return combined
        end
    catch e
        println("    Warning: Could not plot histograms for $obs_name: $e")
        return nothing
    end
end

# ============================================================================
# SECTION 3: Vector Observable Plots
# ============================================================================

"""
Plot Lxx_qx and Lxx_qy as function of momentum q (like plot_Lxx_means).
"""
function plot_Lxx_vs_q(data::LatticeData; direction1="x", direction2="x", std_scheme="correlation", cached_means=nothing)
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    Lx, Ly = sim_data.Lx, sim_data.Ly
    
    # Map directions to cache keys and field names
    if (direction1, direction2) == ("x", "x")
        qy_key, qx_key, k_key = "Lxx_qy", "Lxx_qx", "Kx"
        qy_field, qx_field, k_field = :Lxx_qy, :Lxx_qx, :Kx
        L_name = "Λxx"
        L_q = Ly
    elseif (direction1, direction2) == ("y", "y")
        qy_key, qx_key, k_key = "Lyy_qy", "Lyy_qx", "Ky"
        qy_field, qx_field, k_field = :Lyy_qy, :Lyy_qx, :Ky
        L_name = "Λyy"
        L_q = Lx
    elseif (direction1, direction2) == ("x", "y")
        qy_key, qx_key, k_key = "Lxy_qy", "Lxy_qx", "Kx"
        qy_field, qx_field, k_field = :Lxy_qy, :Lxy_qx, :Kx
        L_name = "Λxy"
        L_q = Ly
    elseif (direction1, direction2) == ("y", "x")
        qy_key, qx_key, k_key = "Lyx_qy", "Lyx_qx", "Ky"
        qy_field, qx_field, k_field = :Lyx_qy, :Lyx_qx, :Ky
        L_name = "Λyx"
        L_q = Lx
    else
        error("Invalid direction: ($direction1, $direction2)")
    end
    
    # Try cached means first
    if !isnothing(cached_means) && haskey(cached_means, qy_key) && isa(cached_means[qy_key], Dict) &&
       haskey(cached_means, qx_key) && isa(cached_means[qx_key], Dict) &&
       haskey(cached_means, k_key) && isa(cached_means[k_key], Dict)
        Lxx_qy_mean = cached_means[qy_key]["value"]
        Lxx_qy_err  = cached_means[qy_key]["error"]
        Lxx_qx_mean = cached_means[qx_key]["value"]
        Lxx_qx_err  = cached_means[qx_key]["error"]
        Kx_mean     = cached_means[k_key]["value"]
        Kx_err      = cached_means[k_key]["error"]
    else
        Lxx_qy = [getfield(data.measurement_data, qy_field)[i] for i in 1:num_meas]
        Lxx_qx = [getfield(data.measurement_data, qx_field)[i] for i in 1:num_meas]
        Kx_raw = getfield(data.measurement_data, k_field)[1:num_meas]
        Lxx_qy_mean, Lxx_qy_err = observable_mean(Lxx_qy, num_meas; std_scheme=std_scheme)
        Lxx_qx_mean, Lxx_qx_err = observable_mean(Lxx_qx, num_meas; std_scheme=std_scheme)
        Kx_mean, Kx_err = observable_mean(Kx_raw, num_meas; std_scheme=std_scheme)
    end
    n_avg = mean(data.measurement_data.n[1:num_meas])
    
    qy_vals = 2π/Ly * (1:length(Lxx_qy_mean))
    qx_vals = 2π/Lx * (1:length(Lxx_qx_mean))
    
    fig = plot(xlabel="q", ylabel=L_name, 
               title="$(L_name), <n>=$(round(n_avg, digits=3)), β=$(sim_data.beta)",
               size=(700, 500), legend=:topright, leftmargin=10mm, bottommargin=10mm)
    
    plot!(fig, qy_vals, Lxx_qy_mean, yerr=Lxx_qy_err, 
          label="$(L_name)(qx=0, qy)", linewidth=2, marker=:circle, color=:blue)
    plot!(fig, qx_vals, Lxx_qx_mean, yerr=Lxx_qx_err, 
          label="$(L_name)(qx, qy=0)", linewidth=2, marker=:square, color=:green)
    scatter!(fig, [0], [-Kx_mean], yerr=[Kx_err], 
             label="<-Kx>=$(round(-Kx_mean, sigdigits=4))", marker=:star5, color=:red, markersize=10)
    
    return fig
end

"""
Plot all Lxx directions (xx, yy, xy, yx) in a 2x2 layout.
"""
function plot_all_Lxx_directions(data::LatticeData; std_scheme="correlation", cached_means=nothing)
    fig_xx = plot_Lxx_vs_q(data; direction1="x", direction2="x", std_scheme=std_scheme, cached_means=cached_means)
    fig_yy = plot_Lxx_vs_q(data; direction1="y", direction2="y", std_scheme=std_scheme, cached_means=cached_means)
    fig_xy = plot_Lxx_vs_q(data; direction1="x", direction2="y", std_scheme=std_scheme, cached_means=cached_means)
    fig_yx = plot_Lxx_vs_q(data; direction1="y", direction2="x", std_scheme=std_scheme, cached_means=cached_means)
    
    combined = plot(fig_xx, fig_yy, fig_xy, fig_yx, layout=(2, 2), size=(1400, 1000))
    return combined
end

# NOTE: plot_conductivity_vs_omega() and plot_conductivity_specific() removed — 
# conductivity plots are now generated from L*_w_fixed in the fixed-delta analysis section.
# See SLIM_MEASUREDATA_INSTRUCTIONS.md items #8, #9.

"""
Plot Phase_Corr as function of distance/index.
"""
function plot_phase_correlation(data::LatticeData; std_scheme="correlation", cached_means=nothing)
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    
    # Get Phase_Corr data — use cache if available
    if !isnothing(cached_means) && haskey(cached_means, "Phase_Corr") && isa(cached_means["Phase_Corr"], Dict)
        Phase_Corr_mean = cached_means["Phase_Corr"]["value"]
        Phase_Corr_err  = cached_means["Phase_Corr"]["error"]
    else
        Phase_Corr = [data.measurement_data.Phase_Corr[i] for i in 1:num_meas]
        Phase_Corr_mean, Phase_Corr_err = observable_mean(Phase_Corr, num_meas; std_scheme=std_scheme)
    end
    
    n_avg = mean(data.measurement_data.n[1:num_meas])
    
    fig = plot(xlabel="Distance index", ylabel="Phase Correlation", 
               title="Phase Correlation, β=$(sim_data.beta), <n>=$(round(n_avg, digits=3))",
               size=(700, 500), legend=:topright, leftmargin=10mm, bottommargin=10mm)
    
    plot!(fig, 1:length(Phase_Corr_mean), Phase_Corr_mean, yerr=Phase_Corr_err, 
          label="⟨Δ*_0 Δ_r⟩", linewidth=2, marker=:circle, color=:blue)
    
    return fig
end

"""
Plot DC conductivity (σ_dc) as a function of Δω for all tensor components.
Each component is averaged over MC measurements with proper error estimation.
Also plots the mean level spacing as a reference vertical line.
"""
function plot_conductivity_dc_vs_dw(data::LatticeData; std_scheme="correlation", cached_means=nothing)
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    measurements = data.measurement_data
    
    dw_values = sim_data.dw_values
    if isempty(dw_values)
        println("    No dw_values defined — skipping conductivity_dc plot.")
        return nothing
    end
    
    n_dw = length(dw_values)
    n_avg = mean(measurements.n[1:num_meas])
    
    # Collect data for each tensor component
    components = Dict{String, Any}()
    component_info = [
        ("σ_dc_xx", :conductivity_dc_xx, false),
        ("σ_dc_xy", :conductivity_dc_xy, true),
        ("σ_dc_yx", :conductivity_dc_yx, true),
        ("σ_dc_yy", :conductivity_dc_yy, false),
    ]
    
    for (label, field_name, is_complex) in component_info
        if !hasfield(typeof(measurements), field_name)
            continue
        end
        
        cache_key = string(field_name)
        if !isnothing(cached_means) && haskey(cached_means, cache_key) && isa(cached_means[cache_key], Dict)
            # Use cached means — value/error are Vectors indexed by dw_ix
            cm = cached_means[cache_key]
            cv = cm["value"]
            ce = cm["error"]
            ct = get(cm, "correlation_time", zeros(length(cv)))
            means = [isa(cv, AbstractVector) ? cv[i] : cv for i in 1:min(n_dw, length(cv))]
            errs  = [isa(ce, AbstractVector) ? abs(ce[i]) : abs(ce) for i in 1:min(n_dw, length(ce))]
            taus  = [isa(ct, AbstractVector) ? ct[i] : ct for i in 1:min(n_dw, length(means))]
            # Pad if needed
            while length(means) < n_dw; push!(means, 0.0); push!(errs, 0.0); push!(taus, 0.0); end
            components[label] = (means=means, errs=errs, taus=taus, field=field_name)
        else
            obs_data = getfield(measurements, field_name)
            if isempty(obs_data) || length(obs_data) < num_meas
                continue
            end
            
            means = zeros(Float64, n_dw)
            errs = zeros(Float64, n_dw)
            taus = zeros(Float64, n_dw)
            
            for dw_ix in 1:n_dw
                ts = [obs_data[i][dw_ix] for i in 1:num_meas]
                ts_real = is_complex ? real.(ts) : ts
                
                m, e = observable_mean(ts_real, num_meas; std_scheme=std_scheme)
                means[dw_ix] = m
                errs[dw_ix] = e
                taus[dw_ix] = find_correlation_time(ts_real)
            end
            
            components[label] = (means=means, errs=errs, taus=taus, field=field_name)
        end  # end cache vs recompute
    end
    
    if isempty(components)
        println("    No conductivity_dc data found — skipping.")
        return nothing
    end
    
    # --- Plot 1: All components vs Δω ---
    fig = plot(xlabel="Δω", ylabel="σ_dc(Δω)",
               title="DC Conductivity vs Δω Window, β=$(sim_data.beta), <n>=$(round(n_avg, digits=3))",
               size=(900, 600), legend=:topright, leftmargin=10mm, bottommargin=10mm)
    
    markers = [:circle, :diamond, :utriangle, :square]
    for (i, (label, vals)) in enumerate(sort(collect(components), by=x->x[1]))
        plot!(fig, dw_values, vals.means, yerr=vals.errs,
              label=label, linewidth=2, marker=markers[mod1(i, length(markers))])
    end
    
    # Add mean level spacing reference line
    if hasfield(typeof(measurements), :energy_spacing_mean) && !isempty(measurements.energy_spacing_mean)
        mean_spacing = mean(measurements.energy_spacing_mean[1:num_meas])
        vline!(fig, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
               label="⟨Δε⟩ = $(round(mean_spacing, sigdigits=3))")
    end
    
    return fig
end

"""
Plot DC conductivity (individual component) vs Δω with diagonal/off-diagonal details.
"""
function plot_conductivity_dc_component(data::LatticeData, field_name::Symbol; std_scheme="correlation", cached_means=nothing)
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    measurements = data.measurement_data
    
    dw_values = sim_data.dw_values
    if isempty(dw_values) || !hasfield(typeof(measurements), field_name)
        return nothing
    end
    
    n_dw = length(dw_values)
    is_complex = eltype(getfield(measurements, field_name)[1]) <: Complex
    n_avg = mean(measurements.n[1:num_meas])
    
    means = zeros(Float64, n_dw)
    errs = zeros(Float64, n_dw)
    taus = zeros(Float64, n_dw)
    num_uncorr = zeros(Int, n_dw)
    
    cache_key = string(field_name)
    if !isnothing(cached_means) && haskey(cached_means, cache_key) && isa(cached_means[cache_key], Dict)
        cm = cached_means[cache_key]
        cv = cm["value"]
        ce = cm["error"]
        ct = get(cm, "correlation_time", zeros(length(cv)))
        cn = get(cm, "num_uncorrelated_measurements", zeros(Int, length(cv)))
        for dw_ix in 1:min(n_dw, length(cv))
            means[dw_ix] = isa(cv, AbstractVector) ? cv[dw_ix] : cv
            errs[dw_ix]  = isa(ce, AbstractVector) ? abs(ce[dw_ix]) : abs(ce)
            taus[dw_ix]  = isa(ct, AbstractVector) ? ct[dw_ix] : ct
            num_uncorr[dw_ix] = isa(cn, AbstractVector) ? cn[dw_ix] : cn
        end
    else
        obs_data = getfield(measurements, field_name)
        if isempty(obs_data) || length(obs_data) < num_meas
            return nothing
        end
        
        for dw_ix in 1:n_dw
            ts = [obs_data[i][dw_ix] for i in 1:num_meas]
            ts_real = is_complex ? real.(ts) : ts
            
            m, e = observable_mean(ts_real, num_meas; std_scheme=std_scheme)
            means[dw_ix] = m
            errs[dw_ix] = e
            tau_val = find_correlation_time(ts_real)
            taus[dw_ix] = tau_val
            num_uncorr[dw_ix] = Int(floor(num_meas / max(1, tau_val)))
        end
    end
    
    field_str = string(field_name)
    
    fig = plot(xlabel="Δω", ylabel="σ_dc(Δω)",
               title="$field_str vs Δω, β=$(sim_data.beta), <n>=$(round(n_avg, digits=3))",
               size=(900, 500), legend=:topright, leftmargin=10mm, bottommargin=10mm)
    
    plot!(fig, dw_values, means, yerr=errs,
          label=field_str, linewidth=2, marker=:circle, color=:blue)
    
    # Mean level spacing
    if hasfield(typeof(measurements), :energy_spacing_mean) && !isempty(measurements.energy_spacing_mean)
        mean_spacing = mean(measurements.energy_spacing_mean[1:num_meas])
        vline!(fig, [mean_spacing], color=:red, linestyle=:dash, linewidth=2,
               label="⟨Δε⟩ = $(round(mean_spacing, sigdigits=3))")
    end
    
    return fig
end

"""
Analyze conductivity_dc observables: for each Δω index, run the full 6-plot statistical analysis.
Generates per-dw-index stats + summary comparison plots.
"""
function analyze_conductivity_dc(data::LatticeData, output_base_dir::String; 
                                  std_scheme="correlation", tau_mult::Float64=10.0, save_plots::Bool=true)
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    measurements = data.measurement_data
    
    dw_values = sim_data.dw_values
    if isempty(dw_values)
        println("    No dw_values — skipping conductivity_dc analysis.")
        return nothing
    end
    
    dc_fields = [
        (:conductivity_dc_xx, false),
        (:conductivity_dc_xy, true),
        (:conductivity_dc_yx, true),
        (:conductivity_dc_yy, false),
    ]
    
    results = Dict{String, Any}()
    
    for (field_name, is_complex) in dc_fields
        if !hasfield(typeof(measurements), field_name)
            continue
        end
        obs_data = getfield(measurements, field_name)
        if isempty(obs_data) || length(obs_data) < num_meas
            continue
        end
        
        field_str = string(field_name)
        println("  > Analyzing $field_str vs Δω...")
        obs_dir = joinpath(output_base_dir, field_str)
        mkpath(obs_dir)
        
        component_results = Dict{Int, Dict}()
        
        for (dw_ix, dw_val) in enumerate(dw_values)
            # Extract time series for this Δω
            ts = [obs_data[i][dw_ix] for i in 1:num_meas]
            ts_plot = is_complex ? real.(ts) : ts
            
            obs_label = "$(field_str)_dw$(dw_ix)_$(round(dw_val, sigdigits=3))"
            
            # Statistics
            mean_val = mean(ts_plot)
            raw_err = get_raw_std(ts_plot)
            tau = find_correlation_time(ts_plot)
            num_uncorr = Int(floor(num_meas / max(1, tau)))
            block_size_used = max(1, round(Int, tau_mult * tau))
            result_blocked = get_block_means(ts_plot, block_size_used)
            blocked_err = isnothing(result_blocked) ? raw_err : sqrt(abs(result_blocked[2]))
            if isnan(blocked_err)
                blocked_err = raw_err
            end
            final_err = max(raw_err, blocked_err)
            
            component_results[dw_ix] = Dict(
                "dw" => dw_val,
                "mean" => mean_val,
                "raw_error" => raw_err,
                "blocked_error" => blocked_err,
                "final_error" => final_err,
                "tau" => tau,
                "block_size" => block_size_used,
                "num_measurements" => num_meas,
                "num_uncorrelated" => num_uncorr
            )
            
            if save_plots
                dw_dir = joinpath(obs_dir, @sprintf("dw_%d_%.4f", dw_ix, dw_val))
                mkpath(dw_dir)
                
                try
                    p1 = plot_time_series(ts_plot, obs_label)
                    p2 = plot_histogram(ts_plot, obs_label; tau_mult=tau_mult)
                    p3 = plot_cumulative_mean(ts_plot, obs_label)
                    p4 = plot_error_vs_binsize(ts_plot, obs_label; tau_mult=tau_mult)
                    p5, p6 = plot_autocorrelation(ts_plot, obs_label)
                    
                    savefig(p1, joinpath(dw_dir, "01_timeseries.png"))
                    savefig(p2, joinpath(dw_dir, "02_histogram.png"))
                    savefig(p3, joinpath(dw_dir, "03_cum_mean.png"))
                    savefig(p4, joinpath(dw_dir, "04_blocking.png"))
                    savefig(p5, joinpath(dw_dir, "05_autocorr.png"))
                    savefig(p6, joinpath(dw_dir, "06_tau_int.png"))
                catch e
                    println("      Error plotting dw_ix=$dw_ix: $e")
                end
            end
        end
        
        results[field_str] = component_results
    end
    
    # --- Summary statistics text ---
    if save_plots
        summary_path = joinpath(output_base_dir, "conductivity_dc_summary.txt")
        open(summary_path, "w") do io
            println(io, "="^90)
            println(io, "DC CONDUCTIVITY (σ_dc) vs Δω — STATISTICAL SUMMARY")
            println(io, "="^90)
            println(io, "β = $(sim_data.beta), <n> = $(round(mean(measurements.n[1:num_meas]), digits=4))")
            println(io, "Number of MC measurements: $num_meas")
            if hasfield(typeof(measurements), :energy_spacing_mean) && !isempty(measurements.energy_spacing_mean)
                println(io, "Mean level spacing ⟨Δε⟩ = $(round(mean(measurements.energy_spacing_mean[1:num_meas]), sigdigits=4))")
            end
            println(io, "")
            
            for (field_str, comp_results) in sort(collect(results), by=x->x[1])
                println(io, "-"^90)
                println(io, field_str)
                println(io, @sprintf("  %-6s %12s %12s %12s %8s %10s", "Δω_ix", "Δω", "Mean", "Error", "τ", "N_eff"))
                println(io, "-"^90)
                for dw_ix in sort(collect(keys(comp_results)))
                    r = comp_results[dw_ix]
                    println(io, @sprintf("  %-6d %12.6f %12.6g %12.6g %8.2f %10d",
                            dw_ix, r["dw"], r["mean"], r["final_error"], r["tau"], r["num_uncorrelated"]))
                end
                println(io, "")
            end
            println(io, "="^90)
        end
        println("  Summary saved: $summary_path")
    end
    
    return results
end

# NOTE: plot_stiffness_delta() and analyze_stiffness_delta() removed —
# L*_q*_delta fields no longer exist; stiffness-vs-δ is no longer computed.
# See SLIM_MEASUREDATA_INSTRUCTIONS.md item #15.

"""
Plot Ps_l (pair structure factor) as function of index.
"""
function plot_Ps_l(data::LatticeData; std_scheme="correlation", cached_means=nothing)
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    
    # Use cache if available
    if !isnothing(cached_means) && haskey(cached_means, "Ps_l") && isa(cached_means["Ps_l"], Dict)
        Ps_l_mean = cached_means["Ps_l"]["value"]
        Ps_l_err  = cached_means["Ps_l"]["error"]
    else
        Ps_l = [data.measurement_data.Ps_l[i] for i in 1:num_meas]
        Ps_l_mean, Ps_l_err = observable_mean(Ps_l, num_meas; std_scheme=std_scheme)
    end
    
    n_avg = mean(data.measurement_data.n[1:num_meas])
    
    fig = plot(xlabel="l (index)", ylabel="Ps(l)", 
               title="Pair Structure Factor, β=$(sim_data.beta), <n>=$(round(n_avg, digits=3))",
               size=(700, 500), legend=:topright, leftmargin=10mm, bottommargin=10mm)
    
    plot!(fig, 0:(length(Ps_l_mean)-1), Ps_l_mean, yerr=Ps_l_err, 
          label="Ps(l)", linewidth=2, marker=:circle, color=:purple)
    
    return fig
end

# ============================================================================
# SECTION 4: Bootstrap-Derived Quantities
# ============================================================================

"""
Compute and display all bootstrap-derived quantities.
If `cached_means` is provided, reads values from cache instead of recomputing.
"""
function compute_bootstrap_quantities(data::LatticeData; num_blocks=32, B=1000, fit_stiffness=true, cached_means=nothing)
    num_meas = data.current_state.num_current_measurments
    measurements = data.measurement_data
    
    results = Dict{String, Any}()
    
    println("\n" * "="^70)
    println("BOOTSTRAP-DERIVED QUANTITIES")
    if !isnothing(cached_means)
        println("  [Using cached means — skipping bootstrap recomputation]")
    end
    println("="^70)
    
    # --- Stiffness ---
    if !isnothing(cached_means) && haskey(cached_means, "stiffness") && isa(cached_means["stiffness"], Dict)
        stiff_val = cached_means["stiffness"]["value"]
        stiff_err = cached_means["stiffness"]["error"]
        results["stiffness"] = (stiff_val, stiff_err)
        println(@sprintf("  Stiffness ρs:          %.6f ± %.6f  (cached)", stiff_val, stiff_err))
    else
        try
            Kx = measurements.Kx[1:num_meas]
            Lxx_qy = [measurements.Lxx_qy[i] for i in 1:num_meas]
            stiff_val, stiff_err = BT_stiffness(Kx, Lxx_qy; num_blocks=num_blocks, B=B, fit=fit_stiffness)
            results["stiffness"] = (stiff_val, stiff_err)
            println(@sprintf("  Stiffness ρs:          %.6f ± %.6f", stiff_val, stiff_err))
        catch e
            println("  Stiffness: Error - $e")
        end
    end
    
    # --- Drude Weight ---
    if !isnothing(cached_means) && haskey(cached_means, "Drude_weight") && isa(cached_means["Drude_weight"], Dict)
        drude_val = cached_means["Drude_weight"]["value"]
        drude_err = cached_means["Drude_weight"]["error"]
        results["Drude_weight"] = (drude_val, drude_err)
        println(@sprintf("  Drude Weight D:        %.6f ± %.6f  (cached)", drude_val, drude_err))
    else
        try
            # Derive Lxx_D from Lxx_w_fixed at δ closest to 2×MLS
            sim_data = data.sim_data
            delta_values = sim_data.delta_values_fixed
            mls = mean(measurements.energy_spacing_mean[1:num_meas])
            d_idx = argmin(abs.(delta_values .- 2.0 * mls))
            
            Kx = measurements.Kx[1:num_meas]
            Lxx_D = [real.(measurements.Lxx_w_fixed[i][d_idx]) for i in 1:num_meas]
            drude_val, drude_err = BT_stiffness(Kx, Lxx_D; num_blocks=num_blocks, B=B, fit=fit_stiffness)
            # Drude weight is -π * (Kx + Lxx_D(ω→0))
            drude_val = -π * 4 * drude_val  # Convert from stiffness formula
            drude_err = π * 4 * drude_err
            results["Drude_weight"] = (drude_val, drude_err)
            println(@sprintf("  Drude Weight D:        %.6f ± %.6f  (δ=%.4f ≈ 2MLS)", drude_val, drude_err, delta_values[d_idx]))
        catch e
            println("  Drude Weight: Error - $e")
        end
    end
    
    # --- Resistivity (derived from L*_w_fixed at δ closest to 2×MLS) ---
    try
        sim_data = data.sim_data
        delta_values = sim_data.delta_values_fixed
        w_list = sim_data.w_conductivity_range
        mls = mean(measurements.energy_spacing_mean[1:num_meas])
        d_idx = argmin(abs.(delta_values .- 2.0 * mls))
        
        cond_xx = [imag.(measurements.Lxx_w_fixed[i][d_idx]) ./ w_list for i in 1:num_meas]
        cond_xy = [imag.(measurements.Lxy_w_fixed[i][d_idx]) ./ w_list for i in 1:num_meas]
        (res_xx, res_xx_err), (res_xy, res_xy_err) = BT_resistivity(cond_xx, cond_xy; num_blocks=num_blocks, B=B)
        results["resistivity_xx"] = (res_xx, res_xx_err)
        results["resistivity_xy"] = (res_xy, res_xy_err)
        println(@sprintf("  Resistivity ρxx:       %.6f ± %.6f  (δ=%.4f ≈ 2MLS)", res_xx, res_xx_err, delta_values[d_idx]))
        println(@sprintf("  Hall Resistivity ρxy:  %.6f ± %.6f", res_xy, res_xy_err))
    catch e
        println("  Resistivity: Error - $e")
    end
    
    # --- Full Resistivity Tensor ---
    if !isnothing(cached_means) && haskey(cached_means, "resistivity_tensor") && isa(cached_means["resistivity_tensor"], Dict)
        res_tensor = cached_means["resistivity_tensor"]["value"]
        res_tensor_err = cached_means["resistivity_tensor"]["error"]
        corr_time = cached_means["resistivity_tensor"]["correlation_time"]
        nblocks = cached_means["resistivity_tensor"]["num_uncorrelated_measurements"]
        results["resistivity_tensor"] = (res_tensor, res_tensor_err)
        println("  Resistivity Tensor (cached):")
        println(@sprintf("    ρxx = %.6f ± %.6f", res_tensor[1], res_tensor_err[1]))
        println(@sprintf("    ρxy = %.6f ± %.6f", res_tensor[2], res_tensor_err[2]))
        println(@sprintf("    ρyx = %.6f ± %.6f", res_tensor[3], res_tensor_err[3]))
        println(@sprintf("    ρyy = %.6f ± %.6f", res_tensor[4], res_tensor_err[4]))
        println("    (correlation time: $(round(corr_time, sigdigits=3)), num_blocks: $nblocks)")
    else
        try
            sim_data = data.sim_data
            delta_values = sim_data.delta_values_fixed
            w_list = sim_data.w_conductivity_range
            mls = mean(measurements.energy_spacing_mean[1:num_meas])
            d_idx = argmin(abs.(delta_values .- 2.0 * mls))
            
            cond_xx = [imag.(measurements.Lxx_w_fixed[i][d_idx]) ./ w_list for i in 1:num_meas]
            cond_xy = [imag.(measurements.Lxy_w_fixed[i][d_idx]) ./ w_list for i in 1:num_meas]
            cond_yx = [imag.(measurements.Lyx_w_fixed[i][d_idx]) ./ w_list for i in 1:num_meas]
            cond_yy = [imag.(measurements.Lyy_w_fixed[i][d_idx]) ./ w_list for i in 1:num_meas]
            res_tensor, res_tensor_err, corr_time, nblocks = BT_resistivity_tensor(
                cond_xx, cond_xy, cond_yx, cond_yy; num_blocks=num_blocks, B=B)
            results["resistivity_tensor"] = (res_tensor, res_tensor_err)
            println("  Resistivity Tensor (δ==$(round(delta_values[d_idx], sigdigits=4)) ≈ 2MLS):")
            println(@sprintf("    ρxx = %.6f ± %.6f", res_tensor[1], res_tensor_err[1]))
            println(@sprintf("    ρxy = %.6f ± %.6f", res_tensor[2], res_tensor_err[2]))
            println(@sprintf("    ρyx = %.6f ± %.6f", res_tensor[3], res_tensor_err[3]))
            println(@sprintf("    ρyy = %.6f ± %.6f", res_tensor[4], res_tensor_err[4]))
            println("    (correlation time: $(round(corr_time, sigdigits=3)), num_blocks: $nblocks)")
        catch e
            println("  Resistivity Tensor: Error - $e")
        end
    end
    
    println("="^70)
    
    return results
end

# ============================================================================
# SECTION 5: Joint Distributions
# ============================================================================

"""
Plot 2D joint distribution (scatter + marginal histograms).
"""
function plot_joint_distribution(x_data::AbstractVector, y_data::AbstractVector, 
                                  x_name::String, y_name::String; 
                                  alpha=0.3, nbins=30, color_by_step=true)
    n = min(length(x_data), length(y_data))
    x = x_data[1:n]
    y = y_data[1:n]
    
    # Handle complex data
    x = eltype(x) <: Complex ? abs.(x) : x
    y = eltype(y) <: Complex ? abs.(y) : y
    
    # Main scatter plot
    if color_by_step
        fig_main = scatter(x, y, 
            xlabel=x_name, ylabel=y_name,
            title="Joint Distribution: $x_name vs $y_name",
            marker_z=1:n, color=:viridis, colorbar_title="MC Step",
            alpha=alpha, markersize=3, legend=false,
            size=(700, 600), leftmargin=10mm, bottommargin=10mm)
    else
        fig_main = scatter(x, y, 
            xlabel=x_name, ylabel=y_name,
            title="Joint Distribution: $x_name vs $y_name",
            alpha=alpha, markersize=3, legend=false, color=:blue,
            size=(700, 600), leftmargin=10mm, bottommargin=10mm)
    end
    
    # Add correlation coefficient
    corr_coef = cor(x, y)
    annotate!(fig_main, [(minimum(x) + 0.1*(maximum(x)-minimum(x)), 
                          maximum(y) - 0.1*(maximum(y)-minimum(y)), 
                          text("r = $(round(corr_coef, sigdigits=3))", 12, :left))])
    
    # Marginal histograms
    fig_x = histogram(x, bins=nbins, xlabel=x_name, ylabel="Count", 
                      title="$x_name distribution", legend=false, color=:blue, alpha=0.7)
    fig_y = histogram(y, bins=nbins, xlabel=y_name, ylabel="Count", 
                      title="$y_name distribution", legend=false, color=:green, alpha=0.7)
    
    # Combined layout
    combined = plot(fig_main, fig_x, fig_y, 
                   layout=@layout([a{0.7w} [b; c]]), size=(1100, 600))
    
    return combined, corr_coef
end

"""
Generate all requested joint distribution plots.
"""
function plot_all_joint_distributions(data::LatticeData)
    num_meas = data.current_state.num_current_measurments
    measurements = data.measurement_data
    
    plots_list = []
    correlations = Dict{String, Float64}()
    
    # 1. Lxx_qy[1] vs Kx
    try
        Lxx_qy_1 = [measurements.Lxx_qy[i][1] for i in 1:num_meas]
        Kx = measurements.Kx[1:num_meas]
        fig, corr = plot_joint_distribution(Lxx_qy_1, Kx, "Λxx(qy)[1]", "Kx")
        push!(plots_list, ("Lxx_qy_1_vs_Kx", fig))
        correlations["Lxx_qy[1] vs Kx"] = corr
    catch e
        println("Could not plot Lxx_qy[1] vs Kx: $e")
    end
    
    # 2. Lxx_D[1] vs Kx — derived from Lxx_w_fixed at δ ≈ 2×MLS
    try
        delta_values = data.sim_data.delta_values_fixed
        mls = mean(measurements.energy_spacing_mean[1:num_meas])
        d_idx = argmin(abs.(delta_values .- 2.0 * mls))
        Lxx_D_1 = [real(measurements.Lxx_w_fixed[i][d_idx][1]) for i in 1:num_meas]
        Kx = measurements.Kx[1:num_meas]
        delta_used = round(delta_values[d_idx], digits=4)
        fig, corr = plot_joint_distribution(Lxx_D_1, Kx, "Λxx_D[1] (δ=$delta_used≈2MLS)", "Kx")
        push!(plots_list, ("Lxx_D_1_vs_Kx", fig))
        correlations["Lxx_D[1] vs Kx"] = corr
    catch e
        println("Could not plot Lxx_D[1] vs Kx: $e")
    end
    
    # 3. n vs DdD
    try
        n = measurements.n[1:num_meas]
        DdD = measurements.DdD[1:num_meas]
        fig, corr = plot_joint_distribution(n, DdD, "n (occupation)", "DdD (pair density)")
        push!(plots_list, ("n_vs_DdD", fig))
        correlations["n vs DdD"] = corr
    catch e
        println("Could not plot n vs DdD: $e")
    end

    # 4. Lxx_qy[1] vs Lxx_qy[2]
    try
        Lxx_qy_1 = [measurements.Lxx_qy[i][1] for i in 1:num_meas]
        Lxx_qy_2 = [measurements.Lxx_qy[i][2] for i in 1:num_meas]
        fig, corr = plot_joint_distribution(Lxx_qy_1, Lxx_qy_2, "Λxx(qy)[1]", "Λxx(qy)[2]")
        push!(plots_list, ("Lxx_qy_1_vs_Lxx_qy_2", fig))
        correlations["Lxx_qy[1] vs Lxx_qy[2]"] = corr
    catch e
        println("Could not plot Lxx_qy[1] vs Lxx_qy[2]: $e")
    end

    # 5. Lyy_qx[1] vs Ky
    try
        Lyy_qx_1 = [measurements.Lyy_qx[i][1] for i in 1:num_meas]
        Ky = measurements.Ky[1:num_meas]
        fig, corr = plot_joint_distribution(Lyy_qx_1, Ky, "Λyy(qx)[1]", "Ky")
        push!(plots_list, ("Lyy_qx_1_vs_Ky", fig))
        correlations["Lyy_qx[1] vs Ky"] = corr
    catch e
        println("Could not plot Lyy_qx[1] vs Ky: $e")
    end

    # 6. Lyy_D[1] vs Ky — derived from Lyy_w_fixed at δ ≈ 2×MLS
    try
        delta_values = data.sim_data.delta_values_fixed
        mls = mean(measurements.energy_spacing_mean[1:num_meas])
        d_idx = argmin(abs.(delta_values .- 2.0 * mls))
        Lyy_D_1 = [real(measurements.Lyy_w_fixed[i][d_idx][1]) for i in 1:num_meas]
        Ky = measurements.Ky[1:num_meas]
        delta_used = round(delta_values[d_idx], digits=4)
        fig, corr = plot_joint_distribution(Lyy_D_1, Ky, "Λyy_D[1] (δ=$delta_used≈2MLS)", "Ky")
        push!(plots_list, ("Lyy_D_1_vs_Ky", fig))
        correlations["Lyy_D[1] vs Ky"] = corr
    catch e
        println("Could not plot Lyy_D[1] vs Ky: $e")
    end

    # 7.  Lyy_qx[1] vs Lyy_qx[2]
    try
        Lyy_qx_1 = [measurements.Lyy_qx[i][1] for i in 1:num_meas]
        Lyy_qx_2 = [measurements.Lyy_qx[i][2] for i in 1:num_meas]
        fig, corr = plot_joint_distribution(Lyy_qx_1, Lyy_qx_2, "Λyy(qx)[1]", "Λyy(qx)[2]")
        push!(plots_list, ("Lyy_qx_1_vs_Lyy_qx_2", fig))
        correlations["Lyy_qx[1] vs Lyy_qx[2]"] = corr
    catch e
        println("Could not plot Lyy_qx[1] vs Lyy_qx[2]: $e")
    end

    
    # Print correlation summary
    println("\n" * "-"^50)
    println("JOINT DISTRIBUTION CORRELATIONS:")
    println("-"^50)
    for (name, corr) in correlations
        println("  $name: r = $(round(corr, sigdigits=4))")
    end
    println("-"^50)
    
    return plots_list, correlations
end

# ============================================================================
# SECTION 6: Complex Mean Field Trajectory
# ============================================================================

"""
Plot the complex mean field Δ (s-wave component) in the complex plane, colored by MC step.
Note: Delta_MC is a vector [s-wave, d-wave_x, d-wave_y]. We use index 1 for s-wave.
"""
function plot_complex_field_trajectory(data::LatticeData; max_points=nothing, subsample=1, component::Int=1)
    num_meas = data.current_state.num_current_measurments
    measurements = data.measurement_data
    
    # Get Delta_MC (complex mean field) - it's a vector [s-wave, d-wave_x, d-wave_y]
    # Extract the specified component (default: 1 = s-wave)
    Delta_MC_raw = measurements.Delta_MC[1:num_meas]
    
    # Handle both scalar and vector Delta_MC formats
    if eltype(Delta_MC_raw) <: AbstractVector
        missing_component = any(length(d) < component for d in Delta_MC_raw)
        if missing_component
            @warn "Requested Delta_MC component $component is unavailable for this pairing. Filling missing values with zero."
        end
        Delta_MC = [length(d) >= component ? d[component] : 0.0 + 0.0im for d in Delta_MC_raw]
        component_name = component == 1 ? "s-wave" : (component == 2 ? "d-wave_x" : (component == 3 ? "d-wave_y" : "component_$component"))
    else
        Delta_MC = Delta_MC_raw
        component_name = "s-wave"
    end
    
    # Subsample if too many points
    if !isnothing(max_points) && num_meas > max_points
        subsample = max(1, div(num_meas, max_points))
    end
    indices = 1:subsample:num_meas
    Delta_plot = Delta_MC[indices]
    
    re_vals = real.(Delta_plot)
    im_vals = imag.(Delta_plot)
    
    # Main trajectory plot
    fig1 = scatter(re_vals, im_vals,
        xlabel="Re(Δ)", ylabel="Im(Δ)",
        title="Complex Mean Field ($component_name) Trajectory\n($(length(Delta_plot)) points, subsample=$subsample)",
        marker_z=1:length(Delta_plot), color=:viridis, colorbar_title="MC Step",
        alpha=0.5, markersize=3, legend=false,
        size=(700, 700), leftmargin=10mm, bottommargin=10mm,
        aspect_ratio=:equal)
    
    # Add circle of mean amplitude
    mean_amp = mean(abs.(Delta_plot))
    θ = range(0, 2π, length=100)
    plot!(fig1, mean_amp .* cos.(θ), mean_amp .* sin.(θ), 
          color=:red, linewidth=2, linestyle=:dash, label="⟨|Δ|⟩ = $(round(mean_amp, sigdigits=4))")
    
    # Add origin marker
    scatter!(fig1, [0], [0], color=:red, markersize=8, markershape=:cross, label=nothing)
    
    # Phase angle histogram
    phases = angle.(Delta_plot)
    fig2 = histogram(phases, bins=50, 
        xlabel="Phase angle θ = arg(Δ)", ylabel="Count",
        title="Phase Distribution ($component_name)\n(should be uniform for ergodic sampling)",
        color=:purple, alpha=0.7, legend=false,
        size=(700, 400), leftmargin=10mm, bottommargin=10mm)
    
    # Add expected uniform line
    expected_count = length(phases) / 50
    hline!(fig2, [expected_count], color=:red, linewidth=2, linestyle=:dash, 
           label="Expected uniform")
    
    # Amplitude histogram
    amplitudes = abs.(Delta_plot)
    fig3 = histogram(amplitudes, bins=50,
        xlabel="|Δ|", ylabel="Count",
        title="Amplitude Distribution ($component_name)",
        color=:blue, alpha=0.7, legend=false,
        size=(700, 400), leftmargin=10mm, bottommargin=10mm)
    
    # Amplitude vs MC step
    fig4 = plot(indices, amplitudes,
        xlabel="MC Step", ylabel="|Δ|",
        title="Amplitude vs MC Step ($component_name)",
        linewidth=0.5, alpha=0.7, legend=false,
        size=(900, 300), leftmargin=10mm, bottommargin=10mm)
    mean_amp = mean(amplitudes)
    hline!(fig4, [mean_amp], color=:red, linewidth=2, linestyle=:dash)
    
    # Phase vs MC step
    fig5 = scatter(indices, phases,
        xlabel="MC Step", ylabel="arg(Δ)",
        title="Phase vs MC Step ($component_name)",
        markersize=1, alpha=0.5, legend=false,
        size=(900, 300), leftmargin=10mm, bottommargin=10mm)
    
    return fig1, fig2, fig3, fig4, fig5, component_name
end

# ============================================================================
# SECTION 7: Summary Statistics Table
# ============================================================================

"""
Generate summary table of all observables.
If `cached_means` is provided (from means_cache.jld2), uses cached values instead of recomputing.
"""
function print_summary_table(data::LatticeData; tau_mult=10.0, printoutput=true, cached_means=nothing)
    num_meas = data.current_state.num_current_measurments
    measurements = data.measurement_data
    sim_data = data.sim_data
    
    scalar_obs, vector_obs = list_observables(measurements)

    println("\n" * "="^90)
    println("SIMULATION SUMMARY")
    println("="^90)
    println("  Path: (loaded simulation)")
    println("  System size: $(sim_data.Lx) × $(sim_data.Ly)")
    println("  Parameters: U=$(sim_data.U), t=$(sim_data.t), μ=$(sim_data.mu), β=$(sim_data.beta)")
    println("  Magnetic field: B=$(round(sim_data.B, digits=4)) = $(round(sim_data.B/(2π), digits=4))Φ₀")
    println("  Number of measurements: $num_meas")
    if !isnothing(cached_means)
        println("  [Using cached means — skipping recomputation]")
    end
    println("="^90)

    if printoutput
        println("\n" * "-"^90)
        println(@sprintf("%-25s %15s %15s %10s %12s", "Observable", "Mean", "Error", "τ", "N_eff"))
        println("-"^90)
    end

    
    results_dict = Dict{Symbol, Dict}()
    
    # Scalar observables
    for obs in scalar_obs
        if obs == :BZ
            continue  
        end
        try
            # Try to use cached means if available
            obs_key = string(obs)
            if !isnothing(cached_means) && haskey(cached_means, obs_key) && isa(cached_means[obs_key], Dict)
                cm = cached_means[obs_key]
                tau_val = isa(cm["correlation_time"], Number) ? cm["correlation_time"] : 0.0
                num_uncorr = isa(cm["num_uncorrelated_measurements"], Number) ? cm["num_uncorrelated_measurements"] : num_meas
                mean_val = isa(cm["value"], Number) ? cm["value"] : (isa(cm["value"], AbstractArray) ? cm["value"][1] : 0.0)
                err_val = isa(cm["error"], Number) ? cm["error"] : (isa(cm["error"], AbstractArray) ? cm["error"][1] : 0.0)
                stats = Dict(
                    "name" => obs_key,
                    "mean" => mean_val,
                    "raw_error" => err_val,
                    "blocked_error" => err_val,
                    "final_error" => err_val,
                    "tau" => tau_val,
                    "block_size" => max(1, round(Int, tau_mult * tau_val)),
                    "num_measurements" => num_meas,
                    "num_uncorrelated" => num_uncorr,
                )
                results_dict[obs] = stats
            else
                stats, _ = analyze_single_observable(measurements, obs, num_meas; 
                                                      tau_mult=tau_mult, show_plots=false)
                results_dict[obs] = stats
            end
            stats = results_dict[obs]
            if printoutput
                println(@sprintf("%-25s %15.6g %15.6g %10.2f %12d", 
                        stats["name"], stats["mean"], stats["final_error"], 
                        stats["tau"], stats["num_uncorrelated"]))
            end
        catch e
            println(@sprintf("%-25s %s", string(obs), "Error: $e"))
        end
    end
    
    # Vector observables (all component only in summary)
    for (obs, vec_len) in vector_obs
        if  obs == :A || obs == :LDOS
            continue  
        end
        try
            obs_key = string(obs)
            if !isnothing(cached_means) && haskey(cached_means, obs_key) && isa(cached_means[obs_key], Dict)
                cm = cached_means[obs_key]
                # Build results_dict entries from cached vector means
                results_dict[obs] = Dict{Int, Dict}()
                mean_vals = cm["value"]
                err_vals = cm["error"]
                tau_vals = cm["correlation_time"]
                uncorr_vals = cm["num_uncorrelated_measurements"]
                actual_vec_len = isa(mean_vals, AbstractArray) ? length(mean_vals) : 1
                if printoutput
                    println(@sprintf("%-25s", "$(obs)"))
                end
                for key in 1:actual_vec_len
                    m_val = isa(mean_vals, AbstractArray) ? mean_vals[key] : mean_vals
                    e_val = isa(err_vals, AbstractArray) ? err_vals[key] : err_vals
                    t_val = isa(tau_vals, AbstractArray) ? tau_vals[key] : tau_vals
                    u_val = isa(uncorr_vals, AbstractArray) ? uncorr_vals[key] : uncorr_vals
                    stats = Dict(
                        "name" => "$(obs)[$key/$actual_vec_len]",
                        "mean" => m_val,
                        "raw_error" => e_val,
                        "blocked_error" => e_val,
                        "final_error" => e_val,
                        "tau" => t_val,
                        "block_size" => max(1, round(Int, tau_mult * (isa(t_val, Number) ? t_val : 0.0))),
                        "num_measurements" => num_meas,
                        "num_uncorrelated" => isa(u_val, Number) ? u_val : num_meas,
                    )
                    results_dict[obs][key] = stats
                    if key == 1
                        if printoutput
                            println(@sprintf("    %-25s %15.6g %15.6g %10.2f %12d", 
                                "$(obs)[$key/$actual_vec_len]", stats["mean"], stats["final_error"], 
                                stats["tau"], stats["num_uncorrelated"]))
                        end
                    end
                end
            else
                stats_ixs = analyze_single_observable(measurements, obs, num_meas; 
                                             tau_mult=tau_mult, show_plots=false, analyze_all_ixes=true)
                results_dict[obs] = Dict{Int, Dict}()
                if printoutput
                    println(@sprintf("%-25s", "$(obs)"))
                end
                for (key, val) in stats_ixs
                    stats, _ = val
                    results_dict[obs][key] = stats
                    if key == 1 
                        if printoutput
                            # Print header for vector observable only once
                            println(@sprintf("    %-25s %15.6g %15.6g %10.2f %12d", 
                                "$(obs)[$key/$vec_len]", stats["mean"], stats["final_error"], 
                                stats["tau"], stats["num_uncorrelated"]))
                        end
                    end

                end
            end
        catch e
            println(@sprintf("%-25s %s", "$(obs)", "Error: $e"))
        end
    end
    
    println("-"^90)
    
    return results_dict
end

# ============================================================================
# SECTION 8: Statistics Text File Export
# ============================================================================

"""
Write all statistics to a text file.
"""
function write_statistics_file(output_dir::String, sim_data, results_dict::Dict, 
                                bootstrap_results::Dict, correlations::Dict;
                                filename::String="statistics_summary.txt",
                                report_metadata::Dict{String, Any}=Dict{String, Any}(),
                                seed_diagnostics=nothing,
                                mu_convergence=nothing,
                                delta_extrapolation=nothing,
                                transport_summary=nothing)
    filepath = joinpath(output_dir, filename)
    
    open(filepath, "w") do io
        # Header
        println(io, "="^90)
        println(io, "MONTE CARLO SIMULATION STATISTICS REPORT")
        println(io, "="^90)
        println(io, "Generated: $(Dates.now())")
        println(io, "")
        
        # Simulation Parameters
        println(io, "-"^90)
        println(io, "SIMULATION PARAMETERS")
        println(io, "-"^90)
        println(io, "  System size: $(sim_data.Lx) × $(sim_data.Ly)")
        println(io, "  Number of sites: $(sim_data.N)")
        println(io, "  U (interaction): $(sim_data.U)")
        println(io, "  t (hopping): $(sim_data.t)")
        println(io, "  μ (chemical potential): $(sim_data.mu)")
        println(io, "  β (inverse temperature): $(sim_data.beta)")
        println(io, "  T (temperature): $(round(1/sim_data.beta, sigdigits=4))")
        println(io, "  B (magnetic field): $(round(sim_data.B, digits=6)) = $(round(sim_data.B/(2π), digits=6))Φ₀")
        println(io, "  V (disorder): $(sim_data.V)")
        println(io, "  TCA_size (Traveling Cluster Approximation size): $(sim_data.TCluster_size_x) × $(sim_data.TCluster_size_y)")
        println(io, "  Pairing symmetry: $(sim_data.pairing_symmetry)")
        println(io, "  Diagonalization method: $(sim_data.diag_method)")
        println(io, "")

        if !isempty(report_metadata)
            println(io, "-"^90)
            println(io, "REPORT METADATA")
            println(io, "-"^90)
            for (k, v) in sort(collect(report_metadata), by=x->x[1])
                println(io, "  $k: $v")
            end
            println(io, "")
        end
        
        # Observable Statistics (exclude A and LDOS as they're not measured every MC step)
        println(io, "-"^90)
        println(io, @sprintf("%-25s %15s %15s %10s %12s", "OBSERVABLE", "MEAN", "ERROR", "τ", "N_eff"))
        println(io, "-"^90)
        
        excluded_obs = [:A, :LDOS]  # Don't print these - not measured every step
        for (obs, stats) in sort(collect(results_dict), by=x->string(x[1]))
            if stats isa Dict && !(obs in excluded_obs)
                if haskey(stats, "name")
                    # Scalar observable
                    println(io, @sprintf("%-25s %15.6g %15.6g %10.2f %12d", 
                        stats["name"], stats["mean"], stats["final_error"], 
                        stats["tau"], stats["num_uncorrelated"]))
                else
                    # Vector observable - print each component
                    for (key, val) in sort(collect(stats), by=x->x[1])
                        println(io, @sprintf("    %-25s %15.6g %15.6g %10.2f %12d", 
                            "$(obs)[$key]", val["mean"], val["final_error"], 
                            val["tau"], val["num_uncorrelated"]))
                    end
                end


            end
        end
        println(io, "-"^90)
        println(io, "")

        # Enhanced transport tensor summary
        if transport_summary isa AbstractDict
            println(io, "-"^90)
            println(io, "TRANSPORT TENSOR SUMMARY")
            println(io, "-"^90)
            println(io, "  available: $(get(transport_summary, "available", false))")
            if haskey(transport_summary, "reason")
                println(io, "  reason: $(transport_summary["reason"])")
            end
            if get(transport_summary, "available", false)
                println(io, "  entries: $(get(transport_summary, "num_entries", 0))")
                println(io, "  energy_spacing_mean: $(get(transport_summary, "energy_spacing_mean", "N/A"))")
                println(io, "  closest_delta_to_2MLS: idx=$(get(transport_summary, "closest_delta_to_2MLS_index", "N/A")), value=$(get(transport_summary, "closest_delta_to_2MLS_value", "N/A"))")
                println(io, "  closest_dw_to_MLS: idx=$(get(transport_summary, "closest_dw_to_MLS_index", "N/A")), value=$(get(transport_summary, "closest_dw_to_MLS_value", "N/A"))")
                entries = get(transport_summary, "entries", Dict{String, Any}())
                for (k, entry_any) in sort(collect(entries), by=x->x[1])
                    if !(entry_any isa AbstractDict)
                        continue
                    end
                    entry = entry_any
                    println(io, "  $k:")
                    println(io, "    value_shape=$(get(entry, "value_shape", "N/A")), error_shape=$(get(entry, "error_shape", "N/A"))")
                    println(io, "    sample=$(get(entry, "sample_value", "N/A")) ± $(get(entry, "sample_error", "N/A"))")
                    println(io, "    num_measurements=$(get(entry, "num_measurements", "N/A")), N_eff=$(get(entry, "num_uncorrelated_measurements", "N/A"))")
                end
            end
            println(io, "")
        end

        # Mu/filling convergence summary
        if mu_convergence isa AbstractDict
            println(io, "-"^90)
            println(io, "MU/FILLING CONVERGENCE")
            println(io, "-"^90)
            println(io, "  available: $(get(mu_convergence, "available", false))")
            if haskey(mu_convergence, "reason")
                println(io, "  reason: $(mu_convergence["reason"])")
            end
            if get(mu_convergence, "available", false)
                println(io, "  source: $(get(mu_convergence, "source", "unknown"))")
                println(io, "  num_seeds: $(get(mu_convergence, "num_seeds", "N/A"))")
                println(io, "  num_converged: $(get(mu_convergence, "num_converged", "N/A"))")
                println(io, "  all_converged: $(get(mu_convergence, "all_converged", "N/A"))")
                println(io, "  target_n: $(get(mu_convergence, "target_n", "N/A")), n_tol: $(get(mu_convergence, "n_tol", "N/A"))")
                println(io, "  mu_mean/std: $(get(mu_convergence, "mu_mean", "N/A")) / $(get(mu_convergence, "mu_std", "N/A"))")
                println(io, "  best_n_mean/std: $(get(mu_convergence, "best_n_mean", "N/A")) / $(get(mu_convergence, "best_n_std", "N/A"))")
                println(io, "  seed_list: $(get(mu_convergence, "seed_list", "N/A"))")
            end
            println(io, "")
        end

        # Seed diagnostics and readiness summary
        if seed_diagnostics isa AbstractDict
            println(io, "-"^90)
            println(io, "SEED DIAGNOSTICS & READINESS")
            println(io, "-"^90)
            println(io, "  available: $(get(seed_diagnostics, "available", false))")
            if haskey(seed_diagnostics, "reason")
                println(io, "  reason: $(seed_diagnostics["reason"])")
            end
            if get(seed_diagnostics, "available", false)
                summary = get(seed_diagnostics, "summary", Dict{String, Any}())
                println(io, "  seed_count: $(get(seed_diagnostics, "seed_count", "N/A"))")
                println(io, "  target_num_measurements_per_seed: $(get(seed_diagnostics, "target_num_measurements_per_seed", "N/A"))")
                println(io, "  seeds_finished/running/other: $(get(summary, "seeds_finished", 0))/$(get(summary, "seeds_running", 0))/$(get(summary, "seeds_other", 0))")
                println(io, "  total_measurements: $(get(summary, "total_measurements", "N/A"))")
                println(io, "  total_effective_measurements_estimate: $(round(get(summary, "total_effective_measurements_estimate", 0.0), digits=3))")
                println(io, "  point_worst_tau: $(get(summary, "point_worst_tau", "N/A"))")
                println(io, "  point_worst_seed/observable: $(get(summary, "point_worst_seed", "N/A")) / $(get(summary, "point_worst_observable", "N/A"))")

                readiness = get(seed_diagnostics, "readiness", nothing)
                if readiness isa AbstractDict
                    println(io, "  readiness.ready_to_declare_finished: $(get(readiness, "ready_to_declare_finished", "N/A"))")
                    println(io, "  readiness.criteria: $(get(readiness, "criteria", "N/A"))")
                    reasons = get(readiness, "reasons", String[])
                    println(io, "  readiness.reasons:")
                    for r in reasons
                        println(io, "    - $r")
                    end
                end
            end
            println(io, "")
        end

        # Delta extrapolation summary
        if delta_extrapolation isa AbstractDict
            println(io, "-"^90)
            println(io, "DELTA->0 EXTRAPOLATION")
            println(io, "-"^90)
            println(io, "  available: $(get(delta_extrapolation, "available", false))")
            if haskey(delta_extrapolation, "reason")
                println(io, "  reason: $(delta_extrapolation["reason"])")
            end
            if get(delta_extrapolation, "available", false)
                println(io, "  num_extrapolated_observables: $(get(delta_extrapolation, "num_extrapolated_observables", 0))")
                println(io, "  extrapolated_observables: $(get(delta_extrapolation, "extrapolated_observables", String[]))")
                warnings = get(delta_extrapolation, "warnings", String[])
                if !isempty(warnings)
                    println(io, "  warnings:")
                    for w in warnings
                        println(io, "    - $w")
                    end
                end
            end
            println(io, "")
        end
        
        # Bootstrap-derived quantities
        println(io, "-"^90)
        println(io, "BOOTSTRAP-DERIVED QUANTITIES")
        println(io, "-"^90)
        for (name, val) in sort(collect(bootstrap_results), by=x->x[1])
            if val isa Tuple && length(val) >= 2
                # Check if the values are scalars or arrays
                if val[1] isa Number && val[2] isa Number
                    println(io, @sprintf("  %-30s: %15.6g ± %.6g", name, val[1], val[2]))
                elseif val[1] isa AbstractMatrix
                    # For matrix quantities like resistivity tensor
                    println(io, "  $name (mean):")
                    for row in eachrow(val[1])
                        println(io, "    ", join([@sprintf("%12.6g", x) for x in row], "  "))
                    end
                    println(io, "  $name (error):")
                    for row in eachrow(val[2])
                        println(io, "    ", join([@sprintf("%12.6g", x) for x in row], "  "))
                    end
                else
                    println(io, "  $name: $(val[1]) ± $(val[2])")
                end
            elseif val isa AbstractArray
                println(io, "  $name: $(val)")
            end
        end
        println(io, "-"^90)
        println(io, "")
        
        # Correlations
        println(io, "-"^90)
        println(io, "JOINT DISTRIBUTION CORRELATIONS")
        println(io, "-"^90)
        for (name, corr) in sort(collect(correlations), by=x->x[1])
            println(io, @sprintf("  %-30s: r = %.4f", name, corr))
        end
        println(io, "-"^90)
        println(io, "")
        
        println(io, "="^90)
        println(io, "END OF REPORT")
        println(io, "="^90)
    end
    
    println("  Statistics saved to: $filepath")
    return filepath
end

# ============================================================================
# SECTION 9: PDF Report Generation
# ============================================================================

# Try to load PDFmerger for PDF merging
const HAS_PDFMERGER = try
    @eval import PDFmerger
    true
catch
    false
end

"""
Create a readable text-only PDF page using Python reportlab.
"""
function create_stats_page(output_path::String, stats_text::String)
    # Convert path to forward slashes for Python
    py_output_path = replace(output_path, "\\" => "/")
    
    script_content = """
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Preformatted, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import sys

text_content = sys.stdin.read()

# Create PDF
doc = SimpleDocTemplate(r"$py_output_path", pagesize=letter, topMargin=0.3*inch, bottomMargin=0.3*inch)
styles = getSampleStyleSheet()

# Create a monospace style for the text with a Unicode-capable font
mono_style = ParagraphStyle(
    'Mono',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=9,
    leading=11,
    textColor=colors.black,
    leftIndent=0,
    rightIndent=0
)

# Split text into lines and create story
story = []
for line in text_content.split('\\n'):
    story.append(Preformatted(line, mono_style))

# Build PDF
doc.build(story)
print("SUCCESS")
"""
    
    try
        # Create inline Python command
        proc = open(`python -c $script_content`, "w")
        print(proc, stats_text)
        close(proc)
        
        return isfile(output_path)
    catch e
        println("    Error creating text PDF: $e")
        return false
    end
end

"""
Add bookmarks to a PDF file using Python pypdf.
"""
function add_pdf_bookmarks(pdf_path::String, bookmarks::Dict{String, Int})
    # Convert path to forward slashes for Python
    py_pdf_path = replace(pdf_path, "\\" => "/")
    
    script_content = """
import sys
from PyPDF2 import PdfReader, PdfWriter

try:
    pdf_path = r"$py_pdf_path"
    
    # Read the PDF
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    # Copy all pages
    for page in reader.pages:
        writer.add_page(page)
    
    # Read bookmarks from stdin
    bookmarks_dict = {}
    for line in sys.stdin:
        line = line.strip()
        if line:
            parts = line.split('|')
            if len(parts) == 2:
                try:
                    title = parts[0]
                    page_num = int(parts[1]) - 1  # Convert to 0-indexed
                    bookmarks_dict[title] = page_num
                except:
                    pass
    
    # Add bookmarks in order using add_outline_item (new API)
    for title in sorted(bookmarks_dict.keys(), key=lambda x: bookmarks_dict[x]):
        page_num = bookmarks_dict[title]
        if page_num < len(writer.pages):
            try:
                # Try new API first
                writer.add_outline_item(title, page_num)
            except:
                # Fallback to old API if new one doesn't work
                try:
                    writer.add_bookmark(title, page_num)
                except:
                    pass
    
    # Write output
    with open(pdf_path, 'wb') as f:
        writer.write(f)
    
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    try
        # Create bookmark text
        bookmark_text = ""
        for (title, page) in sort(collect(bookmarks), by=x->x[2])
            bookmark_text *= "$title|$page\n"
        end
        
        # Run Python with bookmarks piped in
        proc = open(`python -c $script_content`, "w")
        print(proc, bookmark_text)
        close(proc)
        
        return true
    catch e
        println("    Error adding bookmarks: $e")
        return false
    end
end

"""
Create a section title page.
"""
function create_section_title(output_path::String, title::String, subtitle::String="")
    p = plot(legend=false, axis=false, size=(1200, 800), margin=0Plots.mm)
    
    # Title
    annotate!(p, 0.5, 0.65, text(title, 36, :center, :bold, font("Arial")))
    
    # Subtitle
    if !isempty(subtitle)
        annotate!(p, 0.5, 0.45, text(subtitle, 20, :center, font("Arial")))
    end
    
    # Line
    plot!([0.1, 0.9], [0.40, 0.40], linewidth=2, color=:black, label="")
    
    xlims!(p, 0, 1)
    ylims!(p, 0, 1)
    
    try
        savefig(p, output_path)
        return true
    catch e
        println("    Error creating title page: $e")
        return false
    end
end

"""
Merge multiple PDF files into a single PDF using PDFmerger.jl.
Install with: using Pkg; Pkg.add("PDFmerger")
"""
function merge_pdfs_julia(pdf_files::Vector{String}, output_path::String)
    if !HAS_PDFMERGER
        return false
    end
    
    try
        # PDFmerger API: merge_pdfs(["file1.pdf", "file2.pdf"], "output.pdf")
        PDFmerger.merge_pdfs(pdf_files, output_path)
        return true
    catch e
        println("    Julia PDF merge error: $e")
        return false
    end
end

"""
Merge PDFs using system Python (more reliable fallback).
"""
function merge_pdfs_python(pdf_files::Vector{String}, output_path::String)
    # Create a temporary Python script file
    script_content = """
import sys
try:
    from PyPDF2 import PdfMerger
except ImportError:
    print("MISSING_PYPDF2")
    sys.exit(1)

# Read file list from the text file
with open(sys.argv[1], 'r') as f:
    files = [line.strip() for line in f if line.strip()]

output = sys.argv[2]

merger = PdfMerger()
for pdf in files:
    merger.append(pdf)
merger.write(output)
merger.close()
print("SUCCESS")
"""
    
    try
        # Create a temp file with the list of PDFs
        temp_dir = dirname(output_path)
        list_file = joinpath(temp_dir, "_pdf_list.txt")
        script_file = joinpath(temp_dir, "_merge_script.py")
        
        # Write PDF file list
        open(list_file, "w") do io
            for pdf in pdf_files
                println(io, abspath(pdf))
            end
        end
        
        # Write Python script
        open(script_file, "w") do io
            print(io, script_content)
        end
        
        # Run Python script
        result = read(`python $script_file $list_file $(abspath(output_path))`, String)
        
        # Cleanup temp files
        rm(list_file, force=true)
        rm(script_file, force=true)
        
        if occursin("SUCCESS", result)
            return true
        elseif occursin("MISSING_PYPDF2", result)
            println("    Python found but PyPDF2 not installed. Run: pip install PyPDF2")
            return false
        end
    catch e
        # Python not available or failed
        println("    Python merge error: $e")
        return false
    end
    return false
end

"""
Generate a PDF report with all plots organized.
Saves individual PDF pages and attempts to merge them.
Includes title pages and statistics.
"""
function generate_pdf_report(output_dir::String, all_plots::Dict, 
                              observables_to_analyze::Vector{Symbol},
                              stats_text::String;
                              filename::String="full_report.pdf")
    
    pdf_paths = String[]  # Full paths
    pdf_names = String[]  # Just filenames
    page_titles = String[]
    page_number = 1
    
    println("  Generating PDF pages...")
    
    # Helper to save a plot as PDF with numbered name
    function save_pdf_page(fig, name::String, title::String)
        pdf_name = @sprintf("%02d_%s.pdf", page_number, name)
        pdf_path = joinpath(output_dir, pdf_name)
        try
            savefig(fig, pdf_path)
            push!(pdf_paths, pdf_path)
            push!(pdf_names, pdf_name)
            push!(page_titles, title)
            page_number += 1
            return true
        catch e
            println("    Warning: Could not save $name as PDF: $e")
            return false
        end
    end
    
    # Helper to save a text file as PDF page
    function save_text_page(text::String, name::String, title::String)
        pdf_name = @sprintf("%02d_%s.pdf", page_number, name)
        pdf_path = joinpath(output_dir, pdf_name)
        try
            create_stats_page(pdf_path, text)
            push!(pdf_paths, pdf_path)
            push!(pdf_names, pdf_name)
            push!(page_titles, title)
            page_number += 1
            return true
        catch e
            println("    Warning: Could not create text page $name: $e")
            return false
        end
    end
    
    # Helper to save a section title
    function save_title_page(title::String, name::String, subtitle::String="")
        pdf_name = @sprintf("%02d_%s.pdf", page_number, name)
        pdf_path = joinpath(output_dir, pdf_name)
        try
            create_section_title(pdf_path, title, subtitle)
            push!(pdf_paths, pdf_path)
            push!(pdf_names, pdf_name)
            push!(page_titles, "Section: $title")
            page_number += 1
            return true
        catch e
            println("    Warning: Could not create title page $name: $e")
            return false
        end
    end
    
    # ===== Cover & Statistics Pages =====
    println("  Adding cover page and statistics...")
    save_title_page("SIMULATION STATISTICS REPORT", "00_cover", "Monte Carlo Results")
    save_text_page(stats_text, "01_statistics", "Statistics Summary")
    
    # ===== Observable Analysis Pages =====
    println("  Adding observable analysis pages...")
    save_title_page("OBSERVABLE ANALYSIS", "02_obs_title", "Detailed Per-Observable Statistics")
    
    for obs in observables_to_analyze
        key = "$(obs)_analysis"
        if haskey(all_plots, key)
            plots = all_plots[key]
            if !isnothing(plots)
                try
                    # Add title for this observable
                    save_title_page("Observable: $(string(obs))", "obs_$(obs)_title_page", "6-plot Analysis")
                    
                    # Add dashboard
                    dashboard = plot(plots[1], plots[3], plots[4], plots[5], plots[6], 
                                    layout=(3, 2), size=(1200, 1200))
                    save_pdf_page(dashboard, "obs_$(obs)_dashboard", "Observable: $obs - Dashboard")
                    
                    # Add histogram
                    save_pdf_page(plots[2], "obs_$(obs)_histogram", "Observable: $obs - Histogram")
                catch e
                    println("    Warning: Could not create pages for $obs: $e")
                end
            end
        end
    end
    
    # ===== Vector Observable Pages =====
    println("  Adding vector observable pages...")
    save_title_page("VECTOR OBSERVABLES", "03_vector_title", "Correlations & Responses")
    
    for (key, title) in [("Lxx_vs_q", "Current Correlations"),
                          ("conductivity_vs_omega", "Conductivity"),
                          ("conductivity_dc_vs_dw", "DC Conductivity vs Δω"),
                          ("stiffness_delta_vs_delta", "Stiffness vs Broadening δ"),
                          ("phase_correlation", "Phase Correlation"),
                          ("Ps_l", "Pair Structure")]
        if haskey(all_plots, key)
            save_title_page(title, "title_$key")
            save_pdf_page(all_plots[key], key, title)
        end
    end
    
    # ===== Joint Distribution Pages =====
    println("  Adding joint distribution pages...")
    save_title_page("JOINT DISTRIBUTIONS", "04_joint_title", "Observable Correlations")
    
    for (i, key) in enumerate(["Lxx_qy_1_vs_Kx", "Lxx_D_1_vs_Kx", "n_vs_DdD"])
        if haskey(all_plots, key)
            title_text = 
                if key == "Lxx_qy_1_vs_Kx"
                    "Current Correlation vs Momentum"
                elseif key == "Lxx_D_1_vs_Kx"
                    "Current Correlation vs Amplitude"
                else
                    "Occupation vs Order Parameter"
                end
            save_pdf_page(all_plots[key], key, "Joint: $title_text")
        end
    end
    
    # ===== Complex Field Pages =====
    println("  Adding complex field pages...")
    save_title_page("COMPLEX MEAN FIELD", "05_complex_title", "s-wave Trajectory Analysis")
    
    if haskey(all_plots, "complex_field_trajectory")
        save_pdf_page(all_plots["complex_field_trajectory"], "complex_trajectory", 
                      "Complex Mean Field - Trajectory")
    end
    if haskey(all_plots, "complex_field_phase_hist")
        save_pdf_page(all_plots["complex_field_phase_hist"], "complex_phase", 
                      "Complex Mean Field - Phase")
    end
    if haskey(all_plots, "complex_field_amp_hist")
        save_pdf_page(all_plots["complex_field_amp_hist"], "complex_amplitude", 
                      "Complex Mean Field - Amplitude")
    end
    
    println("  Saved $(length(pdf_paths)) total PDF pages")
    
    # Create Table of Contents file
    toc_path = joinpath(dirname(output_dir), "00_index.txt")
    open(toc_path, "w") do io
        println(io, "PDF Report - Table of Contents")
        println(io, "="^50)
        println(io, "")
        for (i, (pdf, title)) in enumerate(zip(pdf_names, page_titles))
            println(io, @sprintf("%2d. %-40s  [%s]", i, title, pdf))
        end
    end
    
    # Try to merge PDFs
    merged_path = joinpath(dirname(output_dir), filename)
    merge_success = false
    
    println("  Attempting PDF merge...")
    
    # Try Julia PDFmerger first
    if HAS_PDFMERGER
        println("    Using Julia PDFmerger...")
        merge_success = merge_pdfs_julia(pdf_paths, merged_path)
    end
    
    # Fall back to Python if Julia didn't work
    if !merge_success
        println("    Trying Python PyPDF2...")
        merge_success = merge_pdfs_python(pdf_paths, merged_path)
    end
    
    if merge_success
        println("  ✓ Merged PDF saved: $merged_path")
        
        # Try to add bookmarks
        println("  Adding bookmarks to PDF...")
        try
            # Create bookmarks dictionary: title -> page number
            bookmarks = Dict{String, Int}()
            for (i, title) in enumerate(page_titles)
                bookmarks[title] = i + 1  # Pages are 1-indexed for bookmarks
            end
            
            if add_pdf_bookmarks(merged_path, bookmarks)
                println("  ✓ Bookmarks added successfully")
            else
                println("  Note: Could not add bookmarks. Bookmarks require reportlab and pypdf.")
            end
        catch e
            println("  Note: Bookmarks require: pip install reportlab PyPDF2")
        end
    else
        println("  Auto-merge not available. Options:")
        println("    1. Julia: using Pkg; Pkg.add(\"PDFmerger\"), then re-run")
        println("    2. Python: pip install PyPDF2, then re-run")
        println("    3. Online: ilovepdf.com or smallpdf.com")
    end
    
    return pdf_paths, page_titles, merge_success
end

# ============================================================================
# SECTION 10: Main Report Generator
# ============================================================================

"""
    generate_report(sim_path; save_plots=false, output_dir=nothing, observables_to_analyze=[...],
                    tau_mult=10.0, seed_index=1, generate_pdf=false, kwargs...)
    generate_report(data::LatticeData, actual_path; ...)

Generate a comprehensive statistical analysis report for a simulation.
The `sim_path` method loads the data (`load_simulation_data`) and calls the
`LatticeData` method, which builds the report from data already in memory.

Arguments:
- sim_path: Path to simulation directory
- save_plots: Whether to save plots to disk (default: false)
- output_dir: Directory to save plots (default: sim_path/statistics_report)
- observables_to_analyze: List of observables for detailed analysis (default: key observables + vector observables)
- tau_mult: Multiplier for τ when determining block size (default: 10.0)
- seed_index: Which seed to load for multi-seed simulations (default: 1)
- generate_pdf: Whether to generate PDF report (default: true when save_plots=true)
"""
function generate_report(sim_path::String;
                         save_plots::Bool=false,
                         output_dir::Union{String, Nothing}=nothing,
                         observables_to_analyze::Vector{Symbol}=[:n, :Kx, :DdD, :Delta_mean, :chi_spin, :chi_charge, :DOS,
                                                                 :Lxx_qy, :Lxx_qx, :Lxy_qx, :Lxy_qy, :Lyx_qx, :Lyx_qy, :Lyy_qx, :Lyy_qy,
                                                                   :Phase_Corr, :Ps_l],
                         tau_mult::Float64=10.0,
                         seed_index::Int=1,
                         generate_pdf::Bool=false,
                                                 print_timing::Bool=true,
                                                 include_seed_diagnostics::Bool=true,
                                                 include_delta_extrapolation::Bool=true,
                                                 include_enhanced_transport_summary::Bool=true,
                                                 readiness_min_completed_seeds::Int=1,
                                                 readiness_min_measurements_per_seed::Int=0,
                                                 readiness_min_effective_measurements_per_seed::Float64=0.0)
    println("\n" * "="^80)
    println("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    println("Generated: $(Dates.now())")
    println("="^80)
    
    # Load data
    println("\n[1/9] Loading simulation data...")
    data, actual_path = load_simulation_data(sim_path; seed_index=seed_index)
    println("  Loaded data from: $actual_path")

    result = generate_report(data, actual_path; 
                         save_plots=save_plots,
                         output_dir=output_dir,
                         observables_to_analyze=observables_to_analyze,
                         tau_mult=tau_mult,
                         seed_index=seed_index,
                         generate_pdf=generate_pdf,
                         print_timing=print_timing,
                         include_seed_diagnostics=include_seed_diagnostics,
                         include_delta_extrapolation=include_delta_extrapolation,
                         include_enhanced_transport_summary=include_enhanced_transport_summary,
                         readiness_min_completed_seeds=readiness_min_completed_seeds,
                         readiness_min_measurements_per_seed=readiness_min_measurements_per_seed,
                         readiness_min_effective_measurements_per_seed=readiness_min_effective_measurements_per_seed)
    
    # Release the large loaded data to free memory
    data = nothing
    GC.gc()
    
    return result
end

function generate_report(data::LatticeData, actual_path::String; 
                         save_plots::Bool=false,
                         output_dir::Union{String, Nothing}=nothing,
                         observables_to_analyze::Vector{Symbol}=[:n, :Kx, :DdD, :Delta_mean, :chi_spin, :chi_charge, :DOS,
                                                                 :Lxx_qy, :Lxx_qx, :Lxy_qx, :Lxy_qy, :Lyx_qx, :Lyx_qy, :Lyy_qx, :Lyy_qy,
                                                                   :Phase_Corr, :Ps_l],
                         tau_mult::Float64=10.0,
                         seed_index::Int=1,
                         generate_pdf::Bool=false,
                                                 print_timing::Bool=true,
                                                 include_seed_diagnostics::Bool=true,
                                                 include_delta_extrapolation::Bool=true,
                                                 include_enhanced_transport_summary::Bool=true,
                                                 readiness_min_completed_seeds::Int=1,
                                                 readiness_min_measurements_per_seed::Int=0,
                                                 readiness_min_effective_measurements_per_seed::Float64=0.0)
    
    println("\n" * "="^80)
    println("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    println("Generated: $(Dates.now())")
    println("="^80)
    
    # Load data
    println("\n[1/9] Loading simulation data...")
    measurements = data.measurement_data
    num_meas = data.current_state.num_current_measurments
    sim_data = data.sim_data
    scalar_obs, vector_obs = list_observables(measurements)
    
    # --- Try loading cached means ---
    cached_means = nothing
    cached = load_means_cache(actual_path)
    if cached !== nothing
        cached_N = cached["num_measurements"]
        actual_N = num_meas
        if cached_N == actual_N
            println("  Using exact cached means (N=$cached_N)")
            cached_means = cached["means"]
        elseif cached_N >= actual_N * 0.9
            println("  Using cached means (cached N=$cached_N, actual N=$actual_N, within 10% tolerance)")
            cached_means = cached["means"]
        else
            println("  Cached means stale (cached N=$cached_N, actual N=$actual_N), will recompute.")
        end
    else
        println("  No means cache found, will compute from scratch.")
    end

    simulation_root, selected_seed_name = report_resolve_simulation_root(actual_path)

    seed_diagnostics_report = include_seed_diagnostics ? report_build_seed_diagnostics(
        simulation_root;
        include_readiness=true,
        min_completed_seeds=readiness_min_completed_seeds,
        min_measurements_per_seed=readiness_min_measurements_per_seed,
        min_effective_measurements_per_seed=readiness_min_effective_measurements_per_seed,
    ) : Dict{String, Any}(
        "available" => false,
        "reason" => "Seed diagnostics disabled by caller",
    )

    mu_convergence_report = report_build_mu_convergence(sim_data, seed_diagnostics_report, cached_means)

    enhanced_means = cached_means
    if (include_delta_extrapolation || include_enhanced_transport_summary) && isnothing(enhanced_means)
        println("  Building means dictionary for enhanced report sections...")
        try
            enhanced_means = get_means(data)
        catch e
            println("  Warning: failed to compute enhanced means: $e")
            enhanced_means = nothing
        end
    end

    delta_extrapolation_report = Dict{String, Any}(
        "available" => false,
        "reason" => "Delta extrapolation not requested",
    )

    transport_summary_report = Dict{String, Any}(
        "available" => false,
        "reason" => "Transport summary not requested",
    )
    
    # Setup output directory structure
    if save_plots
        if isnothing(output_dir)
            output_dir = joinpath(actual_path, "statistics_report")
        end
        mkpath(output_dir)
        
        # Create subdirectories
        dirs = Dict{String, String}()
        for subdir in ["observables", "vector_observables", "joint_distributions", 
                       "complex_field", "pdf_pages"]
            dirs[subdir] = joinpath(output_dir, subdir)
            mkpath(dirs[subdir])
        end
        
        println("  Output directory: $output_dir")
        println("  Subdirectories created for organized output")
    end
    
    all_plots = Dict{String, Any}()
    timing_log = Tuple{String, Float64}[]
    
    # --- Summary Table ---
    println("\n[2/9] Generating summary statistics...")
    flush(stdout)
    t_step = time()
    results_dict = print_summary_table(data; tau_mult=tau_mult, printoutput=!save_plots, cached_means=cached_means)
    flush(stdout)
    elapsed_step = time() - t_step
    push!(timing_log, ("[2/9] Summary statistics", elapsed_step))
    print_timing && println("  [2/9] completed in $(round(elapsed_step, digits=1))s")
    
    # --- Per-Observable Detailed Analysis ---
    println("\n[3/9] Detailed analysis of key observables...")
    flush(stdout)
    t_step = time()
    for obs in observables_to_analyze
        if !hasfield(typeof(measurements), obs)
            println("  Skipping $obs (not found)")
            continue
        end
        isvec = obs in keys(vector_obs)
        
        println("  Analyzing: $obs")
        try
            if !isvec
                # Single observable
                stats, plots = analyze_single_observable(measurements, obs, num_meas; 
                                                      tau_mult=tau_mult, show_plots=true)
            
                if !isnothing(plots)
                    all_plots["$(obs)_analysis"] = plots
                    
                    if save_plots
                        # Create subfolder for this observable
                        obs_dir = joinpath(dirs["observables"], string(obs))
                        mkpath(obs_dir)
                        
                        # Save individual plots
                        savefig(plots[1], joinpath(obs_dir, "01_timeseries.png"))
                        savefig(plots[2], joinpath(obs_dir, "02_histogram.png"))
                        savefig(plots[3], joinpath(obs_dir, "03_cumulative_mean.png"))
                        savefig(plots[4], joinpath(obs_dir, "04_blocking_analysis.png"))
                        savefig(plots[5], joinpath(obs_dir, "05_autocorrelation.png"))
                        savefig(plots[6], joinpath(obs_dir, "06_integrated_tau.png"))
                        
                        # Combined dashboard
                        dashboard = plot(plots[1], plots[3], plots[4], plots[5], plots[6], 
                                        layout=(3, 2), size=(1200, 1200))
                        savefig(dashboard, joinpath(obs_dir, "00_dashboard.png"))
                    end
                end
            else
                # Vector observable - analyze all indices
                stats_ixs = analyze_single_observable(measurements, obs, num_meas; 
                                                      tau_mult=tau_mult, show_plots=true, analyze_all_ixes=true)
                all_plots["$(obs)_analysis"] = Dict{String, Any}()
                
                # Subsample: generate 6-plot analysis for at most 20 evenly-spaced indices
                # Exception: chi_spin, chi_charge are q-dependent (few indices) — plot all
                all_ixs = sort(collect(keys(stats_ixs)))
                no_subsample_obs = (:chi_spin, :chi_charge)
                if length(all_ixs) > 20 && !(obs in no_subsample_obs)
                    step_sz = max(1, (length(all_ixs) - 1) ÷ 19)
                    selected_ixs = all_ixs[1:step_sz:end]
                    if last(selected_ixs) != last(all_ixs)
                        push!(selected_ixs, last(all_ixs))
                    end
                else
                    selected_ixs = all_ixs
                end
                
                for (ix, (stats, plots)) in stats_ixs
                    if !isnothing(plots) && ix in selected_ixs
                        all_plots["$(obs)_analysis"]["index_$(ix)"] = plots
                        if save_plots
                            # Create subfolder for this observable
                            obs_dir = joinpath(dirs["observables"], string(obs), "index_$(ix)")
                            mkpath(obs_dir)
                            
                            # Save individual plots
                            savefig(plots[1], joinpath(obs_dir, "01_timeseries.png"))
                            savefig(plots[2], joinpath(obs_dir, "02_histogram.png"))
                            savefig(plots[3], joinpath(obs_dir, "03_cumulative_mean.png"))
                            savefig(plots[4], joinpath(obs_dir, "04_blocking_analysis.png"))
                            savefig(plots[5], joinpath(obs_dir, "05_autocorrelation.png"))
                            savefig(plots[6], joinpath(obs_dir, "06_integrated_tau.png"))
                            
                            # Combined dashboard
                            dashboard = plot(plots[1], plots[3], plots[4], plots[5], plots[6], 
                                            layout=(3, 2), size=(1200, 1200))
                            savefig(dashboard, joinpath(obs_dir, "00_dashboard.png"))
                        end
                    end
                end
                ## ---- make a layout plot for vector observables ----
                # plot a few indices together with the proper q,w,l labels
                

            end
        catch e
            println("    Error: $e")
        end
    end
   
    elapsed_step = time() - t_step
    push!(timing_log, ("[3/9] Per-observable detailed analysis", elapsed_step))
    print_timing && println("  [3/9] completed in $(round(elapsed_step, digits=1))s")
   
    # --- Spectral & Reciprocal Space Plots ---
    println("\n[3b/9] Spectral and reciprocal space plots...")
    flush(stdout)
    t_step = time()
    
    if save_plots
        spectral_dir = joinpath(output_dir, "spectral_plots")
        mkpath(spectral_dir)
    end
    
    # Spectral function A
    println("  Spectral function A(k,ω)...")
    try
        fig_a = plot_spectral_function(data)
        if !isnothing(fig_a)
            all_plots["spectral_function"] = fig_a
            if save_plots
                savefig(fig_a, joinpath(spectral_dir, "01_spectral_function.png"))
            end
        end
    catch e
        println("    Note: Spectral function not available: $e")
    end
    
    # Brillouin zone
    println("  Brillouin zone...")
    try
        fig_bz = plot_brillouin_zone(data)
        if !isnothing(fig_bz)
            all_plots["brillouin_zone"] = fig_bz
            if save_plots
                savefig(fig_bz, joinpath(spectral_dir, "02_brillouin_zone.png"))
            end
        end
    catch e
        println("    Note: Brillouin zone not available: $e")
    end
    
    # Density of states
    println("  Density of states...")
    try
        fig_dos = plot_dos(data; cached_means=cached_means)
        if !isnothing(fig_dos)
            all_plots["dos"] = fig_dos
            if save_plots
                savefig(fig_dos, joinpath(spectral_dir, "03_dos.png"))
            end
        end
    catch e
        println("    Note: DOS not available: $e")
    end

    elapsed_step = time() - t_step
    push!(timing_log, ("[3b/9] Spectral & reciprocal space", elapsed_step))
    print_timing && println("  [3b/9] completed in $(round(elapsed_step, digits=1))s")

    # --- Fixed Delta Observables Analysis ---
    t_step = time()

    # Define the list based on your request
    fixed_delta_obs = [
        :Lxx_w_fixed, :Lxy_w_fixed, :Lyx_w_fixed, :Lyy_w_fixed
    ]

    if save_plots
        println("\n[3c/9] Analyzing Fixed-Delta Observables...")
        fixed_dir = joinpath(output_dir, "fixed_delta_observables")
        mkpath(fixed_dir)
        
        analyze_fixed_delta_group(data, fixed_delta_obs, fixed_dir; tau_mult=tau_mult, cached_means=cached_means)
        
        # Derived conductivity_regular and Drude weight plots
        try
            plot_derived_conductivity(data, fixed_dir; cached_means=cached_means)
        catch e
            println("    Error generating derived conductivity plots: $e")
        end
    end
    
    elapsed_step = time() - t_step
    push!(timing_log, ("[3c/9] Fixed-delta observables", elapsed_step))
    print_timing && println("  [3c/9] completed in $(round(elapsed_step, digits=1))s")

    # --- Delta→0 Extrapolation Diagnostics ---
    println("\n[3d/9] Delta→0 extrapolation diagnostics...")
    flush(stdout)
    t_step = time()
    if include_delta_extrapolation
        delta_extrapolation_report = report_generate_delta_extrapolation!(
            data,
            enhanced_means;
            save_plots=save_plots,
            output_dir=save_plots ? output_dir : nothing,
        )
        if get(delta_extrapolation_report, "available", false)
            enhanced_means = get(delta_extrapolation_report, "means_with_extrapolation", enhanced_means)
            println("  Extrapolated observables: $(get(delta_extrapolation_report, "num_extrapolated_observables", 0))")
        else
            println("  Extrapolation unavailable: $(get(delta_extrapolation_report, "reason", "Unknown reason"))")
        end
    else
        println("  Delta extrapolation section skipped by caller.")
    end

    elapsed_step = time() - t_step
    push!(timing_log, ("[3d/9] Delta extrapolation diagnostics", elapsed_step))
    print_timing && println("  [3d/9] completed in $(round(elapsed_step, digits=1))s")
    
    # --- Vector Observable Plots ---
    println("\n[4/9] Vector observable plots...")
    flush(stdout)
    t_step = time()
    
    if save_plots
        vector_dir = dirs["vector_observables"]
    end
    
    # Lxx vs q
    println("  Λxx vs q plots...")
    try
        fig_lxx = plot_all_Lxx_directions(data; cached_means=cached_means)
        all_plots["Lxx_vs_q"] = fig_lxx
        if save_plots
            lxx_dir = joinpath(vector_dir, "Lxx_vs_q")
            mkpath(lxx_dir)
            savefig(fig_lxx, joinpath(lxx_dir, "Lxx_all_directions.png"))
            # Also save individual directions
            for (dir1, dir2) in [("x","x"), ("y","y"), ("x","y"), ("y","x")]
                try
                    fig_single = plot_Lxx_vs_q(data; direction1=dir1, direction2=dir2, cached_means=cached_means)
                    savefig(fig_single, joinpath(lxx_dir, "L$(dir1)$(dir2)_vs_q.png"))
                catch
                end
            end
        end
    catch e
        println("    Error: $e")
    end
    
    # NOTE: Conductivity vs ω plots removed — now generated from L*_w_fixed
    # inside the fixed-delta analysis section (see fixed_delta_obs block above).
    
    # Phase correlation
    println("  Phase correlation...")
    try
        fig_phase = plot_phase_correlation(data; cached_means=cached_means)
        all_plots["phase_correlation"] = fig_phase
        if save_plots
            phase_dir = joinpath(vector_dir, "phase_correlation")
            mkpath(phase_dir)
            savefig(fig_phase, joinpath(phase_dir, "phase_correlation.png"))
        end
    catch e
        println("    Error: $e")
    end
    
    # Conductivity DC vs Δω
    println("  DC Conductivity vs Δω...")
    try
        fig_dc = plot_conductivity_dc_vs_dw(data; cached_means=cached_means)
        if !isnothing(fig_dc)
            all_plots["conductivity_dc_vs_dw"] = fig_dc
            if save_plots
                dc_dir = joinpath(vector_dir, "conductivity_dc")
                mkpath(dc_dir)
                savefig(fig_dc, joinpath(dc_dir, "conductivity_dc_vs_dw_all.png"))
                
                # Individual component plots
                for field_name in [:conductivity_dc_xx, :conductivity_dc_xy, :conductivity_dc_yx, :conductivity_dc_yy]
                    try
                        fig_comp = plot_conductivity_dc_component(data, field_name; cached_means=cached_means)
                        if !isnothing(fig_comp)
                            all_plots["$(field_name)_vs_dw"] = fig_comp
                            savefig(fig_comp, joinpath(dc_dir, "$(field_name)_vs_dw.png"))
                        end
                    catch
                    end
                end
                
                # Detailed per-Δω statistical analysis
                println("  DC Conductivity — detailed per-Δω analysis...")
                analyze_conductivity_dc(data, joinpath(dc_dir, "per_dw_analysis"); 
                                        tau_mult=tau_mult, save_plots=true)
            end
        end
    catch e
        println("    Error: $e")
    end
    
    # NOTE: Stiffness vs broadening δ plots removed — L*_q*_delta fields no longer exist.
    # Zero-broadening stiffness from Kx + Lxx_qy (via BT_stiffness) remains in bootstrap quantities.
    
    # Ps_l
    println("  Pair structure factor Ps(l)...")
    try
        fig_ps = plot_Ps_l(data; cached_means=cached_means)
        all_plots["Ps_l"] = fig_ps
        if save_plots
            ps_dir = joinpath(vector_dir, "Ps_l")
            mkpath(ps_dir)
            savefig(fig_ps, joinpath(ps_dir, "Ps_l.png"))
        end
    catch e
        println("    Error: $e")
    end
    
    elapsed_step = time() - t_step
    push!(timing_log, ("[4/9] Vector observable plots", elapsed_step))
    print_timing && println("  [4/9] completed in $(round(elapsed_step, digits=1))s")
    
    # --- Bootstrap Quantities ---
    println("\n[5/9] Bootstrap-derived quantities...")
    flush(stdout)
    t_step = time()
    bootstrap_results = compute_bootstrap_quantities(data; cached_means=cached_means)
    
    elapsed_step = time() - t_step
    push!(timing_log, ("[5/9] Bootstrap quantities", elapsed_step))
    print_timing && println("  [5/9] completed in $(round(elapsed_step, digits=1))s")
    
    # --- Joint Distributions ---
    println("\n[6/9] Joint distributions...")
    flush(stdout)
    t_step = time()
    joint_plots, correlations = plot_all_joint_distributions(data)
    
    if save_plots
        joint_dir = dirs["joint_distributions"]
    end
    
    for (name, fig) in joint_plots
        all_plots[name] = fig
        if save_plots
            savefig(fig, joinpath(joint_dir, "$name.png"))
        end
    end
    
    elapsed_step = time() - t_step
    push!(timing_log, ("[6/9] Joint distributions", elapsed_step))
    print_timing && println("  [6/9] completed in $(round(elapsed_step, digits=1))s")
    
    # --- Complex Field Trajectory ---
    println("\n[7/9] Complex mean field trajectory (s-wave)...")
    flush(stdout)
    t_step = time()
    try
        fig_traj, fig_phase_hist, fig_amp_hist, fig_amp_vs_step, fig_phase_vs_step, comp_name = 
            plot_complex_field_trajectory(data; max_points=10000, component=1)
        all_plots["complex_field_trajectory"] = fig_traj
        all_plots["complex_field_phase_hist"] = fig_phase_hist
        all_plots["complex_field_amp_hist"] = fig_amp_hist
        all_plots["complex_field_amp_vs_step"] = fig_amp_vs_step
        all_plots["complex_field_phase_vs_step"] = fig_phase_vs_step
        
        if save_plots
            complex_dir = dirs["complex_field"]
            savefig(fig_traj, joinpath(complex_dir, "01_trajectory.png"))
            savefig(fig_phase_hist, joinpath(complex_dir, "02_phase_distribution.png"))
            savefig(fig_amp_hist, joinpath(complex_dir, "03_amplitude_distribution.png"))
            savefig(fig_amp_vs_step, joinpath(complex_dir, "04_amplitude_vs_step.png"))
            savefig(fig_phase_vs_step, joinpath(complex_dir, "05_phase_vs_step.png"))
            
            # Combined complex field dashboard
            combined_complex = plot(fig_traj, fig_phase_hist, fig_amp_hist, 
                                   layout=(1, 3), size=(2100, 600))
            savefig(combined_complex, joinpath(complex_dir, "00_dashboard.png"))
        end
    catch e
        println("  Error: $e")
    end
    
    elapsed_step = time() - t_step
    push!(timing_log, ("[7/9] Complex field trajectory", elapsed_step))
    print_timing && println("  [7/9] completed in $(round(elapsed_step, digits=1))s")

    if include_enhanced_transport_summary
        transport_summary_report = report_build_transport_tensor_summary(enhanced_means, sim_data)
    end
    
    # --- Save Statistics Text File ---
    println("\n[8/9] Saving statistics text file...")
    flush(stdout)
    t_step = time()
    stats_filepath = nothing
    if save_plots
        report_metadata = Dict{String, Any}(
            "actual_path" => actual_path,
            "simulation_root" => simulation_root,
            "selected_seed" => selected_seed_name,
            "seed_index" => seed_index,
            "used_cached_means" => !isnothing(cached_means),
            "cache_available" => cached !== nothing,
            "cache_num_measurements" => cached === nothing ? nothing : get(cached, "num_measurements", nothing),
            "cache_schema_version" => cached === nothing ? nothing : get(cached, "schema_version", nothing),
            "include_seed_diagnostics" => include_seed_diagnostics,
            "include_delta_extrapolation" => include_delta_extrapolation,
            "include_enhanced_transport_summary" => include_enhanced_transport_summary,
        )
        stats_filepath = write_statistics_file(output_dir, sim_data, results_dict, bootstrap_results, correlations;
            report_metadata=report_metadata,
            seed_diagnostics=seed_diagnostics_report,
            mu_convergence=mu_convergence_report,
            delta_extrapolation=delta_extrapolation_report,
            transport_summary=transport_summary_report)

        marker_path = joinpath(output_dir, "enhanced_report_marker.txt")
        open(marker_path, "w") do io
            println(io, "generated_at=$(Dates.now())")
            println(io, "actual_path=$actual_path")
            println(io, "used_cached_means=$(!isnothing(cached_means))")
            println(io, "seed_diagnostics_available=$(get(seed_diagnostics_report, "available", false))")
            println(io, "mu_convergence_available=$(get(mu_convergence_report, "available", false))")
            println(io, "delta_extrapolation_available=$(get(delta_extrapolation_report, "available", false))")
            println(io, "delta_extrapolated_observables=$(get(delta_extrapolation_report, "num_extrapolated_observables", 0))")
            println(io, "transport_summary_available=$(get(transport_summary_report, "available", false))")
        end
    end
    
    elapsed_step = time() - t_step
    push!(timing_log, ("[8/9] Save statistics text file", elapsed_step))
    print_timing && println("  [8/9] completed in $(round(elapsed_step, digits=1))s")
    
    # --- Generate PDF Report ---
    println("\n[9/9] Generating PDF report...")
    flush(stdout)
    t_step = time()
    if save_plots && generate_pdf
        try
            # Read the statistics text file
            stats_text = ""
            if !isnothing(stats_filepath) && isfile(stats_filepath)
                stats_text = read(stats_filepath, String)
            end
            
            pdf_pages, page_titles, merged = generate_pdf_report(
                dirs["pdf_pages"], all_plots, observables_to_analyze, stats_text)
        catch e
            println("  PDF generation error: $e")
        end
    else
        println("  PDF report generation skipped.")
    end
    
    elapsed_step = time() - t_step
    push!(timing_log, ("[9/9] PDF report generation", elapsed_step))
    print_timing && println("  [9/9] completed in $(round(elapsed_step, digits=1))s")
    
    # --- Timing Summary ---
    total_time = sum(t[2] for t in timing_log)
    if print_timing
        println("\nTIMING SUMMARY:")
        println("-"^70)
        for (name, t) in timing_log
            pct = round(t / total_time * 100, digits=1)
            println("  $(rpad(name, 42)) $(lpad(string(round(t, digits=1)), 8))s  ($pct%)")
        end
        println("-"^70)
        println("  $(rpad("TOTAL", 42)) $(lpad(string(round(total_time, digits=1)), 8))s")
        println("-"^70)
    end
    
    # Always save timing report to file when saving plots
    if save_plots
        timing_path = joinpath(output_dir, "timing_report.txt")
        open(timing_path, "w") do f
            println(f, "TIMING REPORT — $(Dates.now())")
            println(f, "Simulation: $actual_path")
            println(f, "-"^70)
            for (name, t) in timing_log
                pct = round(t / total_time * 100, digits=1)
                println(f, "  $(rpad(name, 42)) $(lpad(string(round(t, digits=1)), 8))s  ($pct%)")
            end
            println(f, "-"^70)
            println(f, "  $(rpad("TOTAL", 42)) $(lpad(string(round(total_time, digits=1)), 8))s")
        end
        println("  Timing report saved to: $timing_path")
    end
    
    # --- Final Summary ---
    println("\n" * "="^80)
    println("REPORT COMPLETE")
    println("="^80)
    n_plots = length(all_plots)
    println("  Total plots generated: $n_plots")
    if save_plots
        println("  Plots saved to: $output_dir")
        println("  Directory structure:")
        println("    ├── observables/          # Per-observable analysis folders")
        println("    ├── vector_observables/   # Lxx(q), σ(ω), σ_dc(Δω), ρs(δ), Phase_Corr, Ps(l)")
        println("    ├── joint_distributions/  # 2D correlation plots")
        println("    ├── complex_field/        # Δ trajectory plots")
        println("    ├── pdf_pages/            # Individual PDF pages")
        println("    └── statistics_summary.txt")
        
        # Release plot objects — they are already saved to disk
        empty!(all_plots)
        println("  Memory: released $n_plots plot objects (saved to disk)")
    end
    println("="^80)
    flush(stdout)
    
    return all_plots, results_dict, bootstrap_results, correlations, data
end

# ============================================================================
# Quick Display Functions
# ============================================================================

"""
Display a specific plot from the report.
"""
function show_plot(all_plots::Dict, plot_name::String)
    if haskey(all_plots, plot_name)
        display(all_plots[plot_name])
    else
        println("Plot '$plot_name' not found. Available plots:")
        for k in sort(collect(keys(all_plots)))
            println("  - $k")
        end
    end
end

"""
Display all plots from observable analysis.
"""
function show_observable_analysis(all_plots::Dict, obs_name::Symbol)
    key = "$(obs_name)_analysis"
    if haskey(all_plots, key)
        plots = all_plots[key]
        println("Displaying analysis for $obs_name:")
        println("  1. Time series")
        display(plots[1])
        println("  2. Histogram")
        display(plots[2])
        println("  3. Cumulative mean")
        display(plots[3])
        println("  4. Blocking analysis")
        display(plots[4])
        println("  5. Autocorrelation")
        display(plots[5])
        println("  6. Integrated τ")
        display(plots[6])
    else
        println("Analysis for $obs_name not found.")
    end
end

"""
List all available plots in the report.
"""
function list_available_plots(all_plots::Dict)
    println("\nAvailable plots:")
    println("-"^50)
    
    # Group by category
    obs_plots = filter(k -> endswith(k, "_analysis"), keys(all_plots))
    vector_plots = filter(k -> k in ["Lxx_vs_q", "conductivity_vs_omega", "phase_correlation", "Ps_l"], keys(all_plots))
    joint_plots = filter(k -> occursin("vs_", k) && !endswith(k, "_analysis"), keys(all_plots))
    complex_plots = filter(k -> startswith(k, "complex_field"), keys(all_plots))
    
    println("Observable Analysis (use show_observable_analysis):")
    for k in sort(collect(obs_plots))
        println("  - $k")
    end
    
    println("\nVector Observables:")
    for k in sort(collect(vector_plots))
        println("  - $k")
    end
    
    println("\nJoint Distributions:")
    for k in sort(collect(joint_plots))
        println("  - $k")
    end
    
    println("\nComplex Field:")
    for k in sort(collect(complex_plots))
        println("  - $k")
    end
    println("-"^50)
end

# ============================================================================
# Example Usage
# ============================================================================

println("""
================================================================================
Simulation Statistics Report Generator Loaded Successfully!
================================================================================

Usage:
  # Generate full report with organized folder structure
  all_plots, stats, bootstrap, correlations, data = generate_report("path/to/sim"; save_plots=true)
  
  # Without saving (just display)
  all_plots, stats, bootstrap, correlations, data = generate_report("path/to/sim")
  
  # Customize which observables to analyze in detail
  generate_report("path/to/sim"; save_plots=true,
      observables_to_analyze=[:n, :Kx, :Delta_mean, :chi_spin])
  
  # Display specific plots
  show_plot(all_plots, "Lxx_vs_q")
  show_plot(all_plots, "complex_field_trajectory")
  show_observable_analysis(all_plots, :n)
  list_available_plots(all_plots)

Output Structure (when save_plots=true):
  statistics_report/
  ├── observables/
  │   ├── n/                    # Each observable gets its own folder
  │   │   ├── 00_dashboard.png
  │   │   ├── 01_timeseries.png
  │   │   ├── 02_histogram.png
  │   │   └── ...
  │   ├── Kx/
  │   └── ...
  ├── vector_observables/
  │   ├── Lxx_vs_q/
  │   ├── conductivity/
  │   └── ...
  ├── joint_distributions/
  ├── complex_field/
  ├── pdf_pages/
  └── statistics_summary.txt
================================================================================
""")
