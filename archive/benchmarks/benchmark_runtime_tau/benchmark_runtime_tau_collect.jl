using JLD2
using Dates
using JSON

include(joinpath(dirname(@__FILE__), "..", "..", "..", "sim_num_control.jl"))

function csv_escape(value)
    if value === nothing
        return ""
    end
    s = string(value)
    if occursin('"', s)
        s = replace(s, '"' => "\"\"")
    end
    if occursin(',', s) || occursin('\n', s) || occursin('\r', s)
        return '"' * s * '"'
    end
    return s
end

function stringify_list(value)
    if value isa AbstractVector
        return join(string.(value), ";")
    end
    return string(value)
end

function sanitize_json_value(x)
    if x isa AbstractFloat
        return isfinite(x) ? x : nothing
    elseif x isa Number
        return x
    elseif x isa DateTime
        return string(x)
    elseif x isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in x
            out[string(k)] = sanitize_json_value(v)
        end
        return out
    elseif x isa AbstractVector
        return [sanitize_json_value(v) for v in x]
    elseif x isa Tuple
        return [sanitize_json_value(v) for v in x]
    else
        return x
    end
end

function get_from_any_dict(d::AbstractDict, key::String, default=nothing)
    if haskey(d, key)
        return d[key]
    end
    skey = Symbol(key)
    if haskey(d, skey)
        return d[skey]
    end
    return default
end

function as_int_default(v, default::Int=0)
    if v isa Integer
        return Int(v)
    end
    if v isa Number
        return Int(round(v))
    end
    p = tryparse(Int, string(v))
    return p === nothing ? default : p
end

function as_float_default(v, default::Float64=NaN)
    if v isa Number
        return Float64(v)
    end
    p = tryparse(Float64, string(v))
    return p === nothing ? default : p
end

function as_datetime_or_nothing(v)
    if v isa DateTime
        return v
    end
    if v isa AbstractString
        try
            return DateTime(v)
        catch
            return nothing
        end
    end
    return nothing
end

function elapsed_seconds(start_dt::DateTime, end_dt::DateTime)
    return max(0.0, Dates.value(end_dt - start_dt) / 1000.0)
end

function extract_datetime_from_line(line::String)
    m = match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)", line)
    if m === nothing
        return nothing
    end
    return as_datetime_or_nothing(m.captures[1])
end

function estimate_active_time_from_logs(log_dir::String; end_time=nothing)
    if !isdir(log_dir)
        return NaN, "unavailable", ""
    end

    log_paths = [joinpath(log_dir, "simulation_log.txt"), joinpath(log_dir, "log.txt")]
    events = Tuple{DateTime, Symbol}[]
    found_any_log = false

    for p in log_paths
        if !isfile(p)
            continue
        end
        found_any_log = true
        for line in eachline(p)
            dt = extract_datetime_from_line(line)
            if dt === nothing
                continue
            end
            if occursin("-saving progress-", line)
                push!(events, (dt, :save))
            elseif occursin("start time:", line) || occursin("Resuming simulation at", line)
                push!(events, (dt, :start))
            end
        end
    end

    if !found_any_log
        return NaN, "unavailable", ""
    end
    if isempty(events)
        return NaN, "unavailable", "No parseable timing events found in logs"
    end

    sort!(events, by = x -> x[1])
    active_sec = 0.0
    segment_start = nothing
    last_save = nothing

    for (dt, kind) in events
        if kind == :start
            segment_start = dt
            last_save = nothing
            continue
        end

        if segment_start === nothing
            segment_start = dt
            last_save = dt
            continue
        end

        base = last_save === nothing ? segment_start : last_save
        if dt > base
            active_sec += elapsed_seconds(base, dt)
        end
        last_save = dt
    end

    if end_time !== nothing && last_save !== nothing && end_time isa DateTime && end_time > last_save
        active_sec += elapsed_seconds(last_save, end_time)
    end

    return active_sec, "estimated_from_logs", "Active time reconstructed from log save segments"
end

function get_seed_tracking(sim_dict::Dict{String, Any}, seed::Int)
    sims_any = get(sim_dict, "simulations", nothing)
    if !(sims_any isa AbstractDict)
        return nothing
    end
    sims = sims_any::AbstractDict

    if seed >= 0
        if haskey(sims, seed) && sims[seed] isa AbstractDict
            return sims[seed]
        end
        seed_str = string(seed)
        if haskey(sims, seed_str) && sims[seed_str] isa AbstractDict
            return sims[seed_str]
        end
        for (k, v) in sims
            if string(k) == seed_str && v isa AbstractDict
                return v
            end
        end
    end

    for (_, v) in sims
        if !(v isa AbstractDict)
            continue
        end
        status = lowercase(string(get_from_any_dict(v, "status", "")))
        if status == "running"
            return v
        end
    end

    for (_, v) in sims
        if v isa AbstractDict
            return v
        end
    end

    return nothing
end

function infer_raw_step_count(num_thermal::Int, num_measurements_done::Int, measuring_interval::Int)
    if measuring_interval <= 1
        return num_thermal + num_measurements_done + 1
    end
    return num_thermal + num_measurements_done * measuring_interval
end

function default_tau_patterns()
    return [
        "^n\$",
        "^Kx\$",
        "^Delta_MC_abs\$",
        "^U4\$",
        "^L(xx|yy)_q.",
        "^conductivity_regular_(xx|yy)_fixed\$",
        "^Drude_weight_fixed\$",
        "^conductivity_dc_(xx|yy)\$",
        "^drude_dc_(xx|yy)\$"
    ]
end

function normalize_tau_patterns(patterns::Vector{String})
    normalized = String[]
    for p in patterns
        if p == "^K0_action\$"
            push!(normalized, "^U4\$")
        elseif p == "^L.._q." || p == "^L.._q"
            push!(normalized, "^L(xx|yy)_q.")
        elseif p == "^conductivity_regular"
            push!(normalized, "^conductivity_regular_(xx|yy)_fixed\$")
        elseif p == "^Drude_weight"
            push!(normalized, "^Drude_weight_fixed\$")
            push!(normalized, "^drude_dc_(xx|yy)\$")
        elseif p == "^L.._w_fixed\$"
            push!(normalized, "^conductivity_regular_(xx|yy)_fixed\$")
            push!(normalized, "^Drude_weight_fixed\$")
        elseif p == "^L_D_dc_"
            push!(normalized, "^drude_dc_(xx|yy)\$")
        else
            push!(normalized, p)
        end
    end
    return unique(normalized)
end

canonical_tau_key(k::String) = (k == "K0_action" ? "U4" : k)

function make_pattern_regexes(patterns::Vector{String})
    regs = Regex[]
    for p in patterns
        push!(regs, Regex(p))
    end
    return regs
end

function key_matches(key::String, regs::Vector{Regex})
    for r in regs
        if occursin(r, key)
            return true
        end
    end
    return false
end

function max_numeric_value(x)
    if x isa Number
        xf = Float64(x)
        return isfinite(xf) ? xf : -Inf
    elseif x isa AbstractArray
        if isempty(x)
            return -Inf
        end
        m = -Inf
        for v in x
            m = max(m, max_numeric_value(v))
        end
        return m
    elseif x isa AbstractDict
        if isempty(x)
            return -Inf
        end
        m = -Inf
        for v in values(x)
            m = max(m, max_numeric_value(v))
        end
        return m
    else
        return -Inf
    end
end

function extract_worst_tau(means_dict::AbstractDict, patterns::Vector{String})
    regs = make_pattern_regexes(patterns)
    tau_by_key = Dict{String, Float64}()

    for (k, v) in means_dict
        if !(v isa AbstractDict)
            continue
        end
        k_str = canonical_tau_key(string(k))
        if !key_matches(k_str, regs)
            continue
        end
        corr_time = get_from_any_dict(v, "correlation_time", nothing)
        if isnothing(corr_time)
            continue
        end
        tau_val = max_numeric_value(corr_time)
        if isfinite(tau_val) && tau_val > 0
            tau_by_key[k_str] = tau_val
        end
    end

    if isempty(tau_by_key)
        return 1.0, tau_by_key, "No matched observables had correlation_time; defaulting tau_worst to 1.0"
    end

    tau_worst = maximum(values(tau_by_key))
    if !isfinite(tau_worst) || tau_worst <= 0
        return 1.0, tau_by_key, "Matched observables had invalid/non-positive correlation_time; defaulting tau_worst to 1.0"
    end

    return tau_worst, tau_by_key, nothing
end

function find_point_dirs(root_path::String)
    dirs = filter(name -> isdir(joinpath(root_path, name)), readdir(root_path))
    out = String[]
    for d in dirs
        if tryparse(Int, d) !== nothing
            push!(out, joinpath(root_path, d))
        end
    end
    sort!(out, by=x -> parse(Int, basename(x)))
    return out
end

function load_sim_manifest(point_dir::String)
    sim_path = joinpath(point_dir, "sim.jld2")
    if !isfile(sim_path)
        return Dict{String, Any}()
    end
    try
        data = JLD2.load(sim_path)
        d = Dict{String, Any}()
        for (k, v) in data
            d[string(k)] = v
        end
        return d
    catch e
        @warn "Failed to load sim manifest from $(sim_path): $(e)"
        return Dict{String, Any}()
    end
end

function load_existing_summary(point_dir::String)
    summary_path = joinpath(point_dir, "benchmark_summary.jld2")
    if !isfile(summary_path)
        return Dict{String, Any}()
    end
    try
        data = JLD2.load(summary_path)
        s = get(data, "summary", Dict{String, Any}())
        if s isa AbstractDict
            out = Dict{String, Any}()
            for (k, v) in s
                out[string(k)] = v
            end
            return out
        end
    catch e
        @warn "Failed to load existing summary from $(summary_path): $(e)"
    end
    return Dict{String, Any}()
end

function load_simulation_results(point_dir::String)
    return nothing, ""
end

function candidate_result_paths(point_dir::String, sim_dict::Dict{String, Any})
    cands = String[]

    # Preferred candidate when seed is known.
    if haskey(sim_dict, "seed")
        seed_val = as_int_default(sim_dict["seed"], -1)
        if seed_val >= 0
            push!(cands, joinpath(point_dir, "seed_$(seed_val)"))
        end
    end

    # Add all seed_* subdirectories.
    if isdir(point_dir)
        for name in readdir(point_dir)
            p = joinpath(point_dir, name)
            if isdir(p) && startswith(name, "seed_")
                push!(cands, p)
            end
        end
    end

    # Backward compatibility: sometimes results are directly in point_dir.
    push!(cands, point_dir)

    return unique(cands)
end

function has_result_files(path::String)
    return isfile(joinpath(path, "results.jld2")) || isfile(joinpath(path, "checkpoint.jld2"))
end

function load_simulation_results(point_dir::String, sim_dict::Dict{String, Any})
    for c in candidate_result_paths(point_dir, sim_dict)
        if !isdir(c) || !has_result_files(c)
            continue
        end

        try
            return load_results(c; to_load_checkpoint=false), c
        catch e
            @warn "load_results(to_load_checkpoint=false) failed in $(c): $(e)"
        end

        try
            return load_results(c; to_load_checkpoint=true), c
        catch e
            @warn "load_results(to_load_checkpoint=true) failed in $(c): $(e)"
        end
    end

    return nothing, ""
end

function compute_summary_for_point(point_dir::String; recompute_tau::Bool=true)
    sim_dict = load_sim_manifest(point_dir)
    existing = load_existing_summary(point_dir)

    job_folder = as_int_default(get(existing, "job_folder", basename(point_dir)), as_int_default(basename(point_dir), -1))
    point_index = as_int_default(get(existing, "point_index", get(sim_dict, "point_index", job_folder)), job_folder)
    seed_value = as_int_default(get(existing, "seed", get(sim_dict, "seed", -1)), -1)
    seed_tracking = get_seed_tracking(sim_dict, seed_value)

    live_step_count = -1
    live_num_measurements = -1
    live_status = ""
    live_start_time = nothing
    live_last_save_time = nothing
    live_end_time = nothing
    live_active_time_sec = NaN
    live_timing_last_accounted_time = nothing
    live_timing_segment_start = nothing
    live_timing_source = ""
    live_timed_mc_step_total_sec = NaN
    live_timed_mc_step_count = -1
    live_timed_measurement_total_sec = NaN
    live_timed_measurement_count = -1
    if seed_tracking !== nothing
        live_step_count = as_int_default(get_from_any_dict(seed_tracking, "step_count", -1), -1)
        live_num_measurements = as_int_default(get_from_any_dict(seed_tracking, "num_current_measurments", -1), -1)
        live_status = lowercase(string(get_from_any_dict(seed_tracking, "status", "")))
        live_start_time = as_datetime_or_nothing(get_from_any_dict(seed_tracking, "start_time", nothing))
        live_last_save_time = as_datetime_or_nothing(get_from_any_dict(seed_tracking, "last_save_time", nothing))
        live_end_time = as_datetime_or_nothing(get_from_any_dict(seed_tracking, "end_time", nothing))
        live_active_time_sec = as_float_default(get_from_any_dict(seed_tracking, "timing_active_sec", NaN), NaN)
        live_timing_last_accounted_time = as_datetime_or_nothing(get_from_any_dict(seed_tracking, "timing_last_accounted_time", nothing))
        live_timing_segment_start = as_datetime_or_nothing(get_from_any_dict(seed_tracking, "timing_segment_start", nothing))
        live_timing_source = string(get_from_any_dict(seed_tracking, "timing_source", ""))
        live_timed_mc_step_total_sec = as_float_default(get_from_any_dict(seed_tracking, "timing_mc_step_total_sec", NaN), NaN)
        live_timed_mc_step_count = as_int_default(get_from_any_dict(seed_tracking, "timing_mc_step_count", -1), -1)
        live_timed_measurement_total_sec = as_float_default(get_from_any_dict(seed_tracking, "timing_measurement_total_sec", NaN), NaN)
        live_timed_measurement_count = as_int_default(get_from_any_dict(seed_tracking, "timing_measurement_count", -1), -1)
    end
    is_running = (live_status == "running")

    tau_patterns_raw = if haskey(sim_dict, "tau_observable_patterns")
        [string(x) for x in sim_dict["tau_observable_patterns"]]
    else
        default_tau_patterns()
    end
    tau_patterns = unique(vcat(normalize_tau_patterns(tau_patterns_raw), default_tau_patterns()))

    data = nothing
    results_data_path = ""
    if recompute_tau
        data, results_data_path = load_simulation_results(point_dir, sim_dict)
    end
    means_ok = false
    tau_worst = as_float_default(get(existing, "tau_worst", 1.0), 1.0)
    tau_by_key = Dict{String, Float64}()
    tau_warning = ""

    raw_step_count = as_int_default(get(existing, "raw_step_count", -1), -1)
    mc_steps_executed = as_int_default(get(existing, "mc_steps_executed", max(0, raw_step_count + 1)), max(0, raw_step_count + 1))
    step_count_source = raw_step_count >= 0 ? "existing_summary" : "unavailable"
    step_count_warning = ""
    num_measurements_done = as_int_default(get(existing, "num_measurements_done", 0), 0)
    measuring_interval = as_int_default(get(existing, "measuring_interval", get(sim_dict, "measuring_interval", 1)), 1)
    if live_num_measurements >= 0
        num_measurements_done = live_num_measurements
    end
    force_reinfer_from_loaded_counts = false

    if data !== nothing
        raw_from_data = Int(data.current_state.step_count)
        if raw_from_data >= 0
            raw_step_count = raw_from_data
            mc_steps_executed = max(0, raw_step_count + 1)
            step_count_source = "loaded_results"
            step_count_warning = ""
        elseif live_step_count >= 0
            raw_step_count = live_step_count
            mc_steps_executed = max(0, raw_step_count + 1)
            step_count_source = "sim_tracking"
            step_count_warning = "Checkpoint step_count unavailable; using live step_count from sim.jld2 tracking"
        else
            force_reinfer_from_loaded_counts = true
        end
        num_measurements_done = Int(data.current_state.num_current_measurments)
        measuring_interval = Int(data.sim_data.measuring_interval)

        means_dict = get_means(data; std_scheme="correlation", no_avg_obs=true, fit_stiffness=false)
        means_ok = means_dict isa AbstractDict
        if means_ok
            tau_worst, tau_by_key, tau_warning_any = extract_worst_tau(means_dict, tau_patterns)
            tau_warning = isnothing(tau_warning_any) ? "" : tau_warning_any
        else
            tau_warning = "get_means did not return a dictionary; defaulting tau_worst to 1.0"
            tau_worst = 1.0
        end
    elseif recompute_tau
        tau_warning = "Could not load simulation data for collector-side tau recomputation"
    end

    if data === nothing && live_step_count >= 0
        raw_step_count = live_step_count
        mc_steps_executed = max(0, raw_step_count + 1)
        step_count_source = "sim_tracking"
        step_count_warning = ""
    end

    if force_reinfer_from_loaded_counts
        raw_step_count = -1
    end

    if raw_step_count < 0
        num_thermal_ref = as_int_default(get(existing, "num_thermal", get(sim_dict, "num_thermal", 0)), 0)
        raw_step_count = infer_raw_step_count(num_thermal_ref, num_measurements_done, measuring_interval)
        mc_steps_executed = max(0, raw_step_count + 1)
        if force_reinfer_from_loaded_counts
            step_count_source = "inferred_from_loaded_counts"
            step_count_warning = "raw_step_count inferred from loaded measurement count because checkpoint step_count was unavailable"
        else
            step_count_source = "inferred_from_counts"
            step_count_warning = "raw_step_count inferred from num_thermal, num_measurements_done, and measuring_interval because checkpoint step_count was unavailable"
        end
    end

    matched_tau_keys = sort(collect(keys(tau_by_key)))
    tau_worst_observable = ""
    if !isempty(tau_by_key)
        tau_worst_value = maximum(values(tau_by_key))
        worst_keys = sort([k for (k, v) in tau_by_key if v == tau_worst_value])
        tau_worst_observable = join(worst_keys, ";")
    end
    missing_patterns = String[]
    for p in tau_patterns
        found = false
        for k in matched_tau_keys
            if occursin(Regex(p), k)
                found = true
                break
            end
        end
        if !found
            push!(missing_patterns, p)
        end
    end

    if isempty(tau_warning) && tau_worst == 1.0 && isempty(tau_by_key)
        tau_warning = "tau_worst is 1.0 because no matched observables had correlation_time"
    end

    wall_time_sec = as_float_default(get(existing, "wall_time_sec", NaN), NaN)
    wall_time_source = isfinite(wall_time_sec) ? "existing_summary" : "unavailable"
    wall_time_warning = ""

    if live_start_time !== nothing
        now_dt = Dates.now()
        if live_status == "finished" && live_end_time !== nothing
            finished_wall = elapsed_seconds(live_start_time, live_end_time)
            if !isfinite(wall_time_sec) || finished_wall >= wall_time_sec
                wall_time_sec = finished_wall
                wall_time_source = "sim_tracking_end_time"
            end
        elseif live_status == "running"
            running_wall = elapsed_seconds(live_start_time, now_dt)
            wall_time_sec = running_wall
            wall_time_source = "sim_tracking_running_elapsed"
            if live_last_save_time !== nothing
                lag_sec = elapsed_seconds(live_last_save_time, now_dt)
                wall_time_warning = "Running wall_time_sec estimated from now-start_time; last_save_time lag=$(round(lag_sec, digits=1))s"
            else
                wall_time_warning = "Running wall_time_sec estimated from now-start_time; no last_save_time found yet"
            end
        elseif live_last_save_time !== nothing
            saved_wall = elapsed_seconds(live_start_time, live_last_save_time)
            if !isfinite(wall_time_sec) || saved_wall >= wall_time_sec
                wall_time_sec = saved_wall
                wall_time_source = "sim_tracking_last_save"
            end
        end
    end

    active_compute_time_sec = NaN
    active_time_source = "unavailable"
    active_time_warning = ""
    if isfinite(live_active_time_sec)
        active_compute_time_sec = live_active_time_sec
        active_time_source = isempty(live_timing_source) ? "sim_tracking_active" : string("sim_tracking_", live_timing_source)
        if live_status == "finished" && live_end_time !== nothing && live_timing_last_accounted_time !== nothing
            base_dt = live_timing_last_accounted_time
            if live_timing_segment_start !== nothing && live_timing_segment_start > base_dt
                base_dt = live_timing_segment_start
            end
            if live_end_time > base_dt
                active_compute_time_sec += elapsed_seconds(base_dt, live_end_time)
                active_time_source = string(active_time_source, "_plus_end")
            end
        elseif live_status == "running"
            active_time_warning = "Active time reflects accounted compute up to last persisted checkpoint"
        end
    else
        timing_log_dir = isempty(results_data_path) ? point_dir : results_data_path
        active_compute_time_sec, active_time_source, active_time_warning = estimate_active_time_from_logs(timing_log_dir; end_time=live_end_time)
    end

    mean_step_time_sec = as_float_default(get(existing, "mean_step_time_sec", NaN), NaN)
    mean_step_time_source = isfinite(mean_step_time_sec) ? "existing_summary" : "unavailable"
    if isfinite(active_compute_time_sec) && active_compute_time_sec > 0 && mc_steps_executed > 0
        mean_step_time_sec = active_compute_time_sec / mc_steps_executed
        mean_step_time_source = "active_compute_time"
    elseif !isfinite(mean_step_time_sec) && isfinite(wall_time_sec) && mc_steps_executed > 0
        mean_step_time_sec = wall_time_sec / mc_steps_executed
        mean_step_time_source = "wall_time_div_steps"
    end

    timed_mc_step_total_sec = as_float_default(get(existing, "timed_mc_step_total_sec", NaN), NaN)
    timed_mc_step_count = as_int_default(get(existing, "timed_mc_step_count", -1), -1)
    timed_measurement_total_sec = as_float_default(get(existing, "timed_measurement_total_sec", NaN), NaN)
    timed_measurement_count = as_int_default(get(existing, "timed_measurement_count", -1), -1)
    timed_counters_source = (isfinite(timed_mc_step_total_sec) || isfinite(timed_measurement_total_sec) || timed_mc_step_count >= 0 || timed_measurement_count >= 0) ? "existing_summary" : "unavailable"

    if data !== nothing
        timed_mc_step_total_sec = Float64(data.current_state.timing_mc_step_total_sec)
        timed_mc_step_count = Int(data.current_state.timing_mc_step_count)
        timed_measurement_total_sec = Float64(data.current_state.timing_measurement_total_sec)
        timed_measurement_count = Int(data.current_state.timing_measurement_count)
        timed_counters_source = "loaded_results_current_state"
    elseif isfinite(live_timed_mc_step_total_sec) || isfinite(live_timed_measurement_total_sec) || live_timed_mc_step_count >= 0 || live_timed_measurement_count >= 0
        timed_mc_step_total_sec = live_timed_mc_step_total_sec
        timed_mc_step_count = live_timed_mc_step_count
        timed_measurement_total_sec = live_timed_measurement_total_sec
        timed_measurement_count = live_timed_measurement_count
        timed_counters_source = "sim_tracking"
    end

    existing_mean_mc_step_time_sec = as_float_default(get(existing, "mean_mc_step_time_sec", NaN), NaN)
    mean_mc_step_time_sec = NaN
    mean_mc_step_time_source = "unavailable"
    if isfinite(timed_mc_step_total_sec) && timed_mc_step_count > 0
        mean_mc_step_time_sec = timed_mc_step_total_sec / timed_mc_step_count
        mean_mc_step_time_source = string(timed_counters_source, "_timed_steps")
    elseif isfinite(mean_step_time_sec)
        mean_mc_step_time_sec = mean_step_time_sec
        mean_mc_step_time_source = "mean_step_time_fallback"
    elseif isfinite(wall_time_sec) && mc_steps_executed > 0
        mean_mc_step_time_sec = wall_time_sec / mc_steps_executed
        mean_mc_step_time_source = "wall_time_div_steps_fallback"
    elseif isfinite(existing_mean_mc_step_time_sec)
        mean_mc_step_time_sec = existing_mean_mc_step_time_sec
        mean_mc_step_time_source = "existing_summary_fallback"
    end

    existing_mean_measurement_time_sec = as_float_default(get(existing, "mean_measurement_time_sec", NaN), NaN)
    mean_measurement_time_sec = NaN
    mean_measurement_time_source = "unavailable"
    if isfinite(timed_measurement_total_sec) && timed_measurement_count > 0
        mean_measurement_time_sec = timed_measurement_total_sec / timed_measurement_count
        mean_measurement_time_source = string(timed_counters_source, "_timed_measurements")
    elseif isfinite(active_compute_time_sec) && active_compute_time_sec > 0 && num_measurements_done > 0
        mean_measurement_time_sec = active_compute_time_sec / num_measurements_done
        mean_measurement_time_source = "active_compute_div_measurements"
    elseif isfinite(wall_time_sec) && num_measurements_done > 0
        mean_measurement_time_sec = wall_time_sec / num_measurements_done
        mean_measurement_time_source = "wall_time_div_measurements"
    elseif isfinite(existing_mean_measurement_time_sec)
        mean_measurement_time_sec = existing_mean_measurement_time_sec
        mean_measurement_time_source = "existing_summary_fallback"
    end

    active_fraction = (isfinite(active_compute_time_sec) && isfinite(wall_time_sec) && wall_time_sec > 0) ? (active_compute_time_sec / wall_time_sec) : NaN

    steps_per_uncorr = tau_worst * measuring_interval
    mean_step_for_uncorr_sec = isfinite(mean_mc_step_time_sec) ? mean_mc_step_time_sec : mean_step_time_sec
    time_per_uncorr_sec = isfinite(mean_step_for_uncorr_sec) ? mean_step_for_uncorr_sec * steps_per_uncorr : NaN
    indep_samples_per_hour = (isfinite(time_per_uncorr_sec) && time_per_uncorr_sec > 0) ? 3600.0 / time_per_uncorr_sec : NaN
    num_uncorr_est = Int(floor(num_measurements_done / max(1.0, tau_worst)))

    summary = Dict{String, Any}(
        "benchmark_kind" => "runtime_tau",
        "timestamp" => string(Dates.now()),
        "sim_path" => point_dir,
        "results_data_path" => isempty(results_data_path) ? point_dir : results_data_path,
        "job_folder" => job_folder,
        "point_index" => point_index,
        "Lx" => as_int_default(get(existing, "Lx", get(sim_dict, "Lx", -1)), -1),
        "Ly" => as_int_default(get(existing, "Ly", get(sim_dict, "Ly", -1)), -1),
        "L_SC" => as_int_default(get(existing, "L_SC", get(sim_dict, "L_SC", -1)), -1),
        "TClustersize_x" => as_int_default(get(existing, "TClustersize_x", get(sim_dict, "TClustersize_x", -1)), -1),
        "TClustersize_y" => as_int_default(get(existing, "TClustersize_y", get(sim_dict, "TClustersize_y", -1)), -1),
        "U" => as_float_default(get(existing, "U", get(sim_dict, "U", NaN)), NaN),
        "beta" => as_float_default(get(existing, "beta", get(sim_dict, "beta", NaN)), NaN),
        "mu" => as_float_default(get(existing, "mu", get(sim_dict, "mu", NaN)), NaN),
        "diag_method" => string(get(existing, "diag_method", get(sim_dict, "diag_method", "N/A"))),
        "pairing_symmetry" => string(get(existing, "pairing_symmetry", get(sim_dict, "pairing_symmetry", "N/A"))),
        "cores" => as_int_default(get(existing, "cores", get(sim_dict, "cores", 1)), 1),
        "seed" => as_int_default(get(existing, "seed", get(sim_dict, "seed", -1)), -1),
        "num_thermal" => as_int_default(get(existing, "num_thermal", get(sim_dict, "num_thermal", -1)), -1),
        "num_measure" => as_int_default(get(existing, "num_measure", get(sim_dict, "num_measure", -1)), -1),
        "measuring_interval" => measuring_interval,
        "is_running" => is_running,
        "wall_time_sec" => wall_time_sec,
        "wall_time_source" => wall_time_source,
        "wall_time_warning" => wall_time_warning,
        "active_compute_time_sec" => active_compute_time_sec,
        "active_time_source" => active_time_source,
        "active_time_warning" => active_time_warning,
        "active_fraction" => active_fraction,
        "raw_step_count" => raw_step_count,
        "mc_steps_executed" => mc_steps_executed,
        "mean_step_time_sec" => mean_step_time_sec,
        "mean_step_time_source" => mean_step_time_source,
        "mean_mc_step_time_sec" => mean_mc_step_time_sec,
        "mean_mc_step_time_source" => mean_mc_step_time_source,
        "mean_measurement_time_sec" => mean_measurement_time_sec,
        "mean_measurement_time_source" => mean_measurement_time_source,
        "timed_mc_step_total_sec" => timed_mc_step_total_sec,
        "timed_mc_step_count" => timed_mc_step_count,
        "timed_measurement_total_sec" => timed_measurement_total_sec,
        "timed_measurement_count" => timed_measurement_count,
        "num_measurements_done" => num_measurements_done,
        "tau_worst" => tau_worst,
        "tau_worst_observable" => tau_worst_observable,
        "steps_per_uncorr" => steps_per_uncorr,
        "time_per_uncorr_sec" => time_per_uncorr_sec,
        "indep_samples_per_hour" => indep_samples_per_hour,
        "num_uncorr_est" => num_uncorr_est,
        "matched_tau_keys" => matched_tau_keys,
        "missing_tau_patterns" => missing_patterns,
        "means_computed" => means_ok,
        "tau_warning" => tau_warning,
        "step_count_source" => step_count_source,
        "step_count_warning" => step_count_warning
    )

    return summary, tau_by_key, tau_patterns
end

function write_point_summary(point_dir::String, summary::Dict{String, Any}, tau_by_key::Dict{String, Float64}, tau_patterns::Vector{String})
    out_jld2 = joinpath(point_dir, "benchmark_summary.jld2")
    out_json = joinpath(point_dir, "benchmark_summary.json")

    JLD2.save(out_jld2,
        "summary", summary,
        "tau_by_key", tau_by_key,
        "tau_patterns", tau_patterns,
        "collector_recomputed_tau", true,
        "collector_updated_at", string(Dates.now()))

    open(out_json, "w") do io
        JSON.print(io, sanitize_json_value(Dict(
            "summary" => summary,
            "tau_by_key" => tau_by_key,
            "tau_patterns" => tau_patterns,
            "collector_recomputed_tau" => true,
            "collector_updated_at" => string(Dates.now())
        )), 2)
    end
end

function collect_summaries(root_path::String; recompute_tau::Bool=true, write_point_summaries::Bool=true)
    point_dirs = find_point_dirs(root_path)
    summaries = Dict{String, Any}[]

    for d in point_dirs
        summary, tau_by_key, tau_patterns = compute_summary_for_point(d; recompute_tau=recompute_tau)
        if write_point_summaries
            write_point_summary(d, summary, tau_by_key, tau_patterns)
        end
        push!(summaries, summary)
    end

    return summaries
end

function write_csv(path::String, summaries::Vector{Dict{String, Any}})
    columns = [
        "job_folder", "point_index", "Lx", "Ly", "L_SC", "TClustersize_x", "TClustersize_y",
        "U", "beta", "mu", "diag_method", "pairing_symmetry", "cores", "seed",
        "num_thermal", "num_measure", "measuring_interval",
        "is_running", "wall_time_sec", "wall_time_source", "wall_time_warning", "active_compute_time_sec", "active_time_source", "active_time_warning", "active_fraction", "raw_step_count", "mc_steps_executed", "mean_step_time_sec", "mean_step_time_source",
        "mean_mc_step_time_sec", "mean_mc_step_time_source", "mean_measurement_time_sec", "mean_measurement_time_source",
        "timed_mc_step_total_sec", "timed_mc_step_count", "timed_measurement_total_sec", "timed_measurement_count",
        "num_measurements_done", "tau_worst", "tau_worst_observable", "steps_per_uncorr", "time_per_uncorr_sec",
        "indep_samples_per_hour", "num_uncorr_est", "matched_tau_keys", "missing_tau_patterns",
        "means_computed", "tau_warning", "step_count_source", "step_count_warning", "timestamp", "sim_path", "results_data_path"
    ]

    open(path, "w") do io
        println(io, join(columns, ","))
        for s in summaries
            row = String[]
            for c in columns
                v = get(s, c, "")
                if c in ["matched_tau_keys", "missing_tau_patterns"]
                    v = stringify_list(v)
                end
                push!(row, csv_escape(v))
            end
            println(io, join(row, ","))
        end
    end
end

function write_json(path::String, summaries::Vector{Dict{String, Any}})
    out = Dict(
        "generated_at" => string(Dates.now()),
        "num_points" => length(summaries),
        "summaries" => summaries
    )
    open(path, "w") do io
        JSON.print(io, sanitize_json_value(out), 2)
    end
end

function write_sim_info(path::String, summaries::Vector{Dict{String, Any}})
    sim_info = Dict{String, Any}()
    sim_info["updated_at"] = string(Dates.now())
    for s in summaries
        key = string(get(s, "point_index", get(s, "job_folder", "unknown")))
        sim_info[key] = s
    end
    JLD2.save(path, sim_info)
end

function main()
    root_path = length(ARGS) >= 1 ? ARGS[1] : error("Usage: julia benchmark_runtime_tau_collect.jl <benchmark_root_path> [--no-recompute-tau] [--no-write-point-summaries]")
    if !isdir(root_path)
        error("Directory does not exist: $(root_path)")
    end

    recompute_tau = !("--no-recompute-tau" in ARGS)
    write_point_summaries = !("--no-write-point-summaries" in ARGS)

    summaries = collect_summaries(root_path;
        recompute_tau=recompute_tau,
        write_point_summaries=write_point_summaries)
    if isempty(summaries)
        println("No point directories found under $(root_path)")
        return
    end

    sort!(summaries, by = s -> begin
        pi = get(s, "point_index", 10^9)
        jf = tryparse(Int, string(get(s, "job_folder", 10^9)))
        return (pi isa Number ? pi : 10^9, jf === nothing ? 10^9 : jf)
    end)

    out_jld2 = joinpath(root_path, "benchmark_runtime_tau_aggregate.jld2")
    out_csv = joinpath(root_path, "benchmark_runtime_tau_aggregate.csv")
    out_json = joinpath(root_path, "benchmark_runtime_tau_aggregate.json")
    out_sim_info = joinpath(root_path, "benchmark_runtime_tau_sim_info.jld2")

    JLD2.save(out_jld2,
        "generated_at", string(Dates.now()),
        "num_points", length(summaries),
        "summaries", summaries)
    write_csv(out_csv, summaries)
    write_json(out_json, summaries)
    write_sim_info(out_sim_info, summaries)

    println("="^80)
    println("Aggregate complete")
    println("points: $(length(summaries))")
    println("recompute_tau: $(recompute_tau)")
    println("write_point_summaries: $(write_point_summaries)")
    println("saved: $(out_jld2)")
    println("saved: $(out_csv)")
    println("saved: $(out_json)")
    println("saved: $(out_sim_info)")
    println("="^80)
end

main()
