using Statistics
using Dates
using JLD2
using ArgParse
using JSON

include(joinpath(dirname(@__FILE__), "..", "..", "..", "sim_num_control.jl"))

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
    elseif x isa Dict
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

function json_safe_summary(summary::Dict{String, Any})
    out = Dict{String, Any}()
    for (k, v) in summary
        if v isa DateTime
            out[k] = string(v)
        elseif v isa AbstractVector
            out[k] = [x isa DateTime ? string(x) : x for x in v]
        else
            out[k] = v
        end
    end
    return out
end

function as_float_vec(v, key::String)
    if !(v isa AbstractVector)
        error("Expected vector for $(key), got $(typeof(v))")
    end
    out = Float64[]
    for x in v
        push!(out, Float64(x))
    end
    return out
end

function run_point(sim_path::String)
    sim_data_file = joinpath(sim_path, "sim.jld2")
    if !isfile(sim_data_file)
        error("sim.jld2 not found in $(sim_path)")
    end

    sim_data_dict = JLD2.load(sim_data_file)

    tau_patterns_raw = if haskey(sim_data_dict, "tau_observable_patterns")
        [string(x) for x in sim_data_dict["tau_observable_patterns"]]
    else
        default_tau_patterns()
    end
    tau_patterns = unique(vcat(normalize_tau_patterns(tau_patterns_raw), default_tau_patterns()))
    compute_tau_in_worker = haskey(sim_data_dict, "compute_tau_in_worker") ? Bool(sim_data_dict["compute_tau_in_worker"]) : false

    dump_hours = haskey(sim_data_dict, "dump_hours") ? Float64(sim_data_dict["dump_hours"]) : 1.0

    println("="^80)
    println("Running runtime+tau benchmark point")
    println("sim_path: $(sim_path)")
    println("Lx=$(sim_data_dict["Lx"]), Ly=$(sim_data_dict["Ly"]), U=$(sim_data_dict["U"]), beta=$(sim_data_dict["beta"]), cores=$(sim_data_dict["cores"])")
    println("="^80)

    wall_start = time()

    data = running_simulation_once(
        Int(sim_data_dict["Lx"]),
        Int(sim_data_dict["Ly"]),
        Int(sim_data_dict["L_SC"]),
        Float64(sim_data_dict["U"]),
        Float64(sim_data_dict["t"]),
        Float64(sim_data_dict["V"]),
        Int(sim_data_dict["seed"]),
        Float64(sim_data_dict["mu"]),
        Float64(sim_data_dict["beta"]),
        Float64(sim_data_dict["B"]),
        Int(sim_data_dict["num_measure"]),
        Int(sim_data_dict["num_thermal"]),
        Int(sim_data_dict["measuring_interval"]),
        as_float_vec(sim_data_dict["w_list"], "w_list"),
        string(sim_data_dict["guess"]),
        string(sim_data_dict["U_disorder"]),
        string(sim_data_dict["diag_method"]),
        string(sim_data_dict["pairing_symmetry"]),
        as_float_vec(sim_data_dict["w_conductivity_range"], "w_conductivity_range"),
        Float64(sim_data_dict["U_impurity"]);
        directory=sim_path,
        num_cores=Int(sim_data_dict["cores"]),
        savedata=haskey(sim_data_dict, "savedata") ? Bool(sim_data_dict["savedata"]) : true,
        saving_interval=haskey(sim_data_dict, "saving_interval") ? Int(sim_data_dict["saving_interval"]) : 500,
        dump_time=Dates.Second(round(Int, dump_hours * 3600)),
        saveplots=haskey(sim_data_dict, "saveplots") ? Bool(sim_data_dict["saveplots"]) : false,
        date_folder=haskey(sim_data_dict, "date_folder") ? Bool(sim_data_dict["date_folder"]) : false,
        numbered=haskey(sim_data_dict, "numbered") ? Bool(sim_data_dict["numbered"]) : false,
        seed_folder=haskey(sim_data_dict, "seed_folder") ? Bool(sim_data_dict["seed_folder"]) : true,
        U_puddle_params=Tuple(Float64.(sim_data_dict["U_puddle_params"])),
        TClustersize_x=Int(sim_data_dict["TClustersize_x"]),
        TClustersize_y=Int(sim_data_dict["TClustersize_y"]),
        U_smooth=(Bool(sim_data_dict["U_smooth"][1]), Float64(sim_data_dict["U_smooth"][2])),
        loop_random=haskey(sim_data_dict, "loop_random") ? Bool(sim_data_dict["loop_random"]) : true,
        gauge_choice=haskey(sim_data_dict, "gauge_choice") ? string(sim_data_dict["gauge_choice"]) : "LandauX",
        boundary_conditions=haskey(sim_data_dict, "boundary_condition") ? string(sim_data_dict["boundary_condition"]) : "PBC",
        delta_values_fixed=haskey(sim_data_dict, "delta_values_fixed") ? as_float_vec(sim_data_dict["delta_values_fixed"], "delta_values_fixed") : Float64[],
        delta_values_stiffness=haskey(sim_data_dict, "delta_values_stiffness") ? as_float_vec(sim_data_dict["delta_values_stiffness"], "delta_values_stiffness") : Float64[],
        dw_values=haskey(sim_data_dict, "dw_values") ? as_float_vec(sim_data_dict["dw_values"], "dw_values") : Float64[]
    )

    wall_time_sec = time() - wall_start

    tau_worst = 1.0
    tau_by_key = Dict{String, Float64}()
    tau_warning = nothing
    means_ok = false
    if compute_tau_in_worker
        means_dict = get_means(data; std_scheme="correlation", no_avg_obs=true, fit_stiffness=false)
        means_ok = means_dict isa AbstractDict
        if means_ok
            tau_worst, tau_by_key, tau_warning = extract_worst_tau(means_dict, tau_patterns)
        else
            tau_warning = "get_means did not return a dictionary; defaulting tau_worst to 1.0"
        end
    else
        tau_warning = "Worker skipped tau calculation (compute_tau_in_worker=false); collector will recompute tau"
    end

    raw_step_count = data.current_state.step_count
    mc_steps_executed = max(0, raw_step_count + 1)
    num_measurements_done = data.current_state.num_current_measurments
    measuring_interval = Int(data.sim_data.measuring_interval)
    timed_mc_step_total_sec = Float64(getfield(data.current_state, :timing_mc_step_total_sec))
    timed_mc_step_count = Int(getfield(data.current_state, :timing_mc_step_count))
    timed_measurement_total_sec = Float64(getfield(data.current_state, :timing_measurement_total_sec))
    timed_measurement_count = Int(getfield(data.current_state, :timing_measurement_count))

    mean_step_time_sec = mc_steps_executed > 0 ? wall_time_sec / mc_steps_executed : NaN
    mean_step_time_source = isfinite(mean_step_time_sec) ? "wall_time_div_steps" : "unavailable"
    mean_mc_step_time_sec = timed_mc_step_count > 0 ? (timed_mc_step_total_sec / timed_mc_step_count) : NaN
    mean_mc_step_time_source = isfinite(mean_mc_step_time_sec) ? "current_state_timed_steps" : "unavailable"
    if !isfinite(mean_mc_step_time_sec) && isfinite(mean_step_time_sec)
        mean_mc_step_time_sec = mean_step_time_sec
        mean_mc_step_time_source = "wall_time_div_steps_fallback"
    end

    mean_measurement_time_sec = timed_measurement_count > 0 ? (timed_measurement_total_sec / timed_measurement_count) : NaN
    mean_measurement_time_source = isfinite(mean_measurement_time_sec) ? "current_state_timed_measurements" : "unavailable"
    if !isfinite(mean_measurement_time_sec) && num_measurements_done > 0
        mean_measurement_time_sec = wall_time_sec / num_measurements_done
        mean_measurement_time_source = "wall_time_div_measurements_fallback"
    end

    steps_per_uncorr = tau_worst * measuring_interval
    time_per_uncorr_sec = isfinite(mean_mc_step_time_sec) ? mean_mc_step_time_sec * steps_per_uncorr : NaN
    num_uncorr_est = Int(floor(num_measurements_done / max(1.0, tau_worst)))
    indep_samples_per_hour = (isfinite(time_per_uncorr_sec) && time_per_uncorr_sec > 0) ? 3600.0 / time_per_uncorr_sec : NaN

    matched_tau_keys = sort(collect(keys(tau_by_key)))
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

    summary = Dict{String, Any}(
        "benchmark_kind" => "runtime_tau",
        "timestamp" => string(Dates.now()),
        "sim_path" => sim_path,
        "job_folder" => get(sim_data_dict, "job_folder", -1),
        "point_index" => get(sim_data_dict, "point_index", -1),
        "Lx" => Int(data.sim_data.Lx),
        "Ly" => Int(data.sim_data.Ly),
        "L_SC" => Int(data.sim_data.SC_L),
        "TClustersize_x" => Int(sim_data_dict["TClustersize_x"]),
        "TClustersize_y" => Int(sim_data_dict["TClustersize_y"]),
        "U" => Float64(data.sim_data.U),
        "beta" => Float64(data.sim_data.beta),
        "mu" => Float64(data.sim_data.mu),
        "diag_method" => string(data.sim_data.diag_method),
        "pairing_symmetry" => string(data.sim_data.pairing_symmetry),
        "cores" => Int(sim_data_dict["cores"]),
        "seed" => Int(sim_data_dict["seed"]),
        "num_thermal" => Int(data.sim_data.num_thermal),
        "num_measure" => Int(data.sim_data.num_measure),
        "measuring_interval" => measuring_interval,
        "wall_time_sec" => wall_time_sec,
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
        "steps_per_uncorr" => steps_per_uncorr,
        "time_per_uncorr_sec" => time_per_uncorr_sec,
        "indep_samples_per_hour" => indep_samples_per_hour,
        "num_uncorr_est" => num_uncorr_est,
        "matched_tau_keys" => matched_tau_keys,
        "missing_tau_patterns" => missing_patterns,
        "means_computed" => means_ok,
        "tau_warning" => isnothing(tau_warning) ? "" : tau_warning
    )

    summary_jld2 = joinpath(sim_path, "benchmark_summary.jld2")
    summary_json = joinpath(sim_path, "benchmark_summary.json")

    JLD2.save(summary_jld2,
        "summary", summary,
        "tau_by_key", tau_by_key,
        "tau_patterns", tau_patterns)

    open(summary_json, "w") do io
        JSON.print(io, Dict(
            "summary" => json_safe_summary(summary),
            "tau_by_key" => tau_by_key,
            "tau_patterns" => tau_patterns
        ), 2)
    end

    println("="^80)
    println("Benchmark point complete")
    println("wall_time_sec=$(round(wall_time_sec, digits=3))")
    println("mean_step_time_sec=$(round(mean_step_time_sec, digits=6))")
    println("mean_mc_step_time_sec=$(round(mean_mc_step_time_sec, digits=6)) [$(mean_mc_step_time_source)]")
    println("mean_measurement_time_sec=$(round(mean_measurement_time_sec, digits=6)) [$(mean_measurement_time_source)]")
    println("tau_worst=$(round(tau_worst, digits=3))")
    if !isnothing(tau_warning)
        println("tau_warning=$(tau_warning)")
    end
    println("steps_per_uncorr=$(round(steps_per_uncorr, digits=3))")
    println("time_per_uncorr_sec=$(round(time_per_uncorr_sec, digits=6))")
    println("saved: $(summary_jld2)")
    println("saved: $(summary_json)")
    println("="^80)
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--sim_path"
            help = "Simulation directory containing sim.jld2"
            required = true
    end

    parsed_args = parse_args(ARGS, s)
    sim_path = parsed_args["sim_path"]
    run_point(sim_path)
end

main()
