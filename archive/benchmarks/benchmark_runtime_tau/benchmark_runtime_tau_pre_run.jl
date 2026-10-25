@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2
using Formatting
using Printf
using Dates
using JSON

function as_int(value, name::String)
    if value isa Integer
        return Int(value)
    end
    try
        return parse(Int, string(value))
    catch
        error("Invalid integer for $(name): $(value)")
    end
end

function as_float(value, name::String)
    if value isa Number
        return Float64(value)
    end
    try
        return parse(Float64, string(value))
    catch
        error("Invalid float for $(name): $(value)")
    end
end

function as_bool(value, name::String)
    if value isa Bool
        return value
    end
    s = lowercase(strip(string(value)))
    if s in ["1", "true", "yes", "y"]
        return true
    elseif s in ["0", "false", "no", "n"]
        return false
    end
    error("Invalid bool for $(name): $(value)")
end

function as_float_vector(value, name::String)
    if !(value isa AbstractVector)
        error("Expected vector for $(name), got $(typeof(value))")
    end
    out = Float64[]
    for v in value
        push!(out, as_float(v, name))
    end
    return out
end

function as_string_any_dict(value, name::String)
    if value isa AbstractDict
        d = Dict{String, Any}()
        for (k, v) in value
            d[string(k)] = v
        end
        return d
    end

    try
        d = Dict{String, Any}()
        for (k, v) in pairs(value)
            d[string(k)] = v
        end
        return d
    catch
        error("Expected dictionary-like value for $(name), got $(typeof(value))")
    end
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

function ensure_required_keys!(d::Dict{String, Any}, keys::Vector{String}, label::String)
    for key in keys
        if !haskey(d, key)
            error("Missing required key $(key) in $(label)")
        end
    end
end

function normalize_u_smooth(value)
    if value isa Tuple && length(value) == 2
        return (as_bool(value[1], "U_smooth[1]"), as_float(value[2], "U_smooth[2]"))
    elseif value isa AbstractVector && length(value) == 2
        return (as_bool(value[1], "U_smooth[1]"), as_float(value[2], "U_smooth[2]"))
    else
        error("U_smooth must contain exactly 2 values: [bool, sigma]")
    end
end

function get_tcluster_xy(d::Dict{String, Any})
    tx = if haskey(d, "TClustersize_x")
        as_int(d["TClustersize_x"], "TClustersize_x")
    elseif haskey(d, "TClustersize")
        as_int(d["TClustersize"], "TClustersize")
    else
        4
    end

    ty = if haskey(d, "TClustersize_y")
        as_int(d["TClustersize_y"], "TClustersize_y")
    elseif haskey(d, "TClustersize")
        as_int(d["TClustersize"], "TClustersize")
    else
        4
    end

    return tx, ty
end

function normalize_point(defaults::Dict{String, Any}, point::Dict{String, Any}, point_index::Int, tau_patterns::Vector{String}, cluster_cores::Int)
    merged = Dict{String, Any}()
    for (k, v) in defaults
        merged[k] = v
    end
    for (k, v) in point
        merged[k] = v
    end

    required_keys = [
        "Lx", "Ly", "L_SC", "U", "t", "V", "seed", "mu", "beta", "B",
        "num_measure", "num_thermal", "measuring_interval", "w_list", "initial_guess",
        "U_disorder", "diag_method", "pairing_symmetry", "w_conductivity_range",
        "U_impurity"
    ]
    ensure_required_keys!(merged, required_keys, "point $(point_index)")

    tx, ty = get_tcluster_xy(merged)

    u_smooth = haskey(merged, "U_smooth") ? normalize_u_smooth(merged["U_smooth"]) : (false, 0.8)

    delta_values_fixed = haskey(merged, "delta_values_fixed") ? as_float_vector(merged["delta_values_fixed"], "delta_values_fixed") : Float64[]
    delta_values_stiffness = haskey(merged, "delta_values_stiffness") ? as_float_vector(merged["delta_values_stiffness"], "delta_values_stiffness") : Float64[]
    dw_values = haskey(merged, "dw_values") ? as_float_vector(merged["dw_values"], "dw_values") : Float64[]

    point_tau_patterns = if haskey(merged, "tau_observable_patterns")
        [string(x) for x in merged["tau_observable_patterns"]]
    else
        tau_patterns
    end
    point_tau_patterns = normalize_tau_patterns(point_tau_patterns)

    return Dict{String, Any}(
        "point_index" => point_index,
        "Lx" => as_int(merged["Lx"], "Lx"),
        "Ly" => as_int(merged["Ly"], "Ly"),
        "L_SC" => as_int(merged["L_SC"], "L_SC"),
        "U" => as_float(merged["U"], "U"),
        "t" => as_float(merged["t"], "t"),
        "V" => as_float(merged["V"], "V"),
        "seed" => as_int(merged["seed"], "seed"),
        "mu" => as_float(merged["mu"], "mu"),
        "beta" => as_float(merged["beta"], "beta"),
        "B" => as_float(merged["B"], "B"),
        "num_measure" => as_int(merged["num_measure"], "num_measure"),
        "num_thermal" => as_int(merged["num_thermal"], "num_thermal"),
        "measuring_interval" => as_int(merged["measuring_interval"], "measuring_interval"),
        "w_list" => as_float_vector(merged["w_list"], "w_list"),
        "guess" => string(merged["initial_guess"]),
        "U_disorder" => string(merged["U_disorder"]),
        "diag_method" => string(merged["diag_method"]),
        "pairing_symmetry" => string(merged["pairing_symmetry"]),
        "w_conductivity_range" => as_float_vector(merged["w_conductivity_range"], "w_conductivity_range"),
        "U_impurity" => as_float(merged["U_impurity"], "U_impurity"),
        "cores" => cluster_cores,
        "savedata" => haskey(merged, "savedata") ? as_bool(merged["savedata"], "savedata") : true,
        "saving_interval" => haskey(merged, "saving_interval") ? as_int(merged["saving_interval"], "saving_interval") : 500,
        "saveplots" => haskey(merged, "saveplots") ? as_bool(merged["saveplots"], "saveplots") : false,
        "seed_folder" => haskey(merged, "seed_folder") ? as_bool(merged["seed_folder"], "seed_folder") : true,
        "date_folder" => haskey(merged, "date_folder") ? as_bool(merged["date_folder"], "date_folder") : false,
        "numbered" => haskey(merged, "numbered") ? as_bool(merged["numbered"], "numbered") : false,
        "dump_hours" => haskey(merged, "dump_hours") ? as_float(merged["dump_hours"], "dump_hours") : 1.0,
        "U_puddle_params" => haskey(merged, "U_puddle_params") ? as_float_vector(merged["U_puddle_params"], "U_puddle_params") : [0.0, 6.0, 0.7, 0.7],
        "TClustersize_x" => tx,
        "TClustersize_y" => ty,
        "U_smooth" => [u_smooth[1], u_smooth[2]],
        "loop_random" => haskey(merged, "loop_random") ? as_bool(merged["loop_random"], "loop_random") : true,
        "boundary_condition" => haskey(merged, "boundary_condition") ? string(merged["boundary_condition"]) : "PBC",
        "gauge_choice" => haskey(merged, "gauge_choice") ? string(merged["gauge_choice"]) : "LandauX",
        "delta_values_fixed" => delta_values_fixed,
        "delta_values_stiffness" => delta_values_stiffness,
        "dw_values" => dw_values,
        "tau_observable_patterns" => point_tau_patterns,
        "compute_tau_in_worker" => haskey(merged, "compute_tau_in_worker") ? as_bool(merged["compute_tau_in_worker"], "compute_tau_in_worker") : false,
        "benchmark_kind" => "runtime_tau",
        "benchmark_created_at" => string(Dates.now())
    )
end

function create_numbered_job_dir(base_path::String)
    i = 1
    c_path = joinpath(base_path, string(i))
    while isdir(c_path)
        i += 1
        c_path = joinpath(base_path, string(i))
    end
    mkpath(c_path)
    return c_path, i
end

function main()
    script_dir = dirname(@__FILE__)
    config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(script_dir, "benchmark_runtime_tau_config.json")
    config_path = abspath(config_path)
    if !isfile(config_path)
        error("Config file not found: $(config_path)")
    end

    config = JSON.parsefile(config_path)
    config_dir = dirname(config_path)

    defaults = haskey(config, "defaults") ? as_string_any_dict(config["defaults"], "defaults") : Dict{String, Any}()
    points = haskey(config, "points") ? config["points"] : Any[]
    if isempty(points)
        error("Config has no points")
    end

    out_dir_raw = haskey(config, "output_base_dir") ? string(config["output_base_dir"]) : joinpath(V6_DATA_ROOT, "benchmark_runtime_tau")
    out_dir = isabspath(out_dir_raw) ? out_dir_raw : normpath(joinpath(config_dir, out_dir_raw))
    folder_name = haskey(config, "folder_name") ? string(config["folder_name"]) : "runtime_tau_$(Dates.format(Dates.now(), "dd-mm-yyyy"))"
    path = joinpath(out_dir, folder_name)
    mkpath(path)

    run_file_raw = haskey(config, "run_file") ? string(config["run_file"]) : joinpath(script_dir, "benchmark_runtime_tau_run.jl")
    run_file = isabspath(run_file_raw) ? run_file_raw : normpath(joinpath(config_dir, run_file_raw))
    if !isfile(run_file)
        error("Run file not found: $(run_file)")
    end
    julia_bin = haskey(config, "julia_bin") ? string(config["julia_bin"]) : JULIA_BIN

    current_depot = get(ENV, "JULIA_DEPOT_PATH", "")
    exclude_nodes = get(ENV, "EXCLUDE_NODES", "")
    cluster_cores = as_int(get(ENV, "CPUS_PER_TASK", "1"), "CPUS_PER_TASK")
    account = haskey(config, "slurm_account") ? string(config["slurm_account"]) : "all-account"
    partition = haskey(config, "slurm_partition") ? string(config["slurm_partition"]) : "allq,infiniq"
    mem_per_cpu = haskey(config, "mem_per_cpu") ? string(config["mem_per_cpu"]) : "32GB"
    skip_submit = haskey(config, "skip_submit") ? as_bool(config["skip_submit"], "skip_submit") : false

    tau_patterns = if haskey(config, "tau_observable_patterns")
        [string(x) for x in config["tau_observable_patterns"]]
    else
        [
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
    tau_patterns = normalize_tau_patterns(tau_patterns)

    templateSLURM = FormatExpr("""#!/bin/bash

#SBATCH --job-name={1}
#SBATCH --output={2}/%j.out
#SBATCH --error={2}/%j.error
#SBATCH --threads-per-core=1
#SBATCH -c {3}
#SBATCH -n 1
#SBATCH --mem-per-cpu={4}
#SBATCH --requeue

#SBATCH -A {5}
#SBATCH -p {6}
#SBATCH --exclude={7}
#SBATCH --chdir={2}

export DIR={2}
export JULIA_DEPOT_PATH={8}
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

srun --chdir=\$DIR {9} {10} --sim_path=\$DIR

""")

    println("="^80)
    println("RUNTIME+TAU BENCHMARK SUBMISSION")
    println("="^80)
    println("Config path: $(config_path)")
    println("Output path: $(path)")
    println("Points: $(length(points))")
    println("Run file: $(run_file)")
    println("Using cluster cores from CPUS_PER_TASK: $(cluster_cores)")
    println("skip_submit: $(skip_submit)")
    println("="^80)

    submitted = 0
    for (idx, point_any) in enumerate(points)
        point = as_string_any_dict(point_any, "point $(idx)")
        sim_dict = normalize_point(defaults, point, idx, tau_patterns, cluster_cores)

        c_path, folder_num = create_numbered_job_dir(path)
        sim_dict["job_folder"] = folder_num
        sim_dict["job_dir"] = c_path
        sim_dict["config_path"] = abspath(config_path)

        sim_file = joinpath(c_path, "sim.jld2")
        JLD2.save(sim_file, sim_dict)

        slurm_file = joinpath(c_path, "slurm.txt")
        job_name = @sprintf("RTau_L%dx%d_U%.2f_b%.2f_Lc%dx%d_p%d", sim_dict["Lx"], sim_dict["Ly"], sim_dict["U"], sim_dict["beta"], sim_dict["TClustersize_x"], sim_dict["TClustersize_y"], idx)

        open(slurm_file, "w") do io
            printfmt(io, templateSLURM,
                job_name,
                c_path,
                sim_dict["cores"],
                mem_per_cpu,
                account,
                partition,
                exclude_nodes,
                current_depot,
                julia_bin,
                run_file
            )
        end

        println(@sprintf("[%d/%d] prepared point %d in %s (cores=%d)", idx, length(points), idx, c_path, sim_dict["cores"]))

        if !skip_submit
            run(`sbatch $slurm_file`)
            submitted += 1
        end
    end

    println("="^80)
    println("Prepared $(length(points)) benchmark jobs")
    if skip_submit
        println("Submission disabled by skip_submit=true")
    else
        println("Submitted $(submitted) jobs")
    end
    println("="^80)
end

main()
