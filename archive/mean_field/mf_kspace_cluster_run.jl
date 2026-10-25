using Dates
using Statistics
using JLD2
using ArgParse

include(joinpath(@__DIR__, "homogeneous_kspace_observables.jl"))

function as_int(value, name::String)
    if value isa Integer
        return Int(value)
    end
    parsed = tryparse(Int, string(value))
    parsed === nothing && error("Invalid integer for $(name): $(value)")
    return parsed
end

function as_float(value, name::String)
    if value isa Number
        return Float64(value)
    end
    parsed = tryparse(Float64, string(value))
    parsed === nothing && error("Invalid float for $(name): $(value)")
    return parsed
end

function as_bool(value, name::String)
    if value isa Bool
        return value
    elseif value isa Number
        x = Float64(value)
        if x == 1.0
            return true
        elseif x == 0.0
            return false
        end
        error("Invalid numeric bool for $(name): $(value)")
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
    value isa AbstractVector || error("Expected vector for $(name), got $(typeof(value))")
    out = Float64[]
    for v in value
        push!(out, as_float(v, name))
    end
    return out
end

function parse_u_smooth(value)
    if value isa Tuple && length(value) == 2
        return (as_bool(value[1], "U_smooth[1]"), as_float(value[2], "U_smooth[2]"))
    elseif value isa AbstractVector && length(value) == 2
        return (as_bool(value[1], "U_smooth[1]"), as_float(value[2], "U_smooth[2]"))
    end
    return (false, 0.8)
end

function first_or_nan(vec)
    return isempty(vec) ? NaN : vec[1]
end

function resolve_point_path(sweep_root::String, point_id::Int)
    if point_id < 1
        error("point_id must be >= 1")
    end
    point_dir = joinpath(sweep_root, string(point_id))
    if !isdir(point_dir)
        error("Point directory does not exist: $(point_dir)")
    end
    return point_dir
end

function get_point_file(point_dir::String)
    point_file = joinpath(point_dir, "sim.jld2")
    isfile(point_file) || error("sim.jld2 not found at point path: $(point_file)")
    return point_file
end

function load_point_dict(point_file::String)
    d = JLD2.load(point_file)
    return Dict{String, Any}(string(k) => v for (k, v) in pairs(d))
end

function save_point_dict(point_file::String, point_dict::Dict{String, Any})
    JLD2.save(point_file, point_dict)
end

function get_Lx_Ly(point::Dict{String, Any})
    if haskey(point, "Lx") && haskey(point, "Ly")
        return as_int(point["Lx"], "Lx"), as_int(point["Ly"], "Ly")
    elseif haskey(point, "L")
        L = as_int(point["L"], "L")
        return L, L
    else
        error("Point is missing Lx/Ly (or legacy L)")
    end
end

function build_sim_data(point::Dict{String, Any}, seed::Int, mu::Float64, beta::Float64, out_dir::String)
    Lx, Ly = get_Lx_Ly(point)

    t = as_float(get(point, "t", 1.0), "t")
    V = as_float(get(point, "V", 0.0), "V")
    U = as_float(get(point, "U", 4.0), "U")
    L_SC = as_int(get(point, "L_SC", 1), "L_SC")

    u_puddle_params = haskey(point, "U_puddle_params") ? as_float_vector(point["U_puddle_params"], "U_puddle_params") : [0.0, 6.0, 0.7, 0.7]
    u_smooth = parse_u_smooth(get(point, "U_smooth", [false, 0.8]))

    w_list = haskey(point, "w_list") ? as_float_vector(point["w_list"], "w_list") : collect(-3.0:0.1:3.0)
    w_cond = haskey(point, "w_conductivity_range") ? as_float_vector(point["w_conductivity_range"], "w_conductivity_range") : collect(0.0:0.1:4.0)
    dw_values = haskey(point, "dw_values") ? as_float_vector(point["dw_values"], "dw_values") : Float64[]
    delta_values_fixed = haskey(point, "delta_values_fixed") ? as_float_vector(point["delta_values_fixed"], "delta_values_fixed") : Float64[]
    delta_values_stiffness = haskey(point, "delta_values_stiffness") ? as_float_vector(point["delta_values_stiffness"], "delta_values_stiffness") : Float64[]

    return SimData(
        Lx=Lx,
        Ly=Ly,
        SC_L=L_SC,
        U=U,
        t=t,
        V=V,
        seed=seed,
        mu=mu,
        beta=beta,
        B=as_float(get(point, "B", 0.0), "B"),
        num_measure=1,
        num_thermal=0,
        measuring_interval=1,
        saving_interval=as_int(get(point, "saving_interval", 1), "saving_interval"),
        w_list=w_list,
        guess=string(get(point, "initial_guess", get(point, "guess", "uniform"))),
        U_disorder=string(get(point, "U_disorder", "uniform")),
        diag_method=string(get(point, "diag_method", "K0")),
        pairing_symmetry=string(get(point, "pairing_symmetry", "s-wave")),
        directory=out_dir,
        num_cores=as_int(get(point, "cores", 1), "cores"),
        num_threads=Threads.nthreads(),
        start_time=Dates.now(),
        w_conductivity_range=w_cond,
        U_disorder_params=u_puddle_params,
        U_impurity=as_float(get(point, "U_impurity", 0.0), "U_impurity"),
        TCluster_size_x=as_int(get(point, "TClustersize_x", get(point, "TCluster_size_x", 4)), "TClustersize_x"),
        TCluster_size_y=as_int(get(point, "TClustersize_y", get(point, "TCluster_size_y", 4)), "TClustersize_y"),
        smooth_U=u_smooth,
        loop_random=as_bool(get(point, "loop_random", false), "loop_random"),
        boundary_conditions=string(get(point, "boundary_condition", get(point, "boundary_conditions", "PBC"))),
        gauge_choice=string(get(point, "gauge_choice", "LandauY")),
        delta_values_fixed=delta_values_fixed,
        delta_values_stiffness=delta_values_stiffness,
        dw_values=dw_values,
    )
end

function run_one_mu(point::Dict{String, Any}, out_dir::String, seed::Int, mu::Float64, beta::Float64)
    sim_data = build_sim_data(point, seed, mu, beta, out_dir)

    d_min = as_float(get(point, "d_min", 1e-4), "d_min")
    dwave_max_iter = as_int(get(point, "dwave_max_iter", 1000), "dwave_max_iter")
    dwave_tol = as_float(get(point, "dwave_tol", 1e-8), "dwave_tol")
    dwave_mix = as_float(get(point, "dwave_mix", 0.6), "dwave_mix")

    data, info = run_homogeneous_kspace_observables(
        sim_data;
        measurement_ix=1,
        compute_k0=true,
        d_min=d_min,
        dwave_max_iter=dwave_max_iter,
        dwave_tol=dwave_tol,
        dwave_mix=dwave_mix,
    )

    n_val = first_or_nan(data.measurement_data.n)
    return data, info, n_val
end

function tune_mu_target_n(point::Dict{String, Any}, out_dir::String, seed::Int, beta::Float64)
    target_n = as_float(get(point, "target_n", NaN), "target_n")
    n_tol = as_float(get(point, "n_resolution", 5e-3), "n_resolution")
    max_iter = as_int(get(point, "mu_tuning_max_iter", 30), "mu_tuning_max_iter")

    mu_lims = haskey(point, "mu_lims") ? as_float_vector(point["mu_lims"], "mu_lims") : [-4.0, 0.0]
    length(mu_lims) == 2 || error("mu_lims must contain exactly two values")

    mu_low = mu_lims[1]
    mu_high = mu_lims[2]
    if mu_low == mu_high
        error("mu_lims endpoints must be different")
    end

    data_low, info_low, n_low = run_one_mu(point, out_dir, seed, mu_low, beta)
    data_high, info_high, n_high = run_one_mu(point, out_dir, seed, mu_high, beta)

    best_data, best_info, best_mu, best_n = abs(n_low - target_n) <= abs(n_high - target_n) ?
        (data_low, info_low, mu_low, n_low) : (data_high, info_high, mu_high, n_high)

    bracketed = (n_low - target_n) * (n_high - target_n) <= 0
    if !bracketed
        return best_data, best_info, best_mu, best_n, false, 0, n_tol, target_n, mu_lims
    end

    converged = abs(best_n - target_n) <= n_tol
    iter_count = 0

    for iter in 1:max_iter
        iter_count = iter
        converged && break

        mu_try = (mu_low + mu_high) / 2
        dn = n_high - n_low
        if isfinite(dn) && abs(dn) > 1e-12
            mu_secant = mu_low + (target_n - n_low) * (mu_high - mu_low) / dn
            lo = min(mu_low, mu_high)
            hi = max(mu_low, mu_high)
            if isfinite(mu_secant) && lo <= mu_secant <= hi
                mu_try = mu_secant
            end
        end

        data_try, info_try, n_try = run_one_mu(point, out_dir, seed, mu_try, beta)

        if abs(n_try - target_n) < abs(best_n - target_n)
            best_data, best_info, best_mu, best_n = data_try, info_try, mu_try, n_try
        end

        if abs(n_try - target_n) <= n_tol
            best_data, best_info, best_mu, best_n = data_try, info_try, mu_try, n_try
            converged = true
            break
        end

        if (n_low - target_n) * (n_try - target_n) <= 0
            mu_high = mu_try
            n_high = n_try
        else
            mu_low = mu_try
            n_low = n_try
        end
    end

    return best_data, best_info, best_mu, best_n, converged, iter_count, n_tol, target_n, mu_lims
end

function ensure_seed_measurement_ready!(data::LatticeData)
    data.current_state.num_current_measurments = max(data.current_state.num_current_measurments, 1)
    data.current_state.saving_measurement_ix = max(data.current_state.saving_measurement_ix, 1)
    data.current_state.last_dump_time = Dates.now()
end

function update_point_status!(point_dict::Dict{String, Any}, seed::Int, seed_dir::String, point_info_path::String, status::String)
    if !haskey(point_dict, "simulations") || !(point_dict["simulations"] isa AbstractDict)
        point_dict["simulations"] = Dict{Any, Any}()
    end
    if !haskey(point_dict["simulations"], seed)
        point_dict["simulations"][seed] = Dict{String, Any}()
    end

    seed_dict = point_dict["simulations"][seed]
    seed_dict["seed"] = seed
    seed_dict["status"] = status
    seed_dict["dir_path"] = seed_dir
    seed_dict["checkpoint_path"] = joinpath(seed_dir, "checkpoint.jld2")
    seed_dict["info_path"] = point_info_path
    seed_dict["start_time"] = get(seed_dict, "start_time", Dates.now())
    seed_dict["end_time"] = Dates.now()

    point_dict["simulations"][seed] = seed_dict
    point_dict["status"] = status
    point_dict["all_seeds_completed"] = status == "finished"
    point_dict["last_completed_seed"] = status == "finished" ? seed : get(point_dict, "last_completed_seed", nothing)
    point_dict["completed_seeds"] = status == "finished" ? [seed] : Int[]
end

function run_point(point_dir::String; force_rerun::Bool=false)
    point_file = get_point_file(point_dir)
    point_dict = load_point_dict(point_file)

    existing_status = string(get(point_dict, "status", "queued"))
    if !force_rerun && existing_status == "finished"
        println("Skipping finished point: $(point_dir)")
        return
    end

    seed = as_int(get(point_dict, "seed", 12345), "seed")
    seed_dir = joinpath(point_dir, "seed_$(seed)")
    mkpath(seed_dir)

    start_time = Dates.now()

    if !haskey(point_dict, "simulations") || !(point_dict["simulations"] isa AbstractDict)
        point_dict["simulations"] = Dict{Any, Any}()
    end
    point_dict["status"] = "running"
    point_dict["start_time"] = get(point_dict, "start_time", start_time)
    point_dict["current_seed"] = seed
    point_dict["simulations"][seed] = Dict{String, Any}(
        "seed" => seed,
        "status" => "running",
        "start_time" => start_time,
        "dir_path" => seed_dir,
        "checkpoint_path" => joinpath(seed_dir, "checkpoint.jld2"),
        "info_path" => point_file,
    )
    save_point_dict(point_file, point_dict)

    try
        T = as_float(get(point_dict, "T", NaN), "T")
        beta = if isfinite(T)
            T > 0 || error("T must be positive, got $(T)")
            1.0 / T
        elseif haskey(point_dict, "beta")
            as_float(point_dict["beta"], "beta")
        else
            error("Point must define T or beta")
        end

        data = nothing
        info = nothing
        best_mu = NaN
        best_n = NaN
        converged = true
        tune_iters = 0
        target_n = NaN
        n_tol = NaN
        mu_lims = Float64[]

        use_target_n = haskey(point_dict, "target_n") && isfinite(as_float(point_dict["target_n"], "target_n"))
        if use_target_n
            data, info, best_mu, best_n, converged, tune_iters, n_tol, target_n, mu_lims = tune_mu_target_n(point_dict, seed_dir, seed, beta)
        else
            mu = haskey(point_dict, "mu") ? as_float(point_dict["mu"], "mu") : error("Point missing both target_n and fixed mu")
            data, info, best_n = run_one_mu(point_dict, seed_dir, seed, mu, beta)
            best_mu = mu
        end

        ensure_seed_measurement_ready!(data)
        dump_measurements(data, seed_dir)

        kx_val = first_or_nan(data.measurement_data.Kx)

        seed_meta = Dict{String, Any}(
            "seed" => seed,
            "status" => "finished",
            "start_time" => start_time,
            "end_time" => Dates.now(),
            "dir_path" => seed_dir,
            "checkpoint_path" => joinpath(seed_dir, "checkpoint.jld2"),
            "info_path" => point_file,
            "num_current_measurments" => data.current_state.num_current_measurments,
            "mu" => best_mu,
            "n" => best_n,
            "Kx" => kx_val,
            "beta" => beta,
            "T" => 1.0 / beta,
            "convergence" => Dict{String, Any}(
                "converged" => converged,
                "target_n" => target_n,
                "best_n" => best_n,
                "best_mu" => best_mu,
                "n_tol" => n_tol,
                "iterations" => tune_iters,
                "mu_lims" => mu_lims,
            ),
            "delta_summary" => hasproperty(info, :delta_summary) ? info.delta_summary : nothing,
        )

        point_dict["beta"] = beta
        point_dict["T"] = 1.0 / beta
        point_dict["mu"] = best_mu
        point_dict["n"] = best_n
        point_dict["num_current_measurments"] = data.current_state.num_current_measurments
        point_dict["convergence"] = seed_meta["convergence"]
        point_dict["end_time"] = Dates.now()
        point_dict["elapsed_sec"] = max(0.0, Dates.value(point_dict["end_time"] - start_time) / 1000.0)

        update_point_status!(point_dict, seed, seed_dir, point_file, "finished")
        point_dict["simulations"][seed] = seed_meta
        save_point_dict(point_file, point_dict)

        println("Finished point=$(basename(point_dir)) seed=$(seed) L=$(data.sim_data.Lx)x$(data.sim_data.Ly) U=$(data.sim_data.U) T=$(1.0/beta) beta=$(beta) mu=$(best_mu) n=$(best_n) converged=$(converged)")
    catch err
        update_point_status!(point_dict, seed, seed_dir, point_file, "failed")
        point_dict["error"] = sprint(showerror, err)
        point_dict["end_time"] = Dates.now()
        save_point_dict(point_file, point_dict)
        rethrow(err)
    end
end

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--sim_path"
            help = "Absolute path to one point directory that contains sim.jld2"
            arg_type = String
        "--sweep_root"
            help = "Root sweep directory containing numbered point folders"
            arg_type = String
        "--point_id"
            help = "1-based point index (folder number under sweep_root)"
            arg_type = String
        "--force_rerun"
            help = "Run even if point status is already finished"
            action = :store_true
    end
    return parse_args(ARGS, s)
end

function parse_point_id(args)
    candidates = String[]

    if haskey(args, "point_id") && !isnothing(args["point_id"])
        push!(candidates, string(args["point_id"]))
    end
    push!(candidates, get(ENV, "SLURM_ARRAY_TASK_ID", ""))

    for c in candidates
        s = strip(c)
        if isempty(s) || s == "4294967294" || lowercase(s) == "no_val"
            continue
        end
        parsed = tryparse(Int, s)
        if parsed !== nothing && parsed >= 1
            return parsed
        end
    end

    return nothing
end

function main()
    args = parse_cli()

    point_dir = nothing
    if haskey(args, "sim_path") && !isnothing(args["sim_path"])
        point_dir = args["sim_path"]
    elseif haskey(args, "sweep_root") && !isnothing(args["sweep_root"])
        point_id = parse_point_id(args)
        if point_id === nothing
            error("Missing valid point_id. Provide --point_id <id> or SLURM_ARRAY_TASK_ID in environment.")
        end
        point_dir = resolve_point_path(args["sweep_root"], point_id)
    else
        error("Provide either --sim_path <point_dir> OR (--sweep_root <root> --point_id <id>)")
    end

    force_rerun = get(args, "force_rerun", false)
    println("Running homogeneous-kspace MF point at $(point_dir) (force_rerun=$(force_rerun))")
    run_point(point_dir; force_rerun=force_rerun)
end

# Only auto-run when invoked as a script; allow `include`-ing for batch drivers.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
