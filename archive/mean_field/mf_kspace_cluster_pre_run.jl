@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using Dates
using Printf
using JLD2
using ArgParse
using JSON

# Compat: `logrange` is Base on Julia >= 1.11 but missing on 1.10 (local). Define
# a fallback so default_config() works regardless of Julia version.
if !isdefined(Main, :logrange) && !isdefined(Base, :logrange)
    logrange(lo, hi, n) = exp.(range(log(lo), log(hi); length=n))
end

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

function as_int_vector(value, name::String)
    value isa AbstractVector || error("Expected vector for $(name), got $(typeof(value))")
    out = Int[]
    for v in value
        push!(out, as_int(v, name))
    end
    return out
end

function as_string_vector(value, name::String)
    value isa AbstractVector || error("Expected vector for $(name), got $(typeof(value))")
    return [string(v) for v in value]
end

function as_string_any_dict(value, name::String)
    if value isa AbstractDict
        d = Dict{String, Any}()
        for (k, v) in value
            d[string(k)] = v
        end
        return d
    end
    error("Expected dictionary for $(name), got $(typeof(value))")
end

function parse_u_smooth(value)
    if value isa Tuple && length(value) == 2
        return [as_bool(value[1], "U_smooth[1]"), as_float(value[2], "U_smooth[2]")]
    elseif value isa AbstractVector && length(value) == 2
        return [as_bool(value[1], "U_smooth[1]"), as_float(value[2], "U_smooth[2]")]
    end
    return [false, 0.8]
end

function default_config()
    return Dict{String, Any}(
        "output_base_dir" => V6_DATA_ROOT,
        "folder_name" => "mf_kspace_sweep__" * Dates.format(Dates.now(), "dd-mm-yyyy_HHMM"),
        "run_file" => joinpath(@__DIR__, "mf_kspace_cluster_run.jl"),
        "julia_bin" => JULIA_BIN,
        "skip_submit" => true,
        "slurm_account" => "all-account",
        "slurm_partition" => "allq, infiniq",
        "mem_per_cpu" => "8GB",
        "L_values" => [8, 10],
        "T_values" => [0.05, 0.10],
        "U_values" => [2.0, 4.0],
        "point_defaults" => Dict{String, Any}(
            "L_SC" => 1,
            "Ly_mode" => "square",
            "t" => 1.0,
            "V" => 0.0,
            "B" => 0.0,
            "target_n" => 0.875,
            "n_resolution" => 5e-3,
            "mu_lims" => [-4.0, 0.0],
            "mu_tuning_max_iter" => 30,
            "initial_guess" => "uniform",
            "U_disorder" => "uniform",
            "pairing_symmetry" => "s-wave",
            "diag_method" => "K0",
            "U_impurity" => 0.0,
            "U_puddle_params" => [0.0, 6.0, 0.7, 0.7],
            "U_smooth" => [false, 0.8],
            "TClustersize_x" => 4,
            "TClustersize_y" => 4,
            "w_list" => collect(-3.0:0.1:3.0),
            "w_conductivity_range" => collect(logrange(1e-3, 3.0, 12)),
            "dw_values_mult" => [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "delta_values_fixed_mult" => [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "delta_values_stiffness_mult" => Float64[],
            "d_min" => 1e-4,
            "dwave_max_iter" => 1000,
            "dwave_tol" => 1e-8,
            "dwave_mix" => 0.6,
            "saving_interval" => 1,
            "savedata" => true,
            "seed_folder" => true,
            "date_folder" => false,
            "numbered" => false,
            "loop_random" => false,
            "boundary_condition" => "PBC",
            "gauge_choice" => "LandauY",
        ),
        "seed_mode" => "base_plus_index",
        "seed_base" => 123450,
    )
end

function load_config(config_path::Union{Nothing, String})
    cfg = default_config()
    if isnothing(config_path)
        return cfg
    end

    config_path_abs = abspath(config_path)
    isfile(config_path_abs) || error("Config file not found: $(config_path_abs)")

    loaded = JSON.parsefile(config_path_abs)
    loaded_dict = as_string_any_dict(loaded, "config")

    # Copy top-level keys EXCEPT point_defaults; that one is deep-merged below so
    # a partial point_defaults in the JSON does not clobber the full defaults.
    for (k, v) in loaded_dict
        k == "point_defaults" && continue
        cfg[k] = v
    end

    if haskey(loaded_dict, "point_defaults")
        loaded_defaults = as_string_any_dict(loaded_dict["point_defaults"], "point_defaults")
        merged_defaults = as_string_any_dict(cfg["point_defaults"], "point_defaults")
        for (k, v) in loaded_defaults
            merged_defaults[k] = v
        end
        cfg["point_defaults"] = merged_defaults
    end

    return cfg
end

function make_seed(seed_mode::String, seed_base::Int, idx::Int)
    if seed_mode == "base_plus_index"
        return seed_base + idx - 1
    elseif seed_mode == "constant"
        return seed_base
    else
        error("Unsupported seed_mode: $(seed_mode)")
    end
end

function build_point_dict(point_defaults::Dict{String, Any}, idx::Int, L::Int, T::Float64, U::Float64, cluster_cores::Int, seed::Int)
    T > 0 || error("T must be positive, got $(T)")

    d = Dict{String, Any}()
    for (k, v) in point_defaults
        d[k] = v
    end

    ly_mode = string(get(d, "Ly_mode", "square"))
    Ly = ly_mode == "square" ? L : as_int(get(d, "Ly", L), "Ly")

    d["point_index"] = idx
    d["Lx"] = L
    d["Ly"] = Ly
    d["L"] = L
    d["U"] = U
    d["T"] = T
    d["beta"] = 1.0 / T
    d["seed"] = seed
    d["cores"] = cluster_cores
    d["status"] = "queued"
    d["created_at"] = string(Dates.now())

    d["U_puddle_params"] = haskey(d, "U_puddle_params") ? as_float_vector(d["U_puddle_params"], "U_puddle_params") : [0.0, 6.0, 0.7, 0.7]
    d["U_smooth"] = parse_u_smooth(get(d, "U_smooth", [false, 0.8]))
    d["w_list"] = haskey(d, "w_list") ? as_float_vector(d["w_list"], "w_list") : collect(-3.0:0.1:3.0)
    d["w_conductivity_range"] = haskey(d, "w_conductivity_range") ? as_float_vector(d["w_conductivity_range"], "w_conductivity_range") : collect(logrange(1e-3, 3.0, 12))

    mls = 4.0 / (L * Ly)
    d["MLS"] = mls
    dw_mult = haskey(d, "dw_values_mult") ? as_float_vector(d["dw_values_mult"], "dw_values_mult") : Float64[]
    delta_fixed_mult = haskey(d, "delta_values_fixed_mult") ? as_float_vector(d["delta_values_fixed_mult"], "delta_values_fixed_mult") : Float64[]
    delta_stiff_mult = haskey(d, "delta_values_stiffness_mult") ? as_float_vector(d["delta_values_stiffness_mult"], "delta_values_stiffness_mult") : Float64[]
    d["dw_values_mult"] = dw_mult
    d["delta_values_fixed_mult"] = delta_fixed_mult
    d["delta_values_stiffness_mult"] = delta_stiff_mult
    d["dw_values"] = dw_mult .* mls
    d["delta_values_fixed"] = delta_fixed_mult .* mls
    d["delta_values_stiffness"] = delta_stiff_mult .* mls

    d["mu_lims"] = haskey(d, "mu_lims") ? as_float_vector(d["mu_lims"], "mu_lims") : [-4.0, 0.0]

    # Keep sim_num aggregation compatible with existing assumptions.
    d["num_measure"] = 1
    d["num_thermal"] = 0
    d["measuring_interval"] = 1
    d["seeds"] = [seed]
    d["simulations"] = Dict{Any, Any}()

    return d
end

function write_sim_list(sweep_path::String, cluster_job_dir::String, points::Vector{Dict{String, Any}}, start_idx::Int, end_idx::Int)
    list_path = joinpath(cluster_job_dir, "sim_list_$(start_idx)_$(end_idx).txt")
    open(list_path, "w") do io
        println(io, "point_id\tpath\tLx\tLy\tU\tT\tbeta\tseed\tpairing")
        for p in points
            pid = p["point_index"]
            pairing = string(get(p, "pairing_symmetry", "s-wave"))
            println(io, "$(pid)\t$(joinpath(sweep_path, string(pid)))\t$(p["Lx"])\t$(p["Ly"])\t$(p["U"])\t$(p["T"])\t$(p["beta"])\t$(p["seed"])\t$(pairing)")
        end
    end
    return list_path
end

function make_slurm_script(path::String, cluster_job_dir::String, run_file::String, julia_bin::String, start_idx::Int, end_idx::Int, cpus_per_task::Int, mem_per_cpu::String, account::String, partition::String, exclude_nodes::String, julia_depot_path::String)
    slurm_file = joinpath(cluster_job_dir, "submit_array_$(start_idx)_$(end_idx).slurm")

    slurm = """
#!/bin/bash
#SBATCH --job-name=MFKSpaceSweep
#SBATCH --output=$(cluster_job_dir)/%A_%a.out
#SBATCH --error=$(cluster_job_dir)/%A_%a.error
#SBATCH --threads-per-core=1
#SBATCH -c $(cpus_per_task)
#SBATCH -n 1
#SBATCH --mem-per-cpu=$(mem_per_cpu)
#SBATCH --requeue
#SBATCH -A $(account)
#SBATCH -p $(partition)
#SBATCH --array=$(start_idx)-$(end_idx)
"""

    if !isempty(exclude_nodes)
        slurm *= "#SBATCH --exclude=$(exclude_nodes)\n"
    end

    slurm *= """
export SWEEP_ROOT="$(path)"
export ARRAY_START=$(start_idx)
export ARRAY_END=$(end_idx)
export JULIA_DEPOT_PATH="$(julia_depot_path)"
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

POINT_ID="\${SLURM_ARRAY_TASK_ID:-}"
if [ -z "\$POINT_ID" ] || [ "\$POINT_ID" = "4294967294" ]; then
    echo "WARNING: invalid SLURM_ARRAY_TASK_ID='\${SLURM_ARRAY_TASK_ID:-<unset>}'; running points \$ARRAY_START..\$ARRAY_END sequentially in this job"
    env | grep '^SLURM_ARRAY' | sort || true
    for POINT_ID in \$(seq \$ARRAY_START \$ARRAY_END); do
        echo "Running fallback point_id=\$POINT_ID (range \$ARRAY_START..\$ARRAY_END)"
        "$(julia_bin)" "$(run_file)" --sweep_root="\$SWEEP_ROOT" --point_id="\$POINT_ID"
    done
    exit 0
fi

"$(julia_bin)" "$(run_file)" --sweep_root="\$SWEEP_ROOT" --point_id="\$POINT_ID"
"""

    open(slurm_file, "w") do io
        write(io, slurm)
    end

    return slurm_file
end

function parse_sbatch_job_id(output::AbstractString)
    m = match(r"Submitted batch job\s+(\d+)", output)
    return isnothing(m) ? nothing : parse(Int, m.captures[1])
end

function find_existing_max_point_index(sweep_path::String)
    isdir(sweep_path) || return 0
    max_idx = 0
    for entry in readdir(sweep_path)
        full = joinpath(sweep_path, entry)
        isdir(full) || continue
        parsed = tryparse(Int, entry)
        if parsed !== nothing && parsed >= 1 && parsed > max_idx
            max_idx = parsed
        end
    end
    return max_idx
end

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config"
            help = "Path to JSON config file"
            arg_type = String
        "--skip_submit"
            help = "Prepare all point folders and slurm file but do not call sbatch"
            action = :store_true
    end
    return parse_args(ARGS, s)
end

function main()
    args = parse_cli()
    cfg = load_config(get(args, "config", nothing))

    output_base_dir = string(cfg["output_base_dir"])
    folder_name = string(cfg["folder_name"])
    path = joinpath(output_base_dir, folder_name)
    mkpath(path)
    cluster_job_dir = joinpath(path, "_cluster_jobs")
    mkpath(cluster_job_dir)

    existing_max = find_existing_max_point_index(path)
    start_idx = existing_max + 1

    run_file_raw = string(cfg["run_file"])
    run_file = isabspath(run_file_raw) ? run_file_raw : normpath(joinpath(@__DIR__, run_file_raw))
    isfile(run_file) || error("Run file not found: $(run_file)")

    julia_bin = string(get(cfg, "julia_bin", "julia"))
    account = string(get(cfg, "slurm_account", "all-account"))
    partition = string(get(cfg, "slurm_partition", "allq, infiniq"))
    mem_per_cpu = string(get(cfg, "mem_per_cpu", "8GB"))

    cpus_per_task = as_int(get(ENV, "CPUS_PER_TASK", get(cfg, "cpus_per_task", 1)), "CPUS_PER_TASK")
    exclude_nodes = string(get(ENV, "EXCLUDE_NODES", get(cfg, "exclude_nodes", "")))
    julia_depot_path = string(get(ENV, "JULIA_DEPOT_PATH", get(cfg, "julia_depot_path", "")))

    L_values = as_int_vector(cfg["L_values"], "L_values")
    T_values = as_float_vector(cfg["T_values"], "T_values")
    U_values = as_float_vector(cfg["U_values"], "U_values")
    point_defaults = as_string_any_dict(cfg["point_defaults"], "point_defaults")

    seed_mode = string(get(cfg, "seed_mode", "base_plus_index"))
    seed_base = as_int(get(cfg, "seed_base", 123450), "seed_base")

    total_points = length(L_values) * length(T_values) * length(U_values)
    end_idx = start_idx + total_points - 1
    println("="^80)
    println("HOMOGENEOUS K-SPACE MF CLUSTER PRE-RUN")
    println("="^80)
    println("Output path:    $(path)")
    println("Run file:       $(run_file)")
    println("Existing max:   $(existing_max)")
    println("New points:     $(total_points)  (indices $(start_idx)..$(end_idx))")
    println("Sweep sizes:    |L|=$(length(L_values)), |T|=$(length(T_values)), |U|=$(length(U_values))")
    println("CPUs/task:      $(cpus_per_task)")
    println("skip_submit:    $(as_bool(get(cfg, "skip_submit", true), "skip_submit") || get(args, "skip_submit", false))")
    println("="^80)

    points = Dict{String, Any}[]
    idx = start_idx
    for L in L_values
        for T in T_values
            for U in U_values
                point_dir = joinpath(path, string(idx))
                if isfile(joinpath(point_dir, "sim.jld2"))
                    error("Refusing to overwrite existing point: $(point_dir). This indicates an inconsistent sweep state (sim.jld2 already at index $(idx) but find_existing_max_point_index returned $(existing_max)).")
                end

                seed = make_seed(seed_mode, seed_base, idx)
                p = build_point_dict(point_defaults, idx, L, T, U, cpus_per_task, seed)
                p["job_dir"] = point_dir
                p["batch_start_idx"] = start_idx
                p["batch_end_idx"] = end_idx
                p["config_summary"] = Dict(
                    "L_values" => L_values,
                    "T_values" => T_values,
                    "U_values" => U_values,
                )

                mkpath(point_dir)
                JLD2.save(joinpath(point_dir, "sim.jld2"), p)
                push!(points, p)

                println(@sprintf("[%d/%d  global=%d] prepared point in %s  (L=%d, U=%.6g, T=%.6g, beta=%.6g, seed=%d)", idx - start_idx + 1, total_points, idx, point_dir, L, U, T, 1.0 / T, seed))
                idx += 1
            end
        end
    end

    list_path = write_sim_list(path, cluster_job_dir, points, start_idx, end_idx)
    slurm_file = make_slurm_script(path, cluster_job_dir, run_file, julia_bin, start_idx, end_idx, cpus_per_task, mem_per_cpu, account, partition, exclude_nodes, julia_depot_path)

    submit_disabled = as_bool(get(cfg, "skip_submit", true), "skip_submit") || get(args, "skip_submit", false)
    if submit_disabled
        println("Submission skipped. Created files:")
        println("  - $(list_path)")
        println("  - $(slurm_file)")
    else
        output = read(`sbatch $slurm_file`, String)
        println("sbatch output: $(strip(output))")
        job_id = parse_sbatch_job_id(output)
        if !isnothing(job_id)
            open(joinpath(cluster_job_dir, "array_job_id_$(start_idx)_$(end_idx).txt"), "w") do io
                println(io, job_id)
            end
        end
    end

    println("="^80)
    println("Prepared $(total_points) homogeneous-kspace MF points (indices $(start_idx)..$(end_idx)) at: $(path)")
    println("="^80)
end

main()
