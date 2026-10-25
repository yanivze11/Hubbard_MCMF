#=
Discovery & Worker Submission Script
=====================================
Runs inside a SLURM job. Discovers all simulations, applies filters,
writes sim_list.txt, and submits the array worker job.

Called by the discovery SLURM job submitted from submit_batch_reports.jl.

Usage (called by SLURM, not directly):
    julia discover_and_submit_workers.jl <sim_path> <log_dir> <cpus_per_task> <depot_path> <exclude_nodes> [options...]

Options (after the 5 positional args):
    <integer>              Number of workers (default: 10)
    --workers=N            Number of workers
    --no-skip              Process all simulations even if reports exist
    --output-subdir=X      Subdirectory name for results (default: results)
    --tau-mult=X           Tau multiplier (default: 10.0)
    key=value              Parameter filters (e.g., Lx=10 beta=30.0)
=#

include(joinpath(@__DIR__, "batch_generate_reports.jl"))  # for find_simulations + JLD2
using Dates

# --- PARSE POSITIONAL ARGUMENTS ---
if length(ARGS) < 5
    error("Usage: julia discover_and_submit_workers.jl <sim_path> <log_dir> <cpus_per_task> <depot_path> <exclude_nodes> [options...]")
end

sim_path = ARGS[1]
log_dir = ARGS[2]
cpus_per_task = parse(Int, ARGS[3])
depot_path = ARGS[4]
exclude_nodes = ARGS[5]

# --- PARSE OPTIONAL ARGUMENTS ---
function parse_discovery_kwargs(args)
    num_workers = 10
    filters = Dict{String, Any}()
    skip_existing = true
    require_enhanced_report = false
    output_subdir = "results"
    tau_mult = 10.0

    for arg in args
        if arg == "--no-skip"
            skip_existing = false
        elseif arg == "--require-enhanced-report"
            require_enhanced_report = true
        elseif startswith(arg, "--output-subdir=")
            output_subdir = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--tau-mult=")
            tau_mult = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--workers=")
            num_workers = parse(Int, split(arg, "=", limit=2)[2])
        elseif occursin("=", arg)
            kv = split(arg, "=", limit=2)
            if length(kv) == 2
                val = tryparse(Float64, kv[2])
                filters[kv[1]] = val === nothing ? kv[2] : val
            end
        else
            nw = tryparse(Int, arg)
            if nw !== nothing
                num_workers = nw
            else
                println("Warning: Ignoring unknown argument '$arg'")
            end
        end
    end
    return num_workers, filters, skip_existing, require_enhanced_report, output_subdir, tau_mult
end

num_workers, filters, skip_existing, require_enhanced_report, output_subdir, tau_mult = parse_discovery_kwargs(ARGS[6:end])

function parse_sbatch_job_id(output::AbstractString)
    m = match(r"Submitted batch job\s+(\d+)", output)
    return isnothing(m) ? nothing : parse(Int, m.captures[1])
end

println("="^80)
println("SIMULATION DISCOVERY & WORKER SUBMISSION")
println("="^80)
println("Simulation path: $sim_path")
println("Log directory:   $log_dir")
println("Num workers:     $num_workers")
println("Filters:         $filters")
println("Skip existing:   $skip_existing")
println("Require enhanced marker: $require_enhanced_report")
println("Output subdir:   $output_subdir")
println("Tau multiplier:  $tau_mult")
println()

# --- DISCOVER SIMULATIONS ---
println("Discovering simulations...")
simulations = find_simulations(sim_path)
println("Found $(length(simulations)) simulation(s)")

if isempty(simulations)
    println("No simulations found in: $sim_path")
    exit(0)
end

# Apply parameter filters
if !isempty(filters)
    println("Applying filters: $filters")
    simulations = filter(sim -> begin
        sim_file = joinpath(sim, "sim.jld2")
        if !isfile(sim_file)
            return false
        end
        try
            sim_info = JLD2.load(sim_file)
            all(k -> get(sim_info, k, nothing) == filters[k], keys(filters))
        catch e
            println("  Warning: Failed to load $sim_file: $e")
            false
        end
    end, simulations)
    println("After filtering: $(length(simulations)) simulation(s)")
end

# Skip simulations that already have reports
if skip_existing
    before = length(simulations)
    simulations = filter(sim -> begin
        report_dir = joinpath(sim, output_subdir, "statistics_report")
        stats_exists = isfile(joinpath(report_dir, "statistics_summary.txt"))
        marker_exists = isfile(joinpath(report_dir, "enhanced_report_marker.txt"))
        completed = require_enhanced_report ? (stats_exists && marker_exists) : stats_exists
        !completed
    end, simulations)
    skipped = before - length(simulations)
    if skipped > 0
        println("Skipped $skipped simulation(s) with existing reports")
    end
    println("Simulations to process: $(length(simulations))")
end

if isempty(simulations)
    println("No simulations need processing. Done.")
    exit(0)
end

# --- WRITE SIMULATION LIST ---
sim_list_path = joinpath(log_dir, "sim_list.txt")
open(sim_list_path, "w") do f
    for sim in simulations
        println(f, sim)
    end
end
N = length(simulations)
println("Wrote $N simulation paths to: $sim_list_path")

# Adjust num_workers
K = min(N, num_workers)
if K < num_workers
    println("Adjusted workers from $num_workers to $K (only $N simulations)")
end

# Update metadata
open(joinpath(log_dir, "run_info.txt"), "a") do f
    println(f, "\n--- Discovery Results ---")
    println(f, "Total Simulations: $N")
    println(f, "Num Workers (K): $K")
    println(f, "Sims per Worker: ~$(ceil(Int, N / K))")
    println(f, "Filters: $filters")
    println(f, "Skip Existing: $skip_existing")
    println(f, "Require Enhanced Report: $require_enhanced_report")
    println(f, "Output Subdir: $output_subdir")
    println(f, "Tau Mult: $tau_mult")
end

# --- LOCATE WORKER SCRIPT ---
# Look relative to this script's directory
script_dir = dirname(@__FILE__)
worker_script = joinpath(script_dir, "report_worker.jl")
if !isfile(worker_script)
    error("Required worker script not found: $worker_script")
end

# --- CREATE AND SUBMIT ARRAY JOB ---
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

slurm_content = """#!/bin/bash
#SBATCH --job-name=ReportWorkers_$timestamp
#SBATCH --output=$log_dir/%A_%a.out
#SBATCH --error=$log_dir/%A_%a.error
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --exclude=$exclude_nodes
#SBATCH --array=1-$K
#SBATCH -A dror-account
#SBATCH -p drorq

export JULIA_DEPOT_PATH=$depot_path

echo "Worker \$SLURM_ARRAY_TASK_ID of $K starting on \$(hostname) at \$(date)"

julia $worker_script "$sim_list_path" \$SLURM_ARRAY_TASK_ID $K \\
    --output-subdir "$output_subdir" \\
    --tau-mult $tau_mult

echo "Worker \$SLURM_ARRAY_TASK_ID finished at \$(date) with exit code \$?"
"""

submit_file = joinpath(log_dir, "submit_workers.slurm")
open(submit_file, "w") do f
    write(f, slurm_content)
end

println("\nSubmitting array job with $K workers...")
submit_output = read(`sbatch $submit_file`, String)
println("sbatch output: $(strip(submit_output))")

array_job_id = parse_sbatch_job_id(submit_output)
run_info_path = joinpath(log_dir, "run_info.txt")
open(run_info_path, "a") do f
    println(f, "\n--- Array Submission ---")
    println(f, "Array sbatch output: $(strip(submit_output))")
    println(f, "Array Job ID: $(isnothing(array_job_id) ? "N/A" : array_job_id)")
    println(f, "Worker script: $worker_script")
    println(f, "Sim list path: $sim_list_path")
end

if !isnothing(array_job_id)
    open(joinpath(log_dir, "worker_array_job_id.txt"), "w") do f
        println(f, array_job_id)
    end
end

println("\n" * "="^80)
println("Array job submitted!")
println("  $K workers will process $N simulations in parallel")
println("  ~$(ceil(Int, N / K)) simulations per worker")
println("  Array job_id: $(isnothing(array_job_id) ? "unknown" : array_job_id)")
println("  Worker outputs: $log_dir/<job_id>_<task_id>.out")
println("="^80)
