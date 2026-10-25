@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using Dates

"""
Lightweight Cluster Submission Script for Batch Report Generation
==================================================================
Submits a single SLURM job that discovers simulations and spawns array workers.
No heavy package loading — just creates the SLURM script and submits it.

Usage:
    julia submit_batch_reports.jl /path/to/simulations/directory [options...]

Options:
    <integer>              Number of workers (default: 10)
    --workers=N            Number of workers (default: 10)
    --no-skip              Process all simulations even if reports exist
    --output-subdir=X      Subdirectory name for results (default: results)
    --tau-mult=X           Tau multiplier (default: 10.0)
    key=value              Parameter filters (e.g., Lx=10 beta=30.0)
"""

# --- PARSE COMMAND-LINE ARGUMENTS ---
if length(ARGS) < 1
    error("Usage: julia submit_batch_reports.jl <path_to_simulations> [num_workers] [--workers=N] [--no-skip] [key=value filters...]")
end

sim_path = ARGS[1]

if !isdir(sim_path)
    error("Simulation path does not exist: $sim_path")
end

# Collect all extra args to forward to the discovery script
extra_args = ARGS[2:end]

# --- SETUP LOGGING DIRECTORY ---
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
log_dir = joinpath(joinpath(V6_DATA_ROOT, "sim_logs", "batch_reports"), "batch_report_$timestamp")
mkpath(log_dir)

println("="^80)
println("BATCH REPORT SUBMISSION")
println("="^80)
println("Simulation path: $sim_path")
println("Log directory:   $log_dir")
println("Extra args:      $(join(extra_args, " "))")
println()

# Copy this submission script for documentation
cp(@__FILE__, joinpath(log_dir, "submit_script_snapshot.jl"))

# Create metadata file
open(joinpath(log_dir, "run_info.txt"), "w") do f
    println(f, "Batch Report Generation")
    println(f, "Run Date: $timestamp")
    println(f, "Submitted from Host: $(gethostname())")
    println(f, "Simulation Path: $sim_path")
    println(f, "Extra Args: $(join(extra_args, " "))")
end

# --- DETECT CLUSTER ---
hostname = gethostname()
depot_path = ""
exclude_nodes = ""
cpus_per_task = 1

if contains(hostname, "landau")
    depot_path = "~/.julia_landau"
    exclude_nodes = "landau59"
    cpus_per_task = 1
    println("Detected Landau Cluster")
elseif contains(hostname, "curie")
    depot_path = "~/.julia_curie"
    exclude_nodes = "curie25,curie31,curie09"
    cpus_per_task = 4
    println("Detected Curie Cluster")
else
    depot_path = get(ENV, "JULIA_DEPOT_PATH", "~/.julia")
    exclude_nodes = get(ENV, "EXCLUDE_NODES", "")
    cpus_per_task = parse(Int, get(ENV, "CPUS_PER_TASK", "1"))
    println("Using default cluster settings")
end

# -- Locate discovery script --
discovery_script = joinpath(@__DIR__, "discover_and_submit_workers.jl")
if !isfile(discovery_script)
    error("Required script not found: $discovery_script")
end

# Forward all extra args to the discovery script
extra_args_str = join(extra_args, " ")

function parse_sbatch_job_id(output::AbstractString)
    m = match(r"Submitted batch job\s+(\d+)", output)
    return isnothing(m) ? nothing : parse(Int, m.captures[1])
end

# --- CREATE SLURM SCRIPT FOR DISCOVERY JOB ---
# This job runs quickly: discovers simulations, writes sim_list.txt, submits array workers
slurm_content = """#!/bin/bash
#SBATCH --job-name=ReportDiscover_$timestamp
#SBATCH --output=$log_dir/discovery.out
#SBATCH --error=$log_dir/discovery.error
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --exclude=$exclude_nodes
#SBATCH -A dror-account
#SBATCH -p drorq

export JULIA_DEPOT_PATH=$depot_path

echo "Discovery job starting on \$(hostname) at \$(date)"

julia $discovery_script "$sim_path" "$log_dir" $cpus_per_task "$depot_path" "$exclude_nodes" $extra_args_str

echo "Discovery job finished at \$(date) with exit code \$?"
"""

submit_file = joinpath(log_dir, "submit_discovery.slurm")
open(submit_file, "w") do f
    write(f, slurm_content)
end

println("SLURM script: $submit_file")
println("\nSubmitting discovery job...")
submit_output = read(`sbatch $submit_file`, String)
println("sbatch output: $(strip(submit_output))")

discovery_job_id = parse_sbatch_job_id(submit_output)
open(joinpath(log_dir, "run_info.txt"), "a") do f
    println(f, "\n--- Submission ---")
    println(f, "Discovery sbatch output: $(strip(submit_output))")
    println(f, "Discovery Job ID: $(isnothing(discovery_job_id) ? "N/A" : discovery_job_id)")
end

if !isnothing(discovery_job_id)
    open(joinpath(log_dir, "discovery_job_id.txt"), "w") do f
        println(f, discovery_job_id)
    end
end

println("\n" * "="^80)
println("Discovery job submitted!")
println("  It will discover simulations and submit array workers automatically.")
println("  Log directory: $log_dir")
println("  Discovery output: $log_dir/discovery.out")
println("  Discovery job_id: $(isnothing(discovery_job_id) ? "unknown" : discovery_job_id)")
println("  Worker outputs:   $log_dir/<job_id>_<task_id>.out")
println("="^80)
