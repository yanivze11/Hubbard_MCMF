#=
Report Worker — SLURM Array Job Worker
=======================================
Processes a chunk of simulations assigned to this worker via SLURM array task ID.
N simulations are split evenly across K workers; each worker handles ceil(N/K) sims.
The worker loops over its chunk repeatedly until a maximum wall time is reached
(default: 1 week), so reports are always regenerated with the latest data.

Usage (called by SLURM, not directly):
    julia report_worker.jl <sim_list_path> <task_id> <num_workers> [--output-subdir results] [--tau-mult 10.0] [--max-hours 168]
    
Arguments:
    sim_list_path  - Path to text file with one simulation path per line
    task_id        - SLURM_ARRAY_TASK_ID (1-based)
    num_workers    - Total number of workers (K)
    
Optional:
    --output-subdir  - Subdirectory name for report output (default: "results")
    --tau-mult       - Tau multiplier for block size (default: 10.0)
    --max-hours      - Maximum wall time in hours before stopping (default: 168 = 1 week)
=#

include(joinpath(@__DIR__, "..", "..", "simulation_statistics_report.jl"))
include(joinpath(@__DIR__, "batch_generate_reports.jl"))  # for generate_report_safe, find_simulations

using Dates

# --- Parse arguments ---
if length(ARGS) < 3
    error("Usage: julia report_worker.jl <sim_list_path> <task_id> <num_workers> [--output-subdir results] [--tau-mult 10.0]")
end

sim_list_path = ARGS[1]
task_id = parse(Int, ARGS[2])
num_workers = parse(Int, ARGS[3])

# Parse optional keyword arguments
function parse_worker_kwargs(args, start_idx)
    output_subdir = "results"
    tau_mult = 10.0
    max_hours = 168.0  # 1 week default
    i = start_idx
    while i <= length(args)
        if args[i] == "--output-subdir" && i + 1 <= length(args)
            output_subdir = args[i+1]
            i += 2
        elseif args[i] == "--tau-mult" && i + 1 <= length(args)
            tau_mult = parse(Float64, args[i+1])
            i += 2
        elseif args[i] == "--max-hours" && i + 1 <= length(args)
            max_hours = parse(Float64, args[i+1])
            i += 2
        else
            println("Warning: Ignoring unknown argument '$(args[i])'")
            i += 1
        end
    end
    return output_subdir, tau_mult, max_hours
end

output_subdir, tau_mult, max_hours = parse_worker_kwargs(ARGS, 4)
max_wall_seconds = max_hours * 3600.0
worker_start_time = time()

# --- Read simulation list ---
if !isfile(sim_list_path)
    error("Simulation list file not found: $sim_list_path")
end

lines = filter(!isempty, strip.(readlines(sim_list_path)))
N = length(lines)

# --- Compute this worker's chunk ---
chunk_size = ceil(Int, N / num_workers)
start_idx = (task_id - 1) * chunk_size + 1
end_idx = min(task_id * chunk_size, N)

if start_idx > N
    println("Worker $task_id/$num_workers: no simulations to process (N=$N, K=$num_workers)")
    exit(0)
end

num_assigned = end_idx - start_idx + 1
println("="^80)
println("REPORT WORKER $task_id/$num_workers")
println("Started: $(Dates.now())")
println("="^80)
println("  Simulation list: $sim_list_path")
println("  Total simulations: $N")
println("  This worker's range: $start_idx to $end_idx ($num_assigned simulations)")
println("  Output subdir: $output_subdir")
println("  Tau multiplier: $tau_mult")
println("  Max wall time: $(max_hours)h ($(round(max_hours/24, digits=1)) days)")
println("  Mode: continuous loop (always regenerate, no skip)")
println("="^80)
flush(stdout)

# --- Process assigned simulations in a continuous loop ---
function run_worker_loop(lines, start_idx, end_idx, N, task_id, num_assigned, 
                         output_subdir, tau_mult, max_wall_seconds, worker_start_time)
    total_successful = 0
    total_failed = 0
    pass_number = 0

    while true
        pass_number += 1
        elapsed_total = time() - worker_start_time
        remaining = max_wall_seconds - elapsed_total
        
        if remaining <= 0
            println("\n⏰ Wall time limit reached before starting pass $pass_number. Stopping.")
            flush(stdout)
            break
        end
        
        println("\n" * "━"^80)
        println("PASS $pass_number — elapsed $(round(elapsed_total/3600, digits=2))h / $(round(max_wall_seconds/3600, digits=1))h")
        println("━"^80)
        flush(stdout)
        
        pass_successful = 0
        pass_failed = 0
        time_exceeded = false

        for i in start_idx:end_idx
            # Check wall time before each simulation
            elapsed_total = time() - worker_start_time
            if elapsed_total >= max_wall_seconds
                println("\n⏰ Wall time limit reached during pass $pass_number. Stopping.")
                flush(stdout)
                time_exceeded = true
                break
            end
            
            sim_path = String(strip(lines[i]))
            local_idx = i - start_idx + 1
            remaining_h = round((max_wall_seconds - elapsed_total) / 3600, digits=2)
            
            println("\n[Pass $pass_number | $local_idx/$num_assigned] (global $i/$N, worker $task_id) remaining=$(remaining_h)h")
            println("  Processing: $sim_path")
            flush(stdout)
            
            t_sim = time()
            
            if generate_report_safe(sim_path; output_subdir=output_subdir, tau_mult=tau_mult)
                pass_successful += 1
            else
                pass_failed += 1
            end
            
            elapsed_sim = time() - t_sim
            println("  Simulation $i completed in $(round(elapsed_sim, digits=1))s")
            flush(stdout)
            
            # Note: generate_report_safe already runs full GC in its finally block.
            # This extra call ensures any worker-level allocations are also swept.
            GC.gc(true)
            free_mb = round(Sys.free_memory() / 1024^2, digits=0)
            println("  [mem] free=$(free_mb) MB / total=$(round(Sys.total_memory()/1024^2, digits=0)) MB")
        end

        total_successful += pass_successful
        total_failed += pass_failed
        
        elapsed_total = time() - worker_start_time
        println("\n  Pass $pass_number summary: $(pass_successful) ok, $(pass_failed) failed ($(round(elapsed_total/3600, digits=2))h elapsed)")
        flush(stdout)
        
        if time_exceeded
            break
        end
    end

    return total_successful, total_failed, pass_number
end

total_successful, total_failed, total_passes = run_worker_loop(
    lines, start_idx, end_idx, N, task_id, num_assigned,
    output_subdir, tau_mult, max_wall_seconds, worker_start_time)

# --- Summary ---
total_elapsed = time() - worker_start_time
println("\n" * "="^80)
println("WORKER $task_id/$num_workers COMPLETE")
println("="^80)
println("  Passes completed: $total_passes")
println("  Total successful: $total_successful")
println("  Total failed:     $total_failed")
println("  Sims per pass:    $num_assigned")
println("  Wall time used:   $(round(total_elapsed/3600, digits=2))h / $(round(max_hours, digits=1))h")
println("  Finished:         $(Dates.now())")
println("="^80)
flush(stdout)
