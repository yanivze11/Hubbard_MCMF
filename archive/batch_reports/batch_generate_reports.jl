#=
Batch Report Generator for Multiple Simulations
===============================================================================
This script generates statistical analysis reports for all simulations in a directory.
It iterates through each simulation folder and creates comprehensive plots in a 
results/ subdirectory within each simulation.

Usage (local):
    julia batch_generate_reports.jl /path/to/simulations/directory

Usage (cluster via pre_pre_run.jl):
    julia pre_pre_run.jl batch_generate_reports.jl
    (The script will be submitted as a SLURM job)
=#

include(joinpath(@__DIR__, "..", "..", "simulation_statistics_report.jl"))
using Dates, ArgParse, JSON

"""
Find all simulation directories that contain seed_* folders with results.jld2.
Directory structure: base_path/sim#/seed_#/results.jld2
"""
function find_simulations(base_path::String)
    simulations = String[]
    
    if !isdir(base_path)
        error("Base path does not exist: $base_path")
    end
    
    # Look through all subdirectories in base_path
    for item in readdir(base_path)
        item_path = joinpath(base_path, item)
        
        if isdir(item_path)
            # Check if this directory contains seed_* folders
            seed_dirs = filter(f -> startswith(f, "seed_") && isdir(joinpath(item_path, f)), 
                             readdir(item_path))
            
            if !isempty(seed_dirs)
                # This is a simulation directory (contains seed folders)
                # Check if at least one seed has results.jld2
                has_results = any(isfile(joinpath(item_path, seed_dir, "results.jld2")) 
                                 for seed_dir in seed_dirs)
                
                if has_results
                    push!(simulations, item_path)
                end
            end
        end
    end
    
    return simulations
end

"""
Run report generation for a single simulation with error handling.
"""
function generate_report_safe(sim_path::String; output_subdir="results", tau_mult=10.0)
    try
        println("\n" * "="^80)
        println("Processing: $sim_path")
        println("="^80)
        
        # Create output directory inside simulation folder
        output_dir = joinpath(sim_path, output_subdir)
        mkpath(output_dir)
        
        # Generate report
        generate_report(
            sim_path; 
            save_plots=true, 
            output_dir=joinpath(output_dir, "statistics_report"),
            tau_mult=tau_mult,
            generate_pdf=false # Avoid generating large PDF reports
        )
        
        println("✓ Successfully generated reports for: $sim_path")
        flush(stdout)
        return true
        
    catch e
        println("✗ Error processing $sim_path:")
        println("  $(string(e))")
        println("  ", sprint(showerror, e))
        flush(stdout)
        return false
    finally
        # Force full garbage collection between simulations to release loaded data and plots
        # GC.gc() alone is incremental — large JLD2 arrays and Plots objects survive it.
        # Call full GC twice: first pass marks, second pass sweeps.
        GC.gc(true)  # full collection (Julia ≥ 1.9)
        GC.gc(true)
        # Close all open Plots figure windows held by the GR backend
        try; Plots.closeall(); catch; end
        free_mb = round(Sys.free_memory() / 1024^2, digits=0)
        total_mb = round(Sys.total_memory() / 1024^2, digits=0)
        println("  [mem] free=$(free_mb) MB / total=$(total_mb) MB after GC")
        flush(stdout)
    end
end

"""
Main batch processing function.
"""
function batch_generate_reports(base_path::String; 
                                output_subdir="results",
                                tau_mult=10.0,
                                max_simulations=nothing,
                                skip_existing=false,
                                require_enhanced_report::Bool=false,
                                filters=Dict{String, Any}())
    
    println("\n" * "="^80)
    println("BATCH REPORT GENERATOR FOR MULTIPLE SIMULATIONS")
    println("Started: $(Dates.now())")
    println("="^80)
    println("Base path: $base_path")
    println("Output subdirectory: $output_subdir")
    println("Tau multiplier: $tau_mult")
    println("Skip existing: $skip_existing")
    println("Require enhanced report marker: $require_enhanced_report")
    println("Filters: $filters")
    println()
    
    # Find all simulations
    simulations = find_simulations(base_path)
    println("Found $(length(simulations)) simulation(s)")
    
    if isempty(simulations)
        println("No simulations found in: $base_path")
        return
    end

    # Filter simulations based on criteria
    if !isempty(filters)
        println("Filtering simulations based on criteria: $filters")
        simulations = filter(sim_path -> begin
            sim_info = Dict{String, Any}()
            sim_file = joinpath(sim_path, "sim.jld2")
            if isfile(sim_file)
                try
                    sim_info = JLD2.load(sim_file)
                catch e
                    println("Warning: Failed to load $sim_file: $e")
                end
            end
            # println("Debug: sim_info for $sim_path: $sim_info")  # Debug output
            all(filter_key -> begin
                value = get(sim_info, filter_key, nothing)
                # println("Debug: Checking filter $filter_key=$value against $filters[filter_key]")  # Debug output
                value == filters[filter_key]
            end, keys(filters))
        end, simulations)
        println("Found $(length(simulations)) simulation(s) matching filters")
    end

    if isempty(simulations)
        println("No simulations match the specified filters.")
        return
    end

    # Optionally limit number of simulations
    if !isnothing(max_simulations) && length(simulations) > max_simulations
        println("Limiting to $max_simulations simulations")
        simulations = simulations[1:max_simulations]
    end
    
    # Process each simulation
    successful = 0
    failed = 0
    skipped = 0
    
    for (idx, sim_path) in enumerate(simulations)
        println("\n[$(idx)/$(length(simulations))] Processing simulation...")
        
        # Check if output already exists
        existing_output = joinpath(sim_path, output_subdir, "statistics_report")
        if skip_existing && isdir(existing_output)
            stats_summary = joinpath(existing_output, "statistics_summary.txt")
            enhanced_marker = joinpath(existing_output, "enhanced_report_marker.txt")
            is_complete = if require_enhanced_report
                isfile(stats_summary) && isfile(enhanced_marker)
            else
                isfile(stats_summary)
            end
            if is_complete
                println("  Skipping (reports already exist at $existing_output)")
                skipped += 1
                continue
            end
        end
        
        # Generate report
        if generate_report_safe(sim_path; output_subdir=output_subdir, tau_mult=tau_mult)
            successful += 1
        else
            failed += 1
        end
    end
    
    # Summary
    println("\n" * "="^80)
    println("BATCH PROCESSING COMPLETE")
    println("="^80)
    println("Successful: $successful")
    println("Failed: $failed")
    println("Skipped: $skipped")
    println("Total: $(length(simulations))")
    println("Completed: $(Dates.now())")
    println("="^80 * "\n")
    
    return (successful=successful, failed=failed, skipped=skipped, total=length(simulations))
end

# ============================================================================
# Command-line interface
# ============================================================================

function main()
    s = ArgParseSettings(
        description="Generate statistical analysis reports for all simulations in a directory"
    )
    
    @add_arg_table s begin
        "sim_path"
            help = "Path to the directory containing simulations"
            required = true
        "--output-subdir"
            help = "Subdirectory name inside each simulation for results (default: results)"
            default = "results"
        "--tau-mult"
            help = "Tau multiplier for block size calculation (default: 10.0)"
            arg_type = Float64
            default = 10.0
        "--max-simulations"
            help = "Maximum number of simulations to process (default: all)"
            arg_type = Int
            default = nothing
        "--no-skip"
            help = "Process all simulations even if results already exist"
            action = :store_true
        "--filters"
            help = "Filters for simulations as a JSON string (e.g., '{\"Lx\": 12, \"mu\": 2.0}')"
            default = "{}"
        "--require-enhanced-report"
            help = "Skip existing only when enhanced_report_marker.txt exists alongside statistics_summary.txt"
            action = :store_true
    end
    
    parsed_args = parse_args(ARGS, s)
    
    sim_path = parsed_args["sim_path"]
    output_subdir = get(parsed_args, "output_subdir", "results")
    tau_mult = get(parsed_args, "tau-mult", 10.0)
    max_simulations = get(parsed_args, "max-simulations", nothing)
    skip_existing = !get(parsed_args, "no-skip", false)
    require_enhanced_report = get(parsed_args, "require-enhanced-report", false)
    filters = JSON.parse(get(parsed_args, "filters", "{}"))
    
    starttime = Dates.now()
    maxtime = Dates.Day(7)  # Maximum allowed time for batch processing
    # Run batch processing
    while Dates.now() - starttime < maxtime
        batch_generate_reports(
            sim_path;
            output_subdir=output_subdir,
            tau_mult=tau_mult,
            max_simulations=max_simulations,
            skip_existing=skip_existing,
            require_enhanced_report=require_enhanced_report,
            filters=filters
        )
        # sleep for a short duration before next iteration (if needed)
        sleep(60 * 60)  # Sleep for 1 hour
        # break  # Remove this line if you want continuous processing
    end
end

# Run if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
