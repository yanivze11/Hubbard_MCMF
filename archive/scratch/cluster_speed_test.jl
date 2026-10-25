@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
"""
Cluster Speed Test
==================
This script tests the Monte Carlo simulation performance for different:
1. System sizes (Lx, Ly)
2. Number of CPUs per node

It measures:
- Preparation time (setting up SimData and LatticeData)
- MC step time (single step with all site updates)
- Measurement time (computing observables)

Usage:
------
Local:  julia -t N cluster_speed_test.jl
Cluster: sbatch cluster_speed_slurm.txt

The script saves results to a JLD2 file for later analysis.
"""

using JLD2, Formatting, Printf, Dates, Statistics
include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))

# Result structure for storing timing data
mutable struct SpeedTestResults
    Lx::Int64
    Ly::Int64
    num_threads::Int64
    num_steps::Int64
    num_measurements::Int64
    prepare_time::Float64
    step_times::Vector{Float64}
    measurement_times::Vector{Float64}
    mean_step_time::Float64
    std_step_time::Float64
    mean_measurement_time::Float64
    std_measurement_time::Float64
    diag_method::String
    hostname::String
    timestamp::DateTime
end

"""
    run_speed_test(Lx, Ly; kwargs...)

Run a speed test for a given system size and return detailed timing results.
"""
function run_speed_test(Lx::Int64, Ly::Int64;
                        num_steps::Int64=20,
                        num_measurements::Int64=10,
                        U::Float64=4.0,
                        t::Float64=1.0,
                        V::Float64=0.0,
                        mu::Float64=0.0,
                        beta::Float64=10.0,
                        diag_method::String="K0",
                        TCluster_size::Int64=4,
                        pairing_symmetry::String="s-wave",
                        seed::Int64=12345)
    
    num_threads = Threads.nthreads()
    hostname = gethostname()
    
    println("\n" * "="^60)
    println("Speed Test: $(Lx)x$(Ly) lattice, $(num_threads) threads")
    println("="^60)
    
    # ============ Preparation Phase ============
    println("Preparing SimData and LatticeData...")
    prep_start = time()
    
    sim_data = SimData(
        Lx=Lx,
        Ly=Ly,
        SC_L=1,
        U=U,
        t=t,
        V=V,
        seed=seed,
        mu=mu,
        beta=beta,
        num_measure=100,
        num_thermal=100,
        measuring_interval=1,
        saving_interval=100,
        w_list=collect(-3.0:0.1:3.0),
        guess="uniform",
        U_disorder="uniform",
        diag_method=diag_method,
        pairing_symmetry=pairing_symmetry,
        directory=".",
        num_cores=1,
        num_threads=num_threads,
        start_time=Dates.now(),
        w_conductivity_range=[10^a for a in -10:0.5:0.5],
        U_disorder_params=[0.0, 4.0, 1.0, 0.7],
        U_impurity=0.0,
        TCluster_size_x=TCluster_size,
        TCluster_size_y=TCluster_size,
        smooth_U=(false, 0.8),
        loop_random=true,
        B=0.0,
        gauge_choice="LandauX",
        boundary_conditions="PBC"
    )
    
    lattice_data = Lattice_Data(sim_data)
    
    prepare_time = time() - prep_start
    println("  Preparation time: $(round(prepare_time, digits=3)) seconds")
    
    # ============ Warmup Phase ============
    println("Warmup (2 steps + 1 measurement)...")
    for _ in 1:2
        MC_stepV2!(lattice_data; method=diag_method, pairing=pairing_symmetry)
    end
    make_measurement_K!(lattice_data, 1)
    
    # ============ MC Step Timing ============
    println("Timing MC steps ($num_steps steps)...")
    step_times = Float64[]
    
    for i in 1:num_steps
        step_start = time()
        MC_stepV2!(lattice_data; method=diag_method, pairing=pairing_symmetry)
        step_time = time() - step_start
        push!(step_times, step_time)
        
        if i % 5 == 0
            println("  Step $i/$(num_steps): $(round(step_time, digits=4)) s")
        end
    end
    
    mean_step = mean(step_times)
    std_step = std(step_times)
    println("  Mean step time: $(round(mean_step, digits=4)) ± $(round(std_step, digits=4)) seconds")
    
    # ============ Measurement Timing ============
    println("Timing measurements ($num_measurements measurements)...")
    measurement_times = Float64[]
    
    for i in 1:num_measurements
        meas_start = time()
        make_measurement_K!(lattice_data, i)
        meas_time = time() - meas_start
        push!(measurement_times, meas_time)
        
        if i % 3 == 0
            println("  Measurement $i/$(num_measurements): $(round(meas_time, digits=4)) s")
        end
    end
    
    mean_meas = mean(measurement_times)
    std_meas = std(measurement_times)
    println("  Mean measurement time: $(round(mean_meas, digits=4)) ± $(round(std_meas, digits=4)) seconds")
    
    # ============ Create Results ============
    results = SpeedTestResults(
        Lx, Ly, num_threads, num_steps, num_measurements,
        prepare_time, step_times, measurement_times,
        mean_step, std_step, mean_meas, std_meas,
        diag_method, hostname, Dates.now()
    )
    
    return results
end

"""
    run_full_speed_test(; kwargs...)

Run speed tests for multiple system sizes and save results.
"""
function run_full_speed_test(;
        system_sizes::Vector{Tuple{Int64,Int64}}=[(6,6), (8,8), (10,10), (10,11), (12,12), (12,13), (14,14)],
        num_steps::Int64=20,
        num_measurements::Int64=10,
        output_dir::String="speed_test_results",
        kwargs...)
    
    # Create output directory
    mkpath(output_dir)
    
    num_threads = Threads.nthreads()
    hostname = gethostname()
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    
    println("\n" * "="^70)
    println("CLUSTER SPEED TEST")
    println("="^70)
    println("Hostname: $hostname")
    println("Threads: $num_threads")
    println("System sizes: $system_sizes")
    println("="^70)
    
    all_results = Dict{String, SpeedTestResults}()
    
    for (Lx, Ly) in system_sizes
        try
            results = run_speed_test(Lx, Ly; 
                                     num_steps=num_steps, 
                                     num_measurements=num_measurements,
                                     kwargs...)
            key = "$(Lx)x$(Ly)"
            all_results[key] = results
        catch e
            println("ERROR for $(Lx)x$(Ly): $e")
            println(stacktrace(catch_backtrace()))
        end
    end
    
    # Save results
    output_file = joinpath(output_dir, "speed_test_$(hostname)_$(num_threads)cpus_$(timestamp).jld2")
    jldopen(output_file, "w") do file
        file["results"] = all_results
        file["system_sizes"] = system_sizes
        file["num_threads"] = num_threads
        file["hostname"] = hostname
        file["timestamp"] = timestamp
    end
    println("\nResults saved to: $output_file")
    
    # Print summary
    print_summary(all_results)
    
    return all_results, output_file
end

"""
    print_summary(results)

Print a formatted summary table of timing results.
"""
function print_summary(results::Dict{String, SpeedTestResults})
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    println(@sprintf("%-12s %8s %12s %12s %12s %12s", 
                     "Size", "Threads", "Prep (s)", "Step (s)", "Meas (s)", "Total/step"))
    println("-"^80)
    
    for key in sort(collect(keys(results)))
        r = results[key]
        total_per_step = r.mean_step_time + r.mean_measurement_time / r.num_steps * r.num_measurements
        println(@sprintf("%-12s %8d %12.4f %12.4f %12.4f %12.4f",
                        "$(r.Lx)x$(r.Ly)", r.num_threads, r.prepare_time,
                        r.mean_step_time, r.mean_measurement_time, total_per_step))
    end
    println("="^80)
end

"""
    analyze_results(results_files::Vector{String})

Load and analyze results from multiple speed test runs (e.g., different CPU counts).
"""
function analyze_results(results_files::Vector{String})
    all_data = Dict{Int64, Dict{String, SpeedTestResults}}()
    
    for file in results_files
        data = JLD2.load(file)
        num_threads = data["num_threads"]
        all_data[num_threads] = data["results"]
    end
    
    # Print comparison table
    println("\n" * "="^90)
    println("CPU SCALING ANALYSIS")
    println("="^90)
    
    # Get all system sizes
    all_sizes = Set{String}()
    for (_, results) in all_data
        for key in keys(results)
            push!(all_sizes, key)
        end
    end
    
    thread_counts = sort(collect(keys(all_data)))
    
    for size in sort(collect(all_sizes))
        println("\nSystem size: $size")
        println(@sprintf("  %8s %12s %12s %10s %10s", 
                        "Threads", "Step (s)", "Meas (s)", "Step Spd", "Meas Spd"))
        println("  " * "-"^60)
        
        base_step = nothing
        base_meas = nothing
        
        for threads in thread_counts
            if haskey(all_data[threads], size)
                r = all_data[threads][size]
                
                if base_step === nothing
                    base_step = r.mean_step_time
                    base_meas = r.mean_measurement_time
                end
                
                step_speedup = base_step / r.mean_step_time
                meas_speedup = base_meas / r.mean_measurement_time
                
                println(@sprintf("  %8d %12.4f %12.4f %10.2fx %10.2fx",
                                threads, r.mean_step_time, r.mean_measurement_time,
                                step_speedup, meas_speedup))
            end
        end
    end
    
    return all_data
end

# ============ SLURM Template Generator ============
"""
    generate_slurm_scripts(; kwargs...)

Generate SLURM scripts for running speed tests with different CPU counts.
"""
function generate_slurm_scripts(;
        output_dir::String="speed_test_slurm",
        cpu_counts::Vector{Int64}=[1, 2, 4, 8],
        system_sizes::Vector{Tuple{Int64,Int64}}=[(6,6), (8,8), (10,10), (10,11), (12,12)],
        julia_bin::String=JULIA_BIN,
        exclude_nodes::Vector{String}=["landau59", "curie25"])
    
    mkpath(output_dir)
    current_depot = get(ENV, "JULIA_DEPOT_PATH", "")
    script_dir = pwd()
    
    for cpus in cpu_counts
        slurm_content = """#!/bin/bash

#SBATCH --job-name=speed_test_$(cpus)cpu
#SBATCH --output=$(output_dir)/speed_test_$(cpus)cpu_%j.out
#SBATCH --error=$(output_dir)/speed_test_$(cpus)cpu_%j.error
#SBATCH --threads-per-core=1
#SBATCH -c $(cpus)
#SBATCH -n 1
#SBATCH --mem-per-cpu=8GB
#SBATCH --requeue

#SBATCH -A all-account
#SBATCH -p allq, infiniq
#SBATCH --exclude=$(join(exclude_nodes, ","))

export JULIA_DEPOT_PATH=$(current_depot)
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

cd $(script_dir)
srun $(julia_bin) -e '
include("cluster_speed_test.jl")
run_full_speed_test(
    system_sizes=$(system_sizes),
    num_steps=20,
    num_measurements=10,
    output_dir="speed_test_results"
)
'
"""
        
        slurm_file = joinpath(output_dir, "speed_test_$(cpus)cpu.slurm")
        open(slurm_file, "w") do f
            write(f, slurm_content)
        end
        println("Generated: $slurm_file")
    end
    
    # Generate a master script to submit all jobs
    submit_script = joinpath(output_dir, "submit_all.sh")
    open(submit_script, "w") do f
        write(f, "#!/bin/bash\n")
        write(f, "# Submit all speed test jobs\n\n")
        for cpus in cpu_counts
            write(f, "sbatch speed_test_$(cpus)cpu.slurm\n")
        end
    end
    println("Generated submit script: $submit_script")
    
    println("\nTo run all tests, execute:")
    println("  cd $output_dir && bash submit_all.sh")
end

# ============ Main execution when run directly ============
if abspath(PROGRAM_FILE) == @__FILE__
    # Default execution: run speed test with current settings
    println("Running cluster speed test...")
    println("Julia threads: $(Threads.nthreads())")
    
    # Check if running on cluster or locally
    hostname = gethostname()
    
    if contains(hostname, "landau") || contains(hostname, "curie")
        # Cluster execution - more comprehensive test
        system_sizes = [(6,6), (8,8), (10,10), (10,11), (12,12), (12,13)]
    else
        # Local execution - smaller test
        system_sizes = [(6,6), (8,8), (10,10)]
    end
    
    results, output_file = run_full_speed_test(
        system_sizes=system_sizes,
        num_steps=20,
        num_measurements=10,
        output_dir="speed_test_results"
    )
end
