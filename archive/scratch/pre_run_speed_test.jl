@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
"""
Pre-run script for cluster speed tests.
Submits SLURM jobs to test MC simulation performance for different:
1. System sizes (Lx, Ly)
2. Number of CPUs per node

Usage: julia pre_run_speed_test.jl
"""

using JLD2, Formatting, Printf, Dates

# ======================== CLUSTER CONFIGURATION ========================
julia_bin = JULIA_BIN
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")
exclude_nodes = get(ENV, "EXCLUDE_NODES", "")
run_file = joinpath(pwd(), "cluster_speed_test.jl")

# Exclude problematic nodes
# exclude_nodes_landau = ["landau59"]
# exclude_nodes_curie = ["curie25"]
# hostname = gethostname()
# if contains(hostname, "landau")
#     exclude_nodes = exclude_nodes_landau
# elseif contains(hostname, "curie")
#     exclude_nodes = exclude_nodes_curie
# else
#     exclude_nodes = String[]
#     println("Warning: Unknown cluster! Hostname is: $hostname")
# end

# ======================== OUTPUT CONFIGURATION ========================
# Change this to your desired output directory
out_dir = joinpath(V6_DATA_ROOT, "speed_tests")
folder_name = "speed_test_$(Dates.format(Dates.now(), "dd-mm-yyyy"))"
output_path = joinpath(out_dir, folder_name)

# ======================== TEST PARAMETERS ========================
# System sizes to test
system_sizes = [
    (6, 6),
    (8, 8),
    (10, 10),
    (10, 11),
    (12, 12),
    (12, 13),
    # (14, 14),  # Uncomment for larger systems
    # (14, 15),
]

# CPU counts to test
cpu_counts = [1, 2, 4, 8]

# Benchmark parameters
num_steps = 20           # Number of MC steps to time
num_measurements = 10    # Number of measurements to time

# ======================== SLURM TEMPLATE ========================
templateSLURM = FormatExpr("""#!/bin/bash

#SBATCH --job-name=speed_{1}cpu
#SBATCH --output={2}/speed_{1}cpu_%j.out
#SBATCH --error={2}/speed_{1}cpu_%j.error
#SBATCH --threads-per-core=1
#SBATCH -c {1}
#SBATCH -n 1
#SBATCH --mem-per-cpu=8GB
#SBATCH --requeue

#SBATCH -A all-account
#SBATCH -p allq, infiniq
#SBATCH --exclude={3}

export JULIA_DEPOT_PATH={4}
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

cd {5}

srun $julia_bin -e '
include("cluster_speed_test.jl")
run_full_speed_test(
    system_sizes={6},
    num_steps={7},
    num_measurements={8},
    output_dir="{2}"
)
'
""")

# ======================== SUBMIT JOBS ========================
# Create output directory
mkpath(output_path)

println("="^60)
println("SPEED TEST JOB SUBMISSION")
println("="^60)
println("Output directory: $output_path")
println("System sizes: $system_sizes")
println("CPU counts: $cpu_counts")
println("="^60)

# Format system_sizes for Julia code
system_sizes_str = "[" * join(["($Lx,$Ly)" for (Lx,Ly) in system_sizes], ", ") * "]"

for num_cpus in cpu_counts
    slurm_file = joinpath(output_path, "speed_test_$(num_cpus)cpu.txt")
    
    exclude_str = isempty(exclude_nodes) ? "none" : join(exclude_nodes, ",")
    
    open(slurm_file, "w") do f
        printfmt(f, templateSLURM, 
                 num_cpus,           # {1} CPU count
                 output_path,        # {2} Output directory
                 exclude_str,        # {3} Excluded nodes
                 current_depot,      # {4} Julia depot path
                 pwd(),              # {5} Working directory
                 system_sizes_str,   # {6} System sizes
                 num_steps,          # {7} Number of steps
                 num_measurements)   # {8} Number of measurements
    end
    
    println("Submitting job with $num_cpus CPU(s)...")
    run(`sbatch $slurm_file`)
end

println("\n" * "="^60)
println("All jobs submitted!")
println("Results will be saved to: $output_path")
println("="^60)
