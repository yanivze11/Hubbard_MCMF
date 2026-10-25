@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2, Formatting, Printf, Base.Iterators, IterTools

run_file = joinpath(pwd(), "run_sim_num_filtered.jl")
julia_bin = JULIA_BIN

# Set your simulation path and filter string here
path = joinpath(V6_DATA_ROOT, "data", "Mag__22-01-2026")
filter_str = "Lx=10"  # <-- EDIT THIS LINE for your filter

current_depot = get(ENV, "JULIA_DEPOT_PATH", "")
exclude_nodes = get(ENV, "EXCLUDE_NODES", "")
cpus_per_task = parse(Int, get(ENV, "CPUS_PER_TASK", "1"))

######################### Generate slurm.txt #################################
templateSLURM = FormatExpr("""#!/bin/bash

#SBATCH --job-name={1}
#SBATCH --output={2}/%A_%a.out
#SBATCH --error={2}/%A_%a.error
#SBATCH --threads-per-core=1
#SBATCH -c {5}
#SBATCH -n 1
#SBATCH --mem-per-cpu=8GB
#SBATCH --requeue
#SBATCH --exclude=$exclude_nodes # exclude problematic nodes
    
#SBATCH -A dror-account
#SBATCH -p drorq

#SBATCH --array=1-1

## SBATCH --mail-type=fail         # send email if job fails

#SBATCH --chdir={2} 
export DIR={2}
export JULIA_DEPOT_PATH={4}  
srun --chdir=\$DIR $julia_bin $run_file --path={2} --filter={6}

""")
###############################################################################

c_path = joinpath(path, "num_control")
if !isdir(c_path)
    mkpath(c_path)
end

slurm_job_name = "num_control_filtered"
base_path = joinpath(c_path, "slurm_filtered")
slurm_file_name = "$(base_path).txt"

# Find a unique filename without stacking suffixes
counter = 1
while isfile(slurm_file_name)
    global slurm_file_name = "$(base_path)_$(counter).txt"
    global counter += 1
end

open(slurm_file_name, "w") do slurm_file
    printfmt(slurm_file, templateSLURM, slurm_job_name, c_path, 1, current_depot, cpus_per_task, filter_str)
end

run(`sbatch $slurm_file_name`)
