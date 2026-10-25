@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2, Formatting, Printf, Base.Iterators, IterTools, ArgParse


run_file = joinpath(@__DIR__, "run_sim_num_control.jl")
julia_bin = JULIA_BIN

# Parse CLI args
s = ArgParseSettings()
@add_arg_table s begin
    "--sim_path"
        help = "Path to simulations root directory to monitor"
        arg_type = String
        required = true
end

parsed_args = parse_args(ARGS, s)
path = parsed_args["sim_path"]

println("Using sim_path: $path")


# exclude_nodes_landau = ["landau59"]  # Add problematic nodes here
# exclude_nodes_curie = ["curie25", "curie31"]  # Add problematic nodes here
# hostname = gethostname()
# if contains(hostname, "landau")
#     exclude_nodes = exclude_nodes_landau
#     cpus_per_task = 1
# elseif contains(hostname, "curie")
#     exclude_nodes = exclude_nodes_curie
#     cpus_per_task = 4
# else
#     error("Unknown cluster! Hostname is: $hostname")
# end

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
#SBATCH --mem-per-cpu=32GB
#SBATCH --requeue
#SBATCH --exclude=$exclude_nodes 
    
#SBATCH -A dror-account
#SBATCH -p drorq

#SBATCH --array=1-{3}

## SBATCH --mail-type=fail         # send email if job fails

#SBATCH --chdir={2} 
export DIR={2}
export JULIA_DEPOT_PATH={4}
srun --chdir=\$DIR $julia_bin $run_file --sim_path="$(path)"

""")
###############################################################################

c_path = joinpath(path, "num_control")
if !isdir(c_path)
    mkpath(c_path)
end
slurm_file_name = joinpath(c_path, "slurm.txt")
slurm_job_name="num_control"
open(slurm_file_name, "w") do slurm_file
    printfmt(slurm_file, templateSLURM, slurm_job_name, c_path, 1, current_depot, cpus_per_task)
end

println("Submitting sim_num control job with sim_path=$path")
run(`sbatch $slurm_file_name`)