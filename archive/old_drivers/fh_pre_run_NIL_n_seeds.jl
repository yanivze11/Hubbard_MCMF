@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2, Formatting, Printf, Base.Iterators
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

# =============================================================================
# CONFIGURATION: Change run_file to match your setup
# =============================================================================
run_file = joinpath(pwd(),"fh_run_n_seeds.jl")  # Uses multi-seed runner
julia_bin = JULIA_BIN
partitions = get(ENV, "PARTITIONS", "allq,infiniq")
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")

######################### Generate slurm.txt #################################
# Note: Increased time limit for multi-seed runs
templateSLURM = FormatExpr("""#!/bin/bash

#SBATCH --job-name={1}
#SBATCH --output={2}/%A_%a.out
#SBATCH --error={2}/%A_%a.error
#SBATCH --threads-per-core=1
#SBATCH -c {4}
#SBATCH -n 1
#SBATCH --mem-per-cpu=8GB
#SBATCH --requeue
    
#SBATCH -A all-account
#SBATCH -p $partitions
#SBATCH --chdir={2}

#SBATCH --array=1-{3}

## SBATCH --mail-type=fail         # send email if job fails
    
export DIR={2}
export JULIA_DEPOT_PATH={5}  
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1  # or MKL equivalent, to avoid oversubscription
srun --chdir=\$DIR $julia_bin $run_file --sim_path=\$DIR

""")
###############################################################################

out_dir = joinpath(V6_DATA_ROOT, "data")
folder_name = "NIL_seeds__10-01-2026"  # Different folder name for multi-seed runs
path = joinpath(out_dir, folder_name)
sim_name = "NIL_seeds"

begin
using IterTools

# System parameters
Ls = [(10,11), (16, 17)] # Lattice sizes to iterate over
Us = [0.0]
t = 1.0
Vs = [1.0] # disorder strength for chemical potential
impurities = [0.0] # impurity strength for interaction

# =============================================================================
# SEEDS CONFIGURATION: Multiple seeds per simulation
# =============================================================================
# Each job runs ALL these seeds sequentially
jobs_per_sim = 2  # Number of parallel jobs per simulation
seeds_per_job = 20  # Number of seeds per simulation job
seeds = [Int64.(rand(1:Int(10e6), seeds_per_job)) for _ in 1:jobs_per_sim]  # Single array with all seeds
# Note: seeds is a vector of vectors for compatibility, but we only have one entry
# Each job will run all seeds_per_job seeds one after another

# Target occupations
wanted_ns = [0.5, 0.875, 1.0]  # Target electron occupations
betas = [30.0]

### Simulation parameters (after convergence)
num_measure = 1;
num_thermal = 0;
measuring_interval = 1;
w_list = [-4.5:0.1:4.5;];
guess = "uniform"
U_disorder = "uniform"
Bs = [0.0]

diag_methods = ["K0"]
pairings = ["s-wave"]
core_nums = [1]
w_conductivity_range = [10^a for a in -10:0.05:0.5]
U_puddle_params = [0.0, 4.0, 1.0, 0.7]
TClustersizes = [4]
SC_Ls = [1]
smooth_U = (false, 0.8)
boundary_condition = "PBC"
gauge_choice = "Mixed"

### Occupation convergence parameters
num_thermal_occ = 0
num_measure_occ = 1
measuring_interval_occ = 1
n_tol = 1e-4
max_iter = 50
num_uncorrelated_measurments = 1

# Magnetic field values for different L values
L_Bs = Dict(
    (16, 17) => 2*pi/(16*17) * [0:1:((16*17)÷2);],
    # (10,11) => 2*pi/(10*11) * [0:1:((10*11)÷2);],
)

# Build iterator - each item gets the FULL seeds array
iterate_over = Base.Iterators.flatten([
    IterTools.product([L], wanted_ns, betas, core_nums, Vs, impurities, seeds, diag_methods, TClustersizes, SC_Ls, Us, pairings, L_Bs[L]) for L in keys(L_Bs)
])

println("Number of simulation jobs to run: ", length(collect(iterate_over)))
println("Seeds per job: ", seeds_per_job)
println("Total simulations: ", length(collect(iterate_over)) * seeds_per_job)

for item in iterate_over
    println("\nSample simulation parameters: ")
    println("L, wanted_n, beta, cores, V, impurity, sim_seeds, diag_method, TClustersize, SC_L, U, pairing, B")
    println(item)
    println("Seeds to run: ", item[7])  # Show the seeds array
    break
end
end

sim_info = load_sim_info(path)

let i = 1 
    for (L, wanted_n, beta, cores, V, impurity, sim_seeds, diag_method, TClustersize, SC_L, U, pairing, B) in iterate_over
        Lx, Ly = L
        
        ### Check if simulation already exists
        skip = false
        for (sim_num, sim) in sim_info
            if sim_num == "updated_at"
                continue
            end
            if typeof(sim) === Dict{String, Any} && 
               sim["Lx"] == Lx && 
               sim["Ly"] == Ly && 
               sim["SC_L"] == SC_L && 
               sim["diag_method"] == diag_method && 
               sim["wanted_n"] == wanted_n && 
               sim["beta"] == beta && 
               sim["B"] == B &&
               sim["cores"] == cores && 
               sim["num_measure"] == num_measure && 
               sim["V"] == V && 
               sim["U_disorder"] == U_disorder && 
               sim["U"] == U && 
               sim["impurity"] == impurity && 
               sim["seeds"] == sim_seeds &&  # Compare full seeds array
               sim["TClustersize"] == TClustersize
                skip = true
                println("Skipping simulation for L=$L, SC_L=$SC_L, diag_method=$diag_method, wanted_n=$wanted_n, beta=$beta, cores=$cores, sim_num=$sim_num, num_measure=$num_measure")
                break
            end
            if U_disorder == "puddles" && U_puddle_params == sim["U_puddle_params"]
                skip = true
            end
        end

        if skip
            continue
        end

        c_sim_data = Dict(
                    "Lx" => Lx,
                    "Ly" => Ly,
                    "SC_L" => SC_L,
                    "U" => U,
                    "t" => t,
                    "V" => V,
                    "seeds" => sim_seeds,  # Full seeds vector
                    "wanted_n" => wanted_n,
                    "beta" => beta,
                    "B" => B,
                    "num_measure" => num_measure,
                    "num_thermal" => num_thermal,
                    "measuring_interval" => measuring_interval,
                    "w_list" => w_list,
                    "guess" => guess,
                    "U_disorder" => U_disorder,
                    "diag_method" => diag_method,
                    "cores" => cores,
                    "w_conductivity_range" => w_conductivity_range,
                    "U_puddle_params" => U_puddle_params,
                    "impurity" => impurity,
                    "TClustersize" => TClustersize,
                    "smooth_U" => smooth_U,
                    "pairing_symmetry" => pairing,
                    # Occupation convergence parameters
                    "num_thermal_occ" => num_thermal_occ,
                    "num_measure_occ" => num_measure_occ,
                    "measuring_interval_occ" => measuring_interval_occ,
                    "n_tol" => n_tol,
                    "max_iter" => max_iter,
                    "num_uncorrelated_measurments" => num_uncorrelated_measurments,
                    "boundary_condition" => boundary_condition,
                    "gauge_choice" => gauge_choice
                )

        c_path = joinpath(path, "$i")
        while isdir(c_path)
            i += 1
            c_path = joinpath(path, "$i")
        end
        mkpath(c_path)
    
        sim_file = joinpath(c_path, "sim.jld2")

        JLD2.save(sim_file, c_sim_data)
        slurm_file_name = joinpath(c_path, "slurm.txt")
        slurm_job_name = joinpath(sim_name, "n$(wanted_n)_Lx$(Lx)Ly$(Ly)U$(U)b$(beta)Lc$(TClustersize)_$i")
        open(slurm_file_name, "w") do slurm_file
            printfmt(slurm_file, templateSLURM, slurm_job_name, c_path, 1, cores, current_depot)
        end

        run(`sbatch $slurm_file_name`)
    end
end
