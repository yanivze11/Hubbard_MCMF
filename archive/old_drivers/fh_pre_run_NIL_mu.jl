@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2, Formatting, Printf, Base.Iterators
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

run_file = joinpath(pwd(), "fh_run.jl")
julia_bin = JULIA_BIN
partitions = get(ENV, "PARTITIONS", "allq,infiniq")
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")

######################### Generate slurm.txt #################################
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
folder_name = "NIL_mu__11-01-2026"
path = joinpath(out_dir, folder_name)
sim_name = "NIL"


begin
using IterTools

# System parameters
Ls = [(16,17)]  # Lattice sizes to iterate over
Us = [0.0]  # Interaction strengths
t = 1.0  # Hopping parameter
Vs = [0.0]  # Disorder strength for chemical potential
impurities = [0.0]  # Impurity strength for interaction

# Seed configuration
seeds_per_sim = 1
jobs_per_sim = 1
seeds = [Int64.(rand(1:10e6, seeds_per_sim ÷ jobs_per_sim)) for _ in 1:jobs_per_sim]

# Chemical potential and temperature
mus = [-1.0, 0.0]  # Chemical potential values
# betas = [0.5, 1.0, 2.0, 3.0,4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]  # Inverse temperatures
betas = [30.0]

### Simulation parameters
num_measure = 1
num_thermal = 0
measuring_interval = 1
w_list = [-4.5:0.1:4.5;]
guess = "uniform"
U_disorder = "uniform"  # Type of U disorder

# Method parameters
diag_methods = ["K0"]  # Diagonalization methods
pairings = ["s-wave"]  # Pairing symmetries
boundary_condition = "periodic"  # Boundary condition
gauge_choice = "Mixed"  # Gauge choice
core_nums = [1]
w_conductivity_range = [0.0:0.001:0.1; 0.12:0.01:1.0; 1.1:0.1:2.0;]  # Frequencies for conductivity
U_puddle_params = [0.0, 12.0, 0.7, 0.7]  # (V0, VI, lambda_U, rz) for puddles
TClustersizes = [4]  # TCluster sizes
SC_Ls = [1]  # Supercell sizes
smooth_U = (true, 0.8)  # (smooth_U, sigma) for smoothing

# System size and cluster size mapping
L_Lc = Dict(
    10 => [10],
)

# Define beta-specific mu ranges (optional, for fine control)
beta_mu_ranges = Dict(
    0.1 => [-3.0:0.1:-2.7;],
    0.2 => [-3.0:0.1:-2.7;],
    0.4 => [-3.0:0.1:-2.7;],
    1.0 => [-3.0:0.1:-2.5;],
)

 # magnetic field values for each L
L_Bs = Dict(

    # (10, 11) => 2*pi/(10*11) * [0:1:((10*11)÷2);],
    # (12, 13) => 2*pi/(12*13) * [0:1:((12*13)÷2);],
    (16, 17) => 2*pi/(16*17) * [0:1:((16*17)÷2);],
)

# Build the iterate_over iterator
# Option 1: Use beta-specific mu ranges
iterate_over = Base.Iterators.flatten([
    IterTools.product([L], mus, betas, core_nums, Vs, impurities, seeds, diag_methods, TClustersizes, SC_Ls, Us, pairings, L_Bs[L]) for L in keys(L_Bs)
])

# Option 2: Use same mu values for all betas (uncomment to use)
# iterate_over = Base.Iterators.flatten([
#     IterTools.product([L], mus, betas, core_nums, Vs, impurities, seeds, diag_methods, L_Lc[L], SC_Ls, Us, pairings) 
#     for L in keys(L_Lc)
# ])

println("Number of simulations to run: ", length(collect(iterate_over)))

for item in iterate_over
    println("Sample simulation parameters: ")
    println("L, mu, beta, cores, V, impurity, sim_seeds, diag_method, TClustersize, SC_L, U, pairing, B")
    println(item)
    break
end
end

sim_info = load_sim_info(path)

let i = 1 
    for (L, mu, beta, cores, V, impurity, sim_seeds, diag_method, TClustersize, SC_L, U, pairing, B) in iterate_over
        Lx, Ly = L
        ### Check if simulation already exists
        skip = false
        for (sim_num, sim) in sim_info
            if typeof(sim) === Dict{String, Any} && 
               sim["L"] == L && 
               sim["SC_L"] == SC_L && 
               sim["diag_method"] == diag_method && 
               sim["mu"] == mu && 
               sim["beta"] == beta && 
               sim["cores"] == cores && 
               sim["num_measure"] == num_measure && 
               sim["V"] == V && 
               sim["U_disorder"] == U_disorder && 
               sim["U"] == U && 
               sim["impurity"] == impurity && 
               sim["seeds"] == sim_seeds && 
               sim["TClustersize"] == TClustersize &&
               sim["B"] == B
                skip = true
                println("Skipping simulation for L=$L, SC_L=$SC_L, diag_method=$diag_method, mu=$mu, beta=$beta, cores=$cores, sim_num=$sim_num, num_measure=$num_measure")
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
            "seeds" => sim_seeds,
            "mu" => mu,
            "beta" => beta,
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
            "B" => B,
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
        slurm_job_name = joinpath(sim_name, "Lx$(Lx)Ly$(Ly)U$(U)b$(beta)mu$(mu)Lc$(TClustersize)_$i")
        open(slurm_file_name, "w") do slurm_file
            printfmt(slurm_file, templateSLURM, slurm_job_name, c_path, 1, cores, current_depot)
        end

        run(`sbatch $slurm_file_name`)
    end
end