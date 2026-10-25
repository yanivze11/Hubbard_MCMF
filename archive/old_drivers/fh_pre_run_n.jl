@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2, Formatting, Printf, Base.Iterators
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

run_file = joinpath(@__DIR__, "..", "..", "fh_run_n.jl")
julia_bin = JULIA_BIN
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")
exclude_nodes = get(ENV, "EXCLUDE_NODES", "")
cpus_per_task = parse(Int, get(ENV, "CPUS_PER_TASK", "1"))
partitions = get(ENV, "PARTITIONS", "allq,infiniq")

# exclude_nodes_landau = ["landau59"]  # Add problematic nodes here
# exclude_nodes_curie = ["curie25"]  # Add problematic nodes here
# hostname = gethostname()
# if contains(hostname, "landau")
#     exclude_nodes = exclude_nodes_landau
# elseif contains(hostname, "curie")
#     exclude_nodes = exclude_nodes_curie
# else
#     error("Unknown cluster! Hostname is: $hostname")
# end

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
#SBATCH --exclude=$exclude_nodes 
#SBATCH --chdir={2}

#SBATCH --array=1-{3}
    
export DIR={2}
export JULIA_DEPOT_PATH={5}  
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1  # or MKL equivalent, to avoid oversubscription
srun --chdir=\$DIR $julia_bin $run_file --sim_path=\$DIR

""")
###############################################################################

out_dir = joinpath(V6_DATA_ROOT, "data")
folder_name = "FullSys_TCA__04-04-2026"
path = joinpath(out_dir, folder_name)
# sim_name = "SW"
sim_name = "FullSys_0404"

begin
using IterTools

# System parameters
Ls = [(4,4), (6,6)] # Lattice sizes to iterate over
Us = [4.0]
t = 1.0
Vs = [0.0] # disorder strength for chemical potential
# impurities = [0.2, 0.5, 0.7, 1.0] # impurity strength for interaction - for on-off puddles is covering precent
impurities = [0.0] # impurity strength for interaction
seeds_per_sim = 1; # number of seeds per simulation
jobs_per_sim = 1; # number of parallel jobs per simulation
seeds = [Int64.(rand(1:10e6, seeds_per_sim÷jobs_per_sim)) for _ in 1:jobs_per_sim] # same seeds for each sim

# Target occupations instead of chemical potentials
wanted_ns = [0.5, 0.875]  # Target electron occupations
# betas = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0]  # Inverse temperatures
# betas = [1.5, 2.5, 3.0, 3.5, 5.5, 6.5, 7.5, 8.5, 9.5]
# betas = [0.5, 1.0, 2.0, 3.0, 3.5 ,4.0, 4.5, 5.0,5.5, 6.0, 6.5, 7.0,  7.5,8.0, 8.5, 9.0, 9.5, 10.0, 20.0]
# betas = [12.0, 15.0, 18.0, 25.0, 30.0, 35.0, 40.0, 50.0]
# betas = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 18.0, 20.0, 30.0]
# betas = [0.25:0.25:0.75; 1.0:1.0:6.0; 7.0:0.5:13.0; 14:1:22;]
# betas = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 50.0]
betas = [0.25:0.25:0.75; 1.0:1.0:6.0; 7.0:0.5:11.0; 12:1:17; 20; 25; 30;]

# betas = [20.0]

### Simulation parameters (after convergence)
num_measure = 20 * 10^3;
num_thermal = 5 * 10^3;
measuring_interval = 20;
w_list = [-4.0:0.1:4.0;];
guess = "uniform"
U_disorder = "uniform" # Type of U disorder
# Bs = round.(2*pi/10 * [1:2:5;], digits=3) # Magnetic field
# Bs = 2*pi/(10*11) * [0:1:9; 10:2:18; 20:5:30; 40; 50]  # Magnetic field values for Lx,Ly=10,11
Bs = [0.0]


diag_methods = ["K0-TCA"] # Diagonalization methods to iterate over
pairings = ["s-wave"] # Pairing symmetries to iterate over
# core_nums = [cpus_per_task]  # Number of CPU cores to use
# w_conductivity_range = [10^a for a in range(-3, 0.5, 10)] # Range of frequencies for conductivity calculations
# w_conductivity_range = [10^a for a in -3:0.1:0.5] # Range of frequencies for conductivity calculations
w_conductivity_range = collect(logrange(10^-3, 3, 12))
# delta_fixed_values = [] # Fixed delta values to iterate over
delta_fixed_values_mult = [0.5; 1:10;]  # Multiplicative factors for delta_fixed based on max delta
delta_stiffness_mult = [0.0; 0.5; 1:10;]
dw_values_mult = [0.5; 1:10;]  # Multiplicative factors for dw_values based on max dw
U_puddle_params = [0.0, 4.0, 1.0, 0.7] # (V0, VI, lambda_U, rz) for U_disorder = puddles , for on-off puddles: (not used, not used, puddle_size, not used)
TClustersizes = [2] # Size of TClusters for TCluster diag_method
SC_Ls = [1] # Supercell sizes to iterate over
smooth_U = (false, 0.8) # (smooth_U, sigma) for smoothing U disorder
boundary_condition = "PBC"  # Boundary condition type
gauge_choice = "Mixed"  # Gauge choice for magnetic field
# Occupation convergence parameters
num_thermal_occ = 200  # Thermal steps during occupation convergence
num_measure_occ = 100   # Measurement steps during occupation convergence
measuring_interval_occ = 5  # Measuring interval during convergence
n_tol = 1e-3  # Tolerance for occupation convergence
max_iter = 100  # Maximum iterations for occupation convergence
num_uncorrelated_measurments = 16  # Number of uncorrelated measurements for convergence

# system size and Cluster size
L_Lc = Dict(
        # (8, 8) => [8],
        # (10, 10)=>[8, 10],
        # (12, 12)=>[8, 10],
        # (14, 14)=>[8, 10],
        # (16, 16)=>[8, 10],
        # (18, 18) => [8, 10],
        # (20,20) => [8, 10],

        # for specific extra points
        (4,4) => [4],
        (6,6) => [6],
        (8,8) => [8],
        (14, 14) => [10],
        (16, 16) => [10],
        # (18, 18) => [10],
        # (20,20) => [10],
)

# Define a dictionary mapping each beta to its corresponding wanted_n range (optional)
# If you want different n values for different temperatures
beta_n_ranges = Dict(
    # 1.0 => [0.5, 0.75, 0.85, 0.9, 0.95, 1.0],
    # 4.0 => [0.5, 0.75, 0.85, 0.9, 0.95, 1.0],
    # 8.0 => [0.5, 0.75, 0.85, 0.9, 0.95, 1.0],
    # 10.0 => [0.5, 0.75, 0.85, 0.9, 0.95, 1.0],
    # 20.0 => [0.5, 0.75, 0.85, 0.9, 0.95, 1.0],
)

# U-dependent beta values (optional)
# If you want different inverse temperatures for different U values
U_betas = Dict(
    # 2.0 => [0.25:0.25:0.75; 1.0:1.0:11.0; 12.0:0.5:20.0; 22:2:26; 30; 35; 40;],
    # 3.0 =>   [0.25:0.25:0.75; 1.0:1.0:6.0; 7.0:0.5:13.0; 14:1:22;],
    # 4.0 =>   [0.25:0.25:0.75; 1.0:1.0:20.0; 25; 30; 35; 40;],
    # 5.0 =>  [0.25:0.25:0.75; 1.0:1.0:6.0; 7.0:0.5:11.0; 12:1:17; 20; 25; 30;35; 40;]

    # for specific extra points
    # 5.0 =>  [25.0; 16.0; 11.0; 8.0:-0.5:4.0;]
    # high T points for U=3,5
    3.0 => [3.3, 1.8, 1.25, 0.9, 0.8, 0.65, 0.6, 0.55, 0.4, 0.35],
    5.0 => [2.5, 1.54, 1.1, 0.9, 0.7, 0.6, 0.45, 0.4, 0.37, 0.35, 0.32, 0.3, 0.27, 0.26]
)

# Magnetic field values for different L values (optional)
L_Bs = Dict(
    # (8, 9) => 2*pi/(8*9) * [0:1:36;],
    # (10, 11) => 2*pi/(10*11) * [0:1:24; 25:2:55;],
    # (10, 11) => 2*pi/(10*11) * [9; 10; 11; 13;14; 15; 16; 21:24; 26:2:54; 41; 39],
    # (12, 13) => 2*pi/(12*13) * [0:1:40; 45:5:75;78;],
    # (12, 13) => 2*pi/(12*13) * [0:40;],
    # (12, 13) => 2*pi/(12*13) * [41:78;],
    # (14, 15) => 2*pi/(14*15) * [0:1:20; 22:2:50; 55:10:105;],
    # (16, 17) => 2*pi/(16*17) * [0:1:70; 45:5:135;136;],
    # (4, 5) => 2*pi/(4*5) * [0:10;]

    # only B=0 and B=1*2pi/(Lx*Ly)
    (8, 9) => 2*pi/(8*9)* [0, 1, 2],
    (10, 11) => 2*pi/(10*11)* [0, 1, 2, 3],
    (12, 13) => 2*pi/(12*13)* [0, 1, 2, 4],
    (14, 15) => 2*pi/(14*15)* [0, 1, 2, 6],
)

# Build the iterate_over iterator
# Option 1: Use beta-specific n ranges
# iterate_over = Base.Iterators.flatten([
#     IterTools.product([L], beta_n_ranges[beta], [beta], core_nums, Vs, impurities, seeds, diag_methods, L_Lc[L], SC_Ls, Us, pairings, Bs) for beta in keys(beta_n_ranges) for L in keys(L_Lc)
# ])

# Option 2: Use same n values for all betas (uncomment to use)
# iterate_over = Base.Iterators.flatten([
#     IterTools.product([L], wanted_ns, betas, Vs, impurities, seeds, diag_methods, L_Lc[L], SC_Ls, Us, pairings, Bs) for L in keys(L_Lc)
# ])

# option 3: Use same n and Lc values for all betas
# iterate_over = Base.Iterators.flatten([
#     IterTools.product(Ls, wanted_ns, betas, Vs, impurities, seeds, diag_methods, TClustersizes, SC_Ls, Us, pairings, Bs) ])

### option 4: Use L-dependent B values
# iterate_over = Base.Iterators.flatten([
#     IterTools.product([L], wanted_ns, betas,  Vs, impurities, seeds, diag_methods, TClustersizes, SC_Ls, Us, pairings, L_Bs[L]) for L in keys(L_Bs)
# ])

# option 5: Add V=0 cases, but with a reduced set of seeds (e.g., only the first seed for each jobs_per_sim)
### original case
# iterate_over = Base.Iterators.flatten([
#     IterTools.product([L], wanted_ns, betas,  Vs, impurities, seeds, diag_methods, TClustersizes, SC_Ls, Us, pairings, L_Bs[L]) for L in keys(L_Bs)
# ])
### add V=0 case with reduced seeds
# reduced_seeds = [seeds[1]]  # Use only the first seed group for V=0
# iterate_over = IterTools.chain(
#     iterate_over,
#     Base.Iterators.flatten([
#         IterTools.product([L], wanted_ns, betas, [0.0], impurities, reduced_seeds, diag_methods, TClustersizes, SC_Ls, Us, pairings, L_Bs[L]) for L in keys(L_Bs)
#     ])
# )

### option 6: Use U-dependent beta values (different betas for different Us)
# iterate_over = Base.Iterators.flatten([
#     IterTools.product(Ls, wanted_ns, U_betas[U], Vs, impurities, seeds, diag_methods, TClustersizes, SC_Ls, [U], pairings, Bs) for U in keys(U_betas) 
# ])

### option 7: Use U-dependent beta values and L dependent Lc values
iterate_over = Base.Iterators.flatten([
    IterTools.product([L], wanted_ns, U_betas[U], Vs, impurities, seeds, diag_methods, L_Lc[L], SC_Ls, [U], pairings, Bs) for U in keys(U_betas) for L in keys(L_Lc)
])

println("Number of simulations to run: ", length(collect(iterate_over)))
println("Sample simulation parameters: ")
println("L, wanted_n, beta, V, impurity, sim_seeds, diag_method, TClustersize, SC_L, U, pairing, B")

for item in iterate_over

    println(item)
    break
end
end

sim_info = load_sim_info(path)

let i = 1 
    for (L, wanted_n, beta, V, impurity, sim_seeds, diag_method, TClustersize, SC_L, U, pairing, B) in iterate_over
        Lx, Ly = L
        delta_fixed_values = 4/(Lx*Ly) .* delta_fixed_values_mult
        delta_values_stiffness = 4/(Lx*Ly) .* delta_stiffness_mult
        dw_values = 4/(Lx*Ly) .* dw_values_mult
        # w_conductivity_range = 4/(Lx*Ly) .* w_conductivity_range_mult
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
               sim["cores"] == cpus_per_task && 
               sim["num_measure"] == num_measure && 
               sim["V"] == V && 
               sim["U_disorder"] == U_disorder && 
               sim["U"] == U && 
               sim["impurity"] == impurity && 
               sim["seeds"] == sim_seeds && 
               sim["TClustersize"] == TClustersize
                skip = true
                println("Skipping simulation for L=$L, SC_L=$SC_L, diag_method=$diag_method, wanted_n=$wanted_n, beta=$beta, cores=$cpus_per_task, sim_num=$sim_num, num_measure=$num_measure")
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
                    "wanted_n" => wanted_n,  # Target occupation instead of mu
                    "beta" => beta,
                    "B" => B,
                    "num_measure" => num_measure,
                    "num_thermal" => num_thermal,
                    "measuring_interval" => measuring_interval,
                    "w_list" => w_list,
                    "guess" => guess,
                    "U_disorder" => U_disorder,
                    "diag_method" => diag_method,
                    "cores" => cpus_per_task,
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
                    "gauge_choice" => gauge_choice,
                    "delta_fixed_values" => delta_fixed_values,
                    "delta_values_stiffness" => delta_values_stiffness,
                    "dw_values" => dw_values
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
            printfmt(slurm_file, templateSLURM,
             slurm_job_name, # {1} SLURM job name
              c_path, # {2} Directory path
               1,   # {3} Number of array jobs
                cpus_per_task, # {4} CPUs per task
                 current_depot # {5} JULIA_DEPOT_PATH
            )
        end

        run(`sbatch $slurm_file_name`)
    end
end
