@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2, Formatting, Printf, Base.Iterators
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

run_file = joinpath(pwd(), "fh_run_n.jl")
julia_bin = JULIA_BIN
current_depot = get(ENV, "JULIA_DEPOT_PATH", "")
exclude_nodes = get(ENV, "EXCLUDE_NODES", "")
cpus_per_task = parse(Int, get(ENV, "CPUS_PER_TASK", "1"))
partitions = get(ENV, "PARTITIONS", "allq,infiniq")

######################### Generate slurm.txt #################################
templateSLURM = FormatExpr("""#!/bin/bash

#SBATCH --job-name={1}
#SBATCH --output={2}/%A_%a.out
#SBATCH --error={2}/%A_%a.error
#SBATCH --threads-per-core=1
#SBATCH -c {4}
#SBATCH -n 1
#SBATCH --mem-per-cpu=4GB
#SBATCH --requeue

#SBATCH -A all-account
#SBATCH -p $partitions
#SBATCH --exclude=$exclude_nodes
#SBATCH --chdir={2}

#SBATCH --array=1-{3}

export DIR={2}
export JULIA_DEPOT_PATH={5}
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
srun --chdir=\$DIR $julia_bin $run_file --sim_path=\$DIR

""")
###############################################################################

out_dir = joinpath(V6_DATA_ROOT, "data")
folder_name = "Mag_new__12-05-2026"
path = joinpath(out_dir, folder_name)
sim_name = "Mag_new"

begin
using IterTools

# System parameters
Ls = [(12,13), (14,15)]
Us = [3.0, 4.0]
t = 1.0
Vs = [0.0]

# One seed per simulation (clean run, no seed averaging)
seeds_per_sim = 1
seed_list = Int64[]
while length(seed_list) < seeds_per_sim
    seed = rand(1:Int(10e6))
    if !(seed in seed_list)
        push!(seed_list, seed)
    end
end

# Target occupation
wanted_ns = [0.875]

# β grid: 19 values, β = 2, 3, ..., 20  ⇔  T = 0.5, 0.333, ..., 0.05
# betas = [0.25; 0.5; 1.0:10.0; 12.0; 15.0; 20.0; 25.0; 30.0;]
# betas = [0.25, 0.5, 0.75, 1.5, 25.0, 30.0]
betas = [2.0:10.0; 12.0;15;] 

# Magnetic flux quantization on the (Lx, Ly) torus: B = 2π·n/(Lx·Ly).
# n = 0..9 (consecutive) then n = 11, 13, ..., 55 (steps of 2). n=55 ⇒ B=π.
Lx_only, Ly_only = Ls[1]
B_quantum  = 2π / (Lx_only * Ly_only)
# B_n_values = vcat(collect(0:9), collect(11:2:55))   # 33 values
B_n_values = collect(0:10)
Bs         = B_quantum .* B_n_values                # 0 → π

# make Bs_L_dict
Bs_L_dict = Dict{Tuple{Int, Int}, Vector{Float64}}()
for L in Ls
    Lx, Ly = L
    B_quantum = 2π / (Lx * Ly)
    max_n = floor(Int, (Lx * Ly) / 2)  # maximum n for B ≤ π
    B_n_values = vcat(collect(0:5))   # 33 values
    Bs_L_dict[L] = B_quantum .* B_n_values
end


### Simulation parameters (after convergence) -- defaults from fh_pre_run_n_seeds.jl
num_measure = 20 * 10^3
num_thermal = 1 * 10^4
measuring_interval = 25
w_list = [-4.0:0.1:4.0;]
guess = "uniform"
U_disorder = "uniform"   # clean system

diag_methods = ["K0"]    # MCMF; TCA (K0-TCA) is not compatible with B ≠ 0
pairings = ["s-wave"]
w_conductivity_range = collect(logrange(10^-3, 3, 12))

# δ grid for transport extrapolation: 4/(Lx·Ly) .* [0.5; 1:10;] = 11 points
delta_fixed_values_mult = [0.5; 1:10;]
delta_stiffness_mult    = []
dw_values_mult          = [0.5; 1:10;]

# Clean-run placeholders (unused when U_disorder == "uniform")
U_puddle_params = [0.0, 4.0, 0.8, 0.7]
smooth_U = (false, 0.8)

TClustersizes = [4]      # matches existing K0 magnetic-field runs (files 1/5/6/7)
SC_Ls = [1]
boundary_condition = "PBC"
gauge_choice = "Mixed"
impurity_clean = 0.0     # clean → impurity field is 0.0 in existing magnetic-field jld2

# Occupation convergence parameters
num_thermal_occ = 200
num_measure_occ = 100
measuring_interval_occ = 5
n_tol = 1e-3
max_iter = 100
num_uncorrelated_measurments = 16

# oprion 1:
# iterate_over = collect(IterTools.product(
#     Ls, wanted_ns, betas, Vs, [impurity_clean], seed_list,
#     diag_methods, TClustersizes, SC_Ls, Us, pairings, Bs,
# ))

# option 2: iterate over all combinations of L, beta, B, and seed, while keeping other parameters fixed
iterate_over = collect(Base.Iterators.flatten([
    IterTools.product([L], wanted_ns, betas, Vs, [impurity_clean], seed_list, diag_methods, TClustersizes, SC_Ls, Us, pairings, Bs_L_dict[L]) for L in keys(Bs_L_dict)
]))

println("Number of simulations to run: ", length(iterate_over))
println("Sample simulation parameters:")
println("L, wanted_n, beta, V, impurity, seed, diag_method, TClustersize, SC_L, U, pairing, B")
if !isempty(iterate_over)
    println(iterate_over[1])
end
end

sim_info = load_sim_info(path)

let i = 1
    for (L, wanted_n, beta, V, impurity, seed, diag_method, TClustersize, SC_L, U, pairing, B) in iterate_over
        Lx, Ly = L
        delta_fixed_values     = 4 / (Lx * Ly) .* delta_fixed_values_mult
        delta_values_stiffness = 4 / (Lx * Ly) .* delta_stiffness_mult
        dw_values              = 4 / (Lx * Ly) .* dw_values_mult

        # Skip if a matching simulation already exists in this folder
        skip = false
        for (sim_num, sim) in sim_info
            if sim_num == "updated_at"
                continue
            end
            if !(sim isa Dict{String, Any})
                continue
            end

            if get(sim, "Lx", nothing) == Lx &&
               get(sim, "Ly", nothing) == Ly &&
               get(sim, "SC_L", nothing) == SC_L &&
               get(sim, "diag_method", nothing) == diag_method &&
               get(sim, "wanted_n", nothing) == wanted_n &&
               get(sim, "beta", nothing) == beta &&
               get(sim, "B", nothing) == B &&
               get(sim, "cores", nothing) == cpus_per_task &&
               get(sim, "num_measure", nothing) == num_measure &&
               get(sim, "V", nothing) == V &&
               get(sim, "U_disorder", nothing) == U_disorder &&
               get(sim, "U", nothing) == U &&
               get(sim, "impurity", nothing) == impurity &&
               get(sim, "seeds", Int64[]) == Int64[seed] &&
               get(sim, "TClustersize", nothing) == TClustersize &&
               get(sim, "gauge_choice", nothing) == gauge_choice
                skip = true
                println("Skipping existing simulation: L=$L, beta=$beta, B=$B, seed=$seed")
                break
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
            "seeds" => Int64[seed],
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
            # Transport-control parameters
            "delta_fixed_values" => delta_fixed_values,
            "delta_values_stiffness" => delta_values_stiffness,
            "dw_values" => dw_values,
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
        slurm_job_name = joinpath(
            sim_name,
            "n$(wanted_n)_Lx$(Lx)Ly$(Ly)U$(U)b$(beta)B$(round(B; digits=4))_seed$(seed)_$i",
        )
        open(slurm_file_name, "w") do slurm_file
            printfmt(
                slurm_file,
                templateSLURM,
                slurm_job_name,
                c_path,
                1,
                cpus_per_task,
                current_depot,
            )
        end

        run(`sbatch $slurm_file_name`)
    end
end
