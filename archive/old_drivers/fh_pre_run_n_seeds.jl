@isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "..", "..", "cluster_config.jl"))   # V6_DATA_ROOT, JULIA_BIN
using JLD2, Formatting, Printf, Base.Iterators
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

run_file = joinpath(@__DIR__, "..", "..", "fh_run_n.jl")
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
# folder_name = "Disorder__26-04-2026"
# folder_name = "TCA_test__26-04-2026"
# folder_name= "test_27-04-2026"
folder_name = "Disorder__03-05-2026"
path = joinpath(out_dir, folder_name)
sim_name = "Disorder"
# sim_name = "test"

# Return true for disorder modes that need explicit U_puddle_params matching.
function disorder_uses_puddle_params(U_disorder::String)
    return U_disorder in ["puddles", "on-off-puddles", "gaussian", "impurity-gaussian-kernel"]
end

function same_puddle_params(a, b)
    try
        return collect(a) == collect(b)
    catch
        return a == b
    end
end

function has_matching_disorder_params(sim::Dict{String, Any}, U_disorder::String, U_puddle_params)
    if !disorder_uses_puddle_params(U_disorder)
        return true
    end
    if !haskey(sim, "U_puddle_params")
        return false
    end
    return same_puddle_params(sim["U_puddle_params"], U_puddle_params)
end

begin
using IterTools

# System parameters
Ls = [(4, 4)]
Us = [4.0]
t = 1.0
Vs = [0.0]
impurities = [1.0]
U_gaussian_sigma = 0.8

# Multi-seed campaign: one seed per simulation folder/job.
# p=1 (clean) rows use only seed_list[1]; p<1 rows use the full seed_list.
seeds_per_sim = 5
seed_list = Int64[]
while length(seed_list) < seeds_per_sim
    seed = rand(1:Int(10e6))
    if !(seed in seed_list)
        push!(seed_list, seed)
    end
end

# Target occupations
wanted_ns = [0.875]
# betas = [0.25:0.25:0.75; 1.0:1.0:3.0; 4.0:0.5:10.5; 11:1:17; 20; 25; 30;]
betas= [20.0]

### Simulation parameters (after convergence)
num_measure = 20 * 10^3
num_thermal = 5 * 10^3
measuring_interval = 25
w_list = [-4.0:0.1:4.0;]
guess = "uniform"
U_disorder = "impurity-gaussian-kernel"
# U_disorder = "uniform"
Bs = [0.0]

diag_methods = ["K0-TCA"]
pairings = ["s-wave"]
w_conductivity_range = collect(logrange(10^-3, 3, 12))
delta_fixed_values_mult = [0.5; 1:10;]
delta_stiffness_mult = []
dw_values_mult = [0.5; 1:10;]
U_puddle_params = [0.0, 4.0, U_gaussian_sigma, 0.7]
TClustersizes = [2]
SC_Ls = [1]
smooth_U = (false, 0.8)
boundary_condition = "PBC"
gauge_choice = "Mixed"

# Occupation convergence parameters
num_thermal_occ = 200
num_measure_occ = 100
measuring_interval_occ = 5
n_tol = 1e-3
max_iter = 100
num_uncorrelated_measurments = 16

# System size and Cluster size
L_Lc = Dict(
    # (6,6) => [6],
    # (8,8) => [8],
    # (10,10) => [8],
    # (12,12) => [4, 6, 8, 10, 12],
    # (14,14) => [8, 10, 12],
    # (20, 20) => [10],
    # (22, 22) => [10],
    # (24, 24) => [10, 12],
    # (26, 26) => [10, 12],
    # (28, 28) => [10, 12]

    # (12, 12)  => [10],
    # (16, 16) => [10],
    (18, 18) => [10],
    # (20, 20) => [10],
)

U_to_p = Dict(
    # for submission 19-04-2026. we want to have constant p*u= 2,4,5
    #U_mean = 2.0
    # 5.0 => [0.4],
    # 4.0 => [0.5],
    # 3.333 => [0.6],
    # 2.5 => [0.8],
    # # U_mean = 4.0
    # 20.0 => [0.2],
    # 8.0 => [0.5],
    # 6.666 => [0.6],
    # # U_mean = 5.0
    # 25.0 => [0.2],
    # 12.5 => [0.4],
    # 10.0 => [0.5],
    # 8.333 => [0.6],
    # 6.25 => [0.8],

    # for submission 26-04 - getting constant Tc
    # 9.2 => [0.6],
    # 7.3 => [0.8],
    # 4.7 => [0.6],
    # 3.75 => [0.8]

    # for submission 30-04-2026 - disorder phase-space data points
    # p = 1.0 (clean) -- 1 seed only
    # 2.5  => [1.0],
    # 8.0  => [1.0],
    # p < 1.0 (disordered) -- 3 seeds (× 2 clusters)
    # 30.0   => [0.2],
    # 40.0   => [0.2],
    # 12.0   => [0.5],
    # 16.0   => [0.5],
    # 13.333 => [0.6],
    # 6.0    => [0.5],
    # 7.5    => [0.4],
    # 15.0   => [0.2],
    # 10.0   => [0.45],

    # for submission 03-05
    # 6.0 => [1.0],
    # 7.3 => [0.8],
    # 9.2 => [0.6],
    # 10.0 => [1.0, 0.5, 0.45, 0.4],
    # 12.25 => [0.8],
    # 16.0 => [0.6],
    # 15.0 => [0.4],
    # 20.0 => [0.4],
    # 24.0 => [0.4]

    # 09-05
    10.0 => [1.0,0.5, 0.45, 0.5],
    # 9.2 => [0.6],
    40.0 => [0.2],
    16.0 => [0.6],
    24.0 => [0.4],
    12.0 => [0.8]
)

# U-dependent beta values (optional)
U_betas = Dict(
    # 10.0 => [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0,  6.0,  7.0,  8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0],
    # 6.0 => [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0,  6.0,  7.0,  8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0],
    # 6.0 => [0.25:0.25:0.75; 1.0:1.0:3.0; 4.0:0.5:10.5; 11:1:17; 20; 25; 30;]

    # # for submission 19-04-2026. we want to have constant p*u= 2,4,5
    # #U_mean = 2.0
    # 5.0 => [0.05:0.01:0.1;].^-1,
    # 4.0 => [0.05:0.01:0.1;].^-1,
    # 3.333 => [0.05:0.01:0.1;].^-1,
    # 2.5 => [0.05:0.01:0.1;].^-1,
    # # U_mean = 4.0
    # 20.0 => [0.09:0.01:0.15;].^-1, 
    # 8.0 => [0.09:0.01:0.15;].^-1,
    # 6.666 => [0.09:0.01:0.15;].^-1,
    # # U_mean = 5.0
    # 25.0 => [0.09:0.01:0.15;].^-1,
    # 12.5 => [0.09:0.01:0.15;].^-1,
    # 10.0 => [0.09:0.01:0.15;].^-1,
    # 8.333 => [0.09:0.01:0.15;].^-1,
    # 6.25 => [0.09:0.01:0.15;].^-1,

    # for submission 23-04-2026. we want to have constant p*u= 2,4,5
    #U_mean = 2.0
    # 5.0 => [0.12:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 4.0 => [0.12:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 3.333 => [0.12:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 2.5 => [0.12:0.02:0.22; 0.25:0.5:2.0].^-1,
    # # U_mean = 4.0
    # 20.0 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 8.0 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 6.666 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,
    # # U_mean = 5.0
    # 25.0 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 12.5 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 10.0 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 8.333 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,
    # 6.25 => [0.18:0.02:0.22; 0.25:0.5:2.0].^-1,

    # for 26-04 submission. TCA test
    # 4.0 => [2.0:1:10; 15; 20;]

    # for submission 26-04-2026. we want to have constant Tc
    # 9.2 => [0.25:0.25:0.75; 1.0:1.0:6.0; 7.0:0.5:9.0; 10; 15.0; 20;], #Tc about 0.1215 so beta_c about 8.2
    # 7.3 => [0.25:0.25:0.75; 1.0:1.0:6.0; 7.0:0.5:9.0; 10; 15.0; 20;], #Tc about 0.1215 so beta_c about 8.2
    # 4.7 => [0.25:0.25:0.75; 1.0:1.0:6.0; 7.0:0.5:9.0; 10; 15.0; 20;], #Tc about 0.1215 so beta_c about 8.2
    # 3.75 => [0.25:0.25:0.75; 1.0:1.0:8.0; 9.0:0.5:12.0; 13; 15.0; 20;], # Tc not known, but between 9.0 and 12.0

    # for submission 30-04-2026 - disorder phase-space data points
    # Tc range covered with T-step 0.005; "v" rows also get a diluted high-T tail [0.25:0.25:0.75; 1.0:1.0:6.0] for rho(T) linearity
    # p = 1.0 (clean)
    # 2.5  => [0.07:0.005:0.11;].^-1,                                                       # x:           ~Tc 0.07-0.11
    # 8.0  => [[0.25:0.25:0.75; 1.0:1.0:6.0]; [0.09:0.005:0.12;].^-1],                      # v + high T:  ~Tc 0.09-0.12
    # p < 1.0 (disordered)
    # 30.0   => [[0.25:0.25:0.75; 1.0:1.0:6.0]; [0.08:0.005:0.105;].^-1],                   # v + high T:  ~Tc 0.08-0.105
    # 40.0   => [[0.25:0.25:0.75; 1.0:1.0:6.0]; [0.07:0.005:0.1;].^-1],                     # v + high T:  ~Tc 0.07-0.1
    # 12.0   => [[0.25:0.25:0.75; 1.0:1.0:6.0]; [0.1:0.005:0.12;].^-1],                     # v + high T:  ~Tc 0.1-0.12
    # 16.0   => [[0.25:0.25:0.75; 1.0:1.0:6.0]; [0.08:0.005:0.1;].^-1],                     # v + high T:  ~Tc 0.08-0.1
    # 13.333 => [[0.25:0.25:0.75; 1.0:1.0:6.0]; [0.08:0.005:0.1;].^-1],                     # v + high T:  ~Tc 0.08-0.1
    # 6.0    => [0.1:0.005:0.115;].^-1,                                                     # x:           ~Tc 0.1-0.115
    # 7.5    => [0.09:0.005:0.11;].^-1,                                                     # x:           ~Tc 0.09-0.11
    # 15.0   => [0.07:0.005:0.11;].^-1,                                                     # x:           ~Tc 0.07-0.11
    # 10.0   => [[0.25:0.25:0.75; 1.0:1.0:6.0]; [0.115:0.005:0.125;].^-1],                  # v + high T:  ~Tc 0.115-0.125


    # for submission 03-05
    # 6.0 => [0.6:0.2:4.0;].^-1,
    # 7.3 => [0.6:0.2:4.0;].^-1,
    # 9.2 => [0.6:0.2:4.0;].^-1,
    # 10.0 => [0.6:0.2:4.0;].^-1,
    # 12.25 => [[0.08:0.005:0.1;].^-1; [0.6:0.2:4.0;].^-1],
    # 16.0 => [[0.08:0.005:0.1;].^-1; [0.6:0.2:4.0;].^-1],
    # 15.0 => [[0.095:0.005:0.115;].^-1; [0.6:0.2:4.0;].^-1],
    # 20.0 => [[0.085:0.005:0.1;].^-1; [0.6:0.2:4.0;].^-1],
    # 24.0 => [[0.08:0.005:0.1;].^-1; [0.6:0.2:4.0;].^-1]

    # 12.25 => [0.105:0.005:0.13;].^-1,
    # 16.0 => [0.105:0.005:0.13;].^-1,
    # 15.0 => [0.105:0.005:0.13;].^-1,
    # 20.0 => [0.105:0.005:0.13;].^-1,
    # 24.0 => [0.105:0.005:0.13;].^-1,

    # 09-05
    10.0 => [15;9;9.5;3;2.5;1.5;0.75;0.6;0.5; [0.6:0.2:4.0;].^-1],
    # 9.2 => [1.4,1.6].^-1,
    40.0 => [10.0, 9.0, 8.0, 7.0, 1.5],
    16.0 => [2.0:7.0;],
    24.0 => [2.0:7.0;],
    12.0 => [2.0:7.0;]
)

# Option 1: shared betas and Ls
# iterate_over = collect(IterTools.product(Ls, wanted_ns, betas, Vs, impurities, seed_list, diag_methods, TClustersizes, SC_Ls, Us, pairings, Bs))

# Option 2: U-dependent betas and L-dependent Lc
# iterate_over = collect(Base.Iterators.flatten([
#     IterTools.product([L], wanted_ns, U_betas[U], Vs, impurities, seed_list, diag_methods, L_Lc[L], SC_Ls, [U], pairings, Bs) for U in keys(U_betas) for L in keys(L_Lc)
# ]))
 # add also p=1.0 clean system with only one seed
# iterate_over_p1 = collect(Base.Iterators.flatten([
#     IterTools.product([L], wanted_ns, U_betas[U], Vs, [1.0], [seed_list[1]], diag_methods, L_Lc[L], SC_Ls, [U], pairings, Bs) for U in keys(U_betas) for L in keys(L_Lc)
# ]))
# iterate_over = vcat(iterate_over, iterate_over_p1)

# Option 3: U-dependent p and beta values, L-dependent Lc.
# Splits clean (p=1, single seed) from disordered (p<1, full seed_list).
clean_us    = [U for U in keys(U_to_p) if all(p == 1.0 for p in U_to_p[U])]
disorder_us = [U for U in keys(U_to_p) if all(p != 1.0 for p in U_to_p[U])]

iterate_over_clean = collect(Base.Iterators.flatten([
    IterTools.product([L], wanted_ns, U_betas[U], Vs, U_to_p[U], [seed_list[1]], diag_methods, L_Lc[L], SC_Ls, [U], pairings, Bs) for U in clean_us for L in keys(L_Lc)
]))
iterate_over_disorder = collect(Base.Iterators.flatten([
    IterTools.product([L], wanted_ns, U_betas[U], Vs, U_to_p[U], seed_list, diag_methods, L_Lc[L], SC_Ls, [U], pairings, Bs) for U in disorder_us for L in keys(L_Lc)
]))
# iterate_over = vcat(iterate_over_clean, iterate_over_disorder)
iterate_over = iterate_over_disorder

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
        delta_fixed_values = 4 / (Lx * Ly) .* delta_fixed_values_mult
        delta_values_stiffness = 4 / (Lx * Ly) .* delta_stiffness_mult
        dw_values = 4 / (Lx * Ly) .* dw_values_mult

        # Check if simulation already exists
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
               has_matching_disorder_params(sim, U_disorder, U_puddle_params)
                skip = true
                println("Skipping existing simulation for L=$L, SC_L=$SC_L, diag_method=$diag_method, wanted_n=$wanted_n, beta=$beta, seed=$seed")
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
        slurm_job_name = joinpath(sim_name, "n$(wanted_n)_Lx$(Lx)Ly$(Ly)U$(U)b$(beta)Lc$(TClustersize)_seed$(seed)_$i")
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
