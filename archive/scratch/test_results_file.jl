include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

# file_path = "<Curie folder>\\MC_Hubbard_V6\\benchmark_runtime_tau\\runtime_tau_large_pilot_28-03-2026\\12\\seed_22345"
file_path = joinpath(get(ENV, "CURIE_ROOT", ""), "MC_Hubbard_V6", "data", "Full_Sys_TCA__25-02-2026", "306", "seed_7523277")

res = load_results(file_path);
meas = get_means(res);