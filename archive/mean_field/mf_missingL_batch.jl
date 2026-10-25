# Batch driver: run every prepared point in the missing-L MF sweep in ONE Julia
# session (avoids ~15s startup/compile per point), then aggregate means so the
# combined sim_info (with means["Ps"]) is written to <sweep>/num_control/.
#
#   GKSwstype=100 OPENBLAS_NUM_THREADS=1 julia mf_missingL_batch.jl
using Printf
include(joinpath(@__DIR__, "mf_kspace_cluster_run.jl"))   # run_point + helpers (no auto-main)
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))         # update_means

const SWEEP = joinpath(@__DIR__, "cluster_results", "mean_field", "mf_sweep_missingL_U4_n05")

point_dirs = sort(filter(d -> isdir(joinpath(SWEEP, d)) && tryparse(Int, d) !== nothing,
                         readdir(SWEEP)); by = x -> parse(Int, x))

println("=== Running $(length(point_dirs)) MF points in $SWEEP ===")
t0 = time()
for (i, d) in enumerate(point_dirs)
    pd = joinpath(SWEEP, d)
    try
        run_point(pd)                  # skips points already finished
    catch err
        println("FAIL point $d :: ", sprint(showerror, err))
    end
    @printf("[%d/%d] point %s done (elapsed %.1f s)\n", i, length(point_dirs), d, time() - t0)
end
println("=== All points run in $(round(time()-t0; digits=1)) s; aggregating means ===")

update_means(SWEEP)   # writes <SWEEP>/num_control/sim_info.jld2 with means (incl. Ps)
println("=== DONE: means written to $(joinpath(SWEEP, "num_control", "sim_info.jld2")) ===")
