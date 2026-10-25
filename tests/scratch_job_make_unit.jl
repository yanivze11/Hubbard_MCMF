# tests/scratch_job_make_unit.jl — write a tiny, fast unit (sim.jld2) for the
# JOB_MODE=scratch tests. Not registered in any means_db.
#
#   julia tests/scratch_job_make_unit.jl <unit_dir> [L] [num_measure] [measuring_interval]
#
# Same keys as fh_pre_run_V.jl build_params, so fh_run_n.jl runs it unchanged.
using JLD2

function make_unit(dir::AbstractString; L::Int=4, num_measure::Int=1200, measuring_interval::Int=2)
    isfile(joinpath(dir, "sim.jld2")) && error("$dir already has a sim.jld2")
    mkpath(dir)
    p = Dict{String,Any}(
        "Lx" => L, "Ly" => L, "SC_L" => 1, "U" => 4.0, "t" => 1.0, "V" => 0.0,
        "disorder_seed" => 1, "alpha" => 1.0, "beta_phi" => 0.0, "Qx" => 0.0, "Qy" => 0.0,
        "seeds" => Int64[1], "wanted_n" => 0.875, "beta" => 5.0, "B" => 0.0,
        "num_measure" => num_measure, "num_thermal" => 200,
        "measuring_interval" => measuring_interval,
        "w_list" => collect(range(-5.0, 5.0, length=11)), "guess" => "uniform",
        "U_disorder" => "uniform", "diag_method" => "K0", "cores" => 1,
        "w_conductivity_range" => [0.1, 0.5, 1.0],
        "U_puddle_params" => [0.0, 6.0, 0.7, 0.7], "impurity" => 0.0,
        "TClustersize" => 8, "smooth_U" => (false, 0.8), "pairing_symmetry" => "s-wave",
        "num_thermal_occ" => 50, "num_measure_occ" => 20, "measuring_interval_occ" => 2,
        "n_tol" => 0.05, "max_iter" => 8, "num_uncorrelated_measurments" => 16,
        "boundary_condition" => "PBC", "gauge_choice" => "Mixed",
        "delta_fixed_values" => 4 / (L * L) .* [0.5, 1.0],
        "delta_values_stiffness" => [], "dw_values" => [],
    )
    JLD2.save(joinpath(dir, "sim.jld2"), p)
    return dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("usage: julia scratch_job_make_unit.jl <unit_dir> [L] [num_measure] [measuring_interval]")
    L  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    nm = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1200
    mi = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2
    make_unit(ARGS[1]; L=L, num_measure=nm, measuring_interval=mi)
    println("made $(ARGS[1])  (L=$L, num_measure=$nm, measuring_interval=$mi)")
end
