# occupation_convergence.jl — run a simulation at a fixed filling n instead of a fixed μ.
#
# Before the production run, a short Monte-Carlo search finds the chemical potential μ
# that gives the wanted <n> (false-position / bisection on μ, with error bars on each
# <n>). The search state is saved in sim.jld2 so a preempted job resumes it, and the
# last search configuration is reused as the thermalized start of the production run.
# Entry points: `running_simulation_wanted_occupation` (one seed) and
# `running_simulation_seeds_wanted_occupation` (several seeds, one after another).

include("MC_Hubburd.jl")
include("statistics.jl")

using Plots

"""
    OccupationConvergence

State of the μ search for a target filling `target_n`.

`mu_edges`/`n_edges` are the current bracket (μ and the <n> measured there),
`mu_list`/`n_list` every μ tried so far and its measured <n>. The search stops when
|<n> - target_n| < `n_tol` (or is within the error bar), or after `max_iter` steps.
`num_thermal_occ`/`num_measure_occ` are the MC steps used for each trial μ.
`best_mu`/`best_n` hold the result once the search is finished.
"""
mutable struct OccupationConvergence
    target_n::Float64
    sim_params::SimData
    mu_edges::Tuple{Float64, Float64}
    n_edges::Tuple{Float64, Float64}
    mu_list::Vector{Float64}
    n_list::Vector{Float64}
    n_tol::Float64
    max_iter::Int64
    num_thermal_occ::Int64
    num_measure_occ::Int64
    converged::Bool
    best_mu::Union{Nothing, Float64}
    best_n::Union{Nothing, Float64}
end

"""
    Occupation_Convergence(target_n, sim_params; mu_edges, n_edges=(0.0, 1.0), n_tol=1e-3,
                           max_iter=20, num_thermal_occ=1000, num_measure_occ=100)

Create a fresh `OccupationConvergence` (empty history, not converged) for `target_n`.
`sim_params` is the `SimData` used for the trial runs; its `mu` is overwritten at each step.
"""
function Occupation_Convergence(target_n::Float64, sim_params::SimData;
    mu_edges::Tuple{Float64, Float64}=(-4.0, target_n > 0.9 ? 0.1 : 0.0),
    n_edges::Tuple{Float64, Float64}=(0.0, 1.0),
    n_tol::Float64=1e-3,
    max_iter::Int64=20,
    num_thermal_occ::Int64=1000,
    num_measure_occ::Int64=100)
    
    mu_list = Float64[]
    n_list = Float64[]
    
    return OccupationConvergence(
        target_n,
        sim_params,
        mu_edges,
        n_edges,
        mu_list,
        n_list,
        n_tol,
        max_iter,
        num_thermal_occ,
        num_measure_occ,
        false,
        nothing,
        nothing
    )
end

# ============================================================================
# Convergence State Save/Load Helpers
# ============================================================================

"""
    save_convergence_state(sim_info_path::String, occ_conv::OccupationConvergence)

Save convergence state to sim.jld2 for resume capability.
"""
function save_convergence_state(sim_info_path::String, occ_conv::OccupationConvergence)
    if !isfile(sim_info_path)
        sim_info = Dict{String, Any}()
    else
        sim_info = JLD2.load(sim_info_path)
    end
    
    sim_info["convergence"] = Dict{String, Any}(
        "target_n" => occ_conv.target_n,
        "mu_edges" => occ_conv.mu_edges,
        "n_edges" => occ_conv.n_edges,
        "mu_list" => occ_conv.mu_list,
        "n_list" => occ_conv.n_list,
        "n_tol" => occ_conv.n_tol,
        "max_iter" => occ_conv.max_iter,
        "num_thermal_occ" => occ_conv.num_thermal_occ,
        "num_measure_occ" => occ_conv.num_measure_occ,
        "converged" => occ_conv.converged,
        "best_mu" => occ_conv.best_mu,
        "best_n" => occ_conv.best_n,
        "updated_at" => Dates.now()
    )
    
    JLD2.save(sim_info_path, sim_info)
end

"""
    load_convergence_state(sim_info_path::String) -> Union{Dict, Nothing}

Load the μ-search state (the `"convergence"` entry) from sim.jld2.
Return `nothing` if the file or the entry does not exist.
"""
function load_convergence_state(sim_info_path::String)
    if !isfile(sim_info_path)
        return nothing
    end
    
    sim_info = JLD2.load(sim_info_path)
    if haskey(sim_info, "convergence")
        return sim_info["convergence"]
    end
    return nothing
end

"""
    restore_occupation_convergence(conv_state::Dict, sim_params::SimData) -> OccupationConvergence

Restore an OccupationConvergence struct from saved state.
"""
function restore_occupation_convergence(conv_state::Dict, sim_params::SimData)
    return OccupationConvergence(
        conv_state["target_n"],
        sim_params,
        Tuple(conv_state["mu_edges"]),
        Tuple(conv_state["n_edges"]),
        Vector{Float64}(conv_state["mu_list"]),
        Vector{Float64}(conv_state["n_list"]),
        conv_state["n_tol"],
        conv_state["max_iter"],
        conv_state["num_thermal_occ"],
        conv_state["num_measure_occ"],
        conv_state["converged"],
        conv_state["best_mu"],
        conv_state["best_n"]
    )
end

"""
    validate_convergence_params(conv_state::Dict, wanted_n::Float64, n_tol::Float64) -> Bool

Check that a saved μ-search state was for the same target filling (within `n_tol`).
"""
function validate_convergence_params(conv_state::Dict, wanted_n::Float64, n_tol::Float64)
    # Check target_n matches (within tolerance)
    if abs(conv_state["target_n"] - wanted_n) > n_tol
        return false
    end
    return true
end

"""
    converge_to_occupation!(sim_params::SimData, wanted_n; num_thermal_occ=2000, num_measure_occ=500,
                            measuring_interval=1, n_tol=1e-3, max_iter=20,
                            num_uncorrelated_measurments=32, sim_info_path="",
                            use_mf_mu=true, mf_mu_halfwidth=0.75)
        -> (occ_conv::OccupationConvergence, sim::LatticeData)

Search for the μ that gives filling `wanted_n`, starting a new search.

The trial runs are short and measure only <n> (`n_only=true`); `sim_params` is
modified in place (MC steps, measurement buffer, ω grids, `mu`).
The starting μ bracket is the clean mean-field μ ± `mf_mu_halfwidth` when
`use_mf_mu` and the mean field applies (Q = 0), otherwise (-4, 0). The bracket is
widened automatically if it does not contain `wanted_n`.
If `sim_info_path` is given, the search state is saved there after every step.
Returns the search state and the last trial simulation (its configuration is used
as the thermalized start of the production run).
"""
function converge_to_occupation!(sim_params::SimData, wanted_n::Float64; num_thermal_occ = 2000, num_measure_occ = 500, measuring_interval=1, n_tol=1e-3, max_iter=20, num_uncorrelated_measurments::Int64=32, sim_info_path::String="", use_mf_mu::Bool=true, mf_mu_halfwidth::Float64=0.75)
    sim_params.num_thermal = num_thermal_occ
    sim_params.num_measure = num_measure_occ
    # Measurement buffer must hold the >=32-measurement batches that
    # n_uncorrelated_measurments grows to (savedata=false => never flushed).
    sim_params.saving_interval = max(num_measure_occ, 32)
    sim_params.measuring_interval = measuring_interval
    sim_params.w_list = [-1.0,1.0]
    sim_params.w_conductivity_range = [0.0, 0.01]

    # Seed the mu bracket from the clean k-space mean field (density channel
    # included) when possible; the legacy (-4, 0) is the fallback. establish_
    # initial_bounds! auto-widens if this bracket misses, so a tight guess only
    # speeds convergence. mf_starting_mu returns nothing for Q!=0 (MF is Q=0 only).
    mu_edges = (-4.0, 0.0)
    if use_mf_mu
        mfmu = mf_starting_mu(sim_params, wanted_n)
        if mfmu !== nothing
            mu_edges = (mfmu - mf_mu_halfwidth, mfmu + mf_mu_halfwidth)
            println("MF starting mu = $(round(mfmu, digits=4)) -> mu_edges = $(mu_edges)")
        end
    end

    occ_conv = Occupation_Convergence(
        wanted_n, #target_n
        sim_params;
        mu_edges = mu_edges,
        n_edges = (0.0, 1.0),
        n_tol = n_tol,
        max_iter = max_iter,
        num_thermal_occ = num_thermal_occ,
        num_measure_occ = num_measure_occ
    )
    return converge_to_occupation!(occ_conv; num_uncorrelated_measurments=num_uncorrelated_measurments, sim_info_path=sim_info_path)
end


"""
    converge_to_occupation!(data::LatticeData, wanted_n; kwargs...)

Same as the `SimData` method, using a copy of `data.sim_data` (`data` is not changed).
"""
function converge_to_occupation!(data::LatticeData, wanted_n::Float64; num_thermal_occ = 2000, num_measure_occ = 500, measuring_interval=1, n_tol=1e-3, max_iter=20, num_uncorrelated_measurments::Int64=32, sim_info_path::String="")
    sim_params = deepcopy(data.sim_data)
    return converge_to_occupation!(sim_params, wanted_n; num_thermal_occ=num_thermal_occ, num_measure_occ=num_measure_occ, measuring_interval=measuring_interval, n_tol=n_tol, max_iter=max_iter, num_uncorrelated_measurments=num_uncorrelated_measurments, sim_info_path=sim_info_path)
end


"""
    n_uncorrelated_measurments(occ_conv::OccupationConvergence; num_uncorrelated_measurments=32)
        -> (n_values::Vector{Float64}, sim::LatticeData)

Run MC at `occ_conv.sim_params.mu` until there are `num_uncorrelated_measurments`
statistically independent values of the filling n.

The raw n series is cut into blocks of one correlation time and the block means are
kept. If there are too few, the run continues (same configuration) with a longer
`measuring_interval`, and the new block means are added.
Only n is measured (`n_only=true`), so this is much cheaper than a full measurement.
"""
function n_uncorrelated_measurments(occ_conv::OccupationConvergence; num_uncorrelated_measurments::Int64=32)
    # n_only: only <n> is read back here, so the measurements skip every other
    # observable and never allocate the measurement buffers.

    sim = MonteCarloSimulation(occ_conv.sim_params, ""; saveplots=false, savedata=false, n_only=true)
    
    n_data = sim.measurement_data.n[1:sim.current_state.num_current_measurments]
    raw_corr_time = find_correlation_time(n_data)
    raw_corr_time = isnan(raw_corr_time) ? 1 : raw_corr_time
    corr_time = Int64(round(raw_corr_time))+1
    n_uncorrelated = Float64[]
    if corr_time > 2
        append!(n_uncorrelated, get_block_means(n_data[corr_time:end], corr_time, return_nothing=false)[1])
    else
        append!(n_uncorrelated, n_data)
    end

    current_uncorrelated_measurments = length(n_uncorrelated)

    while current_uncorrelated_measurments < num_uncorrelated_measurments
        measurements_left = num_uncorrelated_measurments - current_uncorrelated_measurments
        # Grow the batch toward the >=32 floor; the buffer (saving_interval) is
        # sized max(num_measure_occ, 32) in converge_to_occupation! so a 32-batch
        # always fits (savedata=false => the buffer is never flushed).
        sim.sim_data.num_measure = maximum([minimum([measurements_left, sim.sim_data.num_measure]), 32])
        sim.sim_data.measuring_interval += corr_time + 1
        sim.current_state.num_current_measurments = 0
        sim.current_state.saving_measurement_ix = 0
        fill!(sim.measurement_data.n, 0.0)
        sim = MonteCarloSimulation(sim, ""; saveplots=false, savedata=false, n_only=true)

        curr_n_data = sim.measurement_data.n[1:sim.current_state.num_current_measurments]
        corr_time = Int64(round(find_correlation_time(curr_n_data)))+1
        append!(n_uncorrelated, get_block_means(curr_n_data, corr_time, return_nothing=false)[1])
        current_uncorrelated_measurments = length(n_uncorrelated)
    end
    return n_uncorrelated, sim
end


"""
    probe_mu_edge(occ_conv, mu, edge_name; num_uncorrelated_measurments=32) -> Float64

Measure the mean filling <n> at chemical potential `mu` (one edge of the μ bracket).
`edge_name` ("lower"/"upper") is only used in the printed message.
"""
function probe_mu_edge(occ_conv::OccupationConvergence, mu::Float64, edge_name::String; num_uncorrelated_measurments::Int64=32)
    occ_conv.sim_params.mu = mu
    n_measurements, _ = n_uncorrelated_measurments(occ_conv; num_uncorrelated_measurments=num_uncorrelated_measurments)
    measured_n = mean(n_measurements)
    println("Probing $edge_name edge: mu = $mu, measured n = $measured_n")
    return measured_n
end

"""
    establish_initial_bounds!(occ_conv; num_uncorrelated_measurments=32, sim_info_path="")

Measure <n> at both ends of `occ_conv.mu_edges` and make sure they bracket the target.

Returns one of
- `(lower_mu, upper_mu, lower_n, upper_n)`: a valid bracket;
- `(:converged_at_edge, mu, n)`: one edge already gives the target filling;
- the result of a new `converge_to_occupation!(occ_conv)` call, after moving the
  edge that misses the target by 2 in μ.
"""
function establish_initial_bounds!(occ_conv::OccupationConvergence; num_uncorrelated_measurments::Int64=32, sim_info_path::String="")
    lower_mu, upper_mu = occ_conv.mu_edges
    
    println("Starting occupation convergence to n=$(occ_conv.target_n) with mu edges: ($lower_mu, $upper_mu)")
    
    # Probe lower mu edge
    lower_n = probe_mu_edge(occ_conv, lower_mu, "lower"; num_uncorrelated_measurments=num_uncorrelated_measurments)
    
    # Check if lower edge already matches target
    if abs(lower_n - occ_conv.target_n) < occ_conv.n_tol
        println("Lower edge measurement matches target! n=$lower_n at mu=$lower_mu")
        occ_conv.converged = true
        occ_conv.best_mu = lower_mu
        occ_conv.best_n = lower_n
        occ_conv.n_edges = (lower_n, 1.0)
        return :converged_at_edge, lower_mu, lower_n
    end
    
    if lower_n > occ_conv.target_n
        println("Lower mu edge gives n > target. Extending lower bound.")
        occ_conv.mu_edges = (lower_mu - 2.0, upper_mu)
        return converge_to_occupation!(occ_conv; num_uncorrelated_measurments=num_uncorrelated_measurments, sim_info_path=sim_info_path)
    end
    
    # Always probe the upper mu edge. Never assume n=1 there: with the MF-seeded
    # (or any shifted) bracket the upper edge has no reason to sit at half filling,
    # and a fake upper_n=1 corrupts the false-position interpolation from step 1.
    upper_n = probe_mu_edge(occ_conv, upper_mu, "upper"; num_uncorrelated_measurments=num_uncorrelated_measurments)

    # Check if upper edge already matches target
    if abs(upper_n - occ_conv.target_n) < occ_conv.n_tol
        println("Upper edge measurement matches target! n=$upper_n at mu=$upper_mu")
        occ_conv.converged = true
        occ_conv.best_mu = upper_mu
        occ_conv.best_n = upper_n
        occ_conv.n_edges = (lower_n, upper_n)
        return :converged_at_edge, upper_mu, upper_n
    end

    if upper_n < occ_conv.target_n
        println("Upper mu edge gives n < target. Extending upper bound.")
        occ_conv.mu_edges = (lower_mu, upper_mu + 2.0)
        occ_conv.n_edges = (lower_n, upper_n)
        return converge_to_occupation!(occ_conv; num_uncorrelated_measurments=num_uncorrelated_measurments, sim_info_path=sim_info_path)
    end

    occ_conv.n_edges = (lower_n, upper_n)
    return lower_mu, upper_mu, lower_n, upper_n
end


"""
    converge_to_occupation!(occ_conv::OccupationConvergence; num_uncorrelated_measurments=32,
                            sim_info_path="") -> (occ_conv, sim::LatticeData)

Run (or resume) the μ search described by `occ_conv`.

A fresh search first measures both bracket edges (`establish_initial_bounds!`).
A resumed search (more than 2 entries in `mu_list`) rebuilds the tightest bracket
from the saved history. Each step tries the false-position estimate of μ (the
midpoint if it lands within 0.01 of an edge or the edge fillings are equal),
measures <n>, and replaces one edge. The search stops when <n> is within `n_tol`
or within its own error bar of the target.

If `max_iter` is reached, the last μ is taken as the best estimate
(`converged=false`). In that case the function returns only when `sim_info_path`
is set; otherwise it returns `nothing`.
"""
function converge_to_occupation!(occ_conv::OccupationConvergence; num_uncorrelated_measurments::Int64=32, sim_info_path::String="")
    max_iter = occ_conv.max_iter
    lower_mu, upper_mu = occ_conv.mu_edges
    lower_n, upper_n = occ_conv.n_edges
    
    # Check if we're resuming from a partial convergence
    start_iter = 1
    if length(occ_conv.mu_list) > 2
        # We have prior iterations, resume from where we left off
        start_iter = length(occ_conv.mu_list) - 1  # -2 for initial edges, +1 for next iter
        println("Resuming occupation convergence from iteration $start_iter")
        println("Prior mu_list: $(occ_conv.mu_list)")
        println("Prior n_list: $(occ_conv.n_list)")
        
        # Restore edge values from last known state
        # Find the best lower and upper bounds from history
        for i in 1:length(occ_conv.n_list)
            if occ_conv.n_list[i] < occ_conv.target_n && occ_conv.n_list[i] > lower_n
                lower_n = occ_conv.n_list[i]
                lower_mu = occ_conv.mu_list[i]
            elseif occ_conv.n_list[i] >= occ_conv.target_n && occ_conv.n_list[i] < upper_n
                upper_n = occ_conv.n_list[i]
                upper_mu = occ_conv.mu_list[i]
            end
        end
    else
        # Fresh start: probe edges to establish valid bounds
        result = establish_initial_bounds!(occ_conv; num_uncorrelated_measurments=num_uncorrelated_measurments, sim_info_path=sim_info_path)
        
        # If result is not a tuple, it means we recursed and already returned
        if !(result isa Tuple)
            return result
        end
        
        # If establish_initial_bounds! recursively called converge_to_occupation!
        # (e.g. when extending bounds), it returns a 2-tuple (OccupationConvergence, LatticeData).
        # Pass it through directly — convergence is already done.
        if length(result) == 2
            return result
        end
        
        # Check if we converged at an edge (3-tuple: :converged_at_edge, mu, n)
        if result[1] === :converged_at_edge
            # Already converged at edge measurement
            best_mu, best_n = result[2], result[3]
            push!(occ_conv.mu_list, best_mu)
            push!(occ_conv.n_list, best_n)
            
            # Save final convergence state
            if sim_info_path != ""
                save_convergence_state(sim_info_path, occ_conv)
            end
            
            # Create a simulation with the converged mu to return
            occ_conv.sim_params.mu = best_mu
            _, sim = n_uncorrelated_measurments(occ_conv; num_uncorrelated_measurments=num_uncorrelated_measurments)
            return occ_conv, sim
        end
        
        lower_mu, upper_mu, lower_n, upper_n = result
    end


    # Only append initial edges if this is a fresh start
    if start_iter == 1
        append!(occ_conv.mu_list, [lower_mu, upper_mu])
        lower_n, upper_n = occ_conv.n_edges
        append!(occ_conv.n_list, [lower_n, upper_n])
    end

    for iter in start_iter:max_iter
        # Linear interpolation to estimate mu for target occupation
        if abs(upper_n - lower_n) < occ_conv.n_tol 
            test_mu = (lower_mu + upper_mu) / 2
        else
            # Find mu where n crosses target_n
            test_mu = lower_mu + (occ_conv.target_n - lower_n) * (upper_mu - lower_mu) / (upper_n - lower_n)
            if abs(test_mu - lower_mu) < 1e-2 || abs(test_mu - upper_mu) < 1e-2
                test_mu = (lower_mu + upper_mu) / 2
            end
        end
        occ_conv.sim_params.mu = test_mu
        n_data, sim = n_uncorrelated_measurments(occ_conv; num_uncorrelated_measurments= num_uncorrelated_measurments)

        measured_n = mean(n_data)
        error_n = std(n_data) / sqrt(length(n_data))
        
        push!(occ_conv.mu_list, test_mu)
        push!(occ_conv.n_list, measured_n)
        
        println("Iteration $iter: mu = $test_mu, measured n = $measured_n ± $error_n")
        
        if abs(measured_n - occ_conv.target_n) < occ_conv.n_tol || abs(measured_n - occ_conv.target_n) < error_n
            println("Converged to target occupation: $measured_n ± $error_n at mu = $test_mu")
            push!(occ_conv.mu_list, test_mu)
            push!(occ_conv.n_list, measured_n)
            occ_conv.converged = true
            occ_conv.best_mu = test_mu
            occ_conv.best_n = measured_n
            
            # Save final convergence state
            if sim_info_path != ""
                save_convergence_state(sim_info_path, occ_conv)
            end
            return occ_conv, sim

        elseif measured_n < occ_conv.target_n
            # change edges
            lower_mu = test_mu
            lower_n = measured_n

        else
            # change mu edges
            upper_mu = test_mu
            upper_n = measured_n
        end
        
        # Save state after each iteration for resume capability
        if sim_info_path != ""
            save_convergence_state(sim_info_path, occ_conv)
        end

        if iter == max_iter
            println("Failed to converge to target occupation within $max_iter iterations.")
            println("Final mu bounds: ($lower_mu, $upper_mu), n bounds: ($lower_n, $upper_n)")
            println("taking best estimate mu = $(occ_conv.mu_list[end]), n = $(occ_conv.n_list[end])")
            if sim_info_path != ""
                occ_conv.converged = false
                occ_conv.best_mu = occ_conv.mu_list[end]
                occ_conv.best_n = occ_conv.n_list[end]
                save_convergence_state(sim_info_path, occ_conv)
                return occ_conv, sim
            end
        end
    end

end


"""
    running_simulation_wanted_occupation(Lx, Ly, L_SC, U, t, V, seed, wanted_n, beta, B,
        num_measure, num_thermal, measuring_interval, w_list, initial_guess, U_disorder,
        diag_method, pairing_symmetry, w_conductivity_range, U_impurity; kwargs...) -> LatticeData

Run one simulation (one MC seed) at filling `wanted_n`: find μ, then run the production MC.

The positional arguments are the same as for `running_simulation_once`, with the
filling `wanted_n` in place of μ (see Appendix A of docs/WORKFLOW.md for each parameter).

Steps:
1. Build the `SimData` and check that the production run fits in the job's memory.
2. Resume, if `resume_path` holds a previous attempt: continue from `checkpoint.jld2`
   if the μ search had finished, else continue the μ search.
3. μ search (`converge_to_occupation!`) with `num_thermal_occ`, `num_measure_occ`,
   `measuring_interval_occ`, `n_tol`, `max_iter`. Snapshots and field histograms are
   switched off during the search.
4. Production run at the found μ, starting from the last search configuration.
   `measuring_interval` is raised to the one the search needed, if larger.

Keyword arguments not listed above are those of `running_simulation_once`
(output folder, saving, cluster size, gauge, density channel `alpha`/`beta_phi`/
`Q_density`, μ disorder, field snapshots and histograms).
With `return_converegede_state=true` the result is `(data, occ_conv)`.
"""
function running_simulation_wanted_occupation(Lx::Int64, Ly::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, wanted_n::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1},
    initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
	directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=500, dump_time=Dates.Hour(1), saveplots=false, date_folder=false, numbered=true, seed_folder=false, U_puddle_params=(0.0, 6.0, 0.7, 0.7), TClustersize_x=4, TClustersize_y=4, U_smooth=(false, 0.8), loop_random=true, boundary_conditions="PBC", gauge_choice="LandauX",
     delta_fixed_values=Float64[], delta_values_stiffness=Float64[], dw_values=Float64[],
    disorder_seed::Int64=0,   # which disorder realization; 0 => derive from the MC seed
    num_thermal_occ::Int64=1000, num_measure_occ::Int64=500, measuring_interval_occ::Int64=5, n_tol::Float64=1e-3, max_iter::Int64=20, num_uncorrelated_measurments::Int64=32,
    resume_path::String=directory, return_converegede_state=false,
    # density-channel decoupling. Defaults reproduce the legacy (channel-off) run.
    # These flow into sim_params below, so the SAME channel is active during the
    # mu-finding convergence phase (MC on both Delta and phi) as in production.
    alpha::Float64=1.0, beta_phi::Float64=0.0, Q_density::Array{Float64,1}=[0.0, 0.0],
    # mu-U disorder + field observables. All default OFF so existing campaigns that
    # do not pass them are unchanged. The observables are FORCED OFF during the
    # mu-finding convergence phase (see sim_params_occ below) and only accumulate in
    # production, so histograms never see wrong-mu samples.
    mu_disorder::String="none", mu_disorder_f::Float64=0.0,
    snapshot_interval::Int64=0,
    phi_hist_range::Array{Float64,1}=[-10.0, 10.0], phi_hist_nbins::Int64=0,
    Delta_hist_abs_range::Array{Float64,1}=[0.0, 10.0], Delta_hist_abs_nbins::Int64=0,
    Delta_hist_arg_nbins::Int64=0
    )

    start_time = Dates.now()
    
    # Adjust saving interval if not saving
    if !savedata
        saving_interval = num_measure
    end
    
    # Validate diag_method parameters
    L_min = min(Lx, Ly)
    if diag_method == "TCluster" && (TClustersize_x >= Lx || TClustersize_y >= Ly)
        println("Warning: TClustersize should be smaller than L. setting Lc=L")
        TClustersize_x = Lx
        TClustersize_y = Ly
        diag_method = "exact"
    end
    
    if diag_method == "K0-TCA" && (TClustersize_x >= Lx || TClustersize_y >= Ly)
        println("Warning: TClustersize should be smaller than L. setting Lc=L")
        TClustersize_x = Lx
        TClustersize_y = Ly
        diag_method = "K0"
    end

    # Create sim_params with initial mu (will be updated after convergence)
    N = Lx * Ly
    sim_params = SimData(Lx, Ly, N, L_SC, U, t, V, seed, -0.0, beta, num_measure, num_thermal, measuring_interval, saving_interval,
        w_list, initial_guess, U_disorder, diag_method, pairing_symmetry, directory, num_cores, Threads.nthreads(), start_time, w_conductivity_range, delta_fixed_values, delta_values_stiffness, collect(U_puddle_params), U_impurity, TClustersize_x, TClustersize_y, U_smooth, loop_random, B, gauge_choice, boundary_conditions, dw_values, false,  # compute_jP: never compute for constant filling
        disorder_seed,
        alpha, beta_phi, Q_density,  # density-channel decoupling (off by default)
        snapshot_interval,           # raw-field snapshots (per-measurement, production only)
        phi_hist_range, phi_hist_nbins, Delta_hist_abs_range, Delta_hist_abs_nbins,
        Delta_hist_arg_nbins,        # field histograms (production only; see sim_params_occ)
        mu_disorder, mu_disorder_f   # coupled / potential mu-U disorder
    )

    # Fail before the (possibly hours-long) mu-search if the production run cannot
    # fit in the job's memory (the mu-search itself, n_only, needs far less).
    # Override: MC_SKIP_MEMORY_CHECK=1.
    check_memory_budget(sim_params; verbose=savedata)

    # =========================================================================
    # Resume logic: Check if resuming from existing simulation
    # =========================================================================
    resuming = false
    resuming_convergence = false
    resuming_simulation = false
    conv_state = nothing
    
    if resume_path != "" && isdir(resume_path)
        # When seed_folder=true, the actual simulation data is in seed_$seed subfolder
        # resume_path is the parent directory, so we need to construct the seed folder path
        if seed_folder
            dir_path = joinpath(resume_path, "seed_$seed")
        else
            dir_path = resume_path
        end
        assert_not_slimmed(dir_path)

        sim_info_path = joinpath(resume_path, "sim.jld2")
        checkpoint_path = joinpath(dir_path, "checkpoint.jld2")
        log_filename = joinpath(dir_path, "log.txt")
        
        # Only attempt resume if the seed folder actually exists (for seed_folder=true case)
        # or if sim_info_path exists (for seed_folder=false case)
        if isdir(dir_path) && isfile(sim_info_path)
            # Load convergence state
            conv_state = load_convergence_state(sim_info_path)
            
            if conv_state !== nothing
                # Validate parameters match
                if !validate_convergence_params(conv_state, wanted_n, n_tol)
                    error("Resume failed: Target occupation ($(wanted_n)) does not match saved state ($(conv_state["target_n"]))")
                end
                
                if conv_state["converged"]
                    # Convergence complete, check if we need to resume simulation
                    if isfile(checkpoint_path)
                        resuming = true
                        resuming_simulation = true
                        println("Resuming from simulation checkpoint...")
                        println("Convergence was complete. Loading checkpoint and continuing main simulation.")
                    else
                        # Convergence complete but no checkpoint - need to restart main simulation
                        resuming = true
                        resuming_convergence = false
                        resuming_simulation = false
                        println("Convergence complete but no checkpoint found. Restarting main simulation.")
                    end
                else
                    # Convergence incomplete, resume it
                    resuming = true
                    resuming_convergence = true
                    println("Resuming incomplete convergence phase...")
                end
            end
            
            # Log resume - only if we're actually resuming
            if resuming && savedata
                open(log_filename, "a") do io
                    println(io, "----------------------------------------")
                    println(io, "Resuming simulation at $(Dates.now())")
                    if resuming_convergence
                        println(io, "State: Resuming convergence phase")
                    elseif resuming_simulation
                        println(io, "State: Resuming main simulation from checkpoint")
                    else
                        println(io, "State: Convergence complete, restarting main simulation")
                    end
                    println(io, "----------------------------------------")
                end
            end
        end
    end
    
    # =========================================================================
    # Path 1: Resume from simulation checkpoint
    # =========================================================================
    if resuming_simulation
        # Load checkpoint and continue using the proper load_checkpoint function
        data = load_checkpoint(dir_path)
        if data === nothing
            @warn "Failed to load checkpoint. Falling back to main simulation restart with saved converged state." dir_path
            if conv_state === nothing
                conv_state = load_convergence_state(sim_info_path)
            end
            if conv_state === nothing || !get(conv_state, "converged", false)
                error("Failed to load checkpoint from $dir_path and no valid converged state found in $sim_info_path")
            end
            resuming_simulation = false
            resuming_convergence = false
            resuming = true
            println("Checkpoint load failed. Restarting main simulation from saved converged state.")
        else
            # Update start_time reference but keep measurement counts
            data.sim_data.start_time = start_time

            # Run the remaining simulation
            data = MonteCarloSimulation(data, dir_path; saveplots=saveplots, savedata=savedata,
                saving_interval=saving_interval, dump_time=dump_time,
                sim_info_path=sim_info_path, log_filename=log_filename)

            # Finalize tracking
            if savedata
                finalize_simulation_tracking(sim_info_path, seed, log_filename, start_time)
            end
            return data
        end
    end
    
    # =========================================================================
    # Path 2 & 3: Fresh start or resume convergence
    # =========================================================================
    if !resuming
        # Fresh start: Setup paths
        dir_path, sim_info_path, log_filename, checkpoint_path = setup_simulation_paths(directory, sim_params; date_folder=date_folder, numbered=numbered, seed_folder=seed_folder)

        # Initialize tracking if saving
        if savedata
            initialize_sim_tracking(sim_info_path, seed, start_time, checkpoint_path, dir_path, directory)
        end

        # Log convergence phase start
        if savedata
            open(log_filename, "a") do io
                println(io, "----------------------------------------")
                println(io, "Starting occupation convergence phase")
                println(io, "Target occupation: n = $(wanted_n)")
                println(io, "Convergence parameters: n_tol=$(n_tol), max_iter=$(max_iter)")
                println(io, "Convergence thermal steps: $(num_thermal_occ), measurements: $(num_measure_occ)")
                println(io, "----------------------------------------")
            end
        end
    end
    
    # =========================================================================
    # Convergence phase (fresh or resume)
    # =========================================================================
    sim_params_occ = deepcopy(sim_params)
    # Field observables must NOT accumulate during the mu-finding convergence phase
    # (those samples are at the wrong, still-searching mu). Force them off on the
    # convergence copy; production starts them fresh (ensure_field_histograms! re-sizes
    # the transferred state, and snapshots key off the reset measurement counter).
    sim_params_occ.snapshot_interval    = 0
    sim_params_occ.phi_hist_nbins       = 0
    sim_params_occ.Delta_hist_abs_nbins = 0
    sim_params_occ.Delta_hist_arg_nbins = 0
    best_sim = nothing
    
    if resuming_convergence
        # Restore convergence struct from saved state
        conv_state = load_convergence_state(sim_info_path)
        occ_conv = restore_occupation_convergence(conv_state, sim_params_occ)
        occ_conv.sim_params.num_thermal = num_thermal_occ
        occ_conv.sim_params.num_measure = num_measure_occ
        # Buffer sized for the >=32 batches (see converge_to_occupation!).
        occ_conv.sim_params.saving_interval = max(num_measure_occ, 32)
        occ_conv.sim_params.measuring_interval = measuring_interval_occ
        occ_conv.sim_params.w_list = [-1.0, 1.0]
        occ_conv.sim_params.w_conductivity_range = [0.0, 0.01]
        
        converged_struct, best_sim = converge_to_occupation!(occ_conv; 
            num_uncorrelated_measurments=num_uncorrelated_measurments,
            sim_info_path=sim_info_path)
    elseif resuming && conv_state !== nothing && get(conv_state, "converged", false)
        # Convergence already finished previously: skip re-running it.
        converged_struct = restore_occupation_convergence(conv_state, sim_params_occ)
        if converged_struct.best_mu === nothing
            error("Saved convergence state is marked converged but best_mu is missing in $sim_info_path")
        end
    else
        # Fresh convergence
        converged_struct, best_sim = converge_to_occupation!(sim_params_occ, wanted_n; 
            num_thermal_occ=num_thermal_occ, num_measure_occ=num_measure_occ, 
            measuring_interval=measuring_interval_occ, n_tol=n_tol, max_iter=max_iter, 
            num_uncorrelated_measurments=num_uncorrelated_measurments,
            sim_info_path=sim_info_path)
    end

    if !converged_struct.converged
        println("Occupation convergence failed.")
        println("Using best estimate mu for wanted_n : ", wanted_n)
    else
        println("occupation convergence successful for $(Lx)x$(Ly) system. wanted n: ", wanted_n)
    end

    println("Best mu found: ", converged_struct.best_mu)
    println("Best n found: ", converged_struct.best_n)

    # Log convergence results
    if savedata
        open(log_filename, "a") do io
            println(io, "Occupation convergence successful!")
            println(io, "Converged mu: $(converged_struct.best_mu)")
            println(io, "Achieved occupation: $(converged_struct.best_n)")
            println(io, "Convergence iterations: $(length(converged_struct.mu_list) - 2)")
            best_interval = best_sim === nothing ? measuring_interval : best_sim.sim_data.measuring_interval
            println(io, "Final measuring interval: $(best_interval)")
            println(io, "----------------------------------------")
            println(io, "Starting main simulation with converged mu")
            println(io, "Using convergence state as thermalization (reducing waste)")
        end
        
        # Log full simulation parameters
        log_simulation_start(log_filename, sim_params, start_time; continuing=false)
    end

    # Update sim_params with converged mu and optimized measuring_interval
    sim_params.mu = converged_struct.best_mu
    best_interval = best_sim === nothing ? measuring_interval : best_sim.sim_data.measuring_interval
    sim_params.measuring_interval = maximum([best_interval, measuring_interval])
    
    # Create LatticeData and transfer thermalized state from convergence
    # This reuses the converged configuration as thermalization, avoiding wasted steps
    data = Lattice_Data(sim_params)
    if best_sim !== nothing
        data.current_state = best_sim.current_state
        data.current_state.num_current_measurments = 0
        data.current_state.saving_measurement_ix = 0
    end

    # Run the final simulation
    data = MonteCarloSimulation(data, dir_path; saveplots=saveplots, savedata=savedata, 
        saving_interval=saving_interval, dump_time=dump_time, 
        sim_info_path=sim_info_path, log_filename=log_filename)

    # Finalize tracking
    if savedata
        finalize_simulation_tracking(sim_info_path, seed, log_filename, start_time)
    end

    if return_converegede_state
        return data, converged_struct
    end
    return data
end



"""
    running_simulation_seeds_wanted_occupation(Lx, Ly, L_SC, U, t, V, seeds::Vector{Int64},
        wanted_n, beta, B, ...; seed_folder=true, return_data=false, kwargs...) -> Vector

Run `running_simulation_wanted_occupation` for each MC seed in `seeds`, one after another.

Each seed gets its own μ search and, with `seed_folder=true`, its own `seed_<seed>`
folder. Finished seeds are recorded in `directory/sim.jld2` (`"completed_seeds"`)
and skipped on a restart. Returns the `LatticeData` of every seed if `return_data`,
else an empty vector.

Note: this method passes only part of the keywords on (no density channel, μ disorder,
field histograms, or `disorder_seed`); the cluster drivers call the single-seed
function directly.
"""
function running_simulation_seeds_wanted_occupation(Lx::Int64, Ly::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seeds::Vector{Int64}, wanted_n::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1},
    initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
    directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=500, dump_time=Dates.Hour(1), saveplots=false, date_folder=false, numbered=false, seed_folder=true, U_puddle_params=(0.0, 6.0, 0.7, 0.7), TClustersize_x=4, TClustersize_y=4, U_smooth=(false, 0.8), loop_random=true, boundary_conditions="PBC", gauge_choice="LandauX",
    delta_fixed_values=Float64[], delta_values_stiffness=Float64[], dw_values=Float64[],
    num_thermal_occ::Int64=1000, num_measure_occ::Int64=500, measuring_interval_occ::Int64=5, n_tol::Float64=1e-3, max_iter::Int64=20, num_uncorrelated_measurments::Int64=32, return_data::Bool=false
    )

    out_data = []
    sim_info_path = joinpath(directory, "sim.jld2")
    
    # =========================================================================
    # Resume logic: Check which seeds have already been completed
    # =========================================================================
    completed_seeds = Int64[]
    if savedata && isfile(sim_info_path)
        sim_info = JLD2.load(sim_info_path)
        if haskey(sim_info, "completed_seeds")
            completed_seeds = sim_info["completed_seeds"]
            println("Resuming multi-seed run. Already completed seeds: ", completed_seeds)
        end
    end
    
    for seed in seeds
        # Skip already completed seeds
        if seed in completed_seeds
            println("Skipping seed $seed - already completed")
            continue
        end
        
        # Determine resume path for this seed
        seed_resume_path = seed_folder ? joinpath(directory, "seed_$seed") : directory
        
        println("----------------------------------------")
        println("Starting seed $seed ($(findfirst(==(seed), seeds))/$(length(seeds)))")
        println("----------------------------------------")
        
        data = running_simulation_wanted_occupation(Lx, Ly, L_SC, U, t, V, seed, wanted_n, beta, B, num_measure, num_thermal, measuring_interval, w_list, initial_guess, U_disorder, diag_method, pairing_symmetry, w_conductivity_range, U_impurity;
                directory=directory, num_cores=num_cores, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, saveplots=saveplots, date_folder=date_folder, numbered=numbered, seed_folder=seed_folder,
                    U_puddle_params=U_puddle_params, TClustersize_x=TClustersize_x, TClustersize_y=TClustersize_y, U_smooth=U_smooth, loop_random=loop_random, boundary_conditions=boundary_conditions, gauge_choice=gauge_choice,
                    delta_fixed_values=delta_fixed_values, delta_values_stiffness=delta_values_stiffness, dw_values=dw_values,
                    num_thermal_occ=num_thermal_occ, num_measure_occ=num_measure_occ, measuring_interval_occ=measuring_interval_occ, n_tol=n_tol, max_iter=max_iter, num_uncorrelated_measurments=num_uncorrelated_measurments,
                    resume_path=seed_resume_path)
        
        # Mark seed as completed in sim.jld2
        if savedata
            sim_info = isfile(sim_info_path) ? JLD2.load(sim_info_path) : Dict{String, Any}()
            if !haskey(sim_info, "completed_seeds")
                sim_info["completed_seeds"] = Int64[]
            end
            push!(sim_info["completed_seeds"], seed)
            sim_info["last_completed_seed"] = seed
            sim_info["last_completed_time"] = Dates.now()
            JLD2.save(sim_info_path, sim_info)
            println("Seed $seed completed and saved to tracking.")
        end
        
        if return_data
            push!(out_data, data)
        end
    end

    if savedata && isfile(sim_info_path)
        sim_info = JLD2.load(sim_info_path)
        sim_info["endtime"] = Dates.now()
        sim_info["all_seeds_completed"] = true
        JLD2.save(sim_info_path, sim_info)
    end
    return out_data
end

# Square-lattice (L×L) versions, kept for older scripts.
"""
    running_simulation_wanted_occupation(L, L_SC, U, ...; TClustersize=4, kwargs...)

Square-lattice form: same as the `(Lx, Ly, ...)` method with `Lx = Ly = L` and a
`TClustersize × TClustersize` cluster.
"""
function running_simulation_wanted_occupation(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, wanted_n::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1},
    initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
    TClustersize=4, kwargs...)

    return running_simulation_wanted_occupation(L, L, L_SC, U, t, V, seed, wanted_n, beta, B, num_measure, num_thermal, measuring_interval, w_list,
        initial_guess, U_disorder, diag_method, pairing_symmetry, w_conductivity_range, U_impurity;
        TClustersize_x=TClustersize, TClustersize_y=TClustersize, kwargs...
    )
end


"""
    running_simulation_seeds_wanted_occupation(L, L_SC, U, ...; TClustersize=4, kwargs...)

Square-lattice form: same as the `(Lx, Ly, ...)` method with `Lx = Ly = L`.
"""
function running_simulation_seeds_wanted_occupation(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seeds::Vector{Int64}, wanted_n::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1},
    initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
    TClustersize=4, kwargs...)

    return running_simulation_seeds_wanted_occupation(L, L, L_SC, U, t, V, seeds, wanted_n, beta, B, num_measure, num_thermal, measuring_interval, w_list,
        initial_guess, U_disorder, diag_method, pairing_symmetry, w_conductivity_range, U_impurity;
        TClustersize_x=TClustersize, TClustersize_y=TClustersize, kwargs...    
    )
end

"""
    plot_convergence_output(output::String)

Parse the output string from occupation convergence and create a plot of mu vs n with error bars.
The plot includes the target occupation as a horizontal dashed line and initial mu edges as vertical dotted lines.
"""
function plot_convergence_output(output::String)
    lines = split(output, '\n')
    
    mu_list = Float64[]
    n_list = Float64[]
    error_list = Float64[]
    target_n = 0.0
    lower_mu = 0.0
    upper_mu = 0.0
    
    for line in lines
        line = strip(line)
        
        if startswith(line, "Starting occupation convergence to n=")
            # Parse target_n and mu_edges
            # "Starting occupation convergence to n=0.875 with mu edges: (-4.0, 0.0)"
            parts = split(line, "n=")
            target_part = parts[2]
            target_str, edges_str = split(target_part, " with mu edges: ")
            target_n = parse(Float64, target_str)
            
            edges_str = replace(replace(edges_str, "(" => ""), ")" => "")
            lower_str, upper_str = split(edges_str, ", ")
            lower_mu = parse(Float64, lower_str)
            upper_mu = parse(Float64, upper_str)
            
        elseif startswith(line, "Probing lower edge:")
            # "Probing lower edge: mu = -4.0, measured n = 0.36558521494750384"
            parts = split(line, ", ")
            mu_part = split(parts[1], "mu = ")[2]
            n_part = split(parts[2], "measured n = ")[2]
            mu = parse(Float64, mu_part)
            n = parse(Float64, n_part)
            push!(mu_list, mu)
            push!(n_list, n)
            push!(error_list, 0.0)  # No error for probe
            
        elseif startswith(line, "Iteration ")
            # "Iteration 1: mu = -0.7881279121807134, measured n = 0.8560638968117702 ± 0.0011338849728516618"
            parts = split(line, ": ")
            data_part = parts[2]
            mu_n_part, error_part = split(data_part, " ± ")
            mu_str, n_str = split(mu_n_part, ", measured n = ")
            mu_str = split(mu_str, "mu = ")[2]
            mu = parse(Float64, mu_str)
            n = parse(Float64, n_str)
            error = parse(Float64, error_part)
            push!(mu_list, mu)
            push!(n_list, n)
            push!(error_list, error)
        end
    end
    
    # Create the plot
    p = plot(mu_list, n_list, yerror=error_list, 
             label="Measured n", 
             xlabel="μ", 
             ylabel="n", 
             title="Occupation Convergence to n = $target_n",
             marker=:circle,
             line=:solid)
    
    # Add target occupation line
    hline!([target_n], label="Target n", linestyle=:dash, color=:red)
    
    # Add initial mu edges
    vline!([lower_mu, upper_mu], label="Initial μ edges", linestyle=:dot, color=:gray)
    
    # Display the plot
    display(p)
    
    return p
end
