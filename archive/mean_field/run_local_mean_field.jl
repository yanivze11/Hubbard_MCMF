# Local BDG mean-field runner (no cluster jobs).
#
# Runs a configurable sweep (beta or mu) using BDG_MF.jl and writes
# a compact CSV summary for easy comparison against MC outputs.

include(joinpath(@__DIR__, "BDG_MF.jl"))
include(joinpath(@__DIR__, "homogeneous_kspace_observables.jl"))

using Dates
using Printf
using Statistics

function first_or_nan(vec)
    return isempty(vec) ? NaN : vec[1]
end

function summarize_mf_run(sim::MFSim; sweep_name::String, sweep_value::Float64, beta::Float64, mu::Float64)
    data = sim.lattice_data
    meas = data.measurement_data

    n_val = first_or_nan(meas.n)
    kx_val = first_or_nan(meas.Kx)

    lxx_qy0 = NaN
    if !isempty(meas.Lxx_qy)
        lxx_qy0 = first_or_nan(meas.Lxx_qy[1])
    end

    stiffness_lowest_q = isfinite(lxx_qy0) ? -0.25 * (kx_val + lxx_qy0) : NaN

    delta_s_abs_mean = mean(abs.(data.current_state.Delta_s))
    delta_dx_abs_mean = mean(abs.(data.current_state.Delta_dx))
    delta_dy_abs_mean = mean(abs.(data.current_state.Delta_dy))

    delta_s_abs = abs(mean(data.current_state.Delta_s))
    delta_dx_abs = abs(mean(data.current_state.Delta_dx))
    delta_dy_abs = abs(mean(data.current_state.Delta_dy))

    max_eq = sim.mf_state.iter > 0 ? sim.mf_state.max_eq_vec[sim.mf_state.iter] : NaN

    return (
        sweep_name=sweep_name,
        sweep_value=sweep_value,
        beta=beta,
        mu=mu,
        n=n_val,
        Kx=kx_val,
        Lxx_qy0=lxx_qy0,
        stiffness_lowest_q=stiffness_lowest_q,
        delta_s_abs_mean=delta_s_abs_mean,
        delta_dx_abs_mean=delta_dx_abs_mean,
        delta_dy_abs_mean=delta_dy_abs_mean,
        delta_s_abs=delta_s_abs,
        delta_dx_abs=delta_dx_abs,
        delta_dy_abs=delta_dy_abs,
        converged=sim.mf_state.converged,
        iter=sim.mf_state.iter,
        max_eq=max_eq,
    )
end

function summarize_homogeneous_run(data::LatticeData, info; sweep_name::String, sweep_value::Float64, beta::Float64, mu::Float64)
    meas = data.measurement_data

    n_val = first_or_nan(meas.n)
    kx_val = first_or_nan(meas.Kx)

    lxx_qy0 = NaN
    if !isempty(meas.Lxx_qy)
        lxx_qy0 = first_or_nan(meas.Lxx_qy[1])
    end

    stiffness_lowest_q = isfinite(lxx_qy0) ? -0.25 * (kx_val + lxx_qy0) : NaN

    delta_s_abs_mean = mean(abs.(data.current_state.Delta_s))
    delta_dx_abs_mean = mean(abs.(data.current_state.Delta_dx))
    delta_dy_abs_mean = mean(abs.(data.current_state.Delta_dy))

    delta_s_abs = abs(mean(data.current_state.Delta_s))
    delta_dx_abs = abs(mean(data.current_state.Delta_dx))
    delta_dy_abs = abs(mean(data.current_state.Delta_dy))

    max_eq = NaN
    converged = true
    iter = 0
    if hasproperty(info, :delta_summary)
        ds = info.delta_summary
        if ds isa NamedTuple
            max_eq = get(ds, :residual_max, NaN)
            converged = get(ds, :converged, true)
            iter = get(ds, :iterations, 0)
        end
    end

    return (
        sweep_name=sweep_name,
        sweep_value=sweep_value,
        beta=beta,
        mu=mu,
        n=n_val,
        Kx=kx_val,
        Lxx_qy0=lxx_qy0,
        stiffness_lowest_q=stiffness_lowest_q,
        delta_s_abs_mean=delta_s_abs_mean,
        delta_dx_abs_mean=delta_dx_abs_mean,
        delta_dy_abs_mean=delta_dy_abs_mean,
        delta_s_abs=delta_s_abs,
        delta_dx_abs=delta_dx_abs,
        delta_dy_abs=delta_dy_abs,
        converged=converged,
        iter=iter,
        max_eq=max_eq,
    )
end

function write_summary_csv(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "sweep_name,sweep_value,beta,mu,n,Kx,Lxx_qy0,stiffness_lowest_q,delta_s_abs_mean,delta_dx_abs_mean,delta_dy_abs_mean,delta_s_abs,delta_dx_abs,delta_dy_abs,converged,iter,max_eq")
        for r in rows
            @printf(
                io,
                "%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s,%d,%.10g\n",
                r.sweep_name,
                r.sweep_value,
                r.beta,
                r.mu,
                r.n,
                r.Kx,
                r.Lxx_qy0,
                r.stiffness_lowest_q,
                r.delta_s_abs_mean,
                r.delta_dx_abs_mean,
                r.delta_dy_abs_mean,
                r.delta_s_abs,
                r.delta_dx_abs,
                r.delta_dy_abs,
                string(r.converged),
                r.iter,
                r.max_eq,
            )
        end
    end
    return path
end

function run_local_mf_sweep(; 
    solution_mode::Symbol=:bdg_real_space,
    sweep::Symbol=:beta,
    sweep_values::Vector{Float64}=collect(1.0:1.0:10.0),
    fixed_beta::Float64=10.0,
    fixed_mu::Float64=-1.7,
    target_n::Union{Nothing, Float64}=nothing,
    mu_lims::Vector{Float64}=[-4.0, 0.0],
    n_resolution::Float64=5e-3,
    L::Int64=10,
    L_SC::Int64=1,
    U::Float64=4.0,
    t::Float64=1.0,
    V::Float64=0.0,
    seed::Int64=12345,
    w_list::Vector{Float64}=collect(-3.0:0.1:3.0),
    w_conductivity_range::Vector{Float64}=collect(0.0:0.1:4.0),
    initial_guess::String="uniform",
    U_disorder::String="uniform",
    pairing_symmetry::String="s-wave",
    U_impurity::Float64=0.0,
    max_iter::Int64=5000,
    mu_tune_max_iter::Int64=30,
    tol::Float64=1e-6,
    mixing_a::Float64=0.6,
    save_data::Bool=false,
    output_dir::String=joinpath(@__DIR__, "demo_results", "local_mf")
)
    if !(sweep in (:beta, :mu))
        error("sweep must be :beta or :mu")
    end
    if !(solution_mode in (:bdg_real_space, :homogeneous_kspace))
        error("solution_mode must be :bdg_real_space or :homogeneous_kspace")
    end

    rows = NamedTuple[]

    println("="^70)
    println("Running local BDG mean-field sweep")
    println("mode=$(solution_mode), sweep=$(sweep), L=$(L), U=$(U), pairing=$(pairing_symmetry)")
    println("target_n=$(target_n === nothing ? "none" : string(target_n))")
    println("="^70)

    for (ix, sv) in enumerate(sweep_values)
        beta = sweep == :beta ? sv : fixed_beta
        mu = sweep == :mu ? sv : fixed_mu

        @printf("[%d/%d] beta=%.6g, mu=%.6g\n", ix, length(sweep_values), beta, mu)

        row = nothing
        if solution_mode == :bdg_real_space
            sim = nothing
            if target_n === nothing
                sim = running_MF(
                    L,
                    L_SC,
                    U,
                    t,
                    V,
                    seed + ix,
                    mu,
                    beta,
                    w_list,
                    initial_guess,
                    U_disorder,
                    pairing_symmetry,
                    w_conductivity_range,
                    U_impurity;
                    directory=output_dir,
                    savedata=save_data,
                    date_folder=false,
                    numbered=true,
                    seed_folder=false,
                    saving_interval=1,
                    max_iter=max_iter,
                    tol=tol,
                    a=mixing_a,
                )
            else
                sim, mu_found, _ = run_MF_lion_hunting_n(
                    L,
                    L_SC,
                    U,
                    t,
                    V,
                    seed + ix,
                    mu_lims,
                    target_n,
                    n_resolution,
                    beta,
                    w_list,
                    initial_guess,
                    U_disorder,
                    pairing_symmetry,
                    w_conductivity_range,
                    U_impurity;
                    directory=output_dir,
                    savedata=save_data,
                    date_folder=false,
                    numbered=true,
                    seed_folder=false,
                    saving_interval=1,
                    max_iter=max_iter,
                    tol=tol,
                    a=mixing_a,
                )
                mu = mu_found
            end

            row = summarize_mf_run(
                sim;
                sweep_name=String(sweep),
                sweep_value=sv,
                beta=beta,
                mu=mu,
            )
        else
            data = nothing
            info = nothing
            if target_n === nothing
                data, info = run_local_homogeneous_kspace(
                    L=L,
                    L_SC=L_SC,
                    U=U,
                    t=t,
                    V=V,
                    seed=seed + ix,
                    mu=mu,
                    beta=beta,
                    pairing_symmetry=pairing_symmetry,
                    w_list=w_list,
                    w_conductivity_range=w_conductivity_range,
                    dw_values=[0.01, 0.05, 0.1],
                    U_disorder=U_disorder,
                    U_impurity=U_impurity,
                    diag_method="K0",
                )
            else
                data, info, mu_found, _ = run_local_homogeneous_kspace_target_n(
                    L=L,
                    L_SC=L_SC,
                    U=U,
                    t=t,
                    V=V,
                    seed=seed + ix,
                    mu_lims=mu_lims,
                    target_n=target_n,
                    n_resolution=n_resolution,
                    beta=beta,
                    pairing_symmetry=pairing_symmetry,
                    w_list=w_list,
                    w_conductivity_range=w_conductivity_range,
                    dw_values=[0.01, 0.05, 0.1],
                    U_disorder=U_disorder,
                    U_impurity=U_impurity,
                    diag_method="K0",
                    max_iter=mu_tune_max_iter,
                )
                mu = mu_found
            end

            row = summarize_homogeneous_run(
                data,
                info;
                sweep_name=String(sweep),
                sweep_value=sv,
                beta=beta,
                mu=mu,
            )
        end
        push!(rows, row)

        @printf(
            "   n=%.6g, |Delta_s|=%.6g, stiff_low_q=%.6g, converged=%s, iter=%d\n",
            row.n,
            row.delta_s_abs_mean,
            row.stiffness_lowest_q,
            string(row.converged),
            row.iter,
        )
    end

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    csv_path = joinpath(output_dir, "local_mf_summary_$(timestamp).csv")
    write_summary_csv(csv_path, rows)

    println("\nSaved summary: $(abspath(csv_path))")
    return rows, csv_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    # =========================
    # Local run configuration
    # =========================
    solution_mode = :bdg_real_space      # :bdg_real_space or :homogeneous_kspace
    sweep = :beta                        # :beta or :mu
    sweep_values = collect(2.0:2.0:20.0)

    fixed_beta = 10.0                    # used when sweep=:mu
    fixed_mu = -1.7                      # used when sweep=:beta

    # If target_n is set, each beta point will tune mu locally to match this filling.
    target_n = nothing                   # e.g. 0.875, or keep nothing
    mu_lims = [-4.0, 0.0]
    n_resolution = 5e-3

    L = 10
    L_SC = 1
    U = 4.0
    t = 1.0
    V = 0.0
    seed = 12345

    initial_guess = "uniform"
    U_disorder = "uniform"
    pairing_symmetry = "s-wave"
    U_impurity = 0.0

    max_iter = 5000
    tol = 1e-6
    mixing_a = 0.6

    save_data = false
    output_dir = joinpath(@__DIR__, "demo_results", "local_mf")

    run_local_mf_sweep(
        ;
        solution_mode=solution_mode,
        sweep=sweep,
        sweep_values=sweep_values,
        fixed_beta=fixed_beta,
        fixed_mu=fixed_mu,
        target_n=target_n,
        mu_lims=mu_lims,
        n_resolution=n_resolution,
        L=L,
        L_SC=L_SC,
        U=U,
        t=t,
        V=V,
        seed=seed,
        initial_guess=initial_guess,
        U_disorder=U_disorder,
        pairing_symmetry=pairing_symmetry,
        U_impurity=U_impurity,
        max_iter=max_iter,
        tol=tol,
        mixing_a=mixing_a,
        save_data=save_data,
        output_dir=output_dir,
    )
end
