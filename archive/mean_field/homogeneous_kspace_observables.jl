# Homogeneous k-space mean-field observables workflow.
#
# This file builds homogeneous pairing fields from k-space equations and
# reuses the standard lattice measurement pipeline through make_measurement_K!.

if !(isdefined(@__MODULE__, :SimData) && isdefined(@__MODULE__, :Lattice_Data) && isdefined(@__MODULE__, :MCMF_action_K) && isdefined(@__MODULE__, :Lattice_update!) && isdefined(@__MODULE__, :make_measurement_K!) && isdefined(@__MODULE__, :saddle_point_delta) && isdefined(@__MODULE__, :running_dwave_MF))
    include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))
end

using Dates
using Statistics

function _first_or_nan(vec)
    return isempty(vec) ? NaN : vec[1]
end

function _first_lxx_qy_or_nan(data::LatticeData)
    if isempty(data.measurement_data.Lxx_qy)
        return NaN
    end
    first_entry = data.measurement_data.Lxx_qy[1]
    if isempty(first_entry)
        return NaN
    end
    return first_entry[1]
end

function _scalar_from_delta_result(delta_result)
    delta_value = delta_result
    if delta_result isa Tuple
        delta_value = delta_result[1]
    elseif delta_result isa AbstractArray
        if isempty(delta_result)
            delta_value = 0.0
        else
            delta_value = delta_result[1]
        end
    end
    return Float64(real(delta_value))
end

function _build_local_sim_data(
    ;
    L,
    L_SC,
    U,
    t,
    V,
    seed,
    mu,
    beta,
    pairing_symmetry,
    w_list,
    w_conductivity_range,
    dw_values,
    U_disorder,
    U_impurity,
    diag_method,
)
    return SimData(
        Lx=Int64(L),
        Ly=Int64(L),
        SC_L=Int64(L_SC),
        U=Float64(U),
        t=Float64(t),
        V=Float64(V),
        seed=Int64(seed),
        mu=Float64(mu),
        beta=Float64(beta),
        num_measure=1,
        num_thermal=0,
        measuring_interval=1,
        saving_interval=1,
        w_list=Float64.(w_list),
        guess="uniform",
        U_disorder=String(U_disorder),
        diag_method=String(diag_method),
        pairing_symmetry=String(pairing_symmetry),
        directory=".",
        num_cores=1,
        num_threads=Threads.nthreads(),
        start_time=Dates.now(),
        w_conductivity_range=Float64.(w_conductivity_range),
        U_impurity=Float64(U_impurity),
        loop_random=false,
        dw_values=Float64.(dw_values),
    )
end

function build_homogeneous_delta(
    sim_data::SimData;
    d_min=1e-4,
    dwave_max_iter=1000,
    dwave_tol=1e-8,
    dwave_mix=0.6,
    Ux_override=nothing,
    Uy_override=nothing,
)
    N = sim_data.N
    pairing_symmetry = sim_data.pairing_symmetry

    Delta_s = zeros(ComplexF64, N)
    Delta_dx = zeros(ComplexF64, N)
    Delta_dy = zeros(ComplexF64, N)

    delta_summary = nothing

    if pairing_symmetry == "s-wave"
        delta_result = saddle_point_delta(sim_data; d_min=d_min)
        delta_scalar = _scalar_from_delta_result(delta_result)
        Delta_s .= complex(delta_scalar, 0.0)
        delta_summary = (
            mode="s-wave",
            delta=delta_scalar,
            source=delta_result,
        )
    elseif pairing_symmetry in ["d-wave", "mayr-d-wave", "d-p-wave"]
        Ux = Ux_override === nothing ? sim_data.U : Float64(Ux_override)
        Uy = Uy_override === nothing ? sim_data.U : Float64(Uy_override)

        dwave_result = running_dwave_MF(
            sim_data.Lx,
            sim_data.Ly,
            sim_data.t,
            sim_data.mu,
            sim_data.beta;
            Ux=Ux,
            Uy=Uy,
            max_iter=dwave_max_iter,
            tol=dwave_tol,
            a=dwave_mix,
        )

        dx = ComplexF64(dwave_result[1][1])
        dy_raw = ComplexF64(dwave_result[1][2])
        dy = pairing_symmetry == "d-p-wave" ? -dx : dy_raw

        Delta_dx .= dx
        Delta_dy .= dy

        residual_max = maximum(abs.(dwave_result[2]))
        delta_summary = (
            mode=pairing_symmetry,
            Delta_dx=dx,
            Delta_dy=dy,
            Delta_dy_raw=dy_raw,
            Ux=Ux,
            Uy=Uy,
            residual_max=residual_max,
            converged=residual_max < dwave_tol,
            iterations=length(dwave_result[3]),
        )
    else
        error("pairing symmetry $(pairing_symmetry) is not supported by build_homogeneous_delta")
    end

    return (
        Delta_s=Delta_s,
        Delta_dx=Delta_dx,
        Delta_dy=Delta_dy,
        delta_summary=delta_summary,
    )
end

function run_homogeneous_kspace_observables(
    sim_data::SimData;
    measurement_ix=1,
    compute_k0=true,
    kwargs...
)
    data = Lattice_Data(sim_data)

    homogeneous = build_homogeneous_delta(sim_data; kwargs...)
    Delta_s = homogeneous.Delta_s
    Delta_dx = homogeneous.Delta_dx
    Delta_dy = homogeneous.Delta_dy

    S, energies, U_matrix = MCMF_action_K(data, Delta_s, Delta_dx, Delta_dy)
    Lattice_update!(data, Delta_s, Delta_dx, Delta_dy, S, energies, U_matrix)

    if compute_k0 && sim_data.diag_method == "K0"
        S_K0, _, _ = MCMF_action_K0(data, Delta_s, Delta_dx, Delta_dy)
        data.current_state.S_K0 = S_K0
    end

    measurement_ix_int = Int(measurement_ix)
    if measurement_ix_int < 1
        measurement_ix_int = 1
    end

    make_measurement_K!(data, measurement_ix_int)
    data.current_state.num_current_measurments = max(data.current_state.num_current_measurments, measurement_ix_int)

    info = (
        pairing_symmetry=sim_data.pairing_symmetry,
        delta_summary=homogeneous.delta_summary,
        measurement_ix=measurement_ix_int,
        computed_k0=(compute_k0 && sim_data.diag_method == "K0"),
    )

    return data, info
end

function run_local_homogeneous_kspace(
    ;
    L=10,
    L_SC=1,
    U=4.0,
    t=1.0,
    V=0.0,
    seed=12345,
    mu=-1.7,
    beta=10.0,
    pairing_symmetry="s-wave",
    w_list=collect(-3.0:0.1:3.0),
    w_conductivity_range=collect(0.0:0.1:4.0),
    dw_values=[0.01, 0.05, 0.1],
    U_disorder="uniform",
    U_impurity=0.0,
    diag_method="K0",
    print_summary=true,
    kwargs...
)
    sim_data = _build_local_sim_data(
        L=L,
        L_SC=L_SC,
        U=U,
        t=t,
        V=V,
        seed=seed,
        mu=mu,
        beta=beta,
        pairing_symmetry=pairing_symmetry,
        w_list=w_list,
        w_conductivity_range=w_conductivity_range,
        dw_values=dw_values,
        U_disorder=U_disorder,
        U_impurity=U_impurity,
        diag_method=diag_method,
    )

    data, info = run_homogeneous_kspace_observables(sim_data; measurement_ix=1, compute_k0=true, kwargs...)

    n1 = _first_or_nan(data.measurement_data.n)
    kx1 = _first_or_nan(data.measurement_data.Kx)
    delta_s_mean = mean(abs.(data.current_state.Delta_s))
    delta_dx_mean = mean(abs.(data.current_state.Delta_dx))
    delta_dy_mean = mean(abs.(data.current_state.Delta_dy))

    if print_summary
        println(
            "homogeneous-kspace pairing=$(pairing_symmetry) L=$(L) beta=$(beta) mu=$(mu) |Delta_s|=$(delta_s_mean) |Delta_dx|=$(delta_dx_mean) |Delta_dy|=$(delta_dy_mean) n=$(n1) Kx=$(kx1)"
        )
    end

    return data, info
end

function run_local_homogeneous_kspace_target_n(
    ;
    L=10,
    L_SC=1,
    U=4.0,
    t=1.0,
    V=0.0,
    seed=12345,
    mu_lims=[-4.0, 0.0],
    target_n=0.875,
    n_resolution=5e-3,
    beta=10.0,
    pairing_symmetry="s-wave",
    w_list=collect(-3.0:0.1:3.0),
    w_conductivity_range=collect(0.0:0.1:4.0),
    dw_values=[0.01, 0.05, 0.1],
    U_disorder="uniform",
    U_impurity=0.0,
    diag_method="K0",
    max_iter=30,
    print_summary=true,
    kwargs...
)
    if length(mu_lims) != 2
        error("mu_lims must contain exactly two values [mu_low, mu_high]")
    end
    mu_low = Float64(mu_lims[1])
    mu_high = Float64(mu_lims[2])
    if mu_low == mu_high
        error("mu_lims endpoints must be different")
    end

    eval_at_mu(mu_try, seed_offset=0) = begin
        data, info = run_local_homogeneous_kspace(
            L=L,
            L_SC=L_SC,
            U=U,
            t=t,
            V=V,
            seed=seed + seed_offset,
            mu=mu_try,
            beta=beta,
            pairing_symmetry=pairing_symmetry,
            w_list=w_list,
            w_conductivity_range=w_conductivity_range,
            dw_values=dw_values,
            U_disorder=U_disorder,
            U_impurity=U_impurity,
            diag_method=diag_method,
            print_summary=false,
            kwargs...,
        )
        n_val = _first_or_nan(data.measurement_data.n)
        return data, info, n_val
    end

    data_low, info_low, n_low = eval_at_mu(mu_low, 0)
    data_high, info_high, n_high = eval_at_mu(mu_high, 1)

    best_data, best_info, best_mu, best_n = abs(n_low - target_n) <= abs(n_high - target_n) ?
        (data_low, info_low, mu_low, n_low) : (data_high, info_high, mu_high, n_high)

    # If target is not bracketed, return closest edge.
    bracketed = (n_low - target_n) * (n_high - target_n) <= 0
    if !bracketed
        tuned_info = merge(best_info, (
            mu_tuning=(
                target_n=target_n,
                best_mu=best_mu,
                best_n=best_n,
                converged=abs(best_n - target_n) <= n_resolution,
                bracketed=false,
                iterations=0,
            ),
        ))
        if print_summary
            println("homogeneous-kspace target_n not bracketed: returning closest edge mu=$(best_mu), n=$(best_n)")
        end
        return best_data, tuned_info, best_mu, best_n
    end

    converged = abs(best_n - target_n) <= n_resolution
    iter_count = 0
    for iter in 1:max_iter
        iter_count = iter
        if converged
            break
        end

        mu_try = (mu_low + mu_high) / 2
        dn = n_high - n_low
        if isfinite(dn) && abs(dn) > 1e-12
            mu_secant = mu_low + (target_n - n_low) * (mu_high - mu_low) / dn
            lo = min(mu_low, mu_high)
            hi = max(mu_low, mu_high)
            if isfinite(mu_secant) && lo <= mu_secant <= hi
                mu_try = mu_secant
            end
        end

        data_try, info_try, n_try = eval_at_mu(mu_try, iter + 1)

        if abs(n_try - target_n) < abs(best_n - target_n)
            best_data, best_info, best_mu, best_n = data_try, info_try, mu_try, n_try
        end

        if abs(n_try - target_n) <= n_resolution
            best_data, best_info, best_mu, best_n = data_try, info_try, mu_try, n_try
            converged = true
            break
        end

        if (n_low - target_n) * (n_try - target_n) <= 0
            mu_high = mu_try
            n_high = n_try
        else
            mu_low = mu_try
            n_low = n_try
        end
    end

    tuned_info = merge(best_info, (
        mu_tuning=(
            target_n=target_n,
            best_mu=best_mu,
            best_n=best_n,
            converged=converged,
            bracketed=true,
            iterations=iter_count,
            n_resolution=n_resolution,
            mu_lims=[mu_lims[1], mu_lims[2]],
        ),
    ))

    if print_summary
        println(
            "homogeneous-kspace target_n=$(target_n) beta=$(beta) best_mu=$(best_mu) best_n=$(best_n) converged=$(converged) iters=$(iter_count)"
        )
    end

    return best_data, tuned_info, best_mu, best_n
end

function run_homogeneous_kspace_smoke_tests()
    data_s, info_s = run_local_homogeneous_kspace(
        L=6,
        beta=6.0,
        mu=-1.5,
        seed=12345,
        pairing_symmetry="s-wave",
    )

    data_d, info_d = run_local_homogeneous_kspace(
        L=6,
        beta=10.0,
        mu=-1.2,
        seed=12346,
        pairing_symmetry="d-wave",
    )

    summary_s = (
        pairing=info_s.pairing_symmetry,
        delta_s_abs_mean=mean(abs.(data_s.current_state.Delta_s)),
        delta_dx_abs_mean=mean(abs.(data_s.current_state.Delta_dx)),
        delta_dy_abs_mean=mean(abs.(data_s.current_state.Delta_dy)),
        n1=_first_or_nan(data_s.measurement_data.n),
        Kx1=_first_or_nan(data_s.measurement_data.Kx),
        Lxx_qy0=_first_lxx_qy_or_nan(data_s),
    )

    summary_d = (
        pairing=info_d.pairing_symmetry,
        delta_s_abs_mean=mean(abs.(data_d.current_state.Delta_s)),
        delta_dx_abs_mean=mean(abs.(data_d.current_state.Delta_dx)),
        delta_dy_abs_mean=mean(abs.(data_d.current_state.Delta_dy)),
        n1=_first_or_nan(data_d.measurement_data.n),
        Kx1=_first_or_nan(data_d.measurement_data.Kx),
        Lxx_qy0=_first_lxx_qy_or_nan(data_d),
    )

    println(
        "smoke s-wave: pairing=$(summary_s.pairing), |Delta_s|=$(summary_s.delta_s_abs_mean), |Delta_dx|=$(summary_s.delta_dx_abs_mean), |Delta_dy|=$(summary_s.delta_dy_abs_mean), n=$(summary_s.n1), Kx=$(summary_s.Kx1), Lxx_qy0=$(summary_s.Lxx_qy0)"
    )
    println(
        "smoke d-wave: pairing=$(summary_d.pairing), |Delta_s|=$(summary_d.delta_s_abs_mean), |Delta_dx|=$(summary_d.delta_dx_abs_mean), |Delta_dy|=$(summary_d.delta_dy_abs_mean), n=$(summary_d.n1), Kx=$(summary_d.Kx1), Lxx_qy0=$(summary_d.Lxx_qy0)"
    )

    return (s_wave=summary_s, d_wave=summary_d)
end
