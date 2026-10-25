"""
    Unified plotting helper methods.

These overload existing plotting helper names so they can work directly with the
unified data flow:
- sim_info (merged dictionary)
- wanted_n
- U
- geometry_pairs = [(L, TCA_size), ...]

No intermediate sims_K0/sims_TCA/full_series/all_series objects are required.

Prerequisites loaded in the session:
- data_summery.jl (for get_sims_all and get_sim_from_sims_all)
- sim_num_control.jl (for get_obs_vs_T)
- plotting_helpers.jl (optional; these are overloads, not replacements)
"""

"""
Build unified series tuples from geometry pairs.
Returns Vector of (label, sim_beta_dict) where each sim is keyed by beta.
"""
function build_unified_series(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs;
    prefer_full::Bool=true,
    sims_all=nothing,
    include_missing::Bool=false,
)
    sims_all_local = sims_all === nothing ? get_sims_all(sim_info; wanted_n=Float64(wanted_n)) : sims_all

    series = Tuple{String, Any}[]
    for (L, TCA_size) in geometry_pairs
        sim, method = get_sim_from_sims_all(sims_all_local, U, L, TCA_size; prefer_full=prefer_full)
        if sim === nothing
            include_missing && push!(series, ("L=$(L), TCA=$(TCA_size), MISSING", Dict{Any, Any}()))
            continue
        end

        lbl = method == "FULL" ? "L=$(L), FULL" : "L=$(L), TCA=$(TCA_size)"
        push!(series, (lbl, sim))
    end

    sort!(series, by = x -> begin
        lbl = x[1]
        m_l = match(r"L=(\d+)", lbl)
        m_tca = match(r"TCA=(\d+)", lbl)
        L = m_l === nothing ? typemax(Int) : parse(Int, m_l[1])
        tca = m_tca === nothing ? typemax(Int) : parse(Int, m_tca[1])
        (L, tca)
    end)

    return series
end

"""
Collect all available (L, TCA_size) pairs for a specific U from unified sims_all.
"""
function get_available_geometry_pairs(
    sims_all::Dict{Any, Any},
    U,
)
    haskey(sims_all, U) || return Tuple{Int, Int}[]

    pairs = Tuple{Int, Int}[]
    for (tca, sims_tca) in sims_all[U]
        for L in keys(sims_tca)
            push!(pairs, (Int(L), Int(tca)))
        end
    end

    unique!(pairs)
    sort!(pairs, by = x -> (x[1], x[2]))
    return pairs
end

"""
    _get_obs_safe(sim, obs_name; obs_index=nothing) -> (Ts, values, errors)

`get_obs_vs_T` (component `obs_index` if given), or empty vectors when `sim` is `nothing`.
"""
function _get_obs_safe(sim, obs_name; obs_index=nothing)
    sim === nothing && return Float64[], Float64[], Float64[]
    if obs_index === nothing
        return get_obs_vs_T(sim, obs_name)
    end
    return get_obs_vs_T(sim, obs_name, obs_index)
end

"""
    _sim_has_observable(sim, obs_name) -> Bool

True if any point of the curve `sim` (β => point) has `obs_name` among its means.
"""
function _sim_has_observable(sim, obs_name)
    sim === nothing && return false
    isempty(sim) && return false

    for (_, sim_beta) in sim
        haskey(sim_beta, "sim_dict") || continue
        sim_dict = sim_beta["sim_dict"]
        haskey(sim_dict, "means") || continue
        means = sim_dict["means"]
        haskey(means, obs_name) && return true
    end

    return false
end

"""
    _find_closest_beta_key(sim, beta_target)

The β key of the curve `sim` equal or closest to `beta_target` (`nothing` if empty).
"""
function _find_closest_beta_key(sim::Dict{Any, Any}, beta_target::Real)
    betas = sort(collect(keys(sim)))
    isempty(betas) && return nothing

    if beta_target in betas
        return beta_target
    end

    diffs = [abs(Float64(b) - Float64(beta_target)) for b in betas]
    ix = argmin(diffs)
    return betas[ix]
end

"""
    _connected_psl_series(sim, beta_target; normalize_by_N=true, subtract_delta2=true, min_l=0)

The pair correlation Ps(l) of the curve point closest to `beta_target`, against the
separation l, for the connected-correlation plot: optionally divided by N and with the
long-range part |⟨Δ̂⟩|² (`Delta_operator`) subtracted; separations below `min_l` dropped. `nothing` if the
point or its `Ps_l` is missing.
"""
function _connected_psl_series(
    sim::Dict{Any, Any},
    beta_target::Real;
    normalize_by_N::Bool=true,
    subtract_delta2::Bool=true,
    min_l::Int=0,
)
    beta_use = _find_closest_beta_key(sim, beta_target)
    beta_use === nothing && return nothing

    sim_beta = sim[beta_use]
    haskey(sim_beta, "sim_dict") || return nothing
    sim_dict = sim_beta["sim_dict"]
    haskey(sim_dict, "means") || return nothing
    means = sim_dict["means"]

    if !haskey(means, "Ps_l")
        @warn "Missing Ps_l observable; skipping beta" beta_target=beta_target beta_used=beta_use
        return nothing
    end

    ps_val_raw = means["Ps_l"]["value"]
    ps_err_raw = means["Ps_l"]["error"]
    ps_val_raw isa AbstractVector || return nothing
    ps_err_raw isa AbstractVector || return nothing

    vals = Float64.(ps_val_raw)
    errs = Float64.(ps_err_raw)

    lx = get(sim_dict, "Lx", 0)
    ly = get(sim_dict, "Ly", 0)
    N = (lx > 0 && ly > 0) ? (lx * ly) : length(vals)
    norm_fac = normalize_by_N ? Float64(N) : 1.0

    vals ./= norm_fac
    errs ./= norm_fac

    if subtract_delta2
        if !haskey(means, "Delta_operator")
            @warn "Missing Delta_operator observable; using Ps_l/N only" beta_target=beta_target beta_used=beta_use
        else
            delta_val_raw = means["Delta_operator"]["value"]
            delta_err_raw = means["Delta_operator"]["error"]

            delta_abs = abs(Float64(delta_val_raw))
            delta_err = abs(Float64(delta_err_raw))

            vals .-= delta_abs^2
            errs = sqrt.(errs.^2 .+ (2.0 * delta_abs * delta_err)^2)
        end
    end

    ls = collect(0:(length(vals) - 1))
    max_l = length(vals) - 1
    if min_l > 0
        keep = findall(>=(min_l), ls)
        ls = ls[keep]
        vals = vals[keep]
        errs = errs[keep]
        if isempty(ls)
            @warn "All l points removed by min_l filter" min_l=min_l max_available_l=max_l
            return nothing
        end
    end

    t_use = 1.0 / Float64(beta_use)
    return ls, vals, errs, Float64(beta_use), t_use
end

"""
Plot connected pair-correlation decay versus l for selected beta values.

Plots Ps_l/N - |Delta_operator|^2 (or Ps_l/N if subtract_delta2=false) for one
selected simulation identified by (n, U, L, TCA_size).

Usage:
plot_paircorr_connected_vs_l(sim_info, wanted_n, U, L, TCA_size, beta_list; ...)
"""
function plot_paircorr_connected_vs_l(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    L::Int,
    TCA_size::Int,
    beta_list::AbstractVector;
    prefer_full::Bool=true,
    sims_all=nothing,
    normalize_by_N::Bool=true,
    subtract_delta2::Bool=true,
    min_l::Int=0,
    xscale=:identity,
    yscale=:identity,
    filter_nonpositive_for_log::Bool=true,
    exact_beta::Bool=false,
    beta_match_tol::Float64=1e-8,
)
    sims_all_local = sims_all === nothing ? get_sims_all(sim_info; wanted_n=Float64(wanted_n)) : sims_all
    sim, method = get_sim_from_sims_all(sims_all_local, U, L, TCA_size; prefer_full=prefer_full)
    sim === nothing && error("No simulation found for n=$(wanted_n), U=$(U), L=$(L), TCA=$(TCA_size).")

    norm_lbl = normalize_by_N ? "Ps_l / N" : "Ps_l"
    ylbl = subtract_delta2 ? "$(norm_lbl) - |Delta_operator|^2" : norm_lbl
    mth_lbl = method == "FULL" ? "FULL" : "TCA=$(TCA_size)"

    fig = plot(
        xlabel="l",
        ylabel=ylbl,
        title="Pair-correlation decay: n=$(wanted_n), U=$(U), L=$(L), $(mth_lbl)",
        legend=:topright,
        size=(900, 560),
        left_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
        xscale=xscale,
        yscale=yscale,
    )

    plotted = 0
    global_min_positive = Inf
    global_max_positive = 0.0
    for beta_target in beta_list
        series = _connected_psl_series(
            sim,
            Float64(beta_target);
            normalize_by_N=normalize_by_N,
            subtract_delta2=subtract_delta2,
            min_l=min_l,
        )

        if series === nothing
            @warn "Could not build connected Ps_l series for beta; skipping" beta_target=beta_target
            continue
        end

        ls, vals, errs, beta_use, t_use = series
        if abs(Float64(beta_use) - Float64(beta_target)) > beta_match_tol
            if exact_beta
                @warn "Requested beta not found exactly; skipping due to exact_beta=true" beta_target=beta_target nearest_beta=beta_use
                continue
            else
                @warn "Requested beta not found exactly; using nearest available beta" beta_target=beta_target nearest_beta=beta_use
            end
        end
        isempty(ls) && continue

        keep_finite = findall(i -> isfinite(vals[i]) && isfinite(errs[i]) && errs[i] >= 0.0, eachindex(vals))
        if length(keep_finite) < length(vals)
            @warn "Removed non-finite/invalid points" beta_used=beta_use removed_points=(length(vals) - length(keep_finite))
            ls = ls[keep_finite]
            vals = vals[keep_finite]
            errs = errs[keep_finite]
        end

        if yscale == :log10 && filter_nonpositive_for_log
            keep = findall(v -> v > 0.0, vals)
            removed = length(vals) - length(keep)
            if removed > 0
                @warn "Removed non-positive points for log10 y-scale" beta_used=beta_use removed_points=removed
            end
            if isempty(keep)
                @warn "No positive points left for log10 plot; skipping beta" beta_used=beta_use min_value=minimum(vals) max_value=maximum(vals)
                continue
            end
            ls = ls[keep]
            vals = vals[keep]
            errs = errs[keep]

            # For log-scale, keep lower error bar positive by clipping oversized y-errors.
            for i in eachindex(vals)
                max_err = 0.95 * vals[i]
                if errs[i] > max_err
                    errs[i] = max_err
                end
            end

            local_min = minimum(vals)
            local_max = maximum(vals)
            if isfinite(local_min) && local_min > 0.0
                global_min_positive = min(global_min_positive, local_min)
            end
            if isfinite(local_max)
                global_max_positive = max(global_max_positive, local_max)
            end
        end

        plot!(
            fig,
            ls,
            vals;
            yerr=errs,
            m=:circle,
            ms=4,
            lw=1.8,
            label="beta=$(round(beta_use, sigdigits=4)) (T=$(round(t_use, sigdigits=4)))",
        )
        plotted += 1
    end

    if yscale == :log10 && plotted > 0 && isfinite(global_min_positive) && global_min_positive > 0.0 && global_max_positive > 0.0
        ymin = max(global_min_positive * 0.8, eps(Float64))
        ymax = global_max_positive * 1.2
        if ymax > ymin
            plot!(fig, ylims=(ymin, ymax))
        end
    end

    if plotted == 0
        if yscale == :log10
            @warn "No valid beta series were plotted for log10. Try smaller min_l, yscale=:identity, or subtract_delta2=false." wanted_n=wanted_n U=U L=L TCA_size=TCA_size
        else
            @warn "No valid beta series were plotted." wanted_n=wanted_n U=U L=L TCA_size=TCA_size
        end
    end
    return fig
end

"""
    _phase_corr_component_series(sim, beta_target; component_ix=1, min_l=0)

The phase correlation `Phase_Corr` (component `component_ix`) of the curve point
closest to `beta_target`, against the separation l (from `min_l`). `nothing` if missing.
"""
function _phase_corr_component_series(
    sim::Dict{Any, Any},
    beta_target::Real;
    component_ix::Int=1,
    min_l::Int=0,
)
    beta_use = _find_closest_beta_key(sim, beta_target)
    beta_use === nothing && return nothing

    sim_beta = sim[beta_use]
    haskey(sim_beta, "sim_dict") || return nothing
    sim_dict = sim_beta["sim_dict"]
    haskey(sim_dict, "means") || return nothing
    means = sim_dict["means"]

    if !haskey(means, "Phase_Corr")
        @warn "Missing Phase_Corr observable; skipping beta" beta_target=beta_target beta_used=beta_use
        return nothing
    end

    raw_vals = means["Phase_Corr"]["value"]
    raw_errs = means["Phase_Corr"]["error"]
    raw_vals isa AbstractVector || return nothing
    raw_errs isa AbstractVector || return nothing
    isempty(raw_vals) && return nothing

    vals = Float64[]
    errs = Float64[]

    first_val = first(raw_vals)
    if first_val isa AbstractArray
        for i in eachindex(raw_vals)
            v = raw_vals[i]
            e = raw_errs[i]
            if length(v) < component_ix || length(e) < component_ix
                push!(vals, NaN)
                push!(errs, NaN)
                continue
            end
            push!(vals, Float64(v[component_ix]))
            push!(errs, abs(Float64(e[component_ix])))
        end
    else
        if component_ix != 1
            @warn "Phase_Corr appears scalar-per-l; forcing component 1" requested_component=component_ix
        end
        vals = Float64.(raw_vals)
        errs = abs.(Float64.(raw_errs))
    end

    ls = collect(0:(length(vals) - 1))
    max_l = length(vals) - 1
    if min_l > 0
        keep = findall(>=(min_l), ls)
        ls = ls[keep]
        vals = vals[keep]
        errs = errs[keep]
        if isempty(ls)
            @warn "All l points removed by min_l filter" min_l=min_l max_available_l=max_l
            return nothing
        end
    end

    t_use = 1.0 / Float64(beta_use)
    return ls, vals, errs, Float64(beta_use), t_use
end

"""
Plot Phase_Corr(l) for selected beta values, using the s-wave component by default.

This is analogous to plot_paircorr_connected_vs_l but for Phase_Corr.

Notes:
- component_ix=1 corresponds to s-wave.
- If Phase_Corr is stored as scalar-per-l, component_ix is ignored and scalar values are used.
"""
function plot_phase_corr_vs_l(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    L::Int,
    TCA_size::Int,
    beta_list::AbstractVector;
    component_ix::Int=1,
    prefer_full::Bool=true,
    sims_all=nothing,
    min_l::Int=0,
    xscale=:identity,
    yscale=:identity,
    filter_nonpositive_for_log::Bool=true,
    exact_beta::Bool=false,
    beta_match_tol::Float64=1e-8,
)
    sims_all_local = sims_all === nothing ? get_sims_all(sim_info; wanted_n=Float64(wanted_n)) : sims_all
    sim, method = get_sim_from_sims_all(sims_all_local, U, L, TCA_size; prefer_full=prefer_full)
    sim === nothing && error("No simulation found for n=$(wanted_n), U=$(U), L=$(L), TCA=$(TCA_size).")

    comp_lbl = component_ix == 1 ? "s-wave" : "component $(component_ix)"
    mth_lbl = method == "FULL" ? "FULL" : "TCA=$(TCA_size)"

    fig = plot(
        xlabel="l",
        ylabel="Phase_Corr ($(comp_lbl))",
        title="Phase correlation decay: n=$(wanted_n), U=$(U), L=$(L), $(mth_lbl)",
        legend=:topright,
        size=(900, 560),
        left_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
        xscale=xscale,
        yscale=yscale,
    )

    plotted = 0
    global_min_positive = Inf
    global_max_positive = 0.0
    for beta_target in beta_list
        series = _phase_corr_component_series(
            sim,
            Float64(beta_target);
            component_ix=component_ix,
            min_l=min_l,
        )

        if series === nothing
            @warn "Could not build Phase_Corr series for beta; skipping" beta_target=beta_target
            continue
        end

        ls, vals, errs, beta_use, t_use = series
        if abs(Float64(beta_use) - Float64(beta_target)) > beta_match_tol
            if exact_beta
                @warn "Requested beta not found exactly; skipping due to exact_beta=true" beta_target=beta_target nearest_beta=beta_use
                continue
            else
                @warn "Requested beta not found exactly; using nearest available beta" beta_target=beta_target nearest_beta=beta_use
            end
        end
        isempty(ls) && continue

        keep_finite = findall(i -> isfinite(vals[i]) && isfinite(errs[i]) && errs[i] >= 0.0, eachindex(vals))
        if length(keep_finite) < length(vals)
            @warn "Removed non-finite/invalid points" beta_used=beta_use removed_points=(length(vals) - length(keep_finite))
            ls = ls[keep_finite]
            vals = vals[keep_finite]
            errs = errs[keep_finite]
        end

        if yscale == :log10 && filter_nonpositive_for_log
            keep = findall(v -> v > 0.0, vals)
            removed = length(vals) - length(keep)
            if removed > 0
                @warn "Removed non-positive points for log10 y-scale" beta_used=beta_use removed_points=removed
            end
            if isempty(keep)
                @warn "No positive points left for log10 plot; skipping beta" beta_used=beta_use
                continue
            end

            ls = ls[keep]
            vals = vals[keep]
            errs = errs[keep]

            for i in eachindex(vals)
                max_err = 0.95 * vals[i]
                if errs[i] > max_err
                    errs[i] = max_err
                end
            end

            local_min = minimum(vals)
            local_max = maximum(vals)
            if isfinite(local_min) && local_min > 0.0
                global_min_positive = min(global_min_positive, local_min)
            end
            if isfinite(local_max)
                global_max_positive = max(global_max_positive, local_max)
            end
        end

        plot!(
            fig,
            ls,
            vals;
            yerr=errs,
            m=:circle,
            ms=4,
            lw=1.8,
            label="beta=$(round(beta_use, sigdigits=4)) (T=$(round(t_use, sigdigits=4)))",
        )
        plotted += 1
    end

    if yscale == :log10 && plotted > 0 && isfinite(global_min_positive) && global_min_positive > 0.0 && global_max_positive > 0.0
        ymin = max(global_min_positive * 0.8, eps(Float64))
        ymax = global_max_positive * 1.2
        if ymax > ymin
            plot!(fig, ylims=(ymin, ymax))
        end
    end

    if plotted == 0
        if yscale == :log10
            @warn "No valid Phase_Corr series were plotted for log10. Try smaller min_l or yscale=:identity." wanted_n=wanted_n U=U L=L TCA_size=TCA_size
        else
            @warn "No valid Phase_Corr series were plotted." wanted_n=wanted_n U=U L=L TCA_size=TCA_size
        end
    end

    return fig
end

"""
Convenience wrapper for s-wave only (component 1).
"""
function plot_phase_corr_swave_vs_l(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    L::Int,
    TCA_size::Int,
    beta_list::AbstractVector;
    kwargs...
)
    return plot_phase_corr_vs_l(
        sim_info,
        wanted_n,
        U,
        L,
        TCA_size,
        beta_list;
        component_ix=1,
        kwargs...
    )
end

"""
    finite_size_scaling_fit(sim_info, wanted_n, U, obs, beta_target; ...)

Extract finite-size data from `sim_info` at the given β (per TCA and optionally FULL),
then fit `y ≈ A · L^a` via `fit_finite_size_power_law`, where `L = sqrt(Lx·Ly)`.

# Keywords
- `w_ix`, `geometry_pairs`, `prefer_full`, `sims_all`, `include_full_reference`, `units_mult`:
  same semantics as in `plot_finite_size_scaling`.
- `fixed_a`: if a `Real`, hold the exponent fixed during the fit.

# Returns
NamedTuple `(; tca_fits, full_fit, tca_points, full_points)` where each point is
`(L, N, y, dy, beta)` with `N = Lx·Ly`, `L = sqrt(Lx·Ly)`.
"""
function finite_size_scaling_fit(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    obs,
    beta_target;
    w_ix::Int=1,
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    sims_all=nothing,
    include_full_reference::Bool=true,
    units_mult::Union{Nothing, Float64}=nothing,
    fixed_a::Union{Nothing, Real}=nothing,
    fit_mode::Symbol=:pooled,
)
    fit_mode in (:pooled, :per_series, :both) ||
        error("fit_mode must be :pooled, :per_series, or :both (got $(fit_mode)).")
    sims_all_local = sims_all === nothing ? get_sims_all(sim_info; wanted_n=Float64(wanted_n)) : sims_all
    geometry_pairs_local = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : geometry_pairs
    isempty(geometry_pairs_local) && error("No geometry pairs found for U=$(U), n~$(wanted_n).")

    all_series = build_unified_series(
        sim_info, wanted_n, U, geometry_pairs_local;
        prefer_full=prefer_full, sims_all=sims_all_local,
    )

    tca_points = Dict{Int, Vector{NTuple{5, Float64}}}()
    full_points = NTuple{5, Float64}[]

    for (lbl, sim) in all_series
        isempty(sim) && continue
        _sim_has_observable(sim, obs) || continue

        betas = sort(collect(keys(sim)))
        isempty(betas) && continue
        beta_use = (beta_target in betas) ? beta_target : continue

        sim_beta = sim[beta_use]
        lx = sim_beta["sim_dict"]["Lx"]
        ly = sim_beta["sim_dict"]["Ly"]
        means = sim_beta["sim_dict"]["means"]
        haskey(means, obs) || continue

        v = means[obs]["value"]
        e = means[obs]["error"]
        if !isnothing(units_mult)
            v = v .* units_mult
            e = e .* units_mult
        end

        if v isa AbstractVector
            length(v) >= w_ix || continue
            y = Float64(v[w_ix])
            dy = Float64(e[w_ix])
        else
            w_ix == 1 || continue
            y = Float64(v)
            dy = Float64(e)
        end

        L = sqrt(Float64(lx) * Float64(ly))
        N = Float64(lx) * Float64(ly)
        pt = (L, N, y, dy, Float64(beta_use))

        if occursin("TCA=", lbl)
            m_tca = match(r"TCA=(\d+)", lbl)
            m_tca === nothing && continue
            tca = parse(Int, m_tca[1])
            push!(get!(tca_points, tca, NTuple{5, Float64}[]), pt)
        elseif include_full_reference && occursin("FULL", lbl)
            push!(full_points, pt)
        end
    end

    min_pts = fixed_a === nothing ? 2 : 1

    tca_fits = Dict{Int, Any}()
    full_fit = nothing
    pooled_fit = nothing

    if fit_mode === :per_series || fit_mode === :both
        for (tca, pts) in tca_points
            length(pts) >= min_pts || continue
            Ls = [p[1] for p in pts]
            ys = [p[3] for p in pts]
            σs = [p[4] for p in pts]
            try
                tca_fits[tca] = fit_finite_size_power_law(Ls, ys, σs; fixed_a=fixed_a)
            catch err
                @warn "finite_size_scaling_fit: fit failed for TCA=$tca" exception=err
            end
        end
        if length(full_points) >= min_pts
            Ls = [p[1] for p in full_points]
            ys = [p[3] for p in full_points]
            σs = [p[4] for p in full_points]
            try
                full_fit = fit_finite_size_power_law(Ls, ys, σs; fixed_a=fixed_a)
            catch err
                @warn "finite_size_scaling_fit: fit failed for FULL" exception=err
            end
        end
    end

    if fit_mode === :pooled || fit_mode === :both
        pooled_pts = NTuple{5, Float64}[]
        for pts in values(tca_points)
            append!(pooled_pts, pts)
        end
        include_full_reference && append!(pooled_pts, full_points)
        if length(pooled_pts) >= min_pts
            Ls = [p[1] for p in pooled_pts]
            ys = [p[3] for p in pooled_pts]
            σs = [p[4] for p in pooled_pts]
            try
                pooled_fit = fit_finite_size_power_law(Ls, ys, σs; fixed_a=fixed_a)
            catch err
                @warn "finite_size_scaling_fit: pooled fit failed" exception=err
            end
        end
    end

    return (; tca_fits=tca_fits, full_fit=full_fit, pooled_fit=pooled_fit,
              tca_points=tca_points, full_points=full_points)
end


"""
Internal: overlay a power-law fit curve `y = A·L^a` onto a finite-size plot.

`xaxis ∈ (:L, :inv_L, :inv_N)`. When `xscale == :log10` the curve spans the data
L-range; otherwise it is extrapolated toward `L → ∞` (i.e. `x → 0` for `:inv_L`
and `:inv_N`, or to `2·max(L)` for `:L`).
"""
function _overlay_power_law_fit!(fig, fr, xaxis::Symbol, xscale::Symbol;
                                 label::String="fit", linecolor=nothing)
    A = fr.A; a = fr.a
    L_lo = minimum(fr.L_used)
    L_hi_data = maximum(fr.L_used)

    L_hi = if xscale == :log10
        L_hi_data
    elseif xaxis === :L
        L_hi_data * 2.0
    else
        L_hi_data * 10.0
    end

    Ls_fit = exp.(range(log(L_lo), log(L_hi), length=200))
    ys_fit = A .* Ls_fit .^ a
    xs_fit = xaxis === :L     ? Ls_fit :
             xaxis === :inv_L ? 1.0 ./ Ls_fit :
                                1.0 ./ (Ls_fit .^ 2)

    a_str = fr.fixed_a ? "$(round(a, sigdigits=3)) (fixed)" :
                         "$(round(a, sigdigits=3))±$(round(fr.a_err, sigdigits=2))"
    lbl = "$label: A=$(round(A, sigdigits=3)), a=$a_str"

    if linecolor === nothing
        plot!(fig, xs_fit, ys_fit; lw=1.5, ls=:dot, label=lbl)
    else
        plot!(fig, xs_fit, ys_fit; lw=1.5, ls=:dot, label=lbl, c=linecolor)
    end
    return fig
end


"""
Unified finite-size scaling overload.

Usage:
plot_finite_size_scaling(sim_info, wanted_n, U, geometry_pairs, obs, beta_target; ...)

# Keywords (scaling-related)
- `xaxis::Symbol=:inv_N`: x-axis variable. One of `:L`, `:inv_L`, `:inv_N`.
- `fit_scaling::Bool=false`: if true, overlay a power-law fit `y = A·L^a` for each
  TCA series (and FULL) using `fit_finite_size_power_law`.
- `fixed_a::Union{Nothing,Real}=nothing`: if a number, fit `A` only with `a` held fixed.
"""
function plot_finite_size_scaling(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs,
    obs,
    beta_target;
    w_ix=1,
    include_full_reference::Bool=true,
    title::String="",
    ylabel::String="",
    units_mult::Union{Nothing, Float64}=nothing,
    units_label::String="",
    prefer_full::Bool=true,
    sims_all=nothing,
    legend_pos=:topright,
    xscale=:identity,
    yscale=:identity,
    xaxis::Symbol=:inv_N,
    fit_scaling::Bool=false,
    fixed_a::Union{Nothing, Real}=nothing,
    fit_mode::Symbol=:pooled,
)
    xaxis in (:L, :inv_L, :inv_N) ||
        error("xaxis must be one of :L, :inv_L, :inv_N (got $(xaxis)).")
    fit_mode in (:pooled, :per_series, :both) ||
        error("fit_mode must be one of :pooled, :per_series, :both (got $(fit_mode)).")

    sims_all_local = sims_all === nothing ? get_sims_all(sim_info; wanted_n=Float64(wanted_n)) : sims_all
    geometry_pairs_local = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : geometry_pairs
    isempty(geometry_pairs_local) && error("No geometry pairs found for U=$(U), n~$(wanted_n).")

    all_series = build_unified_series(
        sim_info,
        wanted_n,
        U,
        geometry_pairs_local;
        prefer_full=prefer_full,
        sims_all=sims_all_local,
    )

    # pt = (L, N, y, dy, beta) with L = sqrt(Lx·Ly), N = Lx·Ly
    tca_points = Dict{Int, Vector{NTuple{5, Float64}}}()
    full_points = NTuple{5, Float64}[]

    for (lbl, sim) in all_series
        isempty(sim) && continue

        if !_sim_has_observable(sim, obs)
            @warn "Observable not found for simulation; skipping" wanted_n=wanted_n U=U label=lbl obs_name=obs
            continue
        end

        betas = sort(collect(keys(sim)))
        isempty(betas) && continue
        beta_use = (beta_target in betas) ? beta_target : continue

        sim_beta = sim[beta_use]
        lx = sim_beta["sim_dict"]["Lx"]
        ly = sim_beta["sim_dict"]["Ly"]
        means = sim_beta["sim_dict"]["means"]
        haskey(means, obs) || continue

        v = means[obs]["value"]
        e = means[obs]["error"]

        if !isnothing(units_mult)
            v = v .* units_mult
            e = e .* units_mult
        end

        if v isa AbstractVector
            length(v) >= w_ix || continue
            y = Float64(v[w_ix])
            dy = Float64(e[w_ix])
        else
            w_ix == 1 || continue
            y = Float64(v)
            dy = Float64(e)
        end

        L = sqrt(Float64(lx) * Float64(ly))
        N = Float64(lx) * Float64(ly)
        pt = (L, N, y, dy, Float64(beta_use))

        if occursin("TCA=", lbl)
            m_tca = match(r"TCA=(\d+)", lbl)
            m_tca === nothing && continue
            tca = parse(Int, m_tca[1])
            push!(get!(tca_points, tca, NTuple{5, Float64}[]), pt)
        elseif include_full_reference && occursin("FULL", lbl)
            push!(full_points, pt)
        end
    end

    _xof(Lv::Real, Nv::Real) =
        xaxis === :L     ? Lv :
        xaxis === :inv_L ? 1.0 / Lv :
                           1.0 / Nv

    xlabel_use =
        xaxis === :L     ? "L"   :
        xaxis === :inv_L ? "1/L" :
                           "1/N"

    ylabel_use = isempty(ylabel) ? obs : ylabel
    if !isempty(units_label)
        ylabel_use *= " ($(units_label))"
    elseif !isnothing(units_mult)
        ylabel_use *= " (x$(units_mult))"
    end

    title_use = isempty(title) ? "Finite-size scaling at beta=$(beta_target), w-index=$(w_ix)" : title

    fig = plot(
        xlabel=xlabel_use,
        ylabel=ylabel_use,
        title=title_use,
        legend=legend_pos,
        size=(900, 520),
        left_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
        xscale=xscale,
        yscale=yscale,
    )

    L_all = Float64[]
    min_pts_fit = fixed_a === nothing ? 2 : 1

    for tca in sort(collect(keys(tca_points)))
        pts = sort(tca_points[tca], by = p -> _xof(p[1], p[2]))
        xs = [_xof(p[1], p[2]) for p in pts]
        ys = [p[3] for p in pts]
        es = [p[4] for p in pts]
        betas = unique(round.(Int, [p[5] * 1000 for p in pts])) ./ 1000
        plot!(fig, xs, ys; yerr=es, m=:circle, lw=2, ms=6, label="TCA=$(tca) (beta=$(join(betas, ",")))")
        append!(L_all, [p[1] for p in pts])
    end

    if include_full_reference && !isempty(full_points)
        pts = sort(full_points, by = p -> _xof(p[1], p[2]))
        xs = [_xof(p[1], p[2]) for p in pts]
        ys = [p[3] for p in pts]
        es = [p[4] for p in pts]
        betas = unique(round.(Int, [p[5] * 1000 for p in pts])) ./ 1000
        plot!(fig, xs, ys; yerr=es, m=:diamond, lw=2, ls=:dash, ms=6, c=:black, label="FULL (beta=$(join(betas, ",")))")
        append!(L_all, [p[1] for p in pts])
    end

    # Fit overlays
    if fit_scaling
        if fit_mode === :per_series || fit_mode === :both
            for tca in sort(collect(keys(tca_points)))
                pts = tca_points[tca]
                length(pts) >= min_pts_fit || continue
                Ls = [p[1] for p in pts]
                ysv = [p[3] for p in pts]
                σs = [p[4] for p in pts]
                try
                    fr = fit_finite_size_power_law(Ls, ysv, σs; fixed_a=fixed_a)
                    _overlay_power_law_fit!(fig, fr, xaxis, xscale; label="TCA=$(tca) fit")
                catch err
                    @warn "plot_finite_size_scaling: fit failed for TCA=$tca" exception=err
                end
            end
            if include_full_reference && length(full_points) >= min_pts_fit
                Ls = [p[1] for p in full_points]
                ysv = [p[3] for p in full_points]
                σs = [p[4] for p in full_points]
                try
                    fr = fit_finite_size_power_law(Ls, ysv, σs; fixed_a=fixed_a)
                    _overlay_power_law_fit!(fig, fr, xaxis, xscale; label="FULL fit", linecolor=:black)
                catch err
                    @warn "plot_finite_size_scaling: fit failed for FULL" exception=err
                end
            end
        end

        if fit_mode === :pooled || fit_mode === :both
            pooled_pts = NTuple{5, Float64}[]
            for pts in values(tca_points)
                append!(pooled_pts, pts)
            end
            include_full_reference && append!(pooled_pts, full_points)
            if length(pooled_pts) >= min_pts_fit
                Ls = [p[1] for p in pooled_pts]
                ysv = [p[3] for p in pooled_pts]
                σs = [p[4] for p in pooled_pts]
                try
                    fr = fit_finite_size_power_law(Ls, ysv, σs; fixed_a=fixed_a)
                    _overlay_power_law_fit!(fig, fr, xaxis, xscale; label="pooled fit", linecolor=:red)
                catch err
                    @warn "plot_finite_size_scaling: pooled fit failed" exception=err
                end
            end
        end
    end

    if !isempty(L_all)
        L_min = minimum(L_all); L_max = maximum(L_all)
        if xaxis === :L
            if xscale == :log10
                plot!(fig, xlims=(L_min * 0.8, L_max * 1.2))
            else
                plot!(fig, xlims=(0.0, L_max * 2.0))
            end
        elseif xaxis === :inv_L
            if xscale == :log10
                plot!(fig, xlims=(0.5 / L_max, 1.5 / L_min))
            else
                plot!(fig, xlims=(0.0, 1.5 / L_min))
            end
        else  # :inv_N
            N_min = L_min^2; N_max = L_max^2
            if xscale == :log10
                plot!(fig, xlims=(0.5 / N_max, 1.5 / N_min))
            else
                plot!(fig, xlims=(0.0, 1.5 / N_min))
            end
        end
    end
    return fig
end

"""
Unified finite-size scaling overload using all available geometries for the chosen U.

Usage:
plot_finite_size_scaling(sim_info, wanted_n, U, obs, beta_target; ...)
"""
function plot_finite_size_scaling(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    obs,
    beta_target;
    w_ix=1,
    include_full_reference::Bool=true,
    title::String="",
    ylabel::String="",
    units_mult::Union{Nothing, Float64}=nothing,
    units_label::String="",
    prefer_full::Bool=true,
    sims_all=nothing,
    xscale=:identity,
    yscale=:identity,
    xaxis::Symbol=:inv_N,
    fit_scaling::Bool=false,
    fixed_a::Union{Nothing, Real}=nothing,
    fit_mode::Symbol=:pooled,
)
    return plot_finite_size_scaling(
        sim_info,
        wanted_n,
        U,
        nothing,
        obs,
        beta_target;
        w_ix=w_ix,
        include_full_reference=include_full_reference,
        title=title,
        ylabel=ylabel,
        units_mult=units_mult,
        units_label=units_label,
        prefer_full=prefer_full,
        sims_all=sims_all,
        xscale=xscale,
        yscale=yscale,
        xaxis=xaxis,
        fit_scaling=fit_scaling,
        fixed_a=fixed_a,
        fit_mode=fit_mode,
    )
end

"""
Unified finite-size scaling overload for one n and one U across many beta values.

Creates one subplot per beta.

Usage:
plot_finite_size_scaling(sim_info, wanted_n, U, obs, beta_list; ...)
"""
function plot_finite_size_scaling(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    obs,
    beta_list::AbstractVector;
    geometry_pairs=nothing,
    w_ix=1,
    include_full_reference::Bool=true,
    ylabel::String="",
    units_mult::Union{Nothing, Float64}=nothing,
    units_label::String="",
    prefer_full::Bool=true,
    ncols::Int=0,
    xscale=:identity,
    yscale=:identity,
    xaxis::Symbol=:inv_N,
    fit_scaling::Bool=false,
    fixed_a::Union{Nothing, Real}=nothing,
    fit_mode::Symbol=:pooled,
)
    return plot_finite_size_scaling(
        sim_info,
        [Float64(wanted_n)],
        [Float64(U)],
        obs,
        collect(beta_list);
        geometry_pairs=geometry_pairs,
        w_ix=w_ix,
        include_full_reference=include_full_reference,
        ylabel=ylabel,
        units_mult=units_mult,
        units_label=units_label,
        prefer_full=prefer_full,
        ncols=ncols,
        xscale=xscale,
        yscale=yscale,
        xaxis=xaxis,
        fit_scaling=fit_scaling,
        fixed_a=fixed_a,
        fit_mode=fit_mode,
    )
end

"""
Unified finite-size scaling overload for multiple n/U/beta values.

Creates one subplot per (n, U, beta) combination.

Usage:
plot_finite_size_scaling(sim_info, n_list, U_list, obs, beta_list; ...)
"""
function plot_finite_size_scaling(
    sim_info::Dict{String, Any},
    n_list::AbstractVector,
    U_list::AbstractVector,
    obs,
    beta_list::AbstractVector;
    geometry_pairs=nothing,
    w_ix=1,
    include_full_reference::Bool=true,
    ylabel::String="",
    units_mult::Union{Nothing, Float64}=nothing,
    units_label::String="",
    prefer_full::Bool=true,
    ncols::Int=0,
    xscale=:identity,
    yscale=:identity,
    xaxis::Symbol=:inv_N,
    fit_scaling::Bool=false,
    fixed_a::Union{Nothing, Real}=nothing,
    fit_mode::Symbol=:pooled,
)
    combos = Tuple{Float64, Float64, Float64}[]
    for n in n_list, U in U_list, beta_target in beta_list
        push!(combos, (Float64(n), Float64(U), Float64(beta_target)))
    end
    isempty(combos) && error("No (n, U, beta) combinations were provided.")

    figs = Plots.Plot[]
    for (n, U, beta_target) in combos
        panel_title = "n=$(n), U=$(U), beta=$(beta_target)"
        fig_i = plot_finite_size_scaling(
            sim_info,
            n,
            U,
            geometry_pairs,
            obs,
            beta_target;
            w_ix=w_ix,
            include_full_reference=include_full_reference,
            title=panel_title,
            ylabel=ylabel,
            units_mult=units_mult,
            units_label=units_label,
            prefer_full=prefer_full,
            xscale=xscale,
            yscale=yscale,
            xaxis=xaxis,
            fit_scaling=fit_scaling,
            fixed_a=fixed_a,
            fit_mode=fit_mode,
        )
        push!(figs, fig_i)
    end

    nplots = length(figs)
    ncols_use = ncols > 0 ? ncols : min(3, nplots)
    nrows_use = ceil(Int, nplots / ncols_use)

    return plot(
        figs...,
        layout=(nrows_use, ncols_use),
        size=(900 * ncols_use, 520 * nrows_use),
        plot_title="Finite-size scaling across n/U/beta",
    )
end

"""
Unified observable-vs-T overload.

Usage:
plot_observable_vs_T(sim_info, wanted_n, U, geometry_pairs, obs_name, obs_index=1; ...)
"""
function plot_observable_vs_T(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs,
    obs_name,
    obs_index=1;
    title::String="",
    ylabel::String="",
    xlims=nothing,
    ylims=nothing,
    prefer_full::Bool=true,
    sims_all=nothing,
)
    all_series = build_unified_series(
        sim_info,
        wanted_n,
        U,
        geometry_pairs;
        prefer_full=prefer_full,
        sims_all=sims_all,
    )

    ylabel_use = isempty(ylabel) ? String(obs_name) : ylabel
    title_use = isempty(title) ? "$(obs_name) vs T" : title

    fig = plot(
        xlabel="T",
        ylabel=ylabel_use,
        title=title_use,
        legend=:topright,
        size=(950, 550),
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm,
    )

    for (lbl, sim) in all_series
        if !_sim_has_observable(sim, obs_name)
            @warn "Observable not found for simulation; skipping" wanted_n=wanted_n U=U label=lbl obs_name=obs_name
            continue
        end

        ts, obs, errs = _get_obs_safe(sim, obs_name; obs_index=obs_index)
        isempty(ts) && continue
        plot!(fig, ts, obs; yerr=errs, label=lbl, m=:circle, lw=2, ms=5)
    end

    !isnothing(xlims) && plot!(fig, xlims=xlims)
    !isnothing(ylims) && plot!(fig, ylims=ylims)

    return fig
end

"""
Unified resistivity overlay overload.

Usage:
plot_resistivity_tensor_overlay(sim_info, wanted_n, U, geometry_pairs)
"""
function plot_resistivity_tensor_overlay(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs;
    prefer_full::Bool=true,
    sims_all=nothing,
)
    all_series = build_unified_series(
        sim_info,
        wanted_n,
        U,
        geometry_pairs;
        prefer_full=prefer_full,
        sims_all=sims_all,
    )

    fig = plot(
        xlabel="T",
        ylabel="rho_xx / 2pi",
        title="Resistivity tensor, U=$(U), n~$(wanted_n)",
        legend=:topright,
        size=(950, 550),
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm,
    )

    for (lbl, sim) in all_series
        if !_sim_has_observable(sim, "resistivity_tensor_fixed_extrap")
            @warn "Observable not found for simulation; skipping" wanted_n=wanted_n U=U label=lbl obs_name="resistivity_tensor_fixed_extrap"
            continue
        end

        Ts, vals, errs = _get_obs_safe(sim, "resistivity_tensor_fixed_extrap"; obs_index=nothing)
        isempty(Ts) && continue

        rho = hcat(vals...) ./ (2 * pi)
        drho = hcat(errs...) ./ (2 * pi)
        plot!(fig, Ts, rho[1, :]; yerr=drho[1, :], label="$(lbl) rho_xx", m=:circle, lw=2, ms=5)
    end

    plot!(fig, xlims=(0.0, 1.0), ylims=(0.2, 1.0))
    return fig
end

"""
Unified resistivity subplot overload.

Usage:
plot_resistivity_tensor_subplots(sim_info, wanted_n, U, geometry_pairs)
"""
function plot_resistivity_tensor_subplots(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs;
    prefer_full::Bool=true,
    sims_all=nothing,
)
    all_series = build_unified_series(
        sim_info,
        wanted_n,
        U,
        geometry_pairs;
        prefer_full=prefer_full,
        sims_all=sims_all,
    )

    n_series = length(all_series)
    ncols = min(n_series, 3)
    nrows = ceil(Int, n_series / ncols)

    fig = plot(
        layout=(nrows, ncols),
        size=(450 * ncols, 400 * nrows),
        left_margin=15Plots.mm,
        bottom_margin=8Plots.mm,
    )

    for (i, (lbl, sim)) in enumerate(all_series)
        if !_sim_has_observable(sim, "resistivity_tensor_fixed_extrap")
            @warn "Observable not found for simulation; skipping" wanted_n=wanted_n U=U label=lbl obs_name="resistivity_tensor_fixed_extrap"
            continue
        end

        obs = _get_obs_safe(sim, "resistivity_tensor_fixed_extrap"; obs_index=nothing)
        isempty(obs[1]) && continue

        Ts = obs[1]
        res_vals = [hcat(obs[2]...)[k, :] for k in 1:4]
        res_errs = [hcat(obs[3]...)[k, :] for k in 1:4]

        plot!(fig, Ts, res_vals[1] ./ (2 * pi); yerr=res_errs[1] ./ (2 * pi),
            label="rho_xx", m=:circle, lw=2, ms=5, subplot=i)
        plot!(fig, Ts, res_vals[2] ./ (2 * pi); yerr=res_errs[2] ./ (2 * pi),
            label="rho_xy", m=:square, lw=2, ms=5, subplot=i)

        title!(fig, lbl, subplot=i)
        xlabel!(fig, "T", subplot=i)
        ylabel!(fig, "rho / 2pi", subplot=i)
    end

    plot!(fig, xlims=(0.0, 2.0), ylims=(-0.5, 5.0))
    plot!(fig, plot_title="Resistivity tensor, U=$(U), n~$(wanted_n)")
    return fig
end

"""
Unified DOS heatmap overload.

Usage:
plot_dos_heatmap(sim_info, wanted_n, U, geometry_pairs; T_max=1.0)
"""
function plot_dos_heatmap(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs;
    T_max::Float64=1.0,
    prefer_full::Bool=true,
    sims_all=nothing,
)
    all_series = build_unified_series(
        sim_info,
        wanted_n,
        U,
        geometry_pairs;
        prefer_full=prefer_full,
        sims_all=sims_all,
    )

    n_series = length(all_series)
    ncols = min(n_series, 3)
    nrows = ceil(Int, n_series / ncols)

    fig = plot(
        layout=(nrows, ncols),
        size=(500 * ncols, 400 * nrows),
        left_margin=15Plots.mm,
        bottom_margin=10Plots.mm,
    )

    for (sp_ix, (lbl, sim)) in enumerate(all_series)
        if !_sim_has_observable(sim, "DOS")
            @warn "Observable not found for simulation; skipping" wanted_n=wanted_n U=U label=lbl obs_name="DOS"
            continue
        end

        betas = sort(collect(keys(sim)))
        isempty(betas) && continue

        obs = _get_obs_safe(sim, "DOS"; obs_index=nothing)
        Ts = reverse(obs[1])
        isempty(Ts) && continue

        w = sim[betas[1]]["sim_dict"]["w_list"]
        n_w = length(w)
        n_T = length(Ts)

        dos_mat = fill(NaN, n_w, n_T)
        for (j, _) in enumerate(Ts)
            if Ts[n_T - j + 1] <= T_max && !isempty(obs[2][j])
                dos_mat[:, n_T - j + 1] = obs[2][j]
            end
        end

        heatmap!(fig, Ts, w, dos_mat;
            xlabel="T", ylabel="w", title=lbl,
            colorbar_title="DOS", subplot=sp_ix, xlims=(0.0, T_max))
    end

    plot!(fig, plot_title="DOS(T, w), U=$(U), n~$(wanted_n)")
    return fig
end

"""
Plot DOS(ω) curves at selected temperatures.

One subplot per U in U_list. Within each subplot, one curve per
(geometry, beta) combination. The x-axis is ω (frequency) and the
y-axis is the density of states.

Usage:
plot_dos_vs_omega(sim_info, wanted_n, U_list, geometry_pairs, beta_list; ...)

Arguments:
- sim_info: merged simulation dictionary
- wanted_n: target density
- U_list: vector of U values, one subplot per value
- geometry_pairs: vector of (L, TCA_size) pairs
- beta_list: vector of beta values; one curve per beta per geometry

Keyword arguments:
- prefer_full: when L==TCA_size, prefer the FULL simulation
- sims_all: optional pre-fetched sims_all for wanted_n
- exact_beta: if true, skip betas that do not match within tolerance
- beta_match_tol: tolerance for matching beta values
- show_missing: emit warnings for skipped curves
- omega_lim: optional (xmin, xmax) to clip the ω axis
- ncols: number of columns in the subplot grid (0 = auto)
- show_tc_tstar: if true, add a phantom legend entry per geometry with Tc and T*
- tc_lowest_q: if true, use stiffness_lowest_q (passed through to find_tc)
- tstar_len: window length for T* fit (passed through to find_Tstar)
- show_error_ribbon: if true, draw a shaded ribbon of ±error around each DOS curve
- ribbon_alpha: opacity of the error ribbon (0..1)
"""
function plot_dos_vs_omega(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U_list::AbstractVector,
    geometry_pairs::AbstractVector,
    beta_list::AbstractVector;
    prefer_full::Bool=true,
    sims_all=nothing,
    exact_beta::Bool=false,
    beta_match_tol::Float64=1e-8,
    show_missing::Bool=false,
    omega_lim=nothing,
    ncols::Int=0,
    show_tc_tstar::Bool=true,
    tc_lowest_q::Bool=true,
    tstar_len::Int=3,
    show_error_ribbon::Bool=false,
    ribbon_alpha::Float64=0.25,
)
    isempty(U_list) && error("U_list must contain at least one value.")
    isempty(geometry_pairs) && error("geometry_pairs must contain at least one (L, TCA_size) pair.")
    isempty(beta_list) && error("beta_list must contain at least one beta value.")

    sims_all_n = sims_all === nothing ? get_sims_all(sim_info; wanted_n=Float64(wanted_n)) : sims_all
    beta_targets = Float64.(collect(beta_list))
    panels = Plots.Plot[]

    for U_raw in U_list
        U = Float64(U_raw)
        subplot_fig = plot(
            xlabel="ω",
            ylabel="DOS",
            title="DOS(ω), n~$(wanted_n), U=$(U)",
            legend=:topright,
            size=(920, 560),
            left_margin=8Plots.mm,
            bottom_margin=8Plots.mm,
        )

        plotted_curves = 0

        for geom in geometry_pairs
            L = Int(geom[1])
            TCA_size = Int(geom[2])

            sim, method = get_sim_from_sims_all(sims_all_n, U, L, TCA_size; prefer_full=prefer_full)
            if sim === nothing || isempty(sim)
                show_missing && @warn "Simulation missing; skipping" U=U L=L TCA_size=TCA_size
                continue
            end

            if !_sim_has_observable(sim, "DOS")
                show_missing && @warn "DOS observable missing; skipping" U=U L=L TCA_size=TCA_size
                continue
            end

            betas_avail = sort(collect(keys(sim)))
            isempty(betas_avail) && continue

            obs_Ts_raw, obs_vals, obs_errs = _get_obs_safe(sim, "DOS"; obs_index=nothing)
            isempty(obs_Ts_raw) && continue
            obs_Ts = Float64.(obs_Ts_raw)

            w_list = nothing
            for b in betas_avail
                sd = sim[b]["sim_dict"]
                if haskey(sd, "w_list")
                    w_list = Float64.(sd["w_list"])
                    break
                end
            end
            if w_list === nothing
                show_missing && @warn "w_list missing for sim; skipping" U=U L=L TCA_size=TCA_size
                continue
            end

            lbl_method = method == "FULL" ? "FULL" : "TCA=$(TCA_size)"

            for beta_target in beta_targets
                beta_use = _find_closest_beta_key(sim, beta_target)
                beta_use === nothing && continue

                if abs(Float64(beta_use) - beta_target) > beta_match_tol
                    if exact_beta
                        show_missing && @warn "Target beta not found exactly; skipping" target_beta=beta_target U=U L=L TCA_size=TCA_size nearest_beta=Float64(beta_use)
                        continue
                    else
                        show_missing && @warn "Target beta not found exactly; using nearest available beta" target_beta=beta_target U=U L=L TCA_size=TCA_size nearest_beta=Float64(beta_use)
                    end
                end

                T_use = 1.0 / Float64(beta_use)
                ix = argmin(abs.(obs_Ts .- T_use))
                dos_vec = obs_vals[ix]
                if dos_vec === nothing || isempty(dos_vec)
                    show_missing && @warn "Empty DOS vector; skipping" target_beta=beta_target U=U L=L TCA_size=TCA_size
                    continue
                end

                ys = Float64.(dos_vec)
                npts = min(length(w_list), length(ys))
                npts == 0 && continue
                xs = w_list[1:npts]
                ys = ys[1:npts]

                ribbon_vec = nothing
                if show_error_ribbon && ix <= length(obs_errs)
                    err_raw = obs_errs[ix]
                    if err_raw !== nothing && !isempty(err_raw)
                        err_vec = Float64.(err_raw)
                        nerr = min(npts, length(err_vec))
                        if nerr == npts
                            ribbon_vec = err_vec[1:npts]
                            for i in eachindex(ribbon_vec)
                                if !isfinite(ribbon_vec[i])
                                    ribbon_vec[i] = 0.0
                                end
                            end
                        else
                            show_missing && @warn "DOS error length mismatch; skipping ribbon" target_beta=beta_target U=U L=L TCA_size=TCA_size n_w=npts n_err=length(err_vec)
                        end
                    end
                end

                lbl = length(geometry_pairs) == 1 ?
                    "T=$(round(T_use, sigdigits=4))" :
                    "L=$(L), $(lbl_method), T=$(round(T_use, sigdigits=4))"

                if ribbon_vec === nothing
                    plot!(subplot_fig, xs, ys; lw=1.6, label=lbl)
                else
                    plot!(subplot_fig, xs, ys;
                        lw=1.6, label=lbl,
                        ribbon=ribbon_vec, fillalpha=ribbon_alpha)
                end
                plotted_curves += 1
            end

            if show_tc_tstar
                tc_val = NaN
                try
                    tc, crosses, _ = find_tc(sim; lowest_q=tc_lowest_q)
                    if crosses && tc !== nothing && isfinite(tc)
                        tc_val = Float64(tc)
                    end
                catch err
                    show_missing && @warn "find_tc failed; skipping Tc annotation" U=U L=L TCA_size=TCA_size exception=err
                end

                tstar_val = NaN
                try
                    tstar, _, worked = find_Tstar(sim; len=tstar_len)
                    if worked && tstar !== nothing && isfinite(tstar)
                        tstar_val = Float64(tstar)
                    end
                catch err
                    show_missing && @warn "find_Tstar failed; skipping T* annotation" U=U L=L TCA_size=TCA_size exception=err
                end

                if isfinite(tc_val) || isfinite(tstar_val)
                    parts = String[]
                    isfinite(tc_val) && push!(parts, "Tc=$(round(tc_val, sigdigits=4))")
                    isfinite(tstar_val) && push!(parts, "T*=$(round(tstar_val, sigdigits=4))")
                    geom_prefix = length(geometry_pairs) == 1 ? "" : "L=$(L), $(lbl_method): "
                    plot!(subplot_fig, Float64[NaN], Float64[NaN];
                        label=geom_prefix * join(parts, ", "),
                        lw=0, markersize=0, markerstrokewidth=0)
                end
            end
        end

        if plotted_curves == 0
            show_missing && @warn "No curves plotted for U" U=U
        end

        if omega_lim !== nothing
            xlims!(subplot_fig, omega_lim)
        end

        push!(panels, subplot_fig)
    end

    nplots = length(panels)
    nplots == 0 && error("No subplots were generated.")
    ncols_use = ncols > 0 ? ncols : min(3, nplots)
    nrows_use = ceil(Int, nplots / ncols_use)

    return plot(
        panels...,
        layout=(nrows_use, ncols_use),
        size=(980 * ncols_use, 560 * nrows_use),
        plot_title="DOS(ω), n~$(wanted_n)",
    )
end

"""
Plot current-current correlation limits for selected systems and betas.

For each subplot, this function draws:
- Lxx_qy(qy)
- Lxx_qx(qx)
- -Kx at q=0

Usage:
plot_lxx_limits_with_kx(sim_info, wanted_n, U, beta_list; geometry_pairs=nothing)

Arguments:
- sim_info: merged simulation dictionary
- wanted_n: target density
- U: interaction value
- beta_list: vector of beta values for columns

Keyword arguments:
- geometry_pairs: vector of (L, TCA_size). If nothing, uses all available for U.
- prefer_full: when L==TCA_size, prefer FULL simulation
- ncols: number of columns; defaults to length(beta_list)
- xlims, ylims: optional axis limits for every subplot
"""
function plot_lxx_limits_with_kx(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    beta_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    ncols::Int=0,
    xlims=nothing,
    ylims=nothing,
)
    sims_all_local = get_sims_all(sim_info; wanted_n=Float64(wanted_n))
    geoms = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : collect(geometry_pairs)

    isempty(geoms) && error("No geometry pairs found for U=$(U), n~$(wanted_n).")
    isempty(beta_list) && error("beta_list must contain at least one beta value.")

    nrows = length(geoms)
    ncols_use = ncols > 0 ? ncols : length(beta_list)

    if ncols_use != length(beta_list)
        @warn "ncols is ignored for this plot shape; using one column per beta" requested_ncols=ncols n_beta=length(beta_list)
    end

    fig = plot(
        layout=(nrows, length(beta_list)),
        size=(520 * length(beta_list), 420 * nrows),
        left_margin=12Plots.mm,
        bottom_margin=10Plots.mm,
    )

    for (row_ix, (L, TCA_size)) in enumerate(geoms)
        sim, method = get_sim_from_sims_all(sims_all_local, U, L, TCA_size; prefer_full=prefer_full)

        if sim === nothing
            @warn "Simulation not found; skipping row" wanted_n=wanted_n U=U L=L TCA_size=TCA_size
            continue
        end

        sys_label = method == "FULL" ? "L=$(L), FULL" : "L=$(L), TCA=$(TCA_size)"

        for (col_ix, beta_target_raw) in enumerate(beta_list)
            subplot_ix = (row_ix - 1) * length(beta_list) + col_ix
            beta_target = Float64(beta_target_raw)

            if !haskey(sim, beta_target)
                @warn "Beta not found for simulation; skipping subplot" wanted_n=wanted_n U=U L=L TCA_size=TCA_size beta=beta_target
                continue
            end

            sim_beta = sim[beta_target]
            sim_dict = sim_beta["sim_dict"]
            means = sim_dict["means"]

            if !haskey(means, "Lxx_qy") || !haskey(means, "Lxx_qx") || !haskey(means, "Kx")
                @warn "Required observables missing; skipping subplot" wanted_n=wanted_n U=U L=L TCA_size=TCA_size beta=beta_target required=("Lxx_qy", "Lxx_qx", "Kx")
                continue
            end

            lxx_qy = means["Lxx_qy"]["value"]
            lxx_qy_err = means["Lxx_qy"]["error"]
            lxx_qx = means["Lxx_qx"]["value"]
            lxx_qx_err = means["Lxx_qx"]["error"]
            kx = means["Kx"]["value"]
            kx_err = means["Kx"]["error"]

            if !(lxx_qy isa AbstractVector) || !(lxx_qx isa AbstractVector)
                @warn "Lxx observables are not vectors; skipping subplot" wanted_n=wanted_n U=U L=L TCA_size=TCA_size beta=beta_target
                continue
            end

            lx = sim_dict["Lx"]
            ly = sim_dict["Ly"]
            qy_vals = collect(1:length(lxx_qy)) .* (2 * pi / ly)
            qx_vals = collect(1:length(lxx_qx)) .* (2 * pi / lx)

            plot!(fig, qy_vals, lxx_qy;
                yerr=lxx_qy_err,
                label="Lxx_qy",
                m=:circle,
                lw=2,
                ms=5,
                subplot=subplot_ix)

            plot!(fig, qx_vals, lxx_qx;
                yerr=lxx_qx_err,
                label="Lxx_qx",
                m=:square,
                lw=2,
                ms=5,
                ls=:dash,
                subplot=subplot_ix)

            scatter!(fig, [0.0], [-kx];
                yerr=[kx_err],
                label="-Kx",
                m=:utriangle,
                ms=7,
                lw=2,
                subplot=subplot_ix)

            title!(fig, "$(sys_label), beta=$(beta_target)", subplot=subplot_ix)
            xlabel!(fig, "q", subplot=subplot_ix)
            ylabel!(fig, "Lxx", subplot=subplot_ix)

            xlims !== nothing && plot!(fig, xlims=xlims, subplot=subplot_ix)
            ylims !== nothing && plot!(fig, ylims=ylims, subplot=subplot_ix)
        end
    end

    plot!(fig, plot_title="Lxx(q) limits with -Kx at q=0, U=$(U), n~$(wanted_n)")
    return fig
end

"""
Plot current-current correlation limits for multiple n and U values.

Creates one panel-block per (n, U), each block containing the same layout as
`plot_lxx_limits_with_kx(sim_info, wanted_n, U, beta_list; ...)`.

Usage:
plot_lxx_limits_with_kx(sim_info, n_list, U_list, beta_list; ...)
"""
function plot_lxx_limits_with_kx(
    sim_info::Dict{String, Any},
    n_list::AbstractVector,
    U_list::AbstractVector,
    beta_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    xlims=nothing,
    ylims=nothing,
    block_ncols::Int=0,
)
    combos = Tuple{Float64, Float64}[]
    for n in n_list, U in U_list
        push!(combos, (Float64(n), Float64(U)))
    end
    isempty(combos) && error("No (n, U) combinations were provided.")

    figs = Plots.Plot[]
    for (n, U) in combos
        fig_i = plot_lxx_limits_with_kx(
            sim_info,
            n,
            U,
            beta_list;
            geometry_pairs=geometry_pairs,
            prefer_full=prefer_full,
            xlims=xlims,
            ylims=ylims,
        )
        push!(figs, fig_i)
    end

    n_blocks = length(figs)
    ncols_use = block_ncols > 0 ? block_ncols : min(2, n_blocks)
    nrows_use = ceil(Int, n_blocks / ncols_use)

    return plot(
        figs...,
        layout=(nrows_use, ncols_use),
        size=(1200 * ncols_use, 900 * nrows_use),
        plot_title="Lxx(q) limits with -Kx across n/U combinations",
    )
end

"""
    _mean_std(vals) -> (mean, std)

Mean and sample standard deviation (0 for one value, NaNs for none).
"""
function _mean_std(vals::Vector{Float64})
    n = length(vals)
    n == 0 && return NaN, NaN
    mu = sum(vals) / n
    if n == 1
        return mu, 0.0
    end
    var = sum((v - mu)^2 for v in vals) / (n - 1)
    return mu, sqrt(var)
end

"""
    _canonical_geometry_pairs_from_l_list(L_list, geometry_pairs) -> Vector{Tuple{Int,Int}}

Sorted unique (L, cluster size) pairs: `geometry_pairs` if given, else (L, L) (full K0)
for each L in `L_list`.
"""
function _canonical_geometry_pairs_from_l_list(
    L_list::AbstractVector,
    geometry_pairs,
)
    geoms = Tuple{Int, Int}[]

    if geometry_pairs === nothing
        for L_raw in L_list
            L = Int(round(Float64(L_raw)))
            push!(geoms, (L, L))
        end
    else
        for g in geometry_pairs
            length(g) >= 2 || error("Each geometry entry must contain at least (L, TCA_size).")
            L = Int(round(Float64(g[1])))
            TCA_size = Int(round(Float64(g[2])))
            push!(geoms, (L, TCA_size))
        end
    end

    unique!(geoms)
    sort!(geoms, by = x -> (x[1], x[2]))
    return geoms
end

"""
    _collect_ps_series_for_collapse(sim_info, wanted_n, U, geometry_pairs; prefer_full=true,
                                    sims_all=nothing, warn_on_missing=true) -> Vector{NamedTuple}

The Ps(T) curve (T, Ps, dPs) of each (L, cluster size), the input of the BKT collapse.
Missing geometries are skipped (with a warning).
"""
function _collect_ps_series_for_collapse(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs::Vector{Tuple{Int, Int}};
    prefer_full::Bool=true,
    sims_all=nothing,
    warn_on_missing::Bool=true,
)
    sims_all_local = sims_all === nothing ? get_sims_all(sim_info; wanted_n=Float64(wanted_n)) : sims_all
    rows = NamedTuple[]

    for (L, TCA_size) in geometry_pairs
        sim, method = get_sim_from_sims_all(sims_all_local, U, L, TCA_size; prefer_full=prefer_full)
        if sim === nothing || isempty(sim)
            warn_on_missing && @warn "Simulation missing for geometry; skipping" wanted_n=wanted_n U=U L=L TCA_size=TCA_size
            continue
        end

        if !_sim_has_observable(sim, "Ps")
            warn_on_missing && @warn "Observable Ps not found for geometry; skipping" wanted_n=wanted_n U=U L=L TCA_size=TCA_size
            continue
        end

        Ts_raw, Ps_raw, dPs_raw = _get_obs_safe(sim, "Ps"; obs_index=nothing)
        npts = min(length(Ts_raw), length(Ps_raw), length(dPs_raw))
        if npts == 0
            warn_on_missing && @warn "No Ps points available for geometry; skipping" wanted_n=wanted_n U=U L=L TCA_size=TCA_size
            continue
        end

        Ts = Float64[]
        Ps = Float64[]
        dPs = Float64[]
        for i in 1:npts
            T = Float64(Ts_raw[i])
            p = Float64(Ps_raw[i])
            dp = abs(Float64(dPs_raw[i]))
            if isfinite(T) && isfinite(p) && isfinite(dp) && dp > 0.0
                push!(Ts, T)
                push!(Ps, p)
                push!(dPs, dp)
            end
        end

        if isempty(Ts)
            warn_on_missing && @warn "No finite Ps points with positive errors for geometry; skipping" wanted_n=wanted_n U=U L=L TCA_size=TCA_size
            continue
        end

        lbl = method == "FULL" ? "L=$(L), FULL" : "L=$(L), TCA=$(TCA_size)"
        push!(rows, (
            label=lbl,
            L=L,
            TCA_size=TCA_size,
            method=method,
            T=Ts,
            Ps=Ps,
            dPs=dPs,
        ))
    end

    return rows
end

"""
    _build_ps_collapse_points(ps_rows, Tc, A; min_points_per_L=1, max_deltaT=nothing)
        -> (x, y, sigma, counts_by_L, points_by_L)

Points of the BKT finite-size collapse for trial (Tc, A): for each T > Tc (and
T - Tc ≤ `max_deltaT`), x = L·exp(-A/√(T - Tc)) (L over the BKT correlation length) and
y = Ps·L^(-7/4) (η = 1/4), with error dPs·L^(-7/4). Returns empty points if some L has
fewer than `min_points_per_L`.
"""
function _build_ps_collapse_points(
    ps_rows::Vector,
    Tc::Float64,
    A::Float64;
    min_points_per_L::Int=1,
    max_deltaT::Union{Nothing, Float64}=nothing,
)
    xs = Float64[]
    ys = Float64[]
    sigmas = Float64[]
    counts_by_L = Dict{Int, Int}()
    x_by_L = Dict{Int, Vector{Float64}}()
    y_by_L = Dict{Int, Vector{Float64}}()
    sigma_by_L = Dict{Int, Vector{Float64}}()
    T_by_L = Dict{Int, Vector{Float64}}()

    for row in ps_rows
        L = Int(row.L)
        L_float = Float64(L)
        ps_scale = L_float^(-7.0 / 4.0)
        local_x = get!(x_by_L, L, Float64[])
        local_y = get!(y_by_L, L, Float64[])
        local_sigma = get!(sigma_by_L, L, Float64[])
        local_T = get!(T_by_L, L, Float64[])
        c = get(counts_by_L, L, 0)

        npts = min(length(row.T), length(row.Ps), length(row.dPs))
        for i in 1:npts
            T = Float64(row.T[i])
            T <= Tc && continue

            deltaT = T - Tc
            deltaT <= 0.0 && continue
            (max_deltaT !== nothing && deltaT > max_deltaT) && continue

            x = L_float * exp(-A / sqrt(deltaT))
            y = Float64(row.Ps[i]) * ps_scale
            s = Float64(row.dPs[i]) * ps_scale

            if isfinite(x) && isfinite(y) && isfinite(s) && s > 0.0
                push!(xs, x)
                push!(ys, y)
                push!(sigmas, s)
                push!(local_x, x)
                push!(local_y, y)
                push!(local_sigma, s)
                push!(local_T, T)
                c += 1
            end
        end

        counts_by_L[L] = c
    end

    if min_points_per_L > 0
        for (_, c) in counts_by_L
            c < min_points_per_L && return (
                x=Float64[],
                y=Float64[],
                sigma=Float64[],
                counts_by_L=counts_by_L,
                points_by_L=Dict{Int, NamedTuple{(:x, :y, :sigma, :T), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}(),
            )
        end
    end

    points_by_L = Dict{Int, NamedTuple{(:x, :y, :sigma, :T), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for L in keys(counts_by_L)
        points_by_L[L] = (
            x=get(x_by_L, L, Float64[]),
            y=get(y_by_L, L, Float64[]),
            sigma=get(sigma_by_L, L, Float64[]),
            T=get(T_by_L, L, Float64[]),
        )
    end

    return (x=xs, y=ys, sigma=sigmas, counts_by_L=counts_by_L, points_by_L=points_by_L)
end

"""
    _weighted_poly_reduced_chi2(x, y, sigma, poly_order) -> (chi2_red, coeffs, yfit, dof)

Weighted least-squares polynomial fit of degree `poly_order` and its reduced χ²: the
quality of a collapse (all points on one curve). `chi2_red = Inf` if there are too few points.
"""
function _weighted_poly_reduced_chi2(
    x::Vector{Float64},
    y::Vector{Float64},
    sigma::Vector{Float64},
    poly_order::Int,
)
    n = length(x)
    length(y) == n || return (chi2_red=Inf, coeffs=Float64[], yfit=Float64[], dof=0)
    length(sigma) == n || return (chi2_red=Inf, coeffs=Float64[], yfit=Float64[], dof=0)

    ncoef = poly_order + 1
    dof = n - ncoef
    dof > 0 || return (chi2_red=Inf, coeffs=Float64[], yfit=Float64[], dof=dof)

    X = Matrix{Float64}(undef, n, ncoef)
    for i in 1:n
        xp = 1.0
        for j in 1:ncoef
            X[i, j] = xp
            xp *= x[i]
        end
    end

    w = Vector{Float64}(undef, n)
    for i in 1:n
        s = sigma[i]
        if !isfinite(s) || s <= 0.0
            return (chi2_red=Inf, coeffs=Float64[], yfit=Float64[], dof=dof)
        end
        w[i] = 1.0 / s
    end

    WX = X .* w
    Wy = y .* w

    coeffs = try
        WX \ Wy
    catch
        return (chi2_red=Inf, coeffs=Float64[], yfit=Float64[], dof=dof)
    end

    yfit = X * coeffs
    resid = (y .- yfit) ./ sigma
    any(r -> !isfinite(r), resid) && return (chi2_red=Inf, coeffs=Float64[], yfit=Float64[], dof=dof)

    chisq = sum(abs2, resid)
    chi2_red = chisq / dof
    return (chi2_red=chi2_red, coeffs=coeffs, yfit=yfit, dof=dof)
end

"""
    _interval_halving_minimize(obj, lo, hi; initial_points=21, refine_iters=6, min_span=1e-4)
        -> (best_x, best_y, xs, ys)

Minimize a 1D function on [lo, hi]: evaluate it on an even grid, then repeatedly zoom
into the neighbourhood of the best point (values are cached). Returns the best point
and every evaluated (x, y).
"""
function _interval_halving_minimize(
    obj::Function,
    lo::Float64,
    hi::Float64;
    initial_points::Int=21,
    refine_iters::Int=6,
    min_span::Float64=1e-4,
)
    (isfinite(lo) && isfinite(hi) && hi > lo) || error("Invalid optimization range: lo=$(lo), hi=$(hi)")

    n0 = max(initial_points, 3)
    cache = Dict{Float64, Float64}()

    eval_at! = function (x::Float64)
        if !haskey(cache, x)
            cache[x] = Float64(obj(x))
        end
        return cache[x]
    end

    for x in range(lo, hi; length=n0)
        eval_at!(Float64(x))
    end

    for _ in 1:refine_iters
        (hi - lo) <= min_span && break

        mid = (lo + hi) / 2.0
        left = (lo + mid) / 2.0
        right = (mid + hi) / 2.0

        y_left = eval_at!(left)
        y_mid = eval_at!(mid)
        y_right = eval_at!(right)

        if y_left <= y_mid && y_left <= y_right
            hi = mid
        elseif y_right <= y_mid && y_right <= y_left
            lo = mid
        else
            lo = left
            hi = right
        end
    end

    xs = sort(collect(keys(cache)))
    ys = [cache[x] for x in xs]
    ibest = argmin(ys)

    return (best_x=xs[ibest], best_y=ys[ibest], xs=xs, ys=ys)
end

"""
    _optimize_A_for_fixed_tc(ps_rows, Tc, poly_order, A_range; initial_A_points=21,
                             A_refine_iters=7, min_span_A=1e-4, min_points_total=10,
                             min_points_per_L=1, max_deltaT=nothing)

For a fixed Tc, the BKT constant A in `A_range` that gives the best collapse (lowest
reduced χ² of a degree-`poly_order` fit, `_interval_halving_minimize`).
"""
function _optimize_A_for_fixed_tc(
    ps_rows::Vector,
    Tc::Float64,
    poly_order::Int,
    A_range::Tuple{Float64, Float64};
    initial_A_points::Int=21,
    A_refine_iters::Int=7,
    min_span_A::Float64=1e-4,
    min_points_total::Int=10,
    min_points_per_L::Int=1,
    max_deltaT::Union{Nothing, Float64}=nothing,
)
    A_lo, A_hi = A_range

    obj_A = function (A::Float64)
        pts = _build_ps_collapse_points(ps_rows, Tc, A; min_points_per_L=min_points_per_L, max_deltaT=max_deltaT)
        length(pts.x) >= min_points_total || return Inf

        fit = _weighted_poly_reduced_chi2(pts.x, pts.y, pts.sigma, poly_order)
        return fit.chi2_red
    end

    return _interval_halving_minimize(
        obj_A,
        A_lo,
        A_hi;
        initial_points=initial_A_points,
        refine_iters=A_refine_iters,
        min_span=min_span_A,
    )
end

"""
Optimize A for a fixed Tc using Ps collapse data extracted from sim_info.

This is a public wrapper around _optimize_A_for_fixed_tc that accepts unified
simulation inputs and returns an object shaped similarly to
fit_tc_from_ps_collapse for compatibility with the same plotting helpers.

Usage:
optimize_A_for_fixed_tc_from_ps_collapse(sim_info, wanted_n, U, Tc, L_list; ...)

By default, each L in L_list maps to geometry pair (L, L). You can override with
explicit geometry_pairs=[(L, TCA_size), ...].
"""
function optimize_A_for_fixed_tc_from_ps_collapse(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    Tc::Real,
    L_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    sims_all=nothing,
    poly_order::Int=2,
    A_range::Tuple{Float64, Float64}=(0.0, 1.0),
    initial_A_points::Int=21,
    A_refine_iters::Int=7,
    min_span_A::Float64=1e-4,
    min_points_total::Int=10,
    min_points_per_L::Int=1,
    max_deltaT::Union{Nothing, Float64}=nothing,
    warn_on_missing::Bool=true,
)
    Tc_f = Float64(Tc)
    isfinite(Tc_f) || error("Tc must be finite.")

    poly_order >= 0 || error("poly_order must be non-negative.")
    min_points_total > poly_order + 1 || error("min_points_total must exceed polynomial parameter count.")
    A_lo, A_hi = A_range
    (isfinite(A_lo) && isfinite(A_hi) && A_hi > A_lo) || error("Invalid A_range=$(A_range)")
    if max_deltaT !== nothing
        (isfinite(max_deltaT) && max_deltaT > 0.0) || error("max_deltaT must be a positive finite number or nothing.")
    end

    geoms = _canonical_geometry_pairs_from_l_list(L_list, geometry_pairs)
    isempty(geoms) && error("No geometries were provided.")

    ps_rows = _collect_ps_series_for_collapse(
        sim_info,
        wanted_n,
        U,
        geoms;
        prefer_full=prefer_full,
        sims_all=sims_all,
        warn_on_missing=warn_on_missing,
    )
    isempty(ps_rows) && error("No valid Ps series were found for requested geometries.")

    A_opt = _optimize_A_for_fixed_tc(
        ps_rows,
        Tc_f,
        poly_order,
        A_range;
        initial_A_points=initial_A_points,
        A_refine_iters=A_refine_iters,
        min_span_A=min_span_A,
        min_points_total=min_points_total,
        min_points_per_L=min_points_per_L,
        max_deltaT=max_deltaT,
    )

    best_A = A_opt.best_x
    best_chi2_red = A_opt.best_y

    best_pts = _build_ps_collapse_points(
        ps_rows,
        Tc_f,
        best_A;
        min_points_per_L=min_points_per_L,
        max_deltaT=max_deltaT,
    )
    best_fit = _weighted_poly_reduced_chi2(best_pts.x, best_pts.y, best_pts.sigma, poly_order)

    return (
        best_Tc=Tc_f,
        best_A=best_A,
        best_reduced_chi2=best_chi2_red,
        chi2_vs_Tc=(Tc=[Tc_f], chi2=[best_chi2_red], A_best=[best_A]),
        chi2_vs_A_at_best_Tc=(A=A_opt.xs, chi2=A_opt.ys),
        # Backward-compatible aliases for prior wrapper users.
        Tc=Tc_f,
        chi2_vs_A=(A=A_opt.xs, chi2=A_opt.ys),
        best_fit=(coeffs=best_fit.coeffs, dof=best_fit.dof, n_points=length(best_pts.x)),
        best_collapse_points=(
            x=best_pts.x,
            y=best_pts.y,
            sigma=best_pts.sigma,
            counts_by_L=best_pts.counts_by_L,
            points_by_L=best_pts.points_by_L,
        ),
        metadata=(
            n=Float64(wanted_n),
            U=Float64(U),
            geometry_pairs=geoms,
            labels=[row.label for row in ps_rows],
            poly_order=poly_order,
            tc_range=(Tc_f, Tc_f),
            A_range=A_range,
            min_points_total=min_points_total,
            min_points_per_L=min_points_per_L,
            max_deltaT=max_deltaT,
        ),
    )
end

"""
Geometry-pairs overload:
optimize_A_for_fixed_tc_from_ps_collapse(sim_info, wanted_n, U, Tc, geometry_pairs; ...)
"""
function optimize_A_for_fixed_tc_from_ps_collapse(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    Tc::Real,
    geometry_pairs::AbstractVector{<:Tuple};
    kwargs...
)
    l_list = [Int(round(Float64(g[1]))) for g in geometry_pairs]
    return optimize_A_for_fixed_tc_from_ps_collapse(
        sim_info,
        wanted_n,
        U,
        Tc,
        l_list;
        geometry_pairs=geometry_pairs,
        kwargs...,
    )
end

"""
Fit Tc from pair susceptibility Ps using pooled finite-size collapse across all selected L values.

For each candidate (Tc, A), this function:
1. Normalizes Ps by L^(-7/4)
2. Builds x = L*exp(-A/sqrt(T-Tc)) using only T > Tc
    (and optionally only T-Tc <= max_deltaT when max_deltaT is provided)
3. Fits one weighted polynomial to all pooled points from all L values together
4. Minimizes reduced chi^2 over A (for fixed Tc)
5. Minimizes reduced chi^2 over Tc

Primary input style:
fit_tc_from_ps_collapse(sim_info, wanted_n, U, L_list; ...)

By default, each L in L_list maps to geometry pair (L, L). You can override with
explicit geometry_pairs=[(L, TCA_size), ...].

Keyword option:
- max_deltaT: if set, only points with 0 < (T - Tc) <= max_deltaT are used in
    both the A/Tc optimization and the returned best-collapse data.

Returns include:
- best_collapse_points.points_by_L::Dict{Int, NamedTuple} so x/y/sigma/T can be
    plotted per-L for the best (Tc, A) fit.
"""
function fit_tc_from_ps_collapse(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    L_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    sims_all=nothing,
    poly_order::Int=2,
    tc_range::Tuple{Float64, Float64}=(0.01, 0.2),
    A_range::Tuple{Float64, Float64}=(0.0, 1.0),
    initial_tc_points::Int=30,
    initial_A_points::Int=21,
    tc_refine_iters::Int=6,
    A_refine_iters::Int=7,
    min_span_tc::Float64=1e-4,
    min_span_A::Float64=1e-4,
    min_points_total::Int=10,
    min_points_per_L::Int=1,
    max_deltaT::Union{Nothing, Float64}=nothing,
    warn_on_missing::Bool=true,
)
    poly_order >= 0 || error("poly_order must be non-negative.")
    min_points_total > poly_order + 1 || error("min_points_total must exceed polynomial parameter count.")
    if max_deltaT !== nothing
        (isfinite(max_deltaT) && max_deltaT > 0.0) || error("max_deltaT must be a positive finite number or nothing.")
    end

    Tc_lo, Tc_hi = tc_range
    A_lo, A_hi = A_range
    (isfinite(Tc_lo) && isfinite(Tc_hi) && Tc_hi > Tc_lo) || error("Invalid tc_range=$(tc_range)")
    (isfinite(A_lo) && isfinite(A_hi) && A_hi > A_lo) || error("Invalid A_range=$(A_range)")

    geoms = _canonical_geometry_pairs_from_l_list(L_list, geometry_pairs)
    isempty(geoms) && error("No geometries were provided.")

    ps_rows = _collect_ps_series_for_collapse(
        sim_info,
        wanted_n,
        U,
        geoms;
        prefer_full=prefer_full,
        sims_all=sims_all,
        warn_on_missing=warn_on_missing,
    )
    isempty(ps_rows) && error("No valid Ps series were found for requested geometries.")

    A_cache = Dict{Float64, NamedTuple}()
    get_A_opt = function (Tc::Float64)
        if !haskey(A_cache, Tc)
            A_cache[Tc] = _optimize_A_for_fixed_tc(
                ps_rows,
                Tc,
                poly_order,
                (A_lo, A_hi);
                initial_A_points=initial_A_points,
                A_refine_iters=A_refine_iters,
                min_span_A=min_span_A,
                min_points_total=min_points_total,
                min_points_per_L=min_points_per_L,
                max_deltaT=max_deltaT,
            )
        end
        return A_cache[Tc]
    end

    obj_Tc = function (Tc::Float64)
        A_opt = get_A_opt(Tc)
        return A_opt.best_y
    end

    Tc_opt = _interval_halving_minimize(
        obj_Tc,
        Tc_lo,
        Tc_hi;
        initial_points=initial_tc_points,
        refine_iters=tc_refine_iters,
        min_span=min_span_tc,
    )

    best_Tc = Tc_opt.best_x
    best_A_result = get_A_opt(best_Tc)
    best_A = best_A_result.best_x
    best_chi2_red = best_A_result.best_y

    best_pts = _build_ps_collapse_points(
        ps_rows,
        best_Tc,
        best_A;
        min_points_per_L=min_points_per_L,
        max_deltaT=max_deltaT,
    )
    best_fit = _weighted_poly_reduced_chi2(best_pts.x, best_pts.y, best_pts.sigma, poly_order)

    tc_curve = Tc_opt.xs
    chi2_vs_tc = Float64[]
    A_best_vs_tc = Float64[]
    for Tc in tc_curve
        Ares = get_A_opt(Tc)
        push!(chi2_vs_tc, Ares.best_y)
        push!(A_best_vs_tc, Ares.best_x)
    end

    return (
        best_Tc=best_Tc,
        best_A=best_A,
        best_reduced_chi2=best_chi2_red,
        chi2_vs_Tc=(Tc=tc_curve, chi2=chi2_vs_tc, A_best=A_best_vs_tc),
        chi2_vs_A_at_best_Tc=(A=best_A_result.xs, chi2=best_A_result.ys),
        best_fit=(coeffs=best_fit.coeffs, dof=best_fit.dof, n_points=length(best_pts.x)),
        best_collapse_points=(
            x=best_pts.x,
            y=best_pts.y,
            sigma=best_pts.sigma,
            counts_by_L=best_pts.counts_by_L,
            points_by_L=best_pts.points_by_L,
        ),
        metadata=(
            n=Float64(wanted_n),
            U=Float64(U),
            geometry_pairs=geoms,
            labels=[row.label for row in ps_rows],
            poly_order=poly_order,
            tc_range=(Tc_lo, Tc_hi),
            A_range=(A_lo, A_hi),
            min_points_total=min_points_total,
            min_points_per_L=min_points_per_L,
            max_deltaT=max_deltaT,
        ),
    )
end

"""
Geometry-pairs overload:
fit_tc_from_ps_collapse(sim_info, wanted_n, U, geometry_pairs; ...)
"""
function fit_tc_from_ps_collapse(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs::AbstractVector{<:Tuple};
    kwargs...
)
    l_list = [Int(round(Float64(g[1]))) for g in geometry_pairs]
    return fit_tc_from_ps_collapse(
        sim_info,
        wanted_n,
        U,
        l_list;
        geometry_pairs=geometry_pairs,
        kwargs...,
    )
end

"""
    _eval_poly_coeffs(coeffs, xvals) -> Vector{Float64}

Evaluate Σ_k coeffs[k+1]·x^k at each x.
"""
function _eval_poly_coeffs(
    coeffs::AbstractVector,
    xvals::AbstractVector,
)
    y = zeros(Float64, length(xvals))
    for (i, x) in enumerate(xvals)
        xp = 1.0
        yi = 0.0
        for c in coeffs
            yi += Float64(c) * xp
            xp *= Float64(x)
        end
        y[i] = yi
    end
    return y
end

"""
Build Ps collapse points using per-L polynomial interpolation to fill missing T values.

Same return shape as `_build_ps_collapse_points`, but:
1. For each L, fits a degree-`poly_order_interp` weighted polynomial to Ps(T) on
   the window T in (Tc + min_deltaT, Tc + max_deltaT] (weights = 1/dPs).
2. Defines a common T grid:
   - If `n_common_T === nothing` (default): union of all measured T_i (across
     all L) that fall in the window.
   - If `n_common_T` is an Int: an evenly-spaced grid of exactly that many T
     points in (Tc + min_deltaT, Tc + max_deltaT] (independent of measurements).
3. Each L contributes one point at every T in the common grid: the measurement
   when present (within `T_match_tol`), otherwise the polynomial prediction with
   its propagated variance phi^T (X^T W^2 X)^{-1} phi.

With `n_common_T` set, per-L count is exactly `n_common_T` for every Tc, so the
DoF is truly constant across the scan (no boundary jumps). Without it, per-L
count = |union of measured T_i in window| and changes whenever a measurement
crosses Tc + min_deltaT or Tc + max_deltaT.

Returns an empty result (which the outer optimizer treats as Inf) if any L has
fewer than `min_points_per_L` in-window points for the polynomial fit.
"""
function _build_ps_collapse_points_interpolated(
    ps_rows::Vector,
    Tc::Float64,
    A::Float64;
    poly_order_interp::Int=4,
    min_points_per_L::Int=poly_order_interp + 1,
    min_deltaT::Union{Nothing, Float64}=nothing,
    max_deltaT::Float64,
    T_match_tol::Float64=1e-9,
    n_common_T::Union{Nothing, Int}=nothing,
)
    poly_order_interp >= 0 || error("poly_order_interp must be non-negative.")
    min_points_per_L >= poly_order_interp + 1 || error("min_points_per_L must be >= poly_order_interp + 1.")
    (isfinite(max_deltaT) && max_deltaT > 0.0) || error("max_deltaT must be a positive finite number.")
    delta_lo = min_deltaT === nothing ? 0.0 : Float64(min_deltaT)
    delta_lo >= 0.0 || error("min_deltaT must be non-negative.")
    max_deltaT > delta_lo || error("max_deltaT must be > min_deltaT.")
    if n_common_T !== nothing
        n_common_T > 0 || error("n_common_T must be a positive integer or nothing.")
    end

    empty_pts_by_L = Dict{Int, NamedTuple{(:x, :y, :sigma, :T), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    empty_result = (
        x=Float64[], y=Float64[], sigma=Float64[],
        counts_by_L=Dict{Int, Int}(),
        points_by_L=empty_pts_by_L,
    )

    ncoef = poly_order_interp + 1

    per_L_data = NamedTuple[]
    union_T = Float64[]

    for row in ps_rows
        L = Int(row.L)
        T_inw = Float64[]
        Ps_inw = Float64[]
        dPs_inw = Float64[]

        npts = min(length(row.T), length(row.Ps), length(row.dPs))
        for i in 1:npts
            T = Float64(row.T[i])
            deltaT = T - Tc
            (deltaT > delta_lo && deltaT <= max_deltaT) || continue
            p = Float64(row.Ps[i])
            s = abs(Float64(row.dPs[i]))
            if isfinite(T) && isfinite(p) && isfinite(s) && s > 0.0
                push!(T_inw, T)
                push!(Ps_inw, p)
                push!(dPs_inw, s)
            end
        end

        length(T_inw) >= min_points_per_L || return empty_result

        n_in = length(T_inw)
        X = Matrix{Float64}(undef, n_in, ncoef)
        for i in 1:n_in
            tp = 1.0
            for j in 1:ncoef
                X[i, j] = tp
                tp *= T_inw[i]
            end
        end
        w_vec = 1.0 ./ dPs_inw
        WX = X .* w_vec
        Wy = Ps_inw .* w_vec

        coeffs = try
            WX \ Wy
        catch
            return empty_result
        end
        # Inverse covariance of the coefficients = (X' W^2 X)
        G = transpose(WX) * WX

        push!(per_L_data, (
            L=L,
            T_meas=T_inw,
            Ps_meas=Ps_inw,
            dPs_meas=dPs_inw,
            coeffs=coeffs,
            G=G,
        ))
        append!(union_T, T_inw)
    end

    isempty(per_L_data) && return empty_result

    common_T = Float64[]
    if n_common_T !== nothing
        grid_lo = Tc + delta_lo
        grid_hi = Tc + max_deltaT
        step = (grid_hi - grid_lo) / n_common_T
        for i in 1:n_common_T
            push!(common_T, grid_lo + step * i)
        end
    else
        sort!(union_T)
        for T in union_T
            if isempty(common_T) || abs(T - last(common_T)) > T_match_tol
                push!(common_T, T)
            end
        end
    end
    isempty(common_T) && return empty_result

    xs = Float64[]
    ys = Float64[]
    sigmas = Float64[]
    counts_by_L = Dict{Int, Int}()
    x_by_L = Dict{Int, Vector{Float64}}()
    y_by_L = Dict{Int, Vector{Float64}}()
    sigma_by_L = Dict{Int, Vector{Float64}}()
    T_by_L = Dict{Int, Vector{Float64}}()

    for data in per_L_data
        L = data.L
        L_float = Float64(L)
        ps_scale = L_float^(-7.0 / 4.0)
        local_x = get!(x_by_L, L, Float64[])
        local_y = get!(y_by_L, L, Float64[])
        local_sigma = get!(sigma_by_L, L, Float64[])
        local_T = get!(T_by_L, L, Float64[])
        c = 0

        for T in common_T
            deltaT = T - Tc
            deltaT <= 0.0 && continue

            i_meas = nothing
            for (k, Tm) in enumerate(data.T_meas)
                if abs(Tm - T) <= T_match_tol
                    i_meas = k
                    break
                end
            end

            Ps_val = 0.0
            Ps_sig = 0.0
            if i_meas !== nothing
                Ps_val = data.Ps_meas[i_meas]
                Ps_sig = data.dPs_meas[i_meas]
            else
                phi = Vector{Float64}(undef, ncoef)
                tp = 1.0
                for j in 1:ncoef
                    phi[j] = tp
                    tp *= T
                end
                Ps_val = sum(phi .* data.coeffs)
                v = try
                    data.G \ phi
                catch
                    continue
                end
                var_pred = sum(phi .* v)
                Ps_sig = var_pred > 0.0 ? sqrt(var_pred) : 0.0
            end

            x = L_float * exp(-A / sqrt(deltaT))
            y = Ps_val * ps_scale
            s = Ps_sig * ps_scale

            if isfinite(x) && isfinite(y) && isfinite(s) && s > 0.0
                push!(xs, x); push!(ys, y); push!(sigmas, s)
                push!(local_x, x); push!(local_y, y); push!(local_sigma, s); push!(local_T, T)
                c += 1
            end
        end

        counts_by_L[L] = c
    end

    points_by_L = Dict{Int, NamedTuple{(:x, :y, :sigma, :T), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    for L in keys(counts_by_L)
        points_by_L[L] = (
            x=get(x_by_L, L, Float64[]),
            y=get(y_by_L, L, Float64[]),
            sigma=get(sigma_by_L, L, Float64[]),
            T=get(T_by_L, L, Float64[]),
        )
    end

    return (x=xs, y=ys, sigma=sigmas, counts_by_L=counts_by_L, points_by_L=points_by_L)
end

"""
    _optimize_A_for_fixed_tc_interpolated(ps_rows, Tc, poly_order, A_range; max_deltaT, ...)

Same as `_optimize_A_for_fixed_tc`, but each L contributes points on a common T grid
(measured where available, else from a per-L polynomial interpolation of Ps(T)), so
every L weighs the same (see `fit_tc_from_ps_collapse_interpolated`).
"""
function _optimize_A_for_fixed_tc_interpolated(
    ps_rows::Vector,
    Tc::Float64,
    poly_order::Int,
    A_range::Tuple{Float64, Float64};
    initial_A_points::Int=21,
    A_refine_iters::Int=7,
    min_span_A::Float64=1e-4,
    min_points_total::Int=10,
    min_points_per_L::Int=5,
    poly_order_interp::Int=4,
    min_deltaT::Union{Nothing, Float64}=nothing,
    max_deltaT::Float64,
    T_match_tol::Float64=1e-9,
    n_common_T::Union{Nothing, Int}=nothing,
)
    A_lo, A_hi = A_range

    obj_A = function (A::Float64)
        pts = _build_ps_collapse_points_interpolated(
            ps_rows, Tc, A;
            poly_order_interp=poly_order_interp,
            min_points_per_L=min_points_per_L,
            min_deltaT=min_deltaT,
            max_deltaT=max_deltaT,
            T_match_tol=T_match_tol,
            n_common_T=n_common_T,
        )
        length(pts.x) >= min_points_total || return Inf
        fit = _weighted_poly_reduced_chi2(pts.x, pts.y, pts.sigma, poly_order)
        return fit.chi2_red
    end

    return _interval_halving_minimize(
        obj_A, A_lo, A_hi;
        initial_points=initial_A_points,
        refine_iters=A_refine_iters,
        min_span=min_span_A,
    )
end

"""
Fit Tc by pooled Ps finite-size collapse with per-L polynomial interpolation.

Same idea as `fit_tc_from_ps_collapse`, but for each candidate Tc:
1. For each L, fits a degree-`poly_order_interp` weighted polynomial to Ps(T) on
   the window T in (Tc + min_deltaT, Tc + max_deltaT].
2. Defines a common T grid = union of measured T_i across all L (within window).
3. Each L contributes one point at every T in the common grid: measured if
   available at that T (within `T_match_tol`), otherwise the polynomial
   prediction with propagated variance.
4. Pools all points and fits a degree-`poly_order` polynomial in
   x = L * exp(-A/sqrt(T - Tc)); minimizes reduced chi^2 over A (inner) and Tc
   (outer) by interval halving.

Each L contributes the same number of points (|common T grid|), so the DoF stays
constant as long as no measured T_i crosses the window boundary during the Tc
scan. Different per-L measurement grids no longer bias the pooled fit toward
the densest-sampled L.

Required: `max_deltaT` (defines the interpolation window).

Primary input style:
fit_tc_from_ps_collapse_interpolated(sim_info, wanted_n, U, L_list; max_deltaT=...)
"""
function fit_tc_from_ps_collapse_interpolated(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    L_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    sims_all=nothing,
    poly_order::Int=2,
    poly_order_interp::Int=4,
    tc_range::Tuple{Float64, Float64}=(0.01, 0.2),
    A_range::Tuple{Float64, Float64}=(0.0, 1.0),
    initial_tc_points::Int=30,
    initial_A_points::Int=21,
    tc_refine_iters::Int=6,
    A_refine_iters::Int=7,
    min_span_tc::Float64=1e-4,
    min_span_A::Float64=1e-4,
    min_points_total::Int=10,
    min_points_per_L::Int=poly_order_interp + 1,
    min_deltaT::Union{Nothing, Float64}=nothing,
    max_deltaT::Float64,
    T_match_tol::Float64=1e-9,
    n_common_T::Union{Nothing, Int}=nothing,
    warn_on_missing::Bool=true,
)
    poly_order >= 0 || error("poly_order must be non-negative.")
    poly_order_interp >= 0 || error("poly_order_interp must be non-negative.")
    min_points_total > poly_order + 1 || error("min_points_total must exceed polynomial parameter count.")
    min_points_per_L >= poly_order_interp + 1 || error("min_points_per_L must be >= poly_order_interp + 1.")
    (isfinite(max_deltaT) && max_deltaT > 0.0) || error("max_deltaT must be a positive finite number.")
    if min_deltaT !== nothing
        (isfinite(min_deltaT) && min_deltaT >= 0.0) || error("min_deltaT must be a non-negative finite number or nothing.")
        max_deltaT > min_deltaT || error("max_deltaT must be strictly greater than min_deltaT.")
    end
    if n_common_T !== nothing
        n_common_T > 0 || error("n_common_T must be a positive integer or nothing.")
    end

    Tc_lo, Tc_hi = tc_range
    A_lo, A_hi = A_range
    (isfinite(Tc_lo) && isfinite(Tc_hi) && Tc_hi > Tc_lo) || error("Invalid tc_range=$(tc_range)")
    (isfinite(A_lo) && isfinite(A_hi) && A_hi > A_lo) || error("Invalid A_range=$(A_range)")

    geoms = _canonical_geometry_pairs_from_l_list(L_list, geometry_pairs)
    isempty(geoms) && error("No geometries were provided.")

    ps_rows = _collect_ps_series_for_collapse(
        sim_info, wanted_n, U, geoms;
        prefer_full=prefer_full, sims_all=sims_all, warn_on_missing=warn_on_missing,
    )
    isempty(ps_rows) && error("No valid Ps series were found for requested geometries.")

    A_cache = Dict{Float64, NamedTuple}()
    get_A_opt = function (Tc::Float64)
        if !haskey(A_cache, Tc)
            A_cache[Tc] = _optimize_A_for_fixed_tc_interpolated(
                ps_rows, Tc, poly_order, (A_lo, A_hi);
                initial_A_points=initial_A_points,
                A_refine_iters=A_refine_iters,
                min_span_A=min_span_A,
                min_points_total=min_points_total,
                min_points_per_L=min_points_per_L,
                poly_order_interp=poly_order_interp,
                min_deltaT=min_deltaT,
                max_deltaT=max_deltaT,
                T_match_tol=T_match_tol,
                n_common_T=n_common_T,
            )
        end
        return A_cache[Tc]
    end

    obj_Tc = function (Tc::Float64)
        return get_A_opt(Tc).best_y
    end

    Tc_opt = _interval_halving_minimize(
        obj_Tc, Tc_lo, Tc_hi;
        initial_points=initial_tc_points,
        refine_iters=tc_refine_iters,
        min_span=min_span_tc,
    )

    best_Tc = Tc_opt.best_x
    best_A_result = get_A_opt(best_Tc)
    best_A = best_A_result.best_x
    best_chi2_red = best_A_result.best_y

    best_pts = _build_ps_collapse_points_interpolated(
        ps_rows, best_Tc, best_A;
        poly_order_interp=poly_order_interp,
        min_points_per_L=min_points_per_L,
        min_deltaT=min_deltaT,
        max_deltaT=max_deltaT,
        T_match_tol=T_match_tol,
        n_common_T=n_common_T,
    )
    best_fit = _weighted_poly_reduced_chi2(best_pts.x, best_pts.y, best_pts.sigma, poly_order)

    tc_curve = Tc_opt.xs
    chi2_vs_tc = Float64[]
    A_best_vs_tc = Float64[]
    for Tc in tc_curve
        Ares = get_A_opt(Tc)
        push!(chi2_vs_tc, Ares.best_y)
        push!(A_best_vs_tc, Ares.best_x)
    end

    return (
        best_Tc=best_Tc,
        best_A=best_A,
        best_reduced_chi2=best_chi2_red,
        chi2_vs_Tc=(Tc=tc_curve, chi2=chi2_vs_tc, A_best=A_best_vs_tc),
        chi2_vs_A_at_best_Tc=(A=best_A_result.xs, chi2=best_A_result.ys),
        best_fit=(coeffs=best_fit.coeffs, dof=best_fit.dof, n_points=length(best_pts.x)),
        best_collapse_points=(
            x=best_pts.x,
            y=best_pts.y,
            sigma=best_pts.sigma,
            counts_by_L=best_pts.counts_by_L,
            points_by_L=best_pts.points_by_L,
        ),
        metadata=(
            n=Float64(wanted_n),
            U=Float64(U),
            geometry_pairs=geoms,
            labels=[row.label for row in ps_rows],
            poly_order=poly_order,
            poly_order_interp=poly_order_interp,
            tc_range=(Tc_lo, Tc_hi),
            A_range=(A_lo, A_hi),
            min_points_total=min_points_total,
            min_points_per_L=min_points_per_L,
            min_deltaT=min_deltaT,
            max_deltaT=max_deltaT,
            T_match_tol=T_match_tol,
            n_common_T=n_common_T,
            interpolated=true,
        ),
    )
end

"""
Geometry-pairs overload:
fit_tc_from_ps_collapse_interpolated(sim_info, wanted_n, U, geometry_pairs; max_deltaT=...)
"""
function fit_tc_from_ps_collapse_interpolated(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    geometry_pairs::AbstractVector{<:Tuple};
    kwargs...
)
    l_list = [Int(round(Float64(g[1]))) for g in geometry_pairs]
    return fit_tc_from_ps_collapse_interpolated(
        sim_info, wanted_n, U, l_list;
        geometry_pairs=geometry_pairs, kwargs...,
    )
end

"""
Plot reduced chi^2 versus Tc using the output of fit_tc_from_ps_collapse.

Usage:
plot_ps_collapse_chi2_vs_tc(fit_result)
"""
function plot_ps_collapse_chi2_vs_tc(
    fit_result;
    title::String="",
    yscale=:identity,
)
    tc_vec = Float64.(fit_result.chi2_vs_Tc.Tc)
    chi_vec = Float64.(fit_result.chi2_vs_Tc.chi2)

    isempty(tc_vec) && error("fit_result.chi2_vs_Tc.Tc is empty.")
    length(tc_vec) == length(chi_vec) || error("Tc and chi2 arrays have inconsistent lengths.")

    p = sortperm(tc_vec)
    tc_plot = tc_vec[p]
    chi_plot = chi_vec[p]

    title_use = isempty(title) ? "Reduced chi^2 vs Tc" : title
    fig = plot(
        tc_plot,
        chi_plot;
        xlabel="Tc",
        ylabel="reduced chi^2",
        title=title_use,
        m=:circle,
        ms=4,
        lw=1.8,
        legend=false,
        size=(900, 540),
        left_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
        yscale=yscale,
    )

    best_tc = Float64(fit_result.best_Tc)
    best_chi = Float64(fit_result.best_reduced_chi2)
    if isfinite(best_tc) && isfinite(best_chi)
        scatter!(fig, [best_tc], [best_chi]; m=:diamond, ms=7, label=false, color=:red)
    end

    return fig
end

"""
Plot reduced chi^2 versus A at the best Tc, from fit_tc_from_ps_collapse output.

Usage:
plot_ps_collapse_chi2_vs_A_best_tc(fit_result)
"""
function plot_ps_collapse_chi2_vs_A_best_tc(
    fit_result;
    title::String="",
    yscale=:identity,
)
    A_vec = Float64.(fit_result.chi2_vs_A_at_best_Tc.A)
    chi_vec = Float64.(fit_result.chi2_vs_A_at_best_Tc.chi2)

    isempty(A_vec) && error("fit_result.chi2_vs_A_at_best_Tc.A is empty.")
    length(A_vec) == length(chi_vec) || error("A and chi2 arrays have inconsistent lengths.")

    p = sortperm(A_vec)
    A_plot = A_vec[p]
    chi_plot = chi_vec[p]

    title_use = isempty(title) ? "Reduced chi^2 vs A at best Tc" : title
    fig = plot(
        A_plot,
        chi_plot;
        xlabel="A",
        ylabel="reduced chi^2",
        title=title_use,
        m=:circle,
        ms=4,
        lw=1.8,
        legend=false,
        size=(900, 540),
        left_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
        yscale=yscale,
    )

    best_A = Float64(fit_result.best_A)
    best_chi = Float64(fit_result.best_reduced_chi2)
    if isfinite(best_A) && isfinite(best_chi)
        scatter!(fig, [best_A], [best_chi]; m=:diamond, ms=7, label=false, color=:red)
    end

    return fig
end

"""
Plot best-collapse points grouped by L with the best polynomial fit overlay.

Usage:
plot_ps_collapse_best_fit(fit_result)

Keyword option:
- max_deltaT_override: optional plotting-only cap on (T - Tc). If not provided,
  uses fit_result.metadata.max_deltaT when available.
"""
function plot_ps_collapse_best_fit(
    fit_result;
    title::String="",
    show_errors::Bool=true,
    xscale=:identity,
    yscale=:identity,
    fit_grid_points::Int=400,
    max_deltaT_override::Union{Nothing, Float64}=nothing,
)
    points_by_L = fit_result.best_collapse_points.points_by_L
    coeffs = fit_result.best_fit.coeffs

    isempty(points_by_L) && error("fit_result.best_collapse_points.points_by_L is empty.")

    n_val = haskey(fit_result, :metadata) && haskey(fit_result.metadata, :n) ? fit_result.metadata.n : NaN
    U_val = haskey(fit_result, :metadata) && haskey(fit_result.metadata, :U) ? fit_result.metadata.U : NaN
    max_deltaT_meta = haskey(fit_result, :metadata) && haskey(fit_result.metadata, :max_deltaT) ? fit_result.metadata.max_deltaT : nothing
    max_deltaT_use = max_deltaT_override === nothing ? max_deltaT_meta : max_deltaT_override
    if max_deltaT_use !== nothing
        (isfinite(max_deltaT_use) && max_deltaT_use > 0.0) || error("max_deltaT_override must be a positive finite number or nothing.")
    end
    best_tc = Float64(fit_result.best_Tc)
    best_A = Float64(fit_result.best_A)

    max_dt_lbl = max_deltaT_use === nothing ? "" : ", max(T-Tc)=$(round(Float64(max_deltaT_use), sigdigits=5))"
    title_use = isempty(title) ? "Best Ps collapse: n=$(n_val), U=$(U_val), Tc=$(round(best_tc, sigdigits=6)), A=$(round(best_A, sigdigits=6))$(max_dt_lbl)" : title
    fig = plot(
        xlabel="x = L*exp(-A/sqrt(T-Tc))",
        ylabel="Ps*L^(-7/4)",
        title=title_use,
        legend=:topright,
        size=(980, 580),
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
        xscale=xscale,
        yscale=yscale,
    )

    pooled_x = Float64[]
    for L in sort(collect(keys(points_by_L)))
        pts = points_by_L[L]
        x = Float64.(pts.x)
        y = Float64.(pts.y)
        s = Float64.(pts.sigma)
        t = Float64.(pts.T)
        npts = min(length(x), length(y), length(s), length(t))
        npts == 0 && continue

        x = x[1:npts]
        y = y[1:npts]
        s = s[1:npts]
        t = t[1:npts]

        keep = findall(i -> isfinite(x[i]) && isfinite(y[i]) && isfinite(s[i]) && isfinite(t[i]) && s[i] >= 0.0, eachindex(x))
        if max_deltaT_use !== nothing
            keep = [i for i in keep if (t[i] - best_tc) > 0.0 && (t[i] - best_tc) <= Float64(max_deltaT_use)]
        end
        isempty(keep) && continue
        x = x[keep]
        y = y[keep]
        s = s[keep]

        p = sortperm(x)
        x = x[p]
        y = y[p]
        s = s[p]

        append!(pooled_x, x)

        if show_errors
            plot!(
                fig,
                x,
                y;
                yerr=s,
                m=:circle,
                ms=4,
                lw=1.3,
                label="L=$(L) (N=$(length(x)))",
            )
        else
            plot!(
                fig,
                x,
                y;
                m=:circle,
                ms=4,
                lw=1.3,
                label="L=$(L) (N=$(length(x)))",
            )
        end
    end

    if !isempty(coeffs) && !isempty(pooled_x)
        xmin = minimum(pooled_x)
        xmax = maximum(pooled_x)
        if isfinite(xmin) && isfinite(xmax) && xmax > xmin
            ngrid = max(fit_grid_points, 40)
            xfit = collect(range(xmin, xmax; length=ngrid))
            yfit = _eval_poly_coeffs(coeffs, xfit)
            plot!(fig, xfit, yfit; lw=2.6, color=:black, label="best polynomial fit")
        end
    end

    return fig
end

"""
Collect Tc values across available geometries for each U.

Returns a vector of NamedTuples:
[(n=..., U=..., tc_mean=..., tc_std=..., tc_err=..., tc_count=...,
  tc_values=[...], tc_errs=[...]), ...]

- `tc_std`: sample standard deviation across geometries (no MC errors used).
- `tc_err`: when `tc_method=:fit`, the inverse-variance-weighted uncertainty of
  `tc_mean` (combines per-geometry fit errors with geometry-to-geometry spread
  via `inverse_variance_weighted_mean`); equal to `tc_std` when
  `tc_method=:crossing` (no per-sim uncertainty available).

Arguments:
- `tc_method`: `:crossing` (default, uses `find_tc`) or `:fit` (uses
  `find_tc_fit`; yields per-geometry Tc error bars).
- `fit_window`, `fit_order`: forwarded to `find_tc_fit`.

Usage:
collect_tc_vs_U(sim_info, wanted_n, U_list; geometry_pairs=nothing)
"""
function collect_tc_vs_U(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    lowest_q::Bool=false,
    tc_method::Symbol=:crossing,
    fit_window::Int=3,
    fit_order::Union{Int, Symbol}=:auto,
)
    results = NamedTuple[]
    sims_all_local = get_sims_all(sim_info; wanted_n=Float64(wanted_n))

    extract_tc = if tc_method === :fit
        sim -> find_tc_fit(sim; lowest_q=lowest_q, window=fit_window, order=fit_order)
    elseif tc_method === :crossing
        function (sim)
            tc, crosses, _ = find_tc(sim; lowest_q=lowest_q)
            return tc, NaN, crosses
        end
    else
        error("Unknown tc_method=$(tc_method); expected :crossing or :fit")
    end

    for U_raw in U_list
        U = Float64(U_raw)
        geoms = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : collect(geometry_pairs)

        if isempty(geoms)
            @warn "No geometries found for U; skipping" wanted_n=wanted_n U=U
            push!(results, (
                n=Float64(wanted_n), U=U,
                tc_mean=NaN, tc_std=NaN, tc_err=NaN,
                tc_count=0, tc_values=Float64[], tc_errs=Float64[],
            ))
            continue
        end

        all_series = build_unified_series(
            sim_info,
            wanted_n,
            U,
            geoms;
            prefer_full=prefer_full,
            sims_all=sims_all_local,
        )

        tc_vals = Float64[]
        tc_errs = Float64[]
        for (lbl, sim) in all_series
            isempty(sim) && continue
            tc, tc_err, worked = extract_tc(sim)
            if worked && isfinite(tc)
                push!(tc_vals, Float64(tc))
                push!(tc_errs, Float64(tc_err))
            else
                @warn "Could not extract Tc for series; skipping" wanted_n=wanted_n U=U label=lbl
            end
        end

        tc_mean, tc_std = _mean_std(tc_vals)

        # Combined error when fit errors are available.
        tc_err_agg = NaN
        if tc_method === :fit && !isempty(tc_vals)
            finite_errs = [(isfinite(e) && e > 0) for e in tc_errs]
            if all(finite_errs)
                # inverse_variance_weighted_mean returns (mean, variance);
                # we keep tc_mean from simple mean for compatibility and
                # report only the combined uncertainty.
                _, wvar = inverse_variance_weighted_mean(tc_vals, tc_errs)
                tc_err_agg = wvar > 0 ? sqrt(wvar) : NaN
            elseif length(tc_vals) == 1
                tc_err_agg = tc_errs[1]
            else
                tc_err_agg = tc_std
            end
        else
            tc_err_agg = tc_std
        end

        push!(results, (
            n=Float64(wanted_n),
            U=U,
            tc_mean=tc_mean,
            tc_std=tc_std,
            tc_err=tc_err_agg,
            tc_count=length(tc_vals),
            tc_values=tc_vals,
            tc_errs=tc_errs,
        ))
    end

    return results
end

"""
Plot Tc vs U using find_tc across geometries.

For each selected n, this function computes Tc for each available geometry at each U,
then plots the mean Tc with error bars (std across geometries).

Usage:
plot_tc_vs_U(sim_info, n_list, U_list; geometry_pairs=nothing)

Returns:
- fig: plot object
- summary: vector of NamedTuples from collect_tc_vs_U
"""
function plot_tc_vs_U(
    sim_info::Dict{String, Any},
    n_list::AbstractVector,
    U_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    lowest_q::Bool=false,
    show_all_points::Bool=true,
    line_per_geometry::Bool=false,
    tc_method::Symbol=:crossing,
    fit_window::Int=3,
    fit_order::Union{Int, Symbol}=:auto,
)
    extract_tc = if tc_method === :fit
        sim -> find_tc_fit(sim; lowest_q=lowest_q, window=fit_window, order=fit_order)
    elseif tc_method === :crossing
        function (sim)
            tc, crosses, _ = find_tc(sim; lowest_q=lowest_q)
            return tc, NaN, crosses
        end
    else
        error("Unknown tc_method=$(tc_method); expected :crossing or :fit")
    end

    fig = plot(
        xlabel="U",
        ylabel="Tc",
        title="Tc vs U (BKT crossing from stiffness)",
        legend=:topright,
        size=(950, 560),
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
    )

    summary = NamedTuple[]
    for n_raw in n_list
        n = Float64(n_raw)

        if line_per_geometry
            sims_all_local = get_sims_all(sim_info; wanted_n=n)
            geometry_curves = Dict{String, Vector{Tuple{Float64, Float64, Float64}}}()

            for U_raw in U_list
                U = Float64(U_raw)
                geoms = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : collect(geometry_pairs)

                if isempty(geoms)
                    @warn "No geometries found for U; skipping" wanted_n=n U=U
                    continue
                end

                all_series = build_unified_series(
                    sim_info,
                    n,
                    U,
                    geoms;
                    prefer_full=prefer_full,
                    sims_all=sims_all_local,
                )

                tc_vals_for_U = Float64[]
                tc_errs_for_U = Float64[]
                for (lbl, sim) in all_series
                    isempty(sim) && continue
                    tc, tc_err, worked = extract_tc(sim)
                    if worked && isfinite(tc)
                        tc_f = Float64(tc)
                        err_f = Float64(tc_err)
                        push!(get!(geometry_curves, lbl, Tuple{Float64, Float64, Float64}[]), (U, tc_f, err_f))
                        push!(tc_vals_for_U, tc_f)
                        push!(tc_errs_for_U, err_f)
                    else
                        @warn "Could not extract Tc for series; skipping" wanted_n=n U=U label=lbl
                    end
                end

                tc_mean, tc_std = _mean_std(tc_vals_for_U)
                push!(summary, (
                    n=n,
                    U=U,
                    tc_mean=tc_mean,
                    tc_std=tc_std,
                    tc_err=tc_std,
                    tc_count=length(tc_vals_for_U),
                    tc_values=tc_vals_for_U,
                    tc_errs=tc_errs_for_U,
                ))
            end

            for (lbl, pts) in sort(collect(geometry_curves), by=x -> x[1])
                isempty(pts) && continue
                p = sortperm(first.(pts))
                U_pts = [pts[i][1] for i in p]
                Tc_pts = [pts[i][2] for i in p]
                Err_pts = [pts[i][3] for i in p]
                yerr_arg = any(isfinite, Err_pts) ? replace(Err_pts, NaN => 0.0) : nothing
                plot!(
                    fig,
                    U_pts,
                    Tc_pts;
                    m=:circle,
                    ms=4,
                    lw=1.8,
                    label="n=$(n), $(lbl)",
                    yerr=yerr_arg,
                )
            end

            continue
        end

        rows = collect_tc_vs_U(
            sim_info,
            n,
            U_list;
            geometry_pairs=geometry_pairs,
            prefer_full=prefer_full,
            lowest_q=lowest_q,
            tc_method=tc_method,
            fit_window=fit_window,
            fit_order=fit_order,
        )
        append!(summary, rows)

        U_pts = Float64[]
        Tc_mean = Float64[]
        Tc_err = Float64[]

        for row in rows
            if row.tc_count > 0 && isfinite(row.tc_mean)
                push!(U_pts, row.U)
                push!(Tc_mean, row.tc_mean)
                push!(Tc_err, isfinite(row.tc_err) ? row.tc_err : row.tc_std)

                if show_all_points
                    vals = row.tc_values
                    if !isempty(vals)
                        scatter!(
                            fig,
                            fill(row.U, length(vals)),
                            vals;
                            label=false,
                            m=:circle,
                            ms=3,
                            alpha=0.35,
                            color=:gray,
                        )
                    end
                end
            end
        end

        if !isempty(U_pts)
            p = sortperm(U_pts)
            yerr_arg = any(isfinite, Tc_err) ? replace(Tc_err[p], NaN => 0.0) : nothing
            plot!(
                fig,
                U_pts[p],
                Tc_mean[p];
                yerr=yerr_arg,
                m=:diamond,
                ms=6,
                lw=2,
                label="n=$(n)",
            )
        else
            @warn "No valid Tc points found for n; nothing plotted" n=n
        end
    end

    return fig, summary
end

"""
Convenience overload for a single n.
"""
function plot_tc_vs_U(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U_list::AbstractVector;
    kwargs...
)
    return plot_tc_vs_U(sim_info, [Float64(wanted_n)], U_list; kwargs...)
end

"""
Collect T* values across available geometries for each U using find_Tstar.

Returns a vector of NamedTuples:
[(n=..., U=..., tstar_mean=..., tstar_std=..., tstar_count=..., tstar_values=[...]), ...]

Usage:
collect_tstar_vs_U(sim_info, wanted_n, U_list; geometry_pairs=nothing)
"""
function collect_tstar_vs_U(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    fit_len::Int=3,
)
    results = NamedTuple[]
    sims_all_local = get_sims_all(sim_info; wanted_n=Float64(wanted_n))

    for U_raw in U_list
        U = Float64(U_raw)
        geoms = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : collect(geometry_pairs)

        if isempty(geoms)
            @warn "No geometries found for U; skipping" wanted_n=wanted_n U=U
            push!(results, (n=Float64(wanted_n), U=U, tstar_mean=NaN, tstar_std=NaN, tstar_count=0, tstar_values=Float64[]))
            continue
        end

        all_series = build_unified_series(
            sim_info,
            wanted_n,
            U,
            geoms;
            prefer_full=prefer_full,
            sims_all=sims_all_local,
        )

        tstar_vals = Float64[]
        for (lbl, sim) in all_series
            isempty(sim) && continue
            tstar, _, worked = find_Tstar(sim; len=fit_len)
            if worked && tstar !== nothing && isfinite(tstar)
                push!(tstar_vals, Float64(tstar))
            else
                @warn "Could not extract T* for series; skipping" wanted_n=wanted_n U=U label=lbl fit_len=fit_len
            end
        end

        tstar_mean, tstar_std = _mean_std(tstar_vals)
        push!(results, (
            n=Float64(wanted_n),
            U=U,
            tstar_mean=tstar_mean,
            tstar_std=tstar_std,
            tstar_count=length(tstar_vals),
            tstar_values=tstar_vals,
        ))
    end

    return results
end

"""
Plot T* vs U using find_Tstar across geometries.

For each selected n, this function computes T* for each available geometry at each U,
then plots the mean T* with error bars (std across geometries).

Usage:
plot_tstar_vs_U(sim_info, n_list, U_list; geometry_pairs=nothing)

Returns:
- fig: plot object
- summary: vector of NamedTuples from collect_tstar_vs_U
"""
function plot_tstar_vs_U(
    sim_info::Dict{String, Any},
    n_list::AbstractVector,
    U_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    fit_len::Int=3,
    show_all_points::Bool=true,
    line_per_geometry::Bool=false,
)
    fig = plot(
        xlabel="U",
        ylabel="T*",
        title="T* vs U (from chi_spin maximum)",
        legend=:topright,
        size=(950, 560),
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
    )

    summary = NamedTuple[]
    for n_raw in n_list
        n = Float64(n_raw)

        if line_per_geometry
            sims_all_local = get_sims_all(sim_info; wanted_n=n)
            geometry_curves = Dict{String, Vector{Tuple{Float64, Float64}}}()

            for U_raw in U_list
                U = Float64(U_raw)
                geoms = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : collect(geometry_pairs)

                if isempty(geoms)
                    @warn "No geometries found for U; skipping" wanted_n=n U=U
                    continue
                end

                all_series = build_unified_series(
                    sim_info,
                    n,
                    U,
                    geoms;
                    prefer_full=prefer_full,
                    sims_all=sims_all_local,
                )

                tstar_vals_for_U = Float64[]
                for (lbl, sim) in all_series
                    isempty(sim) && continue
                    tstar, _, worked = find_Tstar(sim; len=fit_len)
                    if worked && tstar !== nothing && isfinite(tstar)
                        tstar_f = Float64(tstar)
                        push!(get!(geometry_curves, lbl, Tuple{Float64, Float64}[]), (U, tstar_f))
                        push!(tstar_vals_for_U, tstar_f)
                    else
                        @warn "Could not extract T* for series; skipping" wanted_n=n U=U label=lbl fit_len=fit_len
                    end
                end

                tstar_mean, tstar_std = _mean_std(tstar_vals_for_U)
                push!(summary, (
                    n=n,
                    U=U,
                    tstar_mean=tstar_mean,
                    tstar_std=tstar_std,
                    tstar_count=length(tstar_vals_for_U),
                    tstar_values=tstar_vals_for_U,
                ))
            end

            for (lbl, pts) in sort(collect(geometry_curves), by=x -> x[1])
                isempty(pts) && continue
                p = sortperm(first.(pts))
                U_pts = [pts[i][1] for i in p]
                Tstar_pts = [pts[i][2] for i in p]
                plot!(
                    fig,
                    U_pts,
                    Tstar_pts;
                    m=:circle,
                    ms=4,
                    lw=1.8,
                    label="n=$(n), $(lbl)",
                )
            end

            continue
        end

        rows = collect_tstar_vs_U(
            sim_info,
            n,
            U_list;
            geometry_pairs=geometry_pairs,
            prefer_full=prefer_full,
            fit_len=fit_len,
        )
        append!(summary, rows)

        U_pts = Float64[]
        Tstar_mean = Float64[]
        Tstar_err = Float64[]

        for row in rows
            if row.tstar_count > 0 && isfinite(row.tstar_mean)
                push!(U_pts, row.U)
                push!(Tstar_mean, row.tstar_mean)
                push!(Tstar_err, row.tstar_std)

                if show_all_points
                    vals = row.tstar_values
                    if !isempty(vals)
                        scatter!(
                            fig,
                            fill(row.U, length(vals)),
                            vals;
                            label=false,
                            m=:circle,
                            ms=3,
                            alpha=0.35,
                            color=:gray,
                        )
                    end
                end
            end
        end

        if !isempty(U_pts)
            p = sortperm(U_pts)
            plot!(
                fig,
                U_pts[p],
                Tstar_mean[p];
                yerr=Tstar_err[p],
                m=:diamond,
                ms=6,
                lw=2,
                label="n=$(n)",
            )
        else
            @warn "No valid T* points found for n; nothing plotted" n=n
        end
    end

    return fig, summary
end

"""
Convenience overload for a single n.
"""
function plot_tstar_vs_U(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U_list::AbstractVector;
    kwargs...
)
    return plot_tstar_vs_U(sim_info, [Float64(wanted_n)], U_list; kwargs...)
end

"""
Plot a phase diagram with Tc and T* versus U.

For each n in n_list:
- Tc is extracted with find_tc (stiffness crossing)
- T* is extracted with find_Tstar (chi_spin maximum)

Usage:
plot_phase_diagram_tc_tstar(sim_info, n_list, U_list; geometry_pairs=nothing)

Returns:
- fig: plot object
- summary: NamedTuple(tc=..., tstar=...) containing row-wise summaries
"""
function plot_phase_diagram_tc_tstar(
    sim_info::Dict{String, Any},
    n_list::AbstractVector,
    U_list::AbstractVector;
    geometry_pairs=nothing,
    prefer_full::Bool=true,
    lowest_q::Bool=false,
    tstar_fit_len::Int=3,
    show_all_points::Bool=false,
    title::String="Phase diagram: Tc and T* vs U",
    tc_method::Symbol=:crossing,
    fit_window::Int=3,
    fit_order::Union{Int, Symbol}=:auto,
)
    fig = plot(
        xlabel="U",
        ylabel="T",
        title=title,
        legend=:topright,
        size=(980, 580),
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
    )

    tc_summary_all = NamedTuple[]
    tstar_summary_all = NamedTuple[]

    for n_raw in n_list
        n = Float64(n_raw)

        tc_rows = collect_tc_vs_U(
            sim_info,
            n,
            U_list;
            geometry_pairs=geometry_pairs,
            prefer_full=prefer_full,
            lowest_q=lowest_q,
            tc_method=tc_method,
            fit_window=fit_window,
            fit_order=fit_order,
        )
        tstar_rows = collect_tstar_vs_U(
            sim_info,
            n,
            U_list;
            geometry_pairs=geometry_pairs,
            prefer_full=prefer_full,
            fit_len=tstar_fit_len,
        )

        append!(tc_summary_all, tc_rows)
        append!(tstar_summary_all, tstar_rows)

        U_tc = Float64[]
        Tc_mu = Float64[]
        Tc_sd = Float64[]

        for row in tc_rows
            if row.tc_count > 0 && isfinite(row.tc_mean)
                push!(U_tc, row.U)
                push!(Tc_mu, row.tc_mean)
                push!(Tc_sd, isfinite(row.tc_err) ? row.tc_err : row.tc_std)

                if show_all_points && !isempty(row.tc_values)
                    scatter!(
                        fig,
                        fill(row.U, length(row.tc_values)),
                        row.tc_values;
                        label=false,
                        m=:circle,
                        ms=3,
                        alpha=0.25,
                        color=:blue,
                    )
                end
            end
        end

        U_ts = Float64[]
        Ts_mu = Float64[]
        Ts_sd = Float64[]

        for row in tstar_rows
            if row.tstar_count > 0 && isfinite(row.tstar_mean)
                push!(U_ts, row.U)
                push!(Ts_mu, row.tstar_mean)
                push!(Ts_sd, row.tstar_std)

                if show_all_points && !isempty(row.tstar_values)
                    scatter!(
                        fig,
                        fill(row.U, length(row.tstar_values)),
                        row.tstar_values;
                        label=false,
                        m=:circle,
                        ms=3,
                        alpha=0.25,
                        color=:red,
                    )
                end
            end
        end

        if !isempty(U_tc)
            p = sortperm(U_tc)
            yerr_tc = any(isfinite, Tc_sd) ? replace(Tc_sd[p], NaN => 0.0) : nothing
            plot!(
                fig,
                U_tc[p],
                Tc_mu[p];
                yerr=yerr_tc,
                m=:diamond,
                ms=6,
                lw=2.2,
                color=:blue,
                label="Tc, n=$(n)",
            )
        else
            @warn "No valid Tc points found for n; skipping Tc curve" n=n
        end

        if !isempty(U_ts)
            p = sortperm(U_ts)
            plot!(
                fig,
                U_ts[p],
                Ts_mu[p];
                yerr=Ts_sd[p],
                m=:utriangle,
                ms=6,
                lw=2.2,
                ls=:dash,
                color=:red,
                label="T*, n=$(n)",
            )
        else
            @warn "No valid T* points found for n; skipping T* curve" n=n
        end
    end

    summary = (tc=tc_summary_all, tstar=tstar_summary_all)
    return fig, summary
end

"""
Convenience overload for a single n.
"""
function plot_phase_diagram_tc_tstar(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U_list::AbstractVector;
    kwargs...
)
    return plot_phase_diagram_tc_tstar(sim_info, [Float64(wanted_n)], U_list; kwargs...)
end

"""
    _dc_component_series(dc_vals, dc_errs; component_idx=1) -> (values, errors)

One component of a DC observable over its Δω windows: entry `component_idx` of each
window's tensor (or the values themselves if they are scalars).
"""
function _dc_component_series(dc_vals, dc_errs; component_idx::Int=1)
    if dc_vals isa AbstractVector
        if isempty(dc_vals)
            return Float64[], Float64[]
        end
        first_val = first(dc_vals)
        if first_val isa AbstractArray
            ys = Float64[]
            es = Float64[]
            for i in eachindex(dc_vals)
                v = dc_vals[i]
                e = dc_errs[i]
                if v === nothing || e === nothing || length(v) < component_idx || length(e) < component_idx
                    continue
                end
                push!(ys, Float64(v[component_idx]))
                push!(es, Float64(e[component_idx]))
            end
            return ys, es
        else
            ys = Float64[]
            es = Float64[]
            for i in eachindex(dc_vals)
                v = dc_vals[i]
                e = dc_errs[i]
                v === nothing && continue
                e === nothing && continue
                push!(ys, Float64(v))
                push!(es, Float64(e))
            end
            return ys, es
        end
    end
    return Float64[], Float64[]
end

"""
    _overlay_dc_vs_dw!(fig, sim_dict, dc_key; component_idx=1, units_MLS=true, label="dc vs dw")

Overlay the DC observable `dc_key` against its window Δω (in mean-level-spacing units
if `units_MLS`) on a δ-extrapolation plot, so δ and Δω can be compared. Skips (with a
warning) if the observable or the Δω values are missing.
"""
function _overlay_dc_vs_dw!(
    fig,
    sim_dict::Dict{String, Any},
    dc_key::String;
    component_idx::Int=1,
    units_MLS::Bool=true,
    label::String="dc vs dw",
)
    haskey(sim_dict, "means") || return fig
    means = sim_dict["means"]
    if !haskey(means, dc_key)
        @warn "Missing DC observable for overlay; skipping" dc_key=dc_key
        return fig
    end

    if !haskey(sim_dict, "dw_values")
        @warn "Missing dw_values for DC overlay; skipping" dc_key=dc_key
        return fig
    end

    dw = sim_dict["dw_values"]
    dc_vals = means[dc_key]["value"]
    dc_errs = means[dc_key]["error"]
    ys, es = _dc_component_series(dc_vals, dc_errs; component_idx=component_idx)

    if isempty(ys)
        @warn "DC observable has no plottable data; skipping" dc_key=dc_key component_idx=component_idx
        return fig
    end

    n = min(length(dw), length(ys))
    x = Float64.(dw[1:n])
    y = ys[1:n]
    e = es[1:n]

    if units_MLS
        if haskey(means, "energy_spacing_mean")
            mls = Float64(means["energy_spacing_mean"]["value"])
            mls != 0.0 && (x = x ./ mls)
        end
    end

    scatter!(
        fig,
        x,
        y;
        yerr=e,
        label=label,
        m=:square,
        ms=5,
        mc=:orange,
        msc=:orange,
        lc=:orange,
    )
    return fig
end

"""
    _plot_single_transport_delta_panel(sim_dict, sys_label, beta; observable,
        conductivity_component_idx=1, drude_component_idx=1, resistivity_component_idx=1,
        units_MLS=true)

One panel: a transport observable (`:conductivity`, `:drude` or `:resistivity`) against
the broadening δ with its δ → 0 extrapolation, and the matching DC observable against Δω.
"""
function _plot_single_transport_delta_panel(
    sim_dict::Dict{String, Any},
    sys_label::String,
    beta::Float64;
    observable::Symbol,
    conductivity_component_idx::Int=1,
    drude_component_idx::Int=1,
    resistivity_component_idx::Int=1,
    units_MLS::Bool=true,
)
    if observable == :conductivity
        reg_key = "conductivity_regular_xx_fixed"
        dc_key = "conductivity_dc_xx"
        cidx = conductivity_component_idx
    elseif observable == :drude
        reg_key = "Drude_weight_fixed"
        dc_key = "drude_dc_xx"
        cidx = drude_component_idx
    elseif observable == :resistivity
        reg_key = "resistivity_tensor_fixed"
        dc_key = "resistivity_dc_tensor"
        cidx = resistivity_component_idx
    else
        error("Unknown observable selector: $(observable). Use :conductivity, :drude, :resistivity, or :all.")
    end

    fig = plot_delta_extrapolation(
        sim_dict,
        reg_key;
        component_idx=cidx,
        units_MLS=units_MLS,
        title_prefix="$(sys_label), beta=$(beta), ",
        show_dc=false,
    )
    _overlay_dc_vs_dw!(
        fig,
        sim_dict,
        dc_key;
        component_idx=cidx,
        units_MLS=units_MLS,
        label="$(dc_key) vs dw",
    )
    return fig
end

"""
Plot conductivity_regular_xx_fixed_extrap versus omega for multiple simulations.

Creates one subplot per requested beta and overlays curves from all requested
(n, U, L, TCA_size) combinations in each subplot.

Omega axis for each curve is taken from that simulation's `w_conductivity_range`
(with fallback to legacy `w_cond_range` when present).

Usage:
plot_conductivity_regular_fixed_vs_omega(
    sim_info, n_list, U_list, geometry_pairs, beta_list;
    prefer_full=true, exact_beta=false
)
"""
function plot_conductivity_regular_fixed_vs_omega(
    sim_info::Dict{String, Any},
    n_list::AbstractVector,
    U_list::AbstractVector,
    geometry_pairs::AbstractVector,
    beta_list::AbstractVector;
    prefer_full::Bool=true,
    exact_beta::Bool=false,
    beta_match_tol::Float64=1e-8,
    show_missing::Bool=false,
    units_mult::Float64=1.0,
    units_label::String="",
    xscale=:identity,
    yscale=:identity,
    ncols::Int=0,
)
    isempty(beta_list) && error("beta_list must contain at least one beta value.")
    isempty(n_list) && error("n_list must contain at least one value.")
    isempty(U_list) && error("U_list must contain at least one value.")
    isempty(geometry_pairs) && error("geometry_pairs must contain at least one (L, TCA_size) pair.")

    # Cache sims_all per n so repeated lookups are cheap.
    sims_by_n = Dict{Float64, Any}()
    for n_raw in n_list
        n = Float64(n_raw)
        sims_by_n[n] = get_sims_all(sim_info; wanted_n=n)
    end

    beta_targets = Float64.(collect(beta_list))
    panels = Plots.Plot[]

    for beta_target in beta_targets
        T_target = 1.0 / beta_target
        subplot_fig = plot(
            xlabel="ω_cond",
            ylabel=isempty(units_label) ? "conductivity_regular_xx_fixed_extrap" : "conductivity_regular_xx_fixed_extrap ($(units_label))",
            title="conductivity_regular_xx_fixed_extrap vs ω at beta=$(round(beta_target, sigdigits=6))",
            legend=:topright,
            size=(920, 560),
            left_margin=8Plots.mm,
            bottom_margin=8Plots.mm,
            xscale=xscale,
            yscale=yscale,
        )

        plotted_curves = 0

        for n_raw in n_list
            n = Float64(n_raw)
            sims_all_n = sims_by_n[n]

            for U_raw in U_list
                U = Float64(U_raw)

                for geom in geometry_pairs
                    L = Int(geom[1])
                    TCA_size = Int(geom[2])

                    sim, method = get_sim_from_sims_all(sims_all_n, U, L, TCA_size; prefer_full=prefer_full)
                    if sim === nothing || isempty(sim)
                        show_missing && @warn "Simulation missing; skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    available_betas = sort(collect(keys(sim)))
                    isempty(available_betas) && continue

                    beta_use = _find_closest_beta_key(sim, beta_target)
                    beta_use === nothing && continue

                    if abs(Float64(beta_use) - beta_target) > beta_match_tol
                        if exact_beta
                            show_missing && @warn "Target beta not found exactly; skipping" target_beta=beta_target n=n U=U L=L TCA_size=TCA_size nearest_beta=Float64(beta_use)
                            continue
                        else
                            show_missing && @warn "Target beta not found exactly; using nearest available beta" target_beta=beta_target n=n U=U L=L TCA_size=TCA_size nearest_beta=Float64(beta_use)
                        end
                    end

                    sim_beta = sim[beta_use]
                    sim_dict = haskey(sim_beta, "sim_dict") ? sim_beta["sim_dict"] : sim_beta

                    if !haskey(sim_dict, "means")
                        show_missing && @warn "Missing means dictionary; skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end
                    means = sim_dict["means"]

                    if !haskey(means, "conductivity_regular_xx_fixed_extrap")
                        show_missing && @warn "Missing conductivity_regular_xx_fixed_extrap; skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    wkey = if haskey(sim_dict, "w_conductivity_range")
                        "w_conductivity_range"
                    elseif haskey(sim_dict, "w_cond_range")
                        show_missing && @warn "Using legacy omega key w_cond_range (prefer w_conductivity_range)" n=n U=U L=L TCA_size=TCA_size
                        "w_cond_range"
                    else
                        ""
                    end

                    if isempty(wkey)
                        show_missing && @warn "Missing omega range key (w_conductivity_range/w_cond_range); skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    wvals = Float64.(sim_dict[wkey])
                    yraw = means["conductivity_regular_xx_fixed_extrap"]["value"]
                    eraw = means["conductivity_regular_xx_fixed_extrap"]["error"]

                    yvec = yraw isa AbstractVector ? Float64.(yraw) : Float64[yraw]
                    evec = eraw isa AbstractVector ? Float64.(eraw) : Float64[eraw]

                    npts = min(length(wvals), length(yvec), length(evec))
                    if npts == 0
                        show_missing && @warn "No plottable points for conductivity_regular_xx_fixed_extrap" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    xs = wvals[1:npts]
                    ys = yvec[1:npts] .* units_mult
                    es = evec[1:npts] .* units_mult

                    keep = findall(i -> isfinite(xs[i]) && isfinite(ys[i]) && isfinite(es[i]) && es[i] >= 0.0, eachindex(xs))
                    isempty(keep) && continue
                    xs = xs[keep]
                    ys = ys[keep]
                    es = es[keep]

                    if yscale == :log10
                        keep_pos = findall(v -> v > 0.0, ys)
                        isempty(keep_pos) && continue
                        xs = xs[keep_pos]
                        ys = ys[keep_pos]
                        es = es[keep_pos]
                        for i in eachindex(ys)
                            max_err = 0.95 * ys[i]
                            if es[i] > max_err
                                es[i] = max_err
                            end
                        end
                    end

                    p = sortperm(xs)
                    lbl_method = method == "FULL" ? "FULL" : "TCA=$(TCA_size)"
                    plot!(
                        subplot_fig,
                        xs[p],
                        ys[p];
                        yerr=es[p],
                        m=:circle,
                        ms=3,
                        lw=1.6,
                        label="n=$(n), U=$(U), L=$(L), $(lbl_method), beta=$(round(Float64(beta_use), sigdigits=5))",
                    )
                    plotted_curves += 1
                end
            end
        end

        if plotted_curves == 0
            show_missing && @warn "No curves plotted for target beta" target_beta=beta_target
        end

        push!(panels, subplot_fig)
    end

    nplots = length(panels)
    nplots == 0 && error("No subplots were generated.")
    ncols_use = ncols > 0 ? ncols : min(3, nplots)
    nrows_use = ceil(Int, nplots / ncols_use)

    return plot(
        panels...,
        layout=(nrows_use, ncols_use),
        size=(980 * ncols_use, 560 * nrows_use),
        plot_title="conductivity_regular_xx_fixed_extrap vs ω_cond",
    )
end

"""
Plot Lxx_D_fixed_extrap versus omega for multiple simulations.

Lxx_D here is the real part of the current-current correlator contribution
obtained from fixed-delta transport data and extrapolated to delta->0.

For every plotted curve, this helper also overlays a point at ω=0 with value
-Kx from the same simulation/beta.

Usage:
plot_lxx_d_fixed_vs_omega(
    sim_info, n_list, U_list, geometry_pairs, beta_list;
    prefer_full=true, exact_beta=false
)
"""
function plot_lxx_d_fixed_vs_omega(
    sim_info::Dict{String, Any},
    n_list::AbstractVector,
    U_list::AbstractVector,
    geometry_pairs::AbstractVector,
    beta_list::AbstractVector;
    prefer_full::Bool=true,
    exact_beta::Bool=false,
    beta_match_tol::Float64=1e-8,
    show_missing::Bool=false,
    units_mult::Float64=1.0,
    units_label::String="",
    xscale=:identity,
    yscale=:identity,
    ncols::Int=0,
)
    isempty(beta_list) && error("beta_list must contain at least one beta value.")
    isempty(n_list) && error("n_list must contain at least one value.")
    isempty(U_list) && error("U_list must contain at least one value.")
    isempty(geometry_pairs) && error("geometry_pairs must contain at least one (L, TCA_size) pair.")

    sims_by_n = Dict{Float64, Any}()
    for n_raw in n_list
        n = Float64(n_raw)
        sims_by_n[n] = get_sims_all(sim_info; wanted_n=n)
    end

    beta_targets = Float64.(collect(beta_list))
    panels = Plots.Plot[]

    for beta_target in beta_targets
        subplot_fig = plot(
            xlabel="ω_cond",
            ylabel=isempty(units_label) ? "Lxx_D_fixed_extrap" : "Lxx_D_fixed_extrap ($(units_label))",
            title="Lxx_D_fixed_extrap vs ω at beta=$(round(beta_target, sigdigits=6))",
            legend=:topright,
            size=(920, 560),
            left_margin=8Plots.mm,
            bottom_margin=8Plots.mm,
            xscale=xscale,
            yscale=yscale,
        )

        plotted_curves = 0
        added_kx_legend = false

        for n_raw in n_list
            n = Float64(n_raw)
            sims_all_n = sims_by_n[n]

            for U_raw in U_list
                U = Float64(U_raw)

                for geom in geometry_pairs
                    L = Int(geom[1])
                    TCA_size = Int(geom[2])

                    sim, method = get_sim_from_sims_all(sims_all_n, U, L, TCA_size; prefer_full=prefer_full)
                    if sim === nothing || isempty(sim)
                        show_missing && @warn "Simulation missing; skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    available_betas = sort(collect(keys(sim)))
                    isempty(available_betas) && continue

                    beta_use = _find_closest_beta_key(sim, beta_target)
                    beta_use === nothing && continue

                    if abs(Float64(beta_use) - beta_target) > beta_match_tol
                        if exact_beta
                            show_missing && @warn "Target beta not found exactly; skipping" target_beta=beta_target n=n U=U L=L TCA_size=TCA_size nearest_beta=Float64(beta_use)
                            continue
                        else
                            show_missing && @warn "Target beta not found exactly; using nearest available beta" target_beta=beta_target n=n U=U L=L TCA_size=TCA_size nearest_beta=Float64(beta_use)
                        end
                    end

                    sim_beta = sim[beta_use]
                    sim_dict = haskey(sim_beta, "sim_dict") ? sim_beta["sim_dict"] : sim_beta

                    if !haskey(sim_dict, "means")
                        show_missing && @warn "Missing means dictionary; skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end
                    means = sim_dict["means"]

                    if !haskey(means, "Lxx_D_fixed_extrap")
                        show_missing && @warn "Missing Lxx_D_fixed_extrap; skipping (run updated sim_num control to populate this key)" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    if !haskey(means, "Kx")
                        show_missing && @warn "Missing Kx observable; skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    wkey = if haskey(sim_dict, "w_conductivity_range")
                        "w_conductivity_range"
                    elseif haskey(sim_dict, "w_cond_range")
                        show_missing && @warn "Using legacy omega key w_cond_range (prefer w_conductivity_range)" n=n U=U L=L TCA_size=TCA_size
                        "w_cond_range"
                    else
                        ""
                    end

                    if isempty(wkey)
                        show_missing && @warn "Missing omega range key (w_conductivity_range/w_cond_range); skipping" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    wvals = Float64.(sim_dict[wkey])
                    yraw = means["Lxx_D_fixed_extrap"]["value"]
                    eraw = means["Lxx_D_fixed_extrap"]["error"]

                    yvec = yraw isa AbstractVector ? Float64.(yraw) : Float64[yraw]
                    evec = eraw isa AbstractVector ? Float64.(eraw) : Float64[eraw]

                    npts = min(length(wvals), length(yvec), length(evec))
                    if npts == 0
                        show_missing && @warn "No plottable points for Lxx_D_fixed_extrap" n=n U=U L=L TCA_size=TCA_size
                        continue
                    end

                    xs = wvals[1:npts]
                    ys = yvec[1:npts] .* units_mult
                    es = evec[1:npts] .* units_mult

                    keep = findall(i -> isfinite(xs[i]) && isfinite(ys[i]) && isfinite(es[i]) && es[i] >= 0.0, eachindex(xs))
                    isempty(keep) && continue
                    xs = xs[keep]
                    ys = ys[keep]
                    es = es[keep]

                    if yscale == :log10
                        keep_pos = findall(v -> v > 0.0, ys)
                        isempty(keep_pos) && continue
                        xs = xs[keep_pos]
                        ys = ys[keep_pos]
                        es = es[keep_pos]
                        for i in eachindex(ys)
                            max_err = 0.95 * ys[i]
                            if es[i] > max_err
                                es[i] = max_err
                            end
                        end
                    end

                    p = sortperm(xs)
                    lbl_method = method == "FULL" ? "FULL" : "TCA=$(TCA_size)"
                    curve_label = "n=$(n), U=$(U), L=$(L), $(lbl_method), beta=$(round(Float64(beta_use), sigdigits=5))"

                    plot!(
                        subplot_fig,
                        xs[p],
                        ys[p];
                        yerr=es[p],
                        m=:circle,
                        ms=3,
                        lw=1.6,
                        label=curve_label,
                    )
                    plotted_curves += 1

                    kx_val_raw = means["Kx"]["value"]
                    kx_err_raw = get(means["Kx"], "error", NaN)
                    kx_val = -Float64(real(kx_val_raw)) * units_mult
                    kx_err = if kx_err_raw isa Number
                        abs(Float64(real(kx_err_raw))) * units_mult
                    else
                        NaN
                    end

                    if isfinite(kx_val)
                        if yscale == :log10 && kx_val <= 0.0
                            show_missing && @warn "Skipping -Kx point for log10 scale because value is non-positive" n=n U=U L=L TCA_size=TCA_size beta=Float64(beta_use) kx_value=kx_val
                        else
                            if isfinite(kx_err) && kx_err >= 0.0
                                if yscale == :log10
                                    max_err = 0.95 * kx_val
                                    if kx_err > max_err
                                        kx_err = max_err
                                    end
                                end
                                scatter!(
                                    subplot_fig,
                                    [0.0],
                                    [kx_val];
                                    yerr=[kx_err],
                                    m=:utriangle,
                                    ms=6,
                                    mc=:black,
                                    msc=:black,
                                    label=added_kx_legend ? false : "-Kx at ω=0",
                                )
                            else
                                scatter!(
                                    subplot_fig,
                                    [0.0],
                                    [kx_val];
                                    m=:utriangle,
                                    ms=6,
                                    mc=:black,
                                    msc=:black,
                                    label=added_kx_legend ? false : "-Kx at ω=0",
                                )
                            end
                            added_kx_legend = true
                        end
                    end
                end
            end
        end

        if plotted_curves == 0
            show_missing && @warn "No curves plotted for target beta" target_beta=beta_target
        end

        push!(panels, subplot_fig)
    end

    nplots = length(panels)
    nplots == 0 && error("No subplots were generated.")
    ncols_use = ncols > 0 ? ncols : min(3, nplots)
    nrows_use = ceil(Int, nplots / ncols_use)

    return plot(
        panels...,
        layout=(nrows_use, ncols_use),
        size=(980 * ncols_use, 560 * nrows_use),
        plot_title="Lxx_D_fixed_extrap vs ω_cond with -Kx(ω=0)",
    )
end

"""
Compatibility wrapper.
"""
function plot_Lxx_D_vs_omega(args...; kwargs...)
    return plot_lxx_d_fixed_vs_omega(args...; kwargs...)
end

"""
Plot transport delta-extrapolation diagnostics for one geometry across multiple betas.

Selectable observable families:
- :conductivity -> conductivity_regular_xx_fixed + conductivity_dc_xx overlay
- :drude       -> Drude_weight_fixed + drude_dc_xx overlay
- :resistivity -> resistivity_tensor_fixed + resistivity_dc_tensor overlay
- :all         -> all three above side-by-side

Usage:
plot_transport_delta_extrapolation(sim_info, wanted_n, U, L, TCA_size, beta_list; observable=:conductivity)
"""
function plot_transport_delta_extrapolation(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U,
    L::Int,
    TCA_size::Int,
    beta_list::AbstractVector;
    prefer_full::Bool=true,
    conductivity_component_idx::Int=1,
    drude_component_idx::Int=1,
    resistivity_component_idx::Int=1,
    units_MLS::Bool=true,
    observable::Symbol=:all,
)
    sims_all_local = get_sims_all(sim_info; wanted_n=Float64(wanted_n))
    sim, method = get_sim_from_sims_all(sims_all_local, U, L, TCA_size; prefer_full=prefer_full)
    sim === nothing && error("Simulation not found for U=$(U), n~$(wanted_n), L=$(L), TCA=$(TCA_size).")

    sys_label = method == "FULL" ? "L=$(L), FULL" : "L=$(L), TCA=$(TCA_size)"
    isempty(beta_list) && error("beta_list must contain at least one beta value.")

    beta_rows = Plots.Plot[]
    for beta_raw in beta_list
        beta = Float64(beta_raw)
        if !haskey(sim, beta)
            @warn "Beta not found for selected simulation; skipping" beta=beta U=U wanted_n=wanted_n L=L TCA_size=TCA_size
            continue
        end

        sim_beta = sim[beta]
        sim_dict = haskey(sim_beta, "sim_dict") ? sim_beta["sim_dict"] : sim_beta
        haskey(sim_dict, "means") || begin
            @warn "Missing means in selected beta; skipping" beta=beta
            continue
        end

        if observable == :all
            fig_cond = _plot_single_transport_delta_panel(
                sim_dict,
                sys_label,
                beta;
                observable=:conductivity,
                conductivity_component_idx=conductivity_component_idx,
                drude_component_idx=drude_component_idx,
                resistivity_component_idx=resistivity_component_idx,
                units_MLS=units_MLS,
            )
            fig_drude = _plot_single_transport_delta_panel(
                sim_dict,
                sys_label,
                beta;
                observable=:drude,
                conductivity_component_idx=conductivity_component_idx,
                drude_component_idx=drude_component_idx,
                resistivity_component_idx=resistivity_component_idx,
                units_MLS=units_MLS,
            )
            fig_res = _plot_single_transport_delta_panel(
                sim_dict,
                sys_label,
                beta;
                observable=:resistivity,
                conductivity_component_idx=conductivity_component_idx,
                drude_component_idx=drude_component_idx,
                resistivity_component_idx=resistivity_component_idx,
                units_MLS=units_MLS,
            )

            row_fig = plot(
                fig_cond,
                fig_drude,
                fig_res,
                layout=(1, 3),
                size=(400, 520),
                plot_title="Transport delta extrapolation, beta=$(beta)",
            )
        else
            single_fig = _plot_single_transport_delta_panel(
                sim_dict,
                sys_label,
                beta;
                observable=observable,
                conductivity_component_idx=conductivity_component_idx,
                drude_component_idx=drude_component_idx,
                resistivity_component_idx=resistivity_component_idx,
                units_MLS=units_MLS,
            )

            row_fig = plot(
                single_fig,
                layout=(1, 1),
                size=(700, 520),
                plot_title="$(observable) delta extrapolation, beta=$(beta)",
            )
        end

        push!(beta_rows, row_fig)
    end

    isempty(beta_rows) && error("No beta values were plotted. Check beta_list for available values in the selected simulation.")

    final_fig = plot(
        beta_rows...,
        layout=(length(beta_rows), 1),
        size=(2000, 560 * length(beta_rows)),
        plot_title="Transport delta extrapolation with DC overlays: $(sys_label), U=$(U), n~$(wanted_n)",
    )

    return final_fig
end


"""
    plot_tc_vs_inv_L(sim_info, wanted_n, U; ...)

Plot critical temperature Tc vs 1/L for a given (n, U) to estimate Tc(∞) via finite-size scaling.

For each geometry L, computes Tc from stiffness crossing using `find_tc` (or `find_tc_fit`).
When `tc_method=:fit`, per-point uncertainties from the polynomial fit are shown as yerror
bars, and the finite-size scaling fit uses weighted least squares — so `Tc_inf_err` is
returned alongside `Tc_inf`.

# Keywords
- `geometry_pairs`: if nothing, use all available geometries.
- `tc_method::Symbol=:crossing`: `:crossing` (default, uses `find_tc`) or `:fit` (uses `find_tc_fit`, yields Tc errorbars).
- `lowest_q::Bool=false`: use lowest-q stiffness instead of default.
- `fit_scaling::Union{Bool,Symbol}=false`: `false` no fit; `true`/`:per_tca` fit each TCA series;
  `:combined` pool all points into one fit; `:both` draw per-TCA and combined fits.
- `x_axis::Symbol=:inv_L`: `:inv_L` plots vs `1/L` (linear FSS); `:inv_logL` plots vs `1/ln(L)`
  (BKT-style scaling). Fit is linear in the chosen x and extrapolates to x=0 (L→∞).

# Returns
`(fig, fit_result)` where `fit_result` is a `Dict` keyed by `tca_size` (and/or `:combined`),
each entry a NamedTuple with `Tc_inf`, `Tc_inf_err`, `a_coeff`, `a_err`, `Ls`, `Tcs`, `Tc_errs`
(or `nothing` if no fit).
"""
function plot_tc_vs_inv_L(
    sim_info::Dict{String, Any},
    wanted_n::Real,
    U;
    geometry_pairs=nothing,
    tc_method::Symbol=:crossing,
    lowest_q::Bool=false,
    fit_scaling::Union{Bool,Symbol}=false,
    x_axis::Symbol=:inv_L,
)
    tc_method in (:crossing, :fit) || error("tc_method must be :crossing or :fit (got $(tc_method)).")
    x_axis in (:inv_L, :inv_logL) || error("x_axis must be :inv_L or :inv_logL (got $(x_axis)).")
    x_of_L   = x_axis === :inv_logL ? (L -> 1.0 / log(L))   : (L -> 1.0 / L)
    x_label  = x_axis === :inv_logL ? "1/ln(L)"             : "1/L"
    L_valid  = x_axis === :inv_logL ? (L -> L > 1)          : (L -> L > 0)

    # Normalize fit_scaling into two flags.
    fit_mode = fit_scaling === true ? :per_tca : fit_scaling
    fit_mode in (false, :per_tca, :combined, :both) ||
        error("fit_scaling must be one of false, true, :per_tca, :combined, :both (got $(fit_scaling)).")
    do_per_tca  = fit_mode === :per_tca  || fit_mode === :both
    do_combined = fit_mode === :combined || fit_mode === :both

    sims_all_local = get_sims_all(sim_info; wanted_n=Float64(wanted_n))
    geoms = geometry_pairs === nothing ? get_available_geometry_pairs(sims_all_local, U) : collect(geometry_pairs)
    isempty(geoms) && error("No geometries found for U=$(U), n~$(wanted_n).")

    extract_tc = if tc_method === :fit
        sim -> (r = find_tc_fit(sim; lowest_q=lowest_q); (r[1], r[2]))
    else
        sim -> (find_tc(sim; lowest_q=lowest_q)[1], NaN)
    end

    # Group results by TCA size so each TCA becomes its own series.
    by_tca = Dict{Any, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()

    for (L, tca_size) in geoms
        sim, _method = get_sim_from_sims_all(sims_all_local, U, L, tca_size)
        (sim === nothing || isempty(sim)) && continue
        try
            tc, tc_err = extract_tc(sim)
            if isfinite(tc)
                Ls_v, Tcs_v, Errs_v = get!(by_tca, tca_size, (Float64[], Float64[], Float64[]))
                push!(Ls_v, Float64(L))
                push!(Tcs_v, tc)
                push!(Errs_v, tc_err)
            end
        catch err
            @warn "plot_tc_vs_inv_L: failed to extract Tc for L=$L, TCA=$tca_size" exception=err
        end
    end

    isempty(by_tca) && error("No valid Tc values found for U=$(U), n~$(wanted_n).")

    tca_keys = sort!(collect(keys(by_tca)))
    max_x = maximum(x_of_L(minimum(ls)) for (ls, _, _) in values(by_tca))

    fig = plot(
        xlabel=x_label, ylabel="Tc",
        title="Critical temperature vs $x_label (n=$(wanted_n), U=$(U))",
        legend=:topright, size=(800, 520),
        left_margin=6Plots.mm, bottom_margin=6Plots.mm,
    )

    # Helper: linear fit Tc = Tc_inf + a*x (x = 1/L or 1/ln(L)), weighted if errors are fully available.
    function _linfit(xs_v::Vector{Float64}, Tcs_v::Vector{Float64}, Errs_v::Vector{Float64})
        n = length(xs_v)
        X = [ones(n) xs_v]
        use_weights = all(e -> isfinite(e) && e > 0, Errs_v)
        if use_weights
            sw = 1.0 ./ Errs_v
            Xw = X .* sw
            yw = Tcs_v .* sw
            params = Xw \ yw
            covm = inv(Xw' * Xw)
        else
            params = X \ Tcs_v
            resid = Tcs_v .- X * params
            dof = max(n - 2, 1)
            sigma2 = (resid' * resid) / dof
            covm = sigma2 .* inv(X' * X)
        end
        Tc_inf, a_coeff = params[1], params[2]
        Tc_inf_err = (isfinite(covm[1,1]) && covm[1,1] > 0) ? sqrt(covm[1,1]) : NaN
        a_err      = (isfinite(covm[2,2]) && covm[2,2] > 0) ? sqrt(covm[2,2]) : NaN
        return Tc_inf, Tc_inf_err, a_coeff, a_err
    end

    # Helper: draw a fit line + extrapolated-Tc∞ star given fit parameters.
    function _draw_fit!(fig, Tc_inf, Tc_inf_err, a_coeff, color, label)
        x_fit = range(0, max_x * 1.2, length=100)
        Tc_fit = Tc_inf .+ a_coeff .* x_fit
        plot!(fig, x_fit, Tc_fit, linestyle=:dash, lw=2, color=color, label=label)
        if isfinite(Tc_inf_err)
            scatter!(fig, [0.0], [Tc_inf], marker=:star5, markersize=7,
                color=color, yerror=[Tc_inf_err], label="")
        else
            scatter!(fig, [0.0], [Tc_inf], marker=:star5, markersize=7,
                color=color, label="")
        end
    end

    fit_result = Dict{Any, NamedTuple}()

    # Per-TCA scatter (and optional per-TCA fit).
    for (i, tca_size) in enumerate(tca_keys)
        Ls_v, Tcs_v, Errs_v = by_tca[tca_size]
        # Drop Ls not representable on the chosen x axis (e.g. L<=1 for :inv_logL).
        keep = L_valid.(Ls_v)
        if !all(keep)
            dropped = Ls_v[.!keep]
            @warn "plot_tc_vs_inv_L: skipping L=$(dropped) for TCA=$tca_size (invalid on x_axis=$x_axis)"
            Ls_v = Ls_v[keep]; Tcs_v = Tcs_v[keep]; Errs_v = Errs_v[keep]
        end
        isempty(Ls_v) && continue
        order = sortperm(Ls_v)
        Ls_v = Ls_v[order]
        Tcs_v = Tcs_v[order]
        Errs_v = Errs_v[order]
        xs_v = x_of_L.(Ls_v)

        has_errors = any(e -> isfinite(e) && e > 0, Errs_v)
        if has_errors
            yerr = [isfinite(e) && e > 0 ? e : 0.0 for e in Errs_v]
            scatter!(fig, xs_v, Tcs_v,
                marker=:circle, color=i, yerror=yerr, label="TCA=$tca_size")
        else
            scatter!(fig, xs_v, Tcs_v,
                marker=:circle, color=i, label="TCA=$tca_size")
        end

        if do_per_tca && length(Ls_v) >= 2
            Tc_inf, Tc_inf_err, a_coeff, a_err = _linfit(xs_v, Tcs_v, Errs_v)
            err_str = isfinite(Tc_inf_err) ? "±$(round(Tc_inf_err, digits=4))" : ""
            _draw_fit!(fig, Tc_inf, Tc_inf_err, a_coeff, i,
                "Fit TCA=$tca_size: Tc∞=$(round(Tc_inf, digits=4))$err_str")
            fit_result[tca_size] = (; Tc_inf=Tc_inf, Tc_inf_err=Tc_inf_err,
                                      a_coeff=a_coeff, a_err=a_err,
                                      Ls=Ls_v, Tcs=Tcs_v, Tc_errs=Errs_v)
        end
    end

    # Combined fit across all (L, Tc) points from all TCAs.
    if do_combined
        Ls_all = Float64[]; Tcs_all = Float64[]; Errs_all = Float64[]
        for tca_size in tca_keys
            Ls_v, Tcs_v, Errs_v = by_tca[tca_size]
            keep = L_valid.(Ls_v)
            append!(Ls_all, Ls_v[keep])
            append!(Tcs_all, Tcs_v[keep])
            append!(Errs_all, Errs_v[keep])
        end
        if length(Ls_all) >= 2
            xs_all = x_of_L.(Ls_all)
            Tc_inf, Tc_inf_err, a_coeff, a_err = _linfit(xs_all, Tcs_all, Errs_all)
            err_str = isfinite(Tc_inf_err) ? "±$(round(Tc_inf_err, digits=4))" : ""
            combined_color = :black
            _draw_fit!(fig, Tc_inf, Tc_inf_err, a_coeff, combined_color,
                "Combined fit: Tc∞=$(round(Tc_inf, digits=4))$err_str")
            fit_result[:combined] = (; Tc_inf=Tc_inf, Tc_inf_err=Tc_inf_err,
                                       a_coeff=a_coeff, a_err=a_err,
                                       Ls=Ls_all, Tcs=Tcs_all, Tc_errs=Errs_all)
        else
            @warn "plot_tc_vs_inv_L: combined fit needs >=2 points, got $(length(Ls_all))."
        end
    end

    return fig, (isempty(fit_result) ? nothing : fit_result)
end
