"""
    Plotting helper functions for observable comparison notebook.
    Wraps repeated plotting patterns into reusable functions.
"""

"""
    plot_finite_size_scaling(all_series, obs, beta_target; w_ix=1, include_full_reference=true, title="", ylabel="")

Create a finite-size scaling plot (observable vs 1/N) at fixed β.

Arguments:
- `all_series`: Vector of (label, sims_beta_dict) tuples
- `obs`: Observable name (String)
- `beta_target`: Target β value (Float64)
- `w_ix`: Component index for vector observables (default: 1)
- `include_full_reference`: Include FULL simulations (default: true)
- `title`: Custom title (default: auto-generated)
- `ylabel`: Custom y-axis label (default: obs name)

Returns:
- Plots.Figure with finite-size scaling plot grouped by TCA cluster size
"""
function plot_finite_size_scaling(all_series, obs, beta_target; 
                                   w_ix=1, include_full_reference=true, 
                                   title="", ylabel="",
                                   units_mult::Union{Nothing, Float64}=nothing, units_label="")
    # Collect points as: (invL, value, err, L, beta_used)
    tca_points  = Dict{Int, Vector{NTuple{5, Float64}}}()
    full_points = NTuple{5, Float64}[]
    min_N = Inf

    for (lbl, sim) in all_series
        isempty(sim) && continue

        betas = sort(collect(keys(sim)))
        isempty(betas) && continue
        β_use = (beta_target in betas) ? beta_target : continue
        
        sim_beta = sim[β_use]
        lx, ly = sim_beta["sim_dict"]["Lx"], sim_beta["sim_dict"]["Ly"]
        min_N = min(min_N, lx * ly)
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
            σ  = Float64(v[w_ix])
            dσ = Float64(e[w_ix])
        else
            w_ix == 1 || continue
            σ  = Float64(v)
            dσ = Float64(e)
        end

        pt = (1.0 / (lx * ly), σ, dσ, Float64(lx), Float64(β_use))

        if occursin("TCA=", lbl)
            mT = match(r"TCA=(\d+)", lbl)
            mT === nothing && continue
            tca = parse(Int, mT.captures[1])
            push!(get!(tca_points, tca, NTuple{5, Float64}[]), pt)
        elseif include_full_reference && occursin("FULL", lbl)
            push!(full_points, pt)
        end
    end

    # Create plot
    ylabel_use = isempty(ylabel) ? obs : ylabel
    if !isempty(units_label)
        ylabel_use *= " ($units_label)"
    elseif !isnothing(units_mult)
        ylabel_use *= " (×$(units_mult))"
    end
    title_use = isempty(title) ? "Finite-size scaling at β=$(beta_target), ω-index=$w_ix" : title
    
    fig = plot(
        xlabel = "1/N",
        ylabel = ylabel_use,
        title  = title_use,
        legend = :topleft,
        size   = (900, 520),
        left_margin = 6Plots.mm,
        bottom_margin = 6Plots.mm
    )

    for tca in sort(collect(keys(tca_points)))
        pts = sort(tca_points[tca], by = x -> x[1])
        xs  = [p[1] for p in pts]
        ys  = [p[2] for p in pts]
        es  = [p[3] for p in pts]
        βs  = unique(round.(Int, [p[5] * 1000 for p in pts])) ./ 1000
        plot!(fig, xs, ys; yerr=es, m=:circle, lw=2, ms=6, label="TCA=$tca (β=$(join(βs, ",")))")
    end

    if include_full_reference && !isempty(full_points)
        pts = sort(full_points, by = x -> x[1])
        xs  = [p[1] for p in pts]
        ys  = [p[2] for p in pts]
        es  = [p[3] for p in pts]
        βs  = unique(round.(Int, [p[5] * 1000 for p in pts])) ./ 1000
        plot!(fig, xs, ys; yerr=es, m=:diamond, lw=2, ls=:dash, ms=6, c=:black, label="FULL (β=$(join(βs, ",")))")
    end
    
    min_N != Inf && plot!(fig, xlims=(0.0, 1.5/min_N))
    return fig
end


"""
    plot_observable_vs_T(all_series, obs_name, obs_index=1; title="", ylabel="", xlims=nothing, ylims=nothing)

Create a simple observable vs T plot for all series.

Arguments:
- `all_series`: Vector of (label, sims_beta_dict) tuples
- `obs_name`: Observable name (String)
- `obs_index`: Component index for vector observables (default: 1)
- `title`: Custom title
- `ylabel`: Custom y-axis label (default: obs_name)
- `xlims`: x-axis limits tuple (default: nothing)
- `ylims`: y-axis limits tuple (default: nothing)

Returns:
- Plots.Figure with observable vs temperature plot
"""
function plot_observable_vs_T(all_series, obs_name, obs_index=1; 
                               title="", ylabel="", xlims=nothing, ylims=nothing)
    ylabel_use = isempty(ylabel) ? obs_name : ylabel
    title_use = isempty(title) ? "$obs_name vs T" : title
    
    fig = plot(xlabel="T", ylabel=ylabel_use, title=title_use,
        legend=:topright, size=(950, 550), left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    for (lbl, sim) in all_series
        ts, obs, errs = get_obs_vs_T(sim, obs_name, obs_index)
        isempty(ts) && continue
        plot!(fig, ts, obs; yerr=errs, label=lbl, m=:circle, lw=2, ms=5)
    end

    !isnothing(xlims) && plot!(fig, xlims=xlims)
    !isnothing(ylims) && plot!(fig, ylims=ylims)

    return fig
end


"""
    plot_mu_vs_T(all_series; title="", ylabel="μ", xlims=nothing, ylims=nothing)
    plot_mu_vs_T(sims::Dict{Any, Any}; label="", kwargs...)

Plot the chemical potential μ used in each simulation as a function of T.

μ is read from `sim["sim_dict"]["mu"]` (with a fallback to `sim["mu"]` for the
beta-grouped dicts produced by `group_by_beta`). Series with no μ recorded are
silently skipped.

Arguments:
- `all_series`: Vector of `(label, sims_beta_dict)` tuples (same shape as for
  `plot_observable_vs_T`). The single-set form wraps a lone `sims` dict.
- `title`, `ylabel`, `xlims`, `ylims`: standard plot overrides.

Returns:
- `Plots.Figure` with μ vs T.
"""
function plot_mu_vs_T(all_series; title="", ylabel="μ", xlims=nothing, ylims=nothing)
    title_use = isempty(title) ? "μ vs T" : title

    fig = plot(xlabel="T", ylabel=ylabel, title=title_use,
        legend=:topright, size=(950, 550), left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    for (lbl, sim) in all_series
        ts, mus = get_mu_vs_T(sim)
        isempty(ts) && continue
        plot!(fig, ts, mus; label=lbl, m=:circle, lw=2, ms=5)
    end

    !isnothing(xlims) && plot!(fig, xlims=xlims)
    !isnothing(ylims) && plot!(fig, ylims=ylims)

    return fig
end

function plot_mu_vs_T(sims::Dict{Any, Any}; label="μ", kwargs...)
    return plot_mu_vs_T([(label, sims)]; kwargs...)
end


"""
    plot_resistivity_tensor_overlay(all_series; U="", n="")

Create a single-panel overlay plot of resistivity tensor components (ρ_xx, ρ_xy) for all series.

Arguments:
- `all_series`: Vector of (label, sims_beta_dict) tuples
- `U`: Parameter U value for title (default: "")
- `n`: Filling factor for title (default: "")

Returns:
- Plots.Figure with all series overlaid
"""
function plot_resistivity_tensor_overlay(all_series; U="", n="")
    fig = plot(
        xlabel = "T",
        ylabel = "ρ / 2π",
        title  = "Resistivity tensor, U=$(U), n≈$(n)",
        legend = :topright,
        size   = (950, 550),
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm
    )

    for (lbl, sim) in all_series
        Ts, vals, errs = get_obs_vs_T(sim, "resistivity_tensor_fixed_extrap")
        isempty(Ts) && continue

        # vals/errs contain 4 components: xx, xy, yx, yy
        ρ = hcat(vals...)./(2*pi)
        δρ = hcat(errs...)./(2*pi)

        plot!(fig, Ts, ρ[1, :]; yerr=δρ[1, :], label="$(lbl) ρ_xx", m=:circle, lw=2, ms=5)
    end

    plot!(fig, xlims=(0.0, 1.0), ylims=(0.2, 1.0))
    return fig
end


"""
    plot_resistivity_tensor_subplots(all_series; U="", n="")

Create a multi-panel plot of resistivity tensor components (ρ_xx, ρ_xy) for each series.

Arguments:
- `all_series`: Vector of (label, sims_beta_dict) tuples
- `U`: Parameter U value for title (default: "")
- `n`: Filling factor for title (default: "")

Returns:
- Plots.Figure with subplots (one per series)
"""
function plot_resistivity_tensor_subplots(all_series; U="", n="")
    N_series = length(all_series)
    ncols = min(N_series, 3)
    nrows = ceil(Int, N_series / ncols)

    fig = plot(layout=(nrows, ncols), size=(450*ncols, 400*nrows),
        left_margin=15Plots.mm, bottom_margin=8Plots.mm)

    for (i, (lbl, sim)) in enumerate(all_series)
        obs = get_obs_vs_T(sim, "resistivity_tensor_fixed_extrap")
        if isempty(obs[1]); continue; end

        Ts = obs[1]
        res_vals = [hcat(obs[2]...)[k, :] for k in 1:4]   # xx, xy, yx, yy
        res_errs = [hcat(obs[3]...)[k, :] for k in 1:4]

        plot!(fig, Ts, res_vals[1] ./ (2π); yerr=res_errs[1] ./ (2π),
            label="ρ_xx", m=:circle, lw=2, ms=5, subplot=i)
        plot!(fig, Ts, res_vals[2] ./ (2π); yerr=res_errs[2] ./ (2π),
            label="ρ_xy", m=:square, lw=2, ms=5, subplot=i)

        title!(fig, lbl, subplot=i)
        xlabel!(fig, "T", subplot=i)
        ylabel!(fig, "ρ / 2π", subplot=i)
    end
    
    plot!(fig, xlims=(0.0, 2.0), ylims=(-0.5, 5.0))
    plot!(fig, plot_title="Resistivity tensor, U=$(U), n≈$(n)")
    
    return fig
end


"""
    plot_dos_heatmap(all_series; T_max=1.0, U="", n="")

Create multi-panel DOS heatmaps (T, ω) for each series.

Arguments:
- `all_series`: Vector of (label, sims_beta_dict) tuples
- `T_max`: Maximum temperature to display (default: 1.0)
- `U`: Parameter U value for title (default: "")
- `n`: Filling factor for title (default: "")

Returns:
- Plots.Figure with DOS heatmaps in subplots
"""
function plot_dos_heatmap(all_series; T_max=1.0, U="", n="")
    N_series = length(all_series)
    ncols = min(N_series, 3)
    nrows = ceil(Int, N_series / ncols)

    fig = plot(layout=(nrows, ncols), size=(500*ncols, 400*nrows),
        left_margin=15Plots.mm, bottom_margin=10Plots.mm)

    for (sp_ix, (lbl, sim)) in enumerate(all_series)
        betas = sort(collect(keys(sim)))
        if isempty(betas); continue; end

        obs = get_obs_vs_T(sim, "DOS")
        Ts = reverse(obs[1])
        if isempty(Ts); continue; end

        ω = sim[betas[1]]["sim_dict"]["w_list"]
        nω = length(ω)
        nT = length(Ts)

        DOS_mat = fill(NaN, nω, nT)
        for (j, _) in enumerate(Ts)
            if Ts[nT-j+1] <= T_max && !isempty(obs[2][j])
                DOS_mat[:, nT-j+1] = obs[2][j]
            end
        end

        heatmap!(fig, Ts, ω, DOS_mat;
            xlabel="T", ylabel="ω", title=lbl,
            colorbar_title="DOS", subplot=sp_ix, xlims=(0.0, T_max))
    end

    plot!(fig, plot_title="DOS(T, ω), U=$(U), n≈$(n)")
    return fig
end
