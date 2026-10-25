# data_summery.jl — load old single-file means (sim_info.jld2, V6 format) and plot
# observables against T.
#
#   sim_info = load_and_prepare_data([...paths...])     # merge several sim_info files
#   sims_all = get_sims_all(sim_info; wanted_n=0.875)   # [U][cluster size][L] => curve
#   plot_obs_vs_T_from_sims_all(sims_all, U, L, Lc, "stiffness")
# A PlotState (create_plot_state / render_plot) caches get_sims_all per filling for
# interactive plotting. For V7 data use means_db.jl (get_curve) instead.

include("MC_Hubburd.jl")
include("sim_num_control.jl")
include("simulation_obs_inspection.jl")
include("plotting_helpers.jl")

"""
    _load_jld2_entries(path) -> Dict{String,Any}

Read every top-level entry of one JLD2 file. Tries the fast default mmap I/O first and
falls back to (slow) IOStream only if mmap throws ReadOnlyMemoryError, a JLD2 mmap bug
that hits some files (nested SerializedDict reconstruction); mmap is ~14× faster.
"""
function _load_jld2_entries(path::String)
    function read_with(io)
        d = Dict{String, Any}()
        if io === nothing
            JLD2.jldopen(path, "r") do f
                for k in keys(f); d[string(k)] = f[k]; end
            end
        else
            JLD2.jldopen(path, "r"; iotype=io) do f
                for k in keys(f); d[string(k)] = f[k]; end
            end
        end
        return d
    end
    try
        return read_with(nothing)          # fast path: default mmap
    catch e
        e isa ReadOnlyMemoryError || rethrow(e)
        @warn "JLD2 mmap hit ReadOnlyMemoryError; using IOStream fallback (slow) for: $path"
        return read_with(IOStream)
    end
end

"""
    load_results_paths(file_path_list) -> Dict{String,Any}

Load several sim_info.jld2 files and merge their entries into one Dict. A key that
already exists gets a `_1`, `_2`, … suffix instead of overwriting.
"""
function load_results_paths(file_path_list::Vector{String})
    # Merge into one dict (adding suffix to avoid key collisions)
    sim_info_all = Dict{String, Any}()
    for path in file_path_list
        println("Loading simulation info from: $path")
        entries = _load_jld2_entries(path)
        num_added = 0
        for base_key in keys(entries)
            key = base_key
            n = 1
            while haskey(sim_info_all, key)
                key = "$(base_key)_$(n)"
                n += 1
            end
            sim_info_all[key] = entries[base_key]
            num_added += 1
        end
        println("Added $num_added simulations from this file.")
    end
    println("Total simulations loaded: $(length(sim_info_all))")
    return sim_info_all
end

"""
The three V6 clean-system `sim_info` files `load_and_prepare_data` reads by default,
in the folder of env `V6_CLUSTER_RESULTS` (the same setting as means_store.jl).
Empty if that is not set; then pass `file_paths` explicitly.
"""
const DEFAULT_SIM_INFO_PATHS = let d = get(ENV, "V6_CLUSTER_RESULTS", "")
    isempty(d) ? String[] :
        [joinpath(d, f) for f in ("sim_info_tca_12-03.jld2", "sim_info_fullsys.jld2", "sim_info_tca.jld2")]
end

"""
    load_and_prepare_data(; file_paths=DEFAULT_SIM_INFO_PATHS, run_extrapolation=true,
        extrap_verbose=false, min_delta_MLS=1.0, max_delta_MLS=8.0,
        min_points=4, force_full_range=true)

Load simulation dictionaries and optionally run delta extrapolation preprocessing.
Returns the merged `sim_info` dictionary to be used by plotting helpers.
"""
function load_and_prepare_data(
    ;
    file_paths::Vector{String}=DEFAULT_SIM_INFO_PATHS,
    run_extrapolation::Bool=true,
    extrap_verbose::Bool=false,
    min_delta_MLS::Float64=1.0,
    max_delta_MLS::Float64=8.0,
    min_points::Int=4,
    force_full_range::Bool=true,
)
    isempty(file_paths) && error("load_and_prepare_data: no files. Pass file_paths=[...] or set " *
                                 "ENV[\"V6_CLUSTER_RESULTS\"] before including data_summery.jl")
    sim_info = load_results_paths(file_paths)

    if run_extrapolation
        extrapolate_all_delta_observables!(
            sim_info;
            verbose=extrap_verbose,
            min_delta_MLS=min_delta_MLS,
            max_delta_MLS=max_delta_MLS,
            min_points=min_points,
            force_full_range=force_full_range,
        )
    end

    return sim_info
end

mutable struct PlotState
    sim_info::Dict{String, Any}
    sims_by_n_cache::Dict{Float64, Any}
    U_list::Vector{Float64}
    n_list::Vector{Float64}
    geometry_pairs::Vector{Tuple{Int, Int}}
    obs_name::String
    obs_ix::Union{Nothing, Int}
    units_mult::Float64
    ylabel::String
    xlims
    ylims
    subplot_xlims
    subplot_ylims
    plot_against_beta::Bool
    prefer_full::Bool
    show_missing::Bool
    legend_pos
    layout_mode::Symbol
    separate_by::Symbol
end

"""
    create_plot_state(sim_info; ...)

Create a GUI-friendly state object that stores current plotting selections
and a cache of `sims_all` grouped by density `n`.
"""
function create_plot_state(
    sim_info::Dict{String, Any};
    U_list=[2.0, 3.0, 5.0],
    n_list=[0.875],
    geometry_pairs=[(8, 8), (10, 10), (12, 12)],
    obs_name::String="Ps",
    obs_ix=nothing,
    units_mult::Float64=1.0,
    ylabel::String=obs_name,
    xlims=nothing,
    ylims=nothing,
    subplot_xlims=nothing,
    subplot_ylims=nothing,
    plot_against_beta::Bool=false,
    prefer_full::Bool=true,
    show_missing::Bool=false,
    legend_pos=:topright,
    layout_mode::Symbol=:separate,
    separate_by::Symbol=:n,
)
    return PlotState(
        sim_info,
        Dict{Float64, Any}(),
        Float64.(collect(U_list)),
        Float64.(collect(n_list)),
        Tuple{Int, Int}.(collect(geometry_pairs)),
        obs_name,
        obs_ix,
        units_mult,
        ylabel,
        xlims,
        ylims,
        subplot_xlims,
        subplot_ylims,
        plot_against_beta,
        prefer_full,
        show_missing,
        legend_pos,
        layout_mode,
        separate_by,
    )
end

"""
    get_or_build_sims_for_n!(state, wanted_n)

Return cached `sims_all` for a given density `wanted_n`; build and cache on miss.
"""
function get_or_build_sims_for_n!(state::PlotState, wanted_n::Float64)
    if !haskey(state.sims_by_n_cache, wanted_n)
        state.sims_by_n_cache[wanted_n] = get_sims_all(state.sim_info; wanted_n=wanted_n)
    end
    return state.sims_by_n_cache[wanted_n]
end

"""
    warmup_sims_cache!(state::PlotState) -> state

Build (and cache) `get_sims_all` for every filling in `state.n_list`.
"""
function warmup_sims_cache!(state::PlotState)
    for n in state.n_list
        get_or_build_sims_for_n!(state, n)
    end
    return state
end

"""
    get_sims_all(sim_info; wanted_n=0.875) -> Dict

Curves of the uniform (clean) K0 and K0-TCA runs at filling `wanted_n`, merged into one
Dict `[U][TClustersize][L] => (β => point)`. For K0 runs the cluster-size key is L itself.
"""
function get_sims_all(sim_info::Dict{String, Any}; wanted_n::Float64=0.875)
    #load both K0 and TCA sims, with the same n, and only uniform disorder
    sims_K0 = get_sims(sim_info, wanted_n, "U";
        diag_method="K0", disorder_type="uniform", max_n_difference=0.05)
    sims_TCA = get_sims(sim_info, wanted_n, ["U", "TClustersize"];
        diag_method="K0-TCA", disorder_type="uniform", max_n_difference=0.05)
    #merge the two dicts, so it is one dict with keys [U][TClustersize][L].
    # for K0 TClustersize=L, and for K0-TCA TClustersize is the actual TCA cluster size.
    sims = sims_TCA
    Us = keys(sims)
    for (U, sims_U) in sims_K0
        U in Us ? nothing : sims[U] = Dict{Any, Any}()
        tcas = keys(sims[U])
        for (L, sim) in sims_U
            L in tcas ? sims[U][L][L]=sim : sims[U][L] = Dict{Any, Any}(L=>sim)
        end
    end
    return sims
end

"""
    print_avalable_sims(sims_all)

Print a table of the available (U, L, cluster sizes) in a `get_sims_all` result.
"""
function print_avalable_sims(sims_all::Dict{Any, Any})
    Us = sort(collect(keys(sims_all)))
    println("Available simulations:")
    println("U\tL\tTCA")
    println("-" ^ 30)
    for U in Us
        sims_U = sims_all[U]
        L_TCA = Dict{Any, Any}()
        for (TClustersize, sims_T) in sims_U
            for (L, sim) in sims_T
                if !(L in keys(L_TCA))
                    L_TCA[L] = [TClustersize]
                else
                    append!(L_TCA[L], [TClustersize])
                end
            end
        end
        Ls_sort = sort(collect(keys(L_TCA)))
        for L in Ls_sort
            TClustersizes = sort(L_TCA[L])
            TClustersizes_str = join(TClustersizes, ", ")
            println("$U\t$L\t$TClustersizes_str")
        end
    end
end

"""
    get_sim_from_sims_all(sims_all, U, L, TCA_size; prefer_full=true) -> (curve, method)

The β => point curve for (U, L, cluster size) from a `get_sims_all` result, with
`method` = "FULL" (K0), "TCA", or "NONE" (not found, curve `nothing`). If `L == TCA_size`
and `prefer_full`, the full K0 run is used when there is one.
"""
function get_sim_from_sims_all(sims_all::Dict{Any, Any}, U, L, TCA_size; prefer_full::Bool=true)
    haskey(sims_all, U) || return nothing, "NONE"
    sims_U = sims_all[U]

    if prefer_full && L == TCA_size
        if haskey(sims_U, L) && haskey(sims_U[L], L)
            return sims_U[L][L], "FULL"
        end
    end

    if haskey(sims_U, TCA_size) && haskey(sims_U[TCA_size], L)
        method = (L == TCA_size) ? "FULL" : "TCA"
        return sims_U[TCA_size][L], method
    end

    if haskey(sims_U, L) && haskey(sims_U[L], L)
        return sims_U[L][L], "FULL"
    end

    return nothing, "NONE"
end

"""
    _get_obs_tuple_safe(sim, obs_name; obs_ix=nothing) -> (Ts, values, errors)

`get_obs_vs_T` (component `obs_ix` if given), or empty vectors when `sim` is `nothing`.
"""
function _get_obs_tuple_safe(sim, obs_name; obs_ix=nothing)
    sim === nothing && return Float64[], Float64[], Float64[]
    if obs_ix === nothing
        return get_obs_vs_T(sim, obs_name)
    end
    return get_obs_vs_T(sim, obs_name, obs_ix)
end

"""
    _sim_has_observable(sim, obs_name) -> Bool

True if any point of the curve `sim` has `obs_name` among its means.
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
    _resolve_subplot_lim(lims, panel_ix; separate_by=:both, n=nothing, U=nothing)

Axis limits for one panel: `lims` itself if it is a single tuple/range; its
`panel_ix`-th entry if a vector; from a Dict keyed by panel index, by n or U
(`separate_by`), or by (n, U). `nothing` if not given.
"""
function _resolve_subplot_lim(lims, panel_ix; separate_by::Symbol=:both, n=nothing, U=nothing)
    lims === nothing && return nothing

    # If a single tuple/range is provided, apply it to every subplot.
    if !(lims isa AbstractDict) && !(lims isa AbstractVector)
        return lims
    end

    if lims isa AbstractVector
        return panel_ix <= length(lims) ? lims[panel_ix] : nothing
    end

    # Dict keys can be panel index, n/U value, or (n, U) for :both.
    haskey(lims, panel_ix) && return lims[panel_ix]
    if separate_by == :n
        haskey(lims, n) && return lims[n]
    elseif separate_by == :U
        haskey(lims, U) && return lims[U]
    else
        haskey(lims, (n, U)) && return lims[(n, U)]
    end

    return nothing
end

"""
    plot_obs_vs_T_from_sims_all(
        sims_all, U, geometry_pairs, obs_name;
        obs_ix=nothing, units_mult=1.0, ylabel=obs_name,
        wanted_n=0.875, xlims=nothing, ylims=nothing,
        plot_against_beta=false, prefer_full=true,
        show_missing=false, legend_pos=:topright,
    )

Plot one observable versus temperature (or beta) for a fixed `U`, using simulations
stored in `sims_all = get_sims_all(sim_info; wanted_n=...)`.

`geometry_pairs` must be a collection of `(L, TCA_size)` tuples.
"""
function plot_obs_vs_T_from_sims_all(
    sims_all::Dict{Any, Any},
    U,
    geometry_pairs,
    obs_name;
    obs_ix=nothing,
    units_mult::Float64=1.0,
    ylabel::String=obs_name,
    wanted_n::Float64=0.875,
    xlims=nothing,
    ylims=nothing,
    plot_against_beta::Bool=false,
    prefer_full::Bool=true,
    show_missing::Bool=false,
    legend_pos=:topright,
    ls::Float64=2.0
)
    geoms = collect(geometry_pairs)
    isempty(geoms) && error("geometry_pairs must contain at least one (L, TCA_size) pair")

    xlab = plot_against_beta ? "β = 1/T" : "T"
    n_txt = "n≈$(wanted_n)"

    fig = plot(
        xlabel=xlab,
        ylabel=ylabel,
        title="$(obs_name), U=$(U), $(n_txt)",
        legend=legend_pos,
        size=(1000, 560),
        left_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
    )

    for (L, TCA_size) in geoms
        sim, _ = get_sim_from_sims_all(sims_all, U, L, TCA_size; prefer_full=prefer_full)

        if !_sim_has_observable(sim, obs_name)
            show_missing && @warn "Observable not found for simulation; skipping" wanted_n=wanted_n U=U L=L TCA_size=TCA_size obs_name=obs_name
            continue
        end

        Ts, vals, errs = _get_obs_tuple_safe(sim, obs_name; obs_ix=obs_ix)

        if isempty(Ts)
            show_missing && @warn "Missing/empty observable" U=U L=L TCA_size=TCA_size obs_name=obs_name
            continue
        end

        xs = plot_against_beta ? 1.0 ./ Ts : Ts
        ys = vals .* units_mult
        es = errs .* units_mult

        p = sortperm(xs)
        plot!(
            fig,
            xs[p],
            ys[p];
            yerr=es[p],
            label="L=$(L), TCA=$(TCA_size)",
            m=:circle,
            lw=ls,
            ms=5,
        )
    end

    xlims !== nothing && plot!(fig, xlims=xlims)
    ylims !== nothing && plot!(fig, ylims=ylims)
    return fig
end

# Convenience: same function for a single geometry `(L, TCA_size)`.
function plot_obs_vs_T_from_sims_all(sims_all::Dict{Any, Any}, U, L, TCA_size, obs_name; kwargs...)
    return plot_obs_vs_T_from_sims_all(sims_all, U, [(L, TCA_size)], obs_name; kwargs...)
end

"""
    plot_obs_vs_T_multi_U_n(
        sim_info, U_list, n_list, geometry_pairs, obs_name;
        layout_mode=:separate, obs_ix=nothing,
        separate_by=:n,
        units_mult=1.0, ylabel=obs_name,
        xlims=nothing, ylims=nothing,
        subplot_xlims=nothing, subplot_ylims=nothing,
        plot_against_beta=false, prefer_full=true,
        show_missing=false, legend_pos=:topright,
        return_value=false,
    )

Plot one observable for several `U` and `n` values.

- `layout_mode=:separate`: split panels according to `separate_by`.
    - `separate_by=:n` one subplot per `n`.
    - `separate_by=:U` one subplot per `U`.
    - `separate_by=:both` grid with rows=`n` and columns=`U`.
- `layout_mode=:together`: all (`n`, `U`, geometry) curves on one figure.

If `return_value=true`, returns `(fig=fig, data=data)` where `data` is a
`Vector{NamedTuple}` with one entry per plotted curve. Each entry has fields
`n, U, L, TCA_size, xs, ys, errs, label, panel_ix` (post units_mult, post
sort, with `xs` already in β-units when `plot_against_beta=true`). Otherwise
returns just `fig`.
"""
function plot_obs_vs_T_multi_U_n(
    sim_info::Dict{String, Any},
    U_list,
    n_list,
    geometry_pairs,
    obs_name;
    layout_mode::Symbol=:separate,
    obs_ix=nothing,
    separate_by::Symbol=:both,
    units_mult::Float64=1.0,
    ylabel::String=obs_name,
    xlims=nothing,
    ylims=nothing,
    subplot_xlims=nothing,
    subplot_ylims=nothing,
    plot_against_beta::Bool=false,
    prefer_full::Bool=true,
    show_missing::Bool=false,
    legend_pos=:topright,
    sims_by_n_cache=nothing,
    return_value::Bool=false,
)
    layout_mode in (:separate, :together) || error("layout_mode must be :separate or :together")
    separate_by in (:n, :U, :both) || error("separate_by must be :n, :U, or :both")

    geoms = collect(geometry_pairs)
    isempty(geoms) && error("geometry_pairs must contain at least one (L, TCA_size) pair")

    n_vals = collect(n_list)
    U_vals = collect(U_list)
    isempty(n_vals) && error("n_list must contain at least one value")
    isempty(U_vals) && error("U_list must contain at least one value")

    sims_by_n = sims_by_n_cache === nothing ? Dict{Any, Any}() : sims_by_n_cache
    for n in n_vals
        haskey(sims_by_n, n) || (sims_by_n[n] = get_sims_all(sim_info; wanted_n=n))
    end

    xlab = plot_against_beta ? "β = 1/T" : "T"
    data = Vector{NamedTuple}()

    if layout_mode == :together
        fig = plot(
            xlabel=xlab,
            ylabel=ylabel,
            title="$(obs_name), multi-U multi-n",
            legend=legend_pos,
            size=(1200, 650),
            left_margin=10Plots.mm,
            bottom_margin=10Plots.mm,
        )

        for n in n_vals
            sims_all_n = sims_by_n[n]
            for U in U_vals
                for (L, TCA_size) in geoms
                    sim, _ = get_sim_from_sims_all(sims_all_n, U, L, TCA_size; prefer_full=prefer_full)

                    if !_sim_has_observable(sim, obs_name)
                        show_missing && @warn "Observable not found for simulation; skipping" n=n U=U L=L TCA_size=TCA_size obs_name=obs_name
                        continue
                    end

                    Ts, vals, errs = _get_obs_tuple_safe(sim, obs_name; obs_ix=obs_ix)

                    if isempty(Ts)
                        show_missing && @warn "Missing/empty observable" n=n U=U L=L TCA_size=TCA_size obs_name=obs_name
                        continue
                    end

                    xs = plot_against_beta ? 1.0 ./ Ts : Ts
                    ys = vals .* units_mult
                    es = errs .* units_mult
                    p = sortperm(xs)

                    label = "n=$(n), U=$(U), L=$(L), TCA=$(TCA_size)"
                    plot!(
                        fig,
                        xs[p],
                        ys[p];
                        yerr=es[p],
                        label=label,
                        m=:circle,
                        lw=2,
                        ms=4,
                    )
                    push!(data, (; n=n, U=U, L=L, TCA_size=TCA_size,
                                   xs=xs[p], ys=ys[p], errs=es[p],
                                   label=label, panel_ix=1))
                end
            end
        end

        xlims_local = _resolve_subplot_lim(subplot_xlims, 1; separate_by=:both)
        ylims_local = _resolve_subplot_lim(subplot_ylims, 1; separate_by=:both)
        xlims_local === nothing && (xlims_local = xlims)
        ylims_local === nothing && (ylims_local = ylims)

        xlims_local !== nothing && plot!(fig, xlims=xlims_local)
        ylims_local !== nothing && plot!(fig, ylims=ylims_local)
        return return_value ? (fig=fig, data=data) : fig
    end

    if separate_by == :n
        n_panels = length(n_vals)
        fig = plot(
            layout=(1, n_panels),
            size=(600 * n_panels, 560),
            legend=legend_pos,
            left_margin=10Plots.mm,
            bottom_margin=10Plots.mm,
        )

        for (panel_ix, n) in enumerate(n_vals)
            sims_all_n = sims_by_n[n]

            plot!(
                fig;
                subplot=panel_ix,
                xlabel=xlab,
                ylabel=ylabel,
                title="$(obs_name), n≈$(n)",
            )

            for U in U_vals
                for (L, TCA_size) in geoms
                    sim, _ = get_sim_from_sims_all(sims_all_n, U, L, TCA_size; prefer_full=prefer_full)
                    Ts, vals, errs = _get_obs_tuple_safe(sim, obs_name; obs_ix=obs_ix)

                    if isempty(Ts)
                        show_missing && @warn "Missing/empty observable" n=n U=U L=L TCA_size=TCA_size obs_name=obs_name
                        continue
                    end

                    xs = plot_against_beta ? 1.0 ./ Ts : Ts
                    ys = vals .* units_mult
                    es = errs .* units_mult
                    p = sortperm(xs)

                    label = "U=$(U), L=$(L), TCA=$(TCA_size)"
                    plot!(
                        fig,
                        xs[p],
                        ys[p];
                        subplot=panel_ix,
                        yerr=es[p],
                        label=label,
                        m=:circle,
                        lw=2,
                        ms=4,
                    )
                    push!(data, (; n=n, U=U, L=L, TCA_size=TCA_size,
                                   xs=xs[p], ys=ys[p], errs=es[p],
                                   label=label, panel_ix=panel_ix))
                end
            end

            xlims_local = _resolve_subplot_lim(subplot_xlims, panel_ix; separate_by=:n, n=n)
            ylims_local = _resolve_subplot_lim(subplot_ylims, panel_ix; separate_by=:n, n=n)
            xlims_local === nothing && (xlims_local = xlims)
            ylims_local === nothing && (ylims_local = ylims)

            xlims_local !== nothing && plot!(fig; subplot=panel_ix, xlims=xlims_local)
            ylims_local !== nothing && plot!(fig; subplot=panel_ix, ylims=ylims_local)
        end

        return return_value ? (fig=fig, data=data) : fig
    elseif separate_by == :U
        u_panels = length(U_vals)
        fig = plot(
            layout=(1, u_panels),
            size=(600 * u_panels, 560),
            legend=legend_pos,
            left_margin=10Plots.mm,
            bottom_margin=10Plots.mm,
        )

        for (panel_ix, U) in enumerate(U_vals)
            plot!(
                fig;
                subplot=panel_ix,
                xlabel=xlab,
                ylabel=ylabel,
                title="$(obs_name), U=$(U)",
            )

            for n in n_vals
                sims_all_n = sims_by_n[n]
                for (L, TCA_size) in geoms
                    sim, _ = get_sim_from_sims_all(sims_all_n, U, L, TCA_size; prefer_full=prefer_full)

                    if !_sim_has_observable(sim, obs_name)
                        show_missing && @warn "Observable not found for simulation; skipping" n=n U=U L=L TCA_size=TCA_size obs_name=obs_name
                        continue
                    end

                    Ts, vals, errs = _get_obs_tuple_safe(sim, obs_name; obs_ix=obs_ix)

                    if isempty(Ts)
                        show_missing && @warn "Missing/empty observable" n=n U=U L=L TCA_size=TCA_size obs_name=obs_name
                        continue
                    end

                    xs = plot_against_beta ? 1.0 ./ Ts : Ts
                    ys = vals .* units_mult
                    es = errs .* units_mult
                    p = sortperm(xs)

                    label = "n=$(n), L=$(L), TCA=$(TCA_size)"
                    plot!(
                        fig,
                        xs[p],
                        ys[p];
                        subplot=panel_ix,
                        yerr=es[p],
                        label=label,
                        m=:circle,
                        lw=2,
                        ms=4,
                    )
                    push!(data, (; n=n, U=U, L=L, TCA_size=TCA_size,
                                   xs=xs[p], ys=ys[p], errs=es[p],
                                   label=label, panel_ix=panel_ix))
                end
            end

            xlims_local = _resolve_subplot_lim(subplot_xlims, panel_ix; separate_by=:U, U=U)
            ylims_local = _resolve_subplot_lim(subplot_ylims, panel_ix; separate_by=:U, U=U)
            xlims_local === nothing && (xlims_local = xlims)
            ylims_local === nothing && (ylims_local = ylims)

            xlims_local !== nothing && plot!(fig; subplot=panel_ix, xlims=xlims_local)
            ylims_local !== nothing && plot!(fig; subplot=panel_ix, ylims=ylims_local)
        end

        return return_value ? (fig=fig, data=data) : fig
    end

    # separate_by == :both
    n_rows = length(n_vals)
    n_cols = length(U_vals)
    fig = plot(
        layout=(n_rows, n_cols),
        size=(520 * n_cols, 420 * n_rows),
        legend=legend_pos,
        left_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
    )

    for (n_ix, n) in enumerate(n_vals)
        sims_all_n = sims_by_n[n]
        for (u_ix, U) in enumerate(U_vals)
            panel_ix = (n_ix - 1) * n_cols + u_ix

            plot!(
                fig;
                subplot=panel_ix,
                xlabel=xlab,
                ylabel=ylabel,
                title="U=$(U), n≈$(n)",
            )

            for (L, TCA_size) in geoms
                sim, _ = get_sim_from_sims_all(sims_all_n, U, L, TCA_size; prefer_full=prefer_full)

                if !_sim_has_observable(sim, obs_name)
                    show_missing && @warn "Observable not found for simulation; skipping" n=n U=U L=L TCA_size=TCA_size obs_name=obs_name
                    continue
                end

                Ts, vals, errs = _get_obs_tuple_safe(sim, obs_name; obs_ix=obs_ix)

                if isempty(Ts)
                    show_missing && @warn "Missing/empty observable" n=n U=U L=L TCA_size=TCA_size obs_name=obs_name
                    continue
                end

                xs = plot_against_beta ? 1.0 ./ Ts : Ts
                ys = vals .* units_mult
                es = errs .* units_mult
                p = sortperm(xs)

                label = "L=$(L), TCA=$(TCA_size)"
                plot!(
                    fig,
                    xs[p],
                    ys[p];
                    subplot=panel_ix,
                    yerr=es[p],
                    label=label,
                    m=:circle,
                    lw=2,
                    ms=4,
                )
                push!(data, (; n=n, U=U, L=L, TCA_size=TCA_size,
                               xs=xs[p], ys=ys[p], errs=es[p],
                               label=label, panel_ix=panel_ix))
            end

            xlims_local = _resolve_subplot_lim(subplot_xlims, panel_ix; separate_by=:both, n=n, U=U)
            ylims_local = _resolve_subplot_lim(subplot_ylims, panel_ix; separate_by=:both, n=n, U=U)
            xlims_local === nothing && (xlims_local = xlims)
            ylims_local === nothing && (ylims_local = ylims)

            xlims_local !== nothing && plot!(fig; subplot=panel_ix, xlims=xlims_local)
            ylims_local !== nothing && plot!(fig; subplot=panel_ix, ylims=ylims_local)
        end
    end

    return return_value ? (fig=fig, data=data) : fig
end

"""
    render_plot(state)

Render the current figure from `PlotState` (intended for GUI callbacks).
"""
function render_plot(state::PlotState)
    return plot_obs_vs_T_multi_U_n(
        state.sim_info,
        state.U_list,
        state.n_list,
        state.geometry_pairs,
        state.obs_name;
        layout_mode=state.layout_mode,
        obs_ix=state.obs_ix,
        separate_by=state.separate_by,
        units_mult=state.units_mult,
        ylabel=state.ylabel,
        xlims=state.xlims,
        ylims=state.ylims,
        subplot_xlims=state.subplot_xlims,
        subplot_ylims=state.subplot_ylims,
        plot_against_beta=state.plot_against_beta,
        prefer_full=state.prefer_full,
        show_missing=state.show_missing,
        legend_pos=state.legend_pos,
        sims_by_n_cache=state.sims_by_n_cache,
    )
end



# Example: multi-U and multi-n plots.
# sim_info = load_and_prepare_data()
# fig_sep = plot_obs_vs_T_multi_U_n(sim_info, [2.0, 3.0, 5.0], [0.5, 0.875, 1.0], geometry_pairs, "Ps";
#     obs_ix=1, plot_against_beta=true, layout_mode=:separate)
# fig_all = plot_obs_vs_T_multi_U_n(sim_info, [2.0, 3.0, 5.0], [0.5, 0.875, 1.0], geometry_pairs, "Ps";
#     obs_ix=1, plot_against_beta=true, layout_mode=:together)

