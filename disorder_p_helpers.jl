"""
Helpers for disorder-vs-p analysis at fixed (U, n), with geometry-aware lookup.

Design goals:
- Keep TClustersize at the deepest level for TCA data.
- Do not rely on unstable notebook-local dictionary shapes.
- Normalize FULL (K0) data by manually mapping TCA=L.

Expected normalized shapes:
- sims_disorder["K0"][U][p][L][L] -> sim_beta_dict
- sims_disorder["K0-TCA"][U][p][L][TCA] -> sim_beta_dict
"""

"""Return numeric key from dictionary that matches target (exact or approx)."""
function numeric_key_match(d::AbstractDict, target; atol=1e-8)
    if haskey(d, target)
        return target
    end
    for k in keys(d)
        if (k isa Number) && (target isa Number) && isapprox(Float64(k), Float64(target); atol=atol, rtol=0.0)
            return k
        end
    end
    return nothing
end

"""
    _ensure_dict_level!(d, key) -> Dict{Any,Any}

`d[key]`, created as an empty Dict first if missing (for building nested Dicts).
"""
function _ensure_dict_level!(d::Dict{Any, Any}, key)
    if !haskey(d, key)
        d[key] = Dict{Any, Any}()
    end
    return d[key]
end

"""
Normalize K0 structure:
raw K0 from get_sims(..., ["U", "impurity", "L"]) is expected as:
    raw[U][p][L] => sim_beta_dict
into:
    out[U][p][L][L] => sim_beta_dict
"""
function _normalize_full_sims(raw_k0::Dict{Any, Any})
    out = Dict{Any, Any}()

    for (U, by_p_any) in raw_k0
        by_p = by_p_any isa AbstractDict ? by_p_any : Dict{Any, Any}()
        out_u = _ensure_dict_level!(out, U)

        for (p, by_L_any) in by_p
            by_L = by_L_any isa AbstractDict ? by_L_any : Dict{Any, Any}()
            out_up = _ensure_dict_level!(out_u, p)

            for (L, sim_beta_dict) in by_L
                out_upl = _ensure_dict_level!(out_up, L)
                # FULL convention: TCA=L
                out_upl[L] = sim_beta_dict
            end
        end
    end

    return out
end

"""
Build disorder-aware simulation dictionaries for fixed wanted_n.

Important choices requested:
- TClustersize is the final sort key for TCA data.
- FULL sims are manually mapped to TCA=L.

Returns Dict with keys:
- "K0" (normalized FULL map)
- "K0-TCA" (TCA map with TClustersize at deepest level)
"""
function build_disorder_sims(
    sim_info::Dict{String, Any},
    wanted_n::Float64;
    disorder_type::String="impurity-gaussian-kernel",
    max_n_difference::Float64=0.05,
)
    # Keep L explicitly so geometry lookup stays deterministic.
    sims_k0_raw = get_sims(
        sim_info,
        wanted_n,
        ["U", "impurity"];
        diag_method="K0",
        disorder_type=disorder_type,
        max_n_difference=max_n_difference,
    )

    # Keep TClustersize at the end as requested.
    sims_tca = get_sims(
        sim_info,
        wanted_n,
        ["U", "impurity", "TClustersize"];
        diag_method="K0-TCA",
        disorder_type=disorder_type,
        max_n_difference=max_n_difference,
    )

    sims_k0 = _normalize_full_sims(sims_k0_raw)

    return Dict{String, Any}(
        "K0" => sims_k0,
        "K0-TCA" => sims_tca,
    )
end

"""
Lookup a simulation dictionary by (U, L, TCA, p).

Lookup priority:
1) FULL map when prefer_full=true and L==TCA.
2) TCA map for exact (L, TCA).
3) FULL fallback.
"""
function get_disorder_sim(
    sims_disorder::Dict{String, Any},
    U,
    L::Int,
    TCA::Int,
    p;
    prefer_full::Bool=true,
)
    sims_k0 = get(sims_disorder, "K0", Dict{Any, Any}())
    sims_tca = get(sims_disorder, "K0-TCA", Dict{Any, Any}())

    # Expected order in sims_disorder maps:
    # mode -> U -> p -> TCA -> L
    function lookup_sim(sims_mode::Dict{Any, Any}, U, p, TCA::Int, L::Int)
        U_key = numeric_key_match(sims_mode, U)
        U_key === nothing && return nothing, nothing

        p_key = numeric_key_match(sims_mode[U_key], p)
        p_key === nothing && return nothing, nothing

        tca_key = numeric_key_match(sims_mode[U_key][p_key], TCA)
        tca_key === nothing && return nothing, nothing

        L_key = numeric_key_match(sims_mode[U_key][p_key][tca_key], L)
        L_key === nothing && return nothing, nothing

        return sims_mode[U_key][p_key][tca_key][L_key], tca_key
    end

    # 1) FULL preferred when geometry is full-size
    if prefer_full && L == TCA
        sim_full, _ = lookup_sim(sims_k0, U, p, L, L)
        sim_full !== nothing && return sim_full, "FULL"
    end

    # 2) Exact TCA lookup with TClustersize at deepest level
    sim_tca, tca_key = lookup_sim(sims_tca, U, p, TCA, L)
    sim_tca !== nothing && return sim_tca, "TCA=$(tca_key)"

    # 3) FULL fallback
    sim_full, _ = lookup_sim(sims_k0, U, p, L, L)
    sim_full !== nothing && return sim_full, "FULL"

    return nothing, "NONE"
end

"""
Plot observable vs temperature while sweeping p for each geometry panel.

Notes:
- For tensor observables, pass obs_ix (e.g. 1 for xx component).
- Uses get_disorder_sim for consistent FULL/TCA lookup.
"""
function plot_obs_vs_T_by_p(
    sims_disorder::Dict{String, Any},
    U,
    wanted_n,
    geometry_pairs,
    p_list,
    obs_name;
    obs_ix=nothing,
    ylabel::String=obs_name,
    units_mult::Float64=1.0,
    plot_against_beta::Bool=false,
    xlims=nothing,
    ylims=nothing,
    prefer_full::Bool=true,
    legend_pos=:topright,
    return_values=false,
    warn_missing::Bool=true,
)
    geoms = collect(geometry_pairs)
    n_panels = length(geoms)
    n_panels == 0 && error("geometry_pairs is empty")

    fig = plot(
        layout=(1, n_panels),
        size=(580 * n_panels, 470),
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        legend=legend_pos,
    )

    xlab = plot_against_beta ? "beta" : "T"
    plot_data = return_values ? Dict() : nothing

    for (sp, (L, TCA)) in enumerate(geoms)
        plot!(
            fig;
            subplot=sp,
            xlabel=xlab,
            ylabel=ylabel,
            title="U=$(U), n~$(wanted_n), L=$(L), TCA=$(TCA)",
        )

        if return_values
            plot_data[(L, TCA)] = []
        end

        for p in p_list
            sim, method = get_disorder_sim(sims_disorder, U, L, TCA, p; prefer_full=prefer_full)
            if sim === nothing
                warn_missing && @warn "Missing simulation" U=U L=L TCA=TCA p=p obs=obs_name
                continue
            end

            Ts, vals, errs = obs_ix === nothing ? get_obs_vs_T(sim, obs_name) : get_obs_vs_T(sim, obs_name, obs_ix)
            isempty(Ts) && continue

            xs = plot_against_beta ? (1.0 ./ Ts) : Ts
            ys = vals .* units_mult
            es = errs .* units_mult
            ord = sortperm(xs)

            if return_values
                push!(plot_data[(L, TCA)], (p=p, xs=xs[ord], ys=ys[ord], es=es[ord]))
            end

            plot!(
                fig,
                xs[ord],
                ys[ord];
                subplot=sp,
                yerr=es[ord],
                label=@sprintf("p=%.3g, U_mean=%.3g", p, U * p),
                m=:circle,
                lw=2,
                ms=4,
            )
        end

        if xlims !== nothing
            plot!(fig; subplot=sp, xlims=xlims)
        end
        if ylims !== nothing
            plot!(fig; subplot=sp, ylims=ylims)
        end
    end

    return return_values ? (fig, plot_data) : fig
end

"""
Plot disorder sweep and overlay homogeneous reference curves.

Behavior:
- First call plots disordered curves and returns values.
- Second call extracts homogeneous values from `sims_homo` at p=0 (K0-TCA path).
- Homogeneous curves are overlaid with a physical label p=1.

If `return_values=true`, returns `(fig, disorder_plot_data, homo_plot_data)`.
"""
function plot_obs_vs_T_by_p_with_homo(
    sims_disorder::Dict{String, Any},
    sims_homo::Dict{String, Any},
    U,
    wanted_n,
    geometry_pairs,
    p_list,
    obs_name;
    obs_ix=nothing,
    ylabel::String=obs_name,
    units_mult::Float64=1.0,
    plot_against_beta::Bool=false,
    xlims=nothing,
    ylims=nothing,
    prefer_full::Bool=true,
    legend_pos=:topright,
    homogeneous_p::Float64=0.0,
    homogeneous_label=nothing,
    homogeneous_marker=:diamond,
    homogeneous_linestyle=:dash,
    homogeneous_color=:black,
    homogeneous_lw::Float64=2.0,
    homogeneous_ms::Float64=5.0,
    return_values=false,
)
    fig, disorder_plot_data = plot_obs_vs_T_by_p(
        sims_disorder,
        U,
        wanted_n,
        geometry_pairs,
        p_list,
        obs_name;
        obs_ix=obs_ix,
        ylabel=ylabel,
        units_mult=units_mult,
        plot_against_beta=plot_against_beta,
        xlims=xlims,
        ylims=ylims,
        prefer_full=prefer_full,
        legend_pos=legend_pos,
        return_values=true,
    )

    _, homo_plot_data = plot_obs_vs_T_by_p(
        sims_homo,
        U,
        wanted_n,
        geometry_pairs,
        [homogeneous_p],
        obs_name;
        obs_ix=obs_ix,
        ylabel=ylabel,
        units_mult=units_mult,
        plot_against_beta=plot_against_beta,
        xlims=xlims,
        ylims=ylims,
        prefer_full=false,
        legend_pos=legend_pos,
        return_values=true,
    )

    geoms = collect(geometry_pairs)
    homo_label = homogeneous_label === nothing ? @sprintf("p=1, U=%.3g", U) : homogeneous_label

    for (sp, (L, TCA)) in enumerate(geoms)
        homo_entries = get(homo_plot_data, (L, TCA), [])
        if isempty(homo_entries)
            @warn "Missing homogeneous simulation for geometry" U=U L=L TCA=TCA p=homogeneous_p obs=obs_name
            continue
        end

        for (ix, entry) in enumerate(homo_entries)
            plot!(
                fig,
                entry.xs,
                entry.ys;
                subplot=sp,
                yerr=entry.es,
                label=(ix == 1 ? homo_label : ""),
                m=homogeneous_marker,
                lw=homogeneous_lw,
                ms=homogeneous_ms,
                ls=homogeneous_linestyle,
                color=homogeneous_color,
            )
        end
    end

    return return_values ? (fig, disorder_plot_data, homo_plot_data) : fig
end

"""
Compare multiple disorder and homogeneous U choices on the same geometry panels.

Inputs are separate:
- `U_disorder_list`: U values used for disordered sweeps (all `p_list` values)
- `U_homo_list`: U values used for homogeneous references (`p=homogeneous_p`)

This avoids pair-coupling and allows selecting each set independently.

If `U_mean_target` is given, only curves whose effective U_mean is within
tolerance are plotted (disorder: U*p ≈ U_mean_target; homo: U ≈ U_mean_target).
Tolerance is `isapprox(U_eff, U_mean_target; atol=U_mean_atol, rtol=U_mean_rtol)`.
Each p still gets its own color.

If `disorder_pairs` is given (e.g. `[(1.5, 0.1), (3.0, 0.2)]`), only those exact
(U, p) combinations are plotted on the disorder side, overriding the
`U_disorder_list` × `p_list` cross product. `U_disorder_list` is then ignored,
but `p_list` is still used to pick the color for each p (pass the full set of
p values you want to share a color scheme).

If `overlay_geometries=true`, every geometry in `geometry_pairs` is drawn on a
single shared panel instead of one panel per geometry. The visual encoding then
becomes: color = geometry, marker = p (cycling through `overlay_p_markers`), line
style = disorder U. Each homogeneous reference is colored by its geometry too, so
it pairs visually with the disordered curves of the same size. Every legend entry
names its `L`/`TCA`. (In the default per-panel mode the encoding is unchanged:
color = p, marker = circle, one panel per geometry.)

If `return_values=true`, returns `(fig, disorder_data_by_U, homo_data_by_U)`.
The returned data is unchanged by `overlay_geometries`: both dicts are keyed by
U, and each value is itself keyed by the `(L, TCA)` geometry, so every geometry's
curves remain individually addressable whether or not they share a panel. Note
the homogeneous data is the third return value (`homo_data_by_U`); unpack all
three (`fig, vals, vals_homo = ...`) to receive it.

If `merge_values=true`, returns `(fig, merged)` instead — a single dict that holds
BOTH the disorder and homogeneous curves together (no separate homo dict to miss).
It is keyed by U, then by the `(L, TCA)` geometry, mapping to a Vector of entries
`(kind, U, p, L, TCA, xs, ys, es)` where `kind` is `:disorder` or `:homo`. So a
disorder U and a homo U that share the same value also share a U key, with both
sets of entries side by side (tell them apart via `entry.kind`). `merge_values`
implies value collection, so `return_values` need not also be set. Example:

    fig, vals = plot_obs_vs_T_by_p_with_homo_multiU(...; merge_values=true)
    for u in sort(collect(keys(vals))), e in vals[u][(18, 10)]
        plot!(fig2, e.xs, e.ys; yerr=e.es, label="\$(e.kind) U=\$(e.U) p=\$(e.p)")
    end
"""
function plot_obs_vs_T_by_p_with_homo_multiU(
    sims_disorder::Dict{String, Any},
    sims_homo::Dict{String, Any},
        U_disorder_list,
        U_homo_list,
    wanted_n,
    geometry_pairs,
    p_list,
    obs_name;
    obs_ix=nothing,
    ylabel::String=obs_name,
    units_mult::Float64=1.0,
    plot_against_beta::Bool=false,
    xlims=nothing,
    ylims=nothing,
    prefer_full::Bool=true,
    legend_pos=:topright,
    homogeneous_p::Float64=0.0,
    homogeneous_marker=:diamond,
    homogeneous_linestyle=:dash,
    homogeneous_lw::Float64=2.0,
    homogeneous_ms::Float64=5.0,
    return_values=false,
    merge_values::Bool=false,
    warn_missing::Bool=true,
    U_mean_target::Union{Real, Nothing}=nothing,
    U_mean_atol::Float64=1e-6,
    U_mean_rtol::Float64=1e-3,
    disorder_pairs=nothing,
    overlay_geometries::Bool=false,
    overlay_p_markers=[:circle, :square, :utriangle, :diamond, :star5, :hexagon, :cross, :xcross],
)
    geoms = collect(geometry_pairs)
    n_geoms = length(geoms)
    n_geoms == 0 && error("geometry_pairs is empty")
    # When overlaying, all geometries share a single panel; otherwise one panel each.
    n_panels = overlay_geometries ? 1 : n_geoms
    # In overlay mode everything lands on subplot 1 and geometry is shown by color.
    geom_subplot(gix) = overlay_geometries ? 1 : gix

    U_dis_vals = collect(U_disorder_list)
    U_homo_vals = collect(U_homo_list)
    p_vals = collect(p_list)
    isempty(U_dis_vals) && error("U_disorder_list is empty")
    isempty(U_homo_vals) && error("U_homo_list is empty")

    matches_U_mean(U_eff) = U_mean_target === nothing ||
        isapprox(Float64(U_eff), Float64(U_mean_target);
                 atol=U_mean_atol, rtol=U_mean_rtol)

    # Find the index of p within p_vals (shared by the color and marker mappings).
    function _p_index(p)
        for (pix, pv) in enumerate(p_vals)
            if (p isa Number) && (pv isa Number) && isapprox(Float64(p), Float64(pv); atol=1e-8, rtol=0.0)
                return pix
            end
        end
        return 1
    end

    # Per-panel mode: color encodes p. Overlay mode: color encodes geometry, so p
    # is moved onto the marker instead (line style still encodes the disorder U).
    _p_color_index(p) = _p_index(p)
    _p_marker(p) = overlay_p_markers[mod1(_p_index(p), length(overlay_p_markers))]

    dis_linestyles = [:solid, :dash, :dot, :dashdot]

    fig = plot(
        layout=(1, n_panels),
        size=(580 * n_panels, 470),
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        legend=legend_pos,
    )

    xlab = plot_against_beta ? "beta" : "T"
    if overlay_geometries
        geom_str = join(["L=$(L),TCA=$(TCA)" for (L, TCA) in geoms], "; ")
        plot!(
            fig;
            subplot=1,
            xlabel=xlab,
            ylabel=ylabel,
            title="n~$(wanted_n)  [$(geom_str)]",
        )
    else
        for (sp, (L, TCA)) in enumerate(geoms)
            plot!(
                fig;
                subplot=sp,
                xlabel=xlab,
                ylabel=ylabel,
                title="n~$(wanted_n), L=$(L), TCA=$(TCA)",
            )
        end
    end

    # `merge_values` also needs the data, so collect whenever either flag is set.
    collect_values = return_values || merge_values
    disorder_data_by_U = collect_values ? Dict{Any, Any}() : nothing
    homo_data_by_U = collect_values ? Dict{Any, Any}() : nothing
    # Single combined view: U => (L, TCA) => Vector of self-describing entries
    # (each carries `kind` = :disorder / :homo so both sides live in one dict).
    merged_data = merge_values ? Dict{Any, Any}() : nothing

    function _merge_into!(U, plot_data, kind)
        geo_dict = get!(merged_data, U, Dict{Any, Any}())
        for ((L, TCA), entries) in plot_data
            lst = get!(geo_dict, (L, TCA), [])
            for e in entries
                push!(lst, (kind=kind, U=U, p=e.p, L=L, TCA=TCA,
                            xs=e.xs, ys=e.ys, es=e.es))
            end
        end
    end

    disorder_groups = if disorder_pairs === nothing
        [(U, collect(p_list)) for U in U_dis_vals]
    else
        groups = Vector{Tuple{Any, Vector{Any}}}()
        for (U, p) in disorder_pairs
            ix = findfirst(t -> t[1] == U, groups)
            if ix === nothing
                push!(groups, (U, Any[p]))
            else
                push!(groups[ix][2], p)
            end
        end
        groups
    end

    for (uix, (U_disorder, group_ps)) in enumerate(disorder_groups)
        dis_ls = dis_linestyles[mod1(uix, length(dis_linestyles))]

        sub_p_list = U_mean_target === nothing ?
            group_ps :
            [p for p in group_ps if matches_U_mean(U_disorder * p)]
        if isempty(sub_p_list)
            continue
        end

        _, disorder_plot_data = plot_obs_vs_T_by_p(
            sims_disorder,
            U_disorder,
            wanted_n,
            geometry_pairs,
            sub_p_list,
            obs_name;
            obs_ix=obs_ix,
            ylabel=ylabel,
            units_mult=units_mult,
            plot_against_beta=plot_against_beta,
            xlims=xlims,
            ylims=ylims,
            prefer_full=prefer_full,
            legend_pos=legend_pos,
            return_values=true,
            warn_missing=warn_missing,
        )

        if collect_values
            disorder_data_by_U[U_disorder] = disorder_plot_data
        end
        merge_values && _merge_into!(U_disorder, disorder_plot_data, :disorder)

        for (gix, (L, TCA)) in enumerate(geoms)
            sp = geom_subplot(gix)
            disorder_entries = get(disorder_plot_data, (L, TCA), [])
            if isempty(disorder_entries)
                warn_missing && @warn "Missing disorder simulations for geometry" U=U_disorder L=L TCA=TCA obs=obs_name
            end

            for entry in disorder_entries
                dis_label = @sprintf(
                    "dis U=%.3g, p=%.3g, U_mean=%.3g",
                    U_disorder,
                    entry.p,
                    U_disorder * entry.p,
                )
                # Overlay: color = geometry, marker = p. Per-panel: color = p, marker = circle.
                if overlay_geometries
                    dis_label *= ", L=$(L), TCA=$(TCA)"
                    dis_color = gix
                    dis_marker = _p_marker(entry.p)
                else
                    dis_color = _p_color_index(entry.p)
                    dis_marker = :circle
                end

                plot!(
                    fig,
                    entry.xs,
                    entry.ys;
                    subplot=sp,
                    yerr=entry.es,
                    label=dis_label,
                    m=dis_marker,
                    lw=2,
                    ms=4,
                    ls=dis_ls,
                    color=dis_color,
                )
            end

            if xlims !== nothing
                plot!(fig; subplot=sp, xlims=xlims)
            end
            if ylims !== nothing
                plot!(fig; subplot=sp, ylims=ylims)
            end
        end
    end

    base_homo_color = length(U_dis_vals)
    for (hix, U_homo) in enumerate(U_homo_vals)
        if !matches_U_mean(U_homo)
            continue
        end
        _, homo_plot_data = plot_obs_vs_T_by_p(
            sims_homo,
            U_homo,
            wanted_n,
            geometry_pairs,
            [homogeneous_p],
            obs_name;
            obs_ix=obs_ix,
            ylabel=ylabel,
            units_mult=units_mult,
            plot_against_beta=plot_against_beta,
            xlims=xlims,
            ylims=ylims,
            prefer_full=false,
            legend_pos=legend_pos,
            return_values=true,
            warn_missing=warn_missing,
        )

        if collect_values
            homo_data_by_U[U_homo] = homo_plot_data
        end
        merge_values && _merge_into!(U_homo, homo_plot_data, :homo)

        for (gix, (L, TCA)) in enumerate(geoms)
            sp = geom_subplot(gix)
            homo_entries = get(homo_plot_data, (L, TCA), [])
            if isempty(homo_entries)
                warn_missing && @warn "Missing homogeneous simulation for geometry" U=U_homo L=L TCA=TCA p=homogeneous_p obs=obs_name
                continue
            end

            # Overlay: color the homo reference by its geometry so it pairs with the
            # disordered curves of the same color. Per-panel: color by homo U as before.
            homo_color = overlay_geometries ? gix : base_homo_color + hix

            for (ix, entry) in enumerate(homo_entries)
                homo_label = @sprintf("homo U=%.3g, p=1, U_mean=%.3g", U_homo, U_homo)
                overlay_geometries && (homo_label *= ", L=$(L), TCA=$(TCA)")

                plot!(
                    fig,
                    entry.xs,
                    entry.ys;
                    subplot=sp,
                    yerr=entry.es,
                    label=(ix == 1 ? homo_label : ""),
                    m=homogeneous_marker,
                    lw=homogeneous_lw,
                    ms=homogeneous_ms,
                    ls=homogeneous_linestyle,
                    color=homo_color,
                )
            end

            if xlims !== nothing
                plot!(fig; subplot=sp, xlims=xlims)
            end
            if ylims !== nothing
                plot!(fig; subplot=sp, ylims=ylims)
            end
        end
    end

    if merge_values
        return (fig, merged_data)
    elseif return_values
        return (fig, disorder_data_by_U, homo_data_by_U)
    else
        return fig
    end
end

"""
Write the merged values from
`plot_obs_vs_T_by_p_with_homo_multiU(...; merge_values=true)` to a tidy ("long")
CSV — one row per temperature point.

Columns: `kind,U,p,L,TCA,<xname>,y,err`
- `kind` is `disorder` or `homo`.
- The x column holds T (or beta, if the plot used `plot_against_beta=true`); pass
  `xname="beta"` to label it accordingly. Defaults to `"T"`.
- Rows are sorted by `(kind, U, L, TCA, x)` for stable, diff-friendly output.

Returns the path written. Example:

    fig, vals = plot_obs_vs_T_by_p_with_homo_multiU(...; merge_values=true)
    save_merged_values_csv(vals, raw"C:\\path\\to\\chi_s_values.csv")
"""
function save_merged_values_csv(merged::AbstractDict, path::AbstractString; xname::String="T")
    rows = NamedTuple[]
    for (_, byLT) in merged, ((L, TCA), entries) in byLT, e in entries
        es = e.es === nothing ? fill(NaN, length(e.xs)) : e.es
        for (x, y, err) in zip(e.xs, e.ys, es)
            push!(rows, (kind=e.kind, U=e.U, p=e.p, L=L, TCA=TCA, x=x, y=y, err=err))
        end
    end
    sort!(rows, by = r -> (String(r.kind), Float64(r.U), r.L, r.TCA, r.x))
    open(path, "w") do io
        println(io, "kind,U,p,L,TCA,$xname,y,err")
        for r in rows
            println(io, "$(r.kind),$(r.U),$(r.p),$(r.L),$(r.TCA),$(r.x),$(r.y),$(r.err)")
        end
    end
    return path
end

"""
Linearly extrapolate an observable to T=0 using the `n_points` lowest-T values.

Returns a NamedTuple `(val, err, Ts_used, vals_used, errs_used)` or `nothing`
if fewer than 2 valid points are available.

Weighted least-squares is used when `errs` contains positive values; otherwise
an unweighted fit is applied. The reported `err` is the standard error of the
intercept from the fit.
"""
function extrapolate_obs_to_zero_T(sim, obs_name::String; obs_ix=nothing, n_points::Int=3)
    sim === nothing && return nothing
    Ts, vals, errs = obs_ix === nothing ?
        get_obs_vs_T(sim, obs_name) :
        get_obs_vs_T(sim, obs_name, obs_ix)
    isempty(Ts) && return nothing

    order = sortperm(Ts)
    k = min(n_points, length(Ts))
    ix = order[1:k]
    Ts_used = Float64.(Ts[ix])
    vals_used = Float64.(vals[ix])
    errs_used = Float64.(errs[ix])

    length(Ts_used) < 2 && return nothing

    ws = [e > 0 && isfinite(e) ? 1.0 / e^2 : 1.0 for e in errs_used]
    S0  = sum(ws)
    Sx  = sum(ws .* Ts_used)
    Sy  = sum(ws .* vals_used)
    Sxx = sum(ws .* Ts_used.^2)
    Sxy = sum(ws .* Ts_used .* vals_used)
    Δ   = S0 * Sxx - Sx^2
    Δ == 0 && return nothing

    a = (Sxx * Sy - Sx * Sxy) / Δ        # intercept = value at T=0
    var_a = Sxx / Δ
    err_a = sqrt(max(var_a, 0.0))

    return (val=a, err=err_a, Ts_used=Ts_used, vals_used=vals_used, errs_used=errs_used)
end

"""
Plot normalized gap Delta_MC_abs / (U_eff/2 * sqrt(n*(2-n))) vs U_eff,
taking the lowest-T point for each simulation.

For disordered sims, U_eff = U * p (U_mean). For homogeneous sims, U_eff = U.
All points sharing the same p value are connected on the same line; homogeneous
sims are merged into the p=1 line. Each point is annotated with its lowest T.

Arguments mirror `plot_obs_vs_T_by_p_with_homo_multiU`.
"""
function plot_Delta_abs_vs_BEC(
    sims_disorder::Dict{String, Any},
    sims_homo::Dict{String, Any},
    U_disorder_list,
    U_homo_list,
    wanted_n,
    geometry_pairs,
    p_list;
    obs_name::String="Delta_MC_abs",
    obs_ix=1,
    homogeneous_p::Float64=0.0,
    prefer_full::Bool=true,
    legend_pos=:outerright,
    warn_missing::Bool=true,
    xlims=nothing,
    ylims=nothing,
    annotate_T::Bool=true,
    annotation_fontsize::Int=7,
    extrapolate_to_zero_T::Bool=false,
    n_extrap_points::Int=3,
    return_values::Bool=false,
)
    geoms = collect(geometry_pairs)
    isempty(geoms) && error("geometry_pairs is empty")

    norm_factor(Ueff, n) = (Ueff / 2) * sqrt(n * (2 - n))

    title_tag = extrapolate_to_zero_T ?
        @sprintf(" (T→0 extrap, %d lowest-T pts)", n_extrap_points) :
        " (lowest-T point)"
    fig = plot(
        xlabel="U_mean = U*p   (homo: U)",
        ylabel="Δ_MC_abs / (U_eff/2 · √(n(2-n)))",
        title=@sprintf("Normalized gap vs U, n≈%.3g%s", wanted_n, title_tag),
        legend=legend_pos,
        size=(1000, 600),
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
    )

    function _fetch_point(sim)
        sim === nothing && return nothing
        if extrapolate_to_zero_T
            r = extrapolate_obs_to_zero_T(sim, obs_name; obs_ix=obs_ix, n_points=n_extrap_points)
            r === nothing && return nothing
            return (val=r.val, err=r.err,
                    T_lo=minimum(r.Ts_used), T_hi=maximum(r.Ts_used),
                    n_used=length(r.Ts_used))
        else
            Ts, vals, errs = obs_ix === nothing ?
                get_obs_vs_T(sim, obs_name) :
                get_obs_vs_T(sim, obs_name, obs_ix)
            isempty(Ts) && return nothing
            ix = argmin(Ts)
            return (val=vals[ix], err=errs[ix],
                    T_lo=Ts[ix], T_hi=Ts[ix], n_used=1)
        end
    end

    # Group points by p. p=1 collects both p=1 disorder sims and all homo sims.
    grouped = Dict{Float64, Vector{NamedTuple}}()
    push_pt!(p_key, pt) = push!(get!(grouped, Float64(p_key), NamedTuple[]), pt)

    for U_disorder in U_disorder_list
        for (L, TCA) in geoms
            for p in p_list
                sim, _ = get_disorder_sim(sims_disorder, U_disorder, L, TCA, p; prefer_full=prefer_full)
                pt = _fetch_point(sim)
                if pt === nothing
                    warn_missing && @warn "Missing/empty disorder simulation" U=U_disorder L=L TCA=TCA p=p obs=obs_name
                    continue
                end
                U_eff = U_disorder * p
                norm = norm_factor(U_eff, wanted_n)
                push_pt!(p, (
                    x=U_eff, y=pt.val / norm, yerr=pt.err / norm,
                    T_lo=pt.T_lo, T_hi=pt.T_hi, n_used=pt.n_used,
                    U=U_disorder, p=Float64(p), L=L, TCA=TCA,
                    is_homo=false,
                ))
            end
        end
    end

    for U_homo in U_homo_list
        for (L, TCA) in geoms
            sim, _ = get_disorder_sim(sims_homo, U_homo, L, TCA, homogeneous_p; prefer_full=false)
            pt = _fetch_point(sim)
            if pt === nothing
                warn_missing && @warn "Missing/empty homo simulation" U=U_homo L=L TCA=TCA p=homogeneous_p obs=obs_name
                continue
            end
            norm = norm_factor(U_homo, wanted_n)
            push_pt!(1.0, (
                x=U_homo, y=pt.val / norm, yerr=pt.err / norm,
                T_lo=pt.T_lo, T_hi=pt.T_hi, n_used=pt.n_used,
                U=U_homo, p=1.0, L=L, TCA=TCA,
                is_homo=true,
            ))
        end
    end

    for p_val in sort(collect(keys(grouped)))
        pts = grouped[p_val]
        sort!(pts, by=pt -> pt.x)
        xs = [pt.x for pt in pts]
        ys = [pt.y for pt in pts]
        yerrs = [pt.yerr for pt in pts]
        has_homo = any(pt -> pt.is_homo, pts)
        lbl = isapprox(p_val, 1.0) && has_homo ?
            @sprintf("p=1 (disorder + homo)") :
            @sprintf("p=%.3g", p_val)
        plot!(fig, xs, ys; yerr=yerrs, label=lbl, m=:circle, lw=2, ms=5)

        if annotate_T
            for pt in pts
                ann = extrapolate_to_zero_T ?
                    @sprintf(" T∈[%.3g,%.3g]", pt.T_lo, pt.T_hi) :
                    @sprintf(" T=%.3g", pt.T_lo)
                annotate!(fig, pt.x, pt.y,
                    text(ann, annotation_fontsize, :left, :bottom))
            end
        end
    end

    xlims !== nothing && plot!(fig, xlims=xlims)
    ylims !== nothing && plot!(fig, ylims=ylims)

    return return_values ? (fig, grouped) : fig
end

"""
Plot Delta_MC_abs / (√π/2 · √(U_eff · T)) vs T, one curve per (U, p) pair.

For disordered sims, U_eff = U * p. For homogeneous sims, U_eff = U
(plotted with dashed diamond markers).

Returns `fig`, or `(fig, plot_data)` if `return_values=true`; `plot_data`
keys are `(:disorder, U, p, L, TCA)` / `(:homo, U, L, TCA)`.
"""
function plot_Delta_abs_vs_thermal_gaussian(
    sims_disorder::Dict{String, Any},
    sims_homo::Dict{String, Any},
    U_disorder_list,
    U_homo_list,
    wanted_n,
    geometry_pairs,
    p_list;
    obs_name::String="Delta_MC_abs",
    obs_ix=1,
    homogeneous_p::Float64=0.0,
    prefer_full::Bool=true,
    legend_pos=:topright,
    warn_missing::Bool=true,
    xlims=nothing,
    ylims=nothing,
    plot_against_beta::Bool=false,
    reverse_ratio::Bool=false,
    return_values::Bool=false,
    x_is_UT::Bool=false
)
    geoms = collect(geometry_pairs)
    n_panels = length(geoms)
    n_panels == 0 && error("geometry_pairs is empty")

    norm_factor(Ueff, T) = (sqrt(pi) / 2) * sqrt(Ueff * T)

    # y = norm/Δ when reverse_ratio, else Δ/norm. Error propagation assumes
    # norm has no uncertainty, so σ_y/|y| = σ_Δ/|Δ| in both cases.
    function _ratio(vals, errs, norms)
        if reverse_ratio
            ys = norms ./ vals
            es = abs.(ys) .* (errs ./ abs.(vals))
        else
            ys = vals ./ norms
            es = errs ./ norms
        end
        return ys, es
    end

    fig = plot(
        layout=(1, n_panels),
        size=(580 * n_panels, 470),
        legend=legend_pos,
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
    )

    xlab = plot_against_beta ? "β = 1/T" : "T"
    xlab = x_is_UT ? "$xlab · U" : xlab
    ylab = reverse_ratio ?
        "(√π/2 · √(U_eff · T)) / Δ_MC_abs" :
        "Δ_MC_abs / (√π/2 · √(U_eff · T))"
    for (sp, (L, TCA)) in enumerate(geoms)
        plot!(fig; subplot=sp, xlabel=xlab, ylabel=ylab,
              title="n≈$(wanted_n), L=$(L), TCA=$(TCA)")
    end

    plot_data = return_values ? Dict{Any, Any}() : nothing

    function _fetch(sim)
        sim === nothing && return nothing
        Ts, vals, errs = obs_ix === nothing ?
            get_obs_vs_T(sim, obs_name) :
            get_obs_vs_T(sim, obs_name, obs_ix)
        isempty(Ts) && return nothing
        return (Ts=Float64.(Ts), vals=Float64.(vals), errs=Float64.(errs))
    end

    for U_disorder in U_disorder_list
        for p in p_list
            U_eff = U_disorder * p
            for (sp, (L, TCA)) in enumerate(geoms)
                sim, _ = get_disorder_sim(sims_disorder, U_disorder, L, TCA, p; prefer_full=prefer_full)
                d = _fetch(sim)
                if d === nothing
                    warn_missing && @warn "Missing/empty disorder simulation" U=U_disorder L=L TCA=TCA p=p obs=obs_name
                    continue
                end
                norms = norm_factor.(U_eff, d.Ts)
                ys, es = _ratio(d.vals, d.errs, norms)
                xs = plot_against_beta ? (1.0 ./ d.Ts) : d.Ts
                xs = x_is_UT ? (xs .* U_eff) : xs
                ord = sortperm(xs)
                lbl = @sprintf("dis U=%.3g, p=%.3g, U_eff=%.3g",
                    U_disorder, p, U_eff)
                plot!(fig, xs[ord], ys[ord]; subplot=sp, yerr=es[ord],
                      label=lbl, m=:circle, lw=2, ms=4)

                if return_values
                    plot_data[(:disorder, U_disorder, p, L, TCA)] = (
                        xs=xs[ord], ys=ys[ord], es=es[ord], Ts=d.Ts[ord],
                        U=U_disorder, p=p, U_eff=U_eff,
                    )
                end
            end
        end
    end

    for U_homo in U_homo_list
        for (sp, (L, TCA)) in enumerate(geoms)
            sim, _ = get_disorder_sim(sims_homo, U_homo, L, TCA, homogeneous_p; prefer_full=false)
            d = _fetch(sim)
            if d === nothing
                warn_missing && @warn "Missing/empty homo simulation" U=U_homo L=L TCA=TCA p=homogeneous_p obs=obs_name
                continue
            end
            norms = norm_factor.(U_homo, d.Ts)
            ys, es = _ratio(d.vals, d.errs, norms)
            xs = plot_against_beta ? (1.0 ./ d.Ts) : d.Ts
            xs = x_is_UT ? (xs .* U_homo) : xs
            ord = sortperm(xs)
            lbl = @sprintf("homo U=%.3g", U_homo)
            plot!(fig, xs[ord], ys[ord]; subplot=sp, yerr=es[ord],
                  label=lbl, m=:diamond, lw=2, ms=5, ls=:dash)

            if return_values
                plot_data[(:homo, U_homo, L, TCA)] = (
                    xs=xs[ord], ys=ys[ord], es=es[ord], Ts=d.Ts[ord],
                    U=U_homo, U_eff=U_homo,
                )
            end
        end
    end

    for sp in 1:n_panels
        xlims !== nothing && plot!(fig; subplot=sp, xlims=xlims)
        ylims !== nothing && plot!(fig; subplot=sp, ylims=ylims)
    end

    return return_values ? (fig, plot_data) : fig
end

"""
Plot Tc (from BKT-line crossing of the stiffness) vs U_mean = U*p
for a single geometry panel.

Each line corresponds to one p value (distinct color); points along the line
sweep over U values, so the x coordinate is U * p and the y coordinate is Tc.

Tc is extracted by `find_tc` from statistics.jl using the stiffness vs T curve
and its crossing with the BKT line 2T/pi.

Arguments:
- `sims_disorder`: Dict returned by `build_disorder_sims`.
- `U_list`: U values to sweep along each p-line.
- `p_list`: p (impurity fraction) values; one line is drawn per p.
- `geometry`: Tuple `(L, TCA)` specifying the geometry panel.

Keyword arguments:
- `lowest_q`: forwarded to `find_tc`.
- `prefer_full`: forwarded to `get_disorder_sim`.
- `only_crossings`: skip points where the BKT line is not actually crossed.
- `sims_homo`: optional Dict of homogeneous sims (same shape as `sims_disorder`,
  produced by `build_disorder_sims`). When provided, a homogeneous Tc line is
  overlaid using `U_homo_list` (looked up at `homogeneous_p`, labeled p=1,
  so the x coordinate is just U).
- `U_homo_list`: U values for the homogeneous line (defaults to `U_list`).
- `homogeneous_p`: p key used to look up homogeneous sims (default 0.0).
- `tc_method`: `:crossing` (default) uses `find_tc` (linear interpolation
  between the two neighbors of the argmin; no error bar), `:fit` uses
  `find_tc_fit` which does a local weighted polynomial fit and returns a
  statistical Tc uncertainty that is drawn as vertical error bars on Tc.
- `fit_window`, `fit_order`: forwarded to `find_tc_fit` when `tc_method=:fit`.
- `use_U_raw`: if true, plot x = U (raw) instead of U*p (default `false`).
  Axis label and title update accordingly. The data field name in the
  returned NamedTuple stays `U_mean` regardless, but holds raw U values
  in this mode.
- `return_values`: if true, returns `(fig, data)` where `data[p]` holds the
  `(U_mean, Tc, Tc_err)` arrays for that disorder line; the homogeneous line,
  if drawn, is stored under the key `:homo`.
"""
function plot_Tc_vs_Umean_by_p(
    sims_disorder::Dict{String, Any},
    U_list,
    p_list,
    geometry::Tuple{Int, Int};
    lowest_q::Bool=false,
    prefer_full::Bool=true,
    only_crossings::Bool=false,
    use_U_raw::Bool=false,
    sims_homo::Union{Dict{String, Any}, Nothing}=nothing,
    U_homo_list=nothing,
    homogeneous_p::Float64=0.0,
    homogeneous_marker=:diamond,
    homogeneous_linestyle=:dash,
    homogeneous_color=:black,
    homogeneous_lw::Float64=2.0,
    homogeneous_ms::Float64=5.0,
    xlims=nothing,
    ylims=nothing,
    legend_pos=:topleft,
    title_extra::String="",
    return_values::Bool=false,
    warn_missing::Bool=true,
    tc_method::Symbol=:crossing,
    fit_window::Int=3,
    fit_order::Union{Int, Symbol}=:auto,
)
    L, TCA = geometry
    x_label_name = use_U_raw ? "U" : "U_mean"
    title_str = "Tc vs $(x_label_name), L=$(L), TCA=$(TCA)"
    if !isempty(title_extra)
        title_str *= " | " * title_extra
    end

    fig = plot(
        xlabel=use_U_raw ? "U" : "U_mean = U * p",
        ylabel="Tc",
        title=title_str,
        legend=legend_pos,
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        size=(650, 500),
    )

    plot_data = return_values ? Dict{Any, Any}() : nothing

    is_p_one(p) = (p isa Number) && isapprox(Float64(p), 1.0; atol=1e-8, rtol=0.0)

    # Unified Tc extractor. Returns (Tc, Tc_err, crosses).
    extract_tc = if tc_method === :fit
        sim -> find_tc_fit(sim; lowest_q=lowest_q, window=fit_window, order=fit_order)
    elseif tc_method === :crossing
        function (sim)
            Tc, crosses, _ = find_tc(sim; lowest_q=lowest_q)
            return Tc, NaN, crosses
        end
    else
        error("Unknown tc_method=$(tc_method); expected :crossing or :fit")
    end

    homo_xs = Float64[]
    homo_ys = Float64[]
    homo_es = Float64[]

    for (pix, p) in enumerate(p_list)
        xs = Float64[]
        ys = Float64[]
        es = Float64[]

        for U in U_list
            sim, _ = get_disorder_sim(sims_disorder, U, L, TCA, p; prefer_full=prefer_full)
            if sim === nothing
                warn_missing && @warn "Missing simulation for Tc extraction" U=U L=L TCA=TCA p=p
                continue
            end

            Tc, Tc_err, crosses = extract_tc(sim)
            if isnan(Tc)
                continue
            end
            if only_crossings && !crosses
                continue
            end

            push!(xs, use_U_raw ? Float64(U) : Float64(U) * Float64(p))
            push!(ys, Tc)
            push!(es, Tc_err)
        end

        if return_values
            plot_data[p] = (U_mean=copy(xs), Tc=copy(ys), Tc_err=copy(es))
        end

        if isempty(xs)
            warn_missing && @warn "No Tc points collected for p" p=p L=L TCA=TCA
            continue
        end

        if is_p_one(p)
            # Physically p=1 is a homogeneous simulation — merge into homo line.
            append!(homo_xs, xs)
            append!(homo_ys, ys)
            append!(homo_es, es)
            continue
        end

        ord = sortperm(xs)
        es_ord = es[ord]
        yerr_arg = any(isfinite, es_ord) ? replace(es_ord, NaN => 0.0) : nothing
        plot!(
            fig,
            xs[ord],
            ys[ord];
            label=@sprintf("p=%.3g", p),
            m=:circle,
            lw=2,
            ms=5,
            color=pix,
            yerr=yerr_arg,
        )
    end

    if sims_homo !== nothing
        homo_Us = U_homo_list === nothing ? U_list : U_homo_list
        for U in homo_Us
            sim, _ = get_disorder_sim(sims_homo, U, L, TCA, homogeneous_p; prefer_full=false)
            if sim === nothing
                warn_missing && @warn "Missing homogeneous simulation for Tc extraction" U=U L=L TCA=TCA p=homogeneous_p
                continue
            end
            Tc, Tc_err, crosses = extract_tc(sim)
            if isnan(Tc)
                continue
            end
            if only_crossings && !crosses
                continue
            end
            push!(homo_xs, Float64(U))
            push!(homo_ys, Tc)
            push!(homo_es, Tc_err)
        end
    end

    if return_values
        plot_data[:homo] = (U_mean=copy(homo_xs), Tc=copy(homo_ys), Tc_err=copy(homo_es))
    end

    if !isempty(homo_xs)
        ord_h = sortperm(homo_xs)
        es_h = homo_es[ord_h]
        yerr_h = any(isfinite, es_h) ? replace(es_h, NaN => 0.0) : nothing
        plot!(
            fig,
            homo_xs[ord_h],
            homo_ys[ord_h];
            label="p=1 (homo)",
            m=homogeneous_marker,
            lw=homogeneous_lw,
            ms=homogeneous_ms,
            ls=homogeneous_linestyle,
            color=homogeneous_color,
            yerr=yerr_h,
        )
    elseif sims_homo !== nothing
        warn_missing && @warn "No homogeneous Tc points collected" L=L TCA=TCA p=homogeneous_p
    end

    if xlims !== nothing
        plot!(fig; xlims=xlims)
    end
    if ylims !== nothing
        plot!(fig; ylims=ylims)
    end

    return return_values ? (fig, plot_data) : fig
end

"""
Greedy 1-D grouping of `vals` by tolerance.

After sorting ascending, a new group is started whenever the next value
exceeds the current group's minimum by more than `tol`. This guarantees
each group spans at most `tol` in `vals`. Returns a `Vector{Vector{Int}}`
of original indices, one inner vector per group.
"""
function _group_by_tolerance(vals::AbstractVector{<:Real}, tol::Real)
    isempty(vals) && return Vector{Vector{Int}}()
    ord = sortperm(vals)
    groups = Vector{Vector{Int}}()
    current = Int[ord[1]]
    group_min = Float64(vals[ord[1]])
    for k in 2:length(ord)
        i = ord[k]
        v = Float64(vals[i])
        if v - group_min <= tol
            push!(current, i)
        else
            push!(groups, current)
            current = Int[i]
            group_min = v
        end
    end
    push!(groups, current)
    return groups
end

"""
Plot iso-Tc lines in (p, U) space.

Sweeps every `(p, U)` pair in `p_list × U_list`, extracts Tc using the same
machinery as `plot_Tc_vs_Umean_by_p`, then groups all collected points by
Tc using `tc_tol` (default `0.002`). Each group is drawn as a single line
(sorted by p) with x = p and y = U (raw, *not* U*p). Lines are colored by
their group's mean Tc using the `cmap` colormap and a colorbar is drawn.

Arguments and shared kwargs (`sims_disorder`, `U_list`, `p_list`,
`geometry`, `lowest_q`, `prefer_full`, `only_crossings`, `sims_homo`,
`U_homo_list`, `homogeneous_p`, `tc_method`, `fit_window`, `fit_order`,
`warn_missing`, `xlims`, `ylims`) match `plot_Tc_vs_Umean_by_p`.

Iso-Tc-specific kwargs:
- `tc_tol`: tolerance for grouping points into the same Tc line (default `0.002`).
- `group_by_tc`: if false, skip Tc-grouping entirely — every point is plotted
  individually as a scatter marker colored by its own Tc (default `true`).
  When false, `tc_tol` and `min_group_size` are ignored, and `scatter_only`
  is implied.
- `cmap`: colormap symbol for the gradient (default `:viridis`).
- `min_group_size`: minimum points required to draw a group (default `1`).
- `show_legend`: if true, label each line with its mean Tc; default `false`
  (the colorbar already encodes Tc).
- `colorbar_title`: label for the colorbar (default `"Tc"`).
- `swap_axes`: if true, plot x = U and y = p (default `false`, x = p, y = U).
- `scatter_only`: if true, draw markers only (no connecting lines) (default `false`).
- `show_missing`: if true, emit `@warn` for missing simulations and empty
  groupings (default `false` — this sweep is bulk and noisy by nature).
- `return_values`: if true, returns `(fig, data)` where `data` is a
  `Vector{NamedTuple}` with fields `(Tc_mean, Tc_min, Tc_max, p, U, Tc, Tc_err)`
  per group, plus a key `:all_points` with the unfiltered point cloud.
"""
function plot_U_vs_p_by_Tc(
    sims_disorder::Dict{String, Any},
    U_list,
    p_list,
    geometry::Tuple{Int, Int};
    tc_tol::Real=0.002,
    group_by_tc::Bool=true,
    cmap::Symbol=:viridis,
    min_group_size::Int=1,
    show_legend::Bool=false,
    colorbar_title::String="Tc",
    swap_axes::Bool=false,
    scatter_only::Bool=false,
    lowest_q::Bool=false,
    prefer_full::Bool=true,
    only_crossings::Bool=false,
    sims_homo::Union{Dict{String, Any}, Nothing}=nothing,
    U_homo_list=nothing,
    homogeneous_p::Float64=0.0,
    xlims=nothing,
    ylims=nothing,
    title_extra::String="",
    return_values::Bool=false,
    show_missing::Bool=false,
    tc_method::Symbol=:crossing,
    fit_window::Int=3,
    fit_order::Union{Int, Symbol}=:auto,
)
    L, TCA = geometry
    base_axes = swap_axes ? "p vs U" : "U vs p"
    title_str = "$(base_axes), iso-Tc lines (tol=$(tc_tol)), L=$(L), TCA=$(TCA)"
    if !isempty(title_extra)
        title_str *= " | " * title_extra
    end

    extract_tc = if tc_method === :fit
        sim -> find_tc_fit(sim; lowest_q=lowest_q, window=fit_window, order=fit_order)
    elseif tc_method === :crossing
        function (sim)
            Tc, crosses, _ = find_tc(sim; lowest_q=lowest_q)
            return Tc, NaN, crosses
        end
    else
        error("Unknown tc_method=$(tc_method); expected :crossing or :fit")
    end

    all_p   = Float64[]
    all_U   = Float64[]
    all_Tc  = Float64[]
    all_err = Float64[]

    for p in p_list
        for U in U_list
            sim, _ = get_disorder_sim(sims_disorder, U, L, TCA, p; prefer_full=prefer_full)
            if sim === nothing
                show_missing && @warn "Missing simulation for Tc extraction" U=U L=L TCA=TCA p=p
                continue
            end
            Tc, Tc_err, crosses = extract_tc(sim)
            if isnan(Tc) || (only_crossings && !crosses)
                continue
            end
            push!(all_p,   Float64(p))
            push!(all_U,   Float64(U))
            push!(all_Tc,  Tc)
            push!(all_err, Tc_err)
        end
    end

    if sims_homo !== nothing
        homo_Us = U_homo_list === nothing ? U_list : U_homo_list
        for U in homo_Us
            sim, _ = get_disorder_sim(sims_homo, U, L, TCA, homogeneous_p; prefer_full=false)
            if sim === nothing
                show_missing && @warn "Missing homogeneous simulation for Tc extraction" U=U L=L TCA=TCA p=homogeneous_p
                continue
            end
            Tc, Tc_err, crosses = extract_tc(sim)
            if isnan(Tc) || (only_crossings && !crosses)
                continue
            end
            push!(all_p,   1.0)
            push!(all_U,   Float64(U))
            push!(all_Tc,  Tc)
            push!(all_err, Tc_err)
        end
    end

    if isempty(all_Tc)
        show_missing && @warn "No Tc points collected for iso-Tc plot" L=L TCA=TCA
    end

    if group_by_tc
        groups_idx = _group_by_tolerance(all_Tc, tc_tol)
    else
        groups_idx = [[i] for i in 1:length(all_Tc)]
    end
    group_means = [isempty(g) ? NaN : mean(all_Tc[g]) for g in groups_idx]

    Tc_lo, Tc_hi = isempty(all_Tc) ? (0.0, 1.0) : extrema(filter(isfinite, all_Tc))
    if !isfinite(Tc_lo) || !isfinite(Tc_hi) || Tc_lo == Tc_hi
        Tc_lo = isfinite(Tc_lo) ? Tc_lo - 0.5 : 0.0
        Tc_hi = isfinite(Tc_hi) ? Tc_hi + 0.5 : 1.0
    end

    fig = plot(
        xlabel=swap_axes ? "U" : "p",
        ylabel=swap_axes ? "p" : "U",
        title=title_str,
        legend=show_legend ? :outerright : false,
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        size=(700, 500),
        colorbar=true,
        colorbar_title=colorbar_title,
        clims=(Tc_lo, Tc_hi),
    )

    plot_data = return_values ? Any[] : nothing

    if !group_by_tc && !isempty(all_Tc)
        xs_all = swap_axes ? all_U : all_p
        ys_all = swap_axes ? all_p : all_U
        scatter!(
            fig,
            xs_all, ys_all;
            marker_z=all_Tc,
            c=cmap,
            clims=(Tc_lo, Tc_hi),
            colorbar=true,
            colorbar_title=colorbar_title,
            m=:circle,
            ms=5,
            msw=0,
            label="",
        )
        if return_values
            for i in 1:length(all_Tc)
                push!(plot_data, (
                    Tc_mean=all_Tc[i],
                    Tc_min=all_Tc[i],
                    Tc_max=all_Tc[i],
                    p=[all_p[i]],
                    U=[all_U[i]],
                    Tc=[all_Tc[i]],
                    Tc_err=[all_err[i]],
                ))
            end
        end

        if xlims !== nothing
            plot!(fig; xlims=xlims)
        end
        if ylims !== nothing
            plot!(fig; ylims=ylims)
        end

        if return_values
            return fig, (
                groups=plot_data,
                all_points=(p=copy(all_p), U=copy(all_U), Tc=copy(all_Tc), Tc_err=copy(all_err)),
            )
        end
        return fig
    end

    for (gi, g) in enumerate(groups_idx)
        length(g) < min_group_size && continue
        ps = all_p[g]
        Us = all_U[g]
        Tcs = all_Tc[g]
        es  = all_err[g]
        # Sort by the chosen x-axis variable so the connecting line is monotonic.
        ord = swap_axes ? sortperm(Us) : sortperm(ps)
        ps  = ps[ord]
        Us  = Us[ord]
        Tcs = Tcs[ord]
        es  = es[ord]
        tc_mean = group_means[gi]
        zline = fill(tc_mean, length(ps))
        xs_plot = swap_axes ? Us : ps
        ys_plot = swap_axes ? ps : Us
        if scatter_only
            scatter!(
                fig,
                xs_plot, ys_plot;
                marker_z=zline,
                c=cmap,
                clims=(Tc_lo, Tc_hi),
                colorbar=true,
                colorbar_title=colorbar_title,
                m=:circle,
                ms=5,
                msw=0,
                label=show_legend ? @sprintf("Tc≈%.4f (n=%d)", tc_mean, length(g)) : "",
            )
        else
            plot!(
                fig,
                xs_plot, ys_plot;
                line_z=zline,
                marker_z=zline,
                c=cmap,
                clims=(Tc_lo, Tc_hi),
                colorbar=true,
                colorbar_title=colorbar_title,
                m=:circle,
                ms=5,
                lw=2,
                label=show_legend ? @sprintf("Tc≈%.4f (n=%d)", tc_mean, length(g)) : "",
            )
        end
        if return_values
            push!(plot_data, (
                Tc_mean=tc_mean,
                Tc_min=minimum(Tcs),
                Tc_max=maximum(Tcs),
                p=copy(ps),
                U=copy(Us),
                Tc=copy(Tcs),
                Tc_err=copy(es),
            ))
        end
    end

    if xlims !== nothing
        plot!(fig; xlims=xlims)
    end
    if ylims !== nothing
        plot!(fig; ylims=ylims)
    end

    if return_values
        return fig, (
            groups=plot_data,
            all_points=(p=copy(all_p), U=copy(all_U), Tc=copy(all_Tc), Tc_err=copy(all_err)),
        )
    end
    return fig
end

"""
Plot T* (chi_spin maximum from `find_Tstar`) vs U_mean = U*p for a single
geometry panel. Mirrors `plot_Tc_vs_Umean_by_p`: one colored line per p,
points sweep U.

T* is extracted with `find_Tstar(sim; len=tstar_len)`; only points where the
parabolic fit succeeded (`worked == true`) and the resulting T* is finite
are kept. There is no statistical error on T*, so no y-error bars are drawn.

Arguments and shared kwargs (`sims_disorder`, `U_list`, `p_list`,
`geometry`, `prefer_full`, `use_U_raw`, `sims_homo`, `U_homo_list`,
`homogeneous_p`, `homogeneous_marker`, `homogeneous_linestyle`,
`homogeneous_color`, `homogeneous_lw`, `homogeneous_ms`, `xlims`, `ylims`,
`legend_pos`, `title_extra`, `return_values`, `warn_missing`) match
`plot_Tc_vs_Umean_by_p`.

T*-specific kwargs:
- `tstar_len`: window length forwarded to `find_Tstar` (default `3`).
- `tstar_scale`: multiplier applied to every T* value before plotting and
  before being stored in the returned data (default `1.0`). Useful for
  showing e.g. T*/2 alongside Tc on the same axes. The y-axis label is
  updated to reflect the scale when `tstar_scale != 1`.

When `return_values=true`, `data[p]` holds `(U_mean, Tstar)`; the
homogeneous line, if drawn, is stored under `:homo`. The stored Tstar
values are already multiplied by `tstar_scale`.
"""
function plot_Tstar_vs_Umean_by_p(
    sims_disorder::Dict{String, Any},
    U_list,
    p_list,
    geometry::Tuple{Int, Int};
    prefer_full::Bool=true,
    use_U_raw::Bool=false,
    sims_homo::Union{Dict{String, Any}, Nothing}=nothing,
    U_homo_list=nothing,
    homogeneous_p::Float64=0.0,
    homogeneous_marker=:diamond,
    homogeneous_linestyle=:dash,
    homogeneous_color=:black,
    homogeneous_lw::Float64=2.0,
    homogeneous_ms::Float64=5.0,
    xlims=nothing,
    ylims=nothing,
    legend_pos=:topleft,
    title_extra::String="",
    return_values::Bool=false,
    warn_missing::Bool=true,
    tstar_len::Int=3,
    tstar_scale::Real=1.0,
)
    L, TCA = geometry
    x_label_name = use_U_raw ? "U" : "U_mean"
    scale_f = Float64(tstar_scale)
    is_scaled = !isapprox(scale_f, 1.0; atol=1e-12, rtol=0.0)
    ylab = is_scaled ? @sprintf("%.4g · T*", scale_f) : "T*"
    title_str = "$(ylab) vs $(x_label_name), L=$(L), TCA=$(TCA)"
    if !isempty(title_extra)
        title_str *= " | " * title_extra
    end

    fig = plot(
        xlabel=use_U_raw ? "U" : "U_mean = U * p",
        ylabel=ylab,
        title=title_str,
        legend=legend_pos,
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        size=(650, 500),
    )

    plot_data = return_values ? Dict{Any, Any}() : nothing

    is_p_one(p) = (p isa Number) && isapprox(Float64(p), 1.0; atol=1e-8, rtol=0.0)

    function extract_tstar(sim)
        try
            ts, _, worked = find_Tstar(sim; len=tstar_len)
            if worked && ts !== nothing && isfinite(ts)
                return scale_f * Float64(ts)
            end
        catch err
            warn_missing && @warn "find_Tstar failed" exception=err
        end
        return NaN
    end

    homo_xs = Float64[]
    homo_ys = Float64[]

    for (pix, p) in enumerate(p_list)
        xs = Float64[]
        ys = Float64[]

        for U in U_list
            sim, _ = get_disorder_sim(sims_disorder, U, L, TCA, p; prefer_full=prefer_full)
            if sim === nothing
                warn_missing && @warn "Missing simulation for T* extraction" U=U L=L TCA=TCA p=p
                continue
            end

            ts = extract_tstar(sim)
            if isnan(ts)
                continue
            end

            push!(xs, use_U_raw ? Float64(U) : Float64(U) * Float64(p))
            push!(ys, ts)
        end

        if return_values
            plot_data[p] = (U_mean=copy(xs), Tstar=copy(ys))
        end

        if isempty(xs)
            warn_missing && @warn "No T* points collected for p" p=p L=L TCA=TCA
            continue
        end

        if is_p_one(p)
            append!(homo_xs, xs)
            append!(homo_ys, ys)
            continue
        end

        ord = sortperm(xs)
        plot!(
            fig,
            xs[ord],
            ys[ord];
            label=@sprintf("p=%.3g", p),
            m=:circle,
            lw=2,
            ms=5,
            color=pix,
        )
    end

    if sims_homo !== nothing
        homo_Us = U_homo_list === nothing ? U_list : U_homo_list
        for U in homo_Us
            sim, _ = get_disorder_sim(sims_homo, U, L, TCA, homogeneous_p; prefer_full=false)
            if sim === nothing
                warn_missing && @warn "Missing homogeneous simulation for T* extraction" U=U L=L TCA=TCA p=homogeneous_p
                continue
            end
            ts = extract_tstar(sim)
            if isnan(ts)
                continue
            end
            push!(homo_xs, Float64(U))
            push!(homo_ys, ts)
        end
    end

    if return_values
        plot_data[:homo] = (U_mean=copy(homo_xs), Tstar=copy(homo_ys))
    end

    if !isempty(homo_xs)
        ord_h = sortperm(homo_xs)
        plot!(
            fig,
            homo_xs[ord_h],
            homo_ys[ord_h];
            label="p=1 (homo)",
            m=homogeneous_marker,
            lw=homogeneous_lw,
            ms=homogeneous_ms,
            ls=homogeneous_linestyle,
            color=homogeneous_color,
        )
    elseif sims_homo !== nothing
        warn_missing && @warn "No homogeneous T* points collected" L=L TCA=TCA p=homogeneous_p
    end

    if xlims !== nothing
        plot!(fig; xlims=xlims)
    end
    if ylims !== nothing
        plot!(fig; ylims=ylims)
    end

    return return_values ? (fig, plot_data) : fig
end

"""
Plot Tc and T* on the same axes vs U_mean = U*p, one color per p with two
curves (Tc: solid+circle; T*: dashed+square). Combines the data flow of
`plot_Tc_vs_Umean_by_p` and `plot_Tstar_vs_Umean_by_p`.

Tc extraction follows `tc_method` (`:crossing` or `:fit`, same semantics as
`plot_Tc_vs_Umean_by_p`). T* extraction uses `find_Tstar(sim; len=tstar_len)`.

Shared kwargs match `plot_Tc_vs_Umean_by_p` (`lowest_q`, `prefer_full`,
`only_crossings`, `use_U_raw`, `sims_homo`, `U_homo_list`, `homogeneous_p`,
`homogeneous_lw`, `homogeneous_ms`, `xlims`, `ylims`, `legend_pos`,
`title_extra`, `return_values`, `warn_missing`, `tc_method`, `fit_window`,
`fit_order`).

Combined-plot kwargs:
- `tstar_len`: forwarded to `find_Tstar` (default `3`).
- `tstar_scale`: multiplier applied to every T* value before plotting and
  before being stored in the returned data (default `1.0`). Useful for
  showing e.g. T*/2 alongside Tc on the same axes. Tc curve labels and
  data are unaffected. When `tstar_scale != 1`, the legend entries for the
  T* curves include the prefactor.
- `tc_marker`, `tc_linestyle`: marker/linestyle for the Tc curves
  (default `:circle`, `:solid`).
- `tstar_marker`, `tstar_linestyle`: marker/linestyle for the T* curves
  (default `:square`, `:dash`).
- `homo_tc_marker`, `homo_tstar_marker`: markers for the homogeneous Tc/T*
  overlays (default `:diamond`, `:utriangle`).

When `return_values=true`, `data[p]` holds
`(U_mean, Tc, Tc_err, Tstar)`; the homogeneous overlay is under `:homo`.
Note: `U_mean` is the union of x-values across the two extractors for the
*same* p — Tc-only or T*-only points are kept in the per-curve arrays.
The returned dict therefore uses separate per-curve x arrays:
`(U_mean_tc, Tc, Tc_err, U_mean_tstar, Tstar)`.
"""
function plot_Tc_and_Tstar_vs_Umean_by_p(
    sims_disorder::Dict{String, Any},
    U_list,
    p_list,
    geometry::Tuple{Int, Int};
    lowest_q::Bool=false,
    prefer_full::Bool=true,
    only_crossings::Bool=false,
    use_U_raw::Bool=false,
    sims_homo::Union{Dict{String, Any}, Nothing}=nothing,
    U_homo_list=nothing,
    homogeneous_p::Float64=0.0,
    homo_tc_marker=:diamond,
    homo_tstar_marker=:utriangle,
    homogeneous_linestyle=:dash,
    homogeneous_color=:black,
    homogeneous_lw::Float64=2.0,
    homogeneous_ms::Float64=5.0,
    xlims=nothing,
    ylims=nothing,
    legend_pos=:topleft,
    title_extra::String="",
    return_values::Bool=false,
    warn_missing::Bool=true,
    tc_method::Symbol=:crossing,
    fit_window::Int=3,
    fit_order::Union{Int, Symbol}=:auto,
    tstar_len::Int=3,
    tstar_scale::Real=1.0,
    tc_marker=:circle,
    tc_linestyle=:solid,
    tstar_marker=:square,
    tstar_linestyle=:dash,
    disorder_pairs::Union{Nothing, AbstractVector}=nothing,
)
    pair_allowed = if disorder_pairs === nothing
        (U, p) -> true
    else
        normalized_pairs = [(Float64(t[1]), Float64(t[2])) for t in disorder_pairs]
        (U, p) -> any(pr -> isapprox(pr[1], Float64(U); atol=1e-10) &&
                            isapprox(pr[2], Float64(p); atol=1e-10),
                      normalized_pairs)
    end
    L, TCA = geometry
    x_label_name = use_U_raw ? "U" : "U_mean"
    scale_f = Float64(tstar_scale)
    is_scaled = !isapprox(scale_f, 1.0; atol=1e-12, rtol=0.0)
    tstar_label = is_scaled ? @sprintf("%.4g·T*", scale_f) : "T*"
    title_str = "Tc & $(tstar_label) vs $(x_label_name), L=$(L), TCA=$(TCA)"
    if !isempty(title_extra)
        title_str *= " | " * title_extra
    end

    fig = plot(
        xlabel=use_U_raw ? "U" : "U_mean = U * p",
        ylabel="T",
        title=title_str,
        legend=legend_pos,
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        size=(700, 520),
    )

    plot_data = return_values ? Dict{Any, Any}() : nothing

    is_p_one(p) = (p isa Number) && isapprox(Float64(p), 1.0; atol=1e-8, rtol=0.0)

    extract_tc = if tc_method === :fit
        sim -> find_tc_fit(sim; lowest_q=lowest_q, window=fit_window, order=fit_order)
    elseif tc_method === :crossing
        function (sim)
            Tc, crosses, _ = find_tc(sim; lowest_q=lowest_q)
            return Tc, NaN, crosses
        end
    else
        error("Unknown tc_method=$(tc_method); expected :crossing or :fit")
    end

    function extract_tstar(sim)
        try
            ts, _, worked = find_Tstar(sim; len=tstar_len)
            if worked && ts !== nothing && isfinite(ts)
                return scale_f * Float64(ts)
            end
        catch err
            warn_missing && @warn "find_Tstar failed" exception=err
        end
        return NaN
    end

    homo_tc_xs, homo_tc_ys, homo_tc_es = Float64[], Float64[], Float64[]
    homo_ts_xs, homo_ts_ys = Float64[], Float64[]

    for (pix, p) in enumerate(p_list)
        xs_tc, ys_tc, es_tc = Float64[], Float64[], Float64[]
        xs_ts, ys_ts = Float64[], Float64[]

        for U in U_list
            pair_allowed(U, p) || continue
            sim, _ = get_disorder_sim(sims_disorder, U, L, TCA, p; prefer_full=prefer_full)
            if sim === nothing
                warn_missing && @warn "Missing simulation" U=U L=L TCA=TCA p=p
                continue
            end

            x_val = use_U_raw ? Float64(U) : Float64(U) * Float64(p)

            Tc, Tc_err, crosses = extract_tc(sim)
            if !isnan(Tc) && !(only_crossings && !crosses)
                push!(xs_tc, x_val)
                push!(ys_tc, Tc)
                push!(es_tc, Tc_err)
            end

            ts = extract_tstar(sim)
            if !isnan(ts)
                push!(xs_ts, x_val)
                push!(ys_ts, ts)
            end
        end

        if return_values
            plot_data[p] = (
                U_mean_tc=copy(xs_tc), Tc=copy(ys_tc), Tc_err=copy(es_tc),
                U_mean_tstar=copy(xs_ts), Tstar=copy(ys_ts),
            )
        end

        if isempty(xs_tc) && isempty(xs_ts)
            warn_missing && @warn "No Tc/T* points collected for p" p=p L=L TCA=TCA
            continue
        end

        if is_p_one(p)
            append!(homo_tc_xs, xs_tc); append!(homo_tc_ys, ys_tc); append!(homo_tc_es, es_tc)
            append!(homo_ts_xs, xs_ts); append!(homo_ts_ys, ys_ts)
            continue
        end

        if !isempty(xs_tc)
            ord = sortperm(xs_tc)
            es_ord = es_tc[ord]
            yerr_arg = any(isfinite, es_ord) ? replace(es_ord, NaN => 0.0) : nothing
            plot!(
                fig, xs_tc[ord], ys_tc[ord];
                label=@sprintf("Tc p=%.3g", p),
                m=tc_marker, lw=2, ms=5, color=pix, ls=tc_linestyle,
                yerr=yerr_arg,
            )
        end

        if !isempty(xs_ts)
            ord = sortperm(xs_ts)
            plot!(
                fig, xs_ts[ord], ys_ts[ord];
                label=@sprintf("%s p=%.3g", tstar_label, p),
                m=tstar_marker, lw=2, ms=5, color=pix, ls=tstar_linestyle,
            )
        end
    end

    if sims_homo !== nothing
        homo_Us = U_homo_list === nothing ? U_list : U_homo_list
        for U in homo_Us
            sim, _ = get_disorder_sim(sims_homo, U, L, TCA, homogeneous_p; prefer_full=false)
            if sim === nothing
                warn_missing && @warn "Missing homogeneous simulation" U=U L=L TCA=TCA p=homogeneous_p
                continue
            end
            Tc, Tc_err, crosses = extract_tc(sim)
            if !isnan(Tc) && !(only_crossings && !crosses)
                push!(homo_tc_xs, Float64(U)); push!(homo_tc_ys, Tc); push!(homo_tc_es, Tc_err)
            end
            ts = extract_tstar(sim)
            if !isnan(ts)
                push!(homo_ts_xs, Float64(U)); push!(homo_ts_ys, ts)
            end
        end
    end

    if return_values
        plot_data[:homo] = (
            U_mean_tc=copy(homo_tc_xs), Tc=copy(homo_tc_ys), Tc_err=copy(homo_tc_es),
            U_mean_tstar=copy(homo_ts_xs), Tstar=copy(homo_ts_ys),
        )
    end

    if !isempty(homo_tc_xs)
        ord_h = sortperm(homo_tc_xs)
        es_h = homo_tc_es[ord_h]
        yerr_h = any(isfinite, es_h) ? replace(es_h, NaN => 0.0) : nothing
        plot!(
            fig, homo_tc_xs[ord_h], homo_tc_ys[ord_h];
            label="Tc p=1 (homo)",
            m=homo_tc_marker, lw=homogeneous_lw, ms=homogeneous_ms,
            ls=homogeneous_linestyle, color=homogeneous_color,
            yerr=yerr_h,
        )
    end

    if !isempty(homo_ts_xs)
        ord_h = sortperm(homo_ts_xs)
        plot!(
            fig, homo_ts_xs[ord_h], homo_ts_ys[ord_h];
            label="$(tstar_label) p=1 (homo)",
            m=homo_tstar_marker, lw=homogeneous_lw, ms=homogeneous_ms,
            ls=homogeneous_linestyle, color=homogeneous_color,
        )
    end

    if isempty(homo_tc_xs) && isempty(homo_ts_xs) && sims_homo !== nothing
        warn_missing && @warn "No homogeneous Tc/T* points collected" L=L TCA=TCA p=homogeneous_p
    end

    if xlims !== nothing
        plot!(fig; xlims=xlims)
    end
    if ylims !== nothing
        plot!(fig; ylims=ylims)
    end

    return return_values ? (fig, plot_data) : fig
end
