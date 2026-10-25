include(joinpath(@__DIR__, "..", "..", "data_summery.jl"))

using Gtk

function _parse_float_list(s::String)
    parts = split(strip(s), ',')
    vals = Float64[]
    for p in parts
        t = strip(p)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    return vals
end

function _parse_geometry_pairs(s::String)
    # Format: "8-8, 10-10, 14-10"
    parts = split(strip(s), ',')
    pairs = Tuple{Int, Int}[]
    for p in parts
        t = strip(p)
        isempty(t) && continue
        ab = split(t, '-')
        length(ab) == 2 || error("Geometry '$t' must be in L-TCA format (example: 14-10)")
        push!(pairs, (parse(Int, strip(ab[1])), parse(Int, strip(ab[2]))))
    end
    return pairs
end

function _parse_obs_ix(s::String)
    t = strip(s)
    isempty(t) && return nothing
    lowercase(t) == "nothing" && return nothing
    return parse(Int, t)
end

function _parse_symbol(s::String, allowed::Vector{Symbol})
    sym = Symbol(strip(s))
    sym in allowed || error("Invalid value: $sym. Allowed: $(join(string.(allowed), ", "))")
    return sym
end

function _parse_limits(s::String)
    t = strip(s)
    isempty(t) && return nothing
    parts = split(t, ',')
    length(parts) == 2 || error("Limits must be 'min,max' or empty")
    a = parse(Float64, strip(parts[1]))
    b = parse(Float64, strip(parts[2]))
    a < b || error("Limits must satisfy min < max")
    return (a, b)
end

function _selected_symbol_from_dropdown(combo, allowed::Vector{Symbol})
    idx = get_gtk_property(combo, :active, Int)
    idx >= 0 || error("Please choose a dropdown value")
    return allowed[idx + 1]
end

function _state_from_widgets(
    sim_info,
    u_entry,
    n_entry,
    geo_entry,
    obs_entry,
    obsix_entry,
    xlims_entry,
    ylims_entry,
    layout_dropdown,
    separate_dropdown,
    beta_check,
    prefer_check,
    missing_check,
)
    U_list = _parse_float_list(get_gtk_property(u_entry, :text, String))
    n_list = _parse_float_list(get_gtk_property(n_entry, :text, String))
    geometry_pairs = _parse_geometry_pairs(get_gtk_property(geo_entry, :text, String))
    obs_name = String(strip(get_gtk_property(obs_entry, :text, String)))
    obs_ix = _parse_obs_ix(get_gtk_property(obsix_entry, :text, String))
    xlims = _parse_limits(get_gtk_property(xlims_entry, :text, String))
    ylims = _parse_limits(get_gtk_property(ylims_entry, :text, String))
    layout_mode = _selected_symbol_from_dropdown(layout_dropdown, [:separate, :together])
    separate_by = _selected_symbol_from_dropdown(separate_dropdown, [:n, :U, :both])

    state = create_plot_state(
        sim_info;
        U_list=U_list,
        n_list=n_list,
        geometry_pairs=geometry_pairs,
        obs_name=obs_name,
        obs_ix=obs_ix,
        xlims=xlims,
        ylims=ylims,
        plot_against_beta=get_gtk_property(beta_check, :active, Bool),
        prefer_full=get_gtk_property(prefer_check, :active, Bool),
        show_missing=get_gtk_property(missing_check, :active, Bool),
        layout_mode=layout_mode,
        separate_by=separate_by,
    )
    return state
end

function launch_plot_gui()
    win = GtkWindow("Hubbard Summary Plot GUI", 960, 620)
    grid = GtkGrid()
    set_gtk_property!(grid, :column_spacing, 8)
    set_gtk_property!(grid, :row_spacing, 8)
    push!(win, grid)

    row = Ref(0)
    function add_row(label_txt::String, default_txt::String)
        lbl = GtkLabel(label_txt)
        ent = GtkEntry()
        set_gtk_property!(ent, :text, default_txt)
        grid[0, row[]] = lbl
        grid[1, row[]] = ent
        row[] += 1
        return ent
    end

    function add_dropdown_row(label_txt::String, options::Vector{String}; default_index::Int=1)
        lbl = GtkLabel(label_txt)
        combo = GtkComboBoxText()
        for opt in options
            push!(combo, opt)
        end
        set_gtk_property!(combo, :active, default_index - 1)
        grid[0, row[]] = lbl
        grid[1, row[]] = combo
        row[] += 1
        return combo
    end

    u_entry = add_row("U list (comma):", "2.0, 3.0, 5.0")
    n_entry = add_row("n list (comma):", "0.5, 0.875, 1.0")
    geo_entry = add_row("Geometry pairs L-TCA:", "8-8, 10-10, 12-12, 14-10")
    obs_entry = add_row("Observable (manual):", "Ps")
    obs_options = [
        "Ps",
        "n",
        "stiffness",
        "stiffness_lowest_q",
        "Delta_operator",
        "Delta_MC_abs",
        "conductivity_regular_xx_fixed_extrap",
        "conductivity_dc_xx",
        "resistivity_tensor_fixed_extrap",
        "Drude_weight_fixed_extrap",
        "DOS",
        "custom (type manually above)",
    ]
    obs_default_ix = findfirst(==("Ps"), obs_options)
    obs_dropdown = add_dropdown_row("Observable preset:", obs_options; default_index=(obs_default_ix === nothing ? 1 : obs_default_ix))
    obsix_entry = add_row("obs_ix (empty for nothing):", "1")
    xlims_entry = add_row("xlims (min,max or empty):", "")
    ylims_entry = add_row("ylims (min,max or empty):", "")
    layout_dropdown = add_dropdown_row("layout_mode:", ["separate", "together"]; default_index=1)
    separate_dropdown = add_dropdown_row("separate_by:", ["n", "U", "both"]; default_index=3)

    beta_check = GtkCheckButton("Plot against beta")
    prefer_check = GtkCheckButton("Prefer full when L=TCA")
    missing_check = GtkCheckButton("Show missing warnings")
    set_gtk_property!(beta_check, :active, true)
    set_gtk_property!(prefer_check, :active, true)
    set_gtk_property!(missing_check, :active, false)

    grid[0, row[]] = beta_check
    grid[1, row[]] = prefer_check
    row[] += 1
    grid[0, row[]] = missing_check
    row[] += 1

    load_btn = GtkButton("Load + Prepare Data")
    warm_btn = GtkButton("Warm n Cache")
    render_btn = GtkButton("Render Plot")
    exit_btn = GtkButton("Exit")
    warm_hint_lbl = GtkLabel("Warm n Cache: precompute per-n sims for faster rendering")
    status_lbl = GtkLabel("Status: ready")

    grid[0, row[]] = load_btn
    grid[1, row[]] = warm_btn
    row[] += 1
    grid[1, row[]] = warm_hint_lbl
    row[] += 1
    grid[0, row[]] = render_btn
    grid[1, row[]] = status_lbl
    row[] += 1
    grid[0, row[]] = exit_btn

    sim_info_ref = Ref{Union{Nothing, Dict{String, Any}}}(nothing)
    state_ref = Ref{Union{Nothing, PlotState}}(nothing)

    signal_connect(load_btn, "clicked") do _
        try
            set_gtk_property!(status_lbl, :label, "Status: loading data...")
            sim_info = load_and_prepare_data()
            sim_info_ref[] = sim_info

            try
                state_ref[] = _state_from_widgets(
                    sim_info,
                    u_entry,
                    n_entry,
                    geo_entry,
                    obs_entry,
                    obsix_entry,
                    xlims_entry,
                    ylims_entry,
                    layout_dropdown,
                    separate_dropdown,
                    beta_check,
                    prefer_check,
                    missing_check,
                )
                set_gtk_property!(status_lbl, :label, "Status: data ready")
            catch state_err
                state_ref[] = nothing
                set_gtk_property!(status_lbl, :label, "Status: data loaded, settings parse error - $(state_err)")
            end
        catch err
            set_gtk_property!(status_lbl, :label, "Status: load error - $(err)")
        end
    end

    signal_connect(obs_dropdown, "changed") do w
        idx = get_gtk_property(w, :active, Int)
        idx < 0 && return
        selected = obs_options[idx + 1]
        selected == "custom (type manually above)" && return
        set_gtk_property!(obs_entry, :text, selected)
    end

    signal_connect(warm_btn, "clicked") do _
        try
            sim_info = sim_info_ref[]
            sim_info === nothing && error("Please load data first")

            state = state_ref[]
            if state === nothing
                state = _state_from_widgets(
                    sim_info,
                    u_entry,
                    n_entry,
                    geo_entry,
                    obs_entry,
                    obsix_entry,
                    xlims_entry,
                    ylims_entry,
                    layout_dropdown,
                    separate_dropdown,
                    beta_check,
                    prefer_check,
                    missing_check,
                )
                state_ref[] = state
            end

            set_gtk_property!(status_lbl, :label, "Status: warming cache...")
            warmup_sims_cache!(state)
            set_gtk_property!(status_lbl, :label, "Status: cache warmed")
        catch err
            set_gtk_property!(status_lbl, :label, "Status: warmup error - $(err)")
        end
    end

    signal_connect(render_btn, "clicked") do _
        try
            sim_info = sim_info_ref[]
            sim_info === nothing && error("Please load data first")

            state = _state_from_widgets(
                sim_info,
                u_entry,
                n_entry,
                geo_entry,
                obs_entry,
                obsix_entry,
                xlims_entry,
                ylims_entry,
                layout_dropdown,
                separate_dropdown,
                beta_check,
                prefer_check,
                missing_check,
            )

            # Reuse old cache if available.
            if state_ref[] !== nothing
                state.sims_by_n_cache = state_ref[].sims_by_n_cache
            end
            state_ref[] = state

            set_gtk_property!(status_lbl, :label, "Status: rendering...")
            fig = render_plot(state)
            display(fig)
            set_gtk_property!(status_lbl, :label, "Status: rendered")
        catch err
            set_gtk_property!(status_lbl, :label, "Status: render error - $(err)")
        end
    end

    signal_connect(exit_btn, "clicked") do _
        destroy(win)
    end

    showall(win)
    return win
end

launch_plot_gui()