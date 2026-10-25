include(joinpath(@__DIR__, "..", "..", "final_plots.jl"))

csv = joinpath(FINAL_PLOTS_DIR, "Delta_vs_MF_U4.0_n0.5.csv")

# Default — same look as the auto-saved PNG, just from the CSV.
fig = replot_Delta_vs_MF(csv)

# Tweak everything in one call:
fig = replot_Delta_vs_MF(csv;
    xlims=(-0.0, 0.8),                # zoom into the transition window
    ylims=(-0.05, 1.5),
    markersize=5,
    linewidth=2,
    legend_pos=:topright,
    title="\$U=4, n=0.5\$",
    save_to=joinpath(FINAL_PLOTS_DIR, "Delta_vs_MF_U4.0_n0.5_polished.png"),
)

# Restrict to a subset of series:
fig = replot_Delta_vs_MF(csv; series=["Delta_c","Delta_MF"])