include(joinpath(@__DIR__, "..", "..", "final_plots.jl"))

csv = joinpath(FINAL_PLOTS_DIR, "Delta_vs_MF_U4.0_n0.5.csv")
out = joinpath(FINAL_PLOTS_DIR, "Delta_vs_MF_U4.0_n0.5_v2.png")

println(">>> Re-rendering from CSV ...")
replot_Delta_vs_MF(csv; save_to=out)
println(">>> Done.")
