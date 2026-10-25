# Smoke test for final_plots.jl — runs `plot_Delta_vs_MF` for F5.1 (U=4, n=0.875).

include(joinpath(@__DIR__, "..", "..", "final_plots.jl"))

const CR = joinpath(@__DIR__, "cluster_results")

clean_paths = [
    joinpath(CR, "Clean_System", "sim_info_FullSys_TCA__27-03-2026_CURIE.jld2"),       # L=8..20, U={3,4,5}, n={0.5,0.875}
    joinpath(CR, "Clean_System", "sim_info_Full_TCA_Bigger__10-04-2026_CURIE.jld2"),   # L=22,24,26,28, U=4, n={0.5,0.875}
]
mf_paths = [
    joinpath(CR, "mean_field", "sim_info_mf.jld2"),                                    # MF L=10,16,18,20,26 at U=4
]

println(">>> Loading MCMF sim_info ...")
@time sim_info = load_and_prepare_data(file_paths=clean_paths)

println(">>> Loading MF sim_info ...")
@time mf_sim_info = load_and_prepare_data(file_paths=mf_paths)

println(">>> Plotting Δ vs MF ...")
@time plot_Delta_vs_MF(;
    sim_info       = sim_info,
    mf_sim_info    = mf_sim_info,
    u              = 4.0,
    n              = 0.5,
    geometries     = [(10,10), (20,10), (26,10)],
    # mf_geometries  defaults to [(L,L) for (L,_) in geometries] = [(10,10),(20,20),(26,26)]
)

println(">>> Done.")
