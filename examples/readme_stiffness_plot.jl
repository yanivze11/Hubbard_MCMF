# ============================================================================
# examples/readme_stiffness_plot.jl — the README figure: superfluid stiffness
# vs temperature for several U at one lattice size, from the archived V7 means
# (backup drive).
#
#   BACKUP_DRIVE='<where the drive is>/Yaniv-Data' julia examples/readme_stiffness_plot.jl
#
# (or LANDAU_ROOT / CURIE_ROOT / LANDAU_MOUNT / CURIE_MOUNT, docs/WORKFLOW.md §8.2)
#
# Reads only means_db (no slims); writes docs/images/stiffness_vs_T_L20.png.
# Clean system (V = 0), L = 20, n = 0.9, pairing-only decoupling (alpha = 1,
# beta_phi = 0), "K0-TCA" with cluster size 8.
# ============================================================================
include(joinpath(@__DIR__, "..", "means_db.jl"))
isempty(MEANS_DB_ROOTS) && error("set BACKUP_DRIVE (or LANDAU_ROOT/CURIE_ROOT) to where the data is")
using Plots

const L, N = 20, 0.9
const US = [2.0, 4.0, 6.0, 8.0, 10.0]
const TMAX = 0.25

plt = plot(; xlabel="T / t", ylabel="ρₛ / t", title="Superfluid stiffness, L = $L, n = $N",
           xlims=(0, TMAX), ylims=(0, 0.2), legend=:topright, size=(640, 430), dpi=150,
           framestyle=:box, left_margin=3Plots.mm, bottom_margin=3Plots.mm)
for U in US
    c = get_curve(; obs=:stiffness, x=:beta, U=U, Lx=L, Ly=L, V=0.0, wanted_n=N, alpha=1.0,
                    beta_phi=0.0, diag_method="K0-TCA", TClustersize=8, mu_disorder="none")
    T = 1 ./ c.x
    p = sortperm(T)
    plot!(plt, T[p], c.value[p]; yerr=c.error[p], marker=:circle, ms=3, lw=1.5,
          label="U = $(round(Int, U))t")
end
Tl = range(0, TMAX; length=50)
plot!(plt, Tl, 2 .* Tl ./ π; ls=:dash, color=:black, lw=1.2, label="2T/π (BKT)")

out = joinpath(@__DIR__, "..", "docs", "images", "stiffness_vs_T_L$L.png")
mkpath(dirname(out))
savefig(plt, out)
println("wrote ", normpath(out))
