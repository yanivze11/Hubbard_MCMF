# plot_dos_gaussian_overlay.jl — fig:DOS_gaussian overlay.
# Model DOS (Gaussian-sampled Δ, gaussian_model_results/sim_info_gaussian_model.jld2)
# vs the measured MCMF DOS (real run, means store) at U=4, n=0.875, L=16, for
# T = 0.5, 1, 2, 4. Both DOS are μ-relative (Fermi level at ω=0) → directly comparable.
# Expectation: agreement improves at HIGH T (Gaussian/uncorrelated-field limit).

include(joinpath(@__DIR__, "..", "..", "data_summery.jl"))   # get_sim_dimensions etc.
include(joinpath(@__DIR__, "..", "..", "means_store.jl"))     # load_subset
using JLD2, Printf, Plots

const U   = 4.0
const N   = 0.875
const L   = 16
const TS  = [0.5, 1.0, 2.0, 4.0]
const MODEL_FILE = joinpath(@__DIR__, "gaussian_model_results", "sim_info_gaussian_model.jld2")
const MEAS_SOURCE = "sim_info_FullSys_TCA__27-03-2026_CURIE.jld2"
const OUT_PNG = joinpath(@__DIR__, "gaussian_model_results", "DOS_gaussian_overlay.png")
const OUT_CSV = joinpath(@__DIR__, "gaussian_model_results", "DOS_gaussian_overlay.csv")

# Collect T => (w_list, DOS) from a sim_info-shaped dict, keeping the entry whose
# filling is closest to N at each beta (matches U & L).
function collect_dos(si; Uv=U, Lv=L, Nv=N)
    best = Dict{Float64,Tuple{Float64,Vector{Float64},Vector{Float64}}}()  # beta=>(ndist,w,dos)
    for (k, e) in si
        k == "updated_at" && continue
        (e isa AbstractDict) || continue
        get(e, "U", nothing) == Uv || continue
        Lx, Ly = get_sim_dimensions(e); (Lx == Lv) || continue
        m = get(e, "means", nothing)
        (m isa AbstractDict && haskey(m, "DOS") && haskey(m, "n")) || continue
        (m["DOS"] isa AbstractDict && m["DOS"]["value"] isa AbstractVector) || continue
        beta = Float64(e["beta"]); nv = Float64(m["n"]["value"]); nd = abs(nv - Nv)
        if !haskey(best, beta) || nd < best[beta][1]
            best[beta] = (nd, Float64.(e["w_list"]), Float64.(m["DOS"]["value"]))
        end
    end
    return Dict(1/b => (v[2], v[3]) for (b, v) in best)
end

nearestT(d::Dict, T) = (ks = collect(keys(d)); ks[argmin(abs.(ks .- T))])

println("[overlay] loading model DOS: $MODEL_FILE"); flush(stdout)
model = collect_dos(JLD2.load(MODEL_FILE))
@printf("  model T available: %s\n", string(sort(collect(keys(model))))); flush(stdout)

println("[overlay] loading measured DOS from means store (U=$U, L=$L, n=$N) ..."); flush(stdout)
si_meas = load_subset(; source=[MEAS_SOURCE], U=U, n=N, run_extrapolation=false)
meas = collect_dos(si_meas)
@printf("  measured T available: %s\n", string(sort(round.(collect(keys(meas)); digits=3)))); flush(stdout)

# ---- plot: single panel, one color per T (model solid, measured dashed) ----
COLORS = Dict(0.5 => :dodgerblue, 1.0 => :seagreen, 2.0 => :darkorange, 4.0 => :firebrick)
plt = plot(size=(950,680), legendfontsize=15, guidefontsize=19, tickfontsize=15,
           left_margin=6Plots.mm, bottom_margin=6Plots.mm,
           title="DOS: Gaussian model vs measured MCMF  (U=$U, n=$N, L=$L)", titlefontsize=16,
           xlabel="ω", ylabel="ρ(ω)", xlims=(-5, 5))
csv = open(OUT_CSV, "w"); println(csv, "T,source,omega,dos")
for T in TS
    c = COLORS[T]
    wM, dM = model[nearestT(model, T)]
    plot!(plt, wM, dM; label="T = $T", lw=2.4, color=c)                 # model (solid) -> color/T legend
    for (w, d) in zip(wM, dM); println(csv, "$T,model,$w,$d"); end
    Tm = nearestT(meas, T)
    if abs(Tm - T) < 0.15 * max(T, 1)
        wD, dD = meas[Tm]
        plot!(plt, wD, dD; label=false, lw=2.0, ls=:dash, color=c)       # measured (dashed, same color)
        for (w, d) in zip(wD, dD); println(csv, "$T,measured,$w,$d"); end
    end
end
# style key (dummy black series)
plot!(plt, [NaN], [NaN]; label="model",    lw=2.4, color=:black)
plot!(plt, [NaN], [NaN]; label="measured", lw=2.0, ls=:dash, color=:black)
plot!(plt; legend=:topright)
close(csv)
savefig(plt, OUT_PNG)
@printf("[overlay] saved:\n  PNG: %s\n  CSV: %s\n", OUT_PNG, OUT_CSV)
