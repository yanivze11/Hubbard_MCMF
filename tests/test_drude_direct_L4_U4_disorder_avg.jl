"""
Disorder-averaged version of test_drude_direct_L4_U4.jl.

Runs the thermalized L=4, U=4 MC over several potential-disorder configurations
(different disorder_seed → different V_i ~ Uniform[-V/2, V/2]) and compares the
disorder-averaged Drude weight D from the two estimators:
  * direct  ω=δ=0  (drude_direct_xx)
  * δ→0 extrapolation (Drude_weight_fixed_extrap)

Reports the paired per-config difference (D_direct - D_extrap), which cancels
config-to-config scatter and isolates the systematic between the two methods.
Plots the disorder-averaged D(δ) curve with realization-spread error bars, plus the
averaged extrapolated and direct values.

Run:  julia tests/test_drude_direct_L4_U4_disorder_avg.jl
"""

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
using Plots, Statistics, Printf

# ---------------- parameters ----------------
L, U, t, mu, beta, B = 4, 4.0, 1.0, -0.5, 10.0, 0.0
V      = 2.0                 # potential-disorder strength: V_i ~ Uniform[-V/2, V/2]
seed   = 12345               # MC seed (shared; disorder varies via disorder_seed)
dseeds = collect(1:8)        # disorder configurations to average over

num_thermal, num_measure, measuring_interval = 5000, 1500, 10

mls = 4.0 / (L * L)
multipliers  = collect(1:20)
delta_values = multipliers .* mls
w_conductivity_range = [0.01, 0.1, 1.0] .* mls

println("Disorder-averaged L=$L, U=$U, V=$V, β=$beta, μ=$mu, B=$B")
println("  $(length(dseeds)) configs, thermal=$num_thermal, measure=$num_measure, interval=$measuring_interval\n")

D_direct = Float64[]
D_extrap = Float64[]
Dcurves  = Vector{Vector{Float64}}()   # D(δ) per config

for ds in dseeds
    data = running_simulation_once(
        L, L, 1, U, t, V, seed, mu, beta, B,
        num_measure, num_thermal, measuring_interval,
        [-3.0, 3.0], "uniform", "uniform", "K0", "s-wave",
        w_conductivity_range, 0.0;
        savedata=false, saveplots=false,
        delta_values_fixed=delta_values, delta_values_stiffness=Float64[],
        dw_values=[5.0*mls], loop_random=true, gauge_choice="Mixed",
        boundary_conditions="PBC", disorder_seed=ds)
    means = get_means(data)
    dd = means["drude_direct_xx"]["value"]
    de = means["Drude_weight_fixed_extrap"]["value"]
    push!(D_direct, dd); push!(D_extrap, de)
    push!(Dcurves, Float64.(means["Drude_weight_fixed"]["value"]))
    @printf("  dseed=%d  n=%.3f  D_direct=%.5f  D_extrap=%.5f  diff=%+.5f\n",
        ds, avrage_n(data), dd, de, dd-de)
end

n = length(dseeds)
sem(x) = std(x) / sqrt(length(x))
dir_m, dir_e = mean(D_direct), sem(D_direct)
ext_m, ext_e = mean(D_extrap), sem(D_extrap)
paired = D_direct .- D_extrap
pair_m, pair_e = mean(paired), sem(paired)

println("\n===== disorder-averaged over $n configs (V=$V) =====")
@printf("  <D_direct>          = %.5f ± %.5f\n", dir_m, dir_e)
@printf("  <D_extrap>          = %.5f ± %.5f\n", ext_m, ext_e)
@printf("  <D_direct - D_extrap> (paired) = %+.5f ± %.5f\n", pair_m, pair_e)
@printf("  paired diff in σ    = %.2f σ from 0\n", abs(pair_m)/max(pair_e, eps()))

# disorder-averaged D(δ) curve + realization spread
Dmat = reduce(hcat, Dcurves)                 # [n_delta x n_config]
Dδ_m = vec(mean(Dmat, dims=2))
Dδ_e = vec(std(Dmat, dims=2)) ./ sqrt(n)

p = plot(multipliers, Dδ_m, yerror=Dδ_e, marker=:circle, ms=4, color=:steelblue, lw=1.5,
    label="⟨D(δ)⟩  [fixed-δ]",
    title="L=4, U=4, V=$V, β=$beta, μ=$mu  (disorder-averaged, $n configs)",
    titlefontsize=10, xlabel="δ / mls", ylabel="Drude weight D", legend=:best)
# faint per-config curves
for c in Dcurves
    plot!(p, multipliers, c, color=:gray, alpha=0.25, lw=0.8, label="")
end
scatter!(p, [0.0], [ext_m], yerror=[ext_e], marker=(:star5, 10), color=:orange,
    label=@sprintf("δ→0 extrap = %.4f ± %.4f", ext_m, ext_e))
hline!(p, [dir_m], color=:crimson, ls=:dash, lw=2,
    label=@sprintf("direct (ω=δ=0) = %.4f ± %.4f", dir_m, dir_e))
hspan!(p, [dir_m-dir_e, dir_m+dir_e], color=:crimson, alpha=0.12, label="")

out = joinpath(@__DIR__, "D_delta_L4_U4_disorder_avg.png")
savefig(p, out)
println("\nSaved plot to: $out")
