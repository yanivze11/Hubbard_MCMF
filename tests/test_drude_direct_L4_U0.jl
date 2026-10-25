"""
Companion to test_drude_direct_L4_U4.jl but for U=0 (non-interacting, single exact
configuration) and plotting the DRUDE WEIGHT D(δ) = -π(⟨Kx⟩ + Re[Λ_xx(ω_min,δ)])
rather than L_D. Overlays the δ→0 extrapolated D and the direct ω=δ=0 D.

Run:  julia tests/test_drude_direct_L4_U0.jl
"""

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
using Plots, Statistics, Printf

L, U, t, mu, beta, B, V, seed = 4, 0.0, 1.0, -0.5, 10.0, 0.0, 0.0, 12345
mls = 4.0 / (L * L)
multipliers  = collect(1:20)
delta_values = multipliers .* mls
w_conductivity_range = [0.01, 0.1, 1.0] .* mls

println("Non-interacting (U=0) single config: L=$L, β=$beta, μ=$mu, B=$B")
data = running_simulation_once(
    L, L, 1, U, t, V, seed, mu, beta, B,
    1, 0, 1, [-3.0, 3.0], "uniform", "uniform", "K0", "s-wave",
    w_conductivity_range, 0.0;
    savedata=false, saveplots=false,
    delta_values_fixed=delta_values, delta_values_stiffness=Float64[],
    dw_values=[5.0*mls], loop_random=false, gauge_choice="Mixed",
    boundary_conditions="PBC")
println("  filling n = $(round(avrage_n(data), digits=4))")

means = get_means(data)
D_delta  = Float64.(means["Drude_weight_fixed"]["value"])      # D_xx(δ)
D_extrap = means["Drude_weight_fixed_extrap"]["value"]         # δ→0
D_direct = means["drude_direct_xx"]["value"]                   # ω=δ=0

println("\n===== L=4, U=0: Drude weight, direct vs δ→0 extrapolation =====")
@printf("  D direct     = %.5f\n", D_direct)
@printf("  D δ→0 extrap = %.5f\n", D_extrap)
@printf("  rel. diff    = %.3g %%\n",
    100*abs(D_direct-D_extrap)/max(abs(D_direct),abs(D_extrap),eps()))

yc = D_direct
p = plot(multipliers, D_delta, marker=:circle, ms=4, color=:steelblue, lw=1.5,
    label="D(δ)  [fixed-δ]", ylims=(yc-0.5, yc+0.5),
    title="L=4, U=0, β=$beta, μ=$mu  (non-interacting; D(δ) is δ-independent)",
    titlefontsize=10, xlabel="δ / mls", ylabel="Drude weight D", legend=:best)
scatter!(p, [0.0], [D_extrap], marker=(:star5, 10), color=:orange,
    label=@sprintf("δ→0 extrap = %.4f", D_extrap))
hline!(p, [D_direct], color=:crimson, ls=:dash, lw=2,
    label=@sprintf("direct (ω=δ=0) = %.4f", D_direct))

out = joinpath(@__DIR__, "D_delta_L4_U0.png")
savefig(p, out)
println("\nSaved plot to: $out")
