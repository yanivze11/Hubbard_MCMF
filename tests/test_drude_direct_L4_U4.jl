"""
Thermalized MC comparison (interacting) of the two Drude-weight estimators, on a
small L=4, U=4 system that can be run locally.

For L_D = Re[Λ_xx] we plot, WITH MC error bars:
  * L_D(δ)          — the fixed-δ current-current correlator vs δ (Lxx_w_fixed, ω_min slot)
  * δ→0 extrap      — linear extrapolation of L_D(δ) to δ=0
  * direct (ω=δ=0)  — the closed-form L_D_direct_xx observable

and report the derived Drude weight D = -π(⟨Kx⟩ + L_D) for both estimators.

Run:  julia tests/test_drude_direct_L4_U4.jl
"""

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
using Plots, Statistics, Printf

# ---------------- parameters ----------------
L      = 4
U      = 4.0
t      = 1.0
mu     = -0.5
beta   = 10.0
B      = 0.0
V      = 0.0        # clean; interaction U supplies the sampled fluctuations
seed   = 12345

# MC schedule (small system → can afford generous thermalization + sampling locally)
num_thermal        = 5000
num_measure        = 3000
measuring_interval = 10

mls = 4.0 / (L * L)
multipliers  = collect(1:20)
delta_values = multipliers .* mls
w_conductivity_range = [0.01, 0.1, 1.0] .* mls   # ω_min = first slot (used by L_D)

println("Thermalized MC: L=$L, U=$U, β=$beta, μ=$mu, B=$B, V=$V")
println("  thermal=$num_thermal, measure=$num_measure, interval=$measuring_interval")
println("  mls = $(round(mls, sigdigits=4))")

data = running_simulation_once(
    L, L, 1, U, t, V, seed, mu, beta, B,
    num_measure, num_thermal, measuring_interval,
    [-3.0, 3.0], "uniform", "uniform", "K0", "s-wave",
    w_conductivity_range, 0.0;
    savedata=false, saveplots=false,
    delta_values_fixed=delta_values, delta_values_stiffness=Float64[],
    dw_values=[5.0*mls], loop_random=true, gauge_choice="Mixed",
    boundary_conditions="PBC")

ncm = data.current_state.num_current_measurments
meas = data.measurement_data
println("  actual measurements = $ncm,  filling n = $(round(avrage_n(data), digits=4))")

# ---------------- L_D(δ): mean + error per δ ----------------
LD_delta  = Float64[]
LD_err    = Float64[]
for d in eachindex(delta_values)
    vals = [real(m[d][1]) for m in meas.Lxx_w_fixed[1:ncm]]   # Re[Λ_xx(ω_min, δ)]
    m, e = observable_mean(vals, ncm; std_scheme="correlation")
    push!(LD_delta, m); push!(LD_err, e)
end

# ---------------- δ→0 extrapolation of L_D(δ) ----------------
LD_extrap, LD_extrap_err, _ = extrapolate_delta_to_zero(
    Float64.(delta_values), LD_delta, LD_err; MLS=mls, min_delta_MLS=1.0)

# ---------------- direct ω=0, δ=0 value ----------------
LD_direct, LD_direct_err = observable_mean(meas.L_D_direct_xx[1:ncm], ncm; std_scheme="correlation")

# ---------------- derived Drude weights (via pipeline) ----------------
means = get_means(data)
D_direct = means["drude_direct_xx"]["value"]
D_direct_e = means["drude_direct_xx"]["error"]
D_extrap = means["Drude_weight_fixed_extrap"]["value"]
D_extrap_e = means["Drude_weight_fixed_extrap"]["error"]

println("\n===== L=4, U=4 thermalized MC: direct vs δ→0 extrapolation =====")
@printf("  L_D  direct      = %.5f ± %.2g\n", LD_direct, LD_direct_err)
@printf("  L_D  δ→0 extrap  = %.5f ± %.2g\n", LD_extrap, LD_extrap_err)
@printf("  Drude D direct   = %.5f ± %.2g\n", D_direct, D_direct_e)
@printf("  Drude D δ→0 extr = %.5f ± %.2g\n", D_extrap, D_extrap_e)
@printf("  rel. diff (D)    = %.2f %%\n", 100*abs(D_direct-D_extrap)/max(abs(D_direct),abs(D_extrap),eps()))

# ---------------- plot ----------------
p = plot(multipliers, LD_delta, yerror=LD_err, marker=:circle, ms=4, color=:steelblue,
    lw=1.5, label="L_D(δ)  [fixed-δ]",
    title="L=4, U=4, β=$beta, μ=$mu  (thermalized MC, $ncm meas.)",
    titlefontsize=10, xlabel="δ / mls", ylabel="L_D = Re[Λ_xx]", legend=:best)
scatter!(p, [0.0], [LD_extrap], yerror=[LD_extrap_err], marker=(:star5, 10), color=:orange,
    label=@sprintf("δ→0 extrap = %.4f ± %.1g", LD_extrap, LD_extrap_err))
hline!(p, [LD_direct], color=:crimson, ls=:dash, lw=2,
    label=@sprintf("direct (ω=δ=0) = %.4f ± %.1g", LD_direct, LD_direct_err))
# shade direct error band
hspan!(p, [LD_direct-LD_direct_err, LD_direct+LD_direct_err], color=:crimson, alpha=0.12, label="")

out = joinpath(@__DIR__, "LD_delta_L4_U4.png")
savefig(p, out)
println("\nSaved plot to: $out")
