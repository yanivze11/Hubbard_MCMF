"""
Compare two ways of getting the Drude weight D = ⟨-K_a⟩ - Re[Λ_aa(q=0, ω→0)]:

  1. DIRECT method   — the closed-form ω→0, δ→0 observable L_D_direct_* :
         Re[Λ_aa(q=0, ω→0, δ→0)] = -(t²/N) Σ_{ε_λ≠ε_μ} (f_λ-f_μ)/(ε_λ-ε_μ) |G^a_{λμ}|²
     giving  drude_direct_aa = -π(⟨K_a⟩ + ⟨L_D_direct_aa⟩).

  2. δ→0 EXTRAPOLATION method — Re[Λ_aa(ω_min, δ)] measured for a grid of δ values
     (from Lxx_w_fixed / Lyy_w_fixed at the smallest ω), then linearly extrapolated
     to δ→0, giving  drude_extrap_aa = -π(⟨K_a⟩ + Re[Λ_aa(ω_min, δ→0)]).

If the direct observable is a good Drude measure, the two should agree (within error),
which would let us stop storing the bulky ω×δ Lxx_w_fixed data.

Run:  julia tests/test_drude_direct_vs_extrap.jl
"""

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))  # pulls in MC_Hubburd.jl + get_means
using Plots
using Statistics

# ============================================================
# Simulation parameters
# ============================================================
Lx = 10
Ly = 10
beta = 20.0
mu = -0.5          # away from half-filling to get finite conductivity
B = 0.0  # small flux to break degeneracies
U = 0.0            # non-interacting: clean, deterministic check
num_measure = 1
num_thermal = 0
measuring_interval = 1

# Mean level spacing: bandwidth ~ 8t, 2N states → mls ≈ 4/(Lx*Ly)
mls = 4.0 / (Lx * Ly)
multipliers = collect(1:20)

# δ grid for the extrapolation method (1..20 × mls) and the small probe ω.
# Drude_weight_fixed reads the FIRST entry of w_conductivity_range, so keep it tiny.
delta_values = multipliers .* mls
w_conductivity_range = [0.01, 0.1, 1.0] .* mls
dw_values = [5.0 * mls]      # minimal; not used in this comparison
w_list = [-3.0, 3.0]

# ============================================================
# Run simulation
# ============================================================
println("Running U=$(U) simulation on $(Lx)×$(Ly), β=$(beta), μ=$(mu)...")
println("  Mean level spacing (mls) = $(round(mls, sigdigits=4))")
data = running_simulation_once(
    Lx, Ly, 1,            # Lx, Ly, SC_L (=1)
    U, 1.0, 0.0,          # U, t=1, V=0
    12345,                # seed
    mu, beta, B,          # mu, beta, B
    num_measure, num_thermal, measuring_interval,
    w_list,
    "uniform", "uniform", "K0", "s-wave",
    w_conductivity_range,
    0.0;                  # U_impurity
    savedata=false, saveplots=false,
    delta_values_fixed=delta_values,
    delta_values_stiffness=Float64[],
    dw_values=dw_values,
    loop_random=false,
    gauge_choice="Mixed",
    boundary_conditions="PBC",
)

println("  Filling: n = $(round(avrage_n(data), digits=4))")

# ============================================================
# Means (drives both the direct-Drude derivation and the δ→0 extrapolation)
# ============================================================
means = get_means(data)
meas  = data.measurement_data
MLS   = mls

# --- Direct method (from means_dict) ---
D_direct_xx = means["drude_direct_xx"]["value"]
D_direct_yy = means["drude_direct_yy"]["value"]
D_direct_xx_err = means["drude_direct_xx"]["error"]
D_direct_yy_err = means["drude_direct_yy"]["error"]

# --- δ→0 extrapolation method ---
# xx: pipeline already produced Drude_weight_fixed (per δ) + _extrap (δ→0).
Dfix_xx_per_delta = Float64.(means["Drude_weight_fixed"]["value"])   # D_xx(δ)
D_extrap_xx     = means["Drude_weight_fixed_extrap"]["value"]
D_extrap_xx_err = means["Drude_weight_fixed_extrap"]["error"]

# yy: build the same D_yy(δ) curve from Lyy_w_fixed at the smallest ω, then extrapolate.
kx = meas.Kx
ky = meas.Ky
Kx_mean = mean(kx)
Ky_mean = mean(ky)
Dfix_yy_per_delta = Float64[]
Dfix_yy_err       = Float64[]
for d_idx in eachindex(delta_values)
    lyy_d = [real(m[d_idx][1]) for m in meas.Lyy_w_fixed]  # Re[Λ_yy(ω_min, δ)] per step
    (ky_m, ky_v), (lyy_m, lyy_v), covv = get_obs_cov(ky, lyy_d)
    push!(Dfix_yy_per_delta, -pi * (ky_m + lyy_m))
    e2 = ky_v + lyy_v + 2*covv; e2 < 0 && (e2 = 0.0)
    push!(Dfix_yy_err, pi * sqrt(e2))
end
D_extrap_yy, D_extrap_yy_err, _ = extrapolate_delta_to_zero(
    Float64.(delta_values), Dfix_yy_per_delta, Dfix_yy_err; MLS=MLS, min_delta_MLS=1.0)

# ============================================================
# Report
# ============================================================
relerr(a, b) = abs(a - b) / max(abs(a), abs(b), eps())
println("\n================ Drude weight: direct vs δ→0 extrapolation ================")
for (lbl, Ddir, Ddir_e, Dext, Dext_e) in [
        ("xx", D_direct_xx, D_direct_xx_err, D_extrap_xx, D_extrap_xx_err),
        ("yy", D_direct_yy, D_direct_yy_err, D_extrap_yy, D_extrap_yy_err)]
    println("  D_$lbl  direct     = $(round(Ddir, sigdigits=6))  ± $(round(Ddir_e, sigdigits=3))")
    println("  D_$lbl  δ→0 extrap = $(round(Dext, sigdigits=6))  ± $(round(Dext_e, sigdigits=3))")
    println("  D_$lbl  rel. diff  = $(round(100*relerr(Ddir, Dext), sigdigits=3)) %")
    println("  ---")
end
println("Note: Re[Λ_aa(q=0, ω=0)] = -⟨K_a⟩ is an exact Ward identity, so in a FINITE")
println("system the direct ω=0,δ=0 evaluation gives D≈0. A physical Drude weight needs")
println("broadening δ > mean level spacing (the δ→0-from-large-δ extrapolation).")

# ============================================================
# Plot: D(δ) curve + δ→0 intercept + direct horizontal line
# ============================================================
p = plot(title="Drude weight: direct vs δ→0 extrap ($(Lx)×$(Ly), β=$(beta), μ=$(mu))",
    xlabel="δ / mls", ylabel="Drude weight D", legend=:best)

plot!(p, multipliers, Dfix_xx_per_delta, marker=:circle, color=:blue,
    label="D_xx(δ)  [fixed-δ]")
scatter!(p, [0.0], [D_extrap_xx], marker=(:star5, 8), color=:blue,
    label="D_xx δ→0 extrap")
hline!(p, [D_direct_xx], color=:blue, linestyle=:dash, label="D_xx direct (ω=δ=0)")

plot!(p, multipliers, Dfix_yy_per_delta, marker=:diamond, color=:red,
    label="D_yy(δ)  [fixed-δ]")
scatter!(p, [0.0], [D_extrap_yy], marker=(:star5, 8), color=:red,
    label="D_yy δ→0 extrap")
hline!(p, [D_direct_yy], color=:red, linestyle=:dash, label="D_yy direct (ω=δ=0)")

outpath = joinpath(@__DIR__, "drude_direct_vs_extrap.png")
savefig(p, outpath)
println("\nSaved comparison plot to: $outpath")
