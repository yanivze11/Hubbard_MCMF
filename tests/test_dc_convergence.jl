"""
Test DC conductivity and Drude weight convergence: fixed-δ method vs Δω-integral method.

Runs a U=0 (non-interacting) simulation and compares:
  1. Regular conductivity σ_reg = Im[Λ(ω+iδ)] / ω  at small ω, as function of δ
  2. DC conductivity from the Δω integral, as function of Δω
  3. Drude weight Re[Λ(ω→0, δ)]  as function of δ
  4. Drude weight from the Δω integral, as function of Δω

Both δ and Δω are set to multiples 1–10 of the mean level spacing mls = 4/(Lx*Ly).
If both methods are consistent, the limits should converge to the same value.
"""

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
using Plots
using Statistics

# ============================================================
# Simulation parameters
# ============================================================
Lx = 12
Ly = 13
beta = 10.0
mu = -0.5          # away from half-filling to get finite conductivity
B = 2 * 2pi / (Lx * Ly)  # small flux to break degeneracies and get finite conductivity
U = 0.0
num_measure = 1
num_thermal = 0
measuring_interval = 1

# Mean level spacing: bandwidth ~ 8t, 2N states → mls ≈ 4/(Lx*Ly)
mls = 4.0 / (Lx * Ly)
multipliers = collect(1:20)

# δ and Δω values = 1:10 × mls
delta_values = multipliers .* mls
dw_values    = multipliers .* mls

# Small ω for the fixed-δ method (use 1×mls as the probe frequency)
w_conductivity_range = [0.01, 0.1, 1.0] .* mls

w_list = [-3.0, 3.0]

# ============================================================
# Run simulation
# ============================================================
println("Running U=$(U) simulation on $(Lx)×$(Ly) lattice, β=$(beta), μ=$(mu)...")
println("  Mean level spacing (mls) = $(round(mls, sigdigits=4))")
data = running_simulation_once(
    Lx, Ly, 1,           # Lx, Ly, SC_L
    U, 1.0, 0.0,       # U, t=1, V=0
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
    boundary_conditions="PBC"
)

println("  Filling: n = $(round(avrage_n(data), digits=4))")
println("  Energy spacing (mean_diff_low δ): $(round(data.temporary_data.delta, digits=6))")

## ============================================================
# Extract results
# ============================================================
meas = data.measurement_data
i = 1  # single measurement

# --- Fixed-δ method ---
w_smallest_idx = 1
w_smallest = w_conductivity_range[w_smallest_idx]
num_meas = length(meas.Lxx_w_fixed)

sigma_reg_vs_delta_xx = Float64[]
sigma_reg_vs_delta_xy = Float64[]
sigma_reg_vs_delta_xx_err = Float64[]
sigma_reg_vs_delta_xy_err = Float64[]
drude_vs_delta_xx = Float64[]
drude_vs_delta_xy = Float64[]
drude_vs_delta_xx_err = Float64[]
drude_vs_delta_xy_err = Float64[]

for d_idx in eachindex(delta_values)

    Lxx = mean(meas.Lxx_w_fixed)[d_idx][w_smallest_idx]
    Lxy = mean(meas.Lxy_w_fixed)[d_idx][w_smallest_idx]
    
    push!(sigma_reg_vs_delta_xx, imag(Lxx) / w_smallest)
    push!(sigma_reg_vs_delta_xy, imag(Lxy) / w_smallest)
    
    push!(drude_vs_delta_xx, real(Lxx))
    push!(drude_vs_delta_xy, real(Lxy))
end

# --- Δω integral method ---
sigma_dc_vs_dw_xx = mean(meas.conductivity_dc_xx)
sigma_dc_vs_dw_xy = mean(meas.conductivity_dc_xy)

drude_dc_vs_dw_xx = mean(meas.L_D_dc_xx)
drude_dc_vs_dw_xy = mean(meas.L_D_dc_xy)

# ============================================================
# Plot (x-axis in units of mls)
# ============================================================
println("\nCreating convergence plots...")

# --- Plot 1: Regular conductivity σ_xx ---
p1 = plot(title="σ_xx^reg DC convergence ($(Lx)×$(Ly), β=$(beta), μ=$(mu))",
    xlabel="δ/mls  or  Δω/mls", ylabel="σ_xx^reg", legend=:topright)
plot!(p1, multipliers, sigma_reg_vs_delta_xx, label="Im[Λ_xx(ω=mls, δ)]/ω  (δ method)", 
    marker=:circle, color=:blue, linewidth=2)
plot!(p1, multipliers, sigma_dc_vs_dw_xx, label="σ_xx^dc(Δω)  [integral]", 
    marker=:square, color=:red, linewidth=2)

# --- Plot 2: Drude weight Λ_D,xx ---
p2 = plot(title="Drude weight Λ_D,xx convergence",
    xlabel="δ/mls  or  Δω/mls", ylabel="Λ_xx", legend=:topright)
plot!(p2, multipliers, drude_vs_delta_xx, label="Λ_xx(ω=mls, δ)  (δ method)", 
    marker=:circle, color=:blue, linewidth=2)
plot!(p2, multipliers, drude_dc_vs_dw_xx, label="Λ_D,xx(Δω)  [integral]", 
    marker=:square, color=:red, linewidth=2)

# --- Plot 3: Regular conductivity σ_xy (real & imag) ---
p3 = plot(title="σ_xy^reg DC convergence (real & imag)",
    xlabel="δ/mls  or  Δω/mls", ylabel="σ_xy^reg", legend=:topright)
plot!(p3, multipliers, sigma_reg_vs_delta_xy, label="σ_xy (δ method)", 
    marker=:circle, color=:blue, linewidth=2)
plot!(p3, multipliers, sigma_dc_vs_dw_xy, label="σ_xy (Δω integral)", 
    marker=:square, color=:red, linewidth=2)

# --- Plot 4: Drude weight Λ_D,xy (real & imag) ---
p4 = plot(title="Drude weight Λ_D,xy convergence (real & imag)",
    xlabel="δ/mls  or  Δω/mls", ylabel="Λ_xy", legend=:topright)
plot!(p4, multipliers, drude_vs_delta_xy, label="Λ_xy(ω=mls, δ)  (δ method)", 
    marker=:circle, color=:blue, linewidth=2)
plot!(p4, multipliers, drude_dc_vs_dw_xy, label="Λ_D,xy(Δω)  [integral]", 
    marker=:square, color=:red, linewidth=2)


# Combine into a 2×2 layout
fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 800), margin=5Plots.mm)

# Print table of values
println("\n" * "=" ^ 80)
println("Regular conductivity σ_xx^reg   (mls = $(round(mls, sigdigits=4)))")
println("-" ^ 80)
println("  δ method (Im[Λ(ω=mls, δ)]/ω):")
for (n, v) in zip(multipliers, sigma_reg_vs_delta_xx)
    println("    $(n)×mls → σ_xx = $(round(v, sigdigits=6))")
end
println("  Δω integral method:")
for (n, v) in zip(multipliers, sigma_dc_vs_dw_xx)
    println("    $(n)×mls → σ_xx = $(round(v, sigdigits=6))")
end

println("\n" * "=" ^ 80)
println("Drude weight Λ_D,xx")
println("-" ^ 80)
println("  δ method (Re[Λ(ω=mls, δ)]):")
for (n, v) in zip(multipliers, drude_vs_delta_xx)
    println("    $(n)×mls → Λ_xx = $(round(v, sigdigits=6))")
end
println("  Δω integral method:")
for (n, v) in zip(multipliers, drude_dc_vs_dw_xx)
    println("    $(n)×mls → Λ_xx = $(round(v, sigdigits=6))")
end

display(fig)
println("\nDone!")

# ============================================================
# Test get_means from sim_num_control.jl
# ============================================================
println("\n" * "=" ^ 80)
println("Testing get_means() on the simulation data...")
println("=" ^ 80)

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))

means = get_means(data)

if isa(means, String)
    println("ERROR: get_means returned: $means")
else
    println("get_means returned $(length(means)) keys:")
    for k in sort(collect(keys(means)))
        v = means[k]
        if isa(v, Dict)
            val = get(v, "value", nothing)
            err = get(v, "error", nothing)
            if val === nothing
                println("  $k → (no value)")
            elseif isa(val, Number)
                println("  $k → value=$(round(val, sigdigits=6)), error=$(round(err, sigdigits=4))")
            elseif isa(val, AbstractVector) && !isempty(val) && isa(val[1], AbstractVector)
                println("  $k → Vector[$(length(val))] of Vector[$(length(val[1]))]")
            elseif isa(val, AbstractVector) && !isempty(val) && isa(val[1], Number) && length(val) <= 5
                println("  $k → value=$(round.(Float64.(val), sigdigits=5)), error=$(round.(Float64.(err), sigdigits=4))")
            elseif isa(val, AbstractVector) && !isempty(val) && isa(val[1], Number)
                println("  $k → Vector[$(length(val))] first=$(round(Float64(val[1]), sigdigits=5)), last=$(round(Float64(val[end]), sigdigits=5))")
            else
                println("  $k → $(typeof(val))")
            end
        else
            println("  $k → $(typeof(v))")
        end
    end

    # Specifically check the new drude_dc_* keys
    println("\n--- Drude weight DC (from L_D_dc_*) ---")
    for comp in ["drude_dc_xx", "drude_dc_yy", "drude_dc_xy", "drude_dc_yx"]
        if haskey(means, comp)
            vals = means[comp]["value"]
            errs = means[comp]["error"]
            println("  $comp: $(length(vals)) dw values")
            for (i, (v, e)) in enumerate(zip(vals, errs))
                if v !== nothing
                    println("    dw[$i] = $(round(v, sigdigits=6)) ± $(round(e, sigdigits=4))")
                else
                    println("    dw[$i] = nothing")
                end
            end
        else
            println("  $comp: NOT FOUND in means_dict")
        end
    end

    # Check that L_D_dc_* raw averages are also present
    println("\n--- L_D_dc_* raw averages ---")
    for comp in ["L_D_dc_xx", "L_D_dc_yy", "L_D_dc_xy", "L_D_dc_yx"]
        if haskey(means, comp)
            vals = means[comp]["value"]
            println("  $comp: $(length(vals)) dw values, first=$(round(vals[1], sigdigits=6))")
        else
            println("  $comp: NOT FOUND")
        end
    end

    # Check DC conductivities (xy, yx should now be real, not split into Re/Im)
    println("\n--- DC conductivity (should be real, no _Re/_Im) ---")
    for comp in ["conductivity_dc_xx", "conductivity_dc_xy", "conductivity_dc_yx", "conductivity_dc_yy"]
        if haskey(means, comp)
            vals = means[comp]["value"]
            println("  $comp: $(length(vals)) dw values, first=$(round(vals[1], sigdigits=6))")
        else
            println("  $comp: NOT FOUND")
        end
    end
    # Verify old _Re/_Im keys are gone
    for old_key in ["conductivity_dc_xy_Re", "conductivity_dc_xy_Im", "conductivity_dc_yx_Re", "conductivity_dc_yx_Im"]
        if haskey(means, old_key)
            println("  WARNING: old key '$old_key' still present!")
        end
    end

    # Cross-check: compare drude_dc_xx from get_means vs manual calculation
    println("\n--- Cross-check: drude_dc_xx vs manual ---")
    manual_drude_xx = drude_dc_vs_dw_xx  # from earlier in the script (mean of L_D_dc_xx)
    kx_val = means["Kx"]["value"]
    if haskey(means, "drude_dc_xx")
        gm_drude_xx = means["drude_dc_xx"]["value"]
        println("  Kx = $(round(kx_val, sigdigits=6))")
        for i in 1:min(5, length(gm_drude_xx))
            manual_D = -pi * (kx_val + manual_drude_xx[i])
            gm_D = gm_drude_xx[i]
            diff = abs(manual_D - gm_D)
            println("    dw[$i]: get_means=$(round(gm_D, sigdigits=6)), manual=-π(Kx+L_D)=$(round(manual_D, sigdigits=6)), diff=$(round(diff, sigdigits=3))")
        end
    end

    println("\nAll get_means tests completed!")
end
