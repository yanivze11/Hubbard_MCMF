# Test: FT of the unperturbed paramagnetic current density (jP) at finite q with B != 0
#
# Sanity check for the new observable compute_jP_q + jPx_q*/jPy_q* in MeasureData.
# Setup: 4x5 lattice, Mixed gauge (A=(B*y*Lx, B*x*Ly)), minimal flux B = 2π/(Lx*Ly).
# Method: thermalize, then average ⟨jP(q)⟩ over many MC measurements.
# Expectations after averaging:
#   * B != 0  ->  ⟨jP_x(q)⟩ at some finite q is clearly non-zero (Mixed gauge breaks
#                  translation symmetry in both directions, so structure appears in
#                  both qx and qy cuts).
#   * B == 0  ->  ⟨jP_x(q)⟩ ~ 0 for every q.
# Plot: |⟨jP_x(qy)⟩| and |⟨jP_x(qx)⟩| with error bars propagated from per-sample SE.

using Test
using Statistics
using LinearAlgebra
using Plots

println("=" ^ 60)
println("Test: jP(q) on 4x5 lattice with magnetic field (MC averaged)")
println("=" ^ 60)

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

const Lx = 4
const Ly = 5
const B_flux = 2π / (Lx * Ly)   # minimal flux quantum for Mixed gauge (B*Lx*Ly = 2π)
const NUM_THERMAL = 1500
const NUM_MEAS = 1000
const STEPS_BETWEEN_MEAS = 10
const SEED = 1234

function build_lattice(; B::Float64, seed::Int=SEED)
    sim_data = SimData(
        Lx=Lx, Ly=Ly,
        U=4.0, beta=8.0,
        num_measure=NUM_MEAS, num_thermal=NUM_THERMAL,
        measuring_interval=STEPS_BETWEEN_MEAS,
        saving_interval=NUM_MEAS,
        diag_method="K0", pairing_symmetry="s-wave",
        gauge_choice="Mixed", boundary_conditions="PBC",
        B=B, seed=seed,
        w_list=[0.0],
        w_conductivity_range=[0.0],
        delta_values_fixed=[0.1],
        dw_values=[0.1],
        compute_jP=true,
    )
    return Lattice_Data(sim_data)
end

function thermalize_and_measure!(data::LatticeData; n_thermal::Int=NUM_THERMAL,
                                  n_meas::Int=NUM_MEAS, steps_between::Int=STEPS_BETWEEN_MEAS)
    println("    Thermalizing ($n_thermal steps)...")
    for _ in 1:n_thermal
        MC_stepV2!(data)
    end
    println("    Measuring ($n_meas samples, $steps_between steps between)...")
    for i in 1:n_meas
        for _ in 1:steps_between
            MC_stepV2!(data)
        end
        make_measurement_K!(data, i)
    end
end

# Per-sample stats from complex samples:
#   μ (complex mean), SE on Re(μ), SE on Im(μ), |μ|, propagated SE on |μ|.
# σ_|z|² ≈ (Re(μ)/|μ|)² σ_Re² + (Im(μ)/|μ|)² σ_Im²
function stats_from_samples(samples::AbstractVector{<:Complex})
    n = length(samples)
    μ = mean(samples)
    se_re = std(real.(samples); corrected=true) / sqrt(n)
    se_im = std(imag.(samples); corrected=true) / sqrt(n)
    absμ = abs(μ)
    se_abs = absμ < 1e-15 ? sqrt(se_re^2 + se_im^2) :
             sqrt((real(μ)/absμ)^2 * se_re^2 + (imag(μ)/absμ)^2 * se_im^2)
    return (μ=μ, se_re=se_re, se_im=se_im, absμ=absμ, se_abs=se_abs)
end

# Per-q-mode stats from a buffer Array{Array{ComplexF64,1},1}
function stats_per_q_vec(buf::Vector{Vector{ComplexF64}}, n::Int)
    n_q = length(buf[1])
    μ      = zeros(ComplexF64, n_q)
    se_re  = zeros(n_q)
    se_im  = zeros(n_q)
    absμ   = zeros(n_q)
    se_abs = zeros(n_q)
    for iq in 1:n_q
        samples = ComplexF64[buf[i][iq] for i in 1:n]
        st = stats_from_samples(samples)
        μ[iq]      = st.μ
        se_re[iq]  = st.se_re
        se_im[iq]  = st.se_im
        absμ[iq]   = st.absμ
        se_abs[iq] = st.se_abs
    end
    return (μ=μ, se_re=se_re, se_im=se_im, absμ=absμ, se_abs=se_abs)
end

function summarize_jPx(data::LatticeData; n::Int=NUM_MEAS)
    m = data.measurement_data
    st0 = stats_from_samples(m.jPx_q0[1:n])
    sty = stats_per_q_vec(m.jPx_qy, n)
    stx = stats_per_q_vec(m.jPx_qx, n)
    return (q0=st0, qy=sty, qx=stx)
end

# ============================================================
# Run both cases first, then test, then plot.
# ============================================================

println("\n--- 4x5, B = 2π/(Lx*Ly) = $(round(B_flux, sigdigits=4)) (Mixed gauge) ---")
data_B = build_lattice(B=B_flux)
thermalize_and_measure!(data_B)
sB = summarize_jPx(data_B)
println("  ⟨jPx_q0⟩ = $(sB.q0.μ)  (|.|=$(sB.q0.absμ) ± $(sB.q0.se_abs))")
println("  ⟨jPx_qy⟩ = $(sB.qy.μ)")
println("    |.| = $(sB.qy.absμ) ± $(sB.qy.se_abs)")
println("    Re  = $(real.(sB.qy.μ)) ± $(sB.qy.se_re)")
println("  ⟨jPx_qx⟩ = $(sB.qx.μ)")
println("    |.| = $(sB.qx.absμ) ± $(sB.qx.se_abs)")
println("    Re  = $(real.(sB.qx.μ)) ± $(sB.qx.se_re)")

println("\n--- 4x5, B = 0 (Mixed gauge) ---")
data_0 = build_lattice(B=0.0)
thermalize_and_measure!(data_0)
s0 = summarize_jPx(data_0)
println("  ⟨jPx_q0⟩ = $(s0.q0.μ)  (|.|=$(s0.q0.absμ) ± $(s0.q0.se_abs))")
println("  ⟨jPx_qy⟩ = $(s0.qy.μ)")
println("    |.| = $(s0.qy.absμ) ± $(s0.qy.se_abs)")
println("    Re  = $(real.(s0.qy.μ)) ± $(s0.qy.se_re)")
println("  ⟨jPx_qx⟩ = $(s0.qx.μ)")
println("    |.| = $(s0.qx.absμ) ± $(s0.qx.se_abs)")
println("    Re  = $(real.(s0.qx.μ)) ± $(s0.qx.se_re)")

# ============================================================
# Tests
# ============================================================

@testset "B != 0: |⟨jP_x(q≠0)⟩| above noise after MC averaging" begin
    max_abs = maximum(vcat(sB.qy.absμ, sB.qx.absμ))
    println("\n  max |⟨jP_x(q!=0)⟩| (B!=0) = $(round(max_abs, sigdigits=4))")
    @test max_abs > 1e-3

    m = data_B.measurement_data
    @test length(m.jPx_qy[1]) == div(Ly, 2)
    @test length(m.jPx_qx[1]) == div(Lx, 2)
end

@testset "B = 0 baseline: |⟨jP_x(q)⟩| ~ 0" begin
    max_abs0 = maximum(vcat([s0.q0.absμ], s0.qy.absμ, s0.qx.absμ))
    println("  max |⟨jP_x(q)⟩| (B=0) = $(round(max_abs0, sigdigits=4))")
    @test max_abs0 < 2e-2
end

# ============================================================
# Plot |⟨jP_x⟩| and Re ⟨jP_x⟩ with error bars
# ============================================================
qy_grid = vcat([0.0], [2π/Ly * j for j in 1:div(Ly, 2)])
qx_grid = vcat([0.0], [2π/Lx * j for j in 1:div(Lx, 2)])

# abs cuts (stitch q=0 in front)
absB_qy = vcat(sB.q0.absμ,  sB.qy.absμ);    seB_abs_qy = vcat(sB.q0.se_abs, sB.qy.se_abs)
absB_qx = vcat(sB.q0.absμ,  sB.qx.absμ);    seB_abs_qx = vcat(sB.q0.se_abs, sB.qx.se_abs)
abs0_qy = vcat(s0.q0.absμ,  s0.qy.absμ);    se0_abs_qy = vcat(s0.q0.se_abs, s0.qy.se_abs)
abs0_qx = vcat(s0.q0.absμ,  s0.qx.absμ);    se0_abs_qx = vcat(s0.q0.se_abs, s0.qx.se_abs)

# Re cuts
reB_qy = vcat(real(sB.q0.μ), real.(sB.qy.μ));    seB_re_qy = vcat(sB.q0.se_re, sB.qy.se_re)
reB_qx = vcat(real(sB.q0.μ), real.(sB.qx.μ));    seB_re_qx = vcat(sB.q0.se_re, sB.qx.se_re)
re0_qy = vcat(real(s0.q0.μ), real.(s0.qy.μ));    se0_re_qy = vcat(s0.q0.se_re, s0.qy.se_re)
re0_qx = vcat(real(s0.q0.μ), real.(s0.qx.μ));    se0_re_qx = vcat(s0.q0.se_re, s0.qx.se_re)

function plot_with_errors(q_grid, vB, vB_err, v0, v0_err; title::String, ylabel::String, hline_zero::Bool=false)
    plt = plot(title=title, xlabel="q", ylabel=ylabel, legend=:topright)
    if hline_zero
        hline!(plt, [0.0]; color=:gray, linestyle=:dash, label="")
    end
    plot!(plt, q_grid, vB; yerror=vB_err, label="B=2π/(Lx·Ly)",
          color=:blue, lw=2, marker=:circle, markersize=5, msc=:blue)
    plot!(plt, q_grid, v0; yerror=v0_err, label="B=0",
          color=:red,  lw=2, marker=:square, markersize=5, msc=:red)
    return plt
end

p1 = plot_with_errors(qy_grid, absB_qy, seB_abs_qy, abs0_qy, se0_abs_qy;
                      title="|⟨jP_x(qy)⟩| (qx=0)", ylabel="|⟨jP_x(qy)⟩|")
p2 = plot_with_errors(qx_grid, absB_qx, seB_abs_qx, abs0_qx, se0_abs_qx;
                      title="|⟨jP_x(qx)⟩| (qy=0)", ylabel="|⟨jP_x(qx)⟩|")
p3 = plot_with_errors(qy_grid, reB_qy, seB_re_qy, re0_qy, se0_re_qy;
                      title="Re ⟨jP_x(qy)⟩ (qx=0)", ylabel="Re ⟨jP_x(qy)⟩", hline_zero=true)
p4 = plot_with_errors(qx_grid, reB_qx, seB_re_qx, re0_qx, se0_re_qx;
                      title="Re ⟨jP_x(qx)⟩ (qy=0)", ylabel="Re ⟨jP_x(qx)⟩", hline_zero=true)

fig = plot(p1, p2, p3, p4; layout=(2,2), size=(1100, 800),
           plot_title="FT paramagnetic current ⟨jP_x(q)⟩, 4×5, Mixed gauge, B=2π/(Lx·Ly), U=4, β=8")

plot_path = joinpath(@__DIR__, "test_jP_q_magnetic.png")
savefig(fig, plot_path)
println("\nSaved plot to: $plot_path")
display(fig)

println("\n" * "=" ^ 60)
println("Done.")
println("=" ^ 60)
