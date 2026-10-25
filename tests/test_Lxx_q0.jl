# Small test for the new Lxx_q0 / Lxy_q0 / Lyx_q0 / Lyy_q0 observables.
# Runs a 4x4 lattice with one magnetic flux quantum and plots
# Lxx as a function of qy (qx=0) and qx (qy=0), with -<Kx> overlaid.

using Statistics
using Plots

include(joinpath(@__DIR__, "..", "sim_num_control.jl"))

Lx = 6
Ly = 6
SC_L = 1
N = Lx * Ly

U  = 4.0
t  = 1.0
V  = 0.0
mu = -0.9
beta = 20.0
# B = 2*pi / (Lx * Ly)   # one flux quantum on the 4x4 torus (Mixed gauge: B*Lx*Ly = 2π)
B = 0.0

num_thermal = 1000
num_measure = 1000
measuring_interval = 5
saving_interval = num_measure

w_list = [0.0,1.0]
w_conductivity =[0.0,1.0]

diag_method      = "K0"
pairing_symmetry = "s-wave"
initial_guess    = "uniform"
U_disorder       = "uniform"
U_impurity       = 0.0
gauge_choice     = "Mixed"
seed             = 12345

output_dir = joinpath(@__DIR__, "..", "demo_results_Lxx_q0")
mkpath(output_dir)

println("Running $Lx×$Ly test with B = 2π/$(Lx * Ly) ≈ $(round(B; digits=4))")

lattice_data = running_simulation_once(
    Lx, Ly, SC_L,
    U, t, V,
    seed,
    mu, beta, B,
    num_measure, num_thermal, measuring_interval,
    w_list,
    initial_guess, U_disorder,
    diag_method, pairing_symmetry,
    w_conductivity, U_impurity;
    gauge_choice = gauge_choice,
    directory    = output_dir,
    savedata     = false,
    saveplots    = false,
    date_folder  = false,
    numbered     = false,
    saving_interval = saving_interval,
    dw_values = [0.1, 0.5],
)

n_meas = lattice_data.current_state.num_current_measurments
@assert n_meas >= 1 "No measurements collected."
println("Collected $n_meas measurements.")

m = lattice_data.measurement_data

# q-grids that match how the simulation slices q_list_CC.
qs_y = (2*pi/Ly) * collect(1:div(Ly, 2))
qs_x = (2*pi/Lx) * collect(1:div(Lx, 2))
qy_full = vcat(0.0, qs_y)
qx_full = vcat(0.0, qs_x)

# Average over measurements. Lxx_qy[i] is a length-div(Ly,2) vector; q=0 lives in Lxx_q0[i].
Lxx_qy_series = [vcat(m.Lxx_q0[i],     m.Lxx_qy[i]) for i in 1:n_meas]
Lxx_qx_series = [vcat(m.Lxx_q0[i],     m.Lxx_qx[i]) for i in 1:n_meas]

Lxx_qy_mean = mean(Lxx_qy_series)
Lxx_qx_mean = mean(Lxx_qx_series)
Lxx_qy_err  = std(Lxx_qy_series)  ./ sqrt(n_meas)
Lxx_qx_err  = std(Lxx_qx_series)  ./ sqrt(n_meas)

Kx_mean = mean(m.Kx[1:n_meas])
Kx_err  = std(m.Kx[1:n_meas]) / sqrt(n_meas)
println("⟨Kx⟩ = $(round(Kx_mean; digits=5)) ± $(round(Kx_err; digits=5))")
println("⟨Lxx(q=0)⟩ = $(round(mean(m.Lxx_q0[1:n_meas]); digits=5))")

plt = plot(qy_full, Lxx_qy_mean,
    yerr  = Lxx_qy_err,
    marker = :circle, linewidth = 2, color = :blue,
    label = "Λxx(qx=0, qy)",
    xlabel = "q",
    ylabel = "Λxx",
    title  = "Λxx(q, ω=0) — $Lx×$Ly, B=2π/$(Lx * Ly), β=$beta",
    legend = :best,
)
plot!(plt, qx_full, Lxx_qx_mean,
    yerr   = Lxx_qx_err,
    marker = :square, linewidth = 2, color = :green,
    label  = "Λxx(qx, qy=0)")
hline!(plt, [-Kx_mean], color = :red, linestyle = :dash, linewidth = 2,
       label = "-⟨Kx⟩")

savefig(plt, joinpath(output_dir, "Lxx_q_with_q0.png"))
println("Saved plot to $(joinpath(output_dir, "Lxx_q_with_q0.png"))")
display(plt)