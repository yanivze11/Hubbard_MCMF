# Demo Simulation: Full Monte Carlo run with visualization of all observables
# Uses the running_simulation_once function for a complete simulation

using Dates
using Statistics
using Plots

println("=" ^ 70)
println("Monte Carlo Mean-Field Hubbard Model Simulation Demo")
println("Rectangular Lattice with Full Observable Measurement")
println("=" ^ 70)

println("\nLoading MC_Hubburd.jl...")
include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))
println("Module loaded successfully!\n")

# ============================================================================
# Simulation Parameters
# ============================================================================
println("Setting up simulation parameters...")

# Lattice parameters (rectangular to demonstrate generalization)
Lx = 6    # x-dimensions
Ly = 7         # y-dimension
SC_L = 1        # Supercell sizes

# Physical parameters
U = 4.0         # Hubbard U (on-site interaction)
t = 1.0         # Hopping amplitude
V = 0.0         # Disorder potential
mu = -0.9        # Chemical potential (half-filling)
beta = 20.0      # Inverse temperature (1/T)
# B = 2.0         # Magnetic field
B = 2*pi/(Lx*Ly) * 4.0  # Magnetic field for one flux quantum through system

# Simulation parameters
num_thermal = 500    # Thermalization sweeps
num_measure = 200    # Measurement sweeps
measuring_interval = 2
saving_interval = 50  # Number of measurements to save

# Energy/frequency grids
w_list = collect(-4.0:0.1:4.0)           # For DOS and spectral function
w_conductivity = collect(0.0:0.05:1.0)   # For conductivity

# Method and symmetry
diag_method = "K0"              # Diagonalization method ("exact", "K0", "K0-TCA")
pairing_symmetry = "s-wave"     # Pairing symmetry ("s-wave", "d-wave", etc.)
initial_guess = "uniform"       # Initial guess for Delta
U_disorder = "uniform"          # U disorder type
U_impurity = 0.0                # Impurity strength
gauge_choice = "Mixed"       # Gauge choice for magnetic field

# Random seed for reproducibility
seed = 12345

# Output directory
output_dir = "demo_results"

println("  Lattice: $(Lx) x $(Ly) = $(Lx*Ly) sites")
println("  U = $U, t = $t, μ = $mu, β = $beta")
println("  Pairing symmetry: $pairing_symmetry")
println("  Thermalization: $num_thermal sweeps")
println("  Measurements: $num_measure with interval $measuring_interval")
println("  Saving interval: $saving_interval")

# ============================================================================
# Run Simulation using running_simulation_once
# ============================================================================
println("\n" * "="^70)
println("RUNNING SIMULATION")
println("="^70)

lattice_data = running_simulation_once(
    Lx, Ly,                    # Lattice dimensions
    SC_L,                      # Supercell size
    U, t, V,                   # Physical parameters
    seed,                      # Random seed
    mu, beta, B,               # Chemical potential, temperature, magnetic field
    num_measure,               # Number of measurement sweeps
    num_thermal,               # Number of thermalization sweeps
    measuring_interval,        # Measurement interval
    w_list,                    # Frequency list for DOS
    initial_guess,             # Initial guess type
    U_disorder,                # U disorder type
    diag_method,               # Diagonalization method
    pairing_symmetry,          # Pairing symmetry
    w_conductivity,            # Frequency list for conductivity
    U_impurity;                # Impurity strength
    gauge_choice=gauge_choice,
    directory = output_dir,
    savedata = false,          # Don't save data files (just demo)
    saveplots = false,         # We'll make our own plots
    date_folder = false,
    numbered = false,
    saving_interval = saving_interval
)

println("\nSimulation complete!")

# ============================================================================
# Extract Results
# ============================================================================
println("\n" * "="^70)
println("EXTRACTING RESULTS")
println("="^70)

sim_data = lattice_data.sim_data
measurements = lattice_data.measurement_data

# Get time series from measurements (only filled entries)
n_series = filter(!iszero, measurements.n[1:saving_interval])
Kx_series = filter(!iszero, measurements.Kx[1:saving_interval])
stiffness_series = measurements.stiffness[1:length(n_series)]
Ps_series = measurements.Ps[1:length(n_series)]
S_series = measurements.Ss[1:length(n_series)]

# Handle empty case
if isempty(n_series)
    n_series = measurements.n[1:1]
    Kx_series = measurements.Kx[1:1]
    stiffness_series = measurements.stiffness[1:1]
    Ps_series = measurements.Ps[1:1]
    S_series = measurements.Ss[1:1]
end

# Skip initial measurements for equilibrated statistics
skip_init = min(10, div(length(n_series), 4))
n_eq = n_series[skip_init+1:end]
Kx_eq = Kx_series[skip_init+1:end]
stiffness_eq = stiffness_series[skip_init+1:end]
Ps_eq = Ps_series[skip_init+1:end]

# Handle case where skip_init is too large
if isempty(n_eq)
    n_eq = n_series
    Kx_eq = Kx_series
    stiffness_eq = stiffness_series
    Ps_eq = Ps_series
end

# ============================================================================
# Results Summary
# ============================================================================
println("\n--- Thermodynamic Observables ---")
println("  Average occupation <n>     = $(round(mean(n_eq), digits=5)) ± $(round(std(n_eq), digits=5))")
println("  Kinetic energy <Kx>        = $(round(mean(Kx_eq), digits=5)) ± $(round(std(Kx_eq), digits=5))")
println("  Superfluid stiffness <ρs>  = $(round(mean(stiffness_eq), digits=5)) ± $(round(std(stiffness_eq), digits=5))")
println("  Pair susceptibility <Ps>   = $(round(mean(Ps_eq), digits=5)) ± $(round(std(Ps_eq), digits=5))")

# Delta order parameter
Delta_s_mean = mean(lattice_data.current_state.Delta_s)
Delta_dx_mean = mean(lattice_data.current_state.Delta_dx)
Delta_dy_mean = mean(lattice_data.current_state.Delta_dy)
println("\n--- Order Parameters ---")
println("  <Δs>  = $(round(abs(Delta_s_mean), digits=5)) (|Δs|)")
println("  <Δdx> = $(round(abs(Delta_dx_mean), digits=5)) (|Δdx|)")
println("  <Δdy> = $(round(abs(Delta_dy_mean), digits=5)) (|Δdy|)")

# ============================================================================
# Plotting Results
# ============================================================================
println("\n" * "="^70)
println("GENERATING PLOTS")
println("="^70)

# Create output directory
# output_dir = "demo_results"
mkpath(output_dir)

# 1. Occupation time series
println("\n1. Occupation time series...")
p1 = plot(1:length(n_series), n_series,
          xlabel="Measurement", ylabel="<n>",
          title="Average Occupation vs MC Time",
          label="<n>", linewidth=1, alpha=0.7)
hline!([mean(n_eq)], label="mean = $(round(mean(n_eq), digits=3))", 
       linestyle=:dash, linewidth=2)
savefig(p1, joinpath(output_dir, "01_occupation_timeseries.png"))

# 2. Action time series
println("2. Action time series...")
p2 = plot(1:length(S_series), S_series,
          xlabel="Measurement", ylabel="S",
          title="Action vs MC Time",
          label="S", linewidth=1, alpha=0.7)
savefig(p2, joinpath(output_dir, "02_action_timeseries.png"))

# 3. Kinetic energy time series
println("3. Kinetic energy time series...")
p3 = plot(1:length(Kx_series), Kx_series,
          xlabel="Measurement", ylabel="Kx",
          title="Kinetic Energy vs MC Time",
          label="Kx", linewidth=1, alpha=0.7)
hline!([mean(Kx_eq)], label="mean = $(round(mean(Kx_eq), digits=3))",
       linestyle=:dash, linewidth=2)
savefig(p3, joinpath(output_dir, "03_kinetic_energy.png"))

# 4. Superfluid stiffness time series
println("4. Superfluid stiffness...")
p4 = plot(1:length(stiffness_series), stiffness_series,
          xlabel="Measurement", ylabel="ρs",
          title="Superfluid Stiffness vs MC Time",
          label="ρs", linewidth=1, alpha=0.7)
hline!([mean(stiffness_eq)], label="mean = $(round(mean(stiffness_eq), digits=4))",
       linestyle=:dash, linewidth=2)
savefig(p4, joinpath(output_dir, "04_stiffness.png"))

# 5. Pair susceptibility
println("5. Pair susceptibility...")
p5 = plot(1:length(Ps_series), Ps_series,
          xlabel="Measurement", ylabel="Ps",
          title="Pair Susceptibility vs MC Time",
          label="Ps", linewidth=1, alpha=0.7)
hline!([mean(Ps_eq)], label="mean = $(round(mean(Ps_eq), digits=4))",
       linestyle=:dash, linewidth=2)
savefig(p5, joinpath(output_dir, "05_pair_susceptibility.png"))

# 6. DOS (Density of States)
println("6. Density of States (DOS)...")
valid_idx = 1:length(n_series)
DOS_avg = mean([measurements.DOS[i] for i in valid_idx])
p6 = plot(w_list, DOS_avg,
          xlabel="ω", ylabel="DOS(ω)",
          title="Density of States, β=$beta, $(Lx)x$(Ly)",
          label="DOS", linewidth=2, fillalpha=0.3, fill=0)
savefig(p6, joinpath(output_dir, "06_DOS.png"))

# 7. Spectral function along k-path
println("7. Spectral function A(k,ω)...")
try
    p7 = plot_spectral_function(lattice_data)
    savefig(p7, joinpath(output_dir, "07_spectral_function.png"))
catch e
    println("   Warning: Could not plot spectral function: $e")
end

# 8. Brillouin Zone occupation
println("8. Brillouin Zone occupation...")
try
    p8 = plot_BZ(lattice_data)
    savefig(p8, joinpath(output_dir, "08_BZ_occupation.png"))
catch e
    println("   Warning: Could not plot BZ: $e")
end

# 9. Delta field visualization
println("9. Order parameter field Δ...")
try
    p9_s = plot_Delta(lattice_data.current_state.Delta_s; Lx=Lx, Ly=Ly)
    title!("s-wave Δs field")
    savefig(p9_s, joinpath(output_dir, "09a_Delta_s.png"))
    
    p9_dx = plot_Delta(lattice_data.current_state.Delta_dx; Lx=Lx, Ly=Ly)
    title!("d-wave Δdx field")
    savefig(p9_dx, joinpath(output_dir, "09b_Delta_dx.png"))
    
    p9_dy = plot_Delta(lattice_data.current_state.Delta_dy; Lx=Lx, Ly=Ly)
    title!("d-wave Δdy field")
    savefig(p9_dy, joinpath(output_dir, "09c_Delta_dy.png"))
catch e
    println("   Warning: Could not plot Delta fields: $e")
end

# 10. Phase correlations
println("10. Phase correlations...")
try
    ls_list_reduced = lattice_data.const_struct.ls_list_reduced
    
    # Create x-axis labels as (lx, ly) strings
    l_labels = ["($(l[1]),$(l[2]))" for l in ls_list_reduced]
    x_positions = collect(1:length(ls_list_reduced))
    
    # Compute mean and std for each correlation type
    Phase_Corr_all = [measurements.Phase_Corr[i] for i in valid_idx]

    Phase_Corr_avg = mean(Phase_Corr_all)
    Phase_Corr_std = std(Phase_Corr_all)

    has_dwave_phase_corr = !isempty(measurements.Phase_Corr_dwave_x) && !isempty(measurements.Phase_Corr_dwave_y)
    if has_dwave_phase_corr
        Phase_Corr_dx_all = [measurements.Phase_Corr_dwave_x[i] for i in valid_idx]
        Phase_Corr_dy_all = [measurements.Phase_Corr_dwave_y[i] for i in valid_idx]
        Phase_Corr_dx_avg = mean(Phase_Corr_dx_all)
        Phase_Corr_dx_std = std(Phase_Corr_dx_all)
        Phase_Corr_dy_avg = mean(Phase_Corr_dy_all)
        Phase_Corr_dy_std = std(Phase_Corr_dy_all)
    end
    
    p10 = plot(x_positions, abs.(Phase_Corr_avg), 
               ribbon=abs.(Phase_Corr_std),
               xlabel="(lx, ly)", ylabel="|C(l)|",
               title="Phase Correlation Functions",
               label="s-wave", marker=:circle, linewidth=2,
               xticks=(x_positions, l_labels), xrotation=45, fillalpha=0.2)
    if has_dwave_phase_corr
        plot!(x_positions, abs.(Phase_Corr_dx_avg), ribbon=abs.(Phase_Corr_dx_std),
              label="d-wave (x)", marker=:square, linewidth=2, fillalpha=0.2)
        plot!(x_positions, abs.(Phase_Corr_dy_avg), ribbon=abs.(Phase_Corr_dy_std),
              label="d-wave (y)", marker=:diamond, linewidth=2, fillalpha=0.2)
    else
        println("   d-wave phase correlations were skipped for pairing_symmetry=$(pairing_symmetry)")
    end
    savefig(p10, joinpath(output_dir, "10_phase_correlations.png"))
catch e
    println("   Warning: Could not plot phase correlations: $e")
end

# 11. Current-current correlations (Lxx)
println("11. Current correlations Λxx(q)...")
try
    Lxx_qy_all = [measurements.Lxx_qy[i] for i in valid_idx]
    Lxx_qx_all = [measurements.Lxx_qx[i] for i in valid_idx]
    
    Lxx_qy_avg = mean(Lxx_qy_all)
    Lxx_qy_std = std(Lxx_qy_all)
    Lxx_qx_avg = mean(Lxx_qx_all)
    Lxx_qx_std = std(Lxx_qx_all)
    Kx_avg = mean(Kx_eq)
    Kx_std = std(Kx_eq)
    
    Lx_gen = Lx * SC_L
    Ly_gen = Ly * SC_L
    qy_vals = (2π/Ly_gen) .* collect(1:length(Lxx_qy_avg))
    qx_vals = (2π/Lx_gen) .* collect(1:length(Lxx_qx_avg))
    
    p11 = plot(qy_vals, Lxx_qy_avg, ribbon=Lxx_qy_std,
               xlabel="q", ylabel="Λxx(q)",
               title="Current-Current Correlation",
               label="Λxx(qy) (qx=0)", marker=:circle, linewidth=2, fillalpha=0.2)
    plot!(qx_vals, Lxx_qx_avg, ribbon=Lxx_qx_std,
          label="Λxx(qx) (qy=0)", marker=:square, linewidth=2, fillalpha=0.2)
    # Add -Kx at q=0 as star marker with error bar
    scatter!([0.0], [-Kx_avg], yerror=[Kx_std], label="-⟨Kx⟩", marker=:star, markersize=10, color=:red)
    savefig(p11, joinpath(output_dir, "11_current_correlations.png"))
catch e
    println("   Warning: Could not plot current correlations: $e")
end

# 12. Lxx_D and Regular conductivity vs frequency (combined)
println("12. Lxx_D & Regular conductivity vs ω...")
try
    Lxx_D_all = [measurements.Lxx_D[i] for i in valid_idx]
    Lxy_D_all = [measurements.Lxy_D[i] for i in valid_idx]
    cond_reg_all = [measurements.conductivity_regular[i] for i in valid_idx]
    cond_hall_reg_all = [measurements.conductivity_hall_regular[i] for i in valid_idx]
    
    Lxx_D_avg = mean(Lxx_D_all)
    Lxx_D_std = std(Lxx_D_all)
    cond_reg_avg = mean(cond_reg_all)
    cond_reg_std = std(cond_reg_all)
    Kx_avg = mean(Kx_eq)
    Kx_std = std(Kx_eq)
    Lxy_D_avg = mean(Lxy_D_all)
    Lxy_D_std = std(Lxy_D_all)
    cond_hall_reg_avg = mean(cond_hall_reg_all)
    cond_hall_reg_std = std(cond_hall_reg_all)
    
    p12 = plot(w_conductivity, Lxx_D_avg, ribbon=Lxx_D_std,
               xlabel="ω", ylabel="Value",
               title="Optical Response Functions",
               label="Re(Λxx(q=0,ω)) [Lxx_D]", linewidth=2, fillalpha=0.2)
    plot!(w_conductivity, cond_reg_avg, ribbon=cond_reg_std,
          label="σ_reg(ω)", linewidth=2, linestyle=:dash, fillalpha=0.2)
    plot!(w_conductivity, Lxy_D_avg, ribbon=Lxy_D_std,
          label="Re(Λxy(q=0,ω)) [Lxy_D]", linewidth=2, fillalpha=0.2)
    plot!(w_conductivity, cond_hall_reg_avg, ribbon=cond_hall_reg_std,
          label="σ_hall_reg(ω)", linewidth=2, linestyle=:dash, fillalpha=0.2)
    # Add -Kx at ω=0 as star marker with error bar (sum rule reference)
    scatter!([0.0], [-Kx_avg], yerror=[Kx_std], label="-⟨Kx⟩", marker=:star, markersize=10, color=:red)
    savefig(p12, joinpath(output_dir, "12_optical_response.png"))
catch e
    println("   Warning: Could not plot optical response: $e")
end

# 13. Combined summary plot
println("13. Summary plot...")
p_summary = plot(
    plot(1:length(S_series), S_series, title="Action S", xlabel="meas", ylabel="S", legend=false),
    plot(1:length(n_series), n_series, title="<n>", xlabel="meas", ylabel="<n>", legend=false),
    plot(1:length(Kx_series), Kx_series, title="Kx", xlabel="meas", ylabel="Kx", legend=false),
    plot(1:length(stiffness_series), stiffness_series, title="ρs", xlabel="meas", ylabel="ρs", legend=false),
    plot(w_list, DOS_avg, title="DOS", xlabel="ω", ylabel="DOS", legend=false, fill=0, fillalpha=0.3),
    plot(1:length(Ps_series), Ps_series, title="Ps", xlabel="meas", ylabel="Ps", legend=false),
    layout=(2,3), size=(1200, 600),
    plot_title="MCMF Simulation: $(Lx)×$(Ly), U=$U, β=$beta, $pairing_symmetry"
)
savefig(p_summary, joinpath(output_dir, "00_summary.png"))

# ============================================================================
# Final Output
# ============================================================================
println("\n" * "="^70)
println("SIMULATION COMPLETE")
println("="^70)
println("\nResults saved to: $(abspath(output_dir))/")
println("\nGenerated files:")
for f in sort(readdir(output_dir))
    if endswith(f, ".png")
        println("  - $f")
    end
end

println("\n" * "-"^50)
println("Final Statistics:")
println("-"^50)
println("  System:              $(Lx) × $(Ly) = $(Lx*Ly) sites")
println("  Parameters:          U=$U, t=$t, μ=$mu, β=$beta")
println("  Pairing:             $pairing_symmetry")
println("  MC sweeps:           $(num_thermal + num_measure*measuring_interval) total")
println("  Measurements:        $(length(n_series))")
println("-"^50)
println("  <n>                = $(round(mean(n_eq), digits=5)) ± $(round(std(n_eq), digits=5))")
println("  <Kx>               = $(round(mean(Kx_eq), digits=5)) ± $(round(std(Kx_eq), digits=5))")
println("  <ρs> (stiffness)   = $(round(mean(stiffness_eq), digits=5)) ± $(round(std(stiffness_eq), digits=5))")
println("  <Ps> (pair susc.)  = $(round(mean(Ps_eq), digits=5)) ± $(round(std(Ps_eq), digits=5))")
println("="^70)
println("Done!")
