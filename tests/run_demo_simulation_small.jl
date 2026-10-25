# Demo Simulation (Small System, Updated Observable Pipeline)
#
# This demo avoids deprecated direct measurement fields (for example: measurements.stiffness)
# and uses get_means(...) from sim_num_control.jl for derived observables.

using Dates
using Statistics
using Printf
using Plots

println("=" ^ 70)
println("MCMF Hubbard Demo (Small, Updated)")
println("=" ^ 70)

println("\nLoading simulation pipeline...")
try
    include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
catch e
    println("Failed to load sim_num_control.jl. This demo needs all its Julia dependencies.")
    println("Install missing packages, for example:")
    println("  julia -e \"import Pkg; Pkg.add(\\\"CurveFit\\\")\"")
    rethrow(e)
end
println("Loaded sim_num_control.jl (and dependencies).")

function extract_scalar_stat(means_dict::Dict{String, Any}, key::String; component::Int=1)
    if !haskey(means_dict, key)
        return nothing, nothing
    end
    entry = means_dict[key]
    if !(entry isa Dict)
        return nothing, nothing
    end

    value = get(entry, "value", nothing)
    err = get(entry, "error", nothing)

    if value isa AbstractVector
        if isempty(value)
            return nothing, nothing
        end
        idx = min(component, length(value))
        value = value[idx]

        if err isa AbstractVector && !isempty(err)
            err = err[min(component, length(err))]
        end
    end

    return value, err
end

function print_stat_line(means_dict::Dict{String, Any}, key::String, label::String; component::Int=1)
    value, err = extract_scalar_stat(means_dict, key; component=component)
    if value isa Number
        if err isa Number
            @printf("  %-26s = %.6g +- %.3g\n", label, float(real(value)), float(real(err)))
        else
            @printf("  %-26s = %.6g\n", label, float(real(value)))
        end
    else
        @printf("  %-26s = %s\n", label, "N/A")
    end
end

# ============================================================================
# Small-system simulation parameters
# ============================================================================
Lx = 4
Ly = 4
SC_L = 1

U = 4.0
t = 1.0
V = 0.0
mu = -0.9
beta = 16.0
B = 0.0

num_thermal = 60
num_measure = 40
measuring_interval = 1
saving_interval = 40

w_list = collect(-4.0:0.2:4.0)
w_conductivity = collect(0.0:0.1:1.0)

diag_method = "K0"
pairing_symmetry = "s-wave"
initial_guess = "uniform"
U_disorder = "uniform"
U_impurity = 0.0
gauge_choice = "LandauY"
seed = 12345

output_dir = "demo_results_small"
mkpath(output_dir)

println("\nSimulation setup:")
println("  Lattice: $(Lx)x$(Ly) (small demo)")
println("  Parameters: U=$U, t=$t, mu=$mu, beta=$beta")
println("  Method: $diag_method, pairing: $pairing_symmetry")
println("  Thermal sweeps: $num_thermal, measurements: $num_measure")

# ============================================================================
# Run simulation
# ============================================================================
println("\nRunning simulation...")
lattice_data = running_simulation_once(
    Lx, Ly,
    SC_L,
    U, t, V,
    seed,
    mu, beta, B,
    num_measure,
    num_thermal,
    measuring_interval,
    w_list,
    initial_guess,
    U_disorder,
    diag_method,
    pairing_symmetry,
    w_conductivity,
    U_impurity;
    gauge_choice=gauge_choice,
    directory=output_dir,
    savedata=false,
    saveplots=false,
    date_folder=false,
    numbered=false,
    saving_interval=saving_interval,
    dw_values=[0.1, 0.2, 0.5, 1.0],
)

num_meas = lattice_data.current_state.num_current_measurments
if num_meas < 1
    error("No measurements were collected in the demo run.")
end

println("Simulation finished with $num_meas measurements.")

# ============================================================================
# Use modern aggregation path for derived observables
# ============================================================================
means_dict = get_means(lattice_data; std_scheme="correlation", no_avg_obs=false, fit_stiffness=false)

println("\nDerived observables from get_means(...):")
print_stat_line(means_dict, "n", "n")
print_stat_line(means_dict, "Kx", "Kx")
print_stat_line(means_dict, "stiffness", "stiffness")
print_stat_line(means_dict, "stiffness_lowest_q", "stiffness_lowest_q")
print_stat_line(means_dict, "Ps", "Ps")
print_stat_line(means_dict, "K0_action", "K0_action")
print_stat_line(means_dict, "Delta_mean", "Delta_mean (alias)")

# ============================================================================
# Basic plots that rely on current fields
# ============================================================================
measurements = lattice_data.measurement_data
n_series = measurements.n[1:num_meas]
kx_series = measurements.Kx[1:num_meas]
S_series = measurements.Ss[1:num_meas]

p1 = plot(1:num_meas, n_series,
    xlabel="Measurement index", ylabel="n",
    title="Occupation Time Series",
    label="n", linewidth=2)
plot!(p1, 1:num_meas, kx_series, label="Kx", linewidth=2)
plot!(p1, 1:num_meas, S_series, label="S", linewidth=2, alpha=0.8)
savefig(p1, joinpath(output_dir, "01_timeseries_n_kx_S.png"))

DOS_all = measurements.DOS[1:num_meas]
DOS_avg = mean(DOS_all)
p2 = plot(w_list, DOS_avg,
    xlabel="w", ylabel="DOS(w)",
    title="Average DOS",
    label="DOS", linewidth=2)
savefig(p2, joinpath(output_dir, "02_dos_avg.png"))

labels = String[]
values = Float64[]
errors = Float64[]
for (key, label) in [
    ("n", "n"),
    ("Kx", "Kx"),
    ("stiffness", "stiffness"),
    ("stiffness_lowest_q", "stiff_low_q"),
    ("Ps", "Ps"),
    ("K0_action", "K0_action"),
]
    val, err = extract_scalar_stat(means_dict, key)
    if val isa Number
        push!(labels, label)
        push!(values, float(real(val)))
        push!(errors, err isa Number ? abs(float(real(err))) : 0.0)
    end
end

if !isempty(values)
    p3 = bar(1:length(values), values,
        yerror=errors,
        xticks=(1:length(labels), labels),
        xlabel="Observable", ylabel="Value",
        title="Derived Means (get_means)",
        legend=false)
    savefig(p3, joinpath(output_dir, "03_derived_means.png"))
end

println("\nSaved demo outputs to: $(abspath(output_dir))")
println("Generated PNG files:")
for f in sort(readdir(output_dir))
    if endswith(f, ".png")
        println("  - $f")
    end
end

println("\nDone.")
