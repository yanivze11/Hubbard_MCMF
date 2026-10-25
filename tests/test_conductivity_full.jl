"""
Test full non-interacting simulation and plot all conductivity components

This test runs a complete MC simulation for a non-interacting system (U=0)
and plots all conductivity components as a function of frequency omega.
"""

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
using Plots
using Statistics
using JLD2

"""
Run full non-interacting simulation and plot conductivity components
"""
function test_full_conductivity(; 
    Lx=10, 
    Ly=11, 
    beta=8.0, 
    B=2π/(10*11)*20,
    num_thermal=0,
    num_measure=1,
    measuring_interval=1,
    save_path="conductivity_test_results")
    
    mkpath(save_path)
    
    println("\n" * "=" ^ 70)
    println("FULL CONDUCTIVITY TEST - Non-Interacting System")
    println("=" ^ 70)
    println("System parameters:")
    println("  Lattice: $(Lx) × $(Ly)")
    println("  Temperature: T = $(round(1/beta, digits=3))")
    println("  Magnetic field: B = $(round(B, digits=6))")
    println("  Flux per plaquette: $(round(B/(2π/(Lx*Ly)), digits=3))")
    println("  Thermalization steps: $(num_thermal)")
    println("  Measurement steps: $(num_measure)")
    println("  Measuring interval: $(measuring_interval)")
    println("=" ^ 70)
    
    # Define frequency range for conductivity
    w_conductivity_range = [0.0001:0.0005:0.01; 0.015:0.005:0.1; 0.12:0.02:1.0;]
    w_list = [-4.5:0.1:4.5;]
    
    # Run full simulation using running_simulation_once
    println("\nRunning simulation...")
    data = running_simulation_once(
        Lx, Ly, 1,  # Lx, Ly, SC_L
        0.0, 1.0, 0.0,  # U, t, V (non-interacting)
        12345,  # seed
        0.0, beta, B,  # mu, beta, B
        num_measure, num_thermal, measuring_interval,
        w_list,
        "uniform",  # initial_guess
        "uniform",  # U_disorder
        "K0",  # diag_method
        "s-wave",  # pairing_symmetry
        w_conductivity_range,
        0.0;  # U_impurity
        savedata=false,  # Don't save to disk
        saveplots=false,
        gauge_choice="Mixed"
    )
    
    # Get final values
    filling_mean = avrage_n(data)
    delta_mean = data.temporary_data.delta
    
    println("\nFinal values:")
    println("  Filling: n = $(round(filling_mean, digits=4))")
    println("  Delta: δ = $(round(delta_mean, digits=6))")
    
    # Get measurements from data
    measurements = data.measurement_data
    
    # Average measurements across all samples
    # measurements.conductivity_regular_xx is an array of arrays (one per measurement)
    num_samples = length(measurements.conductivity_regular_xx)
    
    sigma_xx_avg = mean([measurements.conductivity_regular_xx[i] for i in 1:num_samples])
    sigma_yy_avg = mean([measurements.conductivity_regular_yy[i] for i in 1:num_samples])
    sigma_xy_avg = mean([measurements.conductivity_regular_xy[i] for i in 1:num_samples])
    sigma_yx_avg = mean([measurements.conductivity_regular_yx[i] for i in 1:num_samples])
    
    # Drude weights
    Lxx_D_avg = mean([measurements.Lxx_D[i] for i in 1:num_samples])
    Lyy_D_avg = mean([measurements.Lyy_D[i] for i in 1:num_samples])
    Lxy_D_avg = mean([measurements.Lxy_D[i] for i in 1:num_samples])
    Lyx_D_avg = mean([measurements.Lyx_D[i] for i in 1:num_samples])
    
    # Kinetic energy (for Drude weight)
    Kx_avg = mean([measurements.Kx[i] for i in 1:num_samples])
    
    println("\nMeasurement statistics:")
    println("  Number of samples: $(num_samples)")
    println("  Average <Kx>: $(round(Kx_avg, digits=6))")
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot 1: All conductivities together
    p1 = plot(size=(1000, 700), legend=:topright, palette=:tab10)
    
    plot!(p1, w_conductivity_range, sigma_xx_avg, 
          label="σxx", linewidth=2, marker=:circle, markersize=3)
    plot!(p1, w_conductivity_range, sigma_yy_avg, 
          label="σyy", linewidth=2, marker=:square, markersize=3)
    plot!(p1, w_conductivity_range, -sigma_xy_avg, 
          label="-σxy", linewidth=2, marker=:diamond, markersize=3)
    plot!(p1, w_conductivity_range, sigma_yx_avg, 
          label="σyx", linewidth=2, marker=:utriangle, markersize=3)
    
    plot!(p1, xlabel="ω", ylabel="σ(ω)",
          title="All Conductivity Components\nB=$(round(B, digits=4)), n=$(round(filling_mean, digits=3)), T=$(round(1/beta, digits=3))",
          xscale=:log10)
    
    savefig(p1, joinpath(save_path, "conductivity_all.png"))
    
    # Plot 2: Longitudinal conductivity σxx, σyy
    p2 = plot(size=(1000, 700), legend=:topright, palette=:tab10)
    
    plot!(p2, w_conductivity_range, sigma_xx_avg, 
          label="σxx", linewidth=2, marker=:circle, markersize=3)
    
    plot!(p2, w_conductivity_range, sigma_yy_avg, 
          label="σyy", linewidth=2, marker=:square, markersize=3)
    
    plot!(p2, xlabel="ω", ylabel="σ(ω)",
          title="Longitudinal Conductivity\nB=$(round(B, digits=4)), n=$(round(filling_mean, digits=3)), T=$(round(1/beta, digits=3))",
          xscale=:log10)
    
    savefig(p2, joinpath(save_path, "conductivity_longitudinal.png"))
    
    # Plot 3: Hall conductivity σxy, σyx
    p3 = plot(size=(1000, 700), legend=:topright, palette=:tab10)
    
    plot!(p3, w_conductivity_range, -sigma_xy_avg, 
          label="-σxy", linewidth=2, marker=:circle, markersize=3)
    
    plot!(p3, w_conductivity_range, sigma_yx_avg, 
          label="σyx", linewidth=2, marker=:square, markersize=3)
    
    plot!(p3, xlabel="ω", ylabel="σ(ω)",
          title="Hall Conductivity\nB=$(round(B, digits=4)), n=$(round(filling_mean, digits=3)), T=$(round(1/beta, digits=3))",
          xscale=:log10)
    
    savefig(p3, joinpath(save_path, "conductivity_hall.png"))
    
    # Plot 4: Drude weights (Real parts) with -Kx
    p4 = plot(size=(1000, 700), legend=:topright, palette=:tab10)
    
    plot!(p4, w_conductivity_range, Lxx_D_avg, 
          label="Lxx_D", linewidth=2, marker=:circle, markersize=3)
    plot!(p4, w_conductivity_range, Lyy_D_avg, 
          label="Lyy_D", linewidth=2, marker=:square, markersize=3)
    plot!(p4, w_conductivity_range, Lxy_D_avg, 
          label="Lxy_D", linewidth=2, marker=:diamond, markersize=3)
    plot!(p4, w_conductivity_range, Lyx_D_avg, 
          label="Lyx_D", linewidth=2, marker=:utriangle, markersize=3)
    
    # Add -Kx as horizontal line (it's independent of ω)
    hline!(p4, [-Kx_avg], label="-<Kx>", linestyle=:dash, linewidth=2)
    
    plot!(p4, xlabel="ω", ylabel="Re[L(ω)]",
          title="Drude Weight Components with -<Kx>\nB=$(round(B, digits=4)), n=$(round(filling_mean, digits=3)), T=$(round(1/beta, digits=3))",
          xscale=:log10)
    
    savefig(p4, joinpath(save_path, "drude_weights.png"))
    
    # Plot 5: All conductivity components together in subplots
    p5 = plot(size=(1200, 800), layout=(2,2), palette=:tab10)
    
    # σxx
    plot!(p5, w_conductivity_range, sigma_xx_avg, 
          subplot=1, label="σxx", linewidth=2, marker=:circle, markersize=3,
          xlabel="ω", ylabel="σxx(ω)", title="Longitudinal xx", xscale=:log10)
    
    # σyy
    plot!(p5, w_conductivity_range, sigma_yy_avg, 
          subplot=2, label="σyy", linewidth=2, marker=:square, markersize=3,
          xlabel="ω", ylabel="σyy(ω)", title="Longitudinal yy", xscale=:log10)
    
    # σxy
    plot!(p5, w_conductivity_range, -sigma_xy_avg, 
          subplot=3, label="-σxy", linewidth=2, marker=:diamond, markersize=3,
          xlabel="ω", ylabel="-σxy(ω)", title="Hall xy", xscale=:log10)
    
    # σyx
    plot!(p5, w_conductivity_range, sigma_yx_avg, 
          subplot=4, label="σyx", linewidth=2, marker=:utriangle, markersize=3,
          xlabel="ω", ylabel="σyx(ω)", title="Hall yx", xscale=:log10)
    
    plot!(p5, plot_title="All Conductivity Components\nB=$(round(B, digits=4)), n=$(round(filling_mean, digits=3)), T=$(round(1/beta, digits=3))")
    
    savefig(p5, joinpath(save_path, "conductivity_all_components.png"))
    
    # Save data
    JLD2.save(joinpath(save_path, "conductivity_results.jld2"),
              "w_range", w_conductivity_range,
              "sigma_xx", sigma_xx_avg,
              "sigma_yy", sigma_yy_avg,
              "sigma_xy", -sigma_xy_avg,
              "sigma_yx", sigma_yx_avg,
              "Lxx_D", Lxx_D_avg,
              "Lyy_D", Lyy_D_avg,
              "Lxy_D", Lxy_D_avg,
              "Lyx_D", Lyx_D_avg,
              "Kx_avg", Kx_avg,
              "delta_mean", delta_mean,
              "filling_mean", filling_mean,
              "B", B,
              "beta", beta,
              "Lx", Lx,
              "Ly", Ly,
              "description", "Full conductivity test - non-interacting system")
    
    println("\nResults saved to: $save_path")
    println("=" ^ 70)
    
    return sigma_xx_avg, sigma_yy_avg, -sigma_xy_avg, sigma_yx_avg, Lxx_D_avg, Lyy_D_avg, Lxy_D_avg, Lyx_D_avg
end


# Print usage instructions
println("\n" * "=" ^ 70)
println("Full Conductivity Test Loaded")
println("=" ^ 70)
println("Usage:")
println("  julia> include(\"test_conductivity_full.jl\")")
println("  julia> test_full_conductivity()")
println("\nOptional parameters:")
println("  Lx, Ly          - Lattice dimensions (default: 10, 11)")
println("  beta            - Inverse temperature (default: 8.0)")
println("  B               - Magnetic field (default: 2π/(10*11)*20)")
println("  num_thermal     - Thermalization steps (default: 500)")
println("  num_measure     - Measurement steps (default: 100)")
println("  measuring_interval - Steps between measurements (default: 10)")
println("  save_path       - Output directory (default: 'conductivity_test_results')")
println("=" ^ 70 * "\n")
