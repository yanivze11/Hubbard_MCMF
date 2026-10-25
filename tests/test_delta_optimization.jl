"""
Test file for optimizing δ parameter in current-current correlations and DOS calculations

This test compares different methods for computing the regularization parameter δ:
1. maximum(Energies)/N  (current method)
2. mean(diff(Energies))
3. median(diff(Energies))
4. quantile(diff(Energies), p) for various percentiles

Goal: Find δ that makes Lxx(q→0, w→0) reach a constant value (Drude weight)
"""

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
using Statistics
using Plots
using JLD2

## Helper functions for computing δ with different methods

"""
Compute δ using various methods
Returns: Dict with different δ estimates
"""
function compute_delta_variants(Energies::Array{Float64}; N::Int64=round(Int(sqrt(length(Energies)/2))))
    sorted_E = sort(vec(Energies))
    diffs = diff(sorted_E)
    # Filter out very small differences (numerical noise)
    diffs_filtered = filter(x -> x > 1e-10, diffs)
    
    deltas = Dict{String, Float64}()
    
    # Method 1: Current implementation
    deltas["max/N"] = maximum(sorted_E) / N

    # Method 2: Mean of differences
    if length(diffs_filtered) > 0
        deltas["mean_diff"] = mean(diffs_filtered)
        
        # Method 3: Median of differences
        deltas["median_diff"] = median(diffs_filtered)
        
        # Method 5: RMS (root mean square) of differences
        deltas["rms_diff"] = sqrt(mean(diffs_filtered.^2))
        
    else
        # Fallback if no significant differences found
        for key in ["mean_diff", "median_diff", "q5_diff", "q10_diff", "q25_diff", 
                    "q50_diff", "q75_diff", "rms_diff", "hmean_diff"]
            deltas[key] = deltas["max/N"]
        end
    end

    # filter out large differences (e.g., >   max/N)
    diffs_up_filtered = filter(x -> x <  deltas["max/N"], diffs_filtered)
    
    # Method 3: Mean of differences (with upper filtering)
    if length(diffs_up_filtered) > 0
        deltas["mean_diff_low"] = mean(diffs_up_filtered)
        
        # Method 3: Median of differences
        deltas["median_diff_low"] = median(diffs_up_filtered)
        
        # Method 5: RMS (root mean square) of differences
        deltas["rms_diff_low"] = sqrt(mean(diffs_up_filtered.^2))

    else
        # Fallback if no significant differences found
        for key in ["mean_diff_low", "median_diff_low", "q5_diff", "q10_diff", "q25_diff", 
                    "q50_diff", "q75_diff", "rms_diff_low", "hmean_diff"]
            deltas[key] = deltas["max/N"]
        end
    end
    
    return deltas
end


"""
Compute current-current correlation with specified δ
Modified version of current_correlation_K to accept custom δ
"""
function current_correlation_K_custom(data::LatticeData, w::Float64, δ::Float64; 
                                      pre_G1=nothing, pre_G2=nothing)
    # Extract simulation parameters
    f = data.temporary_data.f
    Energies = data.current_state.Energies
    t = data.sim_data.t
    N = data.sim_data.N
    SC_N = data.sim_data.SC_L^2
    
    Lxx = ComplexF64(0.0)
    
    # Compute the current-current correlation
    for K in 1:SC_N
        # Get G matrices using make_G_matrix_K0 (active function)
        if pre_G1 === nothing
            G1 = make_G_matrix_K0(data; K_list=[K], direction="x")[1][1]
        else
            G1 = pre_G1[K]
        end
        if pre_G2 === nothing
            G2 = make_G_matrix_K0(data; K_list=[K], direction="x")[1][1]
        else
            G2 = pre_G2[K]
        end
        
        for l in 1:2N
            for m in 1:2N
                a = w + Energies[l, K] - Energies[m, K]
                a2 = (a^2 + δ^2)^(-1)
                A = a * a2
                lorenz = a2 * δ
                Lxx += G1[l,m] * conj(G2[l,m]) * (f[l, K] - f[m, K]) * (A - 1im * lorenz)
            end
        end
    end
    Lxx *= - t^2 / (N * SC_N)
    
    return Lxx
end


"""
Compute DOS with specified δ (dw parameter)
"""
function compute_DOS_custom(data::LatticeData, w::Float64, dw::Float64)
    N = data.sim_data.N
    U_matrix = data.current_state.U_matrix
    Energies = data.current_state.Energies
    
    DOS = Float64(0.0)
    for l in 1:(2*N)
        lorents = (dw/pi) * (dw^2 + (w - Energies[l])^2)^-1
        for i in 1:N
            DOS += 2* abs2(U_matrix[i, l]) * lorents
        end
    end
    return DOS
end


## Test 1: Analyze energy spectrum structure vs magnetic field

"""
Test how energy spectrum changes with magnetic field
Creates visualization of energy histograms and diff statistics
"""
function test_energy_spectrum_vs_B(; Lx=16, Ly=17, beta=10.0, 
                                    B_values=[0.0, 2π/(16*17)*10, 2π/(16*17)*30, 2π/(16*17)*60],
                                    save_path="test_results")
    
    mkpath(save_path)
    
    results = Dict[]
    
    println("=" ^ 70)
    println("TEST 1: Energy Spectrum Analysis vs Magnetic Field")
    println("=" ^ 70)
    
    for B in B_values
        println("\n" * "−" ^ 70)
        println("Testing B = $(round(B, digits=6)) (flux per plaquette: $(round(B/(2π/(Lx*Ly)), digits=3)))")
        
        # Create simulation
        sim_data = SimData(;
            Lx=Lx, Ly=Ly, SC_L=1,
            U=0.0, t=1.0, V=0.0,
            mu=0.0, beta=beta, B=B,
            num_measure=1, num_thermal=0, measuring_interval=1,
            w_list=[-4.5:0.1:4.5;],
            guess="uniform",
            U_disorder="uniform",
            diag_method="K0",
            w_conductivity_range=[0.0:0.001:0.1; 0.12:0.01:1.0;],
            pairing_symmetry="s-wave",
            seed=12345,
            gauge_choice="Mixed"
        )
        
        # Initialize and thermalize (minimal - just get energy spectrum)
        data = Lattice_Data(sim_data)
        
        # Get energies
        Energies = data.current_state.Energies
        N = sim_data.N
        
        # Compute delta variants
        deltas = compute_delta_variants(Energies; N=N)
        
        # Statistics on energy differences
        sorted_E = sort(vec(Energies))
        diffs = diff(sorted_E)
        diffs_filtered = filter(x -> x > 1e-10, diffs)
        
        result = Dict(
            "B" => B,
            "flux_per_plaquette" => B / (2π/(Lx*Ly)),
            "E_min" => minimum(Energies),
            "E_max" => maximum(Energies),
            "E_bandwidth" => maximum(Energies) - minimum(Energies),
            "num_levels" => length(sorted_E),
            "diff_min" => minimum(diffs_filtered),
            "diff_max" => maximum(diffs_filtered),
            "diff_mean" => mean(diffs_filtered),
            "diff_median" => median(diffs_filtered),
            "diff_std" => std(diffs_filtered),
            "deltas" => deltas,
            "energies" => sorted_E,
            "diffs" => diffs_filtered
        )
        
        push!(results, result)
        
        # Print summary
        println("\nEnergy spectrum:")
        println("  Range: [$(round(result["E_min"], digits=3)), $(round(result["E_max"], digits=3))]")
        println("  Bandwidth: $(round(result["E_bandwidth"], digits=3))")
        println("\nEnergy differences statistics:")
        println("  Min:    $(round(result["diff_min"], digits=6))")
        println("  Mean:   $(round(result["diff_mean"], digits=6))")
        println("  Median: $(round(result["diff_median"], digits=6))")
        println("  Max:    $(round(result["diff_max"], digits=6))")
        println("  Std:    $(round(result["diff_std"], digits=6))")
        println("\nδ estimates:")
        for (method, val) in sort(collect(deltas), by=x->x[2])
            println("  $(rpad(method, 15)): $(round(val, digits=6))")
        end
    end
    
    # Create visualizations
    println("\n" * "=" ^ 70)
    println("Creating visualizations...")
    
    # Plot 1: Energy histograms
    # res_len = length(results)
    # results_plot = results[[1:div(res_len, 4):end;]]  # Limit to first 4 for subplotting
    # Select 4 values spread equally from the results array for plotting
    if length(results) >= 4
        idxs = round.(Int, range(1, length(results), length=4))
        results_plot = results[idxs]
    else
        results_plot = results
    end
    p1 = plot(layout=(2,2), size=(1200, 900), 
              plot_title="Energy Level Histograms vs Magnetic Field",left_margin=10mm,
            bottom_margin=10mm)
    for (i, res) in enumerate(results_plot)
        histogram!(p1, res["energies"], subplot=i,
                  bins=50, alpha=0.7,
                  title="B = $(round(res["B"], digits=4)), Flux = $(round(res["flux_per_plaquette"], digits=2))\nT = $(round(1/beta, digits=3))",
                  xlabel="Energy", ylabel="Count",
                  legend=false)
    end
    savefig(p1, joinpath(save_path, "energy_histograms.png"))
    
    # Plot 2: Energy difference histograms (log scale)
    p2 = plot(layout=(2,2), size=(1200, 900),
              plot_title="Energy Level Spacing Histograms (log scale)", left_margin=10mm,
            bottom_margin=10mm)
    for (i, res) in enumerate(results_plot)
        histogram!(p2, res["diffs"], subplot=i,
                  bins=50, alpha=0.7,
                  title="B = $(round(res["B"], digits=4)), T = $(round(1/beta, digits=3))",
                  xlabel="ΔE", ylabel="Count",
                  yscale=:log10,
                  legend=false)
    end
    savefig(p2, joinpath(save_path, "energy_diff_histograms.png"))
    
    # Plot 3: δ comparison across B values
    p3 = plot(size=(1000, 600), legend=:topright,
             xlabel="Magnetic Field B", ylabel="δ estimate",
             title="δ Parameter Estimates vs Magnetic Field\nLx=$(Lx), Ly=$(Ly), T=$(round(1/beta, digits=3))",
             palette=:tab10,left_margin=10mm,
            bottom_margin=10mm)
    
    B_vals = [r["B"] for r in results]
    method_names = sort(collect(keys(results[1]["deltas"])))
    
    for method in method_names
        delta_vals = [r["deltas"][method] for r in results]
        plot!(p3, B_vals, delta_vals, marker=:circle, label=method, linewidth=2)
    end
    savefig(p3, joinpath(save_path, "delta_vs_B.png"))
    
    # Save results
    JLD2.save(joinpath(save_path, "test1_results.jld2"), 
              "results", results, 
              "description", "Energy spectrum analysis vs B")
    
    println("Results saved to: $save_path")
    println("=" ^ 70)
    
    return results
end


## Test 2: Compare Lxx(w) for different δ methods

"""
Test current-current correlation convergence to Drude weight
for different δ calculation methods
"""
function test_Lxx_convergence(; Lx=16, Ly=17, beta=10.0, B=0.0,
                               w_range=[0.0:0.0001:0.1; 0.101:0.01:1.0; 1.1:0.1:2.0],
                               save_path="test_results")
    
    mkpath(save_path)
    
    println("\n" * "=" ^ 70)
    println("TEST 2: Current-Current Correlation Lxx(w) Convergence Test")
    println("=" ^ 70)
    println("B = $(round(B, digits=6))")
    
    # Note: For non-interacting system (U=0), expect very small Lxx at B=0
    # Test is most meaningful for B≠0 (Landau level regime) or with interactions/disorder
    
    # Create simulation
    sim_data = SimData(
        Lx=Lx, Ly=Ly, SC_L=1,
        U=0.0, t=1.0, V=0.0,
        mu=0.0, beta=beta, B=B,
        num_measure=1, num_thermal=0, measuring_interval=1,
        w_list=[-4.5:0.1:4.5;],
        guess="uniform",
        U_disorder="uniform",
        diag_method="K0",
        w_conductivity_range=w_range,
        pairing_symmetry="s-wave",
        seed=12345,
        gauge_choice="Mixed"
    )
    
    if B == 0.0
        println("WARNING: Non-interacting system at B=0 will have very small Lxx (~10^-27)")
        println("         This test is mainly for checking numerical stability")
        println("         For meaningful physics, test with B≠0 or add interactions/disorder")
    end
    
    # Initialize
    data = Lattice_Data(sim_data)
    
    # Compute delta variants
    Energies = data.current_state.Energies
    N = sim_data.N
    deltas = compute_delta_variants(Energies; N=N)
    
    # Calculate filling (average occupation)
    n_filling = avrage_n(data)
    
    println("\nδ estimates:")
    for (method, val) in sort(collect(deltas), by=x->x[2])
        println("  $(rpad(method, 15)): $(round(val, digits=6))")
    end
    println("Filling: n = $(round(n_filling, digits=4))")
    
    # Precompute G matrices (same for all w) using make_G_matrix_K0
    println("\nPrecomputing G matrices...")
    SC_N = sim_data.SC_L^2
    G_all = make_G_matrix_K0(data; K_list=[1:SC_N;], direction="x")[1]
    pre_G1 = G_all
    pre_G2 = G_all
    
    # Compute Lxx(w) for each delta method
    println("\nComputing Lxx(w) for different δ methods...")
    results = Dict{String, Vector{ComplexF64}}()
    
    for (method, delta_val) in deltas
        print("  $method... ")
        Lxx_vals = ComplexF64[]
        for w in w_range
            Lxx = current_correlation_K_custom(data, w, delta_val; 
                                               pre_G1=pre_G1, pre_G2=pre_G2)
            push!(Lxx_vals, Lxx)
        end
        results[method] = Lxx_vals
        println("Done")
    end
    
    # Analyze plateau quality
    println("\n" * "−" ^ 70)
    println("Plateau Analysis (w < 0.01):")
    println("−" ^ 70)
    
    plateau_metrics = Dict{String, Dict{String, Float64}}()
    plateau_mask = w_range .< 0.01
    
    if sum(plateau_mask) > 1
        for (method, Lxx_vals) in results
            # Real part (Drude weight region)
            real_vals = real.(Lxx_vals[plateau_mask])
            plateau_mean = mean(real_vals)
            plateau_std = std(real_vals)
            plateau_cv = plateau_std / abs(plateau_mean)  # Coefficient of variation
            
            plateau_metrics[method] = Dict(
                "mean" => plateau_mean,
                "std" => plateau_std,
                "cv" => plateau_cv
            )
            
            println("  $(rpad(method, 15)): mean=$(round(plateau_mean, digits=4)), " *
                   "std=$(round(plateau_std, digits=6)), CV=$(round(plateau_cv, digits=4))")
        end
    end
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot 1: Re[Lxx(w)] - full range with ylims for low omega focus
    # Determine ylims based on low omega values (w < 0.1)
    low_w_vals = vcat([real.(results[method][w_range .< 0.1]) for method in keys(results)]...)
    y_min = minimum(low_w_vals)
    y_max = maximum(low_w_vals)
    y_margin = (y_max - y_min) * 0.1
    
    p1 = plot(size=(1000, 600), legend=:topright,
             xlabel="ω", ylabel="Re[Lxx(ω)]",
             title="Current-Current Correlation (Real Part)\nB=$(round(B, digits=4)), n=$(round(n_filling, digits=3)), T=$(round(1/beta, digits=3))",
             xscale=:log10, ylims=(y_min - y_margin, y_max + y_margin),
             left_margin=10mm, bottom_margin=10mm, palette=:tab10)

    for method in sort(collect(keys(results)))
        Lxx_vals = results[method]
        # println("Lxx values for method $method: ", Lxx_vals)
        plot!(p1, w_range, real.(Lxx_vals), 
              marker=:circle, label=method, linewidth=2)
    end
    savefig(p1, joinpath(save_path, "Lxx_real_vs_delta.png"))
    
    # Plot 2: Re[Lxx(w)] - zoom on low w (linear scale) with ylims
    low_w_mask = w_range .< 0.1
    low_w_real_vals = vcat([real.(results[method][low_w_mask]) for method in keys(results)]...)
    vals_for_margins = vcat([real.(results[method][w_range .<0.01]) for method in keys(results)]...)
    y_min2 = minimum(vals_for_margins)
    y_max2 = maximum(vals_for_margins)
    y_margin2 = (y_max2 - y_min2) * 0.1
    
    p2 = plot(size=(1000, 600), legend=:right,
             xlabel="ω", ylabel="Re[Lxx(ω)]",
             title="Current-Current Correlation - Low ω Region\nB=$(round(B, digits=4)), n=$(round(n_filling, digits=3)), T=$(round(1/beta, digits=3))",
             xlims=(0, maximum(w_range[low_w_mask])),
             ylims=(y_min2 - y_margin2, y_max2 + y_margin2),
             left_margin=10mm, bottom_margin=10mm, palette=:tab10)
    
    for method in sort(collect(keys(results)))
        Lxx_vals = results[method]
        plot!(p2, w_range[low_w_mask], real.(Lxx_vals[low_w_mask]), 
              marker=:circle, label=method, linewidth=2, markersize=4)
    end
    savefig(p2, joinpath(save_path, "Lxx_real_loww.png"))
    
    # Plot 3: Im[Lxx(w)]/w - regular conductivity with ylims
    # Compute all conductivity values to determine ylims
    all_conductivities = []
    for method in keys(results)
        Lxx_vals = results[method]
        conductivity = [-imag(Lxx_vals[i])/w_range[i] for i in 1:length(w_range) if w_range[i] > 1e-10 && w_range[i] < 0.1]
        append!(all_conductivities, conductivity)
    end
    y_min3 = minimum(all_conductivities)
    y_max3 = maximum(all_conductivities)
    y_margin3 = (y_max3 - y_min3) * 0.1
    
    p3 = plot(size=(1000, 600), legend=:topright,
             xlabel="ω", ylabel="-Im[Lxx(ω)]/ω",
             title="Regular Conductivity -Im[Lxx(ω)]/ω\nB=$(round(B, digits=4)), n=$(round(n_filling, digits=3)), T=$(round(1/beta, digits=3))",
             xscale=:log10, ylims=(y_min3 - y_margin3, y_max3 + y_margin3),
             left_margin=10mm, bottom_margin=10mm, palette=:tab10)
    
    for method in sort(collect(keys(results)))
        Lxx_vals = results[method]
        # Compute -Im[Lxx]/w, avoiding division by zero
        conductivity = [-imag(Lxx_vals[i])/w_range[i] for i in 1:length(w_range) if w_range[i] > 1e-10]
        w_nonzero = [w for w in w_range if w > 1e-10]
        plot!(p3, w_nonzero, conductivity, 
              marker=:circle, label=method, linewidth=2)
    end
    savefig(p3, joinpath(save_path, "Lxx_conductivity_vs_delta.png"))
    
    # Save results
    JLD2.save(joinpath(save_path, "test2_results.jld2"),
              "results", results,
              "deltas", deltas,
              "w_range", w_range,
              "plateau_metrics", plateau_metrics,
              "B", B,
              "description", "Lxx(w) convergence test")
    
    println("\nResults saved to: $save_path")
    println("=" ^ 70)
    
    return results, deltas, plateau_metrics
end;


## Test 3: DOS broadening test

"""
Test DOS calculations with different δ (dw) values
"""
function test_DOS_broadening(; Lx=10, Ly=11, beta=10.0, B=0.0,
                              w_range=[-4.0:0.02:4.0;],
                              save_path="test_results")
    
    mkpath(save_path)
    
    println("\n" * "=" ^ 70)
    println("TEST 3: DOS Broadening Test")
    println("=" ^ 70)
    println("B = $(round(B, digits=6))")
    
    # Create simulation
    sim_data = SimData(
        Lx=Lx, Ly=Ly, SC_L=1,
        U=0.0, t=1.0, V=0.0,
        mu=0.0, beta=beta, B=B,
        num_measure=1, num_thermal=0, measuring_interval=1,
        w_list=w_range,
        guess="uniform",
        U_disorder="uniform",
        diag_method="K0",
        w_conductivity_range=[0.0:0.01:1.0;],
        pairing_symmetry="s-wave",
        seed=12345,
        gauge_choice="Mixed"
    )
    
    # Initialize
    data = Lattice_Data(sim_data)
    
    # Compute delta variants
    Energies = data.current_state.Energies
    N = sim_data.N
    deltas = compute_delta_variants(Energies; N=N)
    
    # Calculate filling
    n_filling = avrage_n(data)
    
    println("\nδ (dw) estimates:")
    for (method, val) in sort(collect(deltas), by=x->x[2])
        println("  $(rpad(method, 15)): $(round(val, digits=6))")
    end
    
    # Compute DOS for each method
    println("\nComputing DOS for different broadening parameters...")
    results = Dict{String, Vector{Float64}}()
    
    for (method, dw) in deltas
        print("  $method... ")
        DOS_vals = Float64[]
        for w in w_range
            DOS = compute_DOS_custom(data, w, dw)
            push!(DOS_vals, DOS)
        end
        results[method] = DOS_vals
        println("Done")
    end
    
    # Also compute with fixed small/large values for comparison
    for (label, dw) in [("fixed_0.01", 0.01), ("fixed_0.1", 0.1)]
        print("  $label... ")
        DOS_vals = Float64[]
        for w in w_range
            DOS = compute_DOS_custom(data, w, dw)
            push!(DOS_vals, DOS)
        end
        results[label] = DOS_vals
        println("Done")
    end
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot: DOS comparison
    p = plot(size=(1200, 700), legend=:topright,
            xlabel="Energy ω", ylabel="DOS(ω)",
            title="Density of States - Different Broadening Parameters\nB=$(round(B, digits=4)), n=$(round(n_filling, digits=3)), T=$(round(1/beta, digits=3))",
            palette=:tab10, left_margin=10mm,
            bottom_margin=10mm)
    
    for (method, DOS_vals) in sort(collect(results))
        plot!(p, w_range, DOS_vals, label=method, linewidth=2, alpha=0.8)
    end
    
    # Mark actual energy levels
    vline!(p, sort(vec(Energies)), 
           label="Energy levels", color=:black, alpha=0.1, linewidth=0.5)
    
    savefig(p, joinpath(save_path, "DOS_vs_broadening.png"))
    
    # Save results
    JLD2.save(joinpath(save_path, "test3_results.jld2"),
              "results", results,
              "deltas", deltas,
              "w_range", collect(w_range),
              "energies", sort(vec(Energies)),
              "B", B,
              "description", "DOS broadening test")
    
    println("\nResults saved to: $save_path")
    println("=" ^ 70)
    
    return results, deltas
end


## Main test runner

"""
Run all tests
"""
function run_all_tests(; save_path="test_results")
    
    mkpath(save_path)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    test_dir = joinpath(save_path, "delta_tests_$timestamp")
    mkpath(test_dir)
    
    println("\n" * "═" ^ 70)
    println("DELTA PARAMETER OPTIMIZATION TESTS")
    println("═" ^ 70)
    println("Results will be saved to: $test_dir")
    println("═" ^ 70)
    
    # Test 1: Energy spectrum vs B
    println("\n▸ Running Test 1: Energy Spectrum Analysis")
    test1_results = test_energy_spectrum_vs_B(
        B_values=2π/(16*17)* [0:div(136, 10):136;],
        save_path=joinpath(test_dir, "test1_spectrum")
    )
    
    # Test 2a: Lxx convergence without B
    println("\n▸ Running Test 2a: Lxx Convergence (B=0)")
    test2a_results, test2a_deltas, test2a_metrics = test_Lxx_convergence(
        B=0.0,
        save_path=joinpath(test_dir, "test2a_Lxx_B0")
    )
    
    # Test 2b: Lxx convergence with small B
    println("\n▸ Running Test 2b: Lxx Convergence (B=10)")
    test2b_results, test2b_deltas, test2b_metrics = test_Lxx_convergence(
        B=2π/(16*17)*10,
        save_path=joinpath(test_dir, "test2b_Lxx_B10")
    )
    
    # Test 2c: Lxx convergence with moderate B
    println("\n▸ Running Test 2c: Lxx Convergence (B=30)")
    test2c_results, test2c_deltas, test2c_metrics = test_Lxx_convergence(
        B=2π/(16*17)*30,
        save_path=joinpath(test_dir, "test2c_Lxx_B30")
    )
    
    # Test 3a: DOS without B
    println("\n▸ Running Test 3a: DOS Broadening (B=0)")
    test3a_results, test3a_deltas = test_DOS_broadening(
        B=0.0,
        save_path=joinpath(test_dir, "test3a_DOS_B0")
    )
    
    # Test 3b: DOS with B
    println("\n▸ Running Test 3b: DOS Broadening (B≠0)")
    test3b_results, test3b_deltas = test_DOS_broadening(
        B=2π/(10*11)*20,
        save_path=joinpath(test_dir, "test3b_DOS_B20")
    )
    
    println("\n" * "═" ^ 70)
    println("ALL TESTS COMPLETED")
    println("═" ^ 70)
    println("Results directory: $test_dir")
    println("═" ^ 70)
    
    return test_dir
end


## Quick test function for interactive use

"""
Quick test - just run test 2 with default parameters
"""
function quick_test()
    println("Running quick test (Lxx convergence with B≠0 for meaningful results)...")
    # Use B≠0 to test delta optimization with Landau levels
    results, deltas, metrics = test_Lxx_convergence(
        Lx=10, Ly=11, beta=8.0, B=2π/(10*11)*20,  # Moderate magnetic field
        save_path="quick_test_results"
    )
    
    # Print recommendation
    println("\n" * "=" ^ 70)
    println("RECOMMENDATION")
    println("=" ^ 70)
    
    # Find method with lowest coefficient of variation in plateau region
    best_method = nothing
    best_cv = Inf
    
    for (method, metric) in metrics
        if metric["cv"] < best_cv
            best_cv = metric["cv"]
            best_method = method
        end
    end
    
    if best_method !== nothing
        println("Best δ method based on plateau stability: $best_method")
        println("  δ value: $(round(deltas[best_method], digits=6))")
        println("  Plateau CV: $(round(best_cv, digits=4))")
    end
    println("=" ^ 70)
    
    return results, deltas, metrics
end


# Print usage instructions when file is included
println("\n" * "=" ^ 70)
println("Delta Optimization Test Suite Loaded")
println("=" ^ 70)
println("Available functions:")
println("  • quick_test()                    - Quick test with default parameters")
println("  • run_all_tests()                 - Run complete test suite")
println("  • test_energy_spectrum_vs_B()     - Analyze energy spectrum vs B field")
println("  • test_Lxx_convergence()          - Test Lxx(w) convergence")
println("  • test_DOS_broadening()           - Test DOS broadening")
println("  • compute_delta_variants()        - Helper: compute δ with different methods")
println("\nExample: ")
println("  julia> results = quick_test()")
println("=" ^ 70 * "\n")
