# Test: Verify slimmed MeasureData works end-to-end
# Runs a quick 4x4 simulation and checks that:
#   1. Simulation runs without errors
#   2. MeasureData struct has no removed fields
#   3. get_means() derives all transport observables correctly from L_w_fixed
#   4. Key observables (stiffness, Ps, conductivity, Drude weight, resistivity) are present and finite

using Dates
using Statistics
using Test

println("=" ^ 70)
println("TEST: Slimmed MeasureData — End-to-End Verification")
println("=" ^ 70)

println("\nLoading modules...")
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
println("Modules loaded.\n")

# ============================================================================
# 1. Run a quick simulation
# ============================================================================
println("─" ^ 50)
println("Phase 1: Running a quick 4×4 simulation")
println("─" ^ 50)

Lx = 4
Ly = 4
SC_L = 1
U = 4.0
t = 1.0
V = 0.0
mu = 0.0           # half-filling
beta = 5.0          # moderate temperature
B = 0.0
seed = 42
num_thermal = 20
num_measure = 30
measuring_interval = 1
saving_interval = 30
w_list = collect(-3.0:0.5:3.0)
w_conductivity = collect(0.05:0.1:0.55)
delta_values_fixed = [0.05, 0.1, 0.2, 0.4]
dw_values = [0.05, 0.1, 0.2]

lattice_data = running_simulation_once(
    Lx, Ly, SC_L, U, t, V, seed, mu, beta, B,
    num_measure, num_thermal, measuring_interval,
    w_list, "uniform", "uniform", "K0", "s-wave",
    w_conductivity, 0.0;
    directory = "quick_test_results",
    savedata = false,
    saveplots = false,
    date_folder = false,
    numbered = false,
    saving_interval = saving_interval,
    delta_values_fixed = delta_values_fixed,
    dw_values = dw_values
)

println("Simulation completed!\n")

# ============================================================================
# 2. Verify MeasureData struct — removed fields should NOT exist
# ============================================================================
println("─" ^ 50)
println("Phase 2: Checking MeasureData struct fields")
println("─" ^ 50)

measurements = lattice_data.measurement_data

# Fields that should have been REMOVED (38 fields)
removed_fields = [
    # A: Non-fixed conductivity (16)
    :Lxx_w, :Lxy_w, :Lyx_w, :Lyy_w,
    :Lxx_D, :Lxy_D, :Lyx_D, :Lyy_D,
    :conductivity_regular_xx, :conductivity_regular_xy,
    :conductivity_regular_yx, :conductivity_regular_yy,
    :Drude_weight_xx, :Drude_weight_xy, :Drude_weight_yx, :Drude_weight_yy,
    # B: Fixed-delta derived (12)
    :Lxx_D_fixed, :Lxy_D_fixed, :Lyx_D_fixed, :Lyy_D_fixed,
    :conductivity_regular_xx_fixed, :conductivity_regular_xy_fixed,
    :conductivity_regular_yx_fixed, :conductivity_regular_yy_fixed,
    :Drude_weight_xx_fixed, :Drude_weight_xy_fixed,
    :Drude_weight_yx_fixed, :Drude_weight_yy_fixed,
    # C: q-delta stiffness (8)
    :Lxx_qy_delta, :Lxx_qx_delta, :Lxy_qy_delta, :Lxy_qx_delta,
    :Lyx_qy_delta, :Lyx_qx_delta, :Lyy_qy_delta, :Lyy_qx_delta,
    # D: Derivable scalars (2)
    :stiffness, :Ps
]

# Fields that MUST remain
required_fields = [
    :Ss, :A, :BZ, :DOS, :n, :chi_spin, :chi_charge,
    :Kx, :Ky,
    :Lxx_qy, :Lxx_qx, :Lxy_qy, :Lxy_qx,
    :Lyx_qy, :Lyx_qx, :Lyy_qy, :Lyy_qx,
    :Delta_operator, :Delta_mean, :DdD, :Ps_l,
    :LDOS, :Phase_Corr, :U4,
    :Phase_Corr_dwave_x, :Phase_Corr_dwave_y, :Phase_Corr_mixed,
    :Delta_MC_abs, :Delta_MC,
    :Lxx_w_fixed, :Lxy_w_fixed, :Lyx_w_fixed, :Lyy_w_fixed,
    :energy_spacing_mean, :delta_mean_diff_low,
    :conductivity_dc_xx, :conductivity_dc_xy, :conductivity_dc_yx, :conductivity_dc_yy
]

@testset "MeasureData struct fields" begin
    @testset "Removed fields absent" begin
        for f in removed_fields
            @test !hasfield(typeof(measurements), f)
        end
        println("  ✓ All 38 removed fields confirmed absent.")
    end

    @testset "Required fields present" begin
        for f in required_fields
            @test hasfield(typeof(measurements), f)
        end
        println("  ✓ All $(length(required_fields)) required fields confirmed present.")
    end
end

# ============================================================================
# 3. Run get_means() and verify derived observables
# ============================================================================
println("\n" * "─" ^ 50)
println("Phase 3: Running get_means() — deriving observables")
println("─" ^ 50)

means_dict = get_means(lattice_data; std_scheme="correlation")

println("  get_means() returned $(length(keys(means_dict))) keys.\n")

@testset "get_means() derived observables" begin

    # --- Basic observables ---
    @testset "Basic observables present and finite" begin
        for key in ["n", "Kx", "Ky", "DdD", "Delta_mean"]
            @test haskey(means_dict, key)
            @test isfinite(means_dict[key]["value"])
            @test isfinite(means_dict[key]["error"])
        end
        println("  ✓ Basic observables OK.")
    end

    # --- Stiffness (derived from Kx + Lxx_qy) ---
    @testset "Stiffness derived correctly" begin
        @test haskey(means_dict, "stiffness")
        @test isfinite(means_dict["stiffness"]["value"])
        @test isfinite(means_dict["stiffness"]["error"])
        @test haskey(means_dict, "stiffness_lowest_q")
        @test haskey(means_dict, "stiffness_y")
        println("  ✓ Stiffness (x, y, lowest_q) derived OK.")
    end

    # --- Ps (derived from DdD + n) ---
    @testset "Ps derived correctly" begin
        @test haskey(means_dict, "Ps")
        @test isfinite(means_dict["Ps"]["value"])
        @test isfinite(means_dict["Ps"]["error"])

        # Check the formula: Ps = 2*DdD - n + 1
        n_val = means_dict["n"]["value"]
        DdD_val = means_dict["DdD"]["value"]
        expected_Ps_approx = 2 * DdD_val - n_val + 1
        @test abs(means_dict["Ps"]["value"] - expected_Ps_approx) < 0.1  # tolerance due to covariance correction
        println("  ✓ Ps derived OK (value ≈ 2·DdD - n + 1).")
    end

    # --- conductivity_regular_*_fixed (derived from L_w_fixed) ---
    @testset "Conductivity regular fixed derived" begin
        for dir in ["xx", "xy", "yx", "yy"]
            key = "conductivity_regular_$(dir)_fixed"
            @test haskey(means_dict, key)
            val = means_dict[key]["value"]
            @test !isnothing(val)
            # Should be a nested structure: [num_deltas][num_w]
            @test length(val) == length(delta_values_fixed)
            for d in eachindex(val)
                @test length(val[d]) == length(w_conductivity)
                @test all(isfinite, val[d])
            end
        end
        println("  ✓ conductivity_regular_*_fixed derived for all 4 directions × $(length(delta_values_fixed)) deltas × $(length(w_conductivity)) ω-points.")
    end

    # --- Drude weight fixed (derived from L_w_fixed + Kx) ---
    @testset "Drude weight fixed derived" begin
        @test haskey(means_dict, "Drude_weight_fixed")
        dw = means_dict["Drude_weight_fixed"]
        @test length(dw["value"]) == length(delta_values_fixed)
        for d_idx in eachindex(dw["value"])
            @test isfinite(dw["value"][d_idx])
            @test isfinite(dw["error"][d_idx])
        end
        println("  ✓ Drude_weight_fixed derived for $(length(delta_values_fixed)) delta values.")
    end

    # --- Resistivity tensor fixed (from derived conductivity) ---
    @testset "Resistivity tensor fixed derived" begin
        @test haskey(means_dict, "resistivity_tensor_fixed")
        rt = means_dict["resistivity_tensor_fixed"]
        @test length(rt["value"]) == length(delta_values_fixed)
        for d_idx in eachindex(rt["value"])
            if !isnothing(rt["value"][d_idx])
                @test all(isfinite, rt["value"][d_idx])
                @test all(isfinite, rt["error"][d_idx])
            end
        end
        println("  ✓ resistivity_tensor_fixed derived for $(length(delta_values_fixed)) delta values.")
    end

    # --- DC conductivity ---
    @testset "DC conductivity" begin
        @test haskey(means_dict, "conductivity_dc_xx")
        dc = means_dict["conductivity_dc_xx"]
        @test length(dc["value"]) == length(dw_values)
        @test all(isfinite, dc["value"])
        println("  ✓ DC conductivity OK ($(length(dw_values)) Δω values).")
    end

    # --- Energy spacing diagnostics ---
    @testset "Energy spacing observables" begin
        @test haskey(means_dict, "energy_spacing_mean")
        @test isfinite(means_dict["energy_spacing_mean"]["value"])
        @test means_dict["energy_spacing_mean"]["value"] > 0

        @test haskey(means_dict, "delta_mean_diff_low")
        @test isfinite(means_dict["delta_mean_diff_low"]["value"])
        println("  ✓ Energy spacing observables OK.")
    end

    # --- L_w_fixed raw data ---
    @testset "L_w_fixed raw means" begin
        for dir in ["Lxx_w_fixed", "Lxy_w_fixed", "Lyx_w_fixed", "Lyy_w_fixed"]
            @test haskey(means_dict, dir)
            val = means_dict[dir]["value"]
            @test length(val) == length(delta_values_fixed)
        end
        println("  ✓ L_w_fixed raw means present for all 4 directions.")
    end

    # --- Correlation time and num_uncorrelated_measurements ---
    @testset "Correlation time metadata" begin
        for key in ["n", "Kx", "stiffness", "Ps"]
            @test haskey(means_dict[key], "correlation_time")
            @test haskey(means_dict[key], "num_uncorrelated_measurements")
            @test haskey(means_dict[key], "num_measurements")
            @test means_dict[key]["num_measurements"] == lattice_data.current_state.num_current_measurments
        end
        println("  ✓ Correlation time metadata OK.")
    end
end

# ============================================================================
# 4. Summary
# ============================================================================
println("\n" * "=" ^ 70)
println("ALL TESTS PASSED — Slimmed MeasureData is fully functional!")
println("=" ^ 70)
println("\nKey results from 4×4 test simulation:")
println("  <n>          = $(round(means_dict["n"]["value"], digits=4)) ± $(round(means_dict["n"]["error"], digits=4))")
println("  <Kx>         = $(round(means_dict["Kx"]["value"], digits=4)) ± $(round(means_dict["Kx"]["error"], digits=4))")
println("  stiffness    = $(round(means_dict["stiffness"]["value"], digits=4)) ± $(round(means_dict["stiffness"]["error"], digits=4))")
println("  Ps           = $(round(means_dict["Ps"]["value"], digits=4)) ± $(round(means_dict["Ps"]["error"], digits=4))")
println("  Drude(δ=$(delta_values_fixed[1])) = $(round(means_dict["Drude_weight_fixed"]["value"][1], digits=4)) ± $(round(means_dict["Drude_weight_fixed"]["error"][1], digits=4))")
println("  MLS          = $(round(means_dict["energy_spacing_mean"]["value"], digits=5))")
println("  #measurements = $(means_dict["n"]["num_measurements"])")
println("  #means keys   = $(length(keys(means_dict)))")
