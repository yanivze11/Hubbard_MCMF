# test_load_results_migration.jl — load_results on old or damaged files: old
# drude_dc_* keys renamed to L_D_dc_*, empty chunks skipped and checkpoint counters
# reconciled, and MeasureData fields of the wrong type replaced by NaNs.
using Test
using JLD2

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

function make_sim_data_for_test(save_dir::String)
    return SimData(
        Lx=2,
        Ly=2,
        SC_L=1,
        U=0.0,
        t=1.0,
        V=0.0,
        seed=123,
        mu=0.0,
        beta=2.0,
        num_measure=1,
        num_thermal=0,
        measuring_interval=1,
        saving_interval=1,
        w_list=[-1.0, 1.0],
        guess="uniform",
        U_disorder="uniform",
        diag_method="K0",
        pairing_symmetry="s-wave",
        directory=save_dir,
        num_cores=1,
        num_threads=1,
        w_conductivity_range=[0.1, 0.2],
        delta_values_fixed=[0.1],
        delta_values_stiffness=[0.1],
        U_disorder_params=[0.0, 10.0, 0.7, 0.7],
        U_impurity=0.0,
        TCluster_size_x=1,
        TCluster_size_y=1,
        smooth_U=(false, 0.8),
        loop_random=false,
        B=0.0,
        gauge_choice="LandauX",
        boundary_conditions="PBC",
        dw_values=[0.1, 0.2],
    )
end

function make_measurement_chunk(use_old_names::Bool, ss_val::Float64, offset::Float64)
    xx_key = use_old_names ? "drude_dc_xx" : "L_D_dc_xx"
    xy_key = use_old_names ? "drude_dc_xy" : "L_D_dc_xy"
    yx_key = use_old_names ? "drude_dc_yx" : "L_D_dc_yx"
    yy_key = use_old_names ? "drude_dc_yy" : "L_D_dc_yy"

    return Dict(
        "Ss" => [ss_val],
        xx_key => [[offset + 0.1, offset + 0.2]],
        xy_key => [[offset + 1.1, offset + 1.2]],
        yx_key => [[offset + 2.1, offset + 2.2]],
        yy_key => [[offset + 3.1, offset + 3.2]],
    )
end

function write_mixed_results_file(dir::String; old_first::Bool)
    sim_data = make_sim_data_for_test(dir)
    sim_data_dict = struct2dict(sim_data)

    chunk1 = make_measurement_chunk(old_first, 1.0, 0.0)
    chunk2 = make_measurement_chunk(!old_first, 2.0, 10.0)

    results_path = joinpath(dir, "results.jld2")
    jldopen(results_path, "w") do file
        file["measurement_1"] = chunk1
        file["measurement_2"] = chunk2
        file["sim_data"] = sim_data_dict
    end
    return results_path
end

function write_complex_mismatch_results_file(dir::String)
    sim_data = make_sim_data_for_test(dir)
    sim_data_dict = struct2dict(sim_data)

    chunk1 = Dict(
        "Ss" => [1.0],
        "L_D_dc_xx" => [[1.0 + 0.2im, 2.0 - 0.3im]],
    )
    chunk2 = Dict(
        "Ss" => [2.0],
        "L_D_dc_xx" => [[10.0 + 0.1im, 20.0 + 0.4im]],
    )

    results_path = joinpath(dir, "results.jld2")
    jldopen(results_path, "w") do file
        file["measurement_1"] = chunk1
        file["measurement_2"] = chunk2
        file["sim_data"] = sim_data_dict
    end
    return results_path
end

@testset "load_results migration for mixed drude/L_D keys" begin
    for old_first in (true, false)
        mktempdir() do temp_dir
            write_mixed_results_file(temp_dir; old_first=old_first)

            data = load_results(temp_dir; to_load_checkpoint=false)
            @test data !== nothing

            md = data.measurement_data
            @test length(md.Ss) == 2
            @test md.Ss == [1.0, 2.0]

            @test length(md.L_D_dc_xx) == 2
            @test length(md.L_D_dc_xy) == 2
            @test length(md.L_D_dc_yx) == 2
            @test length(md.L_D_dc_yy) == 2

            @test isapprox(md.L_D_dc_xx[1][1], 0.1; atol=1e-12)
            @test isapprox(md.L_D_dc_xx[1][2], 0.2; atol=1e-12)
            @test isapprox(md.L_D_dc_xx[2][1], 10.1; atol=1e-12)
            @test isapprox(md.L_D_dc_xx[2][2], 10.2; atol=1e-12)

            @test isapprox(md.L_D_dc_yy[1][1], 3.1; atol=1e-12)
            @test isapprox(md.L_D_dc_yy[2][1], 13.1; atol=1e-12)
        end
    end
end

@testset "empty chunk handling and counter reconciliation" begin
    mktempdir() do temp_dir
        results_path = joinpath(temp_dir, "results.jld2")
        jldopen(results_path, "w") do file
            file["measurement_1"] = Dict("Ss" => [0.0, 0.0, 0.0])
            file["measurement_2"] = Dict("Ss" => [1.0, 2.0])
            file["measurement_3"] = Dict("Ss" => [0.0])
        end

        @test count_dumped_measurements(results_path) == 2
    end

    # Stale checkpoint after successful dump: keep num_current, clear saving index.
    @test reconcile_measurement_counters(1000, 500, 1000) == (1000, 0)

    # Interrupted before dump completion: rollback to what is actually dumped.
    @test reconcile_measurement_counters(1000, 500, 500) == (500, 0)

    # Results file ahead of checkpoint: raise num_current to on-disk count.
    @test reconcile_measurement_counters(500, 0, 800) == (800, 0)
end

@testset "NaN fallback for mismatched MeasureData field types" begin
    mktempdir() do temp_dir
        results_path = write_complex_mismatch_results_file(temp_dir)
        mtime_before = stat(results_path).mtime

        data = load_results(temp_dir; to_load_checkpoint=false)
        @test data !== nothing

        md = data.measurement_data
        @test md.Ss == [1.0, 2.0]
        @test length(md.L_D_dc_xx) == 2
        @test all(isnan, md.L_D_dc_xx[1])
        @test all(isnan, md.L_D_dc_xx[2])

        mtime_after = stat(results_path).mtime
        @test mtime_before == mtime_after
    end
end

println("test_load_results_migration.jl passed")