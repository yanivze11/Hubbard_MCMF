# test_prepare_dump_pairing_smoke.jl — prepare_dump for s-wave and d-wave: the
# streams are cut to the filled length, and the d-wave phase correlations are kept
# for d-wave and stay empty for s-wave.
using Test

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

const DWAVE_PAIRINGS = Set(["d-wave", "d-p-wave", "mayr-d-wave", "d+s-wave"])

function make_smoke_data(pairing_symmetry::String; saving_interval::Int=4)
    sim = SimData(
        Lx=2,
        Ly=2,
        SC_L=1,
        U=2.0,
        t=1.0,
        V=0.0,
        seed=1234,
        mu=0.0,
        beta=2.0,
        num_measure=1,
        num_thermal=0,
        measuring_interval=1,
        saving_interval=saving_interval,
        w_list=[0.0],
        guess="uniform",
        U_disorder="uniform",
        diag_method="K0",
        pairing_symmetry=pairing_symmetry,
        directory=pwd(),
        num_cores=1,
        num_threads=1,
        w_conductivity_range=[0.1],
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
        dw_values=[0.1],
    )

    data = Lattice_Data(sim)
    data.current_state.saving_measurement_ix = 2
    data.current_state.num_current_measurments = 2
    data.measurement_data.Ss[1] = 1.0
    data.measurement_data.Ss[2] = 2.0

    return data
end

function run_prepare_dump_smoke(pairing_symmetry::String)
    data = make_smoke_data(pairing_symmetry)
    dump = prepare_dump(data)

    @test dump !== nothing
    @test length(dump["Ss"]) == 2

    if pairing_symmetry in DWAVE_PAIRINGS
        @test length(dump["Phase_Corr_dwave_x"]) == 2
        @test length(dump["Phase_Corr_dwave_y"]) == 2
        @test length(dump["Phase_Corr_mixed"]) == 2
    else
        @test length(dump["Phase_Corr_dwave_x"]) == 0
        @test length(dump["Phase_Corr_dwave_y"]) == 0
        @test length(dump["Phase_Corr_mixed"]) == 0
    end

    println(
        "OK pairing=", pairing_symmetry,
        " Ss=", length(dump["Ss"]),
        " Phase_Corr_dwave_x=", length(dump["Phase_Corr_dwave_x"]),
    )
end

println("Running prepare_dump pairing smoke tests")
run_prepare_dump_smoke("s-wave")
run_prepare_dump_smoke("d-wave")
println("All smoke tests passed.")
