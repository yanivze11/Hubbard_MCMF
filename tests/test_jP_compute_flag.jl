# Test: the compute_jP flag gates the FT paramagnetic current observable.
#
# Verifies that with compute_jP=false (the DEFAULT) the jP observable is really
# not computed: fields are allocated empty, make_measurement_K! writes nothing to
# them and does not error, and prepare_dump handles them (dumps empty, no slicing).
# A positive control with compute_jP=true confirms the fields are populated.

using Test

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

const JP_FIELDS = (:jPx_q0, :jPx_qy, :jPx_qx, :jPy_q0, :jPy_qy, :jPy_qx)

function build(; compute_jP::Bool, B::Float64=0.0, saving_interval::Int=3)
    sim = SimData(
        Lx=4, Ly=4, SC_L=1,
        U=4.0, t=1.0, V=0.0, seed=1234, mu=0.0, beta=8.0,
        num_measure=saving_interval, num_thermal=0,
        measuring_interval=1, saving_interval=saving_interval,
        w_list=[0.0],
        guess="uniform", U_disorder="uniform",
        diag_method="K0", pairing_symmetry="s-wave",
        directory=pwd(), num_cores=1, num_threads=1,
        w_conductivity_range=[0.0],
        delta_values_fixed=[0.1], delta_values_stiffness=[0.1],
        U_impurity=0.0,
        loop_random=false,
        B=B, gauge_choice="Mixed", boundary_conditions="PBC",
        dw_values=[0.1],
        compute_jP=compute_jP,
    )
    return sim, Lattice_Data(sim)
end

println("=" ^ 60)
println("Test: compute_jP flag gating")
println("=" ^ 60)

@testset "Default is OFF" begin
    sim_default = SimData(Lx=4, Ly=4, U=4.0, beta=8.0, num_measure=1, num_thermal=0)
    @test sim_default.compute_jP == false
end

@testset "compute_jP = false: not computed" begin
    sim, data = build(compute_jP=false, B=0.0)
    @test sim.compute_jP == false

    # Allocated empty.
    for f in JP_FIELDS
        @test isempty(getfield(data.measurement_data, f))
    end

    # Run real measurements: must not error and must leave jP empty.
    for i in 1:sim.saving_interval
        MC_stepV2!(data)
        make_measurement_K!(data, i)
    end
    for f in JP_FIELDS
        @test isempty(getfield(data.measurement_data, f))
    end

    # prepare_dump must succeed and emit the jP fields as empty (no slicing crash).
    data.current_state.saving_measurement_ix = sim.saving_interval
    data.current_state.num_current_measurments = sim.saving_interval
    dump = prepare_dump(data)
    @test dump !== nothing
    @test length(dump["Ss"]) == sim.saving_interval
    for f in JP_FIELDS
        @test haskey(dump, string(f))
        @test length(dump[string(f)]) == 0
    end
    println("  OK: jP empty, measurement + dump clean with compute_jP=false")
end

@testset "compute_jP = true: computed (positive control)" begin
    sim, data = build(compute_jP=true, B=2π / (4 * 4))  # minimal flux, Mixed gauge
    @test sim.compute_jP == true

    # Allocated full-size.
    @test length(data.measurement_data.jPx_q0) == sim.saving_interval
    @test length(data.measurement_data.jPx_qy[1]) == div(4, 2)
    @test length(data.measurement_data.jPx_qx[1]) == div(4, 2)

    for i in 1:sim.saving_interval
        MC_stepV2!(data)
        make_measurement_K!(data, i)
    end

    m = data.measurement_data
    # Values are finite and the computation actually wrote something (B != 0 -> nonzero).
    all_vals = vcat(
        m.jPx_q0, m.jPy_q0,
        reduce(vcat, m.jPx_qy), reduce(vcat, m.jPx_qx),
        reduce(vcat, m.jPy_qy), reduce(vcat, m.jPy_qx),
    )
    @test all(isfinite, real.(all_vals))
    @test all(isfinite, imag.(all_vals))
    @test maximum(abs.(all_vals)) > 0.0
    println("  OK: jP populated, max|jP| = $(round(maximum(abs.(all_vals)), sigdigits=4))")
end

println("\nAll compute_jP flag tests passed.")
