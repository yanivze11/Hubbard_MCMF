# test_get_means.jl — runs a small 4×4 K0 simulation in memory (1000 measurements)
# and checks that get_means returns every observable with value, error, correlation
# time and (uncorrelated) measurement counts.
using Test
include(joinpath(@__DIR__, "..", "sim_num_control.jl"))
include(joinpath(@__DIR__, "..", "statistics.jl"))


# Test with a real small simulation (4x4, 1000 measurements)

function run_small_simulation_and_test(;NIL=false)
    Lx, Ly = 4, 4
    num_measurements = 1000
    data = running_simulation_once(Lx, Ly, 1, 4.0, 1.0, 0.0, 1234, -0.65, 20.0, 0.0,
        num_measurements, 0, 1, [-3.0:0.1:3.0;], "uniform", "uniform", "K0", "s-wave",
        [0.0:0.01:1.0;], 0.0;
        directory="", num_cores=1, savedata=false, saving_interval=100, dump_time=Dates.Hour(1),
        saveplots=false, date_folder=true, numbered=true, seed_folder=false,
        U_puddle_params=[0.0, 6.0, 0.7, 0.7], U_smooth=(false, 0.8), loop_random=false,
        boundary_conditions="PBC", gauge_choice="LandauX"
    )
    means = get_means(data)
    for (obs, dict) in means
        if isa(dict, Dict)
            @test haskey(dict, "value")
            @test haskey(dict, "error")
            @test haskey(dict, "correlation_time")
            @test haskey(dict, "num_uncorrelated_measurements")
            @test haskey(dict, "num_measurements")
        end
    end
end

@testset "get_means on small 4x4 simulation" begin
    run_small_simulation_and_test()
end


begin
    Lx = 4
    Ly = 4
    num_measurements = 1
    num_thermal = 0
    measuring_interval = 1
    data = running_simulation_once(Lx, Ly, 1, 0.0, 1.0, 0.0, 1234, -0.65, 20.0, 0.0,
        num_measurements, num_thermal, measuring_interval, [-3.0:0.1:3.0;], "uniform", "uniform", "K0", "s-wave",
        [0.0:0.01:1.0;], 0.0;
        directory="", num_cores=1, savedata=false, saving_interval=100, dump_time=Dates.Hour(1),
        saveplots=false, date_folder=true, numbered=true, seed_folder=false,
        U_puddle_params=[0.0, 6.0, 0.7, 0.7], U_smooth=(false, 0.8), loop_random=false,
        boundary_conditions="PBC", gauge_choice="Mixed"
    );
    means = get_means(data)
    
end;


means = get_means(data)


begin
    
    fig = plot(xlabel="ω", ylabel="σ_μν = Im[Λ_μν(ω)] / ω", title="Conductivity tensor components")
    plot!(fig, data.sim_data.w_conductivity_range, means["conductivity_regular_xx"]["value"], yerr=means["conductivity_regular_xx"]["error"],label="σxx")
    plot!(fig, data.sim_data.w_conductivity_range, -means["conductivity_regular_xy"]["value"], yerr=means["conductivity_regular_xy"]["error"],label="-σxy")
    plot!(fig, data.sim_data.w_conductivity_range, means["conductivity_regular_yx"]["value"], yerr=means["conductivity_regular_yx"]["error"],label="σyx")
    plot!(fig, data.sim_data.w_conductivity_range, means["conductivity_regular_yy"]["value"], yerr=means["conductivity_regular_yy"]["error"],label="σyy")
end

begin
    display(means["resistivity_tensor"])
    display(means["resistivity_regular"])
end

begin
    fig = plot(xlabel="ω", ylabel="ρ_μν = Im[Λ_μν(ω)] / ω", title="Resistivity tensor components")
    scatter!(fig, [1], [means["resistivity_tensor"]["value"][1]], yerr=[means["resistivity_tensor"]["error"][1]], label="ρxx", ms=4)
    scatter!(fig, [2], [means["resistivity_tensor"]["value"][2]], yerr=[means["resistivity_tensor"]["error"][2]], label="ρxy", ms=4)
    scatter!(fig, [2], [-means["resistivity_tensor"]["value"][3]], yerr=[means["resistivity_tensor"]["error"][3]], label="-ρyx", ms=4)
    scatter!(fig, [1], [means["resistivity_tensor"]["value"][4]], yerr=[means["resistivity_tensor"]["error"][4]], label="ρyy", ms=4)
    scatter!(fig, [1], [means["resistivity_regular"]["value"]], yerr=[means["resistivity_regular"]["error"]], label="ρ_xx (assuming σxx=σyy, σxy=-σyx)", m=:square, ms=4)
    scatter!(fig, [2], [means["resistivity_hall_regular"]["value"]], yerr=[means["resistivity_hall_regular"]["error"]], label="ρ_xy (assuming σxx=σyy, σxy=-σyx)", m=:square, ms=4)
end