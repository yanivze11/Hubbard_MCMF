include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))
# data = load_results("<Curie folder>\\MC_Hubbard_V6\\data\\FullSys__11-02-2026\\6\\seed_1886766");

begin
    L = 10;
    my_mls = 4 / L^2;
    println("-----------------------------------------")
    println("Running for L = $L")

    data = running_simulation_once(L, L, 1,
    4.0, 1.0, 0.0, 1234,
    0.0, 20.0, 0.0,
    100, 0, 1,
    [-4.0:0.1:4.0;], "uniform", "uniform", "K0", "s-wave", 
    [10^a for a in range(-1, 1, 12)]*my_mls, 0.0;
    savedata=false,
    delta_values_fixed = [0.5; 1:10;] * (4/L^2),
    delta_values_stiffness = [0.0; 0.5; 1:10;] * (4/L^2),
    dw_values  = [0.5; 1:10;] * (4/L^2)
    );
    println("Simulation completed ")
end



begin
    my_data = data;
    println("-------------------------------")
    println("Calculating memory sizes of observables...")
    meas = my_data.measurement_data;
    obs_names = fieldnames(typeof(meas));
    obs_sizes = Dict{Symbol, Int}();

    for field in obs_names
        # Extract the actual value stored inside the field
        field_value = getfield(meas, field)
        
        # Calculate the memory footprint
        size_in_bytes = Base.summarysize(field_value)
        
        # println("Field: $field | Size: $size_in_bytes bytes")
        obs_sizes[field] = size_in_bytes
    end;
    total_size = sum(values(obs_sizes));
    println("Total memory size of all observables: $(format_size(total_size)) ")
    # Print the results
    println("Observables Sizes:")
    # the sorted list of observables by size
    sorted_obs = sort(collect(obs_sizes), by=x -> x[2], rev=true)
    for (obs, size) in sorted_obs
        println("  $obs: $(format_size(size)) = $(round(size/total_size*100, sigdigits=2)) % of total")
    end
end


begin
    reduct_obs_list = [:conductivity_regular_xy_fixed, :conductivity_regular_yx_fixed, :conductivity_regular_xx_fixed, :conductivity_regular_yy_fixed,
                        :Lyx_D_fixed, :Lxy_D_fixed, :Lxx_D_fixed, :Lyy_D_fixed,
                        :Lyx_qy_delta, :Lxy_qy_delta, :Lxx_qy_delta, :Lyy_qy_delta,
                        :Lyx_qx_delta, :Lxy_qx_delta, :Lxx_qx_delta, :Lyy_qx_delta,
                        :Lyx_w, :Lxy_w, :Lxx_w, :Lyy_w,
                        :conductivity_regular_yy, :conductivity_regular_xx, :conductivity_regular_yx, :conductivity_regular_xy,
                        :Lyx_D, :Lxy_D, :Lxx_D, :Lyy_D
    ]
    new_sizes = Dict{Symbol, Int}();
    println("Reduced list of observables (for plotting):")
    for obs in keys(obs_sizes)
        if !(obs in reduct_obs_list)
            new_sizes[obs] = obs_sizes[obs]
            # println("  $obs: $(format_size(obs_sizes[obs]))")
        end
    end

    total_reduced_size = sum(values(new_sizes));
    println("Total memory size of reduced observables: $(format_size(total_reduced_size)) ")
    println("Reduced Observables Sizes:")
    sorted_reduced_obs = sort(collect(new_sizes), by=x -> x[2], rev=true)
    for (obs, size) in sorted_reduced_obs
        println("  $obs: $(format_size(size)) = $(round(size/total_reduced_size*100, sigdigits=2)) % of total reduced")
    end
end
