# run_sim_num_filtered.jl
# Run update_means_filtered on the cluster for a specified filter of simulations

include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

using ArgParse

function parse_filter_string(filter_str::String)
    filters = Dict{String, Any}()
    for pair in split(filter_str, ",")
        k, v = split(pair, "=")
        # Try to parse as Int or Float, fallback to String
        if occursin(".", v)
            val = tryparse(Float64, v)
            if isnothing(val)
                val = v
            end
        else
            val = tryparse(Int, v)
            if isnothing(val)
                val = v
            end
        end
        filters[k] = val
    end
    return filters
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--path"
            help = "Path to the simulation folder"
            required = true
        "--filter"
            help = "Filter for simulations in the form key1=val1,key2=val2,... (e.g. Lx=12,mu=2.0)"
            required = true
    end
    args = ArgParse.parse_args(s; as_symbols=false)
    path = args["path"]
    # If path ends with "num_control", use its parent directory
    if endswith(path, "num_control")
        println("Path ends with 'num_control', switching to parent directory for simulation search.")
        path = dirname(path)
    end
    filter_str = args["filter"]
    filters = parse_filter_string(filter_str)
    println("Running update_means_filtered on path: $path with filters: $filters")
    update_means_filtered(path; filters=filters)
    println("Completed update_means_filtered.")
end

main()
