include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))
include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))
using JLD2
using FileIO
using ArgParse
using Dates

function run_me(path::String)
    starttime = Dates.now()
    println("Starting monitoring simulations at ", starttime)
    # save the structures as dictionaries
    
    while Dates.now() < starttime + Dates.Day(7)  # Run for 1 week maximum
        println("Starting new iteration at ", Dates.now())
        # save the old simulation data as dictionaries

        # Update the means for each simulation
        println("Updating means for simulations in path: ", path)
        update_means(path)
        # Print the simulation details to a text file
        println("Printing simulation details to sim_details.txt")
        print_sim_details(path)
        # Sleep for _ minutes (adjust as needed)
        println("Sleeping for 10 minute...")
        sleep(60 * 10)
    end
    
end


s = ArgParseSettings()
@add_arg_table s begin
     "--sim_path"
         help = "sim_path"
end

parsed_args = parse_args(ARGS, s)
path = parsed_args["sim_path"]

run_me(path)