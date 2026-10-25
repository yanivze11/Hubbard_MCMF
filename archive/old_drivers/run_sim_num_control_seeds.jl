include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

using ArgParse

function summarize_seed_convergence(path::String; print_details::Bool=true)
    sim_info = load_sim_info(path)

    sim_keys = filter(k -> k != "updated_at", collect(keys(sim_info)))
    total_sims = length(sim_keys)
    sims_with_mu_conv = 0
    fully_converged = 0

    for sim_num in sort(sim_keys, by = x -> something(tryparse(Int, x), typemax(Int)))
        sim = sim_info[sim_num]
        means = get(sim, "means", Dict{String, Any}())

        if !(means isa Dict{String, Any}) || !haskey(means, "mu_convergence")
            continue
        end

        mu_conv = means["mu_convergence"]
        sims_with_mu_conv += 1

        num_converged = get(mu_conv, "num_converged", 0)
        num_seeds = get(mu_conv, "num_seeds", 0)
        target_n = get(mu_conv, "target_n", "N/A")
        mu_mean = get(mu_conv, "mu_mean", "N/A")
        mu_std = get(mu_conv, "mu_std", "N/A")

        if num_seeds > 0 && num_converged == num_seeds
            fully_converged += 1
        end

        if print_details
            println("sim=$sim_num target_n=$target_n converged=$num_converged/$num_seeds mu_mean=$mu_mean mu_std=$mu_std")
        end
    end

    println("----------------------------------------")
    println("Seed convergence summary")
    println("Total simulations: $total_sims")
    println("With mu_convergence stats: $sims_with_mu_conv")
    println("Fully converged (all seeds): $fully_converged")
    println("----------------------------------------")
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--path"
            help = "Path to simulation folder"
            required = true
        "--skip-update"
            help = "Skip update_means and only summarize existing num_control/sim_info.jld2"
            action = :store_true
        "--quiet"
            help = "Print summary only"
            action = :store_true
    end

    args = parse_args(ARGS, s)
    path = args["path"]

    if endswith(path, "num_control")
        println("Path ends with 'num_control', switching to parent directory for simulation search.")
        path = dirname(path)
    end

    if !args["skip-update"]
        println("Updating means at path: $path")
        update_means(path)
    end

    summarize_seed_convergence(path; print_details=!args["quiet"])
end

main()
