using JLD2

const MAG_DIR = joinpath(get(ENV, "V6_CLUSTER_RESULTS", ""), "Magnetic_Field")

safe(d, k, default=missing) = (d isa Dict && haskey(d, k)) ? d[k] : default

function summarize_value_set(vs)
    vs = collect(vs)
    if isempty(vs)
        return "(none)"
    end
    if length(vs) <= 12
        return "[" * join(string.(sort(vs)), ", ") * "]"
    else
        s = sort(vs)
        return "[$(s[1]), $(s[2]), ..., $(s[end-1]), $(s[end])]  (n=$(length(s)))"
    end
end

# Group entries by a parameter signature, then report B/T sweeps within each group.
function inspect_file(path::String)
    fname = basename(path)
    println("="^100)
    println(fname)
    println("="^100)

    JLD2.jldopen(path, "r") do f
        entry_keys = collect(keys(f))
        println("Total entries: ", length(entry_keys))

        # Group by (Lx, Ly, gauge_choice, U, wanted_n, diag_method, pairing_symmetry, boundary_condition)
        groups = Dict{NTuple{8,Any}, Vector{String}}()
        entries_cache = Dict{String,Any}()

        for k in entry_keys
            entry = try
                f[k]
            catch err
                @warn "failed to read entry" file=fname key=k err
                continue
            end
            entries_cache[k] = entry
            sig = (
                safe(entry, "Lx"),
                safe(entry, "Ly"),
                safe(entry, "gauge_choice"),
                safe(entry, "U"),
                safe(entry, "wanted_n"),
                safe(entry, "diag_method"),
                safe(entry, "pairing_symmetry"),
                safe(entry, "boundary_condition"),
            )
            push!(get!(groups, sig, String[]), k)
        end

        for (sig, ks) in sort(collect(groups); by = x -> string(x[1]))
            (Lx, Ly, gauge, U, n_t, diag, sym, bc) = sig
            println("\n--- group: Lx=$Lx Ly=$Ly gauge=$gauge U=$U n=$n_t method=$diag pair=$sym bc=$bc")
            println("    n_entries: ", length(ks))

            Bs = Set{Any}()
            Ts = Set{Any}()
            betas = Set{Any}()
            n_seeds_set = Set{Any}()
            num_meas_set = Set{Any}()
            num_thermal_set = Set{Any}()
            n_unc_set = Set{Any}()
            guess_set = Set{Any}()
            disorder_set = Set{Any}()
            p_set = Set{Any}()
            Lc_set = Set{Any}()
            delta_grids = Set{String}()
            dw_grids = Set{String}()
            wlist_grids = Set{String}()
            obs_sets = Dict{String, Vector{String}}()  # canonical-keys-string => list of entry keys
            n_converged_count = 0
            n_finished_count = 0

            for k in ks
                e = entries_cache[k]
                e isa Dict || continue
                push!(Bs, safe(e, "B"))
                b = safe(e, "beta")
                push!(betas, b)
                if b isa Number && b > 0
                    push!(Ts, round(1/b; digits=6))
                end
                seeds = safe(e, "seeds", missing)
                push!(n_seeds_set, seeds isa AbstractArray ? length(seeds) : missing)
                push!(num_meas_set, safe(e, "num_measure"))
                push!(num_thermal_set, safe(e, "num_thermal"))
                push!(n_unc_set, safe(e, "num_uncorrelated_measurments"))
                push!(guess_set, safe(e, "guess"))
                push!(disorder_set, safe(e, "U_disorder"))
                push!(p_set, safe(e, "impurity"))
                push!(Lc_set, safe(e, "TClustersize"))

                df = safe(e, "delta_fixed_values", missing)
                if df isa AbstractArray
                    push!(delta_grids, "[" * join(string.(df), ", ") * "]  (n=$(length(df)))")
                end
                dw = safe(e, "dw_values", missing)
                if dw isa AbstractArray
                    push!(dw_grids, "[" * join(string.(dw), ", ") * "]  (n=$(length(dw)))")
                end
                wl = safe(e, "w_list", missing)
                if wl isa AbstractArray
                    if length(wl) <= 6
                        push!(wlist_grids, "[" * join(string.(wl), ", ") * "]  (n=$(length(wl)))")
                    else
                        push!(wlist_grids, "[$(wl[1])..$(wl[end])]  (n=$(length(wl)))")
                    end
                end

                conv = safe(e, "convergence", missing)
                if conv isa Dict && safe(conv, "converged", false) === true
                    n_converged_count += 1
                end
                sims = safe(e, "simulations", missing)
                if sims isa Dict
                    for (_, se) in sims
                        se isa Dict || continue
                        if string(safe(se, "status", "")) == "finished"
                            n_finished_count += 1
                        end
                    end
                end

                means = safe(e, "means", missing)
                if means isa Dict
                    sk = sort(string.(collect(keys(means))))
                    canonical = join(sk, "\n")
                    push!(get!(obs_sets, canonical, String[]), k)
                end
            end

            println("    Lc (TCA cluster size): ", summarize_value_set(Lc_set))
            println("    B values: ", summarize_value_set(Bs))
            println("    T values: ", summarize_value_set(Ts))
            println("    beta values: ", summarize_value_set(betas))
            println("    n_seeds: ", summarize_value_set(n_seeds_set))
            println("    num_measure: ", summarize_value_set(num_meas_set))
            println("    num_thermal: ", summarize_value_set(num_thermal_set))
            println("    num_uncorrelated_measurments: ", summarize_value_set(n_unc_set))
            println("    guess: ", summarize_value_set(guess_set))
            println("    U_disorder: ", summarize_value_set(disorder_set))
            println("    impurity p: ", summarize_value_set(p_set))
            println("    n_entries with convergence==true: ", n_converged_count)
            println("    seed-status finished count (sum over entries): ", n_finished_count)

            println("    delta_fixed_values variants: ", length(delta_grids))
            for (i, dg) in enumerate(sort(collect(delta_grids)))
                println("       δ#$i: ", dg)
            end
            println("    dw_values variants: ", length(dw_grids))
            for (i, dg) in enumerate(sort(collect(dw_grids)))
                println("       dw#$i: ", dg)
            end
            println("    w_list variants: ", length(wlist_grids))
            for (i, wg) in enumerate(sort(collect(wlist_grids)))
                println("       w#$i: ", wg)
            end

            # Observable sets in this group
            println("    observable-set variants: ", length(obs_sets))
            for (i, (canon, ks_in_set)) in enumerate(sort(collect(obs_sets); by = x -> -length(x[2])))
                ks_sorted = sort(ks_in_set)
                preview = length(ks_sorted) <= 6 ? join(ks_sorted, ", ") : join(ks_sorted[1:5], ", ") * ", ... (+$(length(ks_sorted)-5) more)"
                names = split(canon, "\n")
                println("\n      observables-set #$i  ($(length(ks_in_set)) entries: $preview)")
                println("        n_obs = $(length(names))")
                # Print observables 4 per line
                row = String[]
                for nm in names
                    push!(row, nm)
                    if length(row) == 4
                        println("          ", join(row, "  |  "))
                        empty!(row)
                    end
                end
                if !isempty(row)
                    println("          ", join(row, "  |  "))
                end
            end
        end
    end
    println()
end

function main()
    files = sort([joinpath(MAG_DIR, f) for f in readdir(MAG_DIR) if endswith(f, ".jld2")])
    println("Found ", length(files), " files in ", MAG_DIR)
    println()
    for f in files
        try
            inspect_file(f)
        catch err
            println("FAILED on $(basename(f)): $err")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
