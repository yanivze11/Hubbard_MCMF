# ============================================================================
# tests/dryrun_means_db_cluster.jl — exercise means_db against REAL cluster data.
#
#   julia -O1 tests/dryrun_means_db_cluster.jl [--campaign <path>] [--limit N]
#
# Reads each simulation's `sim.jld2` (parameters + per-seed tracking) and each
# seed's existing `means_cache.jld2` (per-seed means), then runs them through
# the real write path: identity -> save_entry -> index -> query -> read back.
#
# It deliberately does NOT load results.jld2 (~369 MB per seed), because the
# point is to test the database plumbing on real parameters, not to recompute
# physics. That makes it a seconds-long check instead of an hours-long one.
#
# SAFETY: the cluster mounts are read-only research data. This script only ever
# opens files there with "r", and writes exclusively under a local temp root.
# ============================================================================
using JLD2, Printf, Dates

include(joinpath(@__DIR__, "..", "means_db.jl"))

"Campaign folder to read: `--campaign <path>`, else env `DRYRUN_CAMPAIGN` (no default)."
const DEFAULT_CAMPAIGN = get(ENV, "DRYRUN_CAMPAIGN", "")
const TEST_ROOT = joinpath(tempdir(), "means_db_dryrun")

function parse_args(args)
    campaign = DEFAULT_CAMPAIGN; limit = 8
    for (i, a) in enumerate(args)
        a == "--campaign" && i < length(args) && (campaign = args[i+1])
        a == "--limit"    && i < length(args) && (limit = parse(Int, args[i+1]))
    end
    isempty(campaign) && error("pass --campaign <V6 campaign folder> or set ENV[\"DRYRUN_CAMPAIGN\"]")
    return (campaign=campaign, limit=limit)
end

"""Build an old-style entry + per-seed blocks for one simulation directory."""
function read_simulation(simdir::AbstractString)
    simjld = joinpath(simdir, "sim.jld2")
    isfile(simjld) || return nothing

    entry = jldopen(simjld, "r") do f
        d = Dict{String,Any}()
        for k in keys(f); d[string(k)] = f[k]; end
        d
    end
    tracking = get(entry, "simulations", Dict())

    seed_data = Dict{String,Any}()
    for (seed, sd) in (tracking isa AbstractDict ? tracking : Dict())
        sd isa AbstractDict || continue
        cache = joinpath(simdir, "seed_$(seed)", "means_cache.jld2")
        isfile(cache) || continue
        m = try
            jldopen(g -> haskey(g, "means") ? g["means"] : nothing, cache, "r")
        catch e
            @warn "unreadable means_cache" cache exception=e; nothing
        end
        m isa AbstractDict || continue

        info = Dict{String,Any}("seed" => seed)
        for fld in SEED_INFO_FIELDS
            haskey(sd, fld) && (info[fld] = sd[fld])
        end
        seed_data[string(seed)] = Dict{String,Any}("info" => info, "means" => m)
    end
    isempty(seed_data) && return nothing

    # Seed-average. With one seed that IS the mean; with several, this dry run
    # uses the first seed's means as a stand-in (averaging lives in
    # sim_num_control.jl and is not what we are testing here).
    ks = sort(collect(keys(seed_data)))
    entry["means"] = seed_data[ks[1]]["means"]
    length(ks) > 1 && @printf("    note: %d seeds, using seed %s means as stand-in\n", length(ks), ks[1])
    return (entry=entry, seed_data=seed_data)
end

function main()
    o = parse_args(ARGS)
    println("campaign : ", o.campaign)
    println("db root  : ", TEST_ROOT)
    isdir(o.campaign) || (println("\nCampaign not reachable — is the mount connected?"); return)
    rm(TEST_ROOT; recursive=true, force=true)

    simdirs = filter(d -> isdir(joinpath(o.campaign, d)), readdir(o.campaign))
    sort!(simdirs, by = d -> something(tryparse(Int, d), typemax(Int)))
    simdirs = simdirs[1:min(o.limit, length(simdirs))]
    @printf("\nreading %d simulation(s) from the cluster (read-only)\n", length(simdirs))

    written = String[]; skipped = 0
    t_read = @elapsed for d in simdirs
        r = read_simulation(joinpath(o.campaign, d))
        if r === nothing; skipped += 1; @printf("  sim %-5s  no usable means_cache — skipped\n", d); continue; end
        try
            p = save_entry(r.entry; root=TEST_ROOT, seed_data=r.seed_data, source=basename(o.campaign))
            push!(written, p)
            @printf("  sim %-5s  U=%-5s beta=%-6s L=%sx%s  seeds=%d  ->  %s\n",
                    d, string(get(r.entry,"U","?")), string(get(r.entry,"beta","?")),
                    string(get(r.entry,"Lx","?")), string(get(r.entry,"Ly","?")),
                    length(r.seed_data), relpath(p, TEST_ROOT))
        catch e
            @printf("  sim %-5s  WRITE FAILED: %s\n", d, first(split(sprint(showerror, e), '\n')))
        end
    end
    @printf("\nwrote %d file(s) in %.1fs (%d skipped)\n", length(written), t_read, skipped)
    isempty(written) && return

    sz = sum(filesize.(written)) / length(written)
    @printf("average file size: %.0f KB\n", sz/1024)

    println("\n--- index ---")
    rebuild_index(; root=TEST_ROOT, verbose=false)
    db_summary(; root=TEST_ROOT)

    println("\n--- read back ---")
    idx = load_index(; root=TEST_ROOT)
    U1 = idx["U"][1]
    sel = find_entries(; root=TEST_ROOT, U=U1)
    @printf("find_entries(U=%s) -> %d\n", string(U1), length(sel))

    t_two = @elapsed si = load_means(; root=TEST_ROOT, U=U1, obs=[:n, :stiffness], verbose=false)
    t_all = @elapsed load_means(; root=TEST_ROOT, U=U1, obs=:all, verbose=false)
    @printf("obs=[:n,:stiffness] : %.3f s   obs=:all : %.3f s   (%.1fx)\n",
            t_two, t_all, t_all/max(t_two,1e-9))

    e = first(values(si))
    @printf("\nsample entry: U=%s beta=%s n=%.4f  keys(means)=%s\n",
            string(e["U"]), string(e["beta"]),
            get(get(e["means"], "n", Dict()), "value", NaN),
            string(sort(collect(keys(e["means"])))))

    println("\n--- per-seed data survived ---")
    seeds = read_seeds(written[1])
    for (s, d) in seeds
        @printf("  seed %-10s num_meas=%-7s mu=%-8s obs=%d\n", s,
                string(get(d["info"], "num_current_measurments", "?")),
                string(round(get(d["info"], "mu", NaN), digits=4)),
                haskey(d, "means") ? length(d["means"]) : 0)
    end
    fs = first(keys(seeds))
    @printf("  seed_is_current(%s, its own num_meas) = %s\n", fs,
            seed_is_current(written[1], fs,
                            Int(get(seeds[fs]["info"], "num_current_measurments", 0))))

    println("\n--- identity is stable against a NEW parameter ---")
    r = read_simulation(joinpath(o.campaign, simdirs[1]))
    if r !== nothing
        h0 = params_hash(r.entry)
        e2 = copy(r.entry); e2["some_future_knob"] = 3.7
        e3 = copy(r.entry); delete!(e3, "V")
        @printf("  base hash              : %s\n", h0)
        @printf("  + unregistered param   : %s  %s\n", params_hash(e2), params_hash(e2)==h0 ? "OK" : "CHANGED!")
        @printf("  - V (defaults to 0.0)  : %s  %s\n", params_hash(e3), params_hash(e3)==h0 ? "OK" : "CHANGED!")
    end

    @printf("\ntest db left at %s (delete when done)\n", TEST_ROOT)
end

main()
