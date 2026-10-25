# ============================================================================
# migrate_means_db.jl — one-time: means_store.jld2  ->  per-simulation files.
#
# Reads the old derived store one entry at a time (never loads it whole) and
# fans it out into the means_db layout. Does NOT touch the ~15 GB of raw data;
# the store already holds the stripped {metadata + means} we need.
#
# Resumable: entries whose target file already exists are skipped, so an
# interrupted run can simply be re-run.
#
# NOT migrated: per-seed means. `build_means_store` dropped the raw
# `simulations` blob, so they are not in the store. They still exist as
# `means_cache.jld2` inside each seed directory and can be backfilled later
# by a second pass (see backfill_seed_means.jl — not written yet).
#
#   julia -O1 migrate_means_db.jl [--dry-run] [--limit N]
# ============================================================================
using JLD2, Printf, Dates

include(joinpath(@__DIR__, "..", "..", "means_db.jl"))

const OLD_STORE = joinpath(get(ENV, "V6_CLUSTER_RESULTS", ""), "means_store", "means_store.jld2")
const OLD_MANIFEST = joinpath(get(ENV, "V6_CLUSTER_RESULTS", ""), "means_store", "means_store_manifest.csv")

"""
    migrate(; store, root, limit=nothing, dry_run=false) -> NamedTuple

Fan the old store out into `root`. Returns counts and the collision report.
"""
function migrate(; store::String=OLD_STORE, root::String=MEANS_DB_ROOT,
                 manifest::String=OLD_MANIFEST, limit=nothing, dry_run::Bool=false,
                 verbose::Bool=true)

    # id -> source basename, so each new file records where it came from.
    src_of = Dict{String,String}()
    if isfile(manifest)
        for (i, line) in enumerate(eachline(manifest))
            i == 1 && continue
            isempty(line) && continue
            f = split(line, ',')
            length(f) >= 2 && (src_of[f[1]] = f[2])
        end
    end

    ids = jldopen(store, "r") do f
        sort([k for k in keys(f) if !startswith(k, "__")], by = k -> something(tryparse(Int, k), 0))
    end
    limit !== nothing && (ids = ids[1:min(limit, length(ids))])
    @printf("[migrate] %d entries in store -> %s%s\n", length(ids), root,
            dry_run ? "  (DRY RUN)" : "")

    seen      = Dict{String,Int}()      # relpath -> times seen (collision detection)
    collisions = Tuple{String,String,String}[]   # (relpath, source_a, source_b)
    written = 0; skipped = 0; failed = Tuple{String,String}[]
    bytes = 0

    t = @elapsed jldopen(store, "r") do f
        for (i, id) in enumerate(ids)
            local entry
            try
                entry = f[id]
            catch e
                push!(failed, (id, "read: $(sprint(showerror, e))")); continue
            end
            entry isa AbstractDict || (push!(failed, (id, "not a Dict")); continue)
            get(entry, "means", nothing) isa AbstractDict ||
                (push!(failed, (id, "means missing/not a Dict")); continue)

            src = get(src_of, id, "")
            params = canonical_params(entry)
            rp = entry_relpath(params)

            # Same params from two clusters => same hash. Keep both.
            nseen = get(seen, rp, 0) + 1
            seen[rp] = nseen
            if nseen > 1
                push!(collisions, (rp, "", src))
                stem, ext = splitext(rp)
                rp = string(stem, "-dup", nseen, ext)
            end

            path = joinpath(root, rp)
            if isfile(path) && nseen == 1
                skipped += 1; continue
            end
            if !dry_run
                try
                    save_entry(entry; root=root, source=src, path=path)
                    bytes += filesize(path)
                catch e
                    push!(failed, (id, "write: $(sprint(showerror, e))")); continue
                end
            end
            written += 1
            verbose && i % 250 == 0 &&
                (@printf("  %d/%d  (written %d, skipped %d, collisions %d)\n",
                         i, length(ids), written, skipped, length(collisions)); flush(stdout))
        end
    end

    @printf("\n[migrate] done in %.1f min: %d written, %d skipped, %d failed, %d collisions\n",
            t/60, written, skipped, length(failed), length(collisions))
    written > 0 && @printf("          %.0f MB total (%.0f KB/file)\n", bytes/2^20, bytes/max(written,1)/1024)

    if !isempty(collisions)
        @printf("\n  %d PARAMETER COLLISIONS (same config, >1 store entry — typically the\n", length(collisions))
        @printf("  same run on two clusters with different seeds). Both kept, second as -dupN:\n")
        for (rp, _, src) in first(collisions, 15)
            @printf("    %-70s  %s\n", rp, src)
        end
        length(collisions) > 15 && @printf("    ... and %d more\n", length(collisions) - 15)
    end
    if !isempty(failed)
        @printf("\n  %d FAILED:\n", length(failed))
        for (id, why) in first(failed, 15); @printf("    id=%-6s %s\n", id, why); end
        length(failed) > 15 && @printf("    ... and %d more\n", length(failed) - 15)
    end

    if !dry_run && written > 0
        println("\n[migrate] rebuilding index ...")
        rebuild_index(; root=root)
    end
    return (written=written, skipped=skipped, failed=failed, collisions=collisions)
end

# Arg parsing lives in a function on purpose: assigning to a variable inside a
# top-level `for` loop hits Julia's soft-scope rule and silently creates a local,
# so a script-level loop would leave `--limit` as `nothing`.
function _parse_args(args)
    lim = nothing
    for (i, a) in enumerate(args)
        if a == "--limit" && i < length(args)
            lim = parse(Int, args[i+1])
        end
    end
    return (limit=lim, dry_run=("--dry-run" in args))
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = _parse_args(ARGS)
    migrate(; limit=o.limit, dry_run=o.dry_run)
end
