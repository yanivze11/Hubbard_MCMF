# ============================================================================
# validate_means_db.jl — prove the per-simulation files reproduce the old
# means_store entries EXACTLY, and measure what the new layout actually buys.
#
#   julia -O1 validate_means_db.jl [--sample 60]
#
# Check 1  round-trip fidelity: for a random sample of old store entries, the
#          new file must return an identical entry — same metadata keys, same
#          observables, same values (NaN-aware, recursive through dicts/arrays).
# Check 2  index integrity: every file on disk is in the index, and every index
#          row points at a file that exists.
# Check 3  query timing: old full-entry read vs new partial read.
# ============================================================================
using JLD2, Printf, Random, Dates

include(joinpath(@__DIR__, "..", "..", "means_db.jl"))

const STORE = joinpath(get(ENV, "V6_CLUSTER_RESULTS", ""), "means_store", "means_store.jld2")

# ---------------------------------------------------------------------------
# recursive, NaN-aware comparison
# ---------------------------------------------------------------------------
function deep_diff(a, b, path::String, diffs::Vector{String}; maxdiffs::Int=8)
    length(diffs) >= maxdiffs && return diffs
    if a isa AbstractDict && b isa AbstractDict
        ka = Set(string.(keys(a))); kb = Set(string.(keys(b)))
        for k in setdiff(ka, kb); push!(diffs, "$path.$k: missing in NEW"); end
        for k in setdiff(kb, ka); push!(diffs, "$path.$k: extra in NEW"); end
        for k in sort(collect(intersect(ka, kb)))
            deep_diff(a[k], b[k], "$path.$k", diffs; maxdiffs=maxdiffs)
        end
    elseif a isa AbstractArray && b isa AbstractArray
        if size(a) != size(b)
            push!(diffs, "$path: size $(size(a)) vs $(size(b))")
        else
            for i in eachindex(a)
                length(diffs) >= maxdiffs && break
                deep_diff(a[i], b[i], "$path[$i]", diffs; maxdiffs=maxdiffs)
            end
        end
    elseif a isa AbstractFloat && b isa AbstractFloat
        (isnan(a) && isnan(b)) || a == b || push!(diffs, "$path: $a vs $b")
    elseif a isa Tuple && b isa Tuple
        length(a) == length(b) || push!(diffs, "$path: tuple length")
        for i in 1:min(length(a), length(b))
            deep_diff(a[i], b[i], "$path[$i]", diffs; maxdiffs=maxdiffs)
        end
    else
        a == b || push!(diffs, "$path: $(repr(a)) vs $(repr(b))")
    end
    return diffs
end

"""Candidate new-db paths for an old entry, including -dupN collision variants."""
function candidate_paths(params; root::String=MEANS_DB_ROOT)
    rp = entry_relpath(params)
    stem, ext = splitext(rp)
    out = [joinpath(root, rp)]
    for k in 2:6
        p = joinpath(root, string(stem, "-dup", k, ext))
        isfile(p) && push!(out, p)
    end
    return filter(isfile, out)
end

function check_roundtrip(; sample::Int=60, root::String=MEANS_DB_ROOT, seed::Int=20260723)
    println("\n" * "="^78)
    println("CHECK 1 — round-trip fidelity (old store entry  ==  new means_db file)")
    println("="^78)
    Random.seed!(seed)
    ids = jldopen(STORE, "r") do f
        [k for k in keys(f) if !startswith(k, "__")]
    end
    pick = ids[randperm(length(ids))[1:min(sample, length(ids))]]

    ok = 0; missing_file = 0; mismatched = Tuple{String,Vector{String}}[]
    jldopen(STORE, "r") do f
        for id in pick
            old = f[id]
            (old isa AbstractDict && get(old, "means", nothing) isa AbstractDict) || continue
            cands = candidate_paths(canonical_params(old); root=root)
            if isempty(cands)
                missing_file += 1
                @printf("  MISSING  id=%-6s -> %s\n", id, entry_relpath(canonical_params(old)))
                continue
            end
            best = nothing
            for p in cands
                new = load_entry(p; obs=:all)
                d = deep_diff(old, new, "", String[])
                if isempty(d); best = String[]; break
                elseif best === nothing || length(d) < length(best); best = d; end
            end
            if isempty(best); ok += 1
            else; push!(mismatched, (id, best)); end
        end
    end
    @printf("\n  identical : %d/%d\n", ok, length(pick))
    @printf("  missing   : %d\n", missing_file)
    @printf("  mismatched: %d\n", length(mismatched))
    for (id, d) in first(mismatched, 5)
        @printf("    id=%s\n", id)
        for line in first(d, 6); println("      ", line); end
    end
    return (ok=ok, total=length(pick), missing=missing_file, mismatched=length(mismatched))
end

function check_index(; root::String=MEANS_DB_ROOT)
    println("\n" * "="^78)
    println("CHECK 2 — index integrity")
    println("="^78)
    idx = load_index(; root=root)
    on_disk = Set(relpath.(scan_files(root), root))
    in_index = Set(String.(idx["relpath"]))
    only_disk  = setdiff(on_disk, in_index)
    only_index = setdiff(in_index, on_disk)
    @printf("  files on disk : %d\n", length(on_disk))
    @printf("  rows in index : %d\n", length(in_index))
    @printf("  on disk but NOT indexed : %d%s\n", length(only_disk),
            isempty(only_disk) ? "" : "  <-- rebuild_index()")
    @printf("  indexed but MISSING     : %d\n", length(only_index))
    for p in first(collect(only_disk), 5);  println("    +disk  ", p); end
    for p in first(collect(only_index), 5); println("    -gone  ", p); end
    @printf("  queryable columns: %s\n", join(idx["columns"], ", "))
    return (only_disk=length(only_disk), only_index=length(only_index))
end

function check_timing(; root::String=MEANS_DB_ROOT)
    println("\n" * "="^78)
    println("CHECK 3 — what the layout buys (same selection, different reads)")
    println("="^78)
    idx = load_index(; root=root)
    # a representative T-sweep: pick the (U, Lx) pair with the most simulations
    best = nothing; bestn = 0
    if "U" in idx["columns"] && "Lx" in idx["columns"]
        for (u, l) in unique(collect(zip(idx["U"], idx["Lx"])))
            (u === missing || l === missing) && continue
            c = count(i -> idx["U"][i] === u && idx["Lx"][i] === l, 1:idx["n"])
            c > bestn && (bestn = c; best = (u, l))
        end
    end
    best === nothing && (println("  not enough index columns"); return nothing)
    U, Lx = best
    @printf("  selection: U=%s, Lx=%s  (%d simulations)\n", string(U), string(Lx), bestn)

    load_means(; root=root, U=U, Lx=Lx, obs=[:n], verbose=false)   # warm up
    t_one   = @elapsed s1 = load_means(; root=root, U=U, Lx=Lx, obs=[:n], verbose=false)
    t_three = @elapsed s3 = load_means(; root=root, U=U, Lx=Lx,
                                       obs=[:stiffness, :n, :Ps], verbose=false)
    t_light = @elapsed sl = load_means(; root=root, U=U, Lx=Lx, obs=:light, verbose=false)
    t_all   = @elapsed sa = load_means(; root=root, U=U, Lx=Lx, obs=:all, verbose=false)
    t_idx   = @elapsed find_entries(; root=root, U=U, Lx=Lx)

    @printf("\n  index-only query (no data) : %6.3f s\n", t_idx)
    @printf("  obs=[:n]                   : %6.3f s   (%d sims)\n", t_one, length(s1))
    @printf("  obs=[3 observables]        : %6.3f s   (%d sims)\n", t_three, length(s3))
    @printf("  obs=:light                 : %6.3f s   (%d sims)\n", t_light, length(sl))
    @printf("  obs=:all                   : %6.3f s   (%d sims)\n", t_all, length(sa))
    @printf("\n  speedup of a 3-observable query vs full load: %.1fx\n", t_all / max(t_three, 1e-9))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    # in a function, not a top-level loop — see the soft-scope note in migrate_means_db.jl
    _sample_arg(args) = begin
        v = 60
        for (i, x) in enumerate(args)
            x == "--sample" && i < length(args) && (v = parse(Int, args[i+1]))
        end
        v
    end
    r1 = check_roundtrip(; sample=_sample_arg(ARGS))
    r2 = check_index()
    check_timing()
    println("\n" * "="^78)
    verdict = (r1.mismatched == 0 && r1.missing == 0 && r2.only_disk == 0 && r2.only_index == 0)
    println(verdict ? "VERDICT: PASS — new files reproduce the old entries exactly." :
                      "VERDICT: PROBLEMS FOUND (see above).")
    println("="^78)
end
