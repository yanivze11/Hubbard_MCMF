# ============================================================================
# verify_slims.jl — gate before deleting any results.jld2.
#
# For every results.jld2 in the manifest, check its sibling results_slim.jld2:
#   exists, opens, has slim_format + meta/num_measurements(>0) + a known
#   means_source + non-empty means/ and binned/ + finite means/n, and the
#   original has NOT been written to since the slim was made (a run that
#   continued after slimming must be re-slimmed, not deleted).
#
# V7 slims (means_source == "means_db") are also checked LIVE against the
# means_db record they were copied from, since means_db is what every query
# reads once the raw file is gone:
#   the record still exists, holds means/n, was computed from exactly N
#   measurements, is not marked running, and (with --deep) its n equals the
#   slim's copy.
#
# status=running: a job killed by its time limit, or stalled, never marks its
# seed finished. Such seeds are held back (status_running) unless you pass
# --accept-running, which takes them only if the raw file has been idle for
# over a day. Once slimmed, they can never be continued.
#
# --accept-stale: a slim that slim_results.jl --accept-stale made from a stale
# record (meta/db_stale = true) is checked against that record's own count,
# meta/db_num_measurements, instead of the stream length (decided 2026-10-21:
# every stream is final, the means need not match the bins exactly). Without the
# flag such a slim is held back as db_stale_slim.
#
# Writes:
#   slim_delete_manifest.txt   results.jld2 paths that PASSED (safe to delete)
#   slim_problems.txt          <path>\t<status>\t<why>  for everything else
# Only the whitelist should ever be fed to the delete step.
#
#   julia -O1 verify_slims.jl <manifest> [--deep] [--accept-running] [--accept-stale]
# ============================================================================
using JLD2, Statistics, Printf

const MIN_SLIM_BYTES = 20 * 1024      # a real slim is ~1-4 MB; anything tiny is suspect

# A job killed by its time limit (or a stall) never marks its seed finished, so
# its record keeps status=running forever. --accept-running takes those, but only
# when the raw file has not been touched for DEAD_AFTER_SEC (no live writer).
const DEAD_AFTER_SEC = 24 * 3600

"""
    check_db_record(f, N; deep=false, accept_running=false, accept_stale=false, results_path="")
        -> (status, message)

Check a V7 slim (open file `f`, stream length `N`) against the means_db record it was
copied from: the record exists, has means/n, was computed from the expected number of
measurements (`N`, or the record's own count for a stale slim), is not `running` (unless
`accept_running` and the raw file has been idle over a day), and with `deep` its n equals
the slim's. Returns `(:ok, "")` or the reason it fails.
"""
function check_db_record(f, N::Integer; deep::Bool=false, accept_running::Bool=false,
                         accept_stale::Bool=false, results_path::String="")
    (haskey(f, "meta/db_path") && haskey(f, "meta/db_seed")) || return (:bad, "means_db slim without meta/db_path")
    db, s = f["meta/db_path"], f["meta/db_seed"]
    isfile(db) || return (:db_gone, "means_db file missing: $db")
    stale = haskey(f, "meta/db_stale") && f["meta/db_stale"] === true
    if stale
        rec_n = haskey(f, "meta/db_num_measurements") ? f["meta/db_num_measurements"] : "?"
        accept_stale || return (:db_stale_slim, "means from a stale record (N=$rec_n, stream N=$N); re-verify with --accept-stale")
        haskey(f, "meta/db_num_measurements") || return (:bad, "stale slim without meta/db_num_measurements")
    end
    want = stale ? f["meta/db_num_measurements"] : N      # the count the means were computed from
    return jldopen(db, "r") do g
        haskey(g, "seeds/$s/info")    || return (:db_gone, "seed $s missing in $db")
        haskey(g, "seeds/$s/means/n") || return (:db_no_means, "seed $s has no means/n in $db")
        info = g["seeds/$s/info"]
        nm = get(info, "num_current_measurments", -1)
        nm == want || return (:db_count_mismatch, "means_db N=$nm, slim " * (stale ? "record " : "") * "N=$want ($db)")
        st = string(get(info, "status", ""))
        if st == "running"
            idle = isempty(results_path) ? 0.0 : time() - mtime(results_path)
            (accept_running && idle > DEAD_AFTER_SEC) ||
                return (:status_running, @sprintf("means_db status=running, raw file idle %.1f d (%s)%s",
                        idle/86400, db, accept_running ? "" : "; if the job is dead, re-verify with --accept-running"))
        end
        if deep
            v_db, v_slim = g["seeds/$s/means/n"]["value"], f["means/n"]["value"]
            isequal(v_db, v_slim) || return (:db_differs, "means/n db=$v_db slim=$v_slim")
        end
        return (:ok, "")
    end
end

"""
    check_slim(results_path; deep=false, accept_running=false, accept_stale=false) -> (status, message)

Check the results_slim.jld2 next to `results_path` (the checks of the file header) and,
for a V7 slim, its means_db record (`check_db_record`). `(:ok, summary)` means the raw
file may be deleted; anything else says why not.
"""
function check_slim(results_path::String; deep::Bool=false, accept_running::Bool=false,
                    accept_stale::Bool=false)
    slim = joinpath(dirname(results_path), "results_slim.jld2")
    isfile(slim) || return (:missing_slim, "no results_slim.jld2")
    sz = filesize(slim)
    sz >= MIN_SLIM_BYTES || return (:too_small, "slim only $sz bytes")
    mtime(results_path) > mtime(slim) &&
        return (:stale_slim, "results.jld2 written after the slim (run continued?) - re-slim with --force")
    try
        return jldopen(slim, "r") do f
            haskey(f, "slim_format")            || return (:bad, "no slim_format")
            haskey(f, "meta/num_measurements")  || return (:bad, "no meta/num_measurements")
            N = f["meta/num_measurements"]
            (N isa Integer && N > 0)            || return (:bad, "num_measurements=$N")
            ms = haskey(f, "meta/means_source") ? f["meta/means_source"] : "?"
            ms in ("means_cache", "means_db")   || return (:no_means, "means_source=$ms")
            (haskey(f, "means")  && !isempty(keys(f["means"])))  || return (:bad, "empty means/")
            (haskey(f, "binned") && !isempty(keys(f["binned"]))) || return (:bad, "empty binned/")
            haskey(f, "means/n")                || return (:bad, "no means/n")
            # value must be finite; a NaN error is legitimate (e.g. single-measurement
            # mean-field k-space runs, num_measurements=1, have no statistical error).
            isfinite(f["means/n"]["value"])     || return (:bad, "means/n value not finite")
            if deep && haskey(f, "binned/n")
                isfinite(mean(f["binned/n"]))   || return (:bad, "binned/n mean not finite")
            end
            if ms == "means_db"
                st, msg = check_db_record(f, N; deep=deep, accept_running=accept_running,
                                          accept_stale=accept_stale, results_path=results_path)
                st === :ok || return (st, msg)
            end
            return (:ok, "N=$N, $(length(keys(f["means"]))) means, $(length(keys(f["binned"]))) binned [$ms]")
        end
    catch e
        return (:corrupt, "open failed: $(sprint(showerror, e))")
    end
end

"""
    main(args)

Check every results.jld2 listed in the manifest (`check_slim`) and write
slim_delete_manifest.txt (passed: safe to delete) and slim_problems.txt (everything
else, with the reason) next to the manifest; print a summary.
"""
function main(args)
    pos = filter(a -> !startswith(a, "--"), args)
    isempty(pos) && error("usage: julia verify_slims.jl <manifest> [--deep] [--accept-running] [--accept-stale]")
    manifest = pos[1]
    deep = any(a -> a == "--deep", args)
    accept_running = any(a -> a == "--accept-running", args)
    accept_stale   = any(a -> a == "--accept-stale", args)
    outdir    = dirname(manifest)
    safe_path = joinpath(outdir, "slim_delete_manifest.txt")
    prob_path = joinpath(outdir, "slim_problems.txt")

    lines = filter(!isempty, strip.(readlines(manifest)))
    @printf("[verify] %d results.jld2 from %s  (deep=%s, accept_running=%s, accept_stale=%s)\n",
            length(lines), manifest, deep, accept_running, accept_stale)

    tally = Dict{Symbol,Int}(); reclaim = 0.0
    open(safe_path, "w") do safe
        open(prob_path, "w") do prob
            for (i, rp) in enumerate(lines)
                rp = String(rp)
                st, msg = isfile(rp) ? check_slim(rp; deep=deep, accept_running=accept_running,
                                                  accept_stale=accept_stale) :
                                       (:no_results, "results.jld2 gone")
                tally[st] = get(tally, st, 0) + 1
                if st === :ok
                    println(safe, rp); reclaim += filesize(rp) / 2^30
                else
                    println(prob, rp, "\t", st, "\t", msg)
                end
                i % 500 == 0 && (@printf("  %d/%d ...\n", i, length(lines)); flush(stdout))
            end
        end
    end

    println("\n=== verify summary ===")
    for (k, v) in sort(collect(tally), by = x -> -x[2]); @printf("  %-18s %d\n", k, v); end
    @printf("\nSAFE to delete (verified slim): %d files, %.1f GB\n  -> %s\n",
            get(tally, :ok, 0), reclaim, safe_path)
    probs = length(lines) - get(tally, :ok, 0)
    @printf("PROBLEMS (do NOT delete): %d\n  -> %s\n", probs, prob_path)
    probs == 0 || println("\n*** Investigate slim_problems.txt before deleting. ***")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
