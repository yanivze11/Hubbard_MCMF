# ============================================================================
# run_num_control_V.jl — compute means and write them into means_db.
#
# This is the WORKER, meant to run inside a slurm job (submit it with
# pre_run_num_control_V.jl). It:
#   1. scans the campaign directory for per-simulation sim.jld2 files
#   2. computes per-seed means (skipping any seed already current in means_db)
#   3. averages over MC seeds
#   4. writes one means_db file per simulation and updates the index,
#      flipping each simulation's status from "submitted" to "done"
#
# The legacy monolithic num_control/sim_info.jld2 is NOT written: means_db has
# taken over that role and nothing in this path reads it.
#
# MONITOR LOOP (LOOP_MODE=1): re-run every LOOP_INTERVAL_SEC for up to
# LOOP_MAX_HOURS, so means stay fresh WHILE simulations are still running. This
# is cheap in V7: `seed_is_current` skips any seed whose stored means already
# cover its current measurement count, so each pass only recomputes seeds that
# have produced NEW data since the last pass. It stops early once every seed in
# the campaign reports "finished".
#
# Required environment:
#   MEANS_DB_CLUSTER = landau | curie   (falls back to the hostname)
# Optional: V7_DATA_ROOT, MEANS_DB_ROOT, OUT_DIR, FOLDER_NAME,
#   LOOP_MODE (0/1), LOOP_INTERVAL_SEC (default 600), LOOP_MAX_HOURS (default 168).
#
# Data defaults to /home/data/<user>/data/MC_Hubbard_V7 (cluster_config.jl) — the same path string on
# both clusters, resolving to each one's own storage. /usr (and the code
# checkout) IS shared between the clusters; the Julia DEPOTS are not
# (.julia_landau / .julia_curie), so export the right JULIA_DEPOT_PATH.
# ============================================================================
using JLD2, Printf, Dates

# These ENV vars are captured into consts when means_db.jl is included, so they
# MUST be set first. V7 tree: one path string, cluster-local storage.
include(joinpath(@__DIR__, "cluster_config.jl"))   # V7_DATA_ROOT, detect_cluster_name
haskey(ENV, "MEANS_DB_ROOT") || (ENV["MEANS_DB_ROOT"] = joinpath(V7_DATA_ROOT, "means_db"))

# Which cluster produced these means. pre_pre_run.jl exports MEANS_DB_CLUSTER;
# fall back to the hostname so this also runs directly on a login node. Set
# before means_db.jl is included, so its MEANS_DB_CLUSTER const sees it.
isempty(get(ENV, "MEANS_DB_CLUSTER", "")) && (ENV["MEANS_DB_CLUSTER"] = detect_cluster_name())

include("sim_num_control.jl")

const THIS_CLUSTER = lowercase(get(ENV, "MEANS_DB_CLUSTER", ""))
if !(THIS_CLUSTER in DB_CLUSTERS)
    error("cannot tell which cluster this is (hostname=$(gethostname())). " *
          "Set MEANS_DB_CLUSTER to one of $(DB_CLUSTERS).")
end

const OUT_DIR     = get(ENV, "OUT_DIR", joinpath(V7_DATA_ROOT, "data"))
const FOLDER_NAME = get(ENV, "FOLDER_NAME", "Vdisorder_test__23-07-2026")
const CAMPAIGN    = joinpath(OUT_DIR, FOLDER_NAME)
isdir(CAMPAIGN) || error("campaign directory not found: $CAMPAIGN")

const LOOP_MODE     = get(ENV, "LOOP_MODE", "0") == "1"
const LOOP_INTERVAL = parse(Int, get(ENV, "LOOP_INTERVAL_SEC", "600"))
const LOOP_MAX_HRS  = parse(Float64, get(ENV, "LOOP_MAX_HOURS", "168"))

@printf("cluster       : %s\n", MEANS_DB_CLUSTER)
@printf("campaign      : %s\n", CAMPAIGN)
@printf("means_db root : %s\n", MEANS_DB_ROOT)
@printf("mode          : %s\n", LOOP_MODE ?
        @sprintf("MONITOR every %ds for up to %gh", LOOP_INTERVAL, LOOP_MAX_HRS) : "one-shot")
println()

"""
    do_pass()

One means-computation pass over the whole campaign (`update_means` into means_db,
seeds already current are skipped), then print the index status counts. Never throws:
a running simulation may be mid-write, and a bad pass must not kill a week-long monitor.
"""
function do_pass()
    try
        update_means(CAMPAIGN;
                     export_db=true,          # write means_db files
                     save_monolith=false,     # no sim_info.jld2
                     reuse_db=true,           # skip seeds already current (no reload)
                     write_means_cache=false) # means_cache.jld2 is superseded
    catch e
        @warn "pass failed; will retry next interval" exception=(e, catch_backtrace())
    end
    idx = load_index(; refresh=true)
    @printf("  index: %d simulation(s)", idx["n"])
    "status" in idx["columns"] &&
        print("   ", join(["$s=$(count(==(s), idx["status"]))"
                           for s in sort(unique(idx["status"]))], "  "))
    println()
end

"""
    campaign_all_finished(campaign) -> Bool

True only if every expected seed of every simulation reports `finished`: the signal to
do a last pass and stop the monitor. Unknown or running ⇒ false (keep looping); errors
count as "not done yet".
"""
function campaign_all_finished(campaign)
    any_sim = false
    for d in readdir(campaign)
        sj = joinpath(campaign, d, "sim.jld2")
        isfile(sj) || continue
        any_sim = true
        info = try JLD2.load(sj) catch; return false end
        tracking = get(info, "simulations", Dict())
        (tracking isa AbstractDict) || return false
        for s in get(info, "seeds", Int[])
            sd = get(tracking, s, nothing)
            (sd isa AbstractDict && string(get(sd, "status", "")) == "finished") || return false
        end
    end
    return any_sim
end

# Wrapped in a function on purpose: a `while` loop at top-level SCRIPT scope hits
# Julia's soft-scope rule, so `pass += 1` is treated as a new local and throws
# UndefVarError. Inside a function there is no soft scope and this just works.
"""
    monitor()

`LOOP_MODE=1`: run `do_pass` every `LOOP_INTERVAL_SEC` seconds for up to
`LOOP_MAX_HOURS`, stopping early once every seed of the campaign is finished.
"""
function monitor()
    deadline = now() + Millisecond(round(Int, LOOP_MAX_HRS * 3_600_000))
    pass = 0
    while now() < deadline
        pass += 1
        @printf("=== pass %d at %s ===\n", pass, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        do_pass()
        if campaign_all_finished(CAMPAIGN)
            println("all seeds finished — stopping monitor.")
            break
        end
        # A final sleep past the deadline would waste a whole interval, so guard it.
        now() + Second(LOOP_INTERVAL) < deadline || (println("time budget reached."); break)
        flush(stdout); sleep(LOOP_INTERVAL)
    end
    @printf("monitor exited after %d pass(es) at %s\n", pass, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
end

LOOP_MODE ? monitor() : do_pass()
