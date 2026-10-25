# ============================================================================
# cluster_config.jl — every user- and cluster-specific setting in one place.
#
# A new user edits THIS file (or exports the environment variable named next to
# each setting) and nothing else. Every submit/driver script includes it:
#
#   pre_pre_run.jl, fh_pre_run_*.jl, fh_resubmit_campaign.jl,
#   pre_run_num_control_V.jl, run_num_control_V.jl, tests_pre_run.jl,
#   slurm_job_mode.jl
#
# The simulation engine (MC_Hubburd.jl, means_db.jl, sim_num_control.jl) does
# NOT include it, so loading the engine in a notebook is unaffected.
#
# Precedence for every setting: environment variable > the value below.
# pre_pre_run.jl exports the per-cluster values into the job it submits, so the
# scripts that run inside that job see the same numbers.
#
# Base Julia only (no packages), so it costs nothing to include.
#
# Include it ONCE per Julia session. A file that may be included after the
# config already is (slurm_job_mode.jl) guards the include:
#   @isdefined(CLUSTER_CONFIG_LOADED) || include(joinpath(@__DIR__, "cluster_config.jl"))
# (The guard cannot live in this file: docstrings inside an `if` block are not
# attached, so `?detect_cluster_name` would show nothing.)
# ============================================================================

"Set when this file has been included (see the include guard above)."
const CLUSTER_CONFIG_LOADED = true

# ---------------------------------------------------------------------------
# Paths on the clusters
# ---------------------------------------------------------------------------

"""
Your user name on the clusters (env `CLUSTER_USER`, else `USER`). The default
data and scratch folders below are built from it.
"""
const CLUSTER_USER = get(ENV, "CLUSTER_USER", get(ENV, "USER", get(ENV, "USERNAME", "")))

"""
Root of the V7 data tree on a cluster (env `V7_DATA_ROOT`; default
`/home/data/<CLUSTER_USER>/data/MC_Hubbard_V7`, the FHRC layout).

Holds `data/<campaign>/<unit>/` (simulations), `means_db/` (the means) and
`sim_logs/` (job logs). On the FHRC clusters the same path string exists on both
Landau and Curie, but each points at that cluster's own disk, so every command is
identical on both.
"""
const V7_DATA_ROOT = get(ENV, "V7_DATA_ROOT", "/home/data/$CLUSTER_USER/data/MC_Hubbard_V7")

"""
Root of the old V6 data tree on a cluster (env `V6_DATA_ROOT`; default next to
`V7_DATA_ROOT`). Only the scripts in `archive/` use it.
"""
const V6_DATA_ROOT = get(ENV, "V6_DATA_ROOT", dirname(V7_DATA_ROOT) * "/MC_Hubbard_V6")   # a cluster (POSIX) path

"""
Node-local scratch folder of the scratch-mode jobs (env `SCRATCH_ROOT`; default
`/home/scr/<CLUSTER_USER>`, the FHRC layout). Written into each job script.
"""
const SCRATCH_ROOT = get(ENV, "SCRATCH_ROOT", "/home/scr/$CLUSTER_USER")

"""
Julia executable used inside the SLURM jobs (env `JULIA_BIN`; default: the Julia
that runs this file, i.e. the one you submit with).

`/usr/people` is shared between the FHRC clusters, so one binary serves both.
"""
const JULIA_BIN = get(ENV, "JULIA_BIN", joinpath(Sys.BINDIR, Base.julia_exename()))

# ---------------------------------------------------------------------------
# SLURM accounts and partitions
# ---------------------------------------------------------------------------

"Account for the simulation jobs (env `JOB_ACCOUNT`)."
const JOB_ACCOUNT = get(ENV, "JOB_ACCOUNT", "all-account")

"""
Account and partition of the group's own queue (env `PRIORITY_ACCOUNT`,
`PRIORITY_PARTITION`). Used for the light, long-lived jobs: the pre_run job that
pre_pre_run.jl submits, and the num_control monitor.
"""
const PRIORITY_ACCOUNT   = get(ENV, "PRIORITY_ACCOUNT", "dror-account")
const PRIORITY_PARTITION = get(ENV, "PRIORITY_PARTITION", "drorq")

# ---------------------------------------------------------------------------
# Per-cluster settings
# ---------------------------------------------------------------------------

"""
    CLUSTER_SETTINGS :: Dict{String,NamedTuple}

One entry per cluster, keyed by the name `detect_cluster_name` returns. The keys
must be the same as `DB_CLUSTERS` in means_db.jl, which splits the work between
the clusters.

- `depot`         Julia depot of that cluster (env `JULIA_DEPOT_PATH`). Each
                  cluster needs its own: packages are compiled for its CPUs.
- `partitions`    SLURM partitions for the simulation jobs (env `PARTITIONS`).
                  Comma-separated, no spaces: a space makes sbatch silently
                  ignore every later `#SBATCH` line.
- `cpus_per_task` cores per simulation job (env `CPUS_PER_TASK`).
- `exclude_nodes` nodes never used (env `EXCLUDE_NODES`): landau59/curie25/
                  curie31 are faulty, landau61 is very slow.
"""
const CLUSTER_SETTINGS = Dict(
    "landau" => (depot         = "~/.julia_landau",
                 partitions    = "allq,infiniq",
                 cpus_per_task = 1,
                 exclude_nodes = "landau59,landau61"),
    "curie"  => (depot         = "~/.julia_curie",
                 partitions    = "allq",            # Curie has no infiniq
                 cpus_per_task = 2,
                 exclude_nodes = "curie25,curie31"),
)

"""
    detect_cluster_name(; hostname=gethostname()) -> String

Name of the cluster this runs on: env `MEANS_DB_CLUSTER` if set, else the first
key of `CLUSTER_SETTINGS` that occurs in the hostname, else `""`.
"""
function detect_cluster_name(; hostname::AbstractString=gethostname())
    c = lowercase(get(ENV, "MEANS_DB_CLUSTER", ""))
    isempty(c) || return c
    h = lowercase(hostname)
    for name in sort(collect(keys(CLUSTER_SETTINGS)))
        occursin(name, h) && return name
    end
    return ""
end

"""
    cluster_setting(cluster, key) -> value

One per-cluster setting, with its environment override applied. `key` is one of
`:depot`, `:partitions`, `:cpus_per_task`, `:exclude_nodes`. Throws for an
unknown cluster.
"""
function cluster_setting(cluster::AbstractString, key::Symbol)
    haskey(CLUSTER_SETTINGS, cluster) ||
        error("unknown cluster '$cluster'; known: $(join(sort(collect(keys(CLUSTER_SETTINGS))), ", ")). " *
              "Add it to CLUSTER_SETTINGS in cluster_config.jl or set MEANS_DB_CLUSTER.")
    default = getproperty(CLUSTER_SETTINGS[cluster], key)
    env = Dict(:depot => "JULIA_DEPOT_PATH", :partitions => "PARTITIONS",
               :cpus_per_task => "CPUS_PER_TASK", :exclude_nodes => "EXCLUDE_NODES")[key]
    haskey(ENV, env) || return default
    return default isa Integer ? parse(Int, ENV[env]) : ENV[env]
end

"""
    cluster_setting_or(cluster, key, fallback) -> value

`cluster_setting(cluster, key)`, or `fallback` when `cluster` is not in
`CLUSTER_SETTINGS` (e.g. `""` for a `PREVIEW=1` run on a laptop).
"""
cluster_setting_or(cluster::AbstractString, key::Symbol, fallback) =
    haskey(CLUSTER_SETTINGS, cluster) ? cluster_setting(cluster, key) : fallback
