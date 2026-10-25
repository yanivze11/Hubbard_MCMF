# ============================================================================
# means_db.jl — one light, self-sufficient file per simulation, plus an index.
#
# GOAL: never open a monolithic sim_info again, and never rebuild a derived
# store. Each file IS the source of truth for one simulation:
#   • new simulation = new file (no rebuild, never stale)
#   • cluster jobs write their own files (no shared blob, no clobbering)
#   • corruption costs one simulation, not a campaign
#   • one JLD2 key per observable => read `stiffness` without loading `LDOS`
#
# MEASURED: a 55-temperature sweep, 2 observables = 0.28 s vs 6.8 s for all
# observables (24x). Per-file open (~5 ms) dominates, so cost tracks FILES
# TOUCHED, not bytes.
#
# ---------------------------------------------------------------------------
# FILE LAYOUT
#   <root>/index.jld2                     query index (+ index.csv mirror)
#   <root>/U3/L16x16/U3_n0.875_L16x16_TCA12_b1_<hash>.jld2
#
# INSIDE ONE FILE
#   schema_version        Int
#   param_schema_version  Int      bump when IDENTITY_PARAMS changes meaning
#   params                Dict     FULL parameter dict as recorded
#   identity              Dict     the registered subset that was hashed
#   meta                  Dict     non-identity fields (convergence, sim_num, ...)
#   summary               Dict     tiny header for indexing
#   means/<obs>           Dict     seed-AVERAGED, one key per observable
#   seeds/<seed>/info     Dict     num_current_measurments, mu, convergence, ...
#   seeds/<seed>/means/<obs>       per-seed means
#
# Storing per-seed `info` (not just means) is what makes the file self
# sufficient: `average_over_seeds` needs num_current_measurments / mu /
# convergence per seed, so without them a merged file could not re-derive its
# own average. It also makes this file the recomputation cache that
# `means_cache.jld2` used to be.
#
# ---------------------------------------------------------------------------
# IDENTITY (why a registered list)
# Parameters get added to the code over time. If the hash covered every key,
# the same physical config computed before and after adding a parameter would
# hash differently and silently split into two entries. So:
#   • only params in IDENTITY_PARAMS are hashed
#   • a param equal to its registered default is OMITTED from the signature
#     => registering a new param with its default leaves every existing hash
#        unchanged; only a non-default value creates a new identity
#   • unregistered params are ignored entirely
# The failure mode is now loud instead of silent: if two genuinely different
# simulations map to one filename, `save_entry` sees mismatched `params` and
# refuses rather than overwriting.
#
# USAGE
#   include("means_db.jl")
#   si = load_means(; U=3.0, Lx=12, obs=[:stiffness, :n])
#   db_summary(); rebuild_index()
# ============================================================================
using JLD2, Printf, Dates, SHA

const MEANS_DB_SCHEMA_VERSION = 2
const PARAM_SCHEMA_VERSION    = 1
# The database this process WRITES to (and reads by default when registering):
# the environment variable MEANS_DB_ROOT, set before including this file (the
# cluster drivers set it to <V7_DATA_ROOT>/means_db); otherwise a `means_db`
# folder next to the code. Queries read MEANS_DB_ROOTS (below) instead.
const MEANS_DB_ROOT = get(ENV, "MEANS_DB_ROOT", joinpath(@__DIR__, "means_db"))

# Local cache for files that live on a cluster mount (see _open_jld2).
const MEANS_DB_CACHE = get(ENV, "MEANS_DB_CACHE",
    joinpath(get(ENV, "TEMP", tempdir()), "means_db_cache"))

# Which machine wrote a file. Cluster jobs should set MEANS_DB_CLUSTER=landau /
# curie so that, when the same config runs in both places, every seed can be
# traced back to where it was produced.
const MEANS_DB_CLUSTER = get(ENV, "MEANS_DB_CLUSTER", "local")

# ---------------------------------------------------------------------------
# The backup drive (read-only archive)
#
# After the data reduction the drive holds the only copy of the means_db and the
# slims: <drive>/<cluster>/MC_Hubbard_V7/{means_db,data}/... (docs/WORKFLOW.md §8).
# One setting points the code at it: env BACKUP_DRIVE, the drive's `Yaniv-Data`
# folder, set before including this file. It is the same
# variable MC_Hubburd.jl's fix_cluster_path uses. With it set:
#   * MEANS_DB_ROOTS defaults to the drive's two means_db folders (per cluster,
#     LANDAU_ROOT / CURIE_ROOT win over the drive; see cluster_root);
#   * nothing in this file writes below it (DriveReadOnly).
# ---------------------------------------------------------------------------

"Name of the archive folder on the backup drive. Paths with this folder in them are never written."
const DRIVE_ARCHIVE_DIRNAME = "Yaniv-Data"

"""
    _backup_drive() -> String

The backup drive's top folder (env `BACKUP_DRIVE`, the drive's `Yaniv-Data` folder), read each
time it is called. Empty = not set.
"""
_backup_drive() = String(strip(get(ENV, "BACKUP_DRIVE", "")))

"""
    _norm_path(p) -> String

`p` as an absolute, lower-case path with `/` separators and no trailing `/`, for
comparing paths on any OS.
"""
_norm_path(p::AbstractString) = lowercase(rstrip(replace(abspath(String(p)), '\\' => '/'), '/'))

"""
    _on_backup_drive(path) -> Bool

True if `path` is inside the backup drive folder (`BACKUP_DRIVE`) or has a folder named
`Yaniv-Data` in it (`DRIVE_ARCHIVE_DIRNAME`), so the drive stays protected even when
`BACKUP_DRIVE` was not set.
"""
function _on_backup_drive(path::AbstractString)
    a = _norm_path(path)
    occursin("/" * lowercase(DRIVE_ARCHIVE_DIRNAME) * "/", a * "/") && return true
    d = _backup_drive()
    isempty(d) && return false
    b = _norm_path(d)
    return a == b || startswith(a, b * "/")
end

"""
A write was attempted below the backup drive. The drive is the read-only archive
(PI's backup disk, shared with other people's data): nothing writes there from
Julia: no index rebuild, no `save_entry`, no `update_means`.
"""
struct DriveReadOnly <: Exception
    path::String
    what::String
end
"""
    showerror(io, e::DriveReadOnly)

Say what was refused and what to do instead.
"""
function Base.showerror(io::IO, e::DriveReadOnly)
    print(io, "DriveReadOnly: refused `", e.what, "` at ", e.path, "\n",
          "  This path is on the backup drive (BACKUP_DRIVE / a '", DRIVE_ARCHIVE_DIRNAME,
          "' folder), which is a read-only archive.\n",
          "  To work on a copy, copy the folder to a local disk and point the code there.\n",
          "  If a means_db index.jld2 is missing on the drive, do not rebuild it in place:\n",
          "  check the drive (and _records/<cluster>/<cluster>__means_db__c01/drive.sha256list).")
end

"""
    _assert_writable(path, what)

Throw `DriveReadOnly` if `path` is on the backup drive (see `_on_backup_drive`).
"""
_assert_writable(path::AbstractString, what::AbstractString) =
    _on_backup_drive(path) ? throw(DriveReadOnly(String(path), String(what))) : nothing

# Lifecycle of one simulation, recorded in `summary["status"]`.
#   submitted : the pre-run job registered it; nothing has run yet
#   running   : a job has started
#   done      : means are stored
const DB_STATUS_SUBMITTED = "submitted"
const DB_STATUS_RUNNING   = "running"
const DB_STATUS_DONE      = "done"

"""
    _db_index_jld2(root=MEANS_DB_ROOT)

Path of the index file `<root>/index.jld2`.
"""
_db_index_jld2(root=MEANS_DB_ROOT) = joinpath(root, "index.jld2")

"""
    _db_index_csv(root=MEANS_DB_ROOT)

Path of the CSV copy of the index, `<root>/index.csv`.
"""
_db_index_csv(root=MEANS_DB_ROOT)  = joinpath(root, "index.csv")

# ---------------------------------------------------------------------------
# IDENTITY REGISTRY
#
# `nothing` as the default means REQUIRED: the parameter must be present, and
# it always enters the signature.
#
# TO ADD A PARAMETER: append it here with the value that means "as before".
# Existing hashes stay valid because a default-valued param is omitted from
# the signature. Only bump PARAM_SCHEMA_VERSION if you change what an existing
# entry MEANS (e.g. redefine a default) — that does invalidate old filenames.
# ---------------------------------------------------------------------------
const IDENTITY_PARAMS = Dict{String,Any}(
    # --- required: no sensible "absent" value -------------------------------
    "U"                      => nothing,
    "wanted_n"               => nothing,
    "Lx"                     => nothing,
    "Ly"                     => nothing,
    "beta"                   => nothing,
    "diag_method"            => nothing,
    # --- physics: default == feature off ------------------------------------
    "t"                      => 1.0,
    "B"                      => 0.0,
    "V"                      => 0.0,          # potential disorder strength
    "disorder_seed"          => 0,            # WHICH disorder realization (see below)
    "impurity"               => 0.0,          # p-disorder
    "SC_L"                   => 1,
    "U_disorder"             => "uniform",
    "U_puddle_params"        => [0.0, 6.0, 0.7, 0.7],
    "smooth_U"               => (false, 0.8),
    "pairing_symmetry"       => "s-wave",
    "boundary_condition"     => "PBC",
    "gauge_choice"           => "Mixed",
    "guess"                  => "uniform",
    # --- density-channel (phi) decoupling; defaults == channel OFF (legacy) --
    "alpha"                  => 1.0,          # fraction of U in the pairing channel
    "beta_phi"               => 0.0,          # density penalty coeff; 0 => channel OFF
    "Qx"                     => 0.0,          # density ordering vector Q = (Qx, Qy)
    "Qy"                     => 0.0,
    # --- coupled / potential mu-U disorder; defaults == OFF (legacy) ---------
    "mu_disorder"            => "none",       # "none"|"interaction-only"|"coupled"|"potential-only"
    "mu_disorder_f"          => 0.0,          # fraction of sites with U_i=0 (see below)
    # --- method / numerics ---------------------------------------------------
    "TClustersize"           => 0,
    "max_iter"               => 100,
    "n_tol"                  => 0.001,
    "num_thermal"            => 5000,
    "num_measure"            => 20000,
    "num_thermal_occ"        => 200,
    "num_measure_occ"        => 100,
    "measuring_interval_occ" => 5,
    # --- measurement grids ---------------------------------------------------
    "delta_fixed_values"     => Float64[],
    "delta_values_stiffness" => Float64[],
    "dw_values"              => Float64[],
    "w_list"                 => Float64[],
    "w_conductivity_range"   => Float64[],
)

const REQUIRED_IDENTITY_PARAMS =
    sort([k for (k, v) in IDENTITY_PARAMS if v === nothing])

# Parameters that are only MEANINGFUL under some conditions. When the condition
# is false the parameter is forced to its default before hashing, so a value
# that cannot affect the physics cannot split one config into two entries.
#
# Concretely: `diag_method="K0"` does no cluster approximation, yet runs still
# record a `TClustersize` — without this rule, two identical K0 runs carrying
# different (ignored) TClustersize values would land in different files.
const CONDITIONAL_IDENTITY = Dict{String,Function}(
    "TClustersize"    => e -> occursin("TCA", string(get(e, "diag_method", ""))),
    "U_puddle_params" => e -> string(get(e, "U_disorder", "uniform")) != "uniform",
    # A disorder realization index only means something when there IS disorder.
    # The mu-U disorder draws its site partition from disorder_seed too, so a
    # non-"none" mu_disorder also makes the seed identity-relevant.
    "disorder_seed"   => e -> (get(e, "V", 0.0) != 0) ||
                              (string(get(e, "U_disorder", "uniform")) != "uniform") ||
                              (get(e, "impurity", 0.0) != 0) ||
                              (string(get(e, "mu_disorder", "none")) != "none"),
    # The empty-site fraction only matters when the mu-U disorder is on.
    "mu_disorder_f"   => e -> string(get(e, "mu_disorder", "none")) != "none",
    # The pairing-split alpha and the density ordering vector Q only affect the
    # physics when the density channel is actually on (beta_phi != 0). This keeps
    # every legacy/baseline run (beta_phi == 0) hashing exactly as before.
    "alpha"           => e -> get(e, "beta_phi", 0.0) != 0,
    "Qx"              => e -> get(e, "beta_phi", 0.0) != 0,
    "Qy"              => e -> get(e, "beta_phi", 0.0) != 0,
)

"""
    identity_relevant(entry, param) -> Bool

Whether `param` can affect the physics of `entry`. Unconditional parameters are
always relevant.
"""
identity_relevant(entry::AbstractDict, param::AbstractString) =
    haskey(CONDITIONAL_IDENTITY, param) ? CONDITIONAL_IDENTITY[param](entry) : true

# Fields that are results/bookkeeping, never identity. Kept in `meta`.
const NON_IDENTITY_FIELDS = Set([
    "simulations", "dir_path", "sim_num", "info_path", "start_time", "seeds",
    "means", "measuring_interval", "convergence", "num_uncorrelated_measurments",
    "all_seeds_completed", "last_completed_seed", "last_completed_time",
    "endtime", "completed_seeds", "updated_at", "cores",
    # MC-field histogram / snapshot settings: what gets recorded, not the physics.
    # Kept out of params so runs with and without them don't show up as
    # different "physics variants" in the same query.
    "phi_hist_range", "phi_hist_nbins", "Delta_hist_abs_range",
    "Delta_hist_abs_nbins", "Delta_hist_arg_nbins", "snapshot_interval",
])

# Big, rarely-plotted observables. `obs=:light` skips these.
const HEAVY_OBS = Set([
    "LDOS", "DOS",
    "conductivity_regular_xx_fixed_extrap", "conductivity_regular_xy_fixed_extrap",
    "conductivity_regular_yx_fixed_extrap", "conductivity_regular_yy_fixed_extrap",
    "conductivity_regular_xx_fixed", "conductivity_regular_xy_fixed",
    "conductivity_regular_yx_fixed", "conductivity_regular_yy_fixed",
    "Lxx_w_fixed", "Lxy_w_fixed", "Lyx_w_fixed", "Lyy_w_fixed",
])

# Set by sim_num_control.jl after include, so a merge can re-average pooled
# seeds without means_db depending on (or duplicating) that logic.
const MEANS_DB_AVERAGE_FN = Ref{Any}(nothing)

# Set by sim_num_control.jl after include: (dir_path, obs, vec_index) -> loads
# that seed's raw ~370 MB results.jld2 and returns (stream, n_meas, dir). Kept as
# an injection hook (not a direct call) so THIS file stays JLD2-only — no
# MC_Hubburd / MKL / Plots — and keeps loading fast for pure means queries.
# `get_stream` errors with a wire-me-up message if it is still unset.
const MEANS_DB_STREAM_FN = Ref{Any}(nothing)

# Set by sim_num_control.jl after include: (dir_path, key) -> reads ONE field off
# a seed's stored sim_data (a Dict in results.jld2) WITHOUT pulling the
# measurement chunks. Lets `get_mu` recover outputs (mu, ...) that older runs
# never exported into the means files. Same injection reasoning as above.
const MEANS_DB_SIMDATA_FN = Ref{Any}(nothing)

# ---------------------------------------------------------------------------
# Robust open — the cluster mounts (LANDAU_MOUNT/CURIE_MOUNT, RaiDrive over a Dokan network
# redirector) do not support DOS-path resolution, so libuv's `realpath` fails
# with ENOENT even though the file reads fine. `jldopen` calls `realpath`
# unconditionally, so fall back to a local copy. Mirrors the mmap->IOStream
# fallback in load_results_paths.
# ---------------------------------------------------------------------------
"""
    _is_realpath_failure(e) -> Bool

True if `e` is the `realpath` IOError that a cluster mount raises for a file that reads fine.
"""
_is_realpath_failure(e) = e isa Base.IOError && occursin("realpath", sprint(showerror, e))

"""
    _is_short_read(e) -> Bool

True if `e` is a short read from the mount (an EOFError, possibly wrapped in an IOError).
"""
_is_short_read(e) = e isa EOFError ||
    (e isa Base.IOError && occursin("end of file", lowercase(sprint(showerror, e))))

"""
    _is_not_found(e) -> Bool

True if `e` is ENOENT. A file the cluster wrote AFTER RaiDrive cached its folder's
listing is reported missing until that listing is refreshed (see `_warm`).
"""
_is_not_found(e) = e isa SystemError && e.errnum == Base.Libc.ENOENT
# dir => time() of the last re-list forced by a not-found. A folder's listing is
# stale or fresh for ALL its files, so re-list it at most once per window, not
# once per missing file — that is what kept a curve query fast.
const _RELISTED_DIRS = Dict{String,Float64}()
const _RELIST_WINDOW = 60.0   # seconds

# ---------------------------------------------------------------------------
# RaiDrive caches directory attributes lazily; a COLD directory reports stale or
# short file sizes, which surfaces as a spurious EOF on the first read. Browsing
# to the folder in Explorer clears it — because enumerating the directory forces
# RaiDrive to refresh its attribute cache from the SFTP server. `_warm` does the
# same thing programmatically (readdir + stat). Retrying the read alone does not
# help: it re-reads the SAME stale cache. Warmed dirs are memoized so this is
# cheap; `refresh=true` drops the memo so a retry genuinely re-fetches metadata.
# ---------------------------------------------------------------------------
const _WARMED_DIRS = Set{String}()

"""
    _warm(path; refresh=false)

Make RaiDrive refresh its cached attributes of `path`'s folder (readdir + stat), as
browsing the folder in Explorer does. Done once per folder unless `refresh=true`.
"""
function _warm(path::AbstractString; refresh::Bool=false)
    d = dirname(String(path))
    refresh && delete!(_WARMED_DIRS, d)
    d in _WARMED_DIRS && return
    try; readdir(d); catch; end
    try; stat(String(path)); catch; end
    push!(_WARMED_DIRS, d)
    return
end

"""
    _stable_filesize(path; tries=5) -> Int

The file size on the mount, read until two consecutive reads agree (warming the folder
in between), so a stale attribute cache cannot give a short size to verify against.
"""
function _stable_filesize(path::AbstractString; tries::Int=5)
    prev = filesize(path)
    for _ in 1:tries
        _warm(path; refresh=true)
        cur = filesize(path)
        (cur > 0 && cur == prev) && return cur
        prev = cur
    end
    return prev
end

"""
    _local_copy(path; retries=5) -> String

Copy a mount file into the local cache and return the local path. `Base.cp`
reads through the normal file API and we VERIFY the copy matches the source
size, retrying if a flaky read produced a short file — so the returned path is
always a complete local copy. Cached and reused (the files are immutable once
written).
"""
function _local_copy(path::AbstractString; retries::Int=5)
    mkpath(MEANS_DB_CACHE)
    tag = bytes2hex(sha1(String(path)))[1:12]
    dest = joinpath(MEANS_DB_CACHE, tag * "_" * basename(path))
    want = _stable_filesize(path)
    (want > 0 && isfile(dest) && filesize(dest) == want) && return dest
    for _ in 1:retries
        try
            cp(path, dest; force=true)
            (want > 0 && filesize(dest) == want) && return dest
        catch; end
        sleep(0.2)
    end
    error("could not obtain a complete local copy of $path ($want bytes)")
end

"""
    _open_jld2(f, path)

Read-only `jldopen`, hardened for the cluster mounts (`LANDAU_MOUNT`/`CURIE_MOUNT`, RaiDrive SFTP):

  * **`iotype=IOStream`, never the default mmap** — over a network mount an
    mmap page fault fetches lazily and can return short.
  * **Warm the directory first (`_warm`).** A cold RaiDrive directory reports
    stale/short sizes → spurious EOF; `readdir`+`stat` refreshes the attribute
    cache, exactly as browsing to the folder in Explorer does. Retries also
    re-warm (`refresh=true`), since re-reading the same stale cache is a no-op.
  * **Retry on a short read, with backoff.** RaiDrive intermittently signals a
    premature EOF mid-read (JLD2 0.5.x does not recover; 0.6.x does). The blip is
    transient, so a short `sleep` + a couple of retries usually clears it.
  * **Re-list once on "No such file".** A file the cluster wrote after RaiDrive
    cached its folder's listing (a fresh means export, already in the index) is
    invisible until the listing is refreshed — and the `_warm` memo means the
    up-front warm may not have re-listed it. Re-warm (`refresh=true`) and retry
    immediately, at most once per folder per `_RELIST_WINDOW`; other misses in
    that folder fail fast with an error naming the stale listing.
  * **Copy-local fallback.** If it still flakes, pull a verified-complete copy
    to local disk (`_local_copy`) and read that — local reads never flake. Also
    the path for a realpath failure on the mount.
"""
function _open_jld2(f::Function, path::AbstractString; retries::Int=3)
    p = String(path)
    _warm(p)                              # refresh RaiDrive's attr cache first (what Explorer does)
    for attempt in 1:retries
        try
            return jldopen(f, p, "r"; iotype=IOStream)
        catch e
            if _is_realpath_failure(e)
                try
                    return jldopen(f, _local_copy(path), "r"; iotype=IOStream)
                catch e2
                    error("_open_jld2 failed to read $p : $(sprint(showerror, e2))")
                end
            end
            if _is_not_found(e)
                # Nothing visible to copy locally, so no fallback — just re-list.
                d = dirname(p)
                (attempt == retries || time() - get(_RELISTED_DIRS, d, -Inf) < _RELIST_WINDOW) &&
                    error("_open_jld2: $p not found (folder re-listed). If the cluster wrote " *
                          "it recently, RaiDrive's listing of the folder is stale — open the " *
                          "folder in Explorer, then retry. ($(sprint(showerror, e)))")
                _RELISTED_DIRS[d] = time()
                _warm(p; refresh=true)
                continue
            end
            _is_short_read(e) || error("_open_jld2 failed to read $p : $(sprint(showerror, e))")
            println("$p")
            attempt == retries && break   # fall through to the local-copy fallback
            sleep(0.2 * attempt)          # back off, then force a refresh so the retry isn't a no-op
            _warm(p; refresh=true)
        end
    end
    # Last resort: read a verified-complete local copy. If even that fails, the
    # source file itself is corrupt/truncated (a real EOF, not a mount flake) —
    # surface the path so the caller knows exactly which file to inspect.
    try
        return jldopen(f, _local_copy(path), "r"; iotype=IOStream)
    catch e
        error("_open_jld2 failed to read $p : $(sprint(showerror, e))")
    end
end

# ---------------------------------------------------------------------------
# Identity -> signature -> hash -> path
# ---------------------------------------------------------------------------

"""
    identity_of(entry) -> Dict{String,Any}

The registered identity parameters, resolved: present value, else the
registered default. Errors if a REQUIRED parameter is missing.
"""
function identity_of(entry::AbstractDict)
    out = Dict{String,Any}()
    missing_required = String[]
    for (k, default) in IDENTITY_PARAMS
        # A parameter that cannot affect this entry's physics is pinned to its
        # default, so it drops out of the signature entirely.
        if default !== nothing && !identity_relevant(entry, k)
            out[k] = default
        elseif haskey(entry, k)
            out[k] = entry[k]
        elseif default === nothing
            push!(missing_required, k)
        else
            out[k] = default
        end
    end
    isempty(missing_required) ||
        error("entry is missing required identity parameter(s): " *
              join(sort(missing_required), ", "))
    return out
end

"""
    _canon_str(v) -> String

Canonical text of a parameter value for the identity signature: floats with 10
significant digits, NaN/Inf spelled out, collections and Dicts (sorted keys) recursively.
"""
function _canon_str(v)
    if v isa AbstractFloat
        isnan(v) && return "NaN"
        isinf(v) && return v > 0 ? "Inf" : "-Inf"
        return @sprintf("%.10g", v)
    elseif v isa Bool;            return v ? "true" : "false"
    elseif v isa Integer;         return string(v)
    elseif v isa AbstractString || v isa Symbol; return string(v)
    elseif v isa Tuple;           return "(" * join((_canon_str(x) for x in v), ",") * ")"
    elseif v isa AbstractArray;   return "[" * join((_canon_str(x) for x in v), ",") * "]"
    elseif v isa AbstractDict
        ks = sort(string.(collect(keys(v))))
        return "{" * join(("$k:" * _canon_str(v[k]) for k in ks), ",") * "}"
    else;                         return string(v)
    end
end

"""
    _same_as_default(v, d) -> Bool

Equality via `_canon_str`, so Int/Float and array/tuple element types do not matter.
"""
_same_as_default(v, d) = _canon_str(v) == _canon_str(d)

"""
    params_signature(entry) -> String

Canonical `key=value;...` over the registered identity parameters, with
default-valued parameters OMITTED. Print this when two simulations you expect
to be identical land in different files.
"""
function params_signature(entry::AbstractDict)
    ident = haskey(entry, "__resolved_identity__") ? entry["__resolved_identity__"] : identity_of(entry)
    parts = String[]
    for k in sort(collect(keys(ident)))
        d = IDENTITY_PARAMS[k]
        v = ident[k]
        (d !== nothing && _same_as_default(v, d)) && continue   # omit defaults
        push!(parts, "$k=" * _canon_str(v))
    end
    return join(parts, ";")
end

"""
    params_hash(entry) -> String

8 hex chars of SHA-1 over `params_signature`. SHA rather than `Base.hash` so a
file's name is stable across Julia versions and machines, forever.
"""
params_hash(entry::AbstractDict) = bytes2hex(sha1(params_signature(entry)))[1:8]

"""
    _fmtnum(x) -> String

A number as it appears in a file name (`%g`; "NA" for nothing or NaN).
"""
_fmtnum(x) = x === nothing ? "NA" :
             x isa Integer ? string(x) :
             x isa AbstractFloat ? (isnan(x) ? "NA" : @sprintf("%g", x)) : string(x)

"""
    _sanitize(s) -> String

`s` with every character other than letters, digits and `._+-` replaced by `_`.
"""
_sanitize(s) = replace(string(s), r"[^A-Za-z0-9._+-]" => "_")

"""
    entry_relpath(entry) -> String

`U3/L16x16/U3_n0.875_L16x16_TCA12_b1_<hash>.jld2` — readable enough to eyeball,
hashed so an exact lookup needs no index at all.
"""
function entry_relpath(entry::AbstractDict)
    g(k) = get(entry, k, get(IDENTITY_PARAMS, k, nothing))
    Lx = g("Lx"); Ly = g("Ly")
    stem = string("U", _fmtnum(g("U")), "_n", _fmtnum(g("wanted_n")),
                  "_L", _fmtnum(Lx), "x", _fmtnum(Ly),
                  "_TCA", _fmtnum(g("TClustersize")), "_b", _fmtnum(g("beta")),
                  "_", params_hash(entry))
    return joinpath(string("U", _fmtnum(g("U"))),
                    string("L", _fmtnum(Lx), "x", _fmtnum(Ly)),
                    _sanitize(stem) * ".jld2")
end

"""
    entry_path(entry; root=MEANS_DB_ROOT) -> String

Full path of the means_db file of a simulation identity: `root` + `entry_relpath(entry)`.
"""
entry_path(entry::AbstractDict; root::String=MEANS_DB_ROOT) =
    joinpath(root, entry_relpath(entry))

"""
    is_registered(entry, mc_seed; root=MEANS_DB_ROOT) -> Bool

Whether this EXACT simulation identity is already registered (submitted,
running or done) with MC seed `mc_seed`. Submit scripts use it for dedup.

It checks the identity file directly. A field-by-field `find_entries` filter is
NOT a substitute: any identity parameter the filter omits (wanted_n,
diag_method, TClustersize, num_measure, ...) lets a different simulation count
as "already known", and the unit is silently never run.
"""
function is_registered(entry::AbstractDict, mc_seed; root::String=MEANS_DB_ROOT)
    p = entry_path(entry; root=root)
    isfile(p) || return false
    s = try
        _open_jld2(f -> haskey(f, "summary") ? f["summary"] : Dict{String,Any}(), p)
    catch; Dict{String,Any}(); end
    seen = Set(string.(vcat(get(s, "registered_seeds", String[]),
                            get(s, "seed_list", String[]))))
    return string(mc_seed) in seen
end

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

"""
    _means_n(m) -> Float64

The filling ⟨n⟩ stored in a means Dict, or NaN.
"""
_means_n(m) = begin
    v = get(get(m, "n", Dict()), "value", NaN)
    v isa Number ? Float64(v) : NaN
end

# Per-seed fields `average_over_seeds`/`get_obs_dict` need, plus provenance.
const SEED_INFO_FIELDS = ["num_current_measurments", "mu", "convergence", "status",
                          "dir_path", "step_count", "end_time", "start_time"]

"""
    read_seeds(path) -> Dict{String,Any}

`seed => Dict("info"=>..., "means"=>...)` already stored in a file (empty if
none). Used both for merging and as the recomputation cache.
"""
function read_seeds(path::AbstractString; with_means::Bool=true, obs=:all)
    isfile(path) || return Dict{String,Any}()
    _open_jld2(path) do f
        out = Dict{String,Any}()
        haskey(f, "seeds") || return out
        for s in keys(f["seeds"])
            d = Dict{String,Any}()
            haskey(f, "seeds/$s/info") && (d["info"] = f["seeds/$s/info"])
            if with_means && haskey(f, "seeds/$s/means")
                d["means"] = _read_means_group!(Dict{String,Any}(), f, "seeds/$s/means", obs)
            end
            out[string(s)] = d
        end
        return out
    end
end

"""
    read_seed_means(path, seed; obs=:all) -> Union{Dict,Nothing}

Stored means for ONE seed, without touching the other seeds. `nothing` if absent.
"""
function read_seed_means(path::AbstractString, seed; obs=:all)
    isfile(path) || return nothing
    try
        _open_jld2(path) do f
            k = "seeds/$(seed)/means"
            haskey(f, k) || return nothing
            return _read_means_group!(Dict{String,Any}(), f, k, obs)
        end
    catch
        return nothing
    end
end

"""
    seed_is_current(path, seed, num_measurements) -> Bool

True if this file already holds means for `seed` computed from at least
`num_measurements` measurements. This is what replaces `means_cache.jld2`:
`update_means` can skip re-deriving a seed from its 369 MB `results.jld2`.
"""
function seed_is_current(path::AbstractString, seed, num_measurements::Integer)
    isfile(path) || return false
    try
        _open_jld2(path) do f
            k = "seeds/$(seed)/info"
            haskey(f, k) || return false
            Int(get(f[k], "num_current_measurments", 0)) >= Int(num_measurements)
        end
    catch
        return false
    end
end

"""
    save_entry(entry; root, seed_data=nothing, source="", merge=true) -> String

Write one simulation, atomically (temp file + rename, so an interrupted write
cannot leave a half-file behind).

`entry` is an old-style sim dict: metadata at top level plus `"means"`.
`seed_data` is `seed => Dict("info"=>..., "means"=>...)`.

MERGE (`merge=true`, the default): if a file already exists for this identity —
which is what happens when the same config runs on two clusters — the seed sets
are POOLED rather than one overwriting the other. If
`MEANS_DB_AVERAGE_FN[]` is set, the seed-average is recomputed over the pooled
seeds; otherwise the incoming `means` is kept and a warning is emitted, because
it then describes only a subset of the stored seeds.

If the existing file has the same hash but genuinely different parameters, this
errors instead of overwriting — that means an identity-relevant parameter is
missing from IDENTITY_PARAMS.
"""
function save_entry(entry::AbstractDict; root::String=MEANS_DB_ROOT,
                    seed_data=nothing, source::AbstractString="",
                    cluster::String=MEANS_DB_CLUSTER,
                    merge::Bool=true, path::Union{Nothing,AbstractString}=nothing)
    haskey(entry, "means") || error("entry has no \"means\"")
    means = entry["means"]
    means isa AbstractDict ||
        error("entry[\"means\"] is $(typeof(means)), not a Dict (failed/incomplete simulation?)")

    ident  = identity_of(entry)
    params = Dict{String,Any}(string(k) => v for (k, v) in entry
                              if !(string(k) in NON_IDENTITY_FIELDS))
    meta   = Dict{String,Any}(string(k) => v for (k, v) in entry
                              if string(k) in NON_IDENTITY_FIELDS && string(k) != "means")

    path = path === nothing ? joinpath(root, entry_relpath(entry)) : String(path)
    _assert_writable(path, "save_entry")
    seeds = seed_data === nothing ? Dict{String,Any}() :
            Dict{String,Any}(string(k) => v for (k, v) in seed_data)

    if merge && isfile(path)
        old_params = try
            _open_jld2(g -> haskey(g, "params") ? g["params"] : Dict{String,Any}(), path)
        catch; Dict{String,Any}(); end

        # Same hash, different physics => an identity param is unregistered.
        conflict = String[]
        for k in intersect(keys(old_params), keys(params))
            _canon_str(old_params[k]) == _canon_str(params[k]) || push!(conflict, k)
        end
        isempty(conflict) || error(
            "hash collision at $path: same identity hash but parameters differ in " *
            join(sort(conflict), ", ") * ". Add the distinguishing parameter to " *
            "IDENTITY_PARAMS (and bump PARAM_SCHEMA_VERSION).")

        old_seeds = read_seeds(path)
        merged = 0
        for (s, d) in old_seeds
            haskey(seeds, s) || (seeds[s] = d; merged += 1)
        end
        if merged > 0
            avg = MEANS_DB_AVERAGE_FN[]
            if avg === nothing
                @warn "pooled $merged pre-existing seed(s) into $(basename(path)) but " *
                      "MEANS_DB_AVERAGE_FN is unset; stored `means` covers only the " *
                      "incoming seeds" path=path
            else
                try
                    means = avg(seeds)
                catch e
                    @warn "re-averaging pooled seeds failed; keeping incoming means" exception=e
                end
            end
        end
    end

    mkpath(dirname(path))
    obs_names = sort(string.(collect(keys(means))))
    summary = Dict{String,Any}(
        "n_measured"     => _means_n(means),
        "n_obs"          => length(obs_names),
        "obs"            => obs_names,
        "n_seeds"        => length(seeds),
        "seed_list"      => sort(collect(keys(seeds))),
        "has_seed_means" => any(d -> haskey(d, "means"), values(seeds)),
        "created_at"     => Dates.now(),
        "source"         => string(source),
        "status"         => DB_STATUS_DONE,
        "cluster"        => cluster,
    )

    tmp = path * ".tmp"
    try
        jldopen(tmp, "w") do f
            f["schema_version"]       = MEANS_DB_SCHEMA_VERSION
            f["param_schema_version"] = PARAM_SCHEMA_VERSION
            f["params"]   = params
            f["identity"] = ident
            f["meta"]     = meta
            f["summary"]  = summary
            for (k, v) in means; f["means/"*string(k)] = v; end
            for (s, d) in seeds
                haskey(d, "info") && (f["seeds/$s/info"] = d["info"])
                if haskey(d, "means") && d["means"] isa AbstractDict
                    for (k, v) in d["means"]; f["seeds/$s/means/$(string(k))"] = v; end
                end
            end
        end
        mv(tmp, path; force=true)
    catch e
        isfile(tmp) && rm(tmp; force=true)
        rethrow(e)
    end
    return path
end

"The clusters that run simulations and keep a means_db each."
const DB_CLUSTERS = ["landau", "curie"]

"""
    assign_cluster(entry_or_hash, seed; clusters=DB_CLUSTERS) -> String

Which cluster owns the work unit `(simulation identity, MC seed)`.

Deliberately a PURE FUNCTION of the unit itself: a submitter on Landau can
decide what is its own without knowing anything about Curie's state. Two
clusters therefore cannot submit the same unit, with no index syncing, no
locking and no staleness window — and re-running submission anywhere is
idempotent. The same configuration still runs in both places, with its seeds
split between them, so statistics accumulate in parallel instead of duplicating.
"""
function assign_cluster(entry_or_hash, seed::Integer; clusters::Vector{String}=DB_CLUSTERS)
    h = entry_or_hash isa AbstractDict ? params_hash(entry_or_hash) : String(entry_or_hash)
    # SHA over hash+seed rather than arithmetic on the hex digits: keeps the
    # split even and insensitive to structure in the parameter values.
    d = sha1(string(h, ":", seed))
    return clusters[1 + mod(Int(d[1]) + 256*Int(d[2]), length(clusters))]
end

"""
    register_simulation(entry; root, cluster=MEANS_DB_CLUSTER, status=DB_STATUS_SUBMITTED,
                        seeds=Int[], source="", update_index=true) -> String

Record a simulation in the database AT SUBMISSION TIME, before it has run and
before any means exist. Writes a stub file with parameters only.

This is what makes the index a true catalogue of every simulation rather than
only of finished ones: the submitting job calls this, so a later submitter (or
`find_entries(; status="submitted")`) can see that the work is already claimed.
`save_entry` later fills the same file in and flips the status to `done`.

Returns the path. Existing files are left alone apart from a status/seed update,
so calling this twice is harmless.
"""
function register_simulation(entry::AbstractDict; root::String=MEANS_DB_ROOT,
                             cluster::String=MEANS_DB_CLUSTER,
                             status::String=DB_STATUS_SUBMITTED,
                             seeds=Int[], source::AbstractString="",
                             update_index::Bool=true)
    ident  = identity_of(entry)
    params = Dict{String,Any}(string(k) => v for (k, v) in entry
                              if !(string(k) in NON_IDENTITY_FIELDS))
    meta   = Dict{String,Any}(string(k) => v for (k, v) in entry
                              if string(k) in NON_IDENTITY_FIELDS && string(k) != "means")
    path = joinpath(root, entry_relpath(entry))
    _assert_writable(path, "register_simulation")

    if isfile(path)
        # Already known. Only widen the recorded seed list; never clobber means.
        old = _open_jld2(f -> haskey(f, "summary") ? f["summary"] : Dict{String,Any}(), path)
        known = Set(string.(get(old, "registered_seeds", [])))
        union!(known, string.(seeds))
        get(old, "status", "") == DB_STATUS_DONE && return path
        _update_summary!(path, Dict{String,Any}("registered_seeds" => sort(collect(known)),
                                                "status" => status))
        update_index && append_index!(root, [path])
        return path
    end

    mkpath(dirname(path))
    summary = Dict{String,Any}(
        "status"           => status,
        "cluster"          => cluster,
        "registered_seeds" => sort(string.(collect(seeds))),
        "n_seeds"          => 0,
        "n_obs"            => 0,
        "n_measured"       => NaN,
        "obs"              => String[],
        "created_at"       => Dates.now(),
        "source"           => string(source),
    )
    tmp = path * ".tmp"
    try
        jldopen(tmp, "w") do f
            f["schema_version"]       = MEANS_DB_SCHEMA_VERSION
            f["param_schema_version"] = PARAM_SCHEMA_VERSION
            f["params"]   = params
            f["identity"] = ident
            f["meta"]     = meta
            f["summary"]  = summary
        end
        mv(tmp, path; force=true)
    catch e
        isfile(tmp) && rm(tmp; force=true)
        rethrow(e)
    end
    update_index && append_index!(root, [path])
    return path
end

"""Rewrite just the `summary` key of an existing file (read-modify-write)."""
function _update_summary!(path::AbstractString, updates::AbstractDict)
    _assert_writable(path, "_update_summary! (set_status!)")
    all = Dict{String,Any}()
    _open_jld2(path) do f
        for k in keys(f); all[string(k)] = f[k]; end
    end
    s = get(all, "summary", Dict{String,Any}())
    merge!(s, updates)
    all["summary"] = s
    tmp = path * ".tmp"
    try
        jldopen(tmp, "w") do f
            for (k, v) in all; f[k] = v; end
        end
        mv(tmp, path; force=true)
    catch e
        isfile(tmp) && rm(tmp; force=true)
        rethrow(e)
    end
    return path
end

"""
    set_status!(entry_or_path, status; root) -> String

Move a simulation along its lifecycle (`submitted` -> `running` -> `done`).
"""
function set_status!(entry_or_path, status::String; root::String=MEANS_DB_ROOT,
                     update_index::Bool=true)
    path = entry_or_path isa AbstractDict ? joinpath(root, entry_relpath(entry_or_path)) :
                                            String(entry_or_path)
    isfile(path) || error("not in the database: $path")
    _update_summary!(path, Dict{String,Any}("status" => status))
    update_index && append_index!(root, [path])
    return path
end

"""
    export_sim_info!(sim_info; root, source="", update_index=true) -> NamedTuple

Write every entry of an old-style `sim_info` into the database. This is the
cluster-side hook: `update_means` calls it, so simulations land in means_db
with no rebuild step and nothing to keep in sync.

Per-seed means AND the per-seed fields needed for re-averaging are taken from
`entry["simulations"][seed]`.
"""
function export_sim_info!(sim_info::AbstractDict; root::String=MEANS_DB_ROOT,
                          source::AbstractString="", cluster::String=MEANS_DB_CLUSTER,
                          update_index::Bool=true, verbose::Bool=true)
    written = 0; skipped = 0; failed = Tuple{String,String}[]
    for (key, entry) in sim_info
        string(key) == "updated_at" && continue
        entry isa AbstractDict || (skipped += 1; continue)
        get(entry, "means", nothing) isa AbstractDict || (skipped += 1; continue)

        sd = Dict{String,Any}()
        if get(entry, "simulations", nothing) isa AbstractDict
            for (seed, s) in entry["simulations"]
                s isa AbstractDict || continue
                info = Dict{String,Any}("seed" => seed)
                for fld in SEED_INFO_FIELDS
                    haskey(s, fld) && (info[fld] = s[fld])
                end
                d = Dict{String,Any}("info" => info)
                get(s, "means", nothing) isa AbstractDict && (d["means"] = s["means"])
                sd[string(seed)] = d
            end
        end
        try
            save_entry(entry; root=root, seed_data=(isempty(sd) ? nothing : sd),
                       source=source, cluster=cluster)
            written += 1
        catch e
            push!(failed, (string(key), first(split(sprint(showerror, e), '\n'))))
        end
    end
    verbose && @printf("[export_sim_info!] %d written, %d skipped (no usable means), %d failed\n",
                       written, skipped, length(failed))
    for (k, why) in first(failed, 5); @printf("    key=%-8s %s\n", k, why); end
    update_index && written > 0 && rebuild_index(; root=root, verbose=verbose)
    return (written=written, skipped=skipped, failed=failed)
end

# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

"""
    _select_obs(available, obs) -> Vector{String}

The observable names in `available` that `obs` asks for: all (`nothing`/`:all`), all
but the heavy ones (`:light`), or the given name(s).
"""
function _select_obs(available::Vector{String}, obs)
    (obs === nothing || obs === :all) && return available
    obs === :light && return filter(k -> !(k in HEAVY_OBS), available)
    want = obs isa AbstractVector ? Set(string.(obs)) : Set([string(obs)])
    return filter(k -> k in want, available)
end

"""
    _named_obs(obs) -> Union{Vector{String}, Nothing}

The requested observable names when `obs` is concrete (a name or list), else `nothing`
(`:all`/`:light` need the group listed). Naming them lets a reader open `group/name`
directly instead of listing the whole group: dozens fewer network round-trips per file.
"""
_named_obs(obs) = (obs === nothing || obs === :all || obs === :light) ? nothing :
                  (obs isa AbstractVector ? string.(obs) : [string(obs)])

"""Fill `dest[k] => group/k` for the selected observables in JLD2 group `gpath`,
without enumerating the group when the names are known."""
function _read_means_group!(dest::AbstractDict, f, gpath::String, obs)
    named = _named_obs(obs)
    if named === nothing
        for k in _select_obs(string.(collect(keys(f[gpath]))), obs)
            dest[k] = f["$gpath/$k"]
        end
    else
        for k in named
            haskey(f, "$gpath/$k") && (dest[k] = f["$gpath/$k"])
        end
    end
    return dest
end

"""
    load_entry(path; obs=:all, with_seeds=false) -> Dict

Read one simulation back into the OLD entry shape
(`params ∪ meta ∪ {"means"=>...}`), so existing get_sims / plot_* helpers
consume it unchanged.

`obs` restricts what is read off disk — `:all`, `:light`, or a vector of names.
This is where the speed comes from: ~5 ms for a few observables vs ~120 ms for
all of them.
"""
function load_entry(path::AbstractString; obs=:all, with_seeds::Bool=false)
    _open_jld2(path) do f
        e = Dict{String,Any}()
        for (k, v) in f["params"]; e[string(k)] = v; end
        haskey(f, "meta") && for (k, v) in f["meta"]; e[string(k)] = v; end
        haskey(f, "summary") && (e["db_status"] = string(get(f["summary"], "status", DB_STATUS_DONE)))
        # A registered-but-not-yet-run simulation has no `means` group at all.
        m = Dict{String,Any}()
        haskey(f, "means") && _read_means_group!(m, f, "means", obs)
        e["means"] = m
        if with_seeds && haskey(f, "seeds")
            sims = Dict{Any,Any}()
            for s in keys(f["seeds"])
                d = Dict{String,Any}()
                haskey(f, "seeds/$s/info") && merge!(d, f["seeds/$s/info"])
                if haskey(f, "seeds/$s/means")
                    d["means"] = _read_means_group!(Dict{String,Any}(), f, "seeds/$s/means", obs)
                end
                sims[something(tryparse(Int, string(s)), string(s))] = d
            end
            e["simulations"] = sims
        end
        return e
    end
end

# ---------------------------------------------------------------------------
# Index — a CACHE, never the authority. Every file carries its own params, so
# rebuild_index() can always regenerate it; a stale index costs a rebuild,
# never data.
# ---------------------------------------------------------------------------

"""
    _is_scalarish(v) -> Bool

True for values that fit in one index column (number, string, Bool, Symbol).
"""
_is_scalarish(v) = v isa Number || v isa AbstractString || v isa Bool || v isa Symbol

"""Collapse a column to a concrete element type so filtering is type-stable."""
function _typed_column(vals::Vector{Any})
    present = Iterators.filter(v -> v !== missing, vals)
    numeric = !isempty(present) && all(v -> v isa Number && !(v isa Bool), present)
    return numeric ? Float64[v === missing ? NaN : Float64(v) for v in vals] :
                     String[v === missing ? ""  : string(v)  for v in vals]
end

# root => (mtime of index.jld2, index). Re-reading a 5000-row index on every
# query cost more than reading the data itself (2.5 s -> 0.001 s cached).
const _INDEX_CACHE = Dict{String,Tuple{Float64,Dict{String,Any}}}()

"""One index row: parameters + the small `summary` header. Reads ~5 ms."""
function _index_row(path::AbstractString, root::AbstractString)
    _open_jld2(path) do f
        r = Dict{String,Any}()
        for (k, v) in f["params"]; _is_scalarish(v) && (r[string(k)] = v); end
        # Old entries predate some identity params (e.g. the density-channel
        # alpha/beta_phi/Qx/Qy). A missing identity param means "as before" = its
        # registered default (that is exactly how identity_of / the hash treat
        # it), so fill scalar defaults here to keep the index queryable by them
        # WITHOUT rewriting any data file. The index is a rebuildable cache.
        for (k, d) in IDENTITY_PARAMS
            (d !== nothing && _is_scalarish(d) && !haskey(r, string(k))) && (r[string(k)] = d)
        end
        if haskey(f, "summary")
            s = f["summary"]
            r["n_measured"] = get(s, "n_measured", NaN)
            r["n_seeds"]    = get(s, "n_seeds", 0)
            r["n_obs"]      = get(s, "n_obs", 0)
            r["status"]     = string(get(s, "status", DB_STATUS_DONE))
            r["cluster"]    = string(get(s, "cluster", ""))
        end
        r["relpath"] = relpath(String(path), String(root))
        r
    end
end

"""
    scan_files(root=MEANS_DB_ROOT) -> Vector{String}

Every means_db file under `root` (all `.jld2` except index.jld2), sorted.
"""
function scan_files(root::String=MEANS_DB_ROOT)
    out = String[]
    isdir(root) || return out
    for (dir, _, files) in walkdir(root)
        for fn in files
            endswith(fn, ".jld2") && fn != "index.jld2" && push!(out, joinpath(dir, fn))
        end
    end
    return sort(out)
end

"""
    rebuild_index(; root, verbose=true) -> Dict

Walk the tree, read only `params` + `summary` from each file, write
`index.jld2` + `index.csv`.

Columns are the UNION of every scalar parameter seen, not a fixed list — so a
parameter you start recording later becomes queryable with no schema change.
"""
function rebuild_index(; root::String=MEANS_DB_ROOT, verbose::Bool=true)
    _assert_writable(root, "rebuild_index")
    files = scan_files(root)
    verbose && @printf("[rebuild_index] scanning %d files under %s\n", length(files), root)
    rows = Dict{String,Any}[]
    bad = String[]
    t = @elapsed for (i, path) in enumerate(files)
        try
            push!(rows, _index_row(path, root))
        catch e
            @warn "unreadable, skipping" path exception=e
            push!(bad, path)
        end
        verbose && i % 500 == 0 && (@printf("  %d/%d\n", i, length(files)); flush(stdout))
    end

    # "relpath" must exist even with zero rows: the first `register_simulation`
    # into a fresh database runs after a `find_entries` that already created an
    # empty index, and `append_index!` needs the column to be there.
    cols = sort(collect(reduce(union!, (Set(keys(r)) for r in rows),
                               init=Set{String}(["relpath"]))))
    index = Dict{String,Any}("columns" => cols, "n" => length(rows),
                             "built_at" => Dates.now(), "root" => root)
    for c in cols
        index[c] = _typed_column(Any[get(r, c, missing) for r in rows])
    end
    mkpath(root)
    JLD2.save(_db_index_jld2(root), index)
    _INDEX_CACHE[root] = (mtime(_db_index_jld2(root)), index)

    ordered = vcat(["relpath"], filter(!=("relpath"), cols))
    open(_db_index_csv(root), "w") do io
        println(io, join(ordered, ","))
        for r in rows
            println(io, join((_csvcell(get(r, c, missing)) for c in ordered), ","))
        end
    end
    verbose && @printf("[rebuild_index] %d entries, %d columns in %.1fs -> %s\n",
                       length(rows), length(cols), t, _db_index_jld2(root))
    isempty(bad) || @printf("  *** %d unreadable file(s) skipped ***\n", length(bad))
    return index
end

"""
    _csvcell(v) -> String

One CSV cell: empty for `missing`, commas replaced by `;`.
"""
_csvcell(v) = v === missing ? "" : replace(string(v), "," => ";")

"""
    append_index!(root, paths) -> Dict

Add or replace the index rows for `paths` without rescanning the tree.

A submission job registers one simulation at a time; a full `rebuild_index`
(~100 s at 5000 files) per job would dominate everything else. Falls back to a
full rebuild if there is no index yet.
"""
function append_index!(root::String, paths::AbstractVector)
    isempty(paths) && return load_index(; root=root)
    _assert_writable(root, "append_index!")
    p = _db_index_jld2(root)
    isfile(p) || return rebuild_index(; root=root, verbose=false)

    idx = load_index(; root=root)
    cols = Set(String.(get(idx, "columns", String[])))
    N = Int(get(idx, "n", 0))

    new_rows = Dict{String,Any}[]
    for path in paths
        try
            push!(new_rows, _index_row(path, root))
        catch e
            @warn "cannot index" path exception=e
        end
    end
    isempty(new_rows) && return idx

    replacing = Set(String[r["relpath"] for r in new_rows])
    rp = get(idx, "relpath", String[])
    keep = (N == 0 || isempty(rp)) ? Int[] :
           [i for i in 1:N if !(String(rp[i]) in replacing)]

    union!(cols, reduce(union!, (Set(keys(r)) for r in new_rows),
                        init=Set{String}(["relpath"])))
    ordered = sort(collect(cols))
    out = Dict{String,Any}("columns" => ordered, "n" => length(keep) + length(new_rows),
                           "built_at" => Dates.now(), "root" => root)
    for c in ordered
        old_vals = haskey(idx, c) ? Any[idx[c][i] for i in keep] :
                                    Any[missing for _ in keep]
        new_vals = Any[get(r, c, missing) for r in new_rows]
        out[c] = _typed_column(vcat(old_vals, new_vals))
    end
    JLD2.save(p, out)
    _INDEX_CACHE[root] = (mtime(p), out)
    return out
end

"""
    _mount_drive_letters() -> String

Drive letters of the cluster mounts on this machine: those of env `LANDAU_MOUNT` /
`CURIE_MOUNT`, of `LANDAU_ROOT` / `CURIE_ROOT` when not on the backup drive, and
any listed in env `MEANS_DB_MOUNT_DRIVES` (e.g. "YZ"). No built-in letters.
"""
function _mount_drive_letters()
    letters = uppercase(get(ENV, "MEANS_DB_MOUNT_DRIVES", ""))
    for v in ("LANDAU_MOUNT", "CURIE_MOUNT", "LANDAU_ROOT", "CURIE_ROOT")
        p = strip(get(ENV, v, ""))
        (length(p) >= 2 && p[2] == ':') || continue
        endswith(v, "_ROOT") && _on_backup_drive(p) && continue
        letters *= uppercase(p[1])
    end
    return letters
end

"Drive letters treated as cluster mounts (`_mount_drive_letters`), fixed when this file is included."
const CLUSTER_MOUNT_DRIVES = _mount_drive_letters()

"""True if `root` is on a RaiDrive cluster mount (a mount drive letter or the RaiDrive UNC path)."""
function _on_cluster_mount(root::AbstractString)
    r = String(root)
    startswith(lowercase(r), raw"\\raidrive") && return true
    return length(r) >= 2 && r[2] == ':' && uppercase(r[1]) in CLUSTER_MOUNT_DRIVES
end

"""
A means_db root on a cluster mount has no `index.jld2`. `load_index` refuses to
rebuild it from here: that would open every file on the cluster's storage master
over SFTP and write a new index there (FHRC cluster rules, 2026-10-20).
"""
struct MountIndexMissing <: Exception
    root::String
end
"""
    showerror(io, e::MountIndexMissing)

Explain why the index is not rebuilt over a mount and what to do instead.
"""
function Base.showerror(io::IO, e::MountIndexMissing)
    print(io, "MountIndexMissing: no index.jld2 under ", e.root, "\n",
          "  This root is on a cluster mount, so the index is NOT rebuilt from this machine:\n",
          "  that would open every means_db file on the cluster's storage master over SFTP\n",
          "  and write a new index there.\n",
          "  Most likely RaiDrive's listing of the folder is stale, or a cluster job is\n",
          "  rewriting the index right now. Open the folder in Explorer, wait a minute, retry.\n",
          "  If the index is really gone, rebuild it ON the cluster, not from here.\n",
          "  To query the other cluster only, pass `roots=[...]` or set MEANS_DB_ROOTS.")
end

"""
    load_index(; root=MEANS_DB_ROOT, verbose=false, refresh=false) -> Dict

The index of one means_db root (one row per simulation: parameters + summary).
Cached in memory and re-read only when index.jld2 changes, or with `refresh=true`.
If there is no index it is rebuilt (`rebuild_index`), except on a cluster mount,
where `MountIndexMissing` is thrown instead.
"""
function load_index(; root::String=MEANS_DB_ROOT, verbose::Bool=false, refresh::Bool=false)
    p = _db_index_jld2(root)
    if !isfile(p)
        _on_cluster_mount(root) && throw(MountIndexMissing(root))
        return rebuild_index(; root=root, verbose=verbose)
    end
    mt = mtime(p)
    if !refresh && haskey(_INDEX_CACHE, root)
        cached_mt, idx = _INDEX_CACHE[root]
        cached_mt == mt && return idx
    end
    # IOStream, not JLD2.load's default mmap — same network-mount reason as
    # _open_jld2. Reads every top-level key into a plain Dict.
    idx = _open_jld2(p) do f
        d = Dict{String,Any}()
        for k in keys(f); d[string(k)] = f[k]; end
        d
    end
    _INDEX_CACHE[root] = (mt, idx)
    return idx
end

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

"""
    _matches(actual, want; tol=1e-9) -> Bool

Whether an index value matches a selector: `nothing` matches anything; a vector, set or
tuple matches any of its elements; numbers match within `tol` (NaN never); otherwise
string equality. `missing` matches nothing.
"""
function _matches(actual, want; tol::Float64=1e-9)
    want === nothing && return true
    if want isa AbstractVector || want isa AbstractSet || want isa Tuple
        return any(w -> _matches(actual, w; tol=tol), want)
    end
    actual === missing && return false
    if actual isa Number && want isa Number
        (isnan(Float64(actual)) || isnan(Float64(want))) && return false
        return isapprox(Float64(actual), Float64(want); atol=tol)
    end
    return string(actual) == string(want)
end

"""
    find_entries(; root, n=nothing, n_tol=0.1, kw...) -> Vector{String}

Relative paths of simulations matching the selectors. Any indexed column is a
valid keyword — `U`, `Lx`, `beta`, `TClustersize`, `V`, `impurity`,
`diag_method`, `B`, ... Scalars or vectors (`U=[3.0, 5.0]`). Unset = wildcard.

`n` is special: it filters the MEASURED filling loosely (`|n - n_measured| <=
n_tol`), since achieved n drifts from `wanted_n`.
"""
function find_entries(; root::String=MEANS_DB_ROOT, index=nothing,
                      n=nothing, n_tol::Float64=0.1, status=nothing, kw...)
    idx = index === nothing ? load_index(; root=root) : index
    N = idx["n"]::Int
    N == 0 && return String[]
    cols = Set(idx["columns"])
    for (k, _) in kw
        string(k) in cols ||
            @warn "no such indexed column: $k (have: $(join(sort(collect(cols)), ", ")))"
    end
    keep = trues(N)
    # `status` is supplied by default rather than by the caller, and databases
    # written before it existed simply have no such column — treat those as
    # finished rather than warning on every query.
    if status !== nothing && "status" in cols
        col = idx["status"]
        @inbounds for i in 1:N
            keep[i] && (keep[i] = _matches(col[i], status))
        end
    end
    for (k, want) in kw
        want === nothing && continue
        sk = string(k); sk in cols || continue
        col = idx[sk]
        @inbounds for i in 1:N
            keep[i] && (keep[i] = _matches(col[i], want))
        end
    end
    if n !== nothing && "n_measured" in cols
        col = idx["n_measured"]
        @inbounds for i in 1:N
            keep[i] || continue
            v = col[i]
            keep[i] = v isa Number && !isnan(v) && abs(Float64(v) - n) <= n_tol
        end
    end
    rp = idx["relpath"]
    return String[rp[i] for i in 1:N if keep[i]]
end

"""
    load_means(; obs=:all, root, n=nothing, n_tol=0.1, kw...) -> Dict

The main entry point. Assemble a `sim_info`-shaped Dict of ONLY the matching
simulations, reading one small file each.

    si = load_means(; U=3.0, Lx=12)                          # everything
    si = load_means(; U=[3.0,5.0], Lx=12, obs=[:stiffness])  # ~24x faster
    si = load_means(; U=3.0, obs=:light)                     # skip LDOS & friends

Pass `obs` whenever you know what you are plotting — that is the whole point.
"""
function load_means(; root::String=MEANS_DB_ROOT, obs=:all, n=nothing, n_tol::Float64=0.1,
                    with_seeds::Bool=false, verbose::Bool=true, index=nothing,
                    status=DB_STATUS_DONE, kw...)
    idx = index === nothing ? load_index(; root=root) : index
    # Registered-but-unrun simulations are in the index too; by default this
    # returns only ones that actually have means. Pass `status=nothing` for all.
    sel = find_entries(; root=root, index=idx, n=n, n_tol=n_tol, status=status, kw...)
    si = Dict{String,Any}()
    t = @elapsed for (i, rp) in enumerate(sel)
        try
            si[string(i)] = load_entry(joinpath(root, rp); obs=obs, with_seeds=with_seeds)
        catch e
            @warn "failed to read entry, skipping" relpath=rp exception=e
        end
    end
    verbose && @printf("[load_means] %d/%d simulations in %.2fs (obs=%s)\n",
                       length(si), idx["n"], t,
                       obs isa Symbol ? string(obs) : "$(length(obs)) selected")
    return si
end

# ---------------------------------------------------------------------------
# Federation across clusters
#
# Landau and Curie each own a means_db on their own filesystem. This machine
# mounts both, so reading is a union over roots. Every simulation carries its
# full parameters, and the identity hash is platform-independent, so entries
# from different clusters compose without any coordination.
# ---------------------------------------------------------------------------

"""
    drive_means_db_roots(drive=_backup_drive()) -> Vector{String}

The two means_db folders on the backup drive, Landau first:
`<drive>/landau/MC_Hubbard_V7/means_db`, `<drive>/curie/MC_Hubbard_V7/means_db`.
"""
drive_means_db_roots(drive::AbstractString=_backup_drive()) =
    String[joinpath(drive, c, "MC_Hubbard_V7", "means_db") for c in ("landau", "curie")]

"""
    cluster_root(cluster) -> String

The folder on this machine that holds `cluster`'s `MC_Hubbard_V7/` and
`MC_Hubbard_V6/` (on the cluster itself: `/home/data/<user>/data`). Each cluster is
looked up on its own, the first that applies:
1. env `LANDAU_ROOT` / `CURIE_ROOT`: any folder with that layout, e.g. a cluster
   mount's `data` folder, the drive's `<drive>/curie`, or a local copy;
2. env `BACKUP_DRIVE` set: `<drive>/<cluster>`;
3. env `LANDAU_MOUNT` / `CURIE_MOUNT` set (a mount of the cluster's
   `/home/data/<user>`, as in `fix_cluster_path`): `<mount>/data`;
4. nothing set: `""` (no folder for that cluster).
So one cluster can be read from the drive and the other from its cluster mount.
There are no built-in locations. Read each time it is called.
"""
function cluster_root(cluster)
    c = lowercase(string(cluster))
    c in ("landau", "curie") || error("unknown cluster '$cluster' (expected landau or curie)")
    r = String(strip(get(ENV, uppercase(c) * "_ROOT", "")))
    isempty(r) || return r
    d = _backup_drive()
    isempty(d) || return joinpath(d, c)
    m = String(strip(get(ENV, uppercase(c) * "_MOUNT", "")))
    isempty(m) && return ""
    occursin(r"^[A-Za-z]:$", m) && (m *= "\\")        # a bare drive letter: joinpath would give "Y:data"
    return joinpath(m, "data")
end

"""
    cluster_means_db_roots() -> Vector{String}

`<cluster_root>/MC_Hubbard_V7/means_db` for Landau and Curie (see `cluster_root`),
leaving out a cluster with no folder set.
"""
cluster_means_db_roots() =
    String[joinpath(cluster_root(c), "MC_Hubbard_V7", "means_db")
           for c in ("landau", "curie") if !isempty(cluster_root(c))]

"""
Roots to search, in order. Fixed when this file is included:
1. env `MEANS_DB_ROOTS` (path separator `;`), if set;
2. else `cluster_means_db_roots()`: per cluster, `LANDAU_ROOT`/`CURIE_ROOT`, else the
   backup drive (`BACKUP_DRIVE`), else the cluster mount (`LANDAU_MOUNT`/`CURIE_MOUNT`).
With none of these set it is empty: set one before including this file.
Roots that do not exist are skipped, so this works with one cluster unmounted.
"""
const MEANS_DB_ROOTS = haskey(ENV, "MEANS_DB_ROOTS") ?
    String.(filter(!isempty, split(ENV["MEANS_DB_ROOTS"], ';'))) :
    cluster_means_db_roots()

"""
    find_all(; roots=MEANS_DB_ROOTS, kw...) -> Vector{Dict}

Rows matching the selectors across every root. Each row carries its index
columns plus `root` and `path`. Same selectors as `find_entries`.
"""
function find_all(; roots=MEANS_DB_ROOTS, n=nothing, n_tol::Float64=0.1,
                  status=DB_STATUS_DONE, verbose::Bool=false, kw...)
    out = Dict{String,Any}[]
    isempty(roots) && @warn "no means_db to read: set ENV[\"BACKUP_DRIVE\"], LANDAU_ROOT/CURIE_ROOT, " *
                            "LANDAU_MOUNT/CURIE_MOUNT or MEANS_DB_ROOTS before including means_db.jl" maxlog=1
    for root in roots
        isdir(root) || (verbose && @info "skipping missing root" root; continue)
        idx = try
            load_index(; root=root)
        catch e
            # must stop the query, not just skip the root
            (e isa MountIndexMissing || e isa DriveReadOnly) && rethrow()
            @warn "unreadable index, skipping root" root exception=e; continue
        end
        sel = find_entries(; root=root, index=idx, n=n, n_tol=n_tol, status=status, kw...)
        isempty(sel) && continue
        rp = String.(idx["relpath"])
        pos = Dict(rp[i] => i for i in 1:length(rp))
        for r in sel
            i = pos[r]
            row = Dict{String,Any}()
            for c in idx["columns"]
                v = idx[c][i]
                (v isa AbstractFloat && isnan(v)) || (v isa AbstractString && isempty(v)) || (row[c] = v)
            end
            row["root"] = root
            row["path"] = joinpath(root, r)
            push!(out, row)
        end
    end
    return out
end

# Columns that are bookkeeping, not physics — differing values here are fine.
const _NON_PHYSICS_COLS = Set(["relpath", "path", "root", "cluster", "status",
                               "n_measured", "n_seeds", "n_obs", "cores"])

"""
    varied_params(rows) -> Vector{String}

Which physical parameters actually differ across a set of matched simulations.

An unpinned selector silently widens an average: asking for `U=5, Lx=20` without
`wanted_n` happily averages n=0.5 together with n=0.875. This makes that visible
instead of leaving it in the numbers.
"""
function varied_params(rows::AbstractVector)
    length(rows) < 2 && return String[]
    cols = reduce(union!, (Set(keys(r)) for r in rows), init=Set{String}())
    out = String[]
    for c in cols
        c in _NON_PHYSICS_COLS && continue
        vals = unique(get(r, c, missing) for r in rows)
        length(vals) > 1 && push!(out, c)
    end
    return sort(out)
end

"""
    variant_groups(rows; ignore=()) -> Vector{NamedTuple}

Partition matched rows into the DISTINCT physical variants hiding inside them.
`varied_params` says *which* params differ; this says *what the actual
combinations are*, so an unpinned selector shows up as "3 variants" instead of
one silently-averaged blob.

Groups by the tuple of values over the varying params, minus `ignore` — the axes
you mean to average over (`disorder_seed`, and whatever `x` a curve sweeps). Each
group carries its distinguishing `params`, the sample count `n`, and its
`disorder_seeds`. Sorted by descending `n` (the dominant variant first).

    for g in variant_groups(hits; ignore=("disorder_seed",))
        @info g.params g.n
    end
"""
function variant_groups(rows::AbstractVector; ignore=())
    ig = Set(string.(ignore))
    vary = filter(c -> !(c in ig), varied_params(rows))
    isempty(vary) && return NamedTuple[]
    keyf(r) = Tuple(get(r, c, missing) for c in vary)
    groups = Dict{Any,Vector{Any}}()
    for r in rows
        push!(get!(groups, keyf(r), Any[]), r)
    end
    out = NamedTuple[]
    for (k, rs) in groups
        push!(out, (params = Dict{String,Any}(vary[i] => k[i] for i in eachindex(vary)),
                    n = length(rs),
                    disorder_seeds = sort(unique(get(r, "disorder_seed", missing) for r in rs))))
    end
    sort!(out, by = g -> -g.n)
    return out
end

"""One line per variant: `[N sims] a=..  b=..`, params in stable order."""
function _format_variants(groups::AbstractVector; indent="    ")
    isempty(groups) && return ""
    ks = sort(collect(mapreduce(g -> keys(g.params), union, groups; init=Set{String}())))
    io = IOBuffer()
    for g in groups
        parts = ["$k=" * _canon_str(g.params[k]) for k in ks if haskey(g.params, k)]
        print(io, indent, @sprintf("[%d sims] ", g.n), join(parts, "  "), "\n")
    end
    return String(take!(io))
end

"""
    _check_variants(hits, groups; label, strict) -> groups

Shared variant reporting for `get_observable`/`get_curve`. Warns (listing each
distinct combination and its count) or, under `strict`, errors — so a query that
silently averaged across physics now says exactly what it averaged.
"""
function _check_variants(hits, groups; label::AbstractString, strict::Bool)
    isempty(groups) && return groups
    msg = "$label matched $(length(groups)) physics variants over " *
          "$(sum(g.n for g in groups)) simulations — pin these or you are " *
          "averaging across them:\n" * _format_variants(groups)
    strict ? error(msg) : @warn msg
    return groups
end

# --- averaging that works for scalars and for vector-valued observables ------
"""
    _as_num(x)

`x` as a Float64 if it is a number, unchanged otherwise (e.g. a vector).
"""
_as_num(x) = x isa Number ? Float64(x) : x

"""
    _mean_of(vs)

Element-wise mean of a list of scalars or equal-length vectors.
"""
_mean_of(vs) = length(vs) == 1 ? vs[1] : sum(vs) ./ length(vs)

"""
    _std_of(vs, mu)

Element-wise sample standard deviation of `vs` around `mu` (zero for one value).
"""
function _std_of(vs, mu)
    length(vs) < 2 && return zero.(mu)
    sqrt.(sum(((v .- mu) .^ 2 for v in vs)) ./ (length(vs) - 1))
end

"""
    get_observable(; obs, seeds=false, mc_seed=nothing, roots=MEANS_DB_ROOTS, kw...)

Read ONE observable for the simulations matching the selectors, across both
clusters, pulling only that observable off disk.

Because `disorder_seed` is part of a simulation's identity, leaving it
unspecified matches every disorder realization of the same physical point, and
those are averaged — a disorder average. The spread between samples is physical,
not noise, so it is reported alongside.

    get_observable(; U=3.0, Lx=4, V=1.0, beta=20.0, obs=:stiffness)
      -> (value, error, std, stat_error, n_samples, disorder_seeds, clusters)

`seeds=true` returns the per-sample breakdown instead of the average:

    get_observable(; U=3.0, Lx=4, V=1.0, beta=20.0, obs=:stiffness, seeds=true)
      -> (seeds = [1, 2, 3, 4], values = [...], errors = [...],
          clusters = ["curie", "landau", ...], mc_seeds = [[1], [1], ...])

`mc_seed=k` reads that Monte-Carlo seed's own means rather than the seed-average
within each sample.

`error` is the uncertainty on the disorder average (sample scatter / sqrt N),
`std` the scatter itself, `stat_error` the propagated Monte-Carlo error. Returns
`nothing` if nothing matches.
"""
function get_observable(; obs, roots=MEANS_DB_ROOTS, seeds::Bool=false,
                        mc_seed=nothing, n=nothing, n_tol::Float64=0.1,
                        verbose::Bool=false, warn_varied::Bool=true,
                        strict::Bool=false, kw...)
    o = string(obs)
    hits = find_all(; roots=roots, n=n, n_tol=n_tol, kw...)
    if isempty(hits)
        verbose && @info "no simulations match" selectors=kw
        return nothing
    end

    # Averaging over anything except the disorder realization is almost always
    # an unpinned selector rather than an intention. Enumerate the distinct
    # variants (not just their names) so the mistake is visible; `strict` errors.
    variants = variant_groups(hits; ignore=("disorder_seed",))
    (warn_varied || strict) && _check_variants(hits, variants; label="get_observable", strict=strict)
    varied = filter(!=("disorder_seed"), varied_params(hits))

    # Lean read: open each file ONCE and go straight to the requested key. This
    # is the difference between ~8 ms and ~700 ms per file over the SFTP mount —
    # `load_entry`/`keys(f["means"])` would enumerate ALL ~60 observables' group
    # structure (dozens of round-trips) just to keep the one you asked for. Seed
    # keys are read only when `seeds=true`, in the SAME open.
    obskey = mc_seed === nothing ? "means/$o" : "seeds/$(mc_seed)/means/$o"
    dseeds = Any[]; vals = Any[]; errs = Any[]; clusters = String[]
    mcseeds = Vector{Any}[]; paths = String[]
    for row in sort(hits, by = r -> get(r, "disorder_seed", 0))
        p = row["path"]
        rd = _open_jld2(p) do f
            haskey(f, obskey) || return nothing
            d = f[obskey]
            (d isa AbstractDict && haskey(d, "value")) || return nothing
            mcs = (seeds && haskey(f, "seeds")) ? sort(string.(collect(keys(f["seeds"])))) : String[]
            (value = d["value"], err = get(d, "error", 0.0), mc_seeds = mcs)
        end
        rd === nothing && continue
        push!(dseeds,  get(row, "disorder_seed", missing))
        push!(vals,    _as_num(rd.value))
        push!(errs,    _as_num(rd.err))
        push!(clusters, string(get(row, "cluster", "")))
        push!(mcseeds, rd.mc_seeds)
        push!(paths,   p)
    end

    if isempty(vals)
        verbose && @info "matched $(length(hits)) simulation(s) but none carry `$o`"
        return nothing
    end

    seeds && return (seeds = dseeds, values = vals, errors = errs,
                     clusters = clusters, mc_seeds = mcseeds, paths = paths,
                     n_samples = length(vals), varied = varied, variants = variants)

    mu = _mean_of(vals)
    sd = _std_of(vals, mu)
    N  = length(vals)
    return (value        = mu,
            error        = N > 1 ? sd ./ sqrt(N) : errs[1],   # on the disorder average
            std          = sd,                                 # sample-to-sample scatter
            stat_error   = sqrt.(sum(e .^ 2 for e in errs)) ./ N,
            n_samples    = N,
            disorder_seeds = dseeds,
            clusters     = clusters,
            varied       = varied,    # parameters averaged over besides disorder_seed
            variants     = variants)  # distinct physics variants folded into this average
end

"""
    get_mu(; roots=MEANS_DB_ROOTS, mc_seed=nothing, n=nothing, n_tol=0.1,
           warn_varied=true, strict=false, verbose=false, kw...) -> NamedTuple

Chemical potential μ for the matched simulation(s). μ is a converged OUTPUT, not
a config parameter — it is NOT in `params` (so `get_param` can't see it) and is
never computed as a measurement mean (so `get_observable` can't either). It is
taken from each seed's stored `info["mu"]` when present, else recovered from
`sim_data["mu"]` in that seed's `results.jld2`.

`disorder_seed` is part of a point's identity, so leaving it unpinned matches
every disorder realization and μ is disorder-averaged, like `get_observable`. The
per-sample values and where each came from (`"info"`/`"sim_data"`) are returned
alongside so an outlier is visible.

    get_mu(; U=4.0, Lx=14, V=0.0, alpha=1.0, beta_phi=0.5, wanted_n=0.9, beta=10.0)
      -> (value, error, std, n_samples, disorder_seeds, mc_seeds, values, sources)

`mc_seed=k` keeps only that Monte-Carlo seed within each match. The `sim_data`
fallback reads the raw files on the cluster mounts and needs the reader wired by
sim_num_control.jl (`MEANS_DB_SIMDATA_FN`) — but if every match already carries μ
in its info block, no raw file is opened. Returns `nothing` if nothing matches or
μ is found nowhere.
"""
function get_mu(; roots=MEANS_DB_ROOTS, mc_seed=nothing, n=nothing, n_tol::Float64=0.1,
                warn_varied::Bool=true, strict::Bool=false, verbose::Bool=false, kw...)
    hits = find_all(; roots=roots, n=n, n_tol=n_tol, kw...)
    if isempty(hits)
        verbose && @info "get_mu: no simulations match" selectors=kw
        return nothing
    end
    variants = variant_groups(hits; ignore=("disorder_seed",))
    (warn_varied || strict) && _check_variants(hits, variants; label="get_mu", strict=strict)

    read_sd = MEANS_DB_SIMDATA_FN[]
    dseeds = Any[]; mcseeds = Any[]; vals = Float64[]; sources = String[]
    for row in sort(hits, by = r -> get(r, "disorder_seed", 0))
        for (s, d) in read_seeds(row["path"]; with_means=false)
            mc_seed === nothing || string(mc_seed) == string(s) || continue
            info = get(d, "info", Dict{String,Any}())
            μ = get(info, "mu", missing)
            src = "info"
            if !(μ isa Number && isfinite(μ))   # not exported here — fall back to raw sim_data
                dp = string(get(info, "dir_path", ""))
                if read_sd !== nothing && !isempty(dp)
                    got = try
                        read_sd(dp, "mu"; cluster=string(get(row, "cluster", "")))
                    catch e
                        @warn "get_mu: could not read sim_data[\"mu\"] from $dp" exception=e
                        nothing
                    end
                    if got isa Number && isfinite(got)
                        μ = got; src = "sim_data"
                    end
                end
            end
            (μ isa Number && isfinite(μ)) || continue
            push!(dseeds, get(row, "disorder_seed", missing))
            push!(mcseeds, something(tryparse(Int, string(s)), s))
            push!(vals, Float64(μ))
            push!(sources, src)
        end
    end

    if isempty(vals)
        if read_sd === nothing
            @warn "get_mu: matched $(length(hits)) sim(s) but none carry μ in their info " *
                  "block, and the sim_data fallback is unwired. `include(\"sim_num_control.jl\")` " *
                  "to enable reading μ from results.jld2, then retry."
        else
            verbose && @info "get_mu: matched $(length(hits)) sim(s) but μ found nowhere"
        end
        return nothing
    end

    mu = sum(vals) / length(vals)
    N  = length(vals)
    sd = N > 1 ? sqrt(sum((v - mu)^2 for v in vals) / (N - 1)) : 0.0
    return (value          = mu,
            error          = N > 1 ? sd / sqrt(N) : 0.0,   # on the disorder average
            std            = sd,                            # sample-to-sample scatter
            n_samples      = N,
            disorder_seeds = dseeds,
            mc_seeds       = mcseeds,
            values         = vals,
            sources        = sources)
end

"""
    get_stream(; obs, roots=MEANS_DB_ROOTS, mc_seed=nothing, vec_index=nothing,
               n=nothing, n_tol=0.1, warn_varied=true, strict=false,
               verbose=false, kw...) -> NamedTuple

`get_observable` for the RAW Monte-Carlo stream. Same selector interface — pin a
physical point with `U=`, `Lx=`, `beta=`, `wanted_n=`/`n=`, `disorder_seed=`, … —
but instead of the seed-averaged mean it returns the per-measurement time series
`obs` produced during the run.

A stream is one Markov chain, so it CANNOT be averaged the way `get_observable`
averages means: each (disorder_seed, mc_seed) is its own chain. They are returned
individually, like `get_observable(...; seeds=true)`:

    r = get_stream(; U=3.0, Lx=12, beta=20.0, wanted_n=0.875, obs=:DdD)
    plot(r.streams[1]; xlabel="MC step")          # first chain
    map(mean, r.streams)                           # per-chain means

`obs` must be a raw field of `MeasureData` (`:DdD`, `:n`, `:Ss`, `:Kx`, `:Ky`,
`:chi_spin`, `:Lxx_qy`, `:Delta_operator`, …) — the quantities recorded once per
measurement. DERIVED observables that `get_observable`/`get_curve` serve
(`:stiffness`, `:resistivity_tensor`, `:conductivity_*`, `:drude_*`) are built by
`get_means` at averaging time and have no stream; asking for one errors with the
list of valid fields.

Vector-valued fields (`:chi_spin`, `:Lxx_qy`, `:Ps_l`, …) carry a vector per
measurement; `vec_index` picks the component (default 1, i.e. q=0). Complex fields
(`:Delta_operator`, `:Lxy_q0`, …) come back raw — take `abs.`/`real.` as needed.
`mc_seed=k` keeps only that Monte-Carlo seed's chain within each match.

Unlike the rest of means_db this reads the ~370 MB per-seed `results.jld2` (via
`load_results`), so it needs the simulation engine loaded: `include`
sim_num_control.jl (or MC_Hubburd.jl) first, which wires `MEANS_DB_STREAM_FN`.
Far heavier than `get_observable`, and needs the cluster mounts online. Returns
`nothing` if nothing matches.

Fields: `streams`, `disorder_seeds`, `mc_seeds`, `n_meas`, `clusters`,
`dir_paths`, `varied`, `variants`.
"""
function get_stream(; obs, roots=MEANS_DB_ROOTS, mc_seed=nothing,
                    vec_index::Union{Int,Nothing}=nothing,
                    n=nothing, n_tol::Float64=0.1, warn_varied::Bool=true,
                    strict::Bool=false, verbose::Bool=false, kw...)
    load_stream = MEANS_DB_STREAM_FN[]
    load_stream === nothing && error(
        "get_stream needs the raw results.jld2 loader, which means_db (JLD2-only) " *
        "does not carry. `include(\"sim_num_control.jl\")` (or MC_Hubburd.jl) first " *
        "to wire MEANS_DB_STREAM_FN, then retry.")

    hits = find_all(; roots=roots, n=n, n_tol=n_tol, kw...)
    if isempty(hits)
        verbose && @info "get_stream: no simulations match" selectors=kw
        return nothing
    end

    # Same "did you leave a physics param unpinned?" guard as get_observable.
    variants = variant_groups(hits; ignore=("disorder_seed",))
    (warn_varied || strict) && _check_variants(hits, variants; label="get_stream", strict=strict)
    varied = filter(!=("disorder_seed"), varied_params(hits))

    streams = Any[]; dseeds = Any[]; mcseeds = Any[]
    nmeas = Int[]; clusters = String[]; dirpaths = String[]

    for row in sort(hits, by = r -> get(r, "disorder_seed", 0))
        # Per-seed dir_path -> the seed's raw results.jld2. It lives inside the
        # means file (seeds/<seed>/info), not the index row, so open once.
        seed_dirs = _open_jld2(row["path"]) do f
            out = Tuple{String,String}[]
            haskey(f, "seeds") || return out
            for s in sort(string.(collect(keys(f["seeds"]))))
                mc_seed === nothing || string(mc_seed) == s || continue
                info = haskey(f, "seeds/$s/info") ? f["seeds/$s/info"] : Dict{String,Any}()
                dp = string(get(info, "dir_path", ""))
                isempty(dp) || push!(out, (s, dp))
            end
            return out
        end

        for (s, dp) in seed_dirs
            # the cluster tells which mount/drive folder holds this dir_path
            res = load_stream(dp, obs, vec_index; cluster=string(get(row, "cluster", "")))
            res === nothing && continue
            stream, nm, ddir = res
            push!(streams, stream)
            push!(dseeds, get(row, "disorder_seed", missing))
            push!(mcseeds, something(tryparse(Int, s), s))
            push!(nmeas, nm)
            push!(clusters, string(get(row, "cluster", "")))
            push!(dirpaths, ddir)
        end
    end

    if isempty(streams)
        verbose && @info "get_stream: matched $(length(hits)) sim(s) but loaded no `$obs` stream"
        return nothing
    end

    return (streams        = streams,
            disorder_seeds = dseeds,
            mc_seeds       = mcseeds,
            n_meas         = nmeas,
            clusters       = clusters,
            dir_paths      = dirpaths,
            varied         = varied,
            variants       = variants)
end

"""
    _diagnose_empty_query(; roots, obs, x=nothing, n, n_tol, kw...) -> String

Human-readable reason a curve/point query came back empty — for the warning
`get_curve` emits before returning `nothing`. Re-runs the match with the status
filter DROPPED (`status=nothing`) and reports which of three situations holds:

  1. nothing matches the selectors at all (any status) => those parameter values
     are not in the database; the printed `db_summary` may be stale.
  2. simulations match but are ALL still `submitted` (no means in means_db yet) =>
     either they have not run / not enough seeds finished, OR they have data on
     the cluster that `num_control` (update_means/export_sim_info!) has not
     exported. means_db cannot tell these apart, so both are named.
  3. `done` simulations match but none carry the requested observable / x-axis.
"""
function _diagnose_empty_query(; roots=MEANS_DB_ROOTS, obs, x=nothing,
                               n=nothing, n_tol::Float64=0.1, kw...)
    selstr = isempty(kw) ? "(no selectors)" :
             join(["$k=$(_canon_str(v))" for (k, v) in kw], ", ")
    all_hits = find_all(; roots=roots, n=n, n_tol=n_tol, status=nothing, kw...)
    if isempty(all_hits)
        return "no simulations match [$selstr] — nothing with these parameters is in " *
               "the database at all (any status). Those values are probably not there " *
               "yet: check the LIVE `db_summary(; root=...)` — a summary printed earlier " *
               "in the notebook can be stale."
    end
    ndone = count(r -> string(get(r, "status", "")) == DB_STATUS_DONE, all_hits)
    nsub  = count(r -> string(get(r, "status", "")) == DB_STATUS_SUBMITTED, all_hits)
    if ndone == 0
        return "$(length(all_hits)) simulation(s) match [$selstr], but ALL are still " *
               "`submitted` — none carry means in means_db yet. Either they have not run " *
               "(or not enough seeds have finished), OR they have run and accumulated data " *
               "on the cluster but `num_control` (update_means / export_sim_info!) has not " *
               "been run to export their means. Run num_control on the cluster for these " *
               "points, then `load_index(; root=..., refresh=true)` (or restart the kernel)."
    end
    return "$ndone done simulation(s) match [$selstr]" *
           (nsub > 0 ? " (plus $nsub still submitted)" : "") *
           ", but none provide observable `$(obs)`" *
           (x === nothing ? "" : " on axis `$(x)`") *
           ". Check the observable name (see the available-observables recipe) and that " *
           "`x` is an indexed column."
end

"""
    _curve_point(records, o) -> Union{NamedTuple, Nothing}

Disorder-average observable `o` over one group of index records (one point of
`get_curve`): `(value, error, std, n_samples)`, where `error` is std/√N for N > 1 samples
(the stored error for one) and `std` the sample-to-sample scatter.

It reads straight from the records `get_curve` already holds, so it avoids a trip
through `find_all` per point and keeps `get_curve`'s relabelling (a full-cluster
K0-TCA run treated as K0). Unreadable files are skipped with a warning. Returns
`nothing` if no record carries `o`.
"""
function _curve_point(records::AbstractVector, o::AbstractString)
    obskey = "means/$o"
    vals = Any[]; errs = Any[]
    for row in sort(records, by = r -> get(r, "disorder_seed", 0))
        rd = try
            _open_jld2(row["path"]) do f
                haskey(f, obskey) || return nothing
                d = f[obskey]
                (d isa AbstractDict && haskey(d, "value")) || return nothing
                (value = d["value"], err = get(d, "error", 0.0))
            end
        catch e
            # A corrupt/unreadable file must not sink the whole curve — report its
            # path (so every bad file surfaces in one pass) and skip it, exactly
            # as we skip a file that simply lacks the observable.
            @warn "skipping unreadable file" path=row["path"] exception=e
            continue
        end
        rd === nothing && continue
        push!(vals, _as_num(rd.value))
        push!(errs, _as_num(rd.err))
    end
    isempty(vals) && return nothing
    mu = _mean_of(vals)
    sd = _std_of(vals, mu)
    N  = length(vals)
    return (value = mu,
            error = N > 1 ? sd ./ sqrt(N) : errs[1],   # on the disorder average
            std   = sd,                                 # sample-to-sample scatter
            n_samples = N)
end

"""
    get_curve(; obs, x=:beta, roots=MEANS_DB_ROOTS, kw...)

`get_observable` swept over one parameter — the usual "observable vs T" shape.
Groups the matching simulations by `x`, disorder-averages within each group, and
returns them sorted by `x`.

    c = get_curve(; U=3.0, Lx=4, V=1.0, obs=:stiffness, x=:beta)
    plot(c.x, c.value, yerr=c.error)

Fields: `x`, `value`, `error`, `std`, `n_samples` (samples behind each point),
`variants` (distinct physics variants folded in — see below).

If the matched simulations differ in any physical parameter other than `x` and
`disorder_seed` (e.g. you left `alpha`/`beta_phi` unpinned so channel-on and
channel-off runs both matched), the curve would silently average across them.
`get_curve` now enumerates those variants in a warning and returns them in
`variants`; pass `strict=true` to error instead of averaging.
"""
function get_curve(; obs, x=:beta, roots=MEANS_DB_ROOTS, n=nothing,
                   n_tol::Float64=0.1, verbose::Bool=false, w_ix::Union{Int,Nothing}=nothing,
                   strict::Bool=false, kw...)
    xs = string(x)
    o  = string(obs)
    hits = find_all(; roots=roots, n=n, n_tol=n_tol, kw...)
    if isempty(hits)
        @warn "get_curve returned nothing: " *
              _diagnose_empty_query(; roots=roots, obs=obs, x=x, n=n, n_tol=n_tol, kw...)
        return nothing
    end

    # A K0-TCA run whose cluster spans the whole lattice performs no cluster
    # approximation — it is a full K0 diagonalization. Treat those records as
    # `diag_method="K0"` so the curve merges them with genuine K0 runs (identical
    # physics) instead of splitting off a spurious method variant. The cluster
    # size is not part of a K0 run's identity (see CONDITIONAL_IDENTITY), so pin
    # it to its default for every non-TCA record too; otherwise the relabelled
    # record and a genuine K0 file — same physics — would still look like two
    # `TClustersize` variants because they carry different raw cluster sizes.
    for r in hits
        if string(get(r, "diag_method", "")) == "K0-TCA" &&
           get(r, "TClustersize", 0) >= max(get(r, "Lx", 0), get(r, "Ly", 0))
            r["diag_method"] = "K0"
        end
        identity_relevant(r, "TClustersize") ||
            (r["TClustersize"] = IDENTITY_PARAMS["TClustersize"])
    end

    groups = Dict{Any,Vector{Dict{String,Any}}}()
    for r in hits
        haskey(r, xs) || continue
        push!(get!(groups, r[xs], Dict{String,Any}[]), r)
    end
    if isempty(groups)
        @warn "get_curve returned nothing: none of the matched simulations carry the " *
              "x-axis column `$xs`. " *
              _diagnose_empty_query(; roots=roots, obs=obs, x=x, n=n, n_tol=n_tol, kw...)
        return nothing
    end

    # Check for unpinned selectors ONCE for the whole curve rather than at every
    # point — `x` is swept on purpose, `disorder_seed` is averaged on purpose,
    # everything else is a distinct physical variant that must not be blended.
    variants = variant_groups(hits; ignore=("disorder_seed", xs))
    _check_variants(hits, variants; label="get_curve", strict=strict)
    varied = filter(c -> c != "disorder_seed" && c != xs, varied_params(hits))

    ks = sort(collect(keys(groups)))
    xv = Any[]; yv = Any[]; ev = Any[]; sv = Any[]; nv = Int[]
    for k in ks
        pt = _curve_point(groups[k], o)
        pt === nothing && continue
        push!(xv, k);
        if w_ix === nothing
            push!(yv, pt.value); 
            push!(ev, pt.error)
        else
            push!(yv, pt.value[w_ix]); 
            push!(ev, pt.error[w_ix])
        end

        push!(sv, pt.std); 
        push!(nv, pt.n_samples)
    end
    if isempty(xv)
        @warn "get_curve produced an empty curve: " *
              _diagnose_empty_query(; roots=roots, obs=obs, x=x, n=n, n_tol=n_tol, kw...)
    end
    return (x = xv, value = yv, error = ev, std = sv, n_samples = nv,
            varied = varied, variants = variants)
end

# ---------------------------------------------------------------------------
# Frequency-resolved observables — read-time ω-stitching
#
# The measurement grids (`w_conductivity_range`, `w_list`, `delta_fixed_values`)
# are part of a simulation's IDENTITY, so extending the ω range makes a NEW file
# with a new hash (correct: the Monte Carlo is recomputed). Those separate files
# are the same physical point measured on different frequency windows, and
# `get_spectrum` stitches them back into one spectrum on the union grid.
#
# It works cheaply because the grids are arrays => not index columns => two
# ω-window variants of one point look identical in the index. So `find_all`
# gathers them, we group by disorder realization, stitch the windows within each
# (inverse-variance where they overlap — independent MC estimates of the same
# point), then disorder-average across realizations exactly like `get_observable`.
#
# Each entry: how to read one file's stored value into (outer-coordinate, ω)
# cells. `outer` is the non-ω axis:
#   :delta => value is [δ][ω]; align δ BY VALUE, read from `dgrid`
#   :index => value is [site][ω]; align by position
#   :none  => value is [ω]
# ---------------------------------------------------------------------------
const FREQ_OBS = Dict{String,NamedTuple}(
    # conductivity family + Kubo L's: [δ][ω] on w_conductivity_range
    (o => (grid="w_conductivity_range", outer=:delta, dgrid="delta_fixed_values")
     for o in ("conductivity_regular_xx_fixed", "conductivity_regular_xy_fixed",
               "conductivity_regular_yx_fixed", "conductivity_regular_yy_fixed",
               "conductivity_regular_xx_fixed_extrap", "conductivity_regular_xy_fixed_extrap",
               "conductivity_regular_yx_fixed_extrap", "conductivity_regular_yy_fixed_extrap",
               "Lxx_w_fixed", "Lxy_w_fixed", "Lyx_w_fixed", "Lyy_w_fixed"))...,
    # density of states: [ω] on w_list
    "DOS"  => (grid="w_list", outer=:none,  dgrid=nothing),
    # local DOS: [site][ω] on w_list
    "LDOS" => (grid="w_list", outer=:index, dgrid=nothing),
)

"""
    _combine_estimates(vs, es) -> (mean, error)

Combine independent estimates `vs ± es` of one value: inverse-variance weighted when
every error is positive and finite, else the plain mean with the propagated error
(a missing or zero error cannot be weighted).
"""
function _combine_estimates(vs::AbstractVector, es::AbstractVector)
    length(vs) == 1 && return (Float64(vs[1]), Float64(es[1]))
    if all(e -> e > 0 && isfinite(e), es)
        w = [1.0 / e^2 for e in es]; W = sum(w)
        return (sum(w .* vs) / W, sqrt(1.0 / W))
    end
    m = sum(vs) / length(vs)
    return (m, sqrt(sum(e^2 for e in es)) / length(es))
end

"""
    _read_spectrum_file(path, o, spec, mc_seed)

Read one file's stored value and error of the spectrum-like observable `o` into aligned
[outer][ω] vectors, with the ω grid and the outer coordinates (`spec` says which grid
and which outer index). `mc_seed` selects one seed's means instead of the average.
Returns `nothing` if the file lacks the observable or the grid.
"""
function _read_spectrum_file(path::AbstractString, o::AbstractString,
                             spec::NamedTuple, mc_seed)
    _open_jld2(path) do f
        obskey = mc_seed === nothing ? "means/$o" : "seeds/$(mc_seed)/means/$o"
        (haskey(f, obskey) && haskey(f, "params")) || return nothing
        d = f[obskey]
        (d isa AbstractDict && haskey(d, "value")) || return nothing
        p = f["params"]
        haskey(p, spec.grid) || return nothing
        w  = Float64.(p[spec.grid])
        nw = length(w)
        val = d["value"]; err = get(d, "error", nothing)
        # Normalize to [outer][ω].
        if spec.outer === :none
            outer = Any[nothing]
            vv = Vector{Float64}[Float64.(val)]
            ee = Vector{Float64}[err === nothing ? fill(NaN, nw) : Float64.(err)]
        else
            nouter = length(val)
            if spec.outer === :delta && spec.dgrid !== nothing && haskey(p, spec.dgrid) &&
               length(p[spec.dgrid]) == nouter
                outer = Any[Float64(x) for x in p[spec.dgrid]]
            else
                spec.outer === :delta && @warn "spectrum: δ grid missing/mismatched, aligning by index" path o
                outer = Any[i for i in 1:nouter]
            end
            vv = Vector{Float64}[Float64.(val[i]) for i in 1:nouter]
            ee = Vector{Float64}[err === nothing ? fill(NaN, length(val[i])) : Float64.(err[i]) for i in 1:nouter]
        end
        return (w = w, outer = outer, vv = vv, ee = ee)
    end
end

"""
    get_spectrum(; obs, roots=MEANS_DB_ROOTS, strict=false, kw...)

A frequency-resolved observable, STITCHED across the separate files that measured
the same physical point on different ω windows, and disorder-averaged. Use this
instead of `get_observable` for anything on a `w_*` grid (`get_observable` would
try to average different-length ω arrays and fail or mislead).

    s = get_spectrum(; U=3.0, Lx=8, V=1.0, wanted_n=0.875, beta=50.0,
                       obs=:conductivity_regular_xx_fixed)
    plot(s.w, s.value[i])          # i-th δ; s.deltas[i] is its value

Known observables (`keys(FREQ_OBS)`): the `conductivity_regular_*_fixed[_extrap]`
family and `Lxx_w_fixed`… ([δ][ω]), `DOS` ([ω]), `LDOS` ([site][ω]).

Stitching: within one disorder realization, overlapping ω points from two windows
are independent MC estimates and are inverse-variance combined; disjoint points
are concatenated. Across disorder realizations each ω point is averaged over the
seeds that measured it (union — a partially-covered point is still returned; a
point no seed has is `NaN`).

Returns a NamedTuple with the ω axis EXPLICIT:
  `w`            union frequency grid (sorted)
  `deltas`       δ values (`:delta` obs) / `sites` (`:index`) / absent (`:none`)
  `value`        `[outer][ω]` matrix (or `[ω]` vector for `DOS`), `NaN` in gaps
  `error`        disorder-average error (scatter/√N, else the single MC error)
  `std`          seed-to-seed scatter
  `stat_error`   propagated Monte-Carlo error
  `n_samples`    disorder seeds behind each cell
  `disorder_seeds`, `variants`   as in `get_observable`

Like `get_curve`, differing physical params other than `disorder_seed` are
enumerated as `variants` (warn; `strict=true` errors).
"""
function get_spectrum(; obs, roots=MEANS_DB_ROOTS, n=nothing, n_tol::Float64=0.1,
                      mc_seed=nothing, strict::Bool=false, warn_varied::Bool=true,
                      verbose::Bool=false, kw...)
    o = string(obs)
    haskey(FREQ_OBS, o) || error("get_spectrum: `$o` is not frequency-resolved. " *
        "Known: $(join(sort(collect(keys(FREQ_OBS))), ", "))")
    spec = FREQ_OBS[o]
    hits = find_all(; roots=roots, n=n, n_tol=n_tol, kw...)
    if isempty(hits)
        verbose && @info "no simulations match" selectors=kw
        return nothing
    end

    # ω/δ grids are NOT index columns, so `variant_groups` sees only real physics
    # variants (alpha/beta_phi/…), never the windows we are about to stitch.
    variants = variant_groups(hits; ignore=("disorder_seed",))
    (warn_varied || strict) && _check_variants(hits, variants; label="get_spectrum", strict=strict)

    # One disorder realization per group; within it, only the ω/δ windows differ.
    seedgroups = Dict{Any,Vector{Dict{String,Any}}}()
    for r in hits
        push!(get!(seedgroups, get(r, "disorder_seed", missing), Dict{String,Any}[]), r)
    end

    # Per seed: (outer, ω) -> combined (value, mc-error). Overlapping windows are
    # inverse-variance combined; the union of windows defines this seed's spectrum.
    seed_specs = Dict{Any,Dict{Tuple{Any,Float64},Tuple{Float64,Float64}}}()
    clusters = String[]
    for (ds, rows) in seedgroups
        raw = Dict{Tuple{Any,Float64},Tuple{Vector{Float64},Vector{Float64}}}()
        for r in sort(rows, by = rr -> get(rr, "cluster", ""))
            fr = _read_spectrum_file(r["path"], o, spec, mc_seed)
            fr === nothing && continue
            push!(clusters, string(get(r, "cluster", "")))
            for (oi, oc) in enumerate(fr.outer), (wi, wv) in enumerate(fr.w)
                cell = (oc, wv)
                vv, ee = get!(raw, cell, (Float64[], Float64[]))
                push!(vv, fr.vv[oi][wi]); push!(ee, fr.ee[oi][wi])
            end
        end
        isempty(raw) && continue
        seed_specs[ds] = Dict(cell => _combine_estimates(ve[1], ve[2]) for (cell, ve) in raw)
    end
    isempty(seed_specs) && (verbose && @info "matched $(length(hits)) sim(s) but none carry `$o`"; return nothing)

    # Disorder-average each cell over the seeds that measured it (union coverage).
    dseeds = sort(collect(keys(seed_specs)), by = x -> (x === missing, x))
    cells = reduce(union!, (Set(keys(s)) for s in values(seed_specs)); init=Set{Tuple{Any,Float64}}())
    outers = sort(unique(c[1] for c in cells), by = x -> (x === nothing, x))
    ws     = sort(unique(c[2] for c in cells))
    oidx = Dict(o_ => i for (i, o_) in enumerate(outers))
    widx = Dict(w_ => j for (j, w_) in enumerate(ws))

    # NB: locals must not be named `error`/`std` — assigning either anywhere in
    # this function would shadow `Base.error` (used above) / `Statistics.std`
    # across the WHOLE body, so a matrix name maps to the NamedTuple field only.
    no, nw = length(outers), length(ws)
    valmat = [fill(NaN, nw) for _ in 1:no]
    errmat = [fill(NaN, nw) for _ in 1:no]
    stdmat = [fill(NaN, nw) for _ in 1:no]
    statmat = [fill(NaN, nw) for _ in 1:no]
    nsamp = [zeros(Int, nw) for _ in 1:no]
    for cell in cells
        i, j = oidx[cell[1]], widx[cell[2]]
        vs = Float64[]; es = Float64[]
        for ds in dseeds
            haskey(seed_specs[ds], cell) || continue
            v, e = seed_specs[ds][cell]
            push!(vs, v); push!(es, e)
        end
        N = length(vs)
        mu = sum(vs) / N
        valmat[i][j] = mu
        nsamp[i][j] = N
        sd = N > 1 ? sqrt(sum((v - mu)^2 for v in vs) / (N - 1)) : 0.0
        stdmat[i][j]  = sd
        statmat[i][j] = sqrt(sum(e^2 for e in es)) / N
        errmat[i][j]  = N > 1 ? sd / sqrt(N) : es[1]
    end

    # `:none` collapses the singleton outer axis so DOS returns plain [ω] vectors.
    if spec.outer === :none
        return (w = ws, value = valmat[1], error = errmat[1], std = stdmat[1],
                stat_error = statmat[1], n_samples = nsamp[1],
                disorder_seeds = dseeds, clusters = clusters, variants = variants, obs = o)
    end
    axisname = spec.outer === :delta ? :deltas : :sites
    base = (w = ws, value = valmat, error = errmat, std = stdmat,
            stat_error = statmat, n_samples = nsamp,
            disorder_seeds = dseeds, clusters = clusters, variants = variants, obs = o)
    return merge((; axisname => outers), base)   # single dynamic key: always valid
end

"""
    get_param(param; roots=MEANS_DB_ROOTS, kw...)

Read one config/identity parameter for the simulation(s) matching the selectors,
scalar OR array — e.g. `w_conductivity_range`, `w_list`, `dw_values`,
`delta_fixed_values`, `pairing_symmetry`.

The index carries only SCALAR params (see `_is_scalarish`), and `params` is
stored as ONE Dict blob rather than a group, so neither `find_all`/`get_curve`
nor a `haskey(f, "params/…")` path can reach an array grid. This loads the whole
`params` Dict off disk — the same place `load_entry` reads it — and pulls out
`param` by name.

    w = get_param(:w_conductivity_range; U=3.0, Lx=8, V=1.0, wanted_n=0.875, beta=50.0)

Params are part of a simulation's identity, so every disorder realization of the
same physical point carries the same value; the FIRST match is returned (arrays
are handed back as stored — convert with `Float64.(w)` if needed). Returns
`nothing` if nothing matches or no match carries `param`.

Special case: `param = :mu` is a converged OUTPUT, not a config param, so it is
not in `params`. Asking for it here delegates to `get_mu` and returns the
disorder-averaged μ scalar (use `get_mu` directly for the error / per-seed
breakdown). `show_keys=true` still dumps the real `params` keys.
"""
function get_param(param; roots=MEANS_DB_ROOTS, n=nothing, n_tol::Float64=0.1, show_keys::Bool=false, kw...)
    key = string(param)
    # μ lives on sim_data, never in `params`; hand it to get_mu (info -> sim_data).
    if key == "mu" && !show_keys
        r = get_mu(; roots=roots, n=n, n_tol=n_tol, kw...)
        return r === nothing ? nothing : r.value
    end
    hits = find_all(; roots=roots, n=n, n_tol=n_tol, kw...)
    isempty(hits) && return nothing
    for r in hits
        v = _open_jld2(r["path"]) do f
            if show_keys
                println("params keys in $(r["path"]): ", haskey(f, "params") ? sort(collect(keys(f["params"]))) : "(none)")
                return nothing
            end
            haskey(f, "params") || return nothing
            p = f["params"]
            haskey(p, key) ? p[key] : nothing
        end
        v === nothing || return v
    end
    return nothing
end

"""
    db_status(; roots=MEANS_DB_ROOTS)

One line per root: how many simulations, and how many at each status. Reads
only the indexes, so it is instant and safe to call while jobs are running.
"""
function db_status(; roots=MEANS_DB_ROOTS)
    isempty(roots) && println("no means_db set: set ENV[\"BACKUP_DRIVE\"], LANDAU_ROOT/CURIE_ROOT, " *
                              "LANDAU_MOUNT/CURIE_MOUNT or MEANS_DB_ROOTS before including means_db.jl")
    for root in roots
        if !isdir(root)
            @printf("%-50s (not mounted)\n", root); continue
        end
        idx = try load_index(; root=root) catch e
            @printf("%-50s unreadable index: %s\n", root, sprint(showerror, e)); continue
        end
        N = idx["n"]::Int
        parts = if N > 0 && "status" in idx["columns"]
            join(["$s=$(count(==(s), idx["status"]))" for s in sort(unique(idx["status"]))], "  ")
        else
            ""
        end
        @printf("%-50s %4d simulation(s)   %s\n", root, N, parts)
    end
    return nothing
end

"""
    db_summary(; root)

Print what is queryable, from the index alone (no data read).
"""
function db_summary(; root::String=MEANS_DB_ROOT)
    idx = load_index(; root=root)
    N = idx["n"]::Int
    if N == 0; println("means_db at $root is empty."); return nothing; end
    @printf("means_db: %d simulations at %s\n", N, root)
    @printf("  index built_at : %s\n", string(get(idx, "built_at", "?")))
    for c in idx["columns"]
        c == "relpath" && continue
        vals = unique(idx[c])
        isempty(vals) && continue
        sv = try sort(collect(vals)) catch; collect(vals) end
        length(sv) <= 12 ? @printf("  %-16s %s\n", c, string(sv)) :
            @printf("  %-16s %d distinct values in [%s, %s]\n", c, length(sv),
                    string(first(sv)), string(last(sv)))
    end
    return nothing
end
