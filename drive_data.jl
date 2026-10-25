# ============================================================================
# drive_data.jl — read the archived V6 + V7 data: slims, V6 sim_info, checks.
#
# After the data reduction the backup drive is the archive (docs/WORKFLOW.md §8),
# and each cluster keeps the same layout under /home/data/<user>/data:
#   <cluster folder>/MC_Hubbard_V7/means_db/...                  means (complete DBs)
#   <cluster folder>/MC_Hubbard_V7/data/<camp>/<unit>/seed_<s>/results_slim.jld2
#   <cluster folder>/MC_Hubbard_V6/data/<camp>/num_control/sim_info.jld2
#   <cluster folder>/MC_Hubbard_V6/data/<camp>/<unit>/seed_<s>/{results_slim,means_cache}.jld2
#   <drive>/_records/<cluster>/<chunk>/   meta.tsv, slim_problems.txt, sha256 lists (drive only)
# On the drive the cluster folder is <drive>/<cluster> (<drive> = the `Yaniv-Data`
# folder, <cluster> = `landau` or `curie`); over a cluster mount it is <mount>/data.
#
# WHERE TO READ, set before any include (e.g. in ~/.julia/config/startup.jl):
#   ENV["BACKUP_DRIVE"] = "<where the drive is>/Yaniv-Data"   # both clusters from the drive
#   ENV["CURIE_ROOT"]   = "<Curie's mount>/data"             # optional, per cluster: wins over the drive
#   include("drive_data.jl")                   # also loads means_db.jl (JLD2 only)
# Per cluster: LANDAU_ROOT / CURIE_ROOT, else <BACKUP_DRIVE>/<cluster>, else the
# cluster mount (<LANDAU_MOUNT>/data, <CURIE_MOUNT>/data); no built-in locations.
# See cluster_root in means_db.jl. The means_db queries (get_curve, ...) and the
# helpers below follow the same choice; `drive=` on a helper forces the drive.
# Nothing in means_db.jl / update_means writes on the drive (DriveReadOnly).
# Reading notes: docs/WORKFLOW.md §8.
#
# V7
#   rows  = find_all(; U=4.0, Lx=20, wanted_n=0.875, beta=10.0)   # means_db rows
#   slims = find_slims(; U=4.0, Lx=20, wanted_n=0.875, beta=10.0) # their seeds' slims
#   s     = read_slim(slims[1].slim; obs=[:n, :stiffness])
#   s.means["stiffness"]["value"], s.binned["n"], s.meta["bin_size"]
# V6
#   v6_campaigns()                                    # what is there
#   si = load_v6_sim_info("curie", "Disorder__05-04-2026")   # sim_info Dict
#   v6_seeds("landau", "TCA_test__26-04-2026")        # seed folders with a slim
# Checks of the DRIVE (metadata only, no data read; never over a cluster mount)
#   drive_summary()       # GB / files / slims per cluster, version, campaign
#   drive_check()         # every archived raw seed vs. its slim on the drive
#
# Everything here only reads. Base + JLD2 + Printf; no MKL/Plots.
# ============================================================================
using JLD2, Printf
isdefined(Main, :MEANS_DB_SCHEMA_VERSION) || include(joinpath(@__DIR__, "means_db.jl"))

"The two clusters on the drive, in the order the code uses everywhere (Landau first)."
const DRIVE_CLUSTERS = ("landau", "curie")

"""
    drive_root() -> String

The backup drive's `Yaniv-Data` folder from env `BACKUP_DRIVE`. Errors with a hint if it
is not set or not there (drive not connected).
"""
function drive_root()
    d = _backup_drive()
    isempty(d) && error("drive_data.jl: set ENV[\"BACKUP_DRIVE\"] to the drive's Yaniv-Data " *
                        "folder before including the code.")
    isdir(d) || error("drive_data.jl: BACKUP_DRIVE=$d does not exist (is the drive connected?)")
    return d
end

"""
    _check_cluster(c) -> String

`c` lower-cased, if it is one of `DRIVE_CLUSTERS`; errors otherwise.
"""
function _check_cluster(c)
    s = lowercase(string(c))
    s in DRIVE_CLUSTERS || error("unknown cluster '$c' (expected one of $(join(DRIVE_CLUSTERS, ", ")))")
    return s
end

"""
    _croot(cluster, drive) -> String

The folder holding `cluster`'s `MC_Hubbard_V6/` and `MC_Hubbard_V7/`: `<drive>/<cluster>`
when `drive` is given, else `cluster_root(cluster)` (means_db.jl: `LANDAU_ROOT` /
`CURIE_ROOT`, else the backup drive, else the cluster mount).
"""
function _croot(cluster, drive::Union{Nothing,AbstractString})
    c = _check_cluster(cluster)
    drive === nothing || return joinpath(drive, c)
    r = cluster_root(c)
    isempty(r) && error("no folder set for $c: set ENV[\"BACKUP_DRIVE\"], $(uppercase(c))_ROOT or " *
                        "$(uppercase(c))_MOUNT before including the code (docs/WORKFLOW.md §8.2)")
    return r
end

"""
    drive_data_dir(cluster, version="V7"; drive=nothing) -> String

`<cluster folder>/MC_Hubbard_<version>/data`, the campaigns of one cluster and version.
The cluster folder is `<drive>/<cluster>` if `drive` is given, else `cluster_root(cluster)`.
"""
drive_data_dir(cluster, version::AbstractString="V7"; drive::Union{Nothing,AbstractString}=nothing) =
    joinpath(_croot(cluster, drive), "MC_Hubbard_" * uppercase(version), "data")

"""
    _jld_read(f, path)

`jldopen(f, path, "r")`, falling back from the default mmap to IOStream if mmap fails
(a JLD2 bug on some large nested files, see data_summery.jl).
"""
function _jld_read(f::Function, path::AbstractString)
    try
        return jldopen(f, String(path), "r")
    catch e
        e isa ReadOnlyMemoryError || rethrow(e)
        return jldopen(f, String(path), "r"; iotype=IOStream)
    end
end

# ---------------------------------------------------------------------------
# V7: means_db rows -> seeds -> slims
# ---------------------------------------------------------------------------

"""
    slim_path(dir_path; cluster, drive=nothing) -> String

The `results_slim.jld2` on this machine for a seed folder given as a cluster path
(`/home/data/<user>/data/MC_Hubbard_V7/data/<camp>/<unit>/seed_<s>`, as stored in
means_db `seeds/<s>/info["dir_path"]`), a mount path, or any path containing
`MC_Hubbard_Vx/data/...`. V6 and V7 alike. It is looked up under `<drive>/<cluster>`
if `drive` is given, else under `cluster_root(cluster)` (the drive or the cluster
mount, see means_db.jl). `cluster` is required: both clusters use the same path
strings for different runs. Does not check that the file exists.
"""
function slim_path(dir_path::AbstractString; cluster, drive::Union{Nothing,AbstractString}=nothing)
    p = rstrip(replace(String(dir_path), '\\' => '/'), '/')
    endswith(p, "/results.jld2") && (p = p[1:end-length("/results.jld2")])
    m = match(r"(MC_Hubbard_V\d+/data/.+)$", p)
    m === nothing && error("slim_path: no MC_Hubbard_Vx/data/ in $dir_path")
    return normpath(joinpath(_croot(cluster, drive), split(m.captures[1], '/')...,
                             "results_slim.jld2"))
end

"""
    _row_cluster(row) -> String

The cluster of a `find_all` row: taken from its root (`<drive>/<cluster>/MC_Hubbard_V7/
means_db`) when that is a drive root, else from the row's `cluster` column.
"""
function _row_cluster(row::AbstractDict)
    root = replace(string(get(row, "root", "")), '\\' => '/')
    parts = split(rstrip(root, '/'), '/')
    if length(parts) >= 3 && lowercase(parts[end]) == "means_db"
        c = lowercase(parts[end-2])
        c in DRIVE_CLUSTERS && return c
    end
    return _check_cluster(get(row, "cluster", ""))
end

"""
    seed_slims(row; drive=nothing) -> Vector{NamedTuple}

The seeds of one means_db row (from `find_all`) and their slims on the drive:
`(cluster, seed, dir_path, slim, exists)`. Reads only the row's seed `info` blocks.
"""
function seed_slims(row::AbstractDict; drive::Union{Nothing,AbstractString}=nothing)
    c = _row_cluster(row)
    out = NamedTuple[]
    for (s, d) in sort(collect(read_seeds(row["path"]; with_means=false)); by=first)
        dp = string(get(get(d, "info", Dict{String,Any}()), "dir_path", ""))
        isempty(dp) && continue
        sp = slim_path(dp; cluster=c, drive=drive)
        push!(out, (cluster=c, seed=string(s), dir_path=dp, slim=sp, exists=isfile(sp)))
    end
    return out
end

"""
    find_slims(; roots=MEANS_DB_ROOTS, drive=nothing, kw...) -> Vector{NamedTuple}

`find_all(; kw...)` (same selectors) followed by `seed_slims` for every row: the slims of
all seeds of all matching simulations. Each item also carries the means_db file
(`db_path`). Seeds whose slim is not on the drive (the few held-back seeds) come back
with `exists=false`.
"""
function find_slims(; roots=MEANS_DB_ROOTS, drive::Union{Nothing,AbstractString}=nothing, kw...)
    out = NamedTuple[]
    for row in find_all(; roots=roots, kw...)
        for s in seed_slims(row; drive=drive)
            push!(out, merge(s, (db_path=row["path"],)))
        end
    end
    return out
end

"""
    read_slim(path; obs=nothing) -> NamedTuple

Read one `results_slim.jld2` (V6 or V7; layout in slim_results.jl and WORKFLOW.md §8.3):
`(path, format, sim_data, meta, means, binned, meanobs)`, each of the last four a Dict
keyed by name. `obs` (a name, Symbol, or vector of them) limits means/binned/meanobs to
those observables; `nothing` reads all. Error bars and nonlinear observables: use
`means` (exact); `binned` is the stream averaged into `meta["num_bins"]` bins.
"""
function read_slim(path::AbstractString; obs=nothing)
    want = obs === nothing ? nothing :
           Set(string.(obs isa Union{AbstractVector,Tuple} ? collect(obs) : [obs]))
    _jld_read(path) do f
        grp(g) = haskey(f, g) ?
            Dict{String,Any}(string(k) => f["$g/$k"] for k in keys(f[g])
                             if want === nothing || string(k) in want) :
            Dict{String,Any}()
        meta = haskey(f, "meta") ?
            Dict{String,Any}(string(k) => f["meta/$k"] for k in keys(f["meta"])) : Dict{String,Any}()
        (path     = String(path),
         format   = haskey(f, "slim_format") ? f["slim_format"] : "",
         sim_data = haskey(f, "sim_data") ? f["sim_data"] : nothing,
         meta     = meta,
         means    = grp("means"),
         binned   = grp("binned"),
         meanobs  = grp("meanobs"))
    end
end

"""
    slim_obs(path) -> NamedTuple

The observable names in a slim, without reading any data: `(means, binned, meanobs)`.
"""
function slim_obs(path::AbstractString)
    _jld_read(path) do f
        ks(g) = haskey(f, g) ? sort(string.(collect(keys(f[g])))) : String[]
        (means = ks("means"), binned = ks("binned"), meanobs = ks("meanobs"))
    end
end

# ---------------------------------------------------------------------------
# V6: campaigns, sim_info, seed folders
# ---------------------------------------------------------------------------

"""
    v6_campaigns(; cluster=nothing, drive=nothing) -> Vector{NamedTuple}

The V6 campaigns under each cluster's folder (the drive, or the cluster: see
`cluster_root`): `(cluster, campaign, dir, sim_info)` where `sim_info` is
the path of `num_control/sim_info.jld2` or `""`. Lists two folder levels only.
The thesis keep-list campaigns have slims; Landau also holds `sim_info.jld2` +
`sim.jld2` (no slims) of a few unused campaigns that were deleted on the cluster.
"""
function v6_campaigns(; cluster=nothing, drive::Union{Nothing,AbstractString}=nothing)
    out = NamedTuple[]
    for c in (cluster === nothing ? DRIVE_CLUSTERS : (_check_cluster(cluster),))
        d = drive_data_dir(c, "V6"; drive=drive)
        isdir(d) || continue
        for camp in sort(readdir(d))
            cdir = joinpath(d, camp)
            isdir(cdir) || continue
            si = joinpath(cdir, "num_control", "sim_info.jld2")
            push!(out, (cluster=c, campaign=camp, dir=cdir, sim_info=isfile(si) ? si : ""))
        end
    end
    return out
end

"""
    v6_sim_info_path(cluster, campaign; drive=nothing) -> String

`<drive>/<cluster>/MC_Hubbard_V6/data/<campaign>/num_control/sim_info.jld2` (errors if absent).
"""
function v6_sim_info_path(cluster, campaign::AbstractString; drive::Union{Nothing,AbstractString}=nothing)
    p = joinpath(drive_data_dir(cluster, "V6"; drive=drive), campaign, "num_control", "sim_info.jld2")
    isfile(p) || error("no V6 sim_info.jld2 for $cluster/$campaign on the drive: $p")
    return p
end

"""
    load_v6_sim_info(cluster, campaign; drive=nothing) -> Dict{String,Any}

The campaign's V6 means file (`sim_num => Dict(params..., "means", "simulations")`), as
`load_sim_info` returns it, read straight from the drive. Feed it to the V6 helpers
(`get_sims`, `get_obs_vs_T`, data_summery.jl plots).

Caveat (checked 2026-10-25): these are the cluster's LAST versions. For the
Clean_System campaigns they are byte-identical to the thesis copies in
`Habburd_MC_V6/cluster_results`, but the disorder ones (Disorder*, DisorderII,
Disorder_U) were rewritten on the cluster on 05-17/05-26, after those copies were
taken. Thesis figures keep using the local copies / means_store.jl.
"""
function load_v6_sim_info(cluster, campaign::AbstractString; drive::Union{Nothing,AbstractString}=nothing)
    p = v6_sim_info_path(cluster, campaign; drive=drive)
    _jld_read(p) do f
        Dict{String,Any}(string(k) => f[k] for k in keys(f))
    end
end

"""
    v6_seeds(cluster, campaign; drive=nothing) -> Vector{NamedTuple}

The seed folders of a V6 campaign that have a slim: `(unit, seed, dir, slim,
means_cache)` (`means_cache` = path or `""`). Units are sorted numerically. Lists
folders only.
"""
function v6_seeds(cluster, campaign::AbstractString; drive::Union{Nothing,AbstractString}=nothing)
    cdir = joinpath(drive_data_dir(cluster, "V6"; drive=drive), campaign)
    isdir(cdir) || error("no V6 campaign $cluster/$campaign on the drive: $cdir")
    units = filter(u -> isdir(joinpath(cdir, u)) && u != "num_control", readdir(cdir))
    sort!(units; by = u -> (something(tryparse(Int, u), typemax(Int)), u))
    out = NamedTuple[]
    for u in units, s in sort(filter(x -> startswith(x, "seed_"), readdir(joinpath(cdir, u))))
        sd = joinpath(cdir, u, s)
        slim = joinpath(sd, "results_slim.jld2")
        isfile(slim) || continue
        mc = joinpath(sd, "means_cache.jld2")
        push!(out, (unit=u, seed=s, dir=sd, slim=slim, means_cache=isfile(mc) ? mc : ""))
    end
    return out
end

# ---------------------------------------------------------------------------
# Checks (metadata only)
# ---------------------------------------------------------------------------

"""
    drive_summary(; drive=drive_root(), io=stdout) -> Vector{NamedTuple}

One row per cluster / version / folder (campaign, or means_db): `(cluster, version,
kind, name, gb, files, slims)`. Walks the folders and stats every file (~70k files)
but reads no data; expect a few minutes on the USB disk. Printed as a table.
"""
function drive_summary(; drive::AbstractString=drive_root(), io::IO=stdout)
    rows = NamedTuple[]
    for c in DRIVE_CLUSTERS, v in ("V7", "V6")
        base = joinpath(drive, c, "MC_Hubbard_$v")
        isdir(base) || continue
        for kind in ("data", "means_db")
            kd = joinpath(base, kind)
            isdir(kd) || continue
            names = kind == "data" ? sort(filter(n -> isdir(joinpath(kd, n)), readdir(kd))) : ["(all)"]
            for n in names
                top = kind == "data" ? joinpath(kd, n) : kd
                nb = 0; nf = 0; ns = 0
                for (dir, _, fs) in walkdir(top), fn in fs
                    nf += 1; nb += filesize(joinpath(dir, fn)); ns += (fn == "results_slim.jld2")
                end
                push!(rows, (cluster=c, version=v, kind=kind, name=n, gb=nb / 2^30, files=nf, slims=ns))
            end
        end
    end
    @printf(io, "%-7s %-3s %-8s %-34s %8s %7s %6s\n", "cluster", "ver", "kind", "name", "GB", "files", "slims")
    for r in rows
        @printf(io, "%-7s %-3s %-8s %-34s %8.2f %7d %6d\n", r.cluster, r.version, r.kind, r.name, r.gb, r.files, r.slims)
    end
    @printf(io, "total %.1f GB, %d files, %d slims\n", sum(r.gb for r in rows; init=0.0),
            sum(r.files for r in rows; init=0), sum(r.slims for r in rows; init=0))
    return rows
end

"""
    drive_check(; drive=drive_root(), io=stdout) -> Vector{NamedTuple}

Every raw seed the data reduction archived (rows `R` of `_records/<cluster>/<chunk>/
meta.tsv`) against its `results_slim.jld2` on the drive. Prints one line per
campaign (seeds, slims on drive, held back) and returns the seeds WITHOUT a slim:
`(cluster, seed_dir, chunk, reason)`, the reason taken from the chunk's
`slim_problems.txt`. Their raw `results.jld2` stayed on the cluster. Only file
existence is checked (no reads, no sha256; ~20 s). Curie's V6 slims are not in it:
they were made on the cluster in Aug 2026 and reached the drive in the backup chunk
(`curie__V6_all__bk__c01/drive.sha256list`), which has no raw seeds.
"""
function drive_check(; drive::AbstractString=drive_root(), io::IO=stdout)
    held = NamedTuple[]
    for c in DRIVE_CLUSTERS
        rec = joinpath(drive, "_records", c)
        isdir(rec) || continue
        seeds = Dict{String,String}(); reasons = Dict{String,String}()
        for ch in sort(readdir(rec))
            occursin(r"__(means_db|V6_all__bk)__c\d+$", ch) && continue      # not raw-seed chunks
            meta = joinpath(rec, ch, "meta.tsv")
            isfile(meta) || continue
            for l in eachline(meta)
                f = split(l, '\t')
                (length(f) >= 4 && f[1] == "R") && (seeds[dirname(f[4])] = ch)
            end
            pr = joinpath(rec, ch, "slim_problems.txt")
            isfile(pr) || continue
            for l in eachline(pr)
                f = split(l, '\t')
                m = match(r"(MC_Hubbard_V\d+/data/.+)/results\.jld2$", f[1])
                (m === nothing || length(f) < 2) && continue
                reasons[m.captures[1]] = join(f[2:end], ": ")
            end
        end
        bycamp = Dict{String,Vector{Int}}()
        for (sd, ch) in seeds
            parts = split(sd, '/')                     # MC_Hubbard_Vx / data / camp / unit / seed
            key = replace(parts[1], "MC_Hubbard_" => "") * " " * parts[3]
            ok = isfile(joinpath(drive, c, parts..., "results_slim.jld2"))
            cnt = get!(bycamp, key, [0, 0]); cnt[1] += 1; cnt[2] += ok
            ok || push!(held, (cluster=c, seed_dir=sd, chunk=ch, reason=get(reasons, sd, "")))
        end
        for k in sort(collect(keys(bycamp)))
            n, s = bycamp[k]
            @printf(io, "%-7s %-38s seeds=%5d  slim_on_drive=%5d  held_back=%d\n", c, k, n, s, n - s)
        end
    end
    @printf(io, "%d seed(s) without a slim on the drive\n", length(held))
    for h in held
        @printf(io, "  %-7s %s  (%s; %s)\n", h.cluster, h.seed_dir, h.chunk, h.reason)
    end
    return held
end
