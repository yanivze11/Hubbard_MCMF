# ============================================================================
# slim_results.jl — shrink one seed's raw MC stream into a small, per-observable
# "slim" file, WITHOUT recomputing any physics.
#
# WHY / DESIGN
#   A V6 results.jld2 is `sim_data` + many `measurement_<k>` chunks; each chunk
#   is a Dict of ~43 observables. Streamed observables hold one value per MC
#   measurement (concatenated across chunks == the full stream). BZ/A/LDOS are
#   stored once PER CHUNK (duplicated ~#chunks times).
#
#   The exact, thesis-time means already live next to it in means_cache.jld2
#   (a `means` Dict of ~63 observables, each {value,error,correlation_time,...}
#   computed by V6 code on the FULL stream). We REUSE those verbatim — so the
#   physics numbers are bit-for-bit the thesis values, with zero V6->V7 drift.
#
#   Binning the stream and THEN recomputing would corrupt every nonlinear
#   observable (S_Delta_Q0=<|Δ_mean|²>, Ps via DdD–n covariance, U4, and all
#   autocorrelation error bars). So: means come from the cache; the binned
#   stream is stored only as coarse, additive optionality for future re-analysis.
#
# OUTPUT: results_slim.jld2, ONE JLD2 dataset per observable (loadable in
# isolation, like the new means_db files):
#   slim_format            "slim_v1"
#   schema_version         Int
#   created_at             DateTime
#   source_results         Dict(basename, orig_bytes, num_chunks)
#   sim_data               Dict (verbatim)
#   meta/num_measurements  Int      full-stream length N
#   meta/num_bins          Int      <= requested max bins (adaptive, see below)
#   meta/bin_size          Int      measurements per bin (equal; tail dropped)
#   meta/dropped_remainder Int      N - bin_size*num_bins  (< num_bins)
#   meta/chunk_sizes       Vector{Int}
#   meta/means_source      "means_cache" | "absent"
#   meta/tau_max           Float64  worst autocorr time over all cached observables
#   meta/tau_factor        Float64  bin_size targeted >= tau_factor*tau_max
#   meta/bins_decorrelated Bool     bin_size >= 2*tau_max  (error re-derivable)
#
# ERROR BARS: the stored means/<obs>/error IS the exact cache error (block std
# at ~10τ on the FULL stream), so error bars are preserved exactly regardless of
# binning. BINS: DEFAULT is a fixed `--bins` (100) — the finest grid we keep, so
# a later re-blocking / autocorrelation analysis has the most to work with. Bins
# are uniform across observables (covariances, hence derived-observable errors
# like Ps, survive). tau_max is recorded as a diagnostic (bins_decorrelated flags
# whether bin_size >= 2*tau_max); it does NOT reduce the bin count unless you opt
# in with --tau-factor>0. (Complex observables store complex bin-means; their
# |·|-based error is only approximately re-derivable — exact value in means/.)
#   means/<obs>            Dict{value,error,...}   (copied from means_cache / means_db)
#   binned/<obs>           Vector length num_bins  (block-averaged stream)
#   meanobs/BZ|A|LDOS      weighted (by chunk size) mean array
#
# HEAVY OBSERVABLES: an observable with more than --heavy-elems numbers per
# measurement (DOS, LDOS, Lxx_w_fixed, ... at production L) keeps only
# --heavy-bins bins. Each heavy bin is exactly k consecutive fine bins, so the
# two grids nest and cross-covariances stay computable at the coarse level.
#   meta/heavy_obs, meta/heavy_bins, meta/heavy_bin_size
#
# V7 (no means_cache): the means come from the seed's record in means_db, found
# through the `dir_path` that record stores (see --build-dbmap / --dbmap). They
# are copied VERBATIM, and a slim is written only if that record was computed
# from exactly the stream in this file (same measurement count). Otherwise the
# seed is SKIPPED as db_stale: run num_control for the campaign, then re-slim.
#   meta/means_source = "means_db", meta/db_path, meta/db_seed, meta/db_info
# --accept-stale (decided 2026-10-21: every stream is final and the means need not
# match the binned data exactly): a db_stale seed takes the means of its MOST
# COMPLETE record instead (largest N), verbatim, and is flagged
#   meta/db_stale = true, meta/db_num_measurements = that record's N.
#
# NON-DESTRUCTIVE: writes results_slim.jld2 alongside the original; NEVER
# deletes/renames results.jld2. Replacement is a separate, gated step.
#
#   julia -O1 slim_results.jl <seed_dir | results.jld2> [options]
#     --bins=100            number of bins (default 100; always kept unless --tau-factor)
#     --heavy-bins=30       at least this many bins for heavy observables; each is a
#                           whole number of fine bins, so 30 of 100 gives 33 bins
#                           of 3 (--heavy-bins=<bins> disables)
#     --heavy-elems=64      numbers per measurement above which an obs is heavy
#     --tau-factor=0        >0 caps bins so bin_size>=factor*max_τ (default 0 = off)
#     --dbmap=<tsv>         V7: dir -> means_db record map (from --build-dbmap)
#     --out=<path>          slim output path (default <dir>/results_slim.jld2)
#     --force               overwrite an existing slim file
#     --allow-missing-means write binned-only if no means are found
#                           (default: SKIP such a seed and report it)
#     --accept-stale        V7: take a db_stale seed's most complete means_db record
#     --verify              reload slim and check linear-mean consistency
#     --quiet
#
#   julia -O1 slim_results.jl --build-dbmap=<means_db root> --out=<tsv>
# ============================================================================
using JLD2, Statistics, Dates, Printf

const MEAN_OBS      = ("BZ", "A", "LDOS")            # per-chunk means, not streams
const DRUDE_ALIASES = (("drude_dc_xx","L_D_dc_xx"), ("drude_dc_xy","L_D_dc_xy"),
                       ("drude_dc_yx","L_D_dc_yx"), ("drude_dc_yy","L_D_dc_yy"))
const SLIM_FORMAT    = "slim_v1"
const SLIM_SCHEMA    = 1

# ---- generic recursive mean over a block of like-shaped values --------------
# `+` on same-shaped nested arrays already recurses elementwise at any depth;
# _rscale then divides through recursively. Handles scalars, Vector{Float64},
# Vector{Vector{ComplexF64}}, Vector{Vector{Vector{...}}} (Lxx_w_fixed), etc.
"""
    _rscale(x, s)

Multiply a number, or every number inside nested arrays, by `s`.
"""
_rscale(x::Number, s)        = x * s
_rscale(x::AbstractArray, s) = _rscale.(x, s)

"""
    block_mean(block)

Mean of a block of like-shaped values (scalars or nested arrays, element-wise).
"""
function block_mean(block)
    n = length(block)
    n == 0 && error("block_mean: empty block")
    return _rscale(reduce(+, block), 1 / n)
end

"""
    _is_empty_ss(ss) -> Bool

True if a chunk's action stream is empty or all zero (an unfilled buffer).
"""
_is_empty_ss(ss) = isempty(ss) || all(iszero, ss)

# ---- pick a bin count that keeps bins ~decorrelated AND uniform -------------
# The cache error uses blocks of ~10τ. Re-deriving the error from our bins only
# reproduces it when each bin is longer than the autocorrelation length. So we
# target `max_bins` (≈100) but force bin_size >= tau_factor*max_τ, taking max_τ
# over ALL observables so the SINGLE bin grid stays uniform (uniform binning is
# required to preserve cross-observable covariances, e.g. Ps = 2⟨DdD⟩-⟨n⟩+1).
"""
    _flat_max(x) -> Float64

Largest finite number in `x` (a number or nested arrays); 0 for none, `nothing`, or
non-numeric entries (some means_cache correlation times are `nothing`).
"""
_flat_max(x::Number)        = isfinite(x) ? Float64(x) : 0.0
_flat_max(x::AbstractArray) = isempty(x) ? 0.0 : maximum(_flat_max, x)
# Some means_cache correlation_time entries hold `nothing` (a component whose τ
# couldn't be computed) or other non-numeric junk; treat those as 0 (ignore).
_flat_max(::Nothing)        = 0.0
_flat_max(::Any)            = 0.0

"""
    cache_max_tau(means) -> Float64

The largest correlation time over all observables of a means Dict (NaN if `nothing`).
"""
function cache_max_tau(means)
    means === nothing && return NaN
    mt = 0.0
    for (_, d) in means
        (d isa AbstractDict && haskey(d, "correlation_time")) || continue
        mt = max(mt, _flat_max(d["correlation_time"]))
    end
    return mt
end

"""
    decide_bins(N, max_tau; tau_factor=0.0, min_bins=20, max_bins=100) -> (num_bins, bin_size)

Number and size of the bins for a stream of N measurements: `max_bins` by default. With
`tau_factor > 0` the count is capped so each bin is at least `tau_factor`·τ_max long
(but never below `min_bins`).
"""
function decide_bins(N::Int, max_tau; tau_factor::Float64=0.0,
                     min_bins::Int=20, max_bins::Int=100)
    nb = min(max_bins, N)                 # DEFAULT: always keep up to max_bins (=100) bins.
    # Finer bins preserve more of the stream for a later re-blocking / autocorr
    # analysis; the exact error already lives in means/<obs>/error. τ is only a
    # recorded diagnostic here. Optional (tau_factor>0): cap bins so each is
    # >= tau_factor*max_τ (decorrelated), never below min_bins.
    if tau_factor > 0 && max_tau isa Number && isfinite(max_tau) && max_tau > 0
        nb_dec = max(min_bins, fld(N, max(1, ceil(Int, tau_factor * max_tau))))
        nb = min(nb, nb_dec)
    end
    nb = clamp(nb, 1, N)
    return nb, fld(N, nb)
end

# ---- bin a full stream into equal contiguous bins (tail remainder dropped) --
"""
    bin_fixed(stream, bs, nb) -> Vector

The means of `nb` consecutive bins of `bs` measurements each (the tail is dropped).
"""
function bin_fixed(stream::AbstractVector, bs::Int, nb::Int)
    binned = Vector{Any}(undef, nb)
    for b in 1:nb
        binned[b] = block_mean(view(stream, (b-1)*bs+1 : b*bs))
    end
    return identity.(binned)              # identity.() narrows the eltype
end

"""
    bin_stream(stream, nbins) -> (binned, bin_size, dropped)

Cut a stream into `nbins` equal bins (fewer if it is shorter) and return the bin means,
the bin size and the number of dropped tail measurements.
"""
function bin_stream(stream::AbstractVector, nbins::Int)
    N  = length(stream)
    nb = min(nbins, N)
    bs = fld(N, nb)                       # equal bin size; drop < nb tail samples
    return bin_fixed(stream, bs, nb), bs, N - bs*nb
end

"""
    _nelem(x) -> Int

How many numbers one measurement of an observable holds (a complex number counts as one).
"""
_nelem(x::Number)        = 1
_nelem(x::AbstractArray) = isempty(x) ? 0 : sum(_nelem, x)
_nelem(::Any)            = 1

# ---- V7: find a seed's means in means_db --------------------------------------
# Each means_db file stores, per seed, the `dir_path` the run wrote to. Key both
# sides on the path from `MC_Hubbard_Vx/data/` on, so a record written with a
# cluster path and a slim run on a mount path (or vice versa) still meet.
"""
    _dirkey(p) -> String

The part of a seed path from `MC_Hubbard_Vx/data/` on, with `/` separators, so a
cluster path and a mount path of the same folder give the same key.
"""
function _dirkey(p::AbstractString)
    q = rstrip(replace(String(p), '\\' => '/'), '/')
    m = match(r"(MC_Hubbard_V\d+/data/.+)$", q)
    return m === nothing ? q : String(m.captures[1])
end

const DBMAP_HEADER = "key\tdb_path\tseed\tnum_meas\tstatus"

"""
    build_dbmap(db_root, out) -> out

One pass over a means_db root: write a TSV with one line per seed record that has means
(dir key, means_db file, seed, measurement count, status). Unreadable files are skipped.
"""
function build_dbmap(db_root::AbstractString, out::AbstractString)
    files = String[]
    for (dir, _, fs) in walkdir(db_root), fn in fs
        endswith(fn, ".jld2") && fn != "index.jld2" && push!(files, joinpath(dir, fn))
    end
    sort!(files)
    rows = 0; bad = 0; t0 = time()
    tmp = out * ".tmp"
    open(tmp, "w") do io
        println(io, DBMAP_HEADER)
        for (i, p) in enumerate(files)
            try
                jldopen(p, "r") do f
                    haskey(f, "seeds") || return
                    for s in keys(f["seeds"])
                        haskey(f, "seeds/$s/info") && haskey(f, "seeds/$s/means") || continue
                        info = f["seeds/$s/info"]
                        dp = get(info, "dir_path", nothing)
                        dp isa AbstractString || continue
                        println(io, join((_dirkey(dp), p, s,
                                          get(info, "num_current_measurments", -1),
                                          get(info, "status", "")), '\t'))
                        rows += 1
                    end
                end
            catch e
                bad += 1
                @warn "unreadable means_db file (skipped)" p exception=e
            end
            i % 500 == 0 && (@printf("  %d/%d files ...\n", i, length(files)); flush(stdout))
        end
    end
    mv(tmp, out; force=true)
    @printf("[dbmap] %d seed records from %d files (%d unreadable) in %.0fs -> %s\n",
            rows, length(files), bad, time() - t0, out)
    return out
end

"""
    load_dbmap(path) -> Dict{String, Vector{NamedTuple}}

Read a TSV written by `build_dbmap`: dir key => its means_db records
`(db_path, seed, num_meas, status)`.
"""
function load_dbmap(path::AbstractString)
    m = Dict{String,Vector{NamedTuple}}()
    for (i, l) in enumerate(eachline(path))
        (i == 1 || isempty(l)) && continue
        c = split(l, '\t')
        length(c) >= 5 || continue
        push!(get!(m, String(c[1]), NamedTuple[]),
              (db_path=String(c[2]), seed=String(c[3]),
               num_meas=something(tryparse(Int, c[4]), -1), status=String(c[5])))
    end
    return m
end

"""
    _read_group(f, g) -> Dict{String,Any}

Every entry of JLD2 group `g` in the open file `f`.
"""
function _read_group(f, g::String)
    d = Dict{String,Any}()
    for k in keys(f[g]); d[string(k)] = f["$g/$k"]; end
    return d
end

"""
    pick_db_record(hits, N; accept_stale=false) -> (status, record_or_nothing, message)

The single means_db record whose measurement count equals the stream length `N`
(`:ok`). Otherwise `:no_db`, `:db_ambiguous` or `:db_stale`; with `accept_stale`, a seed
with no exact record takes its unique most complete one (`:stale`).
"""
function pick_db_record(hits, N::Int; accept_stale::Bool=false)
    isempty(hits) && return (:no_db, nothing, "no means_db record for this dir")
    exact = filter(h -> h.num_meas == N, hits)
    length(exact) == 1 && return (:ok, exact[1], "")
    length(exact) > 1 && return (:db_ambiguous, nothing,
        "$(length(exact)) means_db records with N=$N: " * join((h.db_path for h in exact), ", "))
    have = join((string(h.num_meas) for h in hits), ",")
    if accept_stale
        best = maximum(h.num_meas for h in hits)
        top  = filter(h -> h.num_meas == best, hits)
        (best > 0 && length(top) == 1) && return (:stale, top[1], "stream N=$N, means from the record with N=$best")
        return (:db_ambiguous, nothing, "stale, and no single most complete record (N=$have)")
    end
    return (:db_stale, nothing,
        "stream N=$N but means_db has N=$have (run num_control for this campaign first)")
end

# ---- read all chunks and reproduce load_results' combination logic ----------
"""
    load_and_combine(results_path) -> (sim_data, streams, meanobs, chunk_sizes, N, dropped)

Read every non-empty chunk of a results.jld2 and join them as `load_results` does:
streams concatenated, BZ/A/LDOS averaged weighted by chunk size, old `drude_dc_*` keys
renamed. Observables missing from some chunks are listed in `dropped`.
"""
function load_and_combine(results_path::String)
    chunks      = Dict{String,Any}[]
    chunk_sizes = Int[]
    sim_data    = nothing
    jldopen(results_path, "r") do f
        haskey(f, "sim_data") && (sim_data = f["sim_data"])
        mkeys = filter(k -> startswith(k, "measurement_"), collect(keys(f)))
        sort!(mkeys, by = k -> parse(Int, split(k, "_")[2]))
        for k in mkeys
            raw = f[k]
            raw isa AbstractDict || continue
            chunk = Dict{String,Any}(string(kk) => vv for (kk, vv) in raw)
            for (o, n) in DRUDE_ALIASES
                if haskey(chunk, o) && !haskey(chunk, n)
                    chunk[n] = chunk[o]; delete!(chunk, o)
                end
            end
            (haskey(chunk, "Ss") && !_is_empty_ss(chunk["Ss"])) || continue
            push!(chunks, chunk)
            push!(chunk_sizes, length(chunk["Ss"]))
        end
    end
    isempty(chunks) && error("no non-empty measurement chunks in $results_path")

    shared  = reduce(intersect, (Set(keys(c)) for c in chunks))
    allobs  = reduce(union,     (Set(keys(c)) for c in chunks))
    delete!(shared, "type")
    tot     = sum(chunk_sizes)
    streams = Dict{String,Any}()
    meanobs = Dict{String,Any}()
    for obs in sort(collect(shared))
        if obs in MEAN_OBS
            meanobs[obs] = sum(chunks[i][obs] * chunk_sizes[i] for i in eachindex(chunks)) / tot
        else
            streams[obs] = reduce(vcat, (c[obs] for c in chunks))
        end
    end
    dropped = sort(collect(setdiff(allobs, shared)))
    return (; sim_data, streams, meanobs, chunk_sizes, N = tot, dropped)
end

# ---- write the slim file (one dataset per observable) -----------------------
"""
    write_slim(slim_path, R, means, means_source, orig_size, results_path, nbins; max_tau=NaN,
               tau_factor=NaN, heavy_bins=nbins, heavy_elems=64, dbmeta=nothing) -> NamedTuple

Write results_slim.jld2 (layout in the file header): sim_data, metadata, the binned
streams (heavy observables on the coarser nested grid), the mean observables and the
copied means. Written to a .tmp file and renamed. Returns bin counts and sizes.
"""
function write_slim(slim_path, R, means, means_source, orig_size, results_path,
                    nbins; max_tau=NaN, tau_factor=NaN, heavy_bins::Int=nbins,
                    heavy_elems::Int=64, dbmeta=nothing)
    # One fine grid for every observable; heavy ones get k fine bins per bin.
    nb = min(nbins, R.N)
    binsize = fld(R.N, nb); dropped_rem = R.N - binsize*nb
    k   = max(1, fld(nb, max(1, heavy_bins)))
    nbh = fld(nb, k)
    binned = Dict{String,Any}(); heavy = String[]
    for (obs, stream) in R.streams
        (stream isa AbstractVector && isempty(stream)) && continue
        length(stream) == R.N || (@warn "stream length != N, skipped" obs len=length(stream) N=R.N; continue)
        if k > 1 && _nelem(stream[1]) > heavy_elems
            binned[obs] = bin_fixed(stream, binsize*k, nbh); push!(heavy, obs)
        else
            binned[obs] = bin_fixed(stream, binsize, nb)
        end
    end
    sort!(heavy)
    decorrelated = (max_tau isa Number && isfinite(max_tau)) ? (binsize >= 2*max_tau) : missing
    tmp = slim_path * ".tmp"
    jldopen(tmp, "w") do f
        f["slim_format"]    = SLIM_FORMAT
        f["schema_version"] = SLIM_SCHEMA
        f["created_at"]     = now()
        f["source_results"] = Dict("basename" => basename(results_path),
                                    "orig_bytes" => orig_size,
                                    "num_chunks" => length(R.chunk_sizes))
        R.sim_data === nothing || (f["sim_data"] = R.sim_data)
        f["meta/num_measurements"] = R.N
        f["meta/num_bins"]         = nb
        f["meta/bin_size"]         = binsize
        f["meta/heavy_obs"]        = heavy
        f["meta/heavy_bins"]       = isempty(heavy) ? nb : nbh
        f["meta/heavy_bin_size"]   = isempty(heavy) ? binsize : binsize*k
        f["meta/heavy_elems"]      = heavy_elems
        if dbmeta !== nothing
            f["meta/db_path"] = dbmeta.db_path
            f["meta/db_seed"] = dbmeta.seed
            f["meta/db_info"] = dbmeta.info
            f["meta/db_stale"] = dbmeta.stale                 # means from a record with N != stream N
            f["meta/db_num_measurements"] = dbmeta.num_meas   # that record's N
        end
        f["meta/dropped_remainder"]= dropped_rem
        f["meta/chunk_sizes"]      = R.chunk_sizes
        f["meta/means_source"]     = means_source
        f["meta/tau_max"]          = max_tau              # worst autocorr time (from cache)
        f["meta/tau_factor"]       = tau_factor           # bin_size targeted >= tau_factor*tau_max
        f["meta/bins_decorrelated"]= decorrelated         # bin_size >= 2*tau_max ?
        f["meta/mean_obs"]         = sort(collect(keys(R.meanobs)))
        f["meta/binned_obs"]       = sort(collect(keys(binned)))
        f["meta/dropped_obs"]      = R.dropped
        for (obs, v) in binned;    f["binned/$obs"]  = v; end
        for (obs, v) in R.meanobs; f["meanobs/$obs"] = v; end
        if means !== nothing
            for (obs, d) in means; f["means/$obs"] = d; end
        end
    end
    mv(tmp, slim_path; force = true)
    return (nbins = nb, binsize = binsize, dropped_rem = dropped_rem,
            decorrelated = decorrelated, n_heavy = length(heavy), heavy_bins = nbh,
            n_binned = length(binned), n_meanobs = length(R.meanobs),
            n_means = means === nothing ? 0 : length(means))
end

# ---- optional self-check: binned mean of a linear obs ≈ cached value --------
"""
    verify_slim(slim_path; tol=1e-6) -> Bool

Self-check of a slim: the mean of the binned n stream must match the stored mean of n
(within 1e-3; they differ only by the dropped tail).
"""
function verify_slim(slim_path; tol = 1e-6)
    ok = true
    jldopen(slim_path, "r") do f
        haskey(f, "means/n") && haskey(f, "binned/n") || return
        cached = f["means/n"]["value"]
        binmean = mean(f["binned/n"])          # mean-of-bins == mean-of-stream (equal bins)
        # they differ only by the dropped tail remainder (< num_bins samples)
        if !isapprox(binmean, cached; atol = 1e-3, rtol = 1e-3)
            @warn "verify: binned n mean deviates from cache" binmean cached
            ok = false
        end
    end
    return ok
end

# ---- CLI --------------------------------------------------------------------
"""
    _opt(args, name, default)

Value of `--name=...` in `args`, else `default`.
"""
function _opt(args, name, default)
    for a in args
        startswith(a, "--$name=") && return split(a, "=", limit=2)[2]
    end
    return default
end

"""
    _flag(args, name) -> Bool

True if `--name` is in `args`.
"""
_flag(args, name) = any(a -> a == "--$name", args)

"""
    slim_one(target; nbins=100, out=nothing, tau_factor=0.0, min_bins=20, heavy_bins=30,
             heavy_elems=64, dbmap=nothing, force=false, allow_missing_means=false,
             accept_stale=false, verify=false, quiet=false) -> Symbol

Slim one seed (`target` = seed folder or its results.jld2). Means come from
means_cache.jld2 (V6) or from the matching means_db record via `dbmap` (V7). Returns
`:ok`, or why it was skipped (`:no_results`, `:exists`, `:no_means`, `:no_db`,
`:db_stale`, `:db_ambiguous`). Never deletes results.jld2.
"""
function slim_one(target::String; nbins::Int=100, out::Union{Nothing,String}=nothing,
                  tau_factor::Float64=0.0, min_bins::Int=20,
                  heavy_bins::Int=30, heavy_elems::Int=64, dbmap=nothing,
                  force::Bool=false, allow_missing_means::Bool=false,
                  accept_stale::Bool=false, verify::Bool=false, quiet::Bool=false)
    dir = isdir(target) ? target : dirname(target)
    results_path = isdir(target) ? joinpath(target, "results.jld2") : target
    cache_path   = joinpath(dir, "means_cache.jld2")
    slim_path    = out === nothing ? joinpath(dir, "results_slim.jld2") : out

    isfile(results_path) || (println("SKIP  no results.jld2: $dir"); return :no_results)
    if isfile(slim_path) && !force
        quiet || println("SKIP  slim exists: $slim_path  (use --force)")
        return :exists
    end

    means = nothing; means_source = "absent"
    if isfile(cache_path)
        means = jldopen(cache_path, "r") do f
            haskey(f, "means") ? f["means"] : nothing
        end
        means === nothing || (means_source = "means_cache")
    end
    hits = (means === nothing && dbmap !== nothing) ? get(dbmap, _dirkey(dir), NamedTuple[]) : nothing
    if means === nothing && (hits === nothing || isempty(hits)) && !allow_missing_means
        why = hits === nothing ? "no means_cache.jld2 and no --dbmap" : "no means_db record for this dir"
        println("SKIP  $why (use --allow-missing-means to bin anyway): $dir")
        return :no_means
    end

    orig_size = filesize(results_path)
    R = load_and_combine(results_path)
    if means !== nothing && haskey(means, "n") && haskey(means["n"], "num_measurements")
        mc_N = means["n"]["num_measurements"]
        mc_N == R.N || @warn "stream length != means_cache num_measurements (means are still valid, just from a different count)" stream=R.N means_cache=mc_N dir
    end

    # V7: copy the per-seed means from the ONE means_db record built from this stream.
    dbmeta = nothing
    if hits !== nothing && !isempty(hits)
        st, rec, msg = pick_db_record(hits, R.N; accept_stale=accept_stale)
        if !(st in (:ok, :stale))
            println("SKIP  $st: $msg: $dir")
            return st
        end
        st === :stale && !quiet && println("STALE $msg: $dir")
        means, info = jldopen(rec.db_path, "r") do f
            _read_group(f, "seeds/$(rec.seed)/means"), f["seeds/$(rec.seed)/info"]
        end
        means_source = "means_db"
        dbmeta = (db_path=rec.db_path, seed=rec.seed, info=info,
                  stale=(st === :stale), num_meas=rec.num_meas)
    end

    # adaptive bins: target `nbins` but keep bin_size >= tau_factor*max_tau so the
    # error re-derived from bins reproduces the (uniform, covariance-preserving) cache error.
    max_tau  = cache_max_tau(means)
    nb, _bs  = decide_bins(R.N, max_tau; tau_factor=tau_factor, min_bins=min_bins, max_bins=nbins)

    stats = write_slim(slim_path, R, means, means_source, orig_size, results_path, nb;
                       max_tau=max_tau, tau_factor=tau_factor, heavy_bins=heavy_bins,
                       heavy_elems=heavy_elems, dbmeta=dbmeta)
    slim_size = filesize(slim_path)
    verify && (verify_slim(slim_path) || @warn "verify FAILED for $slim_path")

    if !quiet
        dec = stats.decorrelated === missing ? "?" : (stats.decorrelated ? "yes" : "NO")
        @printf("OK    %s\n", dir)
        @printf("      N=%d  bins=%d (size %d, dropped %d)  heavy=%d obs @ %d bins  τmax=%.1f decorr=%s  obs: %d binned / %d meanobs / %d means [%s]\n",
                R.N, stats.nbins, stats.binsize, stats.dropped_rem, stats.n_heavy, stats.heavy_bins,
                (max_tau isa Number && isfinite(max_tau) ? max_tau : NaN), dec,
                stats.n_binned, stats.n_meanobs, stats.n_means, means_source)
        @printf("      %.1f MB -> %.2f MB  (%.1f%% smaller, reclaim %.1f MB)\n",
                orig_size/2^20, slim_size/2^20,
                100*(1 - slim_size/orig_size), (orig_size-slim_size)/2^20)
    end
    return :ok
end

"""
    _tally_str(t) -> String

`status=count` pairs of a tally (zero counts left out, except ok and error).
"""
_tally_str(t) = join(("$k=$v" for (k, v) in sort(collect(t), by=first) if v > 0 || k in (:ok, :error)), " ")

"""
    slim_batch(targets; kw...) -> Dict{Symbol,Int}

`slim_one` on many seeds in ONE process (amortizes Julia/JLD2 startup), with progress
lines and a final tally. A failure is logged and counted, and the batch goes on.
"""
function slim_batch(targets::Vector{String}; kw...)
    tally = Dict{Symbol,Int}(:ok=>0, :exists=>0, :no_means=>0, :no_results=>0, :error=>0)
    reclaimed = 0.0
    t0 = time()
    for (i, tgt) in enumerate(targets)
        status = :error
        try
            dir = isdir(tgt) ? tgt : dirname(tgt)
            res = isdir(tgt) ? joinpath(tgt, "results.jld2") : tgt
            slim = joinpath(dir, "results_slim.jld2")
            before = (isfile(res) ? filesize(res) : 0)
            status = slim_one(tgt; kw...)
            status == :ok && (reclaimed += (before - filesize(slim)) / 2^20)
        catch e
            @error "slim failed for $tgt" exception=(e, catch_backtrace())
            status = :error
        end
        tally[status] = get(tally, status, 0) + 1
        if i % 25 == 0
            @printf("[progress] %d/%d  %s  reclaimed=%.1f GB  (%.0fs)\n", i, length(targets),
                    _tally_str(tally), reclaimed/1024, time()-t0); flush(stdout)
        end
    end
    @printf("\n[batch DONE] %d targets in %.1f min | %s | reclaimed=%.2f GB\n",
            length(targets), (time()-t0)/60, _tally_str(tally), reclaimed/1024)
    return tally
end

if abspath(PROGRAM_FILE) == @__FILE__
    args     = ARGS
    positional = filter(a -> !startswith(a, "--"), args)
    nbins    = parse(Int, _opt(args, "bins", "100"))
    tau_factor = parse(Float64, _opt(args, "tau-factor", "0.0"))
    min_bins   = parse(Int, _opt(args, "min-bins", "20"))
    heavy_bins  = parse(Int, _opt(args, "heavy-bins", "30"))
    heavy_elems = parse(Int, _opt(args, "heavy-elems", "64"))
    manifest = _opt(args, "manifest", nothing)
    dbroot   = _opt(args, "build-dbmap", nothing)
    dbmapf   = _opt(args, "dbmap", nothing)
    dbmap    = dbmapf === nothing ? nothing : load_dbmap(String(dbmapf))
    dbmap === nothing || @printf("[slim] dbmap %s: %d dirs\n", dbmapf, length(dbmap))
    kw = (nbins=nbins, tau_factor=tau_factor, min_bins=min_bins,
          heavy_bins=heavy_bins, heavy_elems=heavy_elems, dbmap=dbmap,
          force=_flag(args,"force"), allow_missing_means=_flag(args,"allow-missing-means"),
          accept_stale=_flag(args,"accept-stale"),
          verify=_flag(args,"verify"), quiet=_flag(args,"quiet"))

    if dbroot !== nothing
        out = _opt(args, "out", nothing)
        out === nothing && error("--build-dbmap needs --out=<tsv>")
        build_dbmap(String(dbroot), String(out))
    elseif manifest !== nothing
        # batch mode: process 1-indexed lines [from,to] of the manifest
        lines = readlines(String(manifest))
        filter!(l -> !isempty(strip(l)), lines)
        from = parse(Int, _opt(args, "from", "1"))
        to   = parse(Int, _opt(args, "to", string(length(lines))))
        to   = min(to, length(lines)); from = max(from, 1)
        sel  = String.(strip.(lines[from:to]))
        @printf("[slim] manifest=%s  lines %d..%d of %d  (bins=%d)\n",
                manifest, from, to, length(lines), nbins)
        slim_batch(sel; kw...)
    elseif !isempty(positional)
        out = _opt(args, "out", nothing)
        slim_one(positional[1]; out=out===nothing ? nothing : String(out), kw...)
    else
        println("usage:")
        println("  julia slim_results.jl <seed_dir | results.jld2> [--bins=100] [--heavy-bins=30] [--dbmap=TSV] [--out=PATH] [--force] [--allow-missing-means] [--verify] [--quiet]")
        println("  julia slim_results.jl --manifest=FILE [--from=N --to=M] [--bins=100] [--heavy-bins=30] [--dbmap=TSV] [--force] [--allow-missing-means] [--accept-stale] [--quiet]")
        println("  julia slim_results.jl --build-dbmap=MEANS_DB_ROOT --out=TSV")
        exit(1)
    end
end
