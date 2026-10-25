# ============================================================================
# means_store.jl — compact partial-load store for sim_info.
#
# WHY: a full load of all .jld2 files reconstructs ~6000 fat entries (~0.43 s
# each, ~19 min) just to read the tiny `means` you actually plot. Per entry,
# `means` is ~60 KB while the raw per-seed `simulations` is ~1.2 MB (95%). This
# store keeps ONLY {metadata + means} per entry (drops `simulations`), so:
#   • store is ~20x smaller (~375 MB total),
#   • a query that needs a few configs loads in ~1 s instead of ~19 min,
#   • existing get_sims / plot_* helpers work UNCHANGED (entries are byte-
#     identical except the dropped `simulations`).
#
# Two artifacts:
#   means_store.jld2          keys "1".."N" -> stripped sim_dict
#   means_store_manifest.csv  id,source,entry_key,U,n,Lx,Ly,Lc,TCA,beta,
#                             diag,disorder,p,B   (the query index)
#
# USAGE (first set where the data lives, see "WHERE THE DATA IS" below)
#   ENV["V6_CLUSTER_RESULTS"] = raw"<...>\cluster_results"
#   include("data_summery.jl"); include("means_store.jl")
#   build_means_store()                              # one-time, ~20 min
#   si = load_subset(; U=[3.0,5.0], L=20, n=0.875)   # ~1 s
#   sims = get_sims_all(si; wanted_n=0.875)          # existing helpers, unchanged
# The two spectral-gap figures (F6.6/F6.7) need raw per-seed data: use
#   e = load_raw_entry(source_basename, entry_key)   # fetches `simulations` on demand
# ============================================================================
using JLD2, Printf

# WHERE THE DATA IS — set by the user, before including this file:
#   ENV["V6_CLUSTER_RESULTS"] = raw"<folder with Clean_System/, disorder/, Magnetic_Field/>"
#   ENV["MEANS_STORE_DIR"]    = raw"<folder for the store>"   # default: <V6_CLUSTER_RESULTS>/means_store
# (e.g. in ~/.julia/config/startup.jl). Unset paths stay empty, and the functions
# that need them stop with a message saying which variable to set.
const V6_CLUSTER_RESULTS = get(ENV, "V6_CLUSTER_RESULTS", "")
const MEANS_STORE_DIR  = get(ENV, "MEANS_STORE_DIR",
                            isempty(V6_CLUSTER_RESULTS) ? "" : joinpath(V6_CLUSTER_RESULTS, "means_store"))
const MEANS_STORE_JLD2 = isempty(MEANS_STORE_DIR) ? "" : joinpath(MEANS_STORE_DIR, "means_store.jld2")
const MEANS_STORE_CSV  = isempty(MEANS_STORE_DIR) ? "" : joinpath(MEANS_STORE_DIR, "means_store_manifest.csv")

"""
    _need_path(path, what)

Stop with an explanation if `path` is empty, i.e. the user has not set where `what` lives.
"""
function _need_path(path::AbstractString, what::AbstractString)
    isempty(path) && error("means_store.jl: no path for the $what. Set ENV[\"V6_CLUSTER_RESULTS\"] " *
                           "(folder with Clean_System/, disorder/, Magnetic_Field/) and/or " *
                           "ENV[\"MEANS_STORE_DIR\"] before including means_store.jl, or pass the path explicitly.")
    return path
end

# Full source set: 13 clean+disorder (== disorder_p_comparison_notebook cell 1)
# + Bigger (E.1/F5.3 L=22-28) + the 3 magnetic files the ch7 driver uses, so the
# store is the single data layer for ch5/ch6/ch7 + the notebook. (mean_field/* is
# loaded separately — different MF structure, small/fast.)
# Empty when V6_CLUSTER_RESULTS is not set.
const _CR = V6_CLUSTER_RESULTS
const DEFAULT_MEANS_SOURCES = isempty(_CR) ? String[] : [
    # clean / homogeneous (p=1)
    joinpath(_CR,"Clean_System","sim_info_Full_Sys_TCA__25-02-2026_CURIE.jld2"),
    joinpath(_CR,"Clean_System","sim_info_FullSys_TCA__12-03-2026_CURIE.jld2"),
    joinpath(_CR,"Clean_System","sim_info_FullSys_TCA__27-03-2026_CURIE.jld2"),
    joinpath(_CR,"Clean_System","sim_info_Full_TCA_Bigger__10-04-2026_CURIE.jld2"),  # L=22-28
    # disordered
    joinpath(_CR,"disorder","sim_info__Disorder__29-03-2026__LANDAU.jld2"),
    joinpath(_CR,"disorder","sim_info_Disorder_U__16-04-2026_LANDAU.jld2"),
    joinpath(_CR,"disorder","sim_info_Disorder_U__16-04-2026_CURIE.jld2"),
    joinpath(_CR,"disorder","sim_info_Disorder__05-04-2026_CURIE.jld2"),
    joinpath(_CR,"disorder","sim_info_DisorderII__26-04-2026_LANDAU.jld2"),
    joinpath(_CR,"disorder","sim_info_Disorder__26-04-2026_CURIE.jld2"),
    joinpath(_CR,"disorder","sim_info__Disorder__30-04-2026_LANDAU.jld2"),
    joinpath(_CR,"disorder","sim_info__Disorder__30-04-2026_CURIE.jld2"),
    joinpath(_CR,"disorder","sim_info__Disorder__03-05-2026_LANDAU.jld2"),
    joinpath(_CR,"disorder","sim_info__Disorder__03-05-2026_CURIE.jld2"),
    # magnetic (B field; ch7)
    joinpath(_CR,"Magnetic_Field","new_measures","Mag_10__05-05-2026_CURIE.jld2"),
    joinpath(_CR,"Magnetic_Field","new_measures","Mag_10__09-05-2026_LANDAU.jld2"),
    joinpath(_CR,"Magnetic_Field","new_measures","Mag_newLxx__10-05-2026_CURIE.jld2"),
]

"""
    _ms_read_entries(path) -> Dict{String,Any}

Load every top-level entry of a JLD2 file, falling back from mmap to IOStream when mmap
fails (network mounts); as in data_summery.jl.
"""
function _ms_read_entries(path::String)
    function rd(io)
        d = Dict{String,Any}()
        f = io === nothing ? jldopen(path, "r") : jldopen(path, "r"; iotype=io)
        try
            for k in keys(f); d[string(k)] = f[k]; end
        finally
            close(f)
        end
        return d
    end
    try
        return rd(nothing)
    catch e
        e isa ReadOnlyMemoryError || rethrow(e)
        return rd(IOStream)
    end
end

"""
    _getf(e, k, default=nothing)

`e[k]` if `e` is a Dict with key `k`, else `default`.
"""
_getf(e, k, default=nothing) = (e isa AbstractDict && haskey(e,k)) ? e[k] : default

"""
    _entry_n(e) -> Float64

The filling ⟨n⟩ stored in an entry's means, or NaN.
"""
function _entry_n(e)
    m = _getf(e, "means", nothing)
    (m isa AbstractDict && haskey(m,"n") && m["n"] isa AbstractDict) || return NaN
    v = get(m["n"], "value", NaN)
    return v isa Number ? Float64(v) : NaN
end
"""
    _csvnum(x) -> String

A number as a CSV cell (empty for `nothing` or a non-number).
"""
_csvnum(x) = x === nothing ? "" : (x isa Number ? string(x) : "")

"""
    _csvstr(x) -> String

A string as a CSV cell (empty for `nothing`, commas replaced by `;`).
"""
_csvstr(x) = x === nothing ? "" : replace(string(x), ","=>";")

"""
    build_means_store(; sources=DEFAULT_MEANS_SOURCES, store=MEANS_STORE_JLD2,
                      manifest_csv=MEANS_STORE_CSV, run_extrapolation=true,
                      drop_keys=["simulations"])

One-time: load all `sources` (with δ-extrapolation so `*_fixed_extrap` are
persisted), strip `drop_keys` (the heavy raw per-seed data) from every entry,
and write the compact store + manifest CSV. Reuses `load_and_prepare_data`.
"""
function build_means_store(; sources::Vector{String}=DEFAULT_MEANS_SOURCES,
                           store::String=MEANS_STORE_JLD2,
                           manifest_csv::String=MEANS_STORE_CSV,
                           run_extrapolation::Bool=true,
                           drop_keys::Vector{String}=["simulations"],
                           drop_invalid::Bool=true,   # skip entries with no valid means+n (failed/incomplete sims)
                           # extrapolation params: MATCH load_and_prepare_data (what made the figures)
                           min_delta_MLS::Float64=1.0, max_delta_MLS::Float64=8.0,
                           min_points::Int=4, force_full_range::Bool=true)
    _need_path(store, "means store (MEANS_STORE_DIR)")
    _need_path(manifest_csv, "means store manifest (MEANS_STORE_DIR)")
    isempty(sources) && _need_path("", "V6 sim_info source files (V6_CLUSTER_RESULTS)")
    sources = filter(isfile, sources)
    mkpath(dirname(store))
    @printf("[build_means_store] %d source files; fresh extrapolation=%s (min_pts=%d, max_MLS=%.1f, full_range=%s)\n",
            length(sources), run_extrapolation, min_points, max_delta_MLS, force_full_range)
    @printf("                    NOTE: source .jld2 files are read-only (not modified).\n")

    # Per-source load: bounds memory AND records provenance. Extrapolation is
    # per-entry (verified), so per-file == merged. Fresh extrap OVERWRITES any
    # (untrusted) *_fixed_extrap saved in the files.
    manifest = IOBuffer()
    println(manifest, "id,source,entry_key,U,n,Lx,Ly,Lc,TCA,beta,diag,disorder,p,B")
    id = 0; failed = String[]; percounts = Tuple{String,Int,Int}[]
    jldopen(store, "w") do hout
        for path in sources
            base = basename(path)
            @printf("  loading %s ...\n", base); flush(stdout)
            si = nothing
            try
                si = Dict{String,Any}(); @time merge!(si, _ms_read_entries(path))
            catch e
                @warn "FAILED to load (possibly corrupted) $base; SKIPPING" exception=e
                push!(failed, base); continue
            end
            if run_extrapolation
                try
                    extrapolate_all_delta_observables!(si; min_delta_MLS=min_delta_MLS,
                        max_delta_MLS=max_delta_MLS, min_points=min_points, force_full_range=force_full_range)
                catch e
                    @warn "extrapolation failed for $base; storing raw means as-is" exception=e
                end
            end
            kept = 0; dropped = 0
            for (key, e) in si
                key == "updated_at" && continue
                (e isa AbstractDict) || continue
                nval = _entry_n(e)
                if drop_invalid && isnan(nval)   # failed/incomplete sim: no usable means+n
                    dropped += 1; continue
                end
                for dk in drop_keys; haskey(e, dk) && delete!(e, dk); end
                id += 1; kept += 1
                hout[string(id)] = e
                Lx, Ly = get_sim_dimensions(e)
                println(manifest, join((id, _csvstr(base), _csvstr(key),
                    _csvnum(_getf(e,"U")), _csvnum(nval),
                    _csvnum(Lx), _csvnum(Ly), _csvnum(_getf(e,"SC_L")),
                    _csvnum(_getf(e,"TClustersize")), _csvnum(_getf(e,"beta")),
                    _csvstr(_getf(e,"diag_method")), _csvstr(_getf(e,"U_disorder")),
                    _csvnum(_getf(e,"impurity")), _csvnum(_getf(e,"B"))), ","))
            end
            push!(percounts, (base, kept, dropped))
            @printf("    stored %d entries, dropped %d invalid (running total %d)\n", kept, dropped, id); flush(stdout)
            si = nothing; GC.gc()   # free this file before the next
        end
        hout["__manifest_csv__"] = String(take!(copy(manifest)))   # embed for portability
    end
    open(manifest_csv, "w") do io; write(io, String(take!(manifest))); end
    @printf("[build_means_store] DONE: %d entries -> %s (%.0f MB)\n", id, store, filesize(store)/2^20)
    @printf("                    manifest -> %s\n", manifest_csv)
    println("  per-file entry counts (stored / dropped-invalid):")
    for (b, c, d) in percounts; @printf("    %-45s %5d  (dropped %d)\n", b, c, d); end
    isempty(failed) || (@printf("  *** %d FILE(S) FAILED TO LOAD: ***\n", length(failed)); foreach(b->println("    ", b), failed))
    return id
end

# ---- manifest row + parsing ------------------------------------------------
struct MSRow
    id::Int; source::String; key::String
    U::Float64; n::Float64; Lx::Float64; Ly::Float64; Lc::Float64; TCA::Float64
    beta::Float64; diag::String; disorder::String; p::Float64; B::Float64
end
"""
    _pf(s) -> Float64

A CSV cell as a Float64 (NaN if empty).
"""
_pf(s) = isempty(s) ? NaN : parse(Float64, s)

"""
    _read_manifest(manifest_csv, store) -> Vector{MSRow}

Read the store's query index: the manifest CSV if it exists, else the copy saved inside
the store under `__manifest_csv__`. Rows with fewer than 14 columns are skipped.
"""
function _read_manifest(manifest_csv::String, store::String)
    (isempty(manifest_csv) && isempty(store)) && _need_path("", "means store (MEANS_STORE_DIR)")
    txt = isfile(manifest_csv) ? read(manifest_csv, String) :
          jldopen(store,"r") do h; h["__manifest_csv__"]; end
    rows = MSRow[]
    for (i, line) in enumerate(split(chomp(txt), '\n'))
        i == 1 && continue           # header
        isempty(line) && continue
        f = split(line, ',')
        length(f) < 14 && continue
        push!(rows, MSRow(parse(Int,f[1]), f[2], f[3], _pf(f[4]), _pf(f[5]),
            _pf(f[6]), _pf(f[7]), _pf(f[8]), _pf(f[9]), _pf(f[10]),
            f[11], f[12], _pf(f[13]), _pf(f[14])))
    end
    return rows
end

"""
    _inset(x, set) -> Bool

True if `set` is `nothing` (a wildcard) or contains `x`.
"""
_inset(x, set) = set === nothing ? true : (x in set)

"""
    _numset(v) -> Union{Set{Float64}, Nothing}

A selector (scalar or vector) as a Set of Float64; `nothing` stays a wildcard.
"""
_numset(v) = v === nothing ? nothing : Set(Float64.(v isa AbstractVector ? v : [v]))

"""
    load_subset(; store=MEANS_STORE_JLD2, manifest_csv=MEANS_STORE_CSV,
                U=nothing, L=nothing, Ly=nothing, TCA=nothing, p=nothing,
                diag_method=nothing, disorder_type=nothing, B=nothing,
                n=nothing, n_tol=0.1, run_extrapolation=false, verbose=true)

Assemble a `sim_info`-shaped Dict with ONLY the entries matching the given
selectors, by reading just those keys from the means store. Unset selectors are
wildcards. Pass `U` as a scalar or vector. `n` filters loosely (|n-actual|≤n_tol);
leave it `nothing` to load all fillings and let `get_sims` pick. Returns a Dict
you feed directly to the existing helpers/plot functions.
"""
function load_subset(; store::String=MEANS_STORE_JLD2, manifest_csv::String=MEANS_STORE_CSV,
                     U=nothing, L=nothing, Ly=nothing, TCA=nothing, p=nothing,
                     diag_method=nothing, disorder_type=nothing, B=nothing,
                     n=nothing, n_tol::Float64=0.1, source=nothing,
                     run_extrapolation::Bool=true,   # redo extrapolation on load (user preference;
                     # store also has baked fresh extrap, so =false is identical+faster if you prefer)
                     # MUST match load_and_prepare_data / build_means_store so re-extrap == baked
                     min_delta_MLS::Float64=1.0, max_delta_MLS::Float64=8.0,
                     min_points::Int=4, force_full_range::Bool=true,
                     verbose::Bool=true)
    rows = _read_manifest(manifest_csv, store)
    Uset = _numset(U); pset = _numset(p); TCAset = _numset(TCA)
    srcset = source === nothing ? nothing : Set(string.(source isa AbstractVector ? source : [source]))
    sel = filter(rows) do r
        _inset(r.U, Uset) &&
        (L === nothing || r.Lx == L) &&
        (Ly === nothing || r.Ly == Ly) &&
        _inset(r.TCA, TCAset) && _inset(r.p, pset) &&
        (diag_method === nothing || r.diag == diag_method) &&
        (disorder_type === nothing || r.disorder == disorder_type) &&
        (B === nothing || (isnan(r.B) ? false : isapprox(r.B, B; atol=1e-9))) &&
        (n === nothing || (!isnan(r.n) && abs(r.n - n) <= n_tol)) &&
        (srcset === nothing || r.source in srcset)
    end
    si = Dict{String,Any}()
    t = @elapsed jldopen(store, "r") do h
        for r in sel; si[string(r.id)] = h[string(r.id)]; end
    end
    if run_extrapolation
        try
            extrapolate_all_delta_observables!(si; min_delta_MLS=min_delta_MLS,
                max_delta_MLS=max_delta_MLS, min_points=min_points, force_full_range=force_full_range)
        catch e; @warn "extrapolation failed" exception=e; end
    end
    verbose && @printf("[load_subset] %d/%d entries in %.2fs\n", length(si), length(rows), t)
    return si
end

"""
    load_raw_config(; U, p, L, TCA, n=0.875, diag_method="K0-TCA",
                    disorder_type="impurity-gaussian-kernel", n_tol=0.1)

Assemble ONE config's entries WITH raw per-seed `simulations`, fetched on demand
from the original .jld2 (via `load_raw_entry`). Returns a `sims`-shaped Dict
`beta => {"sim_dict" => raw_entry}` — exactly what `get_binned_LDOS_all_temps` /
`get_all_SG` (F6.6/F6.7) need. Use this instead of `sims_disorder[mode][U][p][TCA][L]`
when loading via the means store (which drops `simulations`).
"""
function load_raw_config(; U, p, L, TCA, n::Real=0.875, diag_method::String="K0-TCA",
                         disorder_type::String="impurity-gaussian-kernel", n_tol::Float64=0.1,
                         manifest_csv::String=MEANS_STORE_CSV, store::String=MEANS_STORE_JLD2,
                         sources::Vector{String}=DEFAULT_MEANS_SOURCES, verbose::Bool=true)
    rows = _read_manifest(manifest_csv, store)
    sel = filter(rows) do r
        isapprox(r.U, U; atol=1e-9) && r.Lx == L && r.TCA == TCA &&
        isapprox(r.p, p; atol=1e-9) && r.diag == diag_method && r.disorder == disorder_type &&
        (!isnan(r.n) && abs(r.n - n) <= n_tol)
    end
    sim = Dict{Any,Any}()
    for r in sel
        e = load_raw_entry(r.source, r.key; sources=sources)
        b = get(e, "beta", nothing)
        b === nothing && continue
        sim[b] = Dict{String,Any}("sim_dict" => e)
    end
    verbose && @printf("[load_raw_config] U=%s p=%s L=%s TCA=%s -> %d betas (raw, with simulations)\n",
                       string(U), string(p), string(L), string(TCA), length(sim))
    return sim
end

"""
    store_summary(; manifest_csv=MEANS_STORE_CSV, store=MEANS_STORE_JLD2)

Print what's in the means store, read from the manifest only (instant, no data
load): entry/file counts and the distinct U / n / Lx / p / B / disorder / diag
values available. Use it to see what you can query before calling `load_subset`.
"""
function store_summary(; manifest_csv::String=MEANS_STORE_CSV, store::String=MEANS_STORE_JLD2)
    rows = _read_manifest(manifest_csv, store)
    if isempty(rows); println("means store empty / manifest not found at $manifest_csv"); return nothing; end
    unum(f) = sort(unique(round(f(r), digits=4) for r in rows if !isnan(f(r))))
    ustr(f) = sort(unique(f(r) for r in rows if !isempty(f(r))))
    srcs = sort(unique(r.source for r in rows))
    @printf("Means store: %d entries from %d files\n", length(rows), length(srcs))
    # n is the measured n_actual (varies per sim) — bucket to nearest 0.025 to show target fillings
    println("  n (≈)    : ", sort(unique(round(round(r.n/0.025)*0.025, digits=3) for r in rows if !isnan(r.n))))
    println("  U        : ", unum(r->r.U))
    println("  Lx       : ", Int.(unum(r->r.Lx)))
    println("  p        : ", unum(r->r.p))
    println("  disorder : ", ustr(r->r.disorder))
    println("  diag     : ", ustr(r->r.diag))
    Bs = unum(r->r.B)
    println("  B        : ", length(Bs) <= 10 ? Bs : @sprintf("%d values in [%.3f, %.3f]", length(Bs), minimum(Bs), maximum(Bs)))
    println("  per-file entry counts:")
    for s in srcs; @printf("    %-46s %d\n", s, count(r->r.source==s, rows)); end
    return nothing
end

"""
    load_raw_entry(source_basename, entry_key; sources=DEFAULT_MEANS_SOURCES)

Fetch a single FULL entry (incl. raw per-seed `simulations`) from the original
.jld2 — for the two spectral-gap figures (F6.6/F6.7) that need per-seed data.
"""
function load_raw_entry(source_basename::String, entry_key::String;
                        sources::Vector{String}=DEFAULT_MEANS_SOURCES)
    isempty(sources) && _need_path("", "V6 sim_info source files (V6_CLUSTER_RESULTS)")
    path = nothing
    for s in sources; basename(s) == source_basename && (path = s; break); end
    path === nothing && error("source not found: $source_basename")
    function rd(io)
        f = io === nothing ? jldopen(path, "r") : jldopen(path, "r"; iotype=io)
        try
            return f[entry_key]
        finally
            close(f)
        end
    end
    try
        return rd(nothing)
    catch e
        e isa ReadOnlyMemoryError || rethrow(e)
        return rd(IOStream)
    end
end
