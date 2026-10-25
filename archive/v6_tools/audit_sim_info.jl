using JLD2, Dates

const CLUSTER_RESULTS_DIR = get(ENV, "V6_CLUSTER_RESULTS", "")
const CSV_OUT             = joinpath(CLUSTER_RESULTS_DIR, "sim_info_database.csv")
const VAULT_MD_OUT        = get(ENV, "SIM_INFO_MD_OUT", joinpath(CLUSTER_RESULTS_DIR, "sim_info_database.md"))

const SCHEMA_COLS = [
    :file, :entry_key, :category, :cluster, :file_date,
    :method, :Lx, :Ly, :Lc,
    :U, :n_target, :n_actual, :n_converged,
    :T, :beta, :B,
    :disorder_kind, :p, :U_puddle_params,
    :n_seeds, :seed_status, :n_seeds_finished, :pct_measure_done,
    :num_measure, :num_thermal, :num_uncorrelated_measurments,
    :pairing_symmetry, :boundary_condition, :gauge_choice, :guess,
    :start_time, :info_path, :n_obs,
    :feeds_figures, :notes,
]

safe(d, k, default=missing) = (d isa Dict && haskey(d, k)) ? d[k] : default

function category_from_path(file::String)
    p = lowercase(replace(file, '\\' => '/'))
    occursin("/clean_system/", p) && return "clean"
    occursin("/disorder/", p)     && return "disorder"
    occursin("/magnetic_field/", p) && return "magnetic"
    occursin("/mean_field/", p)   && return "mean_field"
    return "other"
end

function cluster_from_filename(name::String)
    occursin("CURIE", name)  && return "CURIE"
    occursin("LANDAU", name) && return "LANDAU"
    return "unknown"
end

# Pull the dd-mm-YYYY token out of filenames like sim_info_..._16-04-2026_CURIE.jld2
function date_from_filename(name::String)
    m = match(r"(\d{2}-\d{2}-\d{4})", name)
    m === nothing && return missing
    try
        return Date(m.captures[1], dateformat"dd-mm-yyyy")
    catch
        return missing
    end
end

function aggregate_seed_status(simulations)
    simulations isa Dict || return ("missing", 0)
    counts = Dict{String,Int}()
    for (_, seed_entry) in simulations
        seed_entry isa Dict || continue
        st = string(safe(seed_entry, "status", "unknown"))
        counts[st] = get(counts, st, 0) + 1
    end
    summary = join(["$k:$v" for (k,v) in sort(collect(counts); by=first)], ",")
    n_finished = get(counts, "finished", 0)
    return (summary, n_finished)
end

# Mean of num_current_measurments / num_measure across seeds.
function pct_done(simulations, num_measure)
    (simulations isa Dict && num_measure isa Number && num_measure > 0) || return missing
    pcts = Float64[]
    for (_, se) in simulations
        se isa Dict || continue
        ncur = safe(se, "num_current_measurments", missing)
        ncur isa Number || continue
        push!(pcts, ncur / num_measure)
    end
    isempty(pcts) && return missing
    return round(sum(pcts) / length(pcts); digits=3)
end

function shape_or_size(x)
    x === missing && return missing
    x isa AbstractArray && return string(size(x))
    return string(x)
end

function format_array_param(x)
    x === missing && return missing
    x isa AbstractArray && return "[" * join(string.(x), ",") * "]"
    return string(x)
end

function extract_entry_row(file::String, entry_key, entry)
    cat = category_from_path(file)
    fname = basename(file)

    row = Dict{Symbol,Any}(c => missing for c in SCHEMA_COLS)
    row[:file]      = fname
    row[:entry_key] = string(entry_key)
    row[:category]  = cat
    row[:cluster]   = cluster_from_filename(fname)
    row[:file_date] = date_from_filename(fname)

    entry isa Dict || (return row)

    diag = string(safe(entry, "diag_method", "unknown"))
    row[:method] = occursin("TCA", diag) ? "TCA" : (diag == "K0" ? "MCMF" : diag)

    row[:Lx] = safe(entry, "Lx")
    row[:Ly] = safe(entry, "Ly")
    row[:Lc] = safe(entry, "TClustersize")
    row[:U]  = safe(entry, "U")
    row[:n_target] = safe(entry, "wanted_n")

    conv = safe(entry, "convergence", missing)
    if conv isa Dict
        row[:n_actual]    = safe(conv, "best_n")
        row[:n_converged] = safe(conv, "converged")
    end

    beta = safe(entry, "beta")
    row[:beta] = beta
    row[:T]    = (beta isa Number && beta > 0) ? round(1/beta; digits=6) : missing
    row[:B]    = safe(entry, "B")

    row[:disorder_kind]    = safe(entry, "U_disorder")
    row[:p]                = safe(entry, "impurity")
    row[:U_puddle_params]  = format_array_param(safe(entry, "U_puddle_params"))

    seeds = safe(entry, "seeds", missing)
    row[:n_seeds] = (seeds isa AbstractArray) ? length(seeds) : missing

    sims = safe(entry, "simulations", missing)
    summary, nfin = aggregate_seed_status(sims)
    row[:seed_status]      = summary
    row[:n_seeds_finished] = nfin

    row[:num_measure]                  = safe(entry, "num_measure")
    row[:num_thermal]                  = safe(entry, "num_thermal")
    row[:num_uncorrelated_measurments] = safe(entry, "num_uncorrelated_measurments")
    row[:pct_measure_done] = pct_done(sims, row[:num_measure])

    row[:pairing_symmetry]    = safe(entry, "pairing_symmetry")
    row[:boundary_condition]  = safe(entry, "boundary_condition")
    row[:gauge_choice]        = safe(entry, "gauge_choice")
    row[:guess]               = safe(entry, "guess")
    row[:start_time]          = safe(entry, "start_time")
    row[:info_path]           = safe(entry, "info_path")

    means = safe(entry, "means", missing)
    row[:n_obs] = (means isa Dict) ? length(means) : missing

    return row
end

function audit_file(file::String)
    rows = Dict{Symbol,Any}[]
    JLD2.jldopen(file, "r") do f
        for k in keys(f)
            entry = try
                f[k]
            catch err
                @warn "failed to read entry" file=basename(file) key=k err
                continue
            end
            push!(rows, extract_entry_row(file, k, entry))
        end
    end
    return rows
end

function gather_jld2_files(dir::String)
    files = String[]
    for (root, _, fs) in walkdir(dir)
        for f in fs
            endswith(f, ".jld2") && push!(files, joinpath(root, f))
        end
    end
    return sort(files)
end

# Minimal CSV parsing: split on commas not inside double quotes, unwrap "" → ".
function _split_csv_line(line::String)
    fields = String[]
    buf = IOBuffer()
    in_q = false
    i = 1
    while i <= lastindex(line)
        c = line[i]
        if in_q
            if c == '"'
                if i+1 <= lastindex(line) && line[i+1] == '"'
                    write(buf, '"'); i += 2; continue
                else
                    in_q = false; i += 1; continue
                end
            else
                write(buf, c); i += 1
            end
        else
            if c == ','
                push!(fields, String(take!(buf))); i += 1
            elseif c == '"'
                in_q = true; i += 1
            else
                write(buf, c); i += 1
            end
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end

# Idempotent merge: keep manual annotations (feeds_figures, notes) from existing CSV.
function load_existing_annotations(path::String)
    out = Dict{Tuple{String,String}, NamedTuple}()
    isfile(path) || return out
    open(path, "r") do io
        header_line = readline(io)
        header = _split_csv_line(header_line)
        idx = Dict(Symbol(h) => i for (i, h) in enumerate(header))
        ifile = get(idx, :file, 0)
        ikey  = get(idx, :entry_key, 0)
        iff   = get(idx, :feeds_figures, 0)
        inote = get(idx, :notes, 0)
        (ifile == 0 || ikey == 0) && return
        for line in eachline(io)
            isempty(line) && continue
            fs = _split_csv_line(line)
            length(fs) < max(ifile, ikey) && continue
            ff = (iff > 0   && iff   <= length(fs)) ? fs[iff]   : ""
            nt = (inote > 0 && inote <= length(fs)) ? fs[inote] : ""
            ((isempty(ff) && isempty(nt))) && continue
            out[(fs[ifile], fs[ikey])] = (
                feeds_figures = isempty(ff) ? missing : ff,
                notes         = isempty(nt) ? missing : nt,
            )
        end
    end
    return out
end

# CSV escaping: quote if contains comma, quote, or newline; escape " as "".
function _csv_field(x)
    s = x === missing ? "" : string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv(path::String, rows::Vector{Dict{Symbol,Any}}, cols::Vector{Symbol})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(cols), ","))
        for r in rows
            println(io, join((_csv_field(get(r, c, missing)) for c in cols), ","))
        end
    end
end

# Sort rows by a tuple of columns; missings sort last.
function _sort_key(r::Dict{Symbol,Any}, cols::Vector{Symbol})
    map(cols) do c
        v = get(r, c, missing)
        v === missing ? (1, "") : (0, v isa Number ? Float64(v) : string(v))
    end
end

_log(args...) = (println(args...); flush(stdout))

function main()
    files = gather_jld2_files(CLUSTER_RESULTS_DIR)
    _log("Found $(length(files)) jld2 files under cluster_results/")
    _log("")

    all_rows = Dict{Symbol,Any}[]
    for f in files
        rel = relpath(f, CLUSTER_RESULTS_DIR)
        print("  ▶ $rel ... "); flush(stdout)
        t0 = time()
        try
            rs = audit_file(f)
            append!(all_rows, rs)
            _log("$(length(rs)) entries  ($(round(time()-t0; digits=1)) s)")
        catch err
            _log("FAILED: $err")
        end
    end

    _log("\nTotal entries: $(length(all_rows))")

    # Restore manual annotations
    existing = load_existing_annotations(CSV_OUT)
    for r in all_rows
        key = (r[:file], r[:entry_key])
        if haskey(existing, key)
            r[:feeds_figures] = existing[key].feeds_figures
            r[:notes]         = existing[key].notes
        end
    end

    sort_cols = [:category, :file, :U, :n_target, :Lx, :T]
    sort!(all_rows; by = r -> _sort_key(r, sort_cols))
    write_csv(CSV_OUT, all_rows, SCHEMA_COLS)
    _log("CSV written: $CSV_OUT  ($(length(all_rows)) rows × $(length(SCHEMA_COLS)) cols)")

    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
