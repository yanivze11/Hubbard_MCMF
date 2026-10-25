# Generate the human-readable Obsidian summary from sim_info_database.csv.
# Run after audit_sim_info.jl. No external deps (base Julia only).

const CSV_IN       = joinpath(get(ENV, "V6_CLUSTER_RESULTS", ""), "sim_info_database.csv")
const VAULT_MD_OUT = get(ENV, "SIM_INFO_MD_OUT", joinpath(get(ENV, "V6_CLUSTER_RESULTS", ""), "sim_info_database.md"))

# Reuse the CSV split helper from audit_sim_info.jl.
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

function read_csv(path::String)
    open(path, "r") do io
        header = _split_csv_line(readline(io))
        rows = Vector{Dict{String,String}}()
        for line in eachline(io)
            isempty(line) && continue
            fs = _split_csv_line(line)
            row = Dict{String,String}()
            for (i, h) in enumerate(header)
                row[h] = i <= length(fs) ? fs[i] : ""
            end
            push!(rows, row)
        end
        return header, rows
    end
end

# Compact representation of a unique-value list (≤ display elements).
function _compact_unique(values; max_show=8)
    uniq = sort(unique(filter(!isempty, values)))
    isempty(uniq) && return "—"
    n = length(uniq)
    if n <= max_show
        return join(uniq, ", ")
    else
        return join(uniq[1:max_show], ", ") * ", … ($(n) total)"
    end
end

# Flag duplicate files (same entry count + identical (U, n_target, Lx, T) multiset).
function detect_duplicate_files(rows)
    by_file = Dict{String, Vector{Dict{String,String}}}()
    for r in rows
        push!(get!(by_file, r["file"], Dict{String,String}[]), r)
    end
    sigs = Dict{String, Vector{NTuple{4,String}}}()
    for (f, rs) in by_file
        sig = sort([(r["U"], r["n_target"], r["Lx"], r["T"]) for r in rs])
        sigs[f] = sig
    end
    dupes = Dict{String, Vector{String}}()  # canonical → [duplicates]
    seen = Dict{Vector{NTuple{4,String}}, String}()
    for (f, sig) in sigs
        if haskey(seen, sig)
            canon = seen[sig]
            push!(get!(dupes, canon, String[]), f)
        else
            seen[sig] = f
        end
    end
    return dupes
end

function file_summary_block(io, file::String, rs::Vector{Dict{String,String}}, duplicate_of=nothing)
    cat   = rs[1]["category"]
    clust = rs[1]["cluster"]
    fdate = rs[1]["file_date"]
    println(io, "### `$file`")
    println(io)
    println(io, "- **Category**: $cat  •  **Cluster**: $clust  •  **Date**: $(isempty(fdate) ? "—" : fdate)  •  **Entries**: $(length(rs))")
    if duplicate_of !== nothing
        println(io, "- ⚠️  **Duplicate of `$duplicate_of`** (same parameter signature). Consider deleting one.")
    end
    methods    = _compact_unique([r["method"] for r in rs])
    Us         = _compact_unique([r["U"] for r in rs])
    ns         = _compact_unique([r["n_target"] for r in rs])
    Ls         = _compact_unique([r["Lx"] for r in rs])
    Lcs        = _compact_unique([r["Lc"] for r in rs])
    Ts         = _compact_unique([r["T"] for r in rs])
    Bs         = _compact_unique([r["B"] for r in rs])
    disorders  = _compact_unique([r["disorder_kind"] for r in rs])
    ps         = _compact_unique([r["p"] for r in rs])
    println(io, "- **Method**: $methods  •  **L**: $Ls  •  **L_c**: $Lcs")
    println(io, "- **U**: $Us  •  **n**: $ns  •  **T**: $Ts  •  **B**: $Bs")
    println(io, "- **Disorder kind**: $disorders  •  **p**: $ps")

    n_conv_true  = count(r -> r["n_converged"] == "true",  rs)
    n_conv_false = count(r -> r["n_converged"] == "false", rs)
    status_breakdown = Dict{String,Int}()
    for r in rs
        s = r["seed_status"]
        status_breakdown[s] = get(status_breakdown, s, 0) + 1
    end
    pcts = filter(!isnan, [parse(Float64, r["pct_measure_done"]) for r in rs if !isempty(r["pct_measure_done"]) && tryparse(Float64, r["pct_measure_done"]) !== nothing])
    mean_pct = isempty(pcts) ? NaN : sum(pcts)/length(pcts)
    println(io, "- **n-converged**: $n_conv_true / $(length(rs))" * (n_conv_false > 0 ? " ($(n_conv_false) failed)" : ""))
    println(io, "- **Seed-status breakdown**: " * join(["`$k`×$v" for (k,v) in sort(collect(status_breakdown); by=x->-x[2])], ", "))
    if !isnan(mean_pct)
        println(io, "- **Mean pct_measure_done**: $(round(100*mean_pct; digits=1))%")
    end
    feeds = unique(filter(!isempty, [r["feeds_figures"] for r in rs]))
    if !isempty(feeds)
        feeds_str = join(feeds, ", ")
        println(io, "- **Feeds figures**: $feeds_str")
    end
    println(io)
end

function main()
    isfile(CSV_IN) || error("CSV not found: $CSV_IN — run audit_sim_info.jl first.")
    _, rows = read_csv(CSV_IN)
    println("Loaded $(length(rows)) rows.")
    dupes = detect_duplicate_files(rows)

    by_cat = Dict{String, Dict{String, Vector{Dict{String,String}}}}()
    for r in rows
        cat = r["category"]
        f   = r["file"]
        get!(by_cat, cat, Dict{String, Vector{Dict{String,String}}}())
        push!(get!(by_cat[cat], f, Dict{String,String}[]), r)
    end

    # Reverse-map the duplicates: for each duplicate file, name its canonical
    duplicate_of = Dict{String,String}()
    for (canon, dupelist) in dupes
        for d in dupelist
            duplicate_of[d] = canon
        end
    end

    mkpath(dirname(VAULT_MD_OUT))
    open(VAULT_MD_OUT, "w") do io
        println(io, "---")
        println(io, "tags: [simulations, database, sim_info, auto-generated]")
        println(io, "created: 2026-04-26")
        println(io, "updated: $(string(today()))")
        println(io, "parent: [[03_Simulations]]")
        println(io, "---")
        println(io)
        println(io, "# Sim_info database — what's where")
        println(io)
        println(io, "> Auto-generated from `cluster_results/sim_info_database.csv` by `generate_sim_info_md.jl`. Do not hand-edit; edit the CSV's `feeds_figures` / `notes` columns and re-run.")
        println(io, ">")
        println(io, "> Schema reference: [[sim_info_structure]]")
        println(io)
        println(io, "## Top-line stats")
        println(io)
        n_files     = length(unique([r["file"] for r in rows]))
        cats_str    = join(sort(collect(keys(by_cat))), ", ")
        n_dupes     = isempty(dupes) ? 0 : sum(length(v) for v in values(dupes))
        println(io, "- **Files**: $n_files")
        println(io, "- **Total entries (sims)**: $(length(rows))")
        println(io, "- **Categories**: $cats_str")
        if n_dupes > 0
            println(io, "- ⚠️  **Likely duplicate files**: $n_dupes (see flags below)")
        end
        println(io)
        for cat in ("clean", "disorder", "magnetic", "mean_field", "other")
            haskey(by_cat, cat) || continue
            println(io, "## Category: `$cat`")
            println(io)
            for (f, rs) in sort(collect(by_cat[cat]); by=first)
                file_summary_block(io, f, rs, get(duplicate_of, f, nothing))
            end
        end
    end
    println("Wrote: $VAULT_MD_OUT")
end

using Dates: today

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
