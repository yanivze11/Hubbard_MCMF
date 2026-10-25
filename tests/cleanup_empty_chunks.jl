# cleanup_empty_chunks.jl — not a test: a repair tool for one results.jld2. Lists
# the measurement chunks whose action stream is empty or all zero (left by an
# interrupted run); with --apply it rewrites the file without them, renumbering
# the remaining chunks. Without --apply it only reports.
#   julia tests/cleanup_empty_chunks.jl <results.jld2 | seed_dir> [--apply]
using JLD2
using Dates

function resolve_results_path(input_path::String)
    if isdir(input_path)
        return joinpath(input_path, "results.jld2")
    end
    return input_path
end

function is_empty_ss_chunk(ss)
    return isempty(ss) || all(x -> x == 0.0, ss)
end

function collect_measurement_keys(file)
    meas_keys = filter(k -> startswith(k, "measurement_"), keys(file))
    return sort(meas_keys, by = k -> parse(Int, split(k, "_")[2]))
end

function inspect_chunks(results_path::String)
    kept = String[]
    dropped = String[]

    jldopen(results_path, "r") do file
        for k in collect_measurement_keys(file)
            chunk = file[k]
            if !haskey(chunk, "Ss")
                push!(dropped, k)
                continue
            end
            if is_empty_ss_chunk(chunk["Ss"])
                push!(dropped, k)
            else
                push!(kept, k)
            end
        end
    end

    return kept, dropped
end

function cleanup_empty_chunks(results_path::String; apply_changes::Bool=false)
    if !isfile(results_path)
        error("results file not found: $results_path")
    end

    kept, dropped = inspect_chunks(results_path)

    println("Results file: $results_path")
    println("total measurement chunks: $(length(kept) + length(dropped))")
    println("kept chunks: $(length(kept))")
    println("empty/malformed chunks: $(length(dropped))")
    if !isempty(dropped)
        println("dropped chunk ids: $(join(dropped, ", "))")
    end

    if !apply_changes
        println("Dry run only. Re-run with --apply to rewrite file without empty chunks.")
        return
    end

    if isempty(dropped)
        println("No empty chunks found. Nothing to rewrite.")
        return
    end

    backup_path = results_path * ".bak_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    cp(results_path, backup_path; force=true)
    println("Backup created: $backup_path")

    tmp_path = results_path * ".tmp"

    jldopen(results_path, "r") do src
        jldopen(tmp_path, "w") do dst
            all_keys = collect(keys(src))
            non_meas_keys = filter(k -> !startswith(k, "measurement_"), all_keys)
            for k in non_meas_keys
                dst[k] = src[k]
            end

            new_idx = 1
            for k in collect_measurement_keys(src)
                if !(k in kept)
                    continue
                end
                dst["measurement_$(new_idx)"] = src[k]
                new_idx += 1
            end
        end
    end

    mv(tmp_path, results_path; force=true)
    println("Rewrote file without empty chunks.")
end

if length(ARGS) < 1
    println("Usage: julia test_code/cleanup_empty_chunks.jl <results.jld2 or simulation_dir> [--apply]")
    exit(1)
end

input_path = ARGS[1]
apply_changes = any(arg -> arg == "--apply", ARGS[2:end])
results_path = resolve_results_path(input_path)
cleanup_empty_chunks(results_path; apply_changes=apply_changes)
