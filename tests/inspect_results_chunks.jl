# inspect_results_chunks.jl — not a test: list the measurement chunks of one
# results.jld2 (length and non-zero entries of the action stream, empty or not) and
# the totals. Read only.
#   julia tests/inspect_results_chunks.jl <results.jld2 | seed_dir>
using JLD2

function resolve_results_path(input_path::String)
    if isdir(input_path)
        return joinpath(input_path, "results.jld2")
    end
    return input_path
end

function count_nonzero(v)
    if isempty(v)
        return 0
    end
    return count(x -> x != 0.0, v)
end

function chunk_health(chunk)
    if !haskey(chunk, "Ss")
        return 0, 0, true
    end
    ss = chunk["Ss"]
    n = length(ss)
    nnz = count_nonzero(ss)
    is_empty = n == 0 || nnz == 0
    return n, nnz, is_empty
end

function inspect_results(results_path::String)
    if !isfile(results_path)
        error("results file not found: $results_path")
    end

    total_chunks = 0
    empty_chunks = 0
    total_meas = 0
    total_nnz_ss = 0

    println("Inspecting: $results_path")
    println("chunk_id | len(Ss) | nonzero(Ss) | empty")

    jldopen(results_path, "r") do file
        meas_keys = filter(k -> startswith(k, "measurement_"), keys(file))
        meas_keys_sorted = sort(meas_keys, by = k -> parse(Int, split(k, "_")[2]))

        for k in meas_keys_sorted
            total_chunks += 1
            chunk = file[k]
            n, nnz, is_empty = chunk_health(chunk)
            empty_chunks += is_empty ? 1 : 0
            total_meas += n
            total_nnz_ss += nnz
            println("$k | $n | $nnz | $is_empty")
        end
    end

    println("---")
    println("total chunks: $total_chunks")
    println("empty chunks: $empty_chunks")
    println("total Ss entries: $total_meas")
    println("nonzero Ss entries: $total_nnz_ss")
end

if length(ARGS) < 1
    println("Usage: julia test_code/inspect_results_chunks.jl <results.jld2 or simulation_dir>")
    exit(1)
end

input_path = ARGS[1]
results_path = resolve_results_path(input_path)
inspect_results(results_path)
