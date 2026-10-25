# tests/scratch_job_check.jl — check a unit that ran with JOB_MODE=scratch.
#
#   julia tests/scratch_job_check.jl [--expect-resume] <unit_dir> [<unit_dir> ...]
#
# PASS needs, for every unit:
#   - no sim.scratch.jld2 and no leftover temp files (.*.sync_tmp, .sim.fix_tmp.jld2)
#   - sim.jld2 records no scratch path ("mcjob_"), and each seed's dir_path is
#     <unit>/seed_N (what num_control / slim use to find the results)
#   - seed_N/results.jld2 holds exactly num_measure measurements: fewer = lost
#     on a requeue, more = measured twice
#   - stdout.log and scratch.log were copied back
#   - with --expect-resume: seed_N/log.txt says the run resumed from a checkpoint
using JLD2

function strings_in(x, out=String[])
    if x isa AbstractString
        push!(out, x)
    elseif x isa AbstractDict
        foreach(v -> strings_in(v, out), values(x))
    elseif x isa AbstractVector
        foreach(v -> strings_in(v, out), x)
    end
    return out
end

samepath(a, b) = normpath(replace(a, '\\' => '/')) == normpath(replace(b, '\\' => '/'))

function count_measurements(results_path)
    n = 0
    jldopen(results_path, "r") do f
        for k in keys(f)
            startswith(k, "measurement_") || continue
            ch = f[k]
            haskey(ch, "Ss") && (n += length(ch["Ss"]))
        end
    end
    return n
end

function check_unit(unit; expect_resume=false)
    problems = String[]
    bad(msg) = push!(problems, msg)
    unit = rstrip(unit, ['/', '\\'])

    isfile(joinpath(unit, "sim.scratch.jld2")) && bad("sim.scratch.jld2 is still there")
    for (root, _, files) in walkdir(unit), f in files
        (endswith(f, ".sync_tmp") || endswith(f, ".fix_tmp.jld2")) && bad("leftover temp file $(joinpath(root, f))")
    end
    for f in ("stdout.log", "scratch.log")
        isfile(joinpath(unit, f)) || bad("$f was not copied back")
    end

    info = JLD2.load(joinpath(unit, "sim.jld2"))
    scr = filter(s -> occursin("mcjob_", s), strings_in(info))
    isempty(scr) || bad("sim.jld2 still records scratch path(s): $(first(scr, 3))")
    sims = get(info, "simulations", nothing)
    if !(sims isa AbstractDict) || isempty(sims)
        bad("sim.jld2 has no 'simulations' entry (the run's sim.jld2 never came back)")
    else
        for (seed, sd) in sims
            dp = string(get(sd, "dir_path", ""))
            samepath(dp, joinpath(unit, "seed_$seed")) || bad("seed $seed dir_path = $dp")
            res = joinpath(unit, "seed_$seed", "results.jld2")
            if !isfile(res)
                bad("seed $seed: no results.jld2")
            else
                n, want = count_measurements(res), info["num_measure"]
                n == want || bad("seed $seed: $n measurements in results.jld2, want $want")
                println("    seed $seed: $n / $want measurements, status=$(get(sd, "status", "?"))")
            end
            if expect_resume
                log = joinpath(unit, "seed_$seed", "log.txt")
                (isfile(log) && occursin("Resuming", read(log, String))) ||
                    bad("seed $seed: log.txt shows no resume")
            end
        end
    end
    return problems
end

function main(args)
    expect_resume = "--expect-resume" in args
    units = filter(a -> !startswith(a, "--"), args)
    isempty(units) && error("usage: julia scratch_job_check.jl [--expect-resume] <unit_dir>...")
    n_fail = 0
    for u in units
        println("$u")
        p = check_unit(u; expect_resume=expect_resume)
        if isempty(p)
            println("  PASS")
        else
            n_fail += 1
            foreach(m -> println("  FAIL: $m"), p)
        end
    end
    println(n_fail == 0 ? "ALL PASS ($(length(units)) unit(s))" : "$n_fail of $(length(units)) unit(s) FAILED")
    exit(n_fail == 0 ? 0 : 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
