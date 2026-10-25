# ============================================================================
# scratch_fix_paths.jl — turn a scratch run's sim.scratch.jld2 into sim.jld2.
#
# A JOB_MODE=scratch job (scratch_job.sh) runs in /home/scr/<user>/mcjob_<id>/<unit>,
# so the sim.jld2 it writes records scratch paths (info_path and, per seed,
# dir_path / checkpoint_path / info_path). num_control, slim and means_db find
# the results through dir_path, so before that file may be called sim.jld2 on
# /home/data every ".../mcjob_<id>/<unit>" prefix is replaced by the unit folder.
#
#   julia scratch_fix_paths.jl <unit_dir> [<unit_dir> ...]
#   julia scratch_fix_paths.jl --campaign <campaign_dir>     # every unit in it
#
# For each unit that has sim.scratch.jld2: load it, fix every string, write
# .sim.fix_tmp.jld2, read it back, rename it to sim.jld2, delete
# sim.scratch.jld2. A unit without sim.scratch.jld2 is left alone.
#
# scratch_job.sh runs this at the end of every job. Run it by hand only to
# recover a unit whose job ended without it (the .out file says so).
# --campaign lists the campaign folder: run it as ONE process, not per unit.
#
# Do NOT run it on a unit whose job is still queued or running: the next
# requeue would stage in sim.jld2 instead of the live mu-search state.
# ============================================================================
using JLD2

# ".../mcjob_<jobid>/<unit>" followed by a separator or the end. Either
# separator, so local Windows test runs are handled too.
const SCRATCH_PREFIX = r"^(.*?[/\\]mcjob_[^/\\]+[/\\][^/\\]+)(?=[/\\]|$)"

"""
    fix_value(x, unit, n::Ref{Int})

Replace the scratch prefix ".../mcjob_<id>/<unit>" by `unit` in `x`: a string, every
string inside a (nested) Dict (changed in place), or a vector of strings. Other values
are returned unchanged. `n` counts the strings changed.
"""
function fix_value(x, unit::AbstractString, n::Ref{Int})
    if x isa AbstractString
        m = match(SCRATCH_PREFIX, x)
        m === nothing && return x
        n[] += 1
        return unit * x[ncodeunits(m.captures[1])+1:end]
    elseif x isa AbstractDict
        for k in collect(keys(x))
            x[k] = fix_value(x[k], unit, n)
        end
        return x
    elseif x isa AbstractVector{<:AbstractString}
        return [fix_value(s, unit, n) for s in x]
    end
    return x
end

"""
    fix_unit(unit_dir) -> Int or nothing

Replace sim.jld2 by the path-fixed sim.scratch.jld2. Returns the number of
strings changed, or `nothing` when the unit has no sim.scratch.jld2.
"""
function fix_unit(unit_dir::AbstractString)
    unit = rstrip(unit_dir, ['/', '\\'])
    src  = joinpath(unit, "sim.scratch.jld2")
    isfile(src) || return nothing
    info = JLD2.load(src)
    n = Ref(0)
    fix_value(info, unit, n)
    tmp = joinpath(unit, ".sim.fix_tmp.jld2")
    JLD2.save(tmp, info)
    JLD2.load(tmp)                 # must read back before it replaces anything
    mv(tmp, joinpath(unit, "sim.jld2"); force=true)
    rm(src)
    return n[]
end

"""
    main(args)

Run `fix_unit` on each unit folder in `args`, or on every folder of a campaign with
`--campaign <dir>`. Prints a summary; exits with code 1 if any unit failed.
"""
function main(args)
    units = String[]
    if length(args) == 2 && args[1] == "--campaign"
        camp = args[2]
        for d in readdir(camp; join=true)
            isdir(d) && push!(units, d)
        end
    elseif !isempty(args) && !any(startswith("--"), args)
        append!(units, args)
    else
        println(stderr, "usage: julia scratch_fix_paths.jl <unit_dir>... | --campaign <dir>")
        exit(2)
    end
    n_fixed = 0; failed = 0
    for u in units
        try
            r = fix_unit(u)
            r === nothing && continue
            n_fixed += 1
            println("fixed $u ($r path(s))")
        catch e
            failed += 1
            println(stderr, "FAILED $u: ", sprint(showerror, e))
        end
    end
    println("scratch_fix_paths: $n_fixed unit(s) fixed, $failed failed, $(length(units)) looked at")
    exit(failed == 0 ? 0 : 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
