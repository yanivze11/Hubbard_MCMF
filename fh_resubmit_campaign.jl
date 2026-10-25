# ============================================================================
# fh_resubmit_campaign.jl — continue every UNFINISHED simulation of a campaign.
#
#   Walks <V7_DATA_ROOT>/data/<folder>/<i>/ on THIS cluster and re-sbatches the
#   existing slurm.txt of every sim that has not reached num_measure.
#   fh_run_n.jl resumes from seed_N/checkpoint.jld2, so each chain CONTINUES —
#   thermalization and accumulated measurements are kept.
#
#   Unlike fh_pre_run_V.jl it creates and registers nothing; unlike
#   fh_pre_run_continue_bad.jl it does not judge statistics — "unfinished" is
#   the only criterion. Safe to re-run: sims that are queued/running (per
#   squeue) or finished are skipped, so run it again whenever jobs get killed
#   by the time limit or preempted.
#
# Run it as a SLURM job through pre_pre_run.jl, like every cluster script
# (arguments after the script name are forwarded):
#
#   julia pre_pre_run.jl fh_resubmit_campaign.jl --folder=Phi_channel__17-09-2026 --dry-run
#   julia pre_pre_run.jl fh_resubmit_campaign.jl --folder=Phi_channel__17-09-2026
#   julia pre_pre_run.jl fh_resubmit_campaign.jl --folder=... --time=7-00:00:00
#
# It only RESUMES existing sim directories. Units that were never submitted are
# fh_pre_run_V.jl's job. Run that AFTER this job has finished: running both at
# once could make this script resubmit a directory fh_pre_run_V has just
# created but not yet sbatch'ed.
#
# FLAGS
#   --folder=NAME     campaign folder under <V7_DATA_ROOT>/data   (required)
#   --dry-run         report only, submit nothing
#   --max-submit=N    cap on jobs submitted this run (default: no cap)
#   --time=D-HH:MM:SS passed to sbatch as --time (default: slurm.txt / partition)
#   --list            print every sim with its state, not just the submitted ones
#
# MEMORY: a slurm.txt that asks for less than slurm_memory.jl's rule for its
# lattice size is submitted with a larger --mem-per-cpu on the sbatch command
# line, which overrides the file. Old campaigns used a flat 4 GB, which
# OOM-killed every Landau L=32 job.
# ============================================================================
using Printf
include(joinpath(@__DIR__, "cluster_config.jl"))   # V7_DATA_ROOT
include(joinpath(@__DIR__, "slurm_memory.jl"))

const _ARGS = Dict{String,String}()
for a in ARGS
    startswith(a, "--") || continue
    k, v = occursin("=", a) ? split(a[3:end], "=", limit=2) : (a[3:end], "1")
    _ARGS[lowercase(String(k))] = String(v)
end

const FOLDER    = get(_ARGS, "folder", "")
const DRY_RUN    = haskey(_ARGS, "dry-run")
const LIST       = haskey(_ARGS, "list")
const MAX_SUBMIT = parse(Int, get(_ARGS, "max-submit", string(typemax(Int))))
const WALLTIME   = get(_ARGS, "time", "")

isempty(FOLDER) && error("pass --folder=<campaign folder under $(joinpath(V7_DATA_ROOT, "data"))>")
const CAMPAIGN = joinpath(V7_DATA_ROOT, "data", FOLDER)
isdir(CAMPAIGN) || error("no such campaign folder on this cluster: $CAMPAIGN")

const _MEAS_RE = r"Measurments taken : (\d+)/(\d+)"

"""
    seed_progress(seed_dir) -> Union{Tuple{Int,Int}, Nothing}

Measurements taken and wanted, `(taken, target)`, from the last "Measurments taken : X/Y"
line of the seed's logs. A fresh run writes simulation_log.txt and a resumed one appends
to log.txt, so the further of the two is used. `nothing` if no such line exists.
"""
function seed_progress(seed_dir)
    best = nothing
    for f in ("log.txt", "simulation_log.txt")
        p = joinpath(seed_dir, f)
        isfile(p) || continue
        m_last = nothing
        for m in eachmatch(_MEAS_RE, read(p, String))
            m_last = m
        end
        m_last === nothing && continue
        x, y = parse(Int, m_last[1]), parse(Int, m_last[2])
        (best === nothing || x > best[1]) && (best = (x, y))
    end
    return best
end

"""
    mem_override(slurm_text) -> Union{Int, Nothing}

The `--mem-per-cpu` (GB) to pass on the sbatch command line, or `nothing` when the
slurm.txt already asks for enough (`mem_per_cpu_gb`). Lattice size comes from the job
name ("..._Lx32Ly32U..."), CPUs and memory from the #SBATCH lines.
"""
function mem_override(slurm_text::AbstractString)
    mL = match(r"Lx(\d+)Ly(\d+)", slurm_text)
    mL === nothing && return nothing
    mc = match(r"#SBATCH\s+(?:-c|--cpus-per-task[= ])\s*(\d+)", slurm_text)
    cpus = mc === nothing ? 1 : parse(Int, mc[1])
    mm = match(r"--mem-per-cpu=(\d+)\s*G", slurm_text)
    have = mm === nothing ? 0 : parse(Int, mm[1])
    need = mem_per_cpu_gb(parse(Int, mL[1]), parse(Int, mL[2]), cpus)
    return need > have ? need : nothing
end

"""
    active_workdirs() -> Set{String}

Working folders of this user's queued and running jobs (`squeue -o %Z`), so nothing is
submitted twice. Errors if squeue fails.
"""
function active_workdirs()
    user = get(ENV, "USER", "")
    out = try
        read(`squeue -h -u $user -o %Z`, String)
    catch e
        error("squeue failed ($e) — refusing to submit without knowing what is already queued")
    end
    return Set(rstrip.(filter(!isempty, split(out, '\n')), '/'))
end

"""
    main()

Classify every `<campaign>/<i>` folder as finished, queued, unfinished or no-slurm, and
`sbatch` the slurm.txt of each unfinished one (with `--time` and a memory override if
needed), up to `--max-submit`. Prints a summary.
"""
function main()
    @printf("campaign : %s\n", CAMPAIGN)
    DRY_RUN && println("DRY RUN — nothing will be submitted")
    active = active_workdirs()
    @printf("%d job(s) of yours currently in squeue\n\n", length(active))

    dirs = sort(filter(d -> isdir(joinpath(CAMPAIGN, d)) && all(isdigit, d), readdir(CAMPAIGN)),
                by = d -> parse(Int, d))
    counts = Dict{String,Int}()
    bump(k) = (counts[k] = get(counts, k, 0) + 1)
    n_sub = 0

    for d in dirs
        sim_dir = joinpath(CAMPAIGN, d)
        slurm   = joinpath(sim_dir, "slurm.txt")
        seeds   = filter(s -> startswith(s, "seed_") && isdir(joinpath(sim_dir, s)), readdir(sim_dir))
        prog    = [seed_progress(joinpath(sim_dir, s)) for s in seeds]

        state = if !isfile(slurm)
            "no-slurm"       # submission never completed — fh_pre_run_V resubmits the unit
        elseif rstrip(sim_dir, '/') in active
            "queued"
        elseif !isempty(prog) && all(p -> p !== nothing && p[1] >= p[2], prog)
            "finished"
        else
            "unfinished"
        end
        bump(state)

        ptxt = isempty(prog) ? "not started" :
               join([p === nothing ? "?" : @sprintf("%d/%d", p...) for p in prog], ", ")
        if state == "unfinished" && n_sub < MAX_SUBMIT
            # A run with no checkpoint restarts from scratch — still the right
            # thing for a never-started sim, so only flag it.
            ck = any(s -> isfile(joinpath(sim_dir, s, "checkpoint.jld2")), seeds)
            mem = mem_override(read(slurm, String))
            @printf("  SUBMIT %-5s %s%s%s\n", d, ptxt, ck ? "" : "   (no checkpoint: starts fresh)",
                    mem === nothing ? "" : "   (mem-per-cpu -> $(mem) GB)")
            if !DRY_RUN
                opts = String[]
                isempty(WALLTIME) || push!(opts, "--time=$WALLTIME")
                mem === nothing || push!(opts, "--mem-per-cpu=$(mem)G")
                cmd = `sbatch $opts $slurm`
                try
                    run(cmd)
                catch e
                    @error "sbatch failed" sim_dir exception=e
                    continue
                end
            end
            n_sub += 1
        elseif LIST || state == "no-slurm"
            @printf("  %-8s %-5s %s\n", state, d, ptxt)
        end
        flush(stdout)
    end

    println()
    for k in ("finished", "queued", "unfinished", "no-slurm")
        @printf("  %-10s : %d\n", k, get(counts, k, 0))
    end
    @printf("\n%s %d job(s)\n", DRY_RUN ? "would submit" : "submitted", n_sub)
    get(counts, "no-slurm", 0) > 0 &&
        println("no-slurm dirs are leftovers of an interrupted submission; re-running " *
                "fh_pre_run_V.jl submits those units into fresh directories.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
