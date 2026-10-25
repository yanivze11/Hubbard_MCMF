# ============================================================================
# tests/scratch_requeue_test.jl — cluster test of JOB_MODE=scratch with a forced
# requeue. Small units, own folder, NOT registered in means_db.
#
# Run every step through SLURM (pre_pre_run.jl), from the V7 checkout:
#
#   1. submit   julia pre_pre_run.jl tests/scratch_requeue_test.jl submit
#               -> <V7_DATA_ROOT>/tests/scratch_requeue__<stamp>/{1,2}
#                  unit 1 runs to the end untouched; unit 2 gets requeued.
#                  Copy-back every SYNC_INTERVAL (default here 120 s).
#   2. requeue  once unit 2 has copied a checkpoint back (its .out file says
#               "start on ..." and `ls <unit 2>/seed_1/` shows checkpoint.jld2),
#               run by hand on the login node:   scontrol requeue <job id>
#   3. check    after both jobs left the queue:
#               julia pre_pre_run.jl tests/scratch_requeue_test.jl check <folder>
#
# Optional env: SYNC_INTERVAL (default 120), TEST_L (6), TEST_NUM_MEASURE (10000),
#               TEST_MEASURING_INTERVAL (2), JULIA_BIN.
# ============================================================================
using JLD2, Dates, Printf

haskey(ENV, "SYNC_INTERVAL") || (ENV["SYNC_INTERVAL"] = "120")
const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "slurm_job_mode.jl"))   # also loads cluster_config.jl (V7_DATA_ROOT, JULIA_BIN)
include(joinpath(@__DIR__, "scratch_job_make_unit.jl"))

const RUN_FILE     = joinpath(REPO, "fh_run_n.jl")

function submit()
    depot = get(ENV, "JULIA_DEPOT_PATH", "")
    isempty(depot) && error("JULIA_DEPOT_PATH is unset: submit through pre_pre_run.jl")
    isfile(JULIA_BIN) || error("JULIA_BIN does not exist: $JULIA_BIN")
    isfile(SCRATCH_JOB_SH) || error("missing $SCRATCH_JOB_SH")
    partitions = get(ENV, "PARTITIONS", "allq")
    exclude    = get(ENV, "EXCLUDE_NODES", "")
    cpus       = parse(Int, get(ENV, "CPUS_PER_TASK", "1"))
    L  = parse(Int, get(ENV, "TEST_L", "6"))
    nm = parse(Int, get(ENV, "TEST_NUM_MEASURE", "10000"))
    mi = parse(Int, get(ENV, "TEST_MEASURING_INTERVAL", "2"))

    folder = joinpath(V7_DATA_ROOT, "tests", "scratch_requeue__" * Dates.format(now(), "dd-mm-yyyy_HHMM"))
    println("folder         : $folder")
    println("sync interval  : $(SYNC_INTERVAL) s;  L=$L num_measure=$nm measuring_interval=$mi;  cpus=$cpus")
    for (i, role) in ((1, "untouched"), (2, "REQUEUE-ME"))
        unit = joinpath(folder, "$i")
        make_unit(unit; L=L, num_measure=nm, measuring_interval=mi)
        slurm = joinpath(unit, "slurm.txt")
        write(slurm, scratch_slurm_script(; job_name="scratch_test/$(role)_$i", dir=unit,
              cpus=cpus, mem_per_cpu_gb=4, partitions=partitions, exclude_nodes=exclude,
              depot=depot, julia_bin=JULIA_BIN, run_file=RUN_FILE))
        jobid = strip(read(`sbatch --parsable $slurm`, String))
        @printf("unit %d (%s): job %s\n", i, role, jobid)
    end
    println("""

    NEXT
      wait until unit 2 copied a checkpoint back:   ls $folder/2/seed_1/
      then on the login node:                       scontrol requeue <job id of unit 2>
      when both jobs are gone from squeue:
        julia pre_pre_run.jl tests/scratch_requeue_test.jl check $folder""")
end

function check(folder)
    for i in (1, 2)
        unit = joinpath(folder, "$i")
        println("── unit $i: SLURM .out lines ──")
        for f in readdir(unit; join=true)
            endswith(f, ".out") && print(read(f, String))
        end
        sl = joinpath(unit, "scratch.log")
        isfile(sl) && (println("── unit $i: scratch.log ──"); print(read(sl, String)))
    end
    println()
    checker = joinpath(@__DIR__, "scratch_job_check.jl")
    ok1 = success(run(ignorestatus(`$(Base.julia_cmd()) $checker $(joinpath(folder, "1"))`)))
    ok2 = success(run(ignorestatus(`$(Base.julia_cmd()) $checker --expect-resume $(joinpath(folder, "2"))`)))
    restarted = any(f -> endswith(f, ".out") && occursin("restart #1", read(joinpath(folder, "2", f), String)),
                    readdir(joinpath(folder, "2")))
    restarted || println("FAIL: unit 2's .out shows no 'restart #1' — was it requeued?")
    println(ok1 && ok2 && restarted ? "\nSCRATCH REQUEUE TEST: PASS" : "\nSCRATCH REQUEUE TEST: FAIL")
end

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = isempty(ARGS) ? "" : ARGS[1]
    if cmd == "submit"
        submit()
    elseif cmd == "check" && length(ARGS) == 2
        check(ARGS[2])
    else
        error("usage: scratch_requeue_test.jl submit | check <folder>")
    end
end
