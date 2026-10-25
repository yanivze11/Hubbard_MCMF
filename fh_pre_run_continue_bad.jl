# ============================================================================
# fh_pre_run_continue_bad.jl — find the points whose statistics look BAD and
# keep the SAME chains running.
#
# HOW IT DIFFERS FROM THE OTHER pre_run SCRIPTS
#   fh_pre_run_V.jl / fh_pre_run_mu_disorder.jl SUBMIT NEW simulations and
#   register them in means_db. This one registers nothing and creates nothing:
#   it scans means_db, flags the points whose averages are untrustworthy, and
#   re-sbatches THOSE simulations' existing directories. fh_run_n.jl passes
#   `resume_path=sim_path`, so a resubmitted job loads its checkpoint.jld2 and
#   continues the same Markov chain toward num_measure — the thermalization and
#   the configuration it reached are preserved.
#
#   An incomplete run is NOT a problem in itself. Only a point whose average is
#   untrustworthy is, and that is what the three tests below decide.
#
# RUN IT THROUGH pre_pre_run.jl (which exports the cluster environment):
#
#   julia pre_pre_run.jl fh_pre_run_continue_bad.jl --dry-run --report-all
#   julia pre_pre_run.jl fh_pre_run_continue_bad.jl
#   julia pre_pre_run.jl fh_pre_run_continue_bad.jl --us=2,4,6 --beta-phis=0,0.5
#   julia pre_pre_run.jl --prerun-queue=general fh_pre_run_continue_bad.jl
#
#   Everything after the script name is forwarded to this script. Direct
#   `julia fh_pre_run_continue_bad.jl ...` works too, provided MEANS_DB_CLUSTER
#   and JULIA_DEPOT_PATH are already exported.
#
# WHAT COUNTS AS BAD — a point needs to fail ONE test
#
#   1. THIN        n_uncorrelated_measurements < --min-uncorr (default 30)
#                  -> too few independent samples behind the mean.
#
#   2. NONMONO     S(0) RISES with temperature along the L-curve, by more than
#                  --rise-factor (default 1.3x), while above the noise floor.
#                  S(0) cannot physically grow with T, so the COLDER point never
#                  ordered -> its chain is not equilibrated.
#
#   3. L_OUTLIER   at fixed (U,T) the point sits below --neigh-factor (default
#                  0.4) times the MEDIAN over the other lattice sizes, while
#                  those neighbours are still well ordered. Finite size moves
#                  S(0) smoothly; a collapse at a single L does not.
#
#   Test 1 alone MISSES the worst point we know of: U=6, L=26, T=0.0599 reported
#   tau=21, 4% relative error and n_uncorr=16, because the chain sat trapped in a
#   disordered configuration for ~700 measurements and the binning analysis saw a
#   quiet, "well converged" flat line. Only tests 2 and 3 catch it. That is why
#   the physics tests sit alongside the statistical one.
#
# AFTERWARDS
#   Re-run the reduction, or the database keeps serving the OLD averages:
#       update_means("<V7_DATA_ROOT>/data/<campaign folder>")
#   The stored means are only as fresh as the last reduction — the L=26 point
#   above reports 347 measurements while 1065 sit in its results.jld2. (Note:
#   update_means_filtered does NOT export to means_db, only the legacy monolith,
#   so use update_means for anything analysis reads.)
#
# FLAGS (all optional; ENV of the same name in CAPS still works as a fallback)
#   --dry-run              scan and report, submit nothing
#   --report-all           print every point with its verdict, not just the bad
#   --us=6                 comma lists: which U ....................... US
#   --ls=8,12,14,18,22,26  which lattice sizes ....................... LS
#   --beta-phis=0.5        which decoupling schemes .................. BETA_PHIS
#   --vs=0.0               which disorder strengths .................. VS
#   --n=0.9                wanted_n .................................. WANTED_N
#   --t-min=0.0 --t-max=0.25   temperature window .................... T_MIN/T_MAX
#   --min-uncorr=30        test 1 threshold .......................... MIN_UNCORR
#   --rise-factor=1.3      test 2 threshold .......................... RISE_FACTOR
#   --neigh-factor=0.4     test 3 threshold .......................... NEIGH_FACTOR
#   --noise-floor=0.05     ignore points below this S/Delta0^2 ....... NOISE_FLOOR
#   --max-submit=10000     cap on resubmitted jobs ................... MAX_SUBMIT
#   --obs=S_Delta_Q0       observable the tests read ................. OBS
# ============================================================================
using JLD2, Printf, Statistics

# --- CLI first: the flags may set the roots the includes capture --------------
const _ARGS = Dict{String,String}()
for a in ARGS
    if startswith(a, "--")
        k, v = occursin("=", a) ? split(a[3:end], "=", limit=2) : (a[3:end], "1")
        _ARGS[lowercase(String(k))] = String(v)
    end
end
if haskey(_ARGS, "help")
    println(read(@__FILE__, String)[1:findfirst("using JLD2", read(@__FILE__, String))[1]-1])
    exit(0)
end
# A flag wins over the environment; the environment wins over the default.
"""
    opt(flag, env, default) -> String

Value of `--flag=...` if given, else of the environment variable `env`, else `default`.
"""
opt(flag, env, default) = get(_ARGS, flag, get(ENV, env, default))

"""
    optf(flag, env, default) -> Float64

`opt` parsed as a Float64.
"""
optf(flag, env, default) = parse(Float64, opt(flag, env, string(default)))

"""
    opti(flag, env, default) -> Int

`opt` parsed as an Int.
"""
opti(flag, env, default) = parse(Int,     opt(flag, env, string(default)))

"""
    optb(flag, env) -> Bool

True if `--flag` is given or the environment variable `env` is "1".
"""
optb(flag, env) = haskey(_ARGS, flag) || get(ENV, env, "0") == "1"

include(joinpath(@__DIR__, "cluster_config.jl"))   # V7_DATA_ROOT, detect_cluster_name
haskey(ENV, "MEANS_DB_ROOT") || (ENV["MEANS_DB_ROOT"] = joinpath(V7_DATA_ROOT, "means_db"))
# means_db's SEARCH roots (plural) come from the laptop settings (BACKUP_DRIVE,
# *_ROOT, *_MOUNT), which are not set on a cluster — leaving them would make the
# scan silently find nothing there. Point them at this cluster's own database
# when it is present; on a laptop that path is absent, so the laptop settings
# still apply. Each
# cluster therefore scans and continues only its OWN runs, which is exactly the
# split we want: run this on Landau AND on Curie.
if !haskey(ENV, "MEANS_DB_ROOTS") && isdir(ENV["MEANS_DB_ROOT"])
    ENV["MEANS_DB_ROOTS"] = ENV["MEANS_DB_ROOT"]
end

include("means_db.jl")
include("mf_kspace_density_channel.jl")
using .MFKSpaceDensityChannel

const DRY_RUN    = optb("dry-run", "DRY_RUN")
const REPORT_ALL = optb("report-all", "REPORT_ALL")
const MAX_SUBMIT = opti("max-submit", "MAX_SUBMIT", 10000)

const MIN_UNCORR   = opti("min-uncorr",   "MIN_UNCORR",   30)
const RISE_FACTOR  = optf("rise-factor",  "RISE_FACTOR",  1.3)
const NEIGH_FACTOR = optf("neigh-factor", "NEIGH_FACTOR", 0.4)
const NOISE_FLOOR  = optf("noise-floor",  "NOISE_FLOOR",  0.05)   # in S/Delta0^2

"""
    parse_list(s, f) -> Vector

Split the comma list `s` and apply `f` to each non-empty item.
"""
parse_list(s, f) = [f(strip(x)) for x in split(s, ",") if !isempty(strip(x))]
const US        = parse_list(opt("us",        "US",        "6.0"), x -> parse(Float64, x))
const LS        = parse_list(opt("ls",        "LS",        "8,12,14,18,22,26"), x -> parse(Int, x))
const BETA_PHIS = parse_list(opt("beta-phis", "BETA_PHIS", "0.5"), x -> parse(Float64, x))
const VS        = parse_list(opt("vs",        "VS",        "0.0"), x -> parse(Float64, x))
const WANTED_N  = optf("n",     "WANTED_N", 0.9)
const T_MIN     = optf("t-min", "T_MIN", 0.0)
const T_MAX     = optf("t-max", "T_MAX", 0.25)
const OBS       = opt("obs", "OBS", "S_Delta_Q0")

# pre_pre_run.jl exports MEANS_DB_CLUSTER; fall back to the hostname.
const THIS_CLUSTER = detect_cluster_name()

"""
    Point

One scanned parameter point: (U, L, V, β_φ, β), the mean of `OBS` with its error,
correlation time and (un)correlated measurement counts (of its worst-sampled seed),
the seed folders behind it, and its means_db file.
"""
struct Point
    U::Float64; L::Int; V::Float64; beta_phi::Float64; beta::Float64
    S::Float64; err::Float64; tau::Float64; n_uncorr::Int; n_meas::Int
    seed_dirs::Vector{Tuple{String,String}}      # (seed, dir_path)
    path::String
end

"""
    collect_points() -> Vector{Point}

Read from means_db every point in the scope (`US`, `LS`, `VS`, `BETA_PHIS`, `WANTED_N`,
T window) and its value of `OBS`. With several seeds, the one with the fewest
uncorrelated measurements is kept, so a good seed cannot hide a bad one.
"""
function collect_points()
    pts = Point[]
    for U in US, L in LS, V in VS, bphi in BETA_PHIS
        hits = find_all(; U=U, Lx=L, Ly=L, V=V, wanted_n=WANTED_N,
                          alpha=1.0, beta_phi=bphi, Qx=0.0, Qy=0.0)
        for row in hits
            beta = Float64(get(row, "beta", NaN))
            isfinite(beta) || continue
            T = 1 / beta
            (T_MIN <= T <= T_MAX) || continue
            info = try
                _open_jld2(row["path"]) do f
                    S = NaN; err = NaN; tau = NaN; nu = 0; nm = 0
                    dirs = Tuple{String,String}[]
                    haskey(f, "seeds") || return (S, err, tau, nu, nm, dirs)
                    for s in sort(string.(collect(keys(f["seeds"]))))
                        g = "seeds/$s/means/$OBS"
                        if haskey(f, g)
                            d = f[g]
                            # several seeds: keep the WORST-sampled one, so a
                            # good seed cannot hide a bad one.
                            if !isfinite(S) || get(d, "num_uncorrelated_measurements", 0) < nu
                                S   = get(d, "value", NaN)
                                err = get(d, "error", NaN)
                                tau = Float64(get(d, "correlation_time", NaN))
                                nu  = Int(get(d, "num_uncorrelated_measurements", 0))
                                nm  = Int(get(d, "num_measurements", 0))
                            end
                        end
                        gi = "seeds/$s/info"
                        if haskey(f, gi)
                            dp = string(get(f[gi], "dir_path", ""))
                            isempty(dp) || push!(dirs, (s, dp))
                        end
                    end
                    return (S, err, tau, nu, nm, dirs)
                end
            catch e
                @warn "unreadable means file" path=row["path"] exception=e
                continue
            end
            (S, err, tau, nu, nm, dirs) = info
            isfinite(S) || continue
            push!(pts, Point(U, L, V, bphi, beta, S, err, tau, nu, nm, dirs, string(row["path"])))
        end
    end
    return pts
end

# S normalized by the T=0 mean-field Delta_0^2 — comparable across U and L.
const _D0 = Dict{Tuple{Float64,Int},Float64}()

"""
    d0sq(U, L) -> Float64

Δ₀², the clean T ≈ 0 k-space mean-field gap squared at (U, L, `WANTED_N`) (cached).
Used to normalize S(0) so that U and L can be compared.
"""
function d0sq(U, L)
    get!(_D0, (U, L)) do
        solve_mf(; t=1.0, U=U, alpha=1.0, beta_phi=0.0, beta=50000.0,
                   Lx=L, Ly=L, target_n=WANTED_N).Delta
    end ^ 2
end

"""
    flag(pts) -> (norm, reasons)

Apply the three tests of the header (THIN, NONMONO, L_OUTLIER) to every point.
`norm` maps (U, L, β_φ, β) to S/Δ₀²; `reasons` maps each flagged point to its failed tests.
"""
function flag(pts)
    norm =Dict((p.U, p.L, p.beta_phi, p.beta) => p.S / d0sq(p.U, p.L) for p in pts)
    reasons = Dict{Any,Vector{String}}()

    for p in pts
        key = (p.U, p.L, p.beta_phi, p.beta)
        rs  = String[]
        s   = norm[key]

        # 1. thin statistics
        p.n_uncorr < MIN_UNCORR &&
            push!(rs, @sprintf("THIN(n_uncorr=%d<%d)", p.n_uncorr, MIN_UNCORR))

        # 2. S rises with T along this L-curve -> the COLDER point is the suspect
        curve = sort([q for q in pts if q.U == p.U && q.L == p.L &&
                                        q.beta_phi == p.beta_phi], by = q -> 1/q.beta)
        i = findfirst(q -> q.beta == p.beta, curve)
        if i !== nothing && i < length(curve)
            hotter = curve[i+1]
            sh = norm[(hotter.U, hotter.L, hotter.beta_phi, hotter.beta)]
            if sh > RISE_FACTOR * s && sh > NOISE_FLOOR
                push!(rs, @sprintf("NONMONO(S=%.3f at T=%.4f but %.3f at T=%.4f)",
                                   s, 1/p.beta, sh, 1/hotter.beta))
            end
        end

        # 3. collapse relative to the other lattice sizes at the same T
        others = [norm[(q.U, q.L, q.beta_phi, q.beta)] for q in pts
                  if q.U == p.U && q.beta_phi == p.beta_phi &&
                     isapprox(q.beta, p.beta; rtol=1e-6) && q.L != p.L]
        if length(others) >= 2
            med = median(others)
            if med > 3 * NOISE_FLOOR && s < NEIGH_FACTOR * med
                push!(rs, @sprintf("L_OUTLIER(S=%.3f vs median %.3f at other L)", s, med))
            end
        end

        isempty(rs) || (reasons[key] = rs)
    end
    return norm, reasons
end

# ---------------------------------------------------------------------------
# Resubmit: sbatch the EXISTING slurm.txt of a flagged run's sim directory.
# dir_path points at <sim dir>/seed_N; slurm.txt and sim.jld2 live one level up.
# ---------------------------------------------------------------------------
"""
    sim_dir_of(seed_dir) -> String

The simulation folder (holding slurm.txt and sim.jld2) of a `.../seed_N` folder.
"""
sim_dir_of(seed_dir) = dirname(rstrip(String(seed_dir), ['/', '\\']))

"""
    resubmit!(p::Point, counter::Ref{Int})

`sbatch` the existing slurm.txt of every seed of a flagged point, so the job resumes from
its checkpoint. Skips seeds whose folder is not on this cluster, that have no
slurm.txt, or no checkpoint (that would restart, not continue); stops at `MAX_SUBMIT`.
Increments `counter` per (would-be) submission.
"""
function resubmit!(p::Point, counter::Ref{Int})
    for (s, dp) in p.seed_dirs
        sim_dir = sim_dir_of(dp)
        slurm   = joinpath(sim_dir, "slurm.txt")
        ckpt    = joinpath(dp, "checkpoint.jld2")
        if !isdir(sim_dir)
            @printf("    - seed %s: %s not on this cluster, skipping\n", s, sim_dir)
            continue
        end
        if !isfile(slurm)
            @printf("    ! seed %s: no slurm.txt in %s — needs a pre_run submission instead\n", s, sim_dir)
            continue
        end
        # Without a checkpoint a resubmit RESTARTS the chain from scratch, which
        # is the opposite of what this script is for.
        if !isfile(ckpt)
            @printf("    ! seed %s: NO checkpoint.jld2 — would restart, not continue; skipping\n", s)
            continue
        end
        if counter[] >= MAX_SUBMIT
            @printf("    - MAX_SUBMIT=%d reached\n", MAX_SUBMIT)
            return
        end
        @printf("    CONTINUE seed %s -> %s\n", s, sim_dir)
        if DRY_RUN
            counter[] += 1
        else
            try
                run(`sbatch $slurm`)
                counter[] += 1
            catch e
                @error "sbatch failed" sim_dir exception=e
            end
        end
    end
end

"""
    main()

Scan the points (`collect_points`), flag the bad ones (`flag`), print a table, and
continue the flagged runs (`resubmit!`; nothing is submitted with `--dry-run`).
"""
function main()
    @printf("cluster       : %s%s\n", isempty(THIS_CLUSTER) ? "(unknown)" : THIS_CLUSTER,
            isempty(THIS_CLUSTER) ? "   — resubmits only what is visible on this filesystem" : "")
    @printf("means_db root : %s\n", MEANS_DB_ROOT)
    @printf("scope         : U=%s  L=%s  beta_phi=%s  V=%s  n=%g  T in [%g, %g]  obs=%s\n",
            string(US), string(LS), string(BETA_PHIS), string(VS), WANTED_N, T_MIN, T_MAX, OBS)
    @printf("thresholds    : min_uncorr=%d  rise=%.2f  neigh=%.2f  noise_floor=%.2f\n\n",
            MIN_UNCORR, RISE_FACTOR, NEIGH_FACTOR, NOISE_FLOOR)
    DRY_RUN && println("DRY RUN — scanning only, nothing will be submitted\n")

    pts = collect_points()
    @printf("scanned %d point(s)\n\n", length(pts))
    isempty(pts) && return

    norm, reasons = flag(pts)

    @printf("%-5s %-4s %-8s %-8s %-9s %-8s %-8s %s\n",
            "U", "L", "T", "S/D0^2", "rel.err%", "tau", "n_uncorr", "verdict")
    counter = Ref(0); nflag = 0
    for p in sort(pts, by = q -> (q.U, q.beta_phi, q.L, 1/q.beta))
        key = (p.U, p.L, p.beta_phi, p.beta)
        bad = haskey(reasons, key)
        bad && (nflag += 1)
        (bad || REPORT_ALL) || continue
        @printf("%-5g %-4d %-8.4f %-8.4f %-9.1f %-8.1f %-8d %s\n",
                p.U, p.L, 1/p.beta, norm[key], 100*p.err/max(p.S, 1e-12),
                p.tau, p.n_uncorr, bad ? join(reasons[key], " ") : "ok")
        bad && resubmit!(p, counter)
    end

    @printf("\n%d of %d point(s) flagged; %s %d job(s)\n",
            nflag, length(pts), DRY_RUN ? "would continue" : "continued", counter[])
    println("\nEach continued job resumes from checkpoint.jld2 and keeps measuring toward")
    println("num_measure. Afterwards run update_means on the campaign folder(s), or the")
    println("database keeps serving the averages from the last reduction.")
end

# ── Script entry point ──────────────────────────────────────────────────────
# Runs ONLY when executed directly, so include()-ing it from the REPL is safe.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
