# ============================================================================
# bench_profile_run.jl — worker of the runtime-profile benchmark.
#
#   One SLURM job = one lattice size L and one core count. It reads sim.jld2
#   (written by bench_profile_pre_run.jl) and runs the requested sections:
#
#     profile  one MC sweep (K0-TCA) vs one make_measurement_K! (REFERENCE path,
#              fast=false, comparable to the 06-10-2026 tables), the measurement
#              split into its kernels, the phi-channel sweep, a checkpoint write,
#              and the projected wall time of one campaign point
#     proto    prototype kernels (bench_profile_kernels.jl) vs the current ones:
#              time and max relative difference
#     integ    the main code after integration: make_measurement_K!(fast=false)
#              vs make_measurement_K!(fast=true) on the same state, each fast
#              kernel of MC_Hubburd.jl alone, the worst MeasureData difference,
#              bytes per call, buffer sizes, memory estimate vs peak RSS
#     occ      one measurement of the occupation-convergence phase vs n only
#     accept   Metropolis acceptance per field for a list of (U, beta, beta_phi)
#     threads  TCA cluster action and full diagonalization vs BLAS threads
#
#   Output, in the job directory:
#     profile_log.txt        human readable, flushed as it goes
#     profile_results.json   machine readable, rewritten after every section
#                            (bench_profile_collect.jl reads these; it does not
#                            summarize "integ" yet, read it from the json / log)
#
#   Usage (from the SLURM script):  julia bench_profile_run.jl --sim_path=DIR
#   Only "integ":  BENCH_SECTIONS=integ julia pre_pre_run.jl bench_profile_pre_run.jl
#   Nothing is written to means_db, and the simulation code is not modified.
# ============================================================================
using Dates, Printf, Statistics, LinearAlgebra, JLD2

include(joinpath(@__DIR__, "..", "..", "..", "MC_Hubburd.jl"))
include(joinpath(@__DIR__, "bench_profile_kernels.jl"))
using JSON

# ── arguments ───────────────────────────────────────────────────────────────
function parse_sim_path(args)
    for (k, a) in enumerate(args)
        startswith(a, "--sim_path=") && return split(a, "=", limit=2)[2]
        a == "--sim_path" && k < length(args) && return args[k+1]
    end
    error("usage: julia bench_profile_run.jl --sim_path=DIR")
end
const SIM_PATH = String(parse_sim_path(ARGS))
const CFG = JLD2.load(joinpath(SIM_PATH, "sim.jld2"))
const CORES = Int(CFG["cores"])
BLAS.set_num_threads(CORES)

# ── logging / results ───────────────────────────────────────────────────────
const LOG_FILE = joinpath(SIM_PATH, "profile_log.txt")
const RES_FILE = joinpath(SIM_PATH, "profile_results.json")
const RESULTS = Dict{String,Any}()

function say(s::AbstractString="")
    println(stdout, s); flush(stdout)
    open(io -> println(io, s), LOG_FILE, "a")
end
function save_results()
    tmp = RES_FILE * ".tmp"
    open(io -> JSON.print(io, RESULTS, 2), tmp, "w")
    mv(tmp, RES_FILE; force=true)
end

# ── campaign numerics (same as fh_pre_run_V.jl) ─────────────────────────────
w_list_of(L)       = collect(range(-5.0, 5.0, length=101))
w_cond_of(L)       = [collect(range(4/L^2 * 0.1, 4/L^2 * 2.0, length=7)); collect(range(4/L^2 * 3.0, 16.0, length=55))]
delta_fixed_of(L)  = 4 / L^2 .* [0.5; 1.0:10.0;]

function make_sd(L; U=Float64(CFG["U"]), beta=Float64(CFG["beta"]), beta_phi=0.0, occ=false)
    Lc = Int(CFG["TClustersize"])
    SimData(; Lx=L, Ly=L, U=U, t=1.0, V=0.0, seed=1, mu=Float64(CFG["mu"]), beta=beta,
        num_measure=10, num_thermal=0, measuring_interval=1, saving_interval=10,
        # the occupation-convergence phase trims these two lists (occupation_convergence.jl)
        w_list=occ ? [-1.0, 1.0] : w_list_of(L),
        w_conductivity_range=occ ? [0.0, 0.01] : w_cond_of(L),
        delta_values_fixed=delta_fixed_of(L),
        delta_values_stiffness=Float64[], dw_values=Float64[],
        guess="uniform", U_disorder="uniform", diag_method="K0-TCA",
        pairing_symmetry="s-wave", TCluster_size_x=Lc, TCluster_size_y=Lc,
        gauge_choice="Mixed", B=0.0, alpha=1.0, beta_phi=beta_phi,
        snapshot_interval=0, phi_hist_nbins=0, Delta_hist_abs_nbins=0, Delta_hist_arg_nbins=0)
end

sweep!(d) = MC_stepV2!(d; method="K0-TCA", loop_random=true, pairing="s-wave")
relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())

# ── section: profile ────────────────────────────────────────────────────────
function measure_parts!(data)
    parts = Pair{String,Float64}[]
    rec(name, f) = push!(parts, name => @elapsed f())
    invalidate_global_diag!(data)
    rec("full diag 2Nx2N (eigvecs)", () -> ensure_global_diag_synced!(data))
    rec("FD + delta (sort)", () -> (get_FD_dist!(data); compute_delta_analitic_continuation!(data)))
    rec("n, Kx, Ky, Delta_op", () -> (avrage_n_K(data); compute_Kx_operator(data; direction="x");
                                      compute_Kx_operator(data; direction="y"); compute_Delta_operator(data)))
    w_list = data.sim_data.w_list
    kxp, kyp = data.const_struct.SF_k_path
    dmax = data.temporary_data.delta_max
    rec("LDOS, all w", () -> for w in w_list; compute_full_LDOS(data, w; delt=dmax); end)
    rec("A(k,w) on k-path, all w", () -> for w in w_list, l in eachindex(kxp)
            kg = data.const_struct.k_general_idx[kxp[l], kyp[l]]
            q_ix, K_ix = data.const_struct.k_general_qK_idx[kg]
            spectral_functionV2(data, K_ix, q_ix, w; delt=dmax)
        end)
    0.0 in w_list && rec("BZ: A(k,w=0) all k", () -> spectral_function_fullV2(data, 0.0; delt=dmax))
    q_list = data.const_struct.q_list_CC
    td = data.temporary_data
    rec("G^x(q) build, all q", () -> make_G_matrix_q_vec(data, q_list; G_out=td.G_matrix_vec_x, direction="x"))
    rec("G^y(q) build, all q", () -> make_G_matrix_q_vec(data, q_list; G_out=td.G_matrix_vec_y, direction="y"))
    Gx = td.G_matrix_vec_x; Gy = td.G_matrix_vec_y
    rec("L_ab(q) x4 from G", () -> for (a, b) in ((Gx, Gx), (Gx, Gy), (Gy, Gx), (Gy, Gy))
            current_correlation_from_G!(data, a, b, td.Lxx_temp; q_length=length(q_list))
        end)
    rec("fixed-delta L(w): ndelta*nw calls", () -> for δ in data.sim_data.delta_values_fixed, w in data.sim_data.w_conductivity_range
            current_correlation_K_4dir(data, w; pre_Gx=Gx, pre_Gy=Gy, δ=δ)
        end)
    rec("L_D_direct", () -> current_correlation_K_4dir_direct(data; pre_Gx=Gx, pre_Gy=Gy))
    rec("DdD (compute_D_dagger_D_K)", () -> compute_D_dagger_D_K(data))
    rec("Ps_l, all l", () -> for l in data.const_struct.ls_list; compute_Ps_l_K(data, l); end)
    # uses G_matrix_vec_x/y[1] as scratch, so it must come after the G consumers
    rec("chi spin/charge (q=0)", () -> susceptibility_q_vec_both!(data, data.const_struct.q_list_sus, zeros(1), zeros(1)))
    rec("phase corr", () -> for l in data.const_struct.ls_list_reduced; compute_phase_correlation(data, l); end)
    return parts
end

function section_profile(L; verbose=true)
    nsweep = Int(CFG["nsweep"])
    t_init = @elapsed (data = Lattice_Data(make_sd(L)))
    N = data.sim_data.N
    sweep!(data)                                                   # warm
    t_sweeps = [@elapsed(sweep!(data)) for _ in 1:nsweep]
    ncl = 200
    t_cl = @elapsed for _ in 1:ncl; MCMF_action_TCluster_K0(data, rand(1:N)); end
    parts = measure_parts!(data)
    invalidate_global_diag!(data)
    t_meas = @elapsed make_measurement_K!(data, 1; fast=false)   # reference path; "integ" times fast=true
    tmp = mktempdir()
    t_ckpt = @elapsed save_current_state(data, tmp)
    ckpt_mb = filesize(joinpath(tmp, "checkpoint.jld2")) / 2^20
    rm(tmp; recursive=true, force=true)
    data = nothing; GC.gc()
    # density channel on: every sweep also updates phi (twice the updates)
    dphi = Lattice_Data(make_sd(L; beta_phi=0.5))
    sweep!(dphi)
    t_sweep_phi = minimum(@elapsed(sweep!(dphi)) for _ in 1:nsweep)
    dphi = nothing; GC.gc()
    verbose || return nothing

    t_sweep = minimum(t_sweeps)
    mi = Int(CFG["measuring_interval"]); nth = Int(CFG["num_thermal"]); nme = Int(CFG["num_measure"])
    proj = (nth * t_sweep + nme * (mi * t_sweep + t_meas)) / 86400
    proj_phi = (nth * t_sweep_phi + nme * (mi * t_sweep_phi + t_meas)) / 86400
    tot = sum(last, parts)
    say(@sprintf("\n=== profile  L=%d  N=%d  2N=%d  (init %.2f s) ===", L, N, 2N, t_init))
    say(@sprintf("MC sweep (K0-TCA, Lc=%d)            %10.3f s   [%d updates, %.3f ms per cluster action]",
                 CFG["TClustersize"], t_sweep, N, 1e3 * t_cl / ncl))
    say(@sprintf("MC sweep, phi channel on            %10.3f s   [%d updates]", t_sweep_phi, 2N))
    say(@sprintf("%d sweeps (= measuring_interval)    %10.3f s", mi, mi * t_sweep))
    say(@sprintf("make_measurement_K! end-to-end      %10.3f s   -> measurement / interval = %.2f", t_meas, t_meas / (mi * t_sweep)))
    for (k, v) in sort(parts; by=last, rev=true)
        say(@sprintf("   %-36s %9.3f s  %5.1f%%", k, v, 100v / tot))
    end
    say(@sprintf("   %-36s %9.3f s", "(sum of parts)", tot))
    say(@sprintf("checkpoint write %.3f s, %.1f MB", t_ckpt, ckpt_mb))
    say(@sprintf("projected campaign point (%d thermal + %d x (%d sweeps + 1 meas)): %.1f days, phi on: %.1f days",
                 nth, nme, mi, proj, proj_phi))
    RESULTS["profile"] = Dict(
        "L" => L, "N" => N, "t_init" => t_init, "t_sweep" => t_sweep, "t_sweeps" => t_sweeps,
        "t_sweep_phi" => t_sweep_phi, "t_cluster_action" => t_cl / ncl,
        "t_measurement" => t_meas, "parts" => Dict(parts), "parts_order" => first.(parts),
        "t_checkpoint" => t_ckpt, "checkpoint_MB" => ckpt_mb,
        "measuring_interval" => mi, "num_thermal" => nth, "num_measure" => nme,
        "projected_days" => proj, "projected_days_phi" => proj_phi)
    save_results()
    return nothing
end

# ── section: proto ──────────────────────────────────────────────────────────
function section_proto(L; verbose=true)
    data = Lattice_Data(make_sd(L))
    for _ in 1:2; sweep!(data); end
    invalidate_global_diag!(data); ensure_global_diag_synced!(data)
    get_FD_dist!(data); compute_delta_analitic_continuation!(data)
    N = data.sim_data.N
    q = data.const_struct.q_list_CC
    nq = length(q)
    td = data.temporary_data
    R = Dict{String,Any}()
    rec(name, tref, tfast, err, bytes; extra...) = (R[name] = Dict{String,Any}("ref" => tref, "fast" => tfast,
        "speedup" => tref / tfast, "relerr" => err, "alloc_MB_per_call" => bytes / 2^20,
        (string(k) => v for (k, v) in extra)...))

    # All prototype buffers: built once per run (in production: in TemporaryData),
    # outside the timed regions. Every timed call below should allocate ~0 bytes.
    t_work = @elapsed (pw = ProtoWork(data))
    work_MB = Base.summarysize(pw) / 2^20
    Gbuf = [zeros(ComplexF64, 2N, 2N) for _ in 1:nq]   # stands in for G_matrix_vec_x/_y; one direction at a time
    labout = zeros(ComplexF64, 4, nq)
    ds = data.sim_data.delta_values_fixed; ws = data.sim_data.w_conductivity_range
    fast4 = Array{NTuple{4,ComplexF64}}(undef, length(ds), length(ws))

    # current vertices: reference first (it fills td.G_matrix_vec_x / _y)
    t_ref = @elapsed (make_G_matrix_q_vec(data, q; G_out=td.G_matrix_vec_x, direction="x");
                      make_G_matrix_q_vec(data, q; G_out=td.G_matrix_vec_y, direction="y"))
    Gx = td.G_matrix_vec_x; Gy = td.G_matrix_vec_y
    t_g = 0.0; e_g = 0.0; b_g = 0; t_rb = 0.0; e_rb = 0.0; b_rb = 0
    for (dir, Gref) in (("x", Gx), ("y", Gy))
        st = @timed proto_current_vertex_gemm!(Gbuf, pw.vertex, data, q, dir)
        t_g += st.time; b_g += st.bytes
        e_g = max(e_g, maximum(relerr(Gbuf[i], Gref[i]) for i in 1:nq))
        st = @timed proto_current_vertex_rowblock!(Gbuf, pw.vertex, data, q, dir)
        t_rb += st.time; b_rb += st.bytes
        e_rb = max(e_rb, maximum(relerr(Gbuf[i], Gref[i]) for i in 1:nq))
    end
    Gbuf = nothing; GC.gc()
    rec("G(q) per-q ZGEMM", t_ref, t_g, e_g, b_g; n_q=nq)
    rec("G(q) row-block", t_ref, t_rb, e_rb, b_rb; n_q=nq)

    ref = zeros(ComplexF64, 4, nq)
    t_ref = @elapsed for (k, (a, b)) in enumerate(((Gx, Gx), (Gx, Gy), (Gy, Gx), (Gy, Gy)))
        current_correlation_from_G!(data, a, b, td.Lxx_temp; q_length=nq)
        ref[k, :] .= td.Lxx_temp[1:nq]
    end
    st = @timed proto_Lab_q!(labout, pw.lab, data, Gx, Gy, nq)
    rec("L_ab(q) x4", t_ref, st.time, relerr(labout, ref), st.bytes)

    t_ref = @elapsed (ref4 = [current_correlation_K_4dir(data, w; pre_Gx=Gx, pre_Gy=Gy, δ=δ) for δ in ds, w in ws])
    st = @timed proto_fixed_delta!(fast4, pw.fd, data, Gx[1], Gy[1])
    err = maximum(maximum(abs.(collect(fast4[i]) .- collect(ref4[i]))) / max(maximum(abs.(collect(ref4[i]))), eps())
                  for i in eachindex(ref4))
    rec("fixed-delta L(w)", t_ref, st.time, err, st.bytes; pairs_kept=st.value[2], n_delta=length(ds), n_w=length(ws))

    t_ref = @elapsed (r = compute_D_dagger_D_K(data))
    st = @timed proto_DdD!(pw.ddd, data)
    rec("DdD", t_ref, st.time, abs(st.value - r) / abs(r), st.bytes)

    w_list = data.sim_data.w_list; dmax = td.delta_max
    kxp, kyp = data.const_struct.SF_k_path
    t_ref = @elapsed begin
        Aref = [begin
                    kg = data.const_struct.k_general_idx[kxp[l], kyp[l]]
                    q_ix, K_ix = data.const_struct.k_general_qK_idx[kg]
                    spectral_functionV2(data, K_ix, q_ix, w; delt=dmax)
                end for l in eachindex(kxp), w in w_list]
        BZref = copy(spectral_function_fullV2(data, 0.0; delt=dmax))
        LDref = reduce(hcat, [copy(compute_full_LDOS(data, w; delt=dmax)) for w in w_list])
    end
    st = @timed proto_spectral!(pw.spec, data)
    Af, BZf, LDf = st.value
    err = max(relerr(Af, Aref), relerr(LDf, LDref), BZf === nothing ? 0.0 : relerr(BZf, BZref[1:length(BZf)]))
    rec("A(k,w)+BZ+LDOS", t_ref, st.time, err, st.bytes)
    R["work"] = Dict("MB" => work_MB, "t_build" => t_work)
    pw = nothing; data = nothing; GC.gc()
    verbose || return nothing

    say(@sprintf("\n=== proto  L=%d  N=%d  n_q=%d  (BLAS threads %d, Julia threads %d) ===",
                 L, N, nq, BLAS.get_num_threads(), Threads.nthreads()))
    say(@sprintf("   preallocated work: %.1f MB, built in %.3f s (once per run)", work_MB, t_work))
    for k in ("G(q) per-q ZGEMM", "G(q) row-block", "L_ab(q) x4", "fixed-delta L(w)", "DdD", "A(k,w)+BZ+LDOS")
        d = R[k]
        say(@sprintf("   %-18s ref %9.3f s | fast %9.3f s | x%6.1f | relerr %.1e | alloc %8.3f MB/call",
                     k, d["ref"], d["fast"], d["speedup"], d["relerr"], d["alloc_MB_per_call"]))
    end
    say(@sprintf("   fixed-delta: %.0f%% of pairs kept", 100R["fixed-delta L(w)"]["pairs_kept"]))
    RESULTS["proto"] = R
    save_results()
    return nothing
end

# ── section: integ ──────────────────────────────────────────────────────────
# The main code after integration. Both paths measure the same state, each
# starting from a copy of the same MeasureData (no second lattice in memory).
function measure_data_diff(a::MeasureData, b::MeasureData)
    md(x::Number, y::Number) = (Float64(abs(x - y)), Float64(max(abs(x), abs(y))))
    function md(x::AbstractArray, y::AbstractArray)
        d = 0.0; s = 0.0
        for (u, v) in zip(x, y)
            du, su = md(u, v); d = max(d, du); s = max(s, su)
        end
        return d, s
    end
    out = Dict{String,Any}()
    for fn in fieldnames(MeasureData)
        d, s = md(getfield(a, fn), getfield(b, fn))
        out[string(fn)] = Dict("relerr" => d / max(s, 1e-300), "absdiff" => d, "scale" => s)
    end
    return out
end

function section_integ(L; verbose=true)
    data = Lattice_Data(make_sd(L))
    for _ in 1:2; sweep!(data); end
    t_buf = @elapsed ensure_measurement_buffers!(data)   # what MonteCarloSimulation does before thermalizing
    td = data.temporary_data
    work = td.meas_work
    N = data.sim_data.N
    m0 = deepcopy(data.measurement_data)

    invalidate_global_diag!(data)
    st_ref = @timed make_measurement_K!(data, 1; fast=false)
    m_ref = data.measurement_data
    data.measurement_data = deepcopy(m0)
    invalidate_global_diag!(data)
    st_fast = @timed make_measurement_K!(data, 1; fast=true)
    diffs = measure_data_diff(m_ref, data.measurement_data)
    m_ref = nothing
    # worst relative difference among fields that are not ~0, and worst absolute one
    big = [(k, v["relerr"]) for (k, v) in diffs if v["scale"] > 1e-8]
    worst_rel = isempty(big) ? ("none", 0.0) : big[argmax(last.(big))]
    worst_abs = maximum(v["absdiff"] for v in values(diffs))

    # each fast kernel alone (diag synced, Fermi factors set by the measurement above)
    q = data.const_struct.q_list_CC
    m = data.measurement_data
    K = Dict{String,Any}()
    rec(name, st) = (K[name] = Dict("t" => st.time, "bytes" => st.bytes))
    rec("G(q) row-block", @timed (current_vertices_fast!(td.G_matrix_vec_x, work, data, q, "x"; method=:rowblock);
                                  current_vertices_fast!(td.G_matrix_vec_y, work, data, q, "y"; method=:rowblock)))
    rec("L_ab(q) x4", @timed current_correlation_q_fast!(work.Lab_q, work, data, td.G_matrix_vec_x, td.G_matrix_vec_y, length(q)))
    rec("fixed-delta L(w)", @timed measure_fixed_delta_fast!(data, m, 1, work, td.G_matrix_vec_x[1], td.G_matrix_vec_y[1]))
    rec("DdD", @timed compute_D_dagger_D_fast!(work, data))
    rec("A(k,w)+BZ+LDOS", @timed measure_spectral_fast!(data, m, 1, work))
    rec("G(q) per-q GEMM", @timed (current_vertices_fast!(td.G_matrix_vec_x, work, data, q, "x"; method=:gemm);
                                   current_vertices_fast!(td.G_matrix_vec_y, work, data, q, "y"; method=:gemm)))
    invalidate_global_diag!(data)
    st_n = @timed make_measurement_n_only!(data, 1)

    work_MB = Base.summarysize(work) / 2^20
    G_MB = (Base.summarysize(td.G_matrix_vec_x) + Base.summarysize(td.G_matrix_vec_y)) / 2^20
    est = estimate_memory_bytes(data.sim_data)
    lim, lim_src = job_memory_limit()
    maxrss = Sys.maxrss()
    data = nothing; m0 = nothing; GC.gc()
    verbose || return nothing

    say(@sprintf("\n=== integ  L=%d  N=%d  n_q=%d  (BLAS threads %d, Julia threads %d) ===",
                 L, N, length(q), BLAS.get_num_threads(), Threads.nthreads()))
    say(@sprintf("   make_measurement_K!  reference %9.3f s (%8.2f MB alloc) | fast %8.3f s (%7.2f MB alloc) | x%.1f",
                 st_ref.time, st_ref.bytes / 2^20, st_fast.time, st_fast.bytes / 2^20, st_ref.time / st_fast.time))
    say(@sprintf("   worst MeasureData difference: relative %.1e (%s), absolute %.1e", worst_rel[2], worst_rel[1], worst_abs))
    for k in ("G(q) row-block", "G(q) per-q GEMM", "L_ab(q) x4", "fixed-delta L(w)", "DdD", "A(k,w)+BZ+LDOS")
        say(@sprintf("   %-18s %9.3f s  %8d bytes", k, K[k]["t"], K[k]["bytes"]))
    end
    say(@sprintf("   n_only measurement %.4f s, %d bytes", st_n.time, st_n.bytes))
    say(@sprintf("   buffers: fast-kernel work %.1f MB, G_matrix_vec_x/_y %.1f MB (allocated in %.2f s)", work_MB, G_MB, t_buf))
    say(@sprintf("   memory: estimate %.2f GB (runtime %.2f + arrays), peak RSS so far %.2f GB, limit %.2f GB (%s)",
                 est.total / 2^30, est.runtime / 2^30, maxrss / 2^30, lim / 2^30, lim_src))
    RESULTS["integ"] = Dict("L" => L, "N" => N, "n_q" => length(q),
        "t_ref" => st_ref.time, "t_fast" => st_fast.time, "speedup" => st_ref.time / st_fast.time,
        "bytes_ref" => st_ref.bytes, "bytes_fast" => st_fast.bytes,
        "worst_relerr" => worst_rel[2], "worst_relerr_field" => worst_rel[1], "worst_absdiff" => worst_abs,
        "fields" => diffs, "kernels" => K, "t_n_only" => st_n.time, "n_only_bytes" => st_n.bytes,
        "work_MB" => work_MB, "G_MB" => G_MB, "t_buffers" => t_buf,
        "estimate_GB" => est.total / 2^30, "estimate_runtime_GB" => est.runtime / 2^30,
        "maxrss_GB" => maxrss / 2^30, "limit_GB" => lim / 2^30, "limit_source" => lim_src)
    save_results()
    return nothing
end

# ── section: occ ────────────────────────────────────────────────────────────
function section_occ(L; verbose=true)
    d = Lattice_Data(make_sd(L; occ=true))
    sweep!(d)
    ensure_measurement_buffers!(d)          # buffer allocation is not part of a measurement
    # what the mu-search measured before n_only (reference path), the fast full
    # measurement, and what it measures now (make_measurement_n_only!)
    invalidate_global_diag!(d); t_full = @elapsed make_measurement_K!(d, 1; fast=false)
    invalidate_global_diag!(d); t_full_fast = @elapsed make_measurement_K!(d, 1; fast=true)
    invalidate_global_diag!(d); st = @timed make_measurement_n_only!(d, 1)
    t_n = st.time; b_n = st.bytes
    t_sweep = @elapsed sweep!(d)
    d = nothing; GC.gc()
    verbose || return nothing
    mio = Int(CFG["measuring_interval_occ"])
    say(@sprintf("\n=== occ  L=%d ===", L))
    say(@sprintf("occupation phase: make_measurement_K! %.3f s (fast path %.3f s) | n_only %.3f s (x%.0f, %.3f MB allocated) | %d sweeps %.3f s",
                 t_full, t_full_fast, t_n, t_full / t_n, b_n / 2^20, mio, mio * t_sweep))
    say(@sprintf("   share of the phase spent in measurements with the full measurement: %.0f%%, with n_only: %.0f%%",
                 100t_full / (t_full + mio * t_sweep), 100t_n / (t_n + mio * t_sweep)))
    RESULTS["occ"] = Dict("L" => L, "t_measurement" => t_full, "t_measurement_fast" => t_full_fast,
                          "t_n_only" => t_n, "n_only_alloc_MB" => b_n / 2^20,
                          "t_sweep" => t_sweep, "measuring_interval_occ" => mio)
    save_results()
    return nothing
end

# ── section: accept ─────────────────────────────────────────────────────────
function sweep_accept!(d, fields)
    N = d.sim_data.N
    acc = Dict(f => 0 for f in fields); tried = Dict(f => 0 for f in fields)
    for _ in 1:(N * length(fields))
        site = rand(1:N)
        field = fields[rand(1:length(fields))]
        dS, sug, _, _ = localized_update(d, site, "K0-TCA", "s-wave", field)
        dS === nothing && continue
        tried[field] += 1
        if check_acceptence(dS)
            acc[field] += 1
            invalidate_global_diag!(d)
        else
            revert_pairing_field!(d, site, "s-wave", field, sug)
        end
    end
    return acc, tried
end

function section_accept(; verbose=true)
    La = Int(CFG["accept_L"])
    nburn = Int(CFG["accept_burn"]); nmeas = Int(CFG["accept_sweeps"])
    rows = Any[]
    verbose && say(@sprintf("\n=== accept  L=%d  (%d burn-in + %d counted sweeps) ===", La, nburn, nmeas))
    for p in CFG["accept_points"]
        U, beta, bphi = Float64(p[1]), Float64(p[2]), Float64(p[3])
        d = Lattice_Data(make_sd(La; U=U, beta=beta, beta_phi=bphi))
        fields = bphi == 0.0 ? [:Delta_s] : [:Delta_s, :phi]
        for _ in 1:nburn; sweep_accept!(d, fields); end
        acc = Dict(f => 0 for f in fields); tried = Dict(f => 0 for f in fields)
        for _ in 1:nmeas
            a, t = sweep_accept!(d, fields)
            for f in fields; acc[f] += a[f]; tried[f] += t[f]; end
        end
        step = sqrt(cov(d.const_struct.dist)[1, 1])
        rate = Dict(string(f) => acc[f] / max(tried[f], 1) for f in fields)
        absD = mean(abs.(d.current_state.Delta_s)); absphi = mean(abs.(d.current_state.phi))
        push!(rows, Dict("U" => U, "beta" => beta, "beta_phi" => bphi, "step_std" => step,
                         "acceptance" => rate, "mean_abs_Delta" => absD, "mean_abs_phi" => absphi))
        if verbose
            say(@sprintf("U=%5.1f beta=%6.2f beta_phi=%.2f  step std %.3f  acc(Delta) %5.1f%%%s   <|Delta|> %.3f",
                         U, beta, bphi, step, 100rate["Delta_s"],
                         bphi == 0.0 ? "" : @sprintf("  acc(phi) %5.1f%%  <|phi|> %.3f", 100rate["phi"], absphi), absD))
        end
    end
    verbose || return nothing
    RESULTS["accept"] = Dict("L" => La, "points" => rows)
    save_results()
    return nothing
end

# ── section: threads ────────────────────────────────────────────────────────
function section_threads(L; verbose=true)
    rows = Any[]
    d = Lattice_Data(make_sd(L))
    for nt in sort(unique(filter(<=(CORES), [1, 2, 4, CORES])))
        BLAS.set_num_threads(nt)
        MCMF_action_TCluster_K0(d, 1)
        tc = @elapsed for _ in 1:300; MCMF_action_TCluster_K0(d, rand(1:d.sim_data.N)); end
        invalidate_global_diag!(d); ensure_global_diag_synced!(d)
        invalidate_global_diag!(d); tf = @elapsed ensure_global_diag_synced!(d)
        push!(rows, Dict("blas_threads" => nt, "t_cluster_action" => tc / 300, "t_full_diag" => tf))
        verbose && say(@sprintf("   BLAS threads %d: TCA cluster action %.3f ms | full diag 2N=%d %.3f s",
                                nt, 1e3tc / 300, 2 * d.sim_data.N, tf))
    end
    BLAS.set_num_threads(CORES)
    d = nothing; GC.gc()
    verbose || return nothing
    RESULTS["threads"] = Dict("L" => L, "rows" => rows)
    save_results()
    return nothing
end

# ── main ────────────────────────────────────────────────────────────────────
function main()
    L = Int(CFG["L"])
    sections = String.(CFG["sections"])
    ci = Sys.cpu_info()
    RESULTS["meta"] = Dict(
        "cluster" => CFG["cluster"], "L" => L, "cores" => CORES, "sections" => sections,
        "hostname" => gethostname(), "cpu_model" => isempty(ci) ? "" : ci[1].model,
        "cpu_threads" => Sys.CPU_THREADS, "total_memory_GB" => Sys.total_memory() / 2^30,
        "loadavg_start" => collect(Sys.loadavg()),
        "julia_version" => string(VERSION), "blas" => string(BLAS.get_config()),
        "blas_threads" => BLAS.get_num_threads(), "julia_threads" => Threads.nthreads(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""), "slurm_nodelist" => get(ENV, "SLURM_JOB_NODELIST", ""),
        "start" => string(now()), "campaign" => Dict(k => CFG[k] for k in
            ("U", "beta", "mu", "TClustersize", "num_thermal", "num_measure", "measuring_interval")))
    save_results()
    say("="^90)
    say(@sprintf("bench_profile  cluster=%s  L=%d  cores=%d  sections=%s", CFG["cluster"], L, CORES, join(sections, ",")))
    say(@sprintf("host %s | %s | %d cpu threads | %.0f GB | load %s", gethostname(),
                 RESULTS["meta"]["cpu_model"], Sys.CPU_THREADS, Sys.total_memory() / 2^30,
                 join(round.(Sys.loadavg(), digits=2), " ")))
    say(@sprintf("julia %s | BLAS threads %d | Julia threads %d | %s", VERSION, BLAS.get_num_threads(),
                 Threads.nthreads(), string(BLAS.get_config())))
    say("start " * string(now()))

    # compile everything on a small lattice first, so no section times the JIT
    t_warm = @elapsed begin
        "profile" in sections && section_profile(10; verbose=false)
        "proto"   in sections && section_proto(10; verbose=false)
        "integ"   in sections && section_integ(10; verbose=false)
        "occ"     in sections && section_occ(10; verbose=false)
        "threads" in sections && section_threads(10; verbose=false)
    end
    say(@sprintf("warm-up (compilation) %.1f s", t_warm))

    for s in sections
        t = @elapsed begin
            s == "profile" && section_profile(L)
            s == "proto"   && section_proto(L)
            s == "integ"   && section_integ(L)
            s == "occ"     && section_occ(L)
            s == "accept"  && section_accept()
            s == "threads" && (say(@sprintf("\n=== threads  L=%d ===", L)); section_threads(L))
        end
        say(@sprintf("[section %s done in %.1f s]", s, t))
    end
    RESULTS["meta"]["end"] = string(now())
    save_results()
    say("done " * string(now()))
end

main()
