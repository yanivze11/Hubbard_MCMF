# ============================================================================
# test_measurement_kernels.jl — the fast measurement kernels of
# make_measurement_K!(...; fast=true) against the reference kernels (fast=false).
#
#   julia -t 1   tests/test_measurement_kernels.jl
#   julia -t 2,1 tests/test_measurement_kernels.jl     (worker + interactive thread, the Julia 1.12 layout)
#   julia -t 4   tests/test_measurement_kernels.jl
#   On the clusters, as SLURM jobs (-t 1, -t 4, -t 4,0):
#   julia pre_pre_run.jl tests_pre_run.jl
#
# Sections
#   1. kernel equality         each fast kernel vs its reference function, same state
#   2. whole measurement       every MeasureData field, 3 consecutive measurements
#                              (running means of A, BZ, LDOS included), old vs fast
#   3. MC trajectory           MeasurmentSteps! with fast vs reference measurements
#                              from the same state and RNG seed: identical trajectory
#   4. allocations             bytes per make_measurement_K! and per fast kernel
#   5. fallbacks               SC_L = 2 (reference path), off-axis q (per-q GEMM)
#   6. n_only                  same n as the full measurement, guard, buffers never
#                              allocated, speed of the mu-search measurements,
#                              a short occupation convergence
#   7. memory estimate         estimate vs actual buffer sizes, SLURM limit parsing,
#                              error and MC_SKIP_MEMORY_CHECK override
# ============================================================================
using Test, Random, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "occupation_convergence.jl"))   # includes MC_Hubburd.jl
BLAS.set_num_threads(1)

const W_LIST = collect(-3.0:0.25:3.0)          # contains 0.0 -> BZ map is measured
const W_COND = [0.0, 0.05, 0.3, 1.0, 2.5]
const DELTAS = [0.02, 0.1, 0.5]
const DW = [0.05, 0.2]
const KERNEL_TOL = 1e-12                        # relative, per kernel
const FIELD_RTOL = 1e-10                        # relative, per MeasureData field
const FIELD_ATOL = 1e-12                        # absolute floor for fields that are ~0 (e.g. Lxy at B=0)

function make_case(; Lx, Ly=Lx, diag="K0", Lc=4, pairing="s-wave", B=0.0, gauge="LandauX", V=0.0,
                   alpha=1.0, beta_phi=0.0, Q=[0.0, 0.0], mu_disorder="none", f=0.0, SC_L=1,
                   compute_jP=false, hist=false, num_measure=4, measuring_interval=1,
                   w_list=W_LIST, w_cond=W_COND, mu=-0.3, beta=8.0, U=3.0)
    SimData(; Lx=Lx, Ly=Ly, SC_L=SC_L, U=U, t=1.0, V=V, seed=11, mu=mu, beta=beta,
        num_measure=num_measure, num_thermal=0, measuring_interval=measuring_interval, saving_interval=5,
        w_list=copy(w_list), guess="uniform", U_disorder="uniform", diag_method=diag, pairing_symmetry=pairing,
        w_conductivity_range=copy(w_cond), delta_values_fixed=copy(DELTAS), delta_values_stiffness=Float64[],
        dw_values=copy(DW), TCluster_size_x=Lc, TCluster_size_y=Lc, B=B, gauge_choice=gauge,
        compute_jP=compute_jP, disorder_seed=3, alpha=alpha, beta_phi=beta_phi, Q_density=copy(Q),
        snapshot_interval=0, phi_hist_nbins=hist ? 20 : 0, Delta_hist_abs_nbins=hist ? 10 : 0,
        Delta_hist_arg_nbins=hist ? 8 : 0, mu_disorder=mu_disorder, mu_disorder_f=f)
end

sweep!(d) = MC_stepV2!(d; method=d.sim_data.diag_method, loop_random=true, pairing=d.sim_data.pairing_symmetry)

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-300)

const CASES = [
    ("L6 K0 s-wave",                 () -> make_case(Lx=6)),
    ("L8 K0-TCA s-wave",             () -> make_case(Lx=8, diag="K0-TCA")),
    ("L12 K0-TCA s-wave",            () -> make_case(Lx=12, diag="K0-TCA")),
    ("8x12 K0 s-wave",               () -> make_case(Lx=8, Ly=12)),
    ("L7 K0 (odd L)",                () -> make_case(Lx=7)),
    ("L8 B Mixed",                   () -> make_case(Lx=8, B=2π/64, gauge="Mixed")),
    ("8x12 B LandauX K0-TCA",        () -> make_case(Lx=8, Ly=12, diag="K0-TCA", B=2π/12, gauge="LandauX")),
    ("L8 V=1 disorder",              () -> make_case(Lx=8, V=1.0)),
    ("L8 beta_phi=0.5 Q=(pi,pi)",    () -> make_case(Lx=8, alpha=0.5, beta_phi=0.5, Q=[Float64(π), Float64(π)])),
    ("L8 mu_disorder coupled",       () -> make_case(Lx=8, mu_disorder="coupled", f=0.3)),
    ("L8 d-wave",                    () -> make_case(Lx=8, pairing="d-wave")),
    ("L8 jP + histograms",           () -> make_case(Lx=8, compute_jP=true, hist=true)),
]

# ── 1. kernel equality ──────────────────────────────────────────────────────
function kernel_errors(data)
    ensure_global_diag_synced!(data)
    get_FD_dist!(data)
    compute_delta_analitic_continuation!(data)
    ensure_measurement_buffers!(data)
    work = measurement_work!(data)
    td = data.temporary_data
    N = data.sim_data.N
    q = data.const_struct.q_list_CC
    nq = length(q)
    errs = Dict{String,Float64}()

    # current vertices: reference into private buffers, both fast methods
    Gref = Dict(dir => [zeros(ComplexF64, 2N, 2N) for _ in 1:nq] for dir in ("x", "y"))
    for dir in ("x", "y")
        make_G_matrix_q_vec(data, q; G_out=Gref[dir], direction=dir)
    end
    Gf = [zeros(ComplexF64, 2N, 2N) for _ in 1:nq]
    for dir in ("x", "y"), method in (:rowblock, :gemm, :auto)
        current_vertices_fast!(Gf, work, data, q, dir; method=method)
        errs["G^$dir(q) $method"] = maximum(relerr(Gf[k], Gref[dir][k]) for k in 1:nq)
    end

    # off-axis q-list: :auto must use the per-q GEMM, the row-block must refuse it.
    # (non-negative components only: q_list_signature cannot hash a negative q,
    # a pre-existing limitation of get_FT_table_for_q_vecs!)
    Lx, Ly = data.sim_data.Lx, data.sim_data.Ly
    qoff = [[0.0, 0.0], [2π/Lx, 2π/Ly], [4π/Lx, 2π/Ly]]
    Goff_ref = [zeros(ComplexF64, 2N, 2N) for _ in 1:3]
    Goff = [zeros(ComplexF64, 2N, 2N) for _ in 1:3]
    make_G_matrix_q_vec(data, qoff; G_out=Goff_ref, direction="y")
    current_vertices_fast!(Goff, work, data, qoff, "y")
    errs["G^y(q) off-axis auto"] = maximum(relerr(Goff[k], Goff_ref[k]) for k in 1:3)
    rowblock_refuses = try
        current_vertices_fast!(Goff, work, data, qoff, "y"; method=:rowblock); false
    catch err
        err isa ErrorException
    end
    errs["row-block refuses off-axis q (0 = yes)"] = rowblock_refuses ? 0.0 : 1.0

    # L_ab(q, w=0)
    Gx = Gref["x"]; Gy = Gref["y"]
    ref = zeros(ComplexF64, 4, nq)
    for (k, (a, b)) in enumerate(((Gx, Gx), (Gx, Gy), (Gy, Gx), (Gy, Gy)))
        current_correlation_from_G!(data, a, b, td.Lxx_temp; q_length=nq)
        ref[k, :] .= td.Lxx_temp[1:nq]
    end
    out = zeros(ComplexF64, 4, nq)
    current_correlation_q_fast!(out, work, data, Gx, Gy, nq)
    errs["L_ab(q)"] = relerr(out, ref)

    # fixed-delta grid
    ref4 = [current_correlation_K_4dir(data, w; pre_Gx=Gx, pre_Gy=Gy, δ=δ) for δ in DELTAS, w in W_COND]
    m = Initial_measurments(data.sim_data)
    measure_fixed_delta_fast!(data, m, 1, work, Gx[1], Gy[1])
    fast4 = [(m.Lxx_w_fixed[1][d][j], m.Lxy_w_fixed[1][d][j], m.Lyx_w_fixed[1][d][j], m.Lyy_w_fixed[1][d][j])
             for d in eachindex(DELTAS), j in eachindex(W_COND)]
    scale = maximum(maximum(abs.(collect(r))) for r in ref4)
    errs["fixed-delta L(w)"] = maximum(maximum(abs.(collect(fast4[k]) .- collect(ref4[k]))) for k in eachindex(ref4)) / scale

    # DdD
    r = compute_D_dagger_D_K(data)
    errs["DdD"] = abs(compute_D_dagger_D_fast!(work, data) - r) / abs(r)

    # spectral function on the k-path, BZ map, LDOS
    dmax = td.delta_max
    cs = data.const_struct
    kxp, kyp = cs.SF_k_path
    Aref = [begin
                q_ix, K_ix = cs.k_general_qK_idx[cs.k_general_idx[kxp[l], kyp[l]]]
                spectral_functionV2(data, K_ix, q_ix, w; delt=dmax)
            end for l in eachindex(kxp), w in W_LIST]
    BZref = copy(spectral_function_fullV2(data, 0.0; delt=dmax))
    LDref = reduce(hcat, [copy(compute_full_LDOS(data, w; delt=dmax)) for w in W_LIST])
    spectral_fast!(work, data)
    errs["A(k,w) path"] = relerr(work.A_path, Aref)
    errs["BZ map"] = relerr(work.BZ, BZref)
    errs["LDOS"] = relerr(work.LDOS, LDref)
    return errs
end

# ── 2. whole-measurement equality ───────────────────────────────────────────
maxdiff_and_scale(x::Number, y::Number) = (Float64(abs(x - y)), Float64(max(abs(x), abs(y))))
function maxdiff_and_scale(x::AbstractArray, y::AbstractArray)
    size(x) == size(y) || error("shape mismatch $(size(x)) vs $(size(y))")
    d = 0.0; s = 0.0
    for (u, v) in zip(x, y)
        du, su = maxdiff_and_scale(u, v)
        d = max(d, du); s = max(s, su)
    end
    return d, s
end

function compare_measure_data(a::MeasureData, b::MeasureData)
    rows = Tuple{Symbol,Float64,Float64,Bool,Float64}[]  # field, relerr, absdiff, same type, scale
    for fn in fieldnames(MeasureData)
        x = getfield(a, fn); y = getfield(b, fn)
        d, s = maxdiff_and_scale(x, y)
        push!(rows, (fn, d / max(s, 1e-300), d, typeof(x) == typeof(y), s))
    end
    return rows
end
field_ok(row) = row[4] && (row[2] <= FIELD_RTOL || row[3] <= FIELD_ATOL)
# worst relative difference among fields that are not ~0 (scale > 1e-8), and worst absolute one
function worst_fields(rows)
    big = [r for r in rows if r[5] > 1e-8]
    wr = isempty(big) ? rows[1] : big[argmax([r[2] for r in big])]
    wa = rows[argmax([r[3] for r in rows])]
    return wr, wa
end

function whole_measurement(sd; nmeas=3, nsweep=1)
    data = Lattice_Data(sd)
    data.sim_data.SC_L == 1 && sweep!(data)
    m_ref = deepcopy(data.measurement_data)
    m_fast = deepcopy(data.measurement_data)
    hist_ok = true
    for i in 1:nmeas
        if data.sim_data.SC_L == 1
            for _ in 1:nsweep
                sweep!(data)
            end
        end
        h0 =copy(data.current_state.phi_hist); H0 = copy(data.current_state.Delta_hist)
        data.measurement_data = m_ref
        make_measurement_K!(data, i; fast=false)
        h1 = copy(data.current_state.phi_hist); H1 = copy(data.current_state.Delta_hist)
        data.measurement_data = m_fast
        make_measurement_K!(data, i; fast=true)
        # the histograms are lifetime accumulators: both paths add the same counts
        hist_ok &= (data.current_state.phi_hist .- h1 == h1 .- h0) && (data.current_state.Delta_hist .- H1 == H1 .- H0)
    end
    return compare_measure_data(m_ref, m_fast), hist_ok, data
end

# ── 3. MC trajectory: fast vs reference measurements inside MeasurmentSteps! ──
function trajectory_check(sd)
    d0 = Lattice_Data(sd)
    dA = deepcopy(d0); dB = deepcopy(d0)
    Random.seed!(2024); MeasurmentSteps!(dA; method=sd.diag_method, savedata=false, fast_measurement=false)
    Random.seed!(2024); MeasurmentSteps!(dB; method=sd.diag_method, savedata=false, fast_measurement=true)
    same_traj = dA.current_state.Delta_s == dB.current_state.Delta_s && dA.current_state.phi == dB.current_state.phi &&
                dA.current_state.step_count == dB.current_state.step_count && dA.current_state.S == dB.current_state.S
    rows = compare_measure_data(dA.measurement_data, dB.measurement_data)
    return same_traj, rows, dA.current_state.num_current_measurments
end

# ── 4. allocations ──────────────────────────────────────────────────────────
function allocation_check(sd)
    data = Lattice_Data(sd)
    sweep!(data)
    make_measurement_K!(data, 1; fast=false)        # warm-up: compiles, allocates G buffers
    make_measurement_K!(data, 1; fast=true)         # warm-up: builds meas_work
    make_measurement_K!(data, 1; fast=true)
    a_old = @allocated make_measurement_K!(data, 1; fast=false)
    a_new = @allocated make_measurement_K!(data, 1; fast=true)
    work = data.temporary_data.meas_work
    td = data.temporary_data
    q = data.const_struct.q_list_CC
    m = data.measurement_data
    k = Dict{String,Int}()
    for _ in 1:2   # second pass is the steady state
        k["G^x(q) row-block"] = @allocated current_vertices_fast!(td.G_matrix_vec_x, work, data, q, "x"; method=:rowblock)
        k["G^y(q) per-q GEMM"] = @allocated current_vertices_fast!(td.G_matrix_vec_y, work, data, q, "y"; method=:gemm)
        k["L_ab(q)"] = @allocated current_correlation_q_fast!(work.Lab_q, work, data, td.G_matrix_vec_x, td.G_matrix_vec_y, length(q))
        k["fixed-delta"] = @allocated measure_fixed_delta_fast!(data, m, 1, work, td.G_matrix_vec_x[1], td.G_matrix_vec_y[1])
        k["DdD"] = @allocated compute_D_dagger_D_fast!(work, data)
        k["spectral+LDOS+BZ"] = @allocated measure_spectral_fast!(data, m, 1, work)
        k["n_only"] = @allocated make_measurement_n_only!(data, 1)
    end
    return a_old, a_new, k
end

# ── 7. memory estimate helpers ──────────────────────────────────────────────
lattice_arrays_bytes(d) = Base.summarysize(d) - Base.summarysize(d.temporary_data.G_matrix_vec_x) -
                          Base.summarysize(d.temporary_data.G_matrix_vec_y) - Base.summarysize(d.temporary_data.meas_work) -
                          Base.summarysize(d.temporary_data.sus_weighted_up) - Base.summarysize(d.temporary_data.sus_weighted_dn_conj)

# ════════════════════════════════════════════════════════════════════════════
println("="^100)
@printf("test_measurement_kernels  julia %s  threads %d (+%d interactive)  BLAS threads %d\n",
        VERSION, Threads.nthreads(), Threads.nthreads(:interactive), BLAS.get_num_threads())
println("="^100)

summary_rows = Any[]

@testset "fast measurement kernels" begin

    @testset "1+2. kernels and whole measurement: $name" for (name, mk) in CASES
        sd = mk()
        data = Lattice_Data(sd)
        sweep!(data)
        errs = kernel_errors(data)
        for (k, e) in errs
            @test e <= KERNEL_TOL
        end
        rows, hist_ok, _ = whole_measurement(sd)
        @test all(field_ok, rows)
        @test hist_ok
        bad = [r for r in rows if !field_ok(r)]
        isempty(bad) || println("  FAILING FIELDS in $name: ", bad)
        wr, wa = worst_fields(rows)
        push!(summary_rows, (name, maximum(values(errs)), wr[1], wr[2], wa[1], wa[3]))
        @printf("%-30s kernels max relerr %.1e | worst field relerr %-16s %.1e | worst abs %-16s %.1e\n",
                name, maximum(values(errs)), wr[1], wr[2], wa[1], wa[3])
        for k in sort(collect(keys(errs)))
            @printf("      %-40s %.2e\n", k, errs[k])
        end
    end

    @testset "work buffers share memory as designed" begin
        data = Lattice_Data(make_case(Lx=6))
        ensure_measurement_buffers!(data)
        w = data.temporary_data.meas_work
        @test pointer(w.pair_dE) == pointer(w.Mlm)
        @test length(data.temporary_data.G_matrix_vec_x) == measurement_G_length(data.sim_data)
        # rebuilt when the grids change after construction
        push!(data.sim_data.w_list, 3.5)
        w2 = measurement_work!(data)
        @test w2 !== w && size(w2.Lor, 2) == length(data.sim_data.w_list)
        @test measurement_work!(data) === w2
    end

    @testset "3. MC trajectory, fast vs reference measurements" begin
        for (name, sd) in (("L8 K0-TCA", make_case(Lx=8, diag="K0-TCA", num_measure=4, measuring_interval=2)),
                           ("8x12 B Mixed", make_case(Lx=8, Ly=12, B=2π/96, gauge="Mixed", num_measure=3)))
            same_traj, rows, nmeas = trajectory_check(sd)
            @test same_traj
            @test nmeas == sd.num_measure
            @test all(field_ok, rows)
            wr, wa = worst_fields(rows)
            @printf("trajectory %-14s identical MC trajectory: %s, %d measurements, worst field relerr %s %.1e, worst abs %s %.1e\n",
                    name, same_traj, nmeas, wr[1], wr[2], wa[1], wa[3])
        end
    end

    @testset "4. allocations" begin
        sd = make_case(Lx=12, diag="K0-TCA")
        a_old, a_new, k = allocation_check(sd)
        @printf("make_measurement_K! L=12 bytes per call: reference %d, fast %d\n", a_old, a_new)
        for name in sort(collect(keys(k)))
            @printf("      %-22s %6d bytes\n", name, k[name])
            @test k[name] <= 16 * 1024          # thread-task bookkeeping only
        end
    end

    @testset "5. fallbacks" begin
        # SC_L > 1: fast=true must run exactly the reference path (the work is never
        # built). NOTE: on HEAD d9df984 the reference measurement itself throws a
        # BoundsError for SC_L > 1 (measure_current_correlations_q! indexes the
        # q-list with L*SC_L); then both paths must throw the same error.
        sd = make_case(Lx=4, Ly=6, SC_L=2)
        d_ref = Lattice_Data(sd)
        d_fast = deepcopy(d_ref)
        run(d, fast) = try
            make_measurement_K!(d, 1; fast=fast); nothing
        catch err
            err
        end
        e_ref = run(d_ref, false)
        e_fast = run(d_fast, true)
        @test d_fast.temporary_data.meas_work === nothing
        if e_ref === nothing
            rows = compare_measure_data(d_ref.measurement_data, d_fast.measurement_data)
            @test e_fast === nothing && all(r -> r[3] == 0.0, rows)
            println("SC_L=2 (4x6): reference runs; fast=true identical: ", all(r -> r[3] == 0.0, rows))
        else
            @test typeof(e_fast) == typeof(e_ref) && sprint(showerror, e_fast) == sprint(showerror, e_ref)
            println("SC_L=2 (4x6): the reference measurement fails (pre-existing): ", sprint(showerror, e_ref),
                    "; fast=true fails identically: ", sprint(showerror, e_fast) == sprint(showerror, e_ref))
        end
    end

    @testset "6. n_only" begin
        sd = make_case(Lx=8, diag="K0-TCA")
        data = Lattice_Data(sd)
        sweep!(data)
        invalidate_global_diag!(data)
        make_measurement_n_only!(data, 1)
        n1, S1 = data.measurement_data.n[1], data.measurement_data.Ss[1]
        @test isempty(data.temporary_data.G_matrix_vec_x) && data.temporary_data.meas_work === nothing
        invalidate_global_diag!(data)
        make_measurement_K!(data, 2)
        @test data.measurement_data.n[2] == n1
        @test data.measurement_data.Ss[2] == S1

        # guard: n_only measurements cannot be saved
        @test_throws ErrorException MonteCarloSimulation(make_case(Lx=6), mktempdir(); savedata=true, n_only=true)

        # the mu-search lattice never allocates the measurement buffers
        sim = MonteCarloSimulation(make_case(Lx=6, num_measure=5), ""; saveplots=false, savedata=false, n_only=true)
        @test isempty(sim.temporary_data.G_matrix_vec_x) && sim.temporary_data.meas_work === nothing
        @test sim.current_state.num_current_measurments == 5
        @test all(>(0), sim.measurement_data.n[1:5])

        # speed of the mu-search phase: occupation_convergence.jl trims w_list and
        # w_conductivity_range to two points each, nothing else
        occ_sd() = make_case(Lx=12, diag="K0-TCA", num_measure=20, measuring_interval=1,
                             w_list=[-1.0, 1.0], w_cond=[0.0, 0.01])
        MonteCarloSimulation(make_case(Lx=6, num_measure=2), ""; saveplots=false, savedata=false)   # compile
        t_full = @elapsed MonteCarloSimulation(occ_sd(), ""; saveplots=false, savedata=false, n_only=false)
        t_n = @elapsed MonteCarloSimulation(occ_sd(), ""; saveplots=false, savedata=false, n_only=true)
        @printf("mu-search MC run, L=12, 20 measurements x 1 sweep: full measurements %.2f s, n_only %.2f s (x%.1f)\n",
                t_full, t_n, t_full / t_n)
        @test t_n < t_full

        # a short occupation convergence at small L converges
        conv_sd = make_case(Lx=6, diag="K0", mu=-0.5, beta=4.0)
        t_conv = @elapsed res = converge_to_occupation!(conv_sd, 0.8; num_thermal_occ=40, num_measure_occ=40,
                                                        measuring_interval=1, n_tol=5e-3, max_iter=12,
                                                        num_uncorrelated_measurments=16)
        @test res isa Tuple                         # nothing when it fails without sim_info_path
        occ, csim = res
        @printf("occupation convergence L=6 target n=0.8: converged=%s, mu=%.4f, n=%.4f, %d iterations, %.1f s\n",
                occ.converged, occ.best_mu, occ.best_n, length(occ.mu_list) - 2, t_conv)
        @test occ.converged
        @test abs(occ.best_n - 0.8) < 0.02
        @test isempty(csim.temporary_data.G_matrix_vec_x)
    end

    @testset "7. memory estimate and check" begin
        sd = make_case(Lx=16, diag="K0-TCA", Lc=8)
        data = Lattice_Data(sd)
        ensure_measurement_buffers!(data)
        td = data.temporary_data
        est = estimate_memory_bytes(sd)
        work_actual = Base.summarysize(td.meas_work)
        G_actual = Base.summarysize(td.G_matrix_vec_x) + Base.summarysize(td.G_matrix_vec_y) +
                   2 * sd.N * 2 * sd.N * sizeof(ComplexF64)      # + susceptibility scratch (allocated at first chi)
        lat_actual = lattice_arrays_bytes(data)
        @printf("L=16 estimate vs actual: fast work %.1f / %.1f MB, G buffers %.1f / %.1f MB, lattice %.1f / %.1f MB\n",
                est.fast_work / 2^20, work_actual / 2^20, est.G_buffers / 2^20, G_actual / 2^20,
                est.lattice / 2^20, lat_actual / 2^20)
        @test isapprox(est.fast_work, work_actual; rtol=0.10)
        @test isapprox(est.G_buffers, G_actual; rtol=0.02)
        @test 0.9 * lat_actual <= est.lattice <= 1.5 * lat_actual
        @test estimate_memory_bytes(sd; measurements=false).G_buffers == 0
        @test estimate_memory_bytes(sd; fast=false).fast_work == 0

        withenv("SLURM_MEM_PER_NODE" => "4096", "SLURM_MEM_PER_CPU" => nothing) do
            @test job_memory_limit() == (4096 * 2^20, "SLURM_MEM_PER_NODE")
        end
        withenv("SLURM_MEM_PER_NODE" => nothing, "SLURM_MEM_PER_CPU" => "1000", "SLURM_CPUS_PER_TASK" => "4") do
            @test job_memory_limit()[1] == 4000 * 2^20
        end
        # a job sbatch'ed from inside pre_pre_run's --mem=4GB job inherits its
        # SLURM_MEM_PER_NODE; the job's own --mem-per-cpu must win
        withenv("SLURM_MEM_PER_NODE" => "4096", "SLURM_MEM_PER_CPU" => "8192", "SLURM_CPUS_PER_TASK" => "1") do
            @test job_memory_limit() == (8192 * 2^20, "SLURM_MEM_PER_CPU x SLURM_CPUS_PER_TASK")
        end
        withenv("SLURM_MEM_PER_NODE" => "64", "SLURM_MEM_PER_CPU" => nothing, "MC_SKIP_MEMORY_CHECK" => nothing) do
            @test_throws ErrorException check_memory_budget(sd)
            # MonteCarloSimulation refuses before thermalizing
            @test_throws ErrorException MonteCarloSimulation(make_case(Lx=6), ""; saveplots=false, savedata=false)
            # ... but the n_only mu-search is not checked (it allocates no measurement buffers)
            @test MonteCarloSimulation(make_case(Lx=6, num_measure=2), ""; saveplots=false, savedata=false, n_only=true) isa LatticeData
        end
        withenv("SLURM_MEM_PER_NODE" => "64", "SLURM_MEM_PER_CPU" => nothing, "MC_SKIP_MEMORY_CHECK" => "1") do
            @test check_memory_budget(sd).total > 64 * 2^20
        end
        # campaign sizes, for the record
        for L in (20, 24, 28, 32)
            c = make_case(Lx=L, diag="K0-TCA", Lc=8)
            c.w_list = collect(range(-5.0, 5.0, length=101))
            c.w_conductivity_range = [collect(range(4/L^2 * 0.1, 4/L^2 * 2.0, length=7)); collect(range(4/L^2 * 3.0, 16.0, length=55))]
            c.delta_values_fixed = 4 / L^2 .* [0.5; 1.0:10.0;]
            c.saving_interval = 500
            e = estimate_memory_bytes(c)
            @printf("campaign L=%d: estimate %.2f GB (G buffers %.2f, fast work %.2f, lattice %.2f, runtime %.2f)\n",
                    L, e.total / 2^30, e.G_buffers / 2^30, e.fast_work / 2^30, e.lattice / 2^30, e.runtime / 2^30)
        end
    end
end

println("\nsummary (kernel max relerr | worst MeasureData field, relative (fields not ~0) and absolute):")
for r in summary_rows
    @printf("  %-30s %.1e | %-16s %.1e | %-16s %.1e\n", r...)
end
@printf("peak RSS of this process: %.0f MB\n", Sys.maxrss() / 2^20)
