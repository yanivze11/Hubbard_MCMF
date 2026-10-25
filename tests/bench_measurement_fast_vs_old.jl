# ============================================================================
# bench_measurement_fast_vs_old.jl — make_measurement_K! reference path
# (fast=false) vs fast path (fast=true), campaign settings of the 17-09
# Phi_channel campaign (K0-TCA Lc=8, 101 w, 62 w_cond, 11 delta, U=4, beta=10).
#
#   julia -t 1 tests/bench_measurement_fast_vs_old.jl 12 16 20 24
#
# One lattice per L; the two paths measure the same state, each starting from a
# copy of the same (empty) MeasureData. Reports end-to-end time and bytes, the
# fast kernels one by one, the worst MeasureData field difference, and the
# buffer sizes. 1 BLAS thread (the production setting).
# ============================================================================
using Printf, LinearAlgebra
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))
BLAS.set_num_threads(1)

w_list_of(L)      = collect(range(-5.0, 5.0, length=101))
w_cond_of(L)      = [collect(range(4/L^2 * 0.1, 4/L^2 * 2.0, length=7)); collect(range(4/L^2 * 3.0, 16.0, length=55))]
delta_fixed_of(L) = 4 / L^2 .* [0.5; 1.0:10.0;]

make_sd(L) = SimData(; Lx=L, Ly=L, U=4.0, t=1.0, V=0.0, seed=1, mu=-0.5, beta=10.0,
    num_measure=10, num_thermal=0, measuring_interval=1, saving_interval=10,
    w_list=w_list_of(L), w_conductivity_range=w_cond_of(L), delta_values_fixed=delta_fixed_of(L),
    delta_values_stiffness=Float64[], dw_values=Float64[], guess="uniform", U_disorder="uniform",
    diag_method="K0-TCA", pairing_symmetry="s-wave", TCluster_size_x=min(8, L - 1), TCluster_size_y=min(8, L - 1),
    gauge_choice="Mixed", B=0.0, alpha=1.0, beta_phi=0.0,
    snapshot_interval=0, phi_hist_nbins=0, Delta_hist_abs_nbins=0, Delta_hist_arg_nbins=0)

sweep!(d) = MC_stepV2!(d; method="K0-TCA", loop_random=true, pairing="s-wave")

function worst_field(a::MeasureData, b::MeasureData)
    md(x::Number, y::Number) = (abs(x - y), max(abs(x), abs(y)))
    function md(x::AbstractArray, y::AbstractArray)
        d = 0.0; s = 0.0
        for (u, v) in zip(x, y)
            du, su = md(u, v); d = max(d, du); s = max(s, su)
        end
        return d, s
    end
    worst = (:none, 0.0, 0.0)
    for fn in fieldnames(MeasureData)
        d, s = md(getfield(a, fn), getfield(b, fn))
        r = d > 1e-12 ? d / max(s, 1e-300) : 0.0          # fields that are ~0 in both paths count as equal
        r > worst[2] && (worst = (fn, r, d))
    end
    return worst
end

function bench(L; verbose=true)
    data = Lattice_Data(make_sd(L))
    for _ in 1:2; sweep!(data); end
    ensure_measurement_buffers!(data)                 # as MonteCarloSimulation does before thermalizing
    td = data.temporary_data
    work = td.meas_work
    m0 = deepcopy(data.measurement_data)

    invalidate_global_diag!(data)
    st_old = @timed make_measurement_K!(data, 1; fast=false)
    m_old = data.measurement_data
    data.measurement_data = deepcopy(m0)
    invalidate_global_diag!(data)
    st_new = @timed make_measurement_K!(data, 1; fast=true)
    m_new = data.measurement_data
    worst = worst_field(m_old, m_new)
    # steady state of the fast path (diag already synced: measurement only)
    data.measurement_data = deepcopy(m0)
    t_new2 = @elapsed make_measurement_K!(data, 1; fast=true)

    # fast kernels one by one
    q = data.const_struct.q_list_CC
    m = data.measurement_data
    k = Pair{String,Float64}[]
    push!(k, "G^x,G^y(q) row-block" => @elapsed (current_vertices_fast!(td.G_matrix_vec_x, work, data, q, "x"; method=:rowblock);
                                                 current_vertices_fast!(td.G_matrix_vec_y, work, data, q, "y"; method=:rowblock)))
    push!(k, "L_ab(q) x4" => @elapsed current_correlation_q_fast!(work.Lab_q, work, data, td.G_matrix_vec_x, td.G_matrix_vec_y, length(q)))
    push!(k, "fixed-delta L(w)" => @elapsed measure_fixed_delta_fast!(data, m, 1, work, td.G_matrix_vec_x[1], td.G_matrix_vec_y[1]))
    push!(k, "DdD" => @elapsed compute_D_dagger_D_fast!(work, data))
    push!(k, "A(k,w)+BZ+LDOS" => @elapsed measure_spectral_fast!(data, m, 1, work))
    push!(k, "G^x,G^y(q) per-q GEMM" => @elapsed (current_vertices_fast!(td.G_matrix_vec_x, work, data, q, "x"; method=:gemm);
                                                   current_vertices_fast!(td.G_matrix_vec_y, work, data, q, "y"; method=:gemm)))
    work_MB = Base.summarysize(work) / 2^20
    G_MB = (Base.summarysize(td.G_matrix_vec_x) + Base.summarysize(td.G_matrix_vec_y)) / 2^20
    verbose || return nothing
    @printf("\nL=%d  N=%d  n_q=%d\n", L, data.sim_data.N, length(q))
    @printf("  make_measurement_K!  reference %8.2f s (%7.2f MB alloc) | fast %7.3f s (%6.2f MB alloc), measurement only %7.3f s | x%.1f\n",
            st_old.time, st_old.bytes / 2^20, st_new.time, st_new.bytes / 2^20, t_new2, st_old.time / st_new.time)
    @printf("  worst MeasureData field: %s relerr %.1e (abs %.1e)\n", worst...)
    for (name, t) in k
        @printf("    %-24s %8.3f s\n", name, t)
    end
    @printf("  buffers: fast-kernel work %.1f MB, G_matrix_vec_x/_y %.1f MB, estimate %.1f / %.1f MB\n",
            work_MB, G_MB, measurement_work_bytes(data.sim_data) / 2^20, estimate_memory_bytes(data.sim_data).G_buffers / 2^20)
    return (L=L, t_old=st_old.time, t_new=st_new.time, t_new2=t_new2, worst=worst)
end

Ls = isempty(ARGS) ? [12, 16, 20, 24] : parse.(Int, ARGS)
@printf("julia %s, %d Julia threads, %d BLAS thread(s), %s\n", VERSION, Threads.nthreads(), BLAS.get_num_threads(),
        Sys.cpu_info()[1].model)
t_warm = @elapsed bench(10; verbose=false)
@printf("warm-up (compilation) %.1f s\n", t_warm)
rows = [bench(L) for L in Ls]
println("\n| L | reference [s] | fast [s] | speed-up | worst field relerr |")
println("|---|---|---|---|---|")
for r in rows
    @printf("| %d | %.2f | %.3f | %.1fx | %.1e (%s) |\n", r.L, r.t_old, r.t_new, r.t_old / r.t_new, r.worst[2], r.worst[1])
end
@printf("peak RSS %.0f MB\n", Sys.maxrss() / 2^20)
