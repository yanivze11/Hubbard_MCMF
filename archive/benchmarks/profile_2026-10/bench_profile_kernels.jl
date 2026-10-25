# ============================================================================
# bench_profile_kernels.jl — PROTOTYPE fast versions of the hot measurement
# kernels of MC_Hubburd.jl.
#
# Nothing here is wired into the simulation: make_measurement_K! is unchanged.
# Each function returns exactly the quantity of the existing kernel named in its
# docstring (checked to ~1e-13 relative error, see bench_profile_run.jl, section
# "proto"), so it can be timed against it and later swapped in.
#
# Memory: every kernel works in buffers that are allocated ONCE per run by
# `ProtoWork(data)` (in production they would live in TemporaryData). The work
# struct holds two kinds of data:
#   * q-independent constants built once from the lattice: bond tables, the
#     sparse pattern of the current operator, per-row/column operators, Fourier
#     tables;
#   * scratch arrays that are overwritten at every measurement.
# A measurement then allocates nothing apart from a few hundred bytes of task
# bookkeeping in the threaded loops; bench_profile_run.jl records the bytes per
# call.
#
# Scope: s-wave, SC_L = 1 (the production setting), like the q-resolved code it
# mirrors. Requires MC_Hubburd.jl to be included first.
#
# Measured on a laptop core at L=20 (campaign settings, see bench_profile_run.jl):
#   current-vertex G^x,G^y(q), all q   6.72 s -> 2.30 s   (row-block)
#   L_ab(q) x4 from G                  0.30 s -> 0.03 s
#   fixed-delta L_ab(w), 11x62 grid    5.03 s -> 0.42 s   (cache-blocked)
#   <Delta^dag Delta>                  0.34 s -> 0.04 s
#   A(k,w) path + BZ + LDOS            1.43 s -> 0.035 s
# ============================================================================
using LinearAlgebra, SparseArrays


# ════════════════════════════════════════════════════════════════════════════
#  1. Current vertices  G^a(q)[l,m] = <l| j^a(q) |m>
# ════════════════════════════════════════════════════════════════════════════

# One lattice row (for the (0, qy) family) or column (for the (qx, 0) family)
# of the row-block method. Everything here is q-independent.
struct ProtoVertexGroup
    axis::Symbol                 # :y -> group of row y, used for q = (0, qy)
                                 # :x -> group of column x, used for q = (qx, 0), qx != 0
    coord::Int                   # the row y (or column x) value, enters e^{i q coord}
    rows::Vector{Int}            # rows of U touched by this group's bonds (up, then down)
    Mg::Matrix{ComplexF64}       # the group's current operator restricted to `rows`
end

# Everything about the +a current operator of one direction that does not
# depend on the eigenvectors or on q.
struct ProtoVertexOp
    nb::Vector{Int}              # nb[i] = site i + a
    Tr::Vector{ComplexF64}       # boundary twist x Peierls phase of the bond i -> i + a
    M::SparseMatrixCSC{ComplexF64,Int}   # 2N x 2N, fixed pattern; nzval rewritten per q
    slot::Vector{Int}            # term k (4 per site) -> its position in M.nzval
    groups::Vector{ProtoVertexGroup}
end

"""
    ProtoVertexWork

Work for `proto_current_vertex_gemm!` / `proto_current_vertex_rowblock!`:
the q-independent operator data of both directions plus scratch shared by them
(T and Grow are (2N)^2 complex: 64 MB each at L=32; UR and MU are max|rows| x 2N).
"""
struct ProtoVertexWork
    x::ProtoVertexOp
    y::ProtoVertexOp
    T::Matrix{ComplexF64}        # M(q) * U        (per-q method)
    UR::Matrix{ComplexF64}       # U[rows, :]      (row-block method)
    MU::Matrix{ComplexF64}       # Mg * U[rows, :] (row-block method)
    Grow::Matrix{ComplexF64}     # one row/column contribution, 2N x 2N
end

function _proto_vertex_op(data::LatticeData, dir::String)
    cs = data.const_struct
    N = data.sim_data.N
    if dir == "x"
        nb, bc, Ld, col = cs.new_ix_list, cs.x, data.sim_data.Lx, 1
    elseif dir == "y"
        nb, bc, Ld, col = cs.new_iy_list, cs.y, data.sim_data.Ly, 2
    else
        error("dir must be \"x\" or \"y\"")
    end
    # same phases as build_G_r_from_S_r_x!/y!
    phase_Tr = exp(-1im * 2π / Ld * cs.SC_K_vectors[1][col])
    Tr = [(bc[i] == Ld ? phase_Tr : 1.0 + 0.0im) * cs.peierls_phases[i, col] for i in 1:N]

    # the 4 terms per site (see proto_current_vertex_gemm!): rows I, columns J
    I = Vector{Int}(undef, 4N); J = Vector{Int}(undef, 4N)
    for i in 1:N
        n = nb[i]
        I[4i-3], J[4i-3] = n, i
        I[4i-2], J[4i-2] = i, n
        I[4i-1], J[4i-1] = N + i, N + n
        I[4i],   J[4i]   = N + n, N + i
    end
    M = sparse(I, J, ones(ComplexF64, 4N), 2N, 2N)        # pattern only; duplicates merged
    slot = Vector{Int}(undef, 4N)
    for k in 1:4N
        r = M.colptr[J[k]]:(M.colptr[J[k]+1] - 1)
        slot[k] = r[findfirst(==(I[k]), view(M.rowval, r))]
    end

    groups = ProtoVertexGroup[]
    for (axis, key) in ((:y, cs.y), (:x, cs.x))
        for g in sort(unique(key))
            S = findall(==(g), key)
            R = unique(vcat(S, nb[S]))
            rows = vcat(R, R .+ N)
            pos = Dict(r => k for (k, r) in enumerate(rows))
            Mg = zeros(ComplexF64, length(rows), length(rows))
            for i in S
                n = nb[i]
                Mg[pos[n],     pos[i]]     +=  conj(Tr[i])
                Mg[pos[i],     pos[n]]     += -Tr[i]
                Mg[pos[N + i], pos[N + n]] += -conj(Tr[i])
                Mg[pos[N + n], pos[N + i]] +=  Tr[i]
            end
            push!(groups, ProtoVertexGroup(axis, g, rows, Mg))
        end
    end
    return ProtoVertexOp(nb, Tr, M, slot, groups)
end

function ProtoVertexWork(data::LatticeData)
    N = data.sim_data.N
    opx = _proto_vertex_op(data, "x")
    opy = _proto_vertex_op(data, "y")
    rmax = maximum(length(g.rows) for g in vcat(opx.groups, opy.groups))
    return ProtoVertexWork(opx, opy, Matrix{ComplexF64}(undef, 2N, 2N),
                           Matrix{ComplexF64}(undef, rmax, 2N), Matrix{ComplexF64}(undef, rmax, 2N),
                           Matrix{ComplexF64}(undef, 2N, 2N))
end

_proto_op(work::ProtoVertexWork, dir::String) =
    dir == "x" ? work.x : dir == "y" ? work.y : error("dir must be \"x\" or \"y\"")


"""
    proto_current_vertex_gemm!(Gout, work, data, q_vecs, dir)

Replaces `make_G_matrix_q_vec(data, q_vecs; direction=dir)`; fills
`Gout[iq][l,m]` with the same numbers (Gout is preallocated, e.g. the existing
temporary_data.G_matrix_vec_x / _y).

The current code loops over all (2N)^2 eigen-pairs (l,m). For each it builds the
N-vector G_r (`build_G_r_from_S_r_x!`) and Fourier-transforms it to every q with a
small GEMV, which costs about 4N^3 (8 + n_q) scalar operations. It also writes
`G_out[iq][l,m]` row by row, i.e. with stride 2N in a column-major matrix.

The site sum is a bilinear form in the eigenvectors:

    G^a(q) = U' * M_a(q) * U

where M_a(q) is the 2N x 2N real-space current operator with 4N non-zeros
(the four terms of build_G_r_from_S_r_*; n = i + a, T = Tr[i], phase = e^{iq.r_i}):

    M[n,   i  ] =  phase * conj(T)      up:   conj(U[n,l]) U[i,m]
    M[i,   n  ] = -phase * T            up:   conj(U[i,l]) U[n,m]
    M[N+i, N+n] = -phase * conj(T)      down: conj(U[N+i,l]) U[N+n,m]
    M[N+n, N+i] =  phase * T            down: conj(U[N+n,l]) U[N+i,m]

The sparsity pattern of M is the same for every q, so it is built once (in
`work`); per q only its 4N values are rewritten. Then one sparse x dense product,
O(N^2), and one ZGEMM, (2N)^3, which MKL runs at near peak and multithreads.
About 2x faster on one core. The cost is still O(N^3) per q, so the number of
q-points stays the main lever.
"""
function proto_current_vertex_gemm!(Gout, work::ProtoVertexWork, data::LatticeData, q_vecs, dir::String)
    op = _proto_op(work, dir)
    N = data.sim_data.N
    U = view(data.current_state.U_matrix, :, :, 1)
    FT = get_FT_table_for_q_vecs!(data, q_vecs)          # FT[i,iq] = e^{+i q.r_i}, precomputed
    nz = op.M.nzval; slot = op.slot; Tr = op.Tr
    for iq in eachindex(q_vecs)
        fill!(nz, 0)
        @inbounds for i in 1:N
            ph = FT[i, iq]; t = Tr[i]
            nz[slot[4i-3]] +=  ph * conj(t)
            nz[slot[4i-2]] += -ph * t
            nz[slot[4i-1]] += -ph * conj(t)
            nz[slot[4i]]   +=  ph * t
        end
        mul!(work.T, op.M, U)
        mul!(Gout[iq], U', work.T)
    end
    return Gout
end


"""
    proto_current_vertex_rowblock!(Gout, work, data, q_vecs, dir)

Same output as `proto_current_vertex_gemm!` and `make_G_matrix_q_vec`, cheaper
when there are many q.

It uses the fact that q_list_CC only holds q = (0, qy) and q = (qx, 0). For the
(0, qy) family the phase e^{i qy y_i} is constant along a lattice row, so

    G(0,qy) = sum_y e^{i qy y} G_row(y),   G_row(y) = U[R_y,:]' * M_y * U[R_y,:]

where R_y holds the up and down rows of the sites of row y and of their
neighbours, and M_y is the q-independent current operator of that row's bonds.
Both R_y and M_y are built once in `work`. The (qx, 0) family is the same with
columns. All the row (column) GEMMs together cost one or two full GEMMs, about
8-16 N^3. The phases are then applied with axpy, so the total for ALL q is about
24-48 N^3 instead of 8 N^3 * n_q. Measured 2.9-3.5x faster than the current
code on one core.
"""
function proto_current_vertex_rowblock!(Gout, work::ProtoVertexWork, data::LatticeData, q_vecs, dir::String)
    op = _proto_op(work, dir)
    N = data.sim_data.N
    M2 = 2N
    U = view(data.current_state.U_matrix, :, :, 1)
    nq = length(q_vecs)
    for iq in 1:nq
        q = q_vecs[iq]
        (q[1] == 0.0 || q[2] == 0.0) || error("row-block needs q on the axes")
        fill!(Gout[iq], 0)
    end
    for grp in op.groups
        r = length(grp.rows)
        UR = view(work.UR, 1:r, :)
        MU = view(work.MU, 1:r, :)
        rows = grp.rows
        @inbounds for c in 1:M2
            for k in 1:r
                UR[k, c] = U[rows[k], c]
            end
        end
        mul!(MU, grp.Mg, UR)
        mul!(work.Grow, UR', MU)                          # this row's (column's) G_row
        for iq in 1:nq
            q = q_vecs[iq]
            if grp.axis === :y
                q[1] == 0.0 || continue                   # (0, qy) family, includes q = 0
                axpy!(cis(q[2] * grp.coord), work.Grow, Gout[iq])
            else
                q[1] != 0.0 || continue                   # (qx, 0) family
                axpy!(cis(q[1] * grp.coord), work.Grow, Gout[iq])
            end
        end
    end
    return Gout
end


# ════════════════════════════════════════════════════════════════════════════
#  2. Static current-current correlation  L_ab(q, w=0)
# ════════════════════════════════════════════════════════════════════════════

"""
    ProtoLabWork(N)

Work for `proto_Lab_q!`: the (2N)^2 Float64 matrix M_lm (32 MB at L=32).
"""
struct ProtoLabWork
    Mlm::Matrix{Float64}
end
ProtoLabWork(N::Int) = ProtoLabWork(Matrix{Float64}(undef, 2N, 2N))

"""
    proto_Lab_q!(out, work, data, Gx, Gy, nq) -> out

`out` is a preallocated 4 x nq ComplexF64 matrix whose rows are (xx, xy, yx, yy).
Replaces the 4 calls to `current_correlation_from_G!` in
`measure_current_correlations_q!`; row k equals `Lxx_temp[1:nq]` of the k-th call:

    L_ab(q) = (t^2/N) [ -sum_lm G^a_lm(q) conj(G^b_lm(q)) M_lm
                        + beta (sum_l G^a_ll f_l) conj(sum_l G^b_ll f_l) ]

with M_lm = (f_l - f_m)/(E_l - E_m), or beta f_l (f_l - 1) for degenerate pairs.

The current code recomputes M_lm, a division plus a branch, for every q and
every one of the 4 components. It also walks `G[l,m]` with m innermost, which
has stride 2N in a column-major matrix.

Here M_lm is built once per measurement into `work`, all 4 components are
accumulated in one column-major pass, and the q loop is threaded. About 9x
faster on one core.
"""
function proto_Lab_q!(out::AbstractMatrix{ComplexF64}, work::ProtoLabWork, data::LatticeData, Gx, Gy, nq::Int)
    f = view(data.temporary_data.f, :, 1)
    E = view(data.current_state.Energies, :, 1)
    β = data.sim_data.beta
    N = data.sim_data.N
    t = data.sim_data.t
    M2 = 2N
    Mlm = work.Mlm
    @inbounds for m in 1:M2, l in 1:M2
        d = E[l] - E[m]
        Mlm[l, m] = abs(d) < 1e-12 ? β * f[l] * (f[l] - 1) : (f[l] - f[m]) / d
    end
    p = t^2 / N
    Threads.@threads :static for iq in 1:nq
        gx = Gx[iq]; gy = Gy[iq]
        sxx = 0.0; syy = 0.0; sxy = 0.0im
        dx = 0.0im; dy = 0.0im
        @inbounds for m in 1:M2
            for l in 1:M2
                a = gx[l, m]; b = gy[l, m]; w = Mlm[l, m]
                sxx += abs2(a) * w
                syy += abs2(b) * w
                sxy += a * conj(b) * w
            end
            dx += gx[m, m] * f[m]
            dy += gy[m, m] * f[m]
        end
        @inbounds begin
            out[1, iq] = p * (-sxx + β * abs2(dx))
            out[2, iq] = p * (-sxy + β * dx * conj(dy))
            out[3, iq] = p * (-conj(sxy) + β * dy * conj(dx))
            out[4, iq] = p * (-syy + β * abs2(dy))
        end
    end
    return out
end


# ════════════════════════════════════════════════════════════════════════════
#  3. Finite-frequency conductivity at fixed broadening  L_ab(w + i delta)
# ════════════════════════════════════════════════════════════════════════════

"""
    ProtoFDWork(N, n_delta, n_w; nchunks=Threads.nthreads())

Work for `proto_fixed_delta!`:
  * 5 packed pair arrays of length (2N)^2: 160 N^2 bytes, i.e. 25 MB at L=20 and
    168 MB at L=32;
  * one 8 x (n_delta n_w) accumulator matrix per chunk of blocks (`nchunks`,
    by default one per worker thread).
"""
struct ProtoFDWork
    dE::Vector{Float64}
    Pxx::Vector{Float64}
    Pyy::Vector{Float64}
    Pr::Vector{Float64}          # real(df gx conj(gy))
    Pi::Vector{Float64}          # imag(df gx conj(gy))
    acc::Array{Float64,3}        # 8 x (n_delta*n_w) x nchunks
end
function ProtoFDWork(N::Int, n_delta::Int, n_w::Int; nchunks::Int=Threads.nthreads())
    n = (2N)^2
    return ProtoFDWork(Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                       Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                       zeros(8, n_delta * n_w, max(nchunks, 1)))
end

# Sum over the packed pairs p0:p1 for one (w, δ). Returns the 8 real accumulators
# (re, im) of xx, yy, xy, yx.
@inline function _proto_fd_kernel(work::ProtoFDWork, p0::Int, p1::Int, w::Float64, δ::Float64)
    dE = work.dE; Pxx = work.Pxx; Pyy = work.Pyy; Pr = work.Pr; Pi = work.Pi
    δ2 = δ * δ
    axx_r = 0.0; axx_i = 0.0; ayy_r = 0.0; ayy_i = 0.0
    axy_r = 0.0; axy_i = 0.0; ayx_r = 0.0; ayx_i = 0.0
    @inbounds @simd for p in p0:p1
        a = w + dE[p]
        inv = 1.0 / (a * a + δ2)
        re = a * inv
        im = δ * inv
        axx_r += Pxx[p] * re;  axx_i -= Pxx[p] * im
        ayy_r += Pyy[p] * re;  ayy_i -= Pyy[p] * im
        axy_r += Pr[p] * re + Pi[p] * im;   axy_i += Pi[p] * re - Pr[p] * im
        ayx_r += Pr[p] * re - Pi[p] * im;   ayx_i += -Pi[p] * re - Pr[p] * im
    end
    return (axx_r, axx_i, ayy_r, ayy_i, axy_r, axy_i, ayx_r, ayx_i)
end

"""
    proto_fixed_delta!(out, work, data, Gx0, Gy0; block=4096) -> (out, frac_kept)

`out[iδ, iw]` is the tuple (Lxx, Lxy, Lyx, Lyy). It equals
`current_correlation_K_4dir(data, w; pre_Gx, pre_Gy, δ)` for every δ in
`delta_values_fixed` and every w in `w_conductivity_range`. Gx0 and Gy0 are the
q = 0 current vertices; `work` is a `ProtoFDWork` and `out` a preallocated
n_delta x n_w array.

The current code makes one full pass over the (2N)^2 pairs per (δ, w), which is
11 x 62 = 682 passes per measurement at campaign settings. Each pass rereads
Gx[l,m] and Gy[l,m] with stride 2N and recomputes the products G G* (f_l - f_m).

1. Pack. The kernel is (a - iδ)/(a^2 + δ^2) = 1/(a + iδ) with a = w + E_l - E_m,
   so each pair needs only its energy difference and three weights
       Pxx = df |gx|^2,   Pyy = df |gy|^2,   Pxy = df gx conj(gy)   (Pyx = conj(Pxy)).
   They are written once, in column-major order, into the preallocated
   contiguous arrays of `work`. Pairs with df == 0 exactly are dropped. This step
   costs O((2N)^2), a few ms.
2. Block. Each (δ, w) is a branch-free SIMD loop with real accumulators. Run
   over all packed pairs, the loop is memory-bandwidth bound: every one of the
   682 passes streams the whole arrays (25 MB at L=20) from RAM. So the pairs are
   processed in blocks of `block` pairs (5*8*block bytes, 160 KB for 4096), and
   each block is used for all 682 (δ, w) while it sits in cache. That moves the
   loop from memory-bound to compute-bound: 1.9x on one core, and it is what lets
   it scale with threads.
3. Threads. Blocks are dealt round-robin to `nchunks` chunks (chunk c takes
   blocks c, c + nchunks, ...). Each chunk has its own accumulators, indexed by
   the CHUNK number, never by Threads.threadid(): thread ids are not
   1:nthreads(). Julia 1.12 adds an interactive thread by default, so with -t 4
   the worker threads are ids 2..5. The chunk sums are reduced in a fixed order,
   so for a given nchunks the result is deterministic whatever the scheduling.

Measured on one core: L=20 5.03 s -> 0.42 s, L=24 22.1 s -> 1.19 s, relative
difference to the current code about 1e-13 (summation order only).
"""
function proto_fixed_delta!(out::AbstractMatrix, work::ProtoFDWork, data::LatticeData,
                            Gx0::AbstractMatrix, Gy0::AbstractMatrix; block::Int=4096)
    f = view(data.temporary_data.f, :, 1)
    E = view(data.current_state.Energies, :, 1)
    N = data.sim_data.N
    M2 = 2N
    ws = data.sim_data.w_conductivity_range
    ds = data.sim_data.delta_values_fixed
    nd = length(ds); ng = nd * length(ws)
    length(work.dE) >= M2^2 || error("ProtoFDWork too small for N=$N")
    size(work.acc, 2) == ng || error("ProtoFDWork built for a different delta/w grid")
    size(out) == (nd, length(ws)) || error("out must be n_delta x n_w")

    # 1) pack, column-major, skipping pairs that contribute exactly nothing
    np = 0
    @inbounds for m in 1:M2
        fm = f[m]; Em = E[m]
        for l in 1:M2
            df = f[l] - fm
            df == 0.0 && continue
            np += 1
            gx = Gx0[l, m]; gy = Gy0[l, m]; c = gx * conj(gy)
            work.dE[np] = E[l] - Em
            work.Pxx[np] = df * abs2(gx); work.Pyy[np] = df * abs2(gy)
            work.Pr[np] = df * real(c);   work.Pi[np] = df * imag(c)
        end
    end

    # 2+3) cache-blocked accumulation, blocks dealt round-robin to chunks.
    # `npairs` is bound once: capturing the reassigned `np` in the threaded closure
    # would box it (Core.Box) and make every kernel call a dynamic dispatch.
    # The accumulator slice is chosen by the chunk index c, NOT Threads.threadid()
    # (see the docstring: thread ids can exceed nthreads()).
    npairs = np
    nblk = cld(npairs, block)
    nch = size(work.acc, 3)
    acc = work.acc
    fill!(acc, 0.0)
    Threads.@threads for c in 1:nch
        for ib in c:nch:nblk
            p0 = (ib - 1) * block + 1
            p1 = min(ib * block, npairs)
            @inbounds for k in 1:ng
                a = _proto_fd_kernel(work, p0, p1, ws[cld(k, nd)], ds[mod1(k, nd)])
                for j in 1:8
                    acc[j, k, c] += a[j]
                end
            end
        end
    end

    # reduce over threads and write (xx, xy, yx, yy)
    pref = -data.sim_data.t^2 / N
    @inbounds for k in 1:ng
        s = ntuple(j -> sum(view(acc, j, k, :)), 8)
        out[mod1(k, nd), cld(k, nd)] = (pref * complex(s[1], s[2]), pref * complex(s[5], s[6]),
                                        pref * complex(s[7], s[8]), pref * complex(s[3], s[4]))
    end
    return out, np / M2^2
end


# ════════════════════════════════════════════════════════════════════════════
#  4. Pair correlation  <Delta^dag Delta>
# ════════════════════════════════════════════════════════════════════════════

"""
    ProtoDdDWork(N)

Work for `proto_DdD!`: the (2N)^2 complex matrix V (64 MB at L=32).
"""
struct ProtoDdDWork
    V::Matrix{ComplexF64}
end
ProtoDdDWork(N::Int) = ProtoDdDWork(Matrix{ComplexF64}(undef, 2N, 2N))

"""
    proto_DdD!(work, data) -> Float64

Replaces `compute_D_dagger_D_K(data)`.

The current code calls `compute_V_k` (an N-term dot product) for each of the
(2N)^2 pairs (l,m): O(4N^3) scalar work, one function call per pair.

V_lm = sum_r conj(U[r,l]) U[N+r,m] is the matrix product V = U_up' * U_dn, so a
single ZGEMM writes it into `work.V`. The rest is an O(N^2) sum:
    DdD = ( sum_lm f_l (1 - f_m) |V_lm|^2 + |sum_l f_l V_ll|^2 ) / N
Measured 8-12x faster on one core.
"""
function proto_DdD!(work::ProtoDdDWork, data::LatticeData)
    N = data.sim_data.N
    U = view(data.current_state.U_matrix, :, :, 1)
    f = view(data.temporary_data.f, :, 1)
    V = work.V
    mul!(V, view(U, 1:N, :)', view(U, N+1:2N, :))
    s = 0.0
    a2 = 0.0im
    @inbounds for m in 1:2N
        om = 1 - f[m]
        for l in 1:2N
            s += f[l] * om * abs2(V[l, m])
        end
        a2 += f[m] * V[m, m]
    end
    return (s + abs2(a2)) / N
end


# ════════════════════════════════════════════════════════════════════════════
#  5. Spectral function on the k-path, BZ map at w = 0, and LDOS
# ════════════════════════════════════════════════════════════════════════════

"""
    ProtoSpectralWork(data)

Work for `proto_spectral!`.
  * Constants built once: the Fourier rows of the k-path points (n_path x N),
    and of all k for the BZ map (N x N, 16 MB at L=32).
  * Scratch: the Lorentzian table (2N x n_w), the projections (complex and |.|^2)
    and the outputs A_path (n_path x n_w), BZ (N) and LDOS (N x n_w).
About 90 MB in total at L=32.
"""
struct ProtoSpectralWork
    F_path::Matrix{ComplexF64}   # F_path[p, n] = e^{-i k_p . r_n}, k_p on SF_k_path
    F_all::Matrix{ComplexF64}    # same for every k in k_vectors (BZ map)
    iw0::Int                     # index of w = 0 in w_list (0 if absent)
    Lor::Matrix{Float64}         # Lor[l, j] = (δ/π) / (δ^2 + (E_l - w_j)^2)
    FU_path::Matrix{ComplexF64}  # F_path * U_up
    W_path::Matrix{Float64}      # |F_path * U_up|^2
    FU_all::Matrix{ComplexF64}   # F_all * U_up
    W_all::Matrix{Float64}       # |F_all * U_up|^2
    absU2::Matrix{Float64}       # |U_up|^2
    A_path::Matrix{Float64}      # output: A(k_p, w_j)
    BZ::Vector{Float64}          # output: A(k, w = 0)
    LDOS::Matrix{Float64}        # output: LDOS(i, w_j)
end

function ProtoSpectralWork(data::LatticeData)
    cs = data.const_struct
    N = data.sim_data.N
    data.sim_data.SC_L == 1 || error("proto_spectral: SC_L = 1 only")
    nw = length(data.sim_data.w_list)
    kxp, kyp = cs.SF_k_path
    kv = [cs.k_vectors[cs.k_general_qK_idx[cs.k_general_idx[kxp[l], kyp[l]]][1]] for l in eachindex(kxp)]
    F_path = [exp(-1im * (k[1] * cs.x[n] + k[2] * cs.y[n])) for k in kv, n in 1:N]
    F_all = [exp(-1im * (k[1] * cs.x[n] + k[2] * cs.y[n])) for k in cs.k_vectors, n in 1:N]
    iw0 = something(findfirst(==(0.0), data.sim_data.w_list), 0)
    np = length(kv)
    return ProtoSpectralWork(F_path, F_all, iw0, Matrix{Float64}(undef, 2N, nw),
                             Matrix{ComplexF64}(undef, np, 2N), Matrix{Float64}(undef, np, 2N),
                             Matrix{ComplexF64}(undef, N, 2N), Matrix{Float64}(undef, N, 2N),
                             Matrix{Float64}(undef, N, 2N),
                             Matrix{Float64}(undef, np, nw), Vector{Float64}(undef, N),
                             Matrix{Float64}(undef, N, nw))
end

"""
    proto_spectral!(work, data) -> (A_path, BZ, LDOS)

Replaces the spectral block of make_measurement_K! (the `for (j,w) in
enumerate(w_list)` loop). The returned arrays are the buffers inside `work`:
  * A_path[p, j] = spectral_functionV2(data, K_ix, q_ix, w_j; delt=delta_max)
    for the p-th point of SF_k_path;
  * BZ           = spectral_function_fullV2(data, 0.0; delt=delta_max)
    (`nothing` when 0.0 is not in w_list);
  * LDOS[:, j]   = compute_full_LDOS(data, w_j; delt=delta_max)
    (and DOS_j = sum(LDOS[:, j]) / N).

The projections |<k|u_l>|^2 do not depend on w. The current code still rebuilds
them, including N exp() calls per k, for each of the ~100 w values, and the BZ
spends O(N) per (k, l) pair. Here the Fourier rows are built once per run, the
projections once per measurement,
    W_path = |F_path * U_up|^2   (n_path x 2N),   W_all = |F_all * U_up|^2   (N x 2N),
and every w at once is a GEMM against the Lorentzian table
Lor[l, j] = (δ/π) / (δ^2 + (E_l - w_j)^2). Measured 40-140x faster.
"""
function proto_spectral!(work::ProtoSpectralWork, data::LatticeData)
    N = data.sim_data.N
    U = view(data.current_state.U_matrix, :, :, 1)
    E = view(data.current_state.Energies, :, 1)
    Uup = view(U, 1:N, :)
    w_list = data.sim_data.w_list
    δ = data.temporary_data.delta_max
    Lor = work.Lor
    @inbounds for j in eachindex(w_list), l in 1:2N
        Lor[l, j] = (1 / (pi * δ)) / (1 + ((E[l] - w_list[j]) / δ)^2)
    end
    # k-path
    mul!(work.FU_path, work.F_path, Uup)
    work.W_path .= abs2.(work.FU_path)
    mul!(work.A_path, work.W_path, Lor)
    rmul!(work.A_path, 1 / N)
    # BZ map at w = 0
    BZ = nothing
    if work.iw0 != 0
        mul!(work.FU_all, work.F_all, Uup)
        work.W_all .= abs2.(work.FU_all)
        mul!(work.BZ, work.W_all, view(Lor, :, work.iw0))
        rmul!(work.BZ, 1 / N)
        BZ = work.BZ
    end
    # LDOS (factor 2: spin)
    work.absU2 .= abs2.(Uup)
    mul!(work.LDOS, work.absU2, Lor)
    rmul!(work.LDOS, 2)
    return work.A_path, BZ, work.LDOS
end


# ════════════════════════════════════════════════════════════════════════════
#  6. Occupation-convergence phase
# ════════════════════════════════════════════════════════════════════════════

"""
    proto_n_only!(data) -> n

Everything the occupation-convergence phase (occupation_convergence.jl) actually
uses from a measurement: the full diagonalization, the Fermi factors and
<n> (`avrage_n_K`). Today that phase calls the full make_measurement_K!, which
builds every q-resolved current vertex, the fixed-delta grid, DdD, Ps_l, ... and
discards them. That is 25-64x more work per measurement at L=12-20. It uses
only existing functions, all of which work in the preallocated
TemporaryData / CurrentState arrays.
"""
function proto_n_only!(data::LatticeData)
    ensure_global_diag_synced!(data; reason="n_only")
    get_FD_dist!(data)
    return avrage_n_K(data)
end


# ════════════════════════════════════════════════════════════════════════════
#  All work buffers together
# ════════════════════════════════════════════════════════════════════════════

"""
    ProtoWork(data)

Every buffer the prototype kernels need, allocated once per run (in production:
inside TemporaryData). `Base.summarysize(work)` gives its size.
"""
struct ProtoWork
    vertex::ProtoVertexWork
    lab::ProtoLabWork
    fd::ProtoFDWork
    ddd::ProtoDdDWork
    spec::ProtoSpectralWork
end
function ProtoWork(data::LatticeData)
    N = data.sim_data.N
    return ProtoWork(ProtoVertexWork(data), ProtoLabWork(N),
                     ProtoFDWork(N, length(data.sim_data.delta_values_fixed), length(data.sim_data.w_conductivity_range)),
                     ProtoDdDWork(N), ProtoSpectralWork(data))
end
