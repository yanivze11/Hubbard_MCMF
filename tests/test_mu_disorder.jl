# test_mu_disorder.jl — tests of the binary μ–U disorder (details below).
using Test
using Random

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

# Tests for coupled / potential mu-U disorder (item 3): mu_i = mu - U_i/2 with
# U_i in {0, U}, an EXACT count floor(f*N) of empty (U_i=0, mu unshifted) sites.
# Covers: exact partition, mu_shift = U_i/2, the fold into V_matrix, the actual BdG
# diagonal, coupled-vs-potential-only interaction, reproducibility, and error paths.

function base_sim(; mu_disorder::String, f::Float64, U::Float64=5.0, mu::Float64=0.0,
                    pairing::String="s-wave", V::Float64=0.0, seed::Int=101, dseed::Int=0)
    return SimData(
        Lx=6, Ly=6, SC_L=1, U=U, t=1.0, V=V, seed=seed, mu=mu, beta=6.0,
        num_measure=1, num_thermal=0, measuring_interval=1, saving_interval=1,
        w_list=[0.0], guess="uniform", U_disorder="uniform",
        diag_method="K0", pairing_symmetry=pairing, directory=".",
        num_cores=1, num_threads=1, w_conductivity_range=[0.0],
        delta_values_fixed=Float64[], delta_values_stiffness=Float64[], dw_values=Float64[],
        U_disorder_params=[0.0, 10.0, 0.7, 0.7], U_impurity=0.0,
        TCluster_size_x=1, TCluster_size_y=1, smooth_U=(false, 0.8),
        loop_random=true, B=0.0, gauge_choice="LandauX", boundary_conditions="PBC",
        compute_jP=false, alpha=1.0, beta_phi=0.0, Q_density=[0.0, 0.0], disorder_seed=dseed,
        snapshot_interval=0, phi_hist_nbins=0, Delta_hist_abs_nbins=0, Delta_hist_arg_nbins=0,
        mu_disorder=mu_disorder, mu_disorder_f=f,
    )
end

@testset "mu-U disorder" begin
    N = 36
    U = 5.0

    # ---------- none: no shift, uniform interaction ----------
    cs_none = Const_Struct(base_sim(mu_disorder="none", f=0.0))
    @test all(==(0.0), cs_none.mu_shift)
    @test all(==(U), cs_none.U_params)

    # ---------- coupled: binary {0,U} interaction + coupled shift ----------
    for f in (0.0, 0.25, 0.3, 0.5, 1.0)
        cs = Const_Struct(base_sim(mu_disorder="coupled", f=f, dseed=7))
        n_zero = floor(Int, f * N)
        zero_mask = cs.U_params .== 0.0
        u_mask    = cs.U_params .== U
        @test count(zero_mask) == n_zero              # EXACT count of empty sites
        @test count(u_mask) == N - n_zero
        @test all(zero_mask .| u_mask)                # every site is exactly 0 or U
        # mu_shift = U_i/2: U/2 on interacting sites, 0 on empty sites
        @test all(cs.mu_shift[u_mask]    .== U / 2)
        @test all(cs.mu_shift[zero_mask] .== 0.0)
        # V=0 here, so V_matrix must equal the folded mu_shift exactly
        @test cs.V_matrix == cs.mu_shift
    end

    # ---------- interaction-only: binary {0,U} interaction, NO mu shift ----------
    for f in (0.1, 0.25, 0.5)
        cs = Const_Struct(base_sim(mu_disorder="interaction-only", f=f, dseed=7))
        n_zero = floor(Int, f * N)
        @test count(cs.U_params .== 0.0) == n_zero
        @test count(cs.U_params .== U)   == N - n_zero
        @test all(==(0.0), cs.mu_shift)               # chemical potential stays uniform
        @test all(==(0.0), cs.V_matrix)               # V=0 and no mu shift folded in
    end

    # ---------- potential-only: uniform U, only mu is disordered ----------
    f = 0.4
    cs_po = Const_Struct(base_sim(mu_disorder="potential-only", f=f))
    n_zero = floor(Int, f * N)
    @test all(==(U), cs_po.U_params)                  # interaction stays uniform
    @test count(cs_po.mu_shift .== U / 2) == N - n_zero
    @test count(cs_po.mu_shift .== 0.0)   == n_zero
    @test cs_po.V_matrix == cs_po.mu_shift

    # ---------- the trio shares ONE site partition at fixed (f, disorder_seed) ----------
    # Empty sites = {U_i=0} for the interaction modes, and {mu unshifted} for the mu modes.
    ff = 0.3; ds = 55
    cs_int  = Const_Struct(base_sim(mu_disorder="interaction-only", f=ff, dseed=ds))
    cs_cpl  = Const_Struct(base_sim(mu_disorder="coupled",          f=ff, dseed=ds))
    cs_pot  = Const_Struct(base_sim(mu_disorder="potential-only",   f=ff, dseed=ds))
    empty_int = Set(findall(==(0.0), cs_int.U_params))     # U_i=0 sites
    empty_cpl = Set(findall(==(0.0), cs_cpl.U_params))     # U_i=0 sites
    empty_pot = Set(findall(==(0.0), cs_pot.mu_shift))     # mu-unshifted sites
    @test empty_int == empty_cpl                            # same partition (U on/off)
    @test empty_int == empty_pot                            # same partition (mu shift)

    # ---------- the BdG diagonal actually carries mu - U_i/2 ----------
    # mu=0, phi=0, V=0  =>  H[i,i] = -(mu - V_matrix[i]) = mu_shift[i]
    sim_h = base_sim(mu_disorder="coupled", f=0.3, mu=0.0, dseed=7)
    data = Lattice_Data(sim_h)
    H = make_H_matrix_K(data, data.current_state.Delta_s,
                        data.current_state.Delta_dx, data.current_state.Delta_dy)
    ms = data.const_struct.mu_shift
    for i in 1:N
        @test real(H[i, i])       ≈  ms[i]  atol=1e-12   # electron block
        @test real(H[i+N, i+N])   ≈ -ms[i]  atol=1e-12   # hole block
    end
    # empty sites (U_i=0) carry NO shift -> diagonal is exactly mu (=0)
    @test all(real(H[i, i]) == 0.0 for i in 1:N if data.const_struct.U_params[i] == 0.0)

    # ---------- reproducibility: partition follows disorder_seed only ----------
    a = Const_Struct(base_sim(mu_disorder="coupled", f=0.3, seed=1, dseed=42))
    b = Const_Struct(base_sim(mu_disorder="coupled", f=0.3, seed=999, dseed=42))  # different MC seed
    c = Const_Struct(base_sim(mu_disorder="coupled", f=0.3, seed=1, dseed=43))    # different disorder seed
    @test a.mu_shift == b.mu_shift          # same disorder seed => identical sample
    @test a.mu_shift != c.mu_shift          # different disorder seed => different sample

    # ---------- error paths ----------
    @test_throws ErrorException Const_Struct(base_sim(mu_disorder="coupled", f=0.3, pairing="d-wave"))
    @test_throws ErrorException Const_Struct(base_sim(mu_disorder="bogus", f=0.3))
    @test_throws ErrorException Const_Struct(base_sim(mu_disorder="coupled", f=1.5))

    # ---------- end-to-end: coupled disorder runs through the full MC loop ----------
    using Dates
    out = joinpath(@__DIR__, "..", "local_smoke_runs", "mu_e2e_" * Dates.format(Dates.now(), "HHMMSS"))
    mkpath(out)
    sim_e2e = SimData(
        Lx=4, Ly=4, SC_L=1, U=5.0, t=1.0, V=0.0, seed=7, mu=0.0, beta=6.0,
        num_measure=5, num_thermal=3, measuring_interval=1, saving_interval=5,
        w_list=[0.0], guess="uniform", U_disorder="uniform", diag_method="K0",
        pairing_symmetry="s-wave", directory=out, num_cores=1, num_threads=1,
        w_conductivity_range=[0.0], delta_values_fixed=Float64[], delta_values_stiffness=Float64[],
        dw_values=Float64[], U_disorder_params=[0.0, 10.0, 0.7, 0.7], U_impurity=0.0,
        TCluster_size_x=1, TCluster_size_y=1, smooth_U=(false, 0.8), loop_random=true,
        B=0.0, gauge_choice="LandauX", boundary_conditions="PBC", compute_jP=false,
        alpha=1.0, beta_phi=0.5, Q_density=[0.0, 0.0], snapshot_interval=0,
        mu_disorder="coupled", mu_disorder_f=0.25,
    )
    res = MonteCarloSimulation(sim_e2e, out; saveplots=false, savedata=true, saving_interval=5,
                               dump_time=Dates.Hour(1), sim_info_path=joinpath(out, "sim.jld2"),
                               log_filename=joinpath(out, "log.txt"))
    @test res.current_state.num_current_measurments == 5
    # empty sites (U_i=0) are never updated: Delta_s and phi stay exactly 0 there
    dead = findall(==(0.0), res.const_struct.U_params)
    @test !isempty(dead)
    @test all(res.current_state.Delta_s[dead] .== 0.0 + 0.0im)
    @test all(res.current_state.phi[dead] .== 0.0)
    rm(out; recursive=true, force=true)

    println("OK mu-U disorder test")
end
