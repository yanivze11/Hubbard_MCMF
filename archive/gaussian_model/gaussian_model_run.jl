# ============================================================================
# gaussian_model_run.jl — the "Gaussian model" for the attractive-Hubbard MCMF
# system at U=4, n=0.875, L=16.
#
# IDEA. Instead of running the full Monte-Carlo over the Hubbard-Stratonovich
# pairing field, we SAMPLE Δ_i directly from an uncorrelated Gaussian:
#     Re[Δ_i], Im[Δ_i]  iid ~ N(0, σ²)   on every site,
# build the BdG (K0) Hamiltonian for each sample, diagonalize it, and measure
# ALL observables with the main code's own `make_measurement_K!`, averaging over
# samples. The amplitude is pinned to the measured high-T pairing amplitude:
#     σ = sqrt(2/π) · Δ_p(T)     ⇔     Δ_p = ⟨|Δ_i|⟩ = σ·sqrt(π/2)   (Rayleigh mean)
# so the model's own ⟨|Δ_i|⟩ reproduces the measured `Delta_MC_abs`.
#
# This mirrors `sample_rpaII` from notebooks/disorder_p_comparison_notebook.ipynb
# (with one fix: `make_measurement_K!` does NOT bump the measurement counter — the
# MC loop normally does — so we set it by hand, else `get_means` sees no data).
#
# DECISIONS (locked by the user):
#   1. Δ_p(T) FROM DATA: measured `Delta_MC_abs` of the real homogeneous MCMF run
#      (U=4, n=0.875) at T = 0.5, 1.0, 2.0, 4.0 (Δ_p is ~L-independent; read at L=16).
#   2. μ(T) SELF-CONSISTENT FOR THE MODEL: for each T, root-find μ so that the
#      GAUSSIAN MODEL itself has ⟨n⟩ = 0.875 (NOT the real run's best_mu).
#
# OUTPUT. A sim_info .jld2 whose entries mirror the real-run schema field-for-
# field, so the existing get_sims / get_obs_vs_T / means-store pipeline reads it
# like a real run. Written to a NEW dir (cluster_results/ is read-only):
#     gaussian_model_results/sim_info_gaussian_model.jld2
#
# Run as a script:   julia gaussian_model_run.jl
# Eventually feeds thesis figure fig:DOS_gaussian (ch.5) — model DOS vs measured
# high-T DOS. This script just produces the data file (+ validates it).
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "data_summery.jl"))   # MC_Hubburd.jl + sim_num_control.jl: SimData, get_means,
                                                 # get_sims_all, get_obs_vs_T (NO Plots beyond MC_Hubburd's)
include(joinpath(@__DIR__, "..", "..", "means_store.jl"))    # load_subset (only used by the one-off Δ_p extraction)

using Random, Statistics, Distributions, JLD2, Dates, LinearAlgebra

# --------------------------------------------------------------------------
# Fixed run parameters
# --------------------------------------------------------------------------
const GM_U            = 4.0
const GM_N            = 0.875
const GM_L            = 16
const GM_TS           = [0.5, 1.0, 2.0, 4.0]          # temperatures
const GM_WS           = collect(-5.0:0.1:5.0)         # DOS energy grid (101 pts; smooth + fast)

# Δ_p(T) = measured Delta_MC_abs (decision 1), extracted ONCE (verified, full
# precision) so the heavy per-T runs need NOT load the means store — that load
# was the main memory source behind the LLVM bad_alloc crashes.
const GM_DELTAS = Dict(0.5 => 2.01897173782932, 1.0 => 2.476387218004027,
                       2.0 => 3.084972665857382, 4.0 => 3.992057196844593)

# Sampling sizes: a cheaper sweep during the μ root-find, a big final run.
const GM_NUM_SAMPLES_MU    = 120                      # μ search (deterministic seed)
const GM_NUM_SAMPLES_FINAL = 300                      # final, smooth-DOS run (cheap in fast mode)
const GM_SEED              = 20260614                 # fixed -> reproducible field draws

# Fast DOS-focused mode: measure DOS, n, P_s, χ_spin/χ_charge, Ps_l, Phase_Corr and
# thermodynamics, but SKIP the expensive transport/spectral observables (spectral
# function over the k-path, Kx/Ky, current-current correlators, conductivity
# tensors). fig:DOS_gaussian needs none of those. ~5-10× faster per sample.
const GM_FAST_DOS = true

const GM_SOURCE  = "sim_info_FullSys_TCA__27-03-2026_CURIE.jld2"
const GM_OUTDIR  = joinpath(@__DIR__, "gaussian_model_results")
const GM_OUTFILE = joinpath(GM_OUTDIR, "sim_info_gaussian_model.jld2")

# n-search tolerances
const GM_NTOL_SEARCH = 0.003
const GM_NTOL_FINAL  = 0.010

# --------------------------------------------------------------------------
# 1. Δ_p(T) from the real MCMF data (decision 1)
# --------------------------------------------------------------------------
# Scan the loaded subset for (U=4, L=16, n≈0.875) entries, build beta -> Δ_p from
# the measured observable "Delta_MC_abs" (a length-1 vector), keeping the entry
# whose filling is closest to 0.875 at each beta.
function delta_p_vs_beta(; U=GM_U, L=GM_L, n=GM_N)
    # run_extrapolation=false: we only read raw means (Delta_MC_abs, n), not the
    # *_extrap transport observables, so skip the expensive on-load δ→0 fit.
    si = load_subset(; source=[GM_SOURCE], U=U, n=n, run_extrapolation=false)
    best = Dict{Float64,Tuple{Float64,Float64}}()   # beta -> (delta_p, n_dist)
    for (k, e) in si
        k == "updated_at" && continue
        (e isa AbstractDict) || continue
        Lx, Ly = get_sim_dimensions(e)
        (Lx == L && Ly == L) || continue
        beta = get(e, "beta", nothing); beta === nothing && continue
        m = get(e, "means", nothing); (m isa AbstractDict && haskey(m, "Delta_MC_abs")) || continue
        haskey(m, "n") || continue
        nval = m["n"]["value"]
        dval = m["Delta_MC_abs"]["value"]
        dp = dval isa AbstractVector ? Float64(dval[1]) : Float64(dval)
        nd = abs(nval - n)
        if !haskey(best, beta) || nd < best[beta][2]
            best[beta] = (dp, nd)
        end
    end
    return Dict(b => v[1] for (b, v) in best)
end

# nearest-beta lookup (target betas are present exactly, but be robust)
function delta_p_for_T(dp_map::Dict{Float64,Float64}, T::Float64)
    beta = 1 / T
    betas = collect(keys(dp_map))
    isempty(betas) && error("no Δ_p data found")
    b = betas[argmin(abs.(betas .- beta))]
    return dp_map[b], b
end

# --------------------------------------------------------------------------
# 2. The Gaussian sampler  (mirrors sample_rpaII; reusable lat_data per T)
# --------------------------------------------------------------------------
# Build a fresh LatticeData configured for amplitude Δ_p at chemical potential mu.
# `dist` per-component std = 0.05·mean(U_params) = 0.05·(σ/0.05) = σ (verified).
function make_model_lat(delta::Float64; mu::Float64, num_samples::Int, seed::Int=GM_SEED)
    sigma = sqrt(2 / pi) * delta
    sim_data = SimData(Lx=GM_L, saving_interval=num_samples, w_list=GM_WS,
                       mu=mu, U=sigma / 0.05, seed=seed,
                       U_disorder="uniform", diag_method="K0",
                       pairing_symmetry="s-wave")
    lat = Lattice_Data(sim_data)
    fill!(lat.current_state.Delta_dx, 0.0)   # Gaussian model: s-wave only
    fill!(lat.current_state.Delta_dy, 0.0)
    return lat, sigma
end

# Draw one Gaussian Δ_s configuration into `ds` (zero d-wave throughout).
@inline function draw_delta_field!(lat, ds, l2)
    sug = lat.temporary_data.Delta_suggestion
    @inbounds for i in 1:l2
        rand!(lat.const_struct.dist, sug)
        ds[i] = sug[1] + 1im * sug[2]
    end
    return ds
end

# FAST filling: redraw fields, diagonalize, average ⟨n⟩ over samples WITHOUT the
# (expensive) full measurement. Reseeds so n(μ) is a deterministic, monotone
# function of μ for a clean bisection. Mutates lat.sim_data.mu.
function model_filling(lat; mu::Float64, num_samples::Int, seed::Int=GM_SEED)
    lat.sim_data.mu = mu
    l2 = GM_L^2
    ds = lat.current_state.Delta_s
    dx = lat.current_state.Delta_dx
    dy = lat.current_state.Delta_dy
    Random.seed!(seed)
    nsum = 0.0
    for _ in 1:num_samples
        draw_delta_field!(lat, ds, l2)
        _, E, Umat = MCMF_action_K0(lat, ds, dx, dy)
        lat.current_state.Energies .= E
        lat.current_state.U_matrix .= Umat
        get_FD_dist!(lat)                 # build Fermi factors at this μ, β
        nsum += avrage_n_K(lat)
    end
    return nsum / num_samples
end

# Self-consistent μ by bisection: find μ with model ⟨n⟩ = target_n. n increases
# monotonically with μ. Expands the bracket outward if not initially bracketing.
function solve_mu(lat; target_n::Float64=GM_N, num_samples::Int=GM_NUM_SAMPLES_MU,
                  seed::Int=GM_SEED, mu_lo=-6.0, mu_hi=1.0, tol=GM_NTOL_SEARCH, maxit=40)
    f(mu) = model_filling(lat; mu=mu, num_samples=num_samples, seed=seed) - target_n
    flo = f(mu_lo); fhi = f(mu_hi)
    expand = 0
    while flo > 0 && expand < 6   # n(mu_lo) already above target -> lower the floor
        mu_lo -= 2.0; flo = f(mu_lo); expand += 1
    end
    while fhi < 0 && expand < 12  # n(mu_hi) still below target -> raise the ceiling
        mu_hi += 2.0; fhi = f(mu_hi); expand += 1
    end
    (flo <= 0 <= fhi) || error("could not bracket μ for n=$target_n (flo=$flo, fhi=$fhi)")
    mu = 0.5 * (mu_lo + mu_hi); fm = 0.0
    for it in 1:maxit
        mu = 0.5 * (mu_lo + mu_hi)
        fm = f(mu)
        @printf("      [μ-bisect %2d] μ=% .5f  ⟨n⟩=% .5f  (Δn=% .1e)\n", it, mu, fm + target_n, fm); flush(stdout)
        abs(fm) <= tol && break
        if fm > 0; mu_hi = mu; else; mu_lo = mu; end
    end
    return mu, fm + target_n
end

# FULL run at converged μ: redraw + diagonalize + FULL measurement of all
# observables, then set the measurement counter that make_measurement_K! omits.
function run_full_model(lat; mu::Float64, num_samples::Int=GM_NUM_SAMPLES_FINAL, seed::Int=GM_SEED)
    lat.sim_data.mu = mu
    l2 = GM_L^2
    ds = lat.current_state.Delta_s
    dx = lat.current_state.Delta_dx
    dy = lat.current_state.Delta_dy
    Random.seed!(seed)
    for samp in 1:num_samples
        draw_delta_field!(lat, ds, l2)
        _, E, Umat = MCMF_action_K0(lat, ds, dx, dy)
        lat.current_state.Energies .= E
        lat.current_state.U_matrix .= Umat
        make_measurement_K!(lat, samp)
    end
    # FIX: make_measurement_K! never bumps this (the MC loop does). get_means needs it.
    lat.current_state.num_current_measurments = num_samples
    return lat
end

# --------------------------------------------------------------------------
# LEAN measurement (fast DOS-focused mode): a trimmed copy of make_measurement_K!
# (MC_Hubburd.jl:1621) that keeps ONLY the cheap observables and skips the
# expensive transport/spectral ones. Reuses the main code's helpers verbatim.
#   KEEP : n, DOS/LDOS, DdD (→Ps), Ps_l, χ_spin, χ_charge, Phase_Corr,
#          Delta_mean/operator/MC/MC_abs, energy_spacing, action
#   SKIP : spectral function A & BZ, Kx/Ky, current-current correlators,
#          conductivity tensors, L_D, jP  (the ~95% cost; unused by fig:DOS_gaussian)
# --------------------------------------------------------------------------
function make_measurement_dos!(data::LatticeData, i::Int64)
    i > data.sim_data.saving_interval && (i = data.sim_data.saving_interval)
    ensure_global_diag_synced!(data; reason="measurement")   # diagonalize current Δ field (1 eig)
    get_FD_dist!(data)
    compute_delta_analitic_continuation!(data; method="mean_diff_low")

    m  = data.measurement_data
    N  = data.sim_data.N
    SC_L = data.sim_data.SC_L
    N_gen = (data.sim_data.Lx * SC_L) * (data.sim_data.Ly * SC_L)

    m.Ss[i] = data.current_state.S

    Energies = data.current_state.Energies
    diffs = diff(sort(vec(Energies)))
    m.energy_spacing_mean[i] = isempty(diffs) ? 0.0 : mean(diffs)
    m.delta_mean_diff_low[i] = data.temporary_data.delta

    m.n[i] = avrage_n_K(data)

    # DOS / LDOS over the ω-grid (the part of the w-loop we keep; skip spectral fn)
    for (j, w) in enumerate(data.sim_data.w_list)
        LDOS_c = compute_full_LDOS(data, w; delt=data.temporary_data.delta_max)
        m.LDOS[j] .*= i-1; m.LDOS[j] .+= LDOS_c; m.LDOS[j] ./= i
        m.DOS[i][j] += sum(LDOS_c) / N_gen
    end

    # cheap thermodynamic / correlation observables
    m.DdD[i] = compute_D_dagger_D_K(data)                      # → Ps = 2⟨DdD⟩-n+1
    for (l_ix, l) in enumerate(data.const_struct.ls_list)
        m.Ps_l[i][l_ix] = compute_Ps_l_K(data, l)
    end
    susceptibility_q_vec_both!(data, data.const_struct.q_list_sus, m.chi_spin[i], m.chi_charge[i])
    for (l_ix, l) in enumerate(data.const_struct.ls_list_reduced)
        m.Phase_Corr[i][l_ix] = compute_phase_correlation(data, l; pairing_field=:Delta_s)
    end

    m.Delta_mean[i]       = mean(data.current_state.Delta_s)
    m.Delta_operator[i]   = compute_Delta_operator(data)
    m.Delta_MC_abs[i][1]  = mean(abs.(data.current_state.Delta_s))
    m.Delta_MC[i][1]      = mean(data.current_state.Delta_s)
    m.U4[i]               = data.current_state.S_K0
    return m
end

# Fast full run: draw field into current_state, force a re-diagonalize, lean measure.
function run_full_model_dos(lat; mu::Float64, num_samples::Int=GM_NUM_SAMPLES_FINAL, seed::Int=GM_SEED)
    lat.sim_data.mu = mu
    l2 = GM_L^2
    ds = lat.current_state.Delta_s
    Random.seed!(seed)
    for samp in 1:num_samples
        draw_delta_field!(lat, ds, l2)            # fills current_state.Delta_s
        lat.current_state.diag_is_synced = false  # force ensure_global_diag_synced! to diagonalize
        make_measurement_dos!(lat, samp)
    end
    lat.current_state.num_current_measurments = num_samples
    return lat
end

# --------------------------------------------------------------------------
# verify the dist really yields Re/Im ~ N(0, σ²) and ⟨|Δ|⟩ ≈ Δ_p
# --------------------------------------------------------------------------
function verify_gaussian_dist(delta::Float64; n=40000, seed::Int=GM_SEED)
    lat, sigma = make_model_lat(delta; mu=0.0, num_samples=2, seed=seed)
    Random.seed!(seed)
    re = Vector{Float64}(undef, n); im = Vector{Float64}(undef, n); ab = Vector{Float64}(undef, n)
    sug = lat.temporary_data.Delta_suggestion
    for k in 1:n
        rand!(lat.const_struct.dist, sug)
        re[k] = sug[1]; im[k] = sug[2]; ab[k] = hypot(sug[1], sug[2])
    end
    @printf("  [verify] Δ_p=%.4f -> σ_target=√(2/π)Δ_p=%.4f | std(Re)=%.4f std(Im)=%.4f | ⟨|Δ|⟩=%.4f (Δ_p=%.4f)\n",
            delta, sigma, std(re; corrected=false), std(im; corrected=false), mean(ab), delta)
    return nothing
end

# --------------------------------------------------------------------------
# Build a sim_info entry that mirrors the real-run schema field-for-field
# --------------------------------------------------------------------------
function build_entry(lat, T::Float64, delta::Float64, sigma::Float64, mu::Float64,
                     n_final::Float64, num_samples::Int, seed::Int, sim_num::Int)
    means = get_means(lat)
    isa(means, String) && error("get_means failed: $means")
    beta = 1 / T
    e = Dict{String,Any}(
        # --- geometry / model identity (read by get_sims/group_by_*) ---
        "Lx" => GM_L, "Ly" => GM_L, "SC_L" => 1,
        "U" => GM_U,                       # physical interaction (for grouping/provenance)
        "U_disorder" => "uniform",
        "diag_method" => "K0",
        "TClustersize" => GM_L,            # K0: full system (== L)
        "pairing_symmetry" => "s-wave",
        "boundary_condition" => "PBC",
        "V" => 0.0, "B" => 0.0, "t" => 1.0, "impurity" => 0.0,
        # --- thermodynamic point ---
        "beta" => beta, "mu" => mu,
        "wanted_n" => GM_N, "n_tol" => GM_NTOL_FINAL,
        # --- the measurements (all observables) ---
        "means" => means,
        "w_list" => copy(GM_WS),
        # --- bookkeeping ---
        "num_measure" => num_samples,
        "num_thermal" => 0,
        "measuring_interval" => 1,
        "seeds" => [seed],
        "sim_num" => sim_num,
        "cores" => 1,
        "start_time" => Dates.now(),
        # --- μ self-consistency record (mirrors real "convergence" block) ---
        "convergence" => Dict{String,Any}(
            "best_mu" => mu, "best_n" => n_final, "target_n" => GM_N,
            "converged" => abs(n_final - GM_N) <= GM_NTOL_FINAL,
            "n_tol" => GM_NTOL_FINAL, "updated_at" => Dates.now(),
            "method" => "gaussian_model_self_consistent_mu"),
        # --- Gaussian-model specific metadata ---
        "model" => "gaussian_pairing",
        "gaussian_delta_p" => delta,       # input Δ_p (= measured Delta_MC_abs)
        "gaussian_sigma" => sigma,         # per-component std σ = √(2/π)Δ_p
        "gaussian_num_samples" => num_samples,
        "gaussian_sim_U" => sigma / 0.05,  # SimData.U used only to width the dist
        "gaussian_source" => GM_SOURCE,
    )
    return e
end

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
function main()
    mkpath(GM_OUTDIR)
    prog = open(joinpath(GM_OUTDIR, "gaussian_model_progress.txt"), "w")
    pmark(s) = (println(prog, string(round(time(); digits=1), "  ", s)); flush(prog))
    println("="^78)
    println("Gaussian model build — U=$GM_U, n=$GM_N, L=$GM_L, T=$(GM_TS)")
    println("  μ-search samples=$(GM_NUM_SAMPLES_MU), final samples=$(GM_NUM_SAMPLES_FINAL)")
    println("="^78); flush(stdout)
    pmark("start")

    # 1) Δ_p(T) from data
    println("\n[1] Δ_p(T) from measured Delta_MC_abs ($GM_SOURCE, U=$GM_U, L=$GM_L, n≈$GM_N):")
    dp_map = delta_p_vs_beta()
    deltas = Dict{Float64,Float64}()
    for T in GM_TS
        dp, b = delta_p_for_T(dp_map, T)
        deltas[T] = dp
        @printf("    T=%.2f (β=%.3f -> matched β=%.3f):  Δ_p = %.5f\n", T, 1/T, b, dp)
    end
    flush(stdout); pmark("delta_p done: $(round.([deltas[T] for T in GM_TS]; digits=4))")

    # quick distribution sanity check (uses the lowest-T amplitude)
    println("\n[2] dist sanity check (Re/Im ~ N(0,σ²), ⟨|Δ|⟩ ≈ Δ_p):")
    verify_gaussian_dist(deltas[GM_TS[1]])

    # 3) per-T: solve μ, then full sampling run, then build entry
    sim_info = Dict{String,Any}("updated_at" => Dates.now())
    sim_num = 0
    for T in GM_TS
        delta = deltas[T]
        println("\n" * "-"^70)
        @printf("[T = %.2f]  Δ_p = %.5f  (σ = %.5f)\n", T, delta, sqrt(2/pi)*delta); flush(stdout)
        pmark("T=$T begin (Δ_p=$(round(delta;digits=4)))")

        # μ root-find on a cheap reusable lattice
        lat_mu, _ = make_model_lat(delta; mu=-0.9, num_samples=GM_NUM_SAMPLES_MU)
        println("    solving μ for ⟨n⟩=$GM_N (model self-consistent):"); flush(stdout)
        mu_star, n_search = solve_mu(lat_mu; target_n=GM_N)
        @printf("    -> μ* = %.5f   (⟨n⟩_search = %.5f)\n", mu_star, n_search); flush(stdout)
        pmark("T=$T μ*=$(round(mu_star;digits=5)) n_search=$(round(n_search;digits=5))")

        # full high-sample run at μ*
        @printf("    full run: %d samples, full measurement ...\n", GM_NUM_SAMPLES_FINAL); flush(stdout)
        lat, sigma = make_model_lat(delta; mu=mu_star, num_samples=GM_NUM_SAMPLES_FINAL)
        t0 = time()
        run_full_model(lat; mu=mu_star, num_samples=GM_NUM_SAMPLES_FINAL)
        n_final = get_means(lat)["n"]["value"]
        @printf("    full run done in %.1fs:  ⟨n⟩_final = %.5f  (target %.3f, Δ=%.1e)\n",
                time()-t0, n_final, GM_N, n_final-GM_N); flush(stdout)
        pmark("T=$T full done in $(round(time()-t0;digits=1))s n_final=$(round(n_final;digits=5))")
        if abs(n_final - GM_N) > GM_NTOL_FINAL
            @warn @sprintf("⟨n⟩_final off target at T=%.2f: %.4f (|Δ|=%.4f > %.4f)",
                           T, n_final, abs(n_final-GM_N), GM_NTOL_FINAL)
        end

        sim_num += 1
        sim_info[string(sim_num)] = build_entry(lat, T, delta, sigma, mu_star,
                                                n_final, GM_NUM_SAMPLES_FINAL, GM_SEED, sim_num)
    end

    # 4) save to NEW dir
    mkpath(GM_OUTDIR)
    JLD2.save(GM_OUTFILE, sim_info)
    @printf("\n[saved] %d entries -> %s (%.1f MB)\n",
            sim_num, GM_OUTFILE, filesize(GM_OUTFILE)/2^20); flush(stdout)
    pmark("saved $sim_num entries -> $GM_OUTFILE")

    # 5) validate: reload through the normal loader and exercise get_obs_vs_T
    validate(GM_OUTFILE)
    pmark("DONE (validated)")
    close(prog)
    return GM_OUTFILE
end

# --------------------------------------------------------------------------
# Validation: reload like a real run; confirm get_sims_all / get_obs_vs_T work
# --------------------------------------------------------------------------
function validate(path::String)
    println("\n" * "="^78)
    println("[validate] reloading $(basename(path)) through the normal pipeline")
    println("="^78)
    si = JLD2.load(path)
    sims_all = get_sims_all(si; wanted_n=GM_N)
    haskey(sims_all, GM_U) || (println("  !! U=$GM_U not found in get_sims_all"); return)
    sim, method = get_sim_from_sims_all(sims_all, GM_U, GM_L, GM_L)
    if sim === nothing
        println("  !! could not retrieve (U=$GM_U, L=$GM_L) sim from sims_all"); return
    end
    println("  retrieved sim (method=$method) with $(length(sim)) temperatures")
    for obs in ["n", "DOS", "Ps", "chi_spin"]
        try
            Ts, vals, errs = get_obs_vs_T(sim, obs)
            order = sortperm(Ts)
            Ts = Ts[order]; vals = vals[order]
            if obs == "DOS"
                # DOS is a vector per T -> report ω=0 value (gap proxy)
                z = argmin(abs.(GM_WS))
                dos0 = [Float64(v[z]) for v in vals]
                @printf("    %-9s OK: T=%s  DOS(ω≈0)=%s\n", obs,
                        string(round.(Ts, digits=3)), string(round.(dos0, digits=4)))
            else
                sval = [v isa AbstractVector ? round(Float64(v[1]), digits=4) : round(Float64(v), digits=4) for v in vals]
                @printf("    %-9s OK: T=%s  val=%s\n", obs, string(round.(Ts, digits=3)), string(sval))
            end
        catch err
            @printf("    %-9s FAILED: %s\n", obs, err)
        end
    end

    # filling check
    try
        Ts, ns, _ = get_obs_vs_T(sim, "n")
        @printf("  ⟨n⟩ per T: %s  (target %.3f)\n", string(round.(Float64.(ns), digits=4)), GM_N)
    catch; end

    # optional: compare to measured high-T DOS CSV (eventual fig:DOS_gaussian)
    csv = joinpath(@__DIR__, "final_plots", "data", "ch05", "DOS_vs_T_L20_n0.875.csv")
    println(isfile(csv) ? "  measured-DOS CSV present for later overlay: $csv" :
                          "  (measured-DOS CSV not found at $csv — overlay is a later step)")
    println("[validate] done.")
end

# --------------------------------------------------------------------------
# Per-temperature isolated runner (robust against the transient LLVM bad_alloc:
# each T runs in its own short-lived, low-memory process and is saved to its own
# file immediately, so a crash costs at most one T's rerun — never the batch).
# --------------------------------------------------------------------------
function run_one_T_to_file(T::Float64, delta::Float64, outfile::String)
    @printf("[one-T] T=%.2f  Δ_p=%.5f  (σ=%.5f)  samples=%d  ws=%d\n",
            T, delta, sqrt(2/pi)*delta, GM_NUM_SAMPLES_FINAL, length(GM_WS)); flush(stdout)
    mkpath(dirname(outfile))
    prog = open(replace(outfile, r"\.jld2$" => "") * "_progress.txt", "w")
    pmark(s) = (println(prog, string(round(time(); digits=1), "  ", s)); flush(prog))
    pmark("start T=$T")

    # μ self-consistency (cheap, diag-only)
    lat_mu, _ = make_model_lat(delta; mu=-0.9, num_samples=GM_NUM_SAMPLES_MU)
    mu_star, n_search = solve_mu(lat_mu; target_n=GM_N)
    @printf("[one-T] μ*=%.5f  ⟨n⟩_search=%.5f\n", mu_star, n_search); flush(stdout)
    pmark("μ*=$(round(mu_star;digits=5)) n_search=$(round(n_search;digits=5))")
    lat_mu = nothing; GC.gc()   # free before the heavy run

    # full high-sample measurement (fast DOS-focused or full observable suite)
    lat, sigma = make_model_lat(delta; mu=mu_star, num_samples=GM_NUM_SAMPLES_FINAL)
    t0 = time()
    if GM_FAST_DOS
        run_full_model_dos(lat; mu=mu_star, num_samples=GM_NUM_SAMPLES_FINAL)
    else
        run_full_model(lat; mu=mu_star, num_samples=GM_NUM_SAMPLES_FINAL)
    end
    n_final = get_means(lat)["n"]["value"]
    @printf("[one-T] full run done in %.1fs  ⟨n⟩_final=%.5f (Δ=%.1e)  [%s]\n",
            time()-t0, n_final, n_final-GM_N, GM_FAST_DOS ? "fast-DOS" : "full-suite"); flush(stdout)
    pmark("full done in $(round(time()-t0;digits=1))s n_final=$(round(n_final;digits=5))")
    abs(n_final - GM_N) > GM_NTOL_FINAL &&
        @warn @sprintf("⟨n⟩_final off target at T=%.2f: %.4f", T, n_final)

    e = build_entry(lat, T, delta, sigma, mu_star, n_final, GM_NUM_SAMPLES_FINAL, GM_SEED, 1)
    JLD2.save(outfile, Dict{String,Any}("entry" => e, "T" => T))
    @printf("[one-T] saved T=%.2f -> %s\n", T, outfile); flush(stdout)
    pmark("saved -> $outfile")
    close(prog)
    return outfile
end

# Merge per-T entry files into one sim_info .jld2 (real-run schema) + validate.
function merge_and_validate(entryfiles::Vector{String}, final::String)
    sim_info = Dict{String,Any}("updated_at" => Dates.now())
    k = 0
    for f in entryfiles
        if !isfile(f); @printf("  [merge] MISSING %s — skipping\n", f); continue; end
        d = JLD2.load(f)
        e = d["entry"]; k += 1; e["sim_num"] = k
        sim_info[string(k)] = e
        @printf("  [merge] + T=%.2f from %s (entry %d)\n", get(d, "T", NaN), basename(f), k)
    end
    mkpath(dirname(final))
    JLD2.save(final, sim_info)
    @printf("[merge] %d entries -> %s (%.1f MB)\n", k, final, filesize(final)/2^20)
    validate(final)
    return final
end

# --------------------------------------------------------------------------
# CLI dispatch
#   julia -O1 gaussian_model_run.jl one-T <T> <delta> <outfile>
#   julia -O1 gaussian_model_run.jl merge <entry1.jld2> ... <final.jld2>
#   julia -O1 gaussian_model_run.jl            # single-process main() (legacy)
# --------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "one-T"
        run_one_T_to_file(parse(Float64, ARGS[2]), parse(Float64, ARGS[3]), ARGS[4])
    elseif !isempty(ARGS) && ARGS[1] == "merge"
        merge_and_validate(String.(ARGS[2:end-1]), ARGS[end])
    else
        main()
    end
end
