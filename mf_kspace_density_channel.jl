# =============================================================================
# mf_kspace_density_channel.jl
#
# Standalone, fast, homogeneous (no-disorder) k-space mean-field solver for the
# COUPLED pairing (Delta) + density (phi) channels of the attractive Hubbard
# model, restricted to ordering vector Q = (0,0).
#
# Source of truth (derivation, signs, saddle equations):
#   Thesis_Obsidian_Vault/Habburd_Project/decoupling in the density.md
#   (sections "# minimum", "## the c-number (energy) term", and
#    "## homogeneous k-space (no disorder), Q=0").
#
# It self-consistently solves, for uniform Delta and uniform phi:
#
#     eps_k = -2t(cos kx + cos ky)          (dispersion, matches MC_Hubburd.jl)
#     xi_k  = eps_k - mu - phi              (phi is a Hartree chemical-pot shift)
#     E_k   = sqrt(xi_k^2 + |Delta|^2)
#
#     gap:  2/(alpha*U) = (1/N) sum_k tanh(beta E_k/2)/E_k
#     n     = (1/N) sum_k [1 - (xi_k/E_k) tanh(beta E_k/2)]
#     phi   = beta_phi*U * n                              (POSITIVE coefficient)
#
# and returns the density n and (optionally) the chemical potential mu that
# reproduces a target density. The mean-field free energy F = S/beta uses the
# note's boxed action
#
#   S = (beta/(alpha U)) sum_i|Delta_i|^2 + (beta/(2 beta_phi U)) sum_i phi_i^2
#       - ln Z'_e - beta sum_i cos(Q.r_i) phi_i .
#
# NOTE on parameters (current convention): alpha and beta_phi are the FRACTIONS
# of U in the pairing and density channels (pairing strength alpha*U, density
# strength beta_phi*U); the density penalty is phi^2/(2 beta_phi U). A strict
# single-U partition is alpha + beta_phi = 1; the EQUAL split (equal penalty
# denominators alpha U = 2 beta_phi U) is alpha = 2/3, beta_phi = 1/3. Also
# distinct beta's -- `beta` is the INVERSE TEMPERATURE, unrelated to beta_phi.
#
# IMPORTANT -- MF is NOT the MC result. This minimizes the mean-field free
# energy over UNIFORM fields with no disorder. Use it to (1) seed the MC's Delta
# and phi fields and (2) provide a starting mu for occupation convergence --
# NOT as a check against the full Monte-Carlo equilibrium.
#
# Usage:
#     include("mf_kspace_density_channel.jl")
#     using .MFKSpaceDensityChannel
#     res = solve_mf(; t=1.0, U=4.0, alpha=1.0, beta_phi=1.0, beta=10.0,
#                      Lx=8, Ly=8, mu=0.0, Delta0=0.5, phi0=0.0)
#     res = solve_mf(; t=1.0, U=4.0, alpha=2/3, beta_phi=1/3, beta=10.0,
#                      Lx=8, Ly=8, target_n=1.0)   # solves for mu
#
# This module deliberately does NOT depend on MC_Hubburd.jl (kept decoupled so
# the eventual MC wiring is a thin adapter).
# =============================================================================

module MFKSpaceDensityChannel

export solve_mf, mf_free_energy, mf_density, mf_gap_residual

# --- small numerical guard: below this |E| we use the E->0 limits ------------
const _E_FLOOR = 1e-12

"""
    _epsilon_grid(t, Nkx, Nky) -> Vector{Float64}

Dispersion eps_k = -2t(cos kx + cos ky) on the lattice k-grid
kx = 2*pi*m/Nkx (m = 0..Nkx-1), ky = 2*pi*n/Nky. Flattened to a length
Nkx*Nky vector. A denser grid (Nkx > Lx) just refines the BZ sum; the
self-consistency uses (1/N) averages, so results are grid-consistent.
"""
function _epsilon_grid(t::Float64, Nkx::Int, Nky::Int)
    eps = Vector{Float64}(undef, Nkx * Nky)
    idx = 1
    @inbounds for mx in 0:(Nkx - 1)
        kx = 2pi * mx / Nkx
        ckx = cos(kx)
        for ny in 0:(Nky - 1)
            ky = 2pi * ny / Nky
            eps[idx] = -2t * (ckx + cos(ky))
            idx += 1
        end
    end
    return eps
end

"""
    mf_density(eps, mu, phi, Delta, beta) -> n

Uniform-field density  n = (1/N) sum_k [1 - (xi_k/E_k) tanh(beta E_k/2)],
with xi_k = eps_k - mu - phi and E_k = sqrt(xi_k^2 + Delta^2). n in [0, 2].
"""
function mf_density(eps::Vector{Float64}, mu::Float64, phi::Float64,
                    Delta::Float64, beta::Float64)
    Nk = length(eps)
    D2 = Delta * Delta
    acc = 0.0
    @inbounds for k in 1:Nk
        xi = eps[k] - mu - phi
        E = sqrt(xi * xi + D2)
        if E < _E_FLOOR
            # xi = Delta = 0: the (xi/E) tanh term -> 0.
            acc += 1.0
        else
            acc += 1.0 - (xi / E) * tanh(0.5 * beta * E)
        end
    end
    return acc / Nk
end

"""
    mf_gap_residual(eps, mu, phi, Delta, U, alpha, beta) -> r

Residual of the gap equation, r = (1/N) sum_k tanh(beta E_k/2)/E_k - 2/(alpha U).
Zero at self-consistency (for Delta > 0). For Delta = 0 it just reports the
LHS minus RHS (used only diagnostically).
"""
function mf_gap_residual(eps::Vector{Float64}, mu::Float64, phi::Float64,
                         Delta::Float64, U::Float64, alpha::Float64, beta::Float64)
    Nk = length(eps)
    D2 = Delta * Delta
    lhs = 0.0
    @inbounds for k in 1:Nk
        xi = eps[k] - mu - phi
        E = sqrt(xi * xi + D2)
        if E < _E_FLOOR
            lhs += 0.5 * beta                 # tanh(x)/x -> 1 as x->0, * beta/... limit
        else
            lhs += tanh(0.5 * beta * E) / E
        end
    end
    lhs /= Nk
    return lhs - 2.0 / (alpha * U)
end

"""
    _delta_fixed_point(eps, mu, phi, Delta, U, alpha, beta) -> Delta_new

One fixed-point step of the s-wave gap equation:
    Delta_new = alpha*U * (1/N) sum_k [Delta/(2 E_k)] tanh(beta E_k/2).
At the fixed point Delta cancels and this reproduces 2/(alpha U) = <tanh/E>.
Returns 0 if the pairing coupling alpha*U is (numerically) zero.
"""
function _delta_fixed_point(eps::Vector{Float64}, mu::Float64, phi::Float64,
                            Delta::Float64, U::Float64, alpha::Float64, beta::Float64)
    aU = alpha * U
    if aU <= 0.0
        return 0.0
    end
    Nk = length(eps)
    D2 = Delta * Delta
    acc = 0.0
    @inbounds for k in 1:Nk
        xi = eps[k] - mu - phi
        E = sqrt(xi * xi + D2)
        if E < _E_FLOOR
            acc += 0.25 * beta * Delta          # Delta/(2E) tanh(bE/2) -> (beta/4)Delta
        else
            acc += (Delta / (2.0 * E)) * tanh(0.5 * beta * E)
        end
    end
    return aU * acc / Nk
end

"""
    mf_free_energy(eps, mu, phi, Delta, U, alpha, beta_phi, beta; N) -> (F, F_per_site)

Mean-field free energy F = S/beta from the note's boxed action (Q=0, uniform,
no disorder), up to the dropped field-independent constants (-2N ln2 from the
Nambu pairing and -N(2tz+mu)). Per-site:

    f = (1/(alpha U)) Delta^2 + (1/(2 beta_phi U)) phi^2 - phi
        - (2/beta) * (1/N) sum_k ln cosh(beta E_k/2).

The linear "-phi" is the c-number Slin term (note: "Keep the -beta sum cos(Q.r)phi").
Its presence is what makes the phi saddle the physical +phi form. F (total) = N*f.
"""
function mf_free_energy(eps::Vector{Float64}, mu::Float64, phi::Float64,
                        Delta::Float64, U::Float64, alpha::Float64,
                        beta_phi::Float64, beta::Float64; N::Int)
    Nk = length(eps)
    D2 = Delta * Delta
    lncosh_sum = 0.0
    @inbounds for k in 1:Nk
        xi = eps[k] - mu - phi
        E = sqrt(xi * xi + D2)
        x = 0.5 * beta * E
        # numerically-stable log(cosh x) = |x| + log((1+exp(-2|x|))/2)
        ax = abs(x)
        lncosh_sum += ax + log1p(exp(-2ax)) - log(2.0)
    end
    ferm = -(2.0 / beta) * lncosh_sum / Nk           # per-site fermionic free energy
    pair = (Delta * Delta) / (alpha * U)
    dens = beta_phi > 0.0 ? (phi * phi) / (2.0 * beta_phi * U) : 0.0
    f = pair + dens - phi + ferm
    return N * f, f
end

"""
    _self_consistent(eps, U, alpha, beta_phi, beta, mu; Delta0, phi0, tol,
                     max_iter, mixing) -> NamedTuple

Inner loop: iterate {Delta, phi} to self-consistency at FIXED mu, with linear
mixing  x <- (1-mixing)*x_old + mixing*x_new. beta_phi = 0 pins phi = 0
(density channel off). Returns (Delta, phi, n, converged, iters, residual).
"""
function _self_consistent(eps::Vector{Float64}, U::Float64, alpha::Float64,
                          beta_phi::Float64, beta::Float64, mu::Float64;
                          Delta0::Float64, phi0::Float64, tol::Float64,
                          max_iter::Int, mixing::Float64)
    Delta = abs(Delta0)
    phi = beta_phi > 0.0 ? phi0 : 0.0
    converged = false
    iters = 0
    resid = Inf
    for it in 1:max_iter
        iters = it
        # phi update uses the density at the current fields.
        n = mf_density(eps, mu, phi, Delta, beta)
        phi_new = beta_phi > 0.0 ? beta_phi * U * n : 0.0
        Delta_new = _delta_fixed_point(eps, mu, phi, Delta, U, alpha, beta)

        dDelta = Delta_new - Delta
        dphi = phi_new - phi
        resid = max(abs(dDelta), abs(dphi))

        Delta += mixing * dDelta
        phi += mixing * dphi

        if resid < tol
            converged = true
            break
        end
    end
    n = mf_density(eps, mu, phi, Delta, beta)
    return (Delta = Delta, phi = phi, n = n, converged = converged,
            iters = iters, residual = resid)
end

"""
    solve_mf(; t, U, alpha, beta_phi, beta, Lx, Ly, mu=nothing, target_n=nothing,
             Delta0=0.5, phi0=0.0, tol=1e-10, max_iter=5000, mixing=0.5,
             Q=(0.0,0.0), grid_factor=1, nkx=nothing, nky=nothing,
             mu_bracket=(-12.0, 12.0), mu_tol=1e-8, mu_max_iter=200,
             verbose=false) -> NamedTuple

Homogeneous k-space mean-field solver for the coupled Delta + phi channels at
Q = (0,0). Supply EITHER `mu` (fixed chemical potential) OR `target_n` (solve
for the mu that yields that density); supplying both, or neither, is an error.

FIXED-mu path: self-consistently iterate {Delta, phi} together at that mu.

TARGET_n path: fix the density, which BREAKS the Hartree feedback loop. With n
pinned, the density saddle gives a constant phi = beta_phi*U*target_n, so the
remaining problem is an ordinary BCS solve for (Delta, mu_eff) at fixed filling
with mu_eff = mu + phi. n(mu_eff) (Delta solved self-consistently, phi held
fixed) is smooth and monotonic -- unlike n(mu) at fixed mu, which for strong
density coupling (e.g. scheme b) is discontinuous and skips a whole range of
densities. We bisect mu_eff to hit target_n, then report mu = mu_eff - phi.
This is robust for every scheme, including the over-complete beta_phi = 1.

Returns a NamedTuple with at least: Delta, phi, n, mu, F (and F_per_site,
converged, iters, gap_residual, and the input alpha/beta_phi/beta/Q/grid).

Only Q = (0,0) is supported; any other Q errors (staggered phi is a separate
follow-up needing a folded-BZ 4x4 BdG).
"""
function solve_mf(; t::Real, U::Real, alpha::Real, beta_phi::Real, beta::Real,
                  Lx::Int, Ly::Int,
                  mu::Union{Nothing,Real} = nothing,
                  target_n::Union{Nothing,Real} = nothing,
                  Delta0::Real = 0.5, phi0::Real = 0.0,
                  tol::Real = 1e-10, max_iter::Int = 10000, mixing::Real = 0.5,
                  Q = (0.0, 0.0),
                  grid_factor::Int = 1,
                  nkx::Union{Nothing,Int} = nothing,
                  nky::Union{Nothing,Int} = nothing,
                  mu_bracket = (-12.0, 12.0), mu_tol::Real = 1e-8,
                  n_tol::Real = 1e-8, mu_max_iter::Int = 200,
                  verbose::Bool = false)

    # --- validate Q: only the uniform (Hartree) channel for now --------------
    if !(abs(Q[1]) < 1e-12 && abs(Q[2]) < 1e-12)
        error("solve_mf: only Q = (0,0) is supported (got Q = $(Q)). " *
              "Staggered phi (Q=(pi,pi)) is a separate folded-BZ follow-up.")
    end
    if (mu === nothing) == (target_n === nothing)
        error("solve_mf: supply exactly one of `mu` or `target_n`.")
    end

    t = Float64(t); U = Float64(U); alpha = Float64(alpha)
    beta_phi = Float64(beta_phi); beta = Float64(beta)
    Delta0 = Float64(Delta0); phi0 = Float64(phi0)
    tol = Float64(tol); mixing = Float64(mixing)

    if alpha <= 0.0
        error("solve_mf: alpha must be > 0 (pairing penalty is |Delta|^2/(alpha U)).")
    end

    Nkx = nkx === nothing ? grid_factor * Lx : nkx
    Nky = nky === nothing ? grid_factor * Ly : nky
    eps = _epsilon_grid(t, Nkx, Nky)
    N = Lx * Ly                      # physical site count for the extensive F

    if mu !== nothing
        # ---- fixed mu -------------------------------------------------------
        mu = Float64(mu)
        sc = _self_consistent(eps, U, alpha, beta_phi, beta, mu;
                              Delta0 = Delta0, phi0 = phi0, tol = tol,
                              max_iter = max_iter, mixing = mixing)
        Ftot, fps = mf_free_energy(eps, mu, sc.phi, sc.Delta, U, alpha,
                                   beta_phi, beta; N = N)
        gres = mf_gap_residual(eps, mu, sc.phi, sc.Delta, U, alpha, beta)
        verbose && println("solve_mf(mu=$(mu)): Delta=$(sc.Delta), phi=$(sc.phi), " *
                           "n=$(sc.n), converged=$(sc.converged)")
        return (Delta = sc.Delta, phi = sc.phi, n = sc.n, mu = mu,
                F = Ftot, F_per_site = fps,
                converged = sc.converged, iters = sc.iters,
                residual = sc.residual, gap_residual = gres,
                mu_converged = true, alpha = alpha, beta_phi = beta_phi,
                beta = beta, U = U, t = t, Q = (0.0, 0.0),
                Nkx = Nkx, Nky = Nky, Lx = Lx, Ly = Ly)
    end

    # ---- target_n: fix the density -> break the Hartree feedback ------------
    # Pinning n makes the density saddle a CONSTANT phi = beta_phi*U*target_n.
    # What remains is a plain BCS solve for (Delta, mu_eff) at fixed filling,
    # with mu_eff = mu + phi. We bisect mu_eff on n(mu_eff) -- which is smooth
    # and monotonic because phi no longer moves with n -- then set mu = mu_eff -
    # phi. (Bisecting the raw n(mu) instead fails for strong density coupling:
    # there n(mu) is discontinuous and skips whole density ranges; see scheme b.)
    target_n = Float64(target_n)
    if !(0.0 < target_n < 2.0)
        error("solve_mf: target_n must be in (0, 2) (got $(target_n)).")
    end

    phi_fixed = beta_phi > 0.0 ? beta_phi * U * target_n : 0.0
    seedD = abs(Delta0)

    # BCS density at a fixed effective chemical potential mu_eff: iterate Delta
    # only (phi held fixed -> use beta_phi=0 so _self_consistent pins phi=0 and
    # xi_k = eps_k - mu_eff). Returns (n, Delta, converged).
    function bcs_at_shift(mu_eff::Float64)
        s = _self_consistent(eps, U, alpha, 0.0, beta, mu_eff;
                             Delta0 = seedD, phi0 = 0.0, tol = tol,
                             max_iter = max_iter, mixing = mixing)
        return s.n, s.Delta, s.converged
    end

    lo, hi = Float64(mu_bracket[1]), Float64(mu_bracket[2])
    n_lo, _, _ = bcs_at_shift(lo)
    n_hi, _, _ = bcs_at_shift(hi)
    # expand the bracket outward until it straddles target_n (n increases with mu_eff)
    expand = 0
    while (n_lo - target_n) * (n_hi - target_n) > 0.0 && expand < 60
        span = hi - lo
        if n_hi < target_n
            hi += span
            n_hi, _, _ = bcs_at_shift(hi)
        elseif n_lo > target_n
            lo -= span
            n_lo, _, _ = bcs_at_shift(lo)
        else
            break
        end
        expand += 1
    end
    if (n_lo - target_n) * (n_hi - target_n) > 0.0
        error("solve_mf: could not bracket target_n=$(target_n) in mu_eff " *
              "(n(lo=$(lo))=$(n_lo), n(hi=$(hi))=$(n_hi)). Widen mu_bracket.")
    end

    mueff_sol = 0.5 * (lo + hi)
    n_mid = 0.5 * (n_lo + n_hi)
    Delta_sol = seedD
    inner_conv = true
    for _ in 1:mu_max_iter
        mueff_sol = 0.5 * (lo + hi)
        n_mid, Delta_sol, inner_conv = bcs_at_shift(mueff_sol)
        if abs(n_mid - target_n) < n_tol || 0.5 * (hi - lo) < mu_tol
            break
        end
        if (n_mid - target_n) > 0.0
            hi = mueff_sol            # n too high -> lower mu_eff
        else
            lo = mueff_sol
        end
    end

    # Density hit? Report honestly instead of trusting a collapsed mu bracket.
    n_hit = abs(n_mid - target_n) < n_tol
    mu_sol = mueff_sol - phi_fixed        # mu_eff = mu + phi  =>  mu = mu_eff - phi

    # Evaluate F and the gap residual at the physical (mu, phi). These use
    # xi_k = eps_k - mu - phi = eps_k - mu_eff, so they are consistent with the
    # BCS solve above.
    Ftot, fps = mf_free_energy(eps, mu_sol, phi_fixed, Delta_sol, U, alpha,
                               beta_phi, beta; N = N)
    gres = mf_gap_residual(eps, mu_sol, phi_fixed, Delta_sol, U, alpha, beta)
    verbose && println("solve_mf(target_n=$(target_n)): mu=$(mu_sol), " *
                       "mu_eff=$(mueff_sol), Delta=$(Delta_sol), phi=$(phi_fixed), " *
                       "n=$(n_mid), n_hit=$(n_hit)")
    return (Delta = Delta_sol, phi = phi_fixed, n = n_mid, mu = mu_sol,
            F = Ftot, F_per_site = fps,
            converged = inner_conv, iters = mu_max_iter,
            residual = abs(n_mid - target_n), gap_residual = gres,
            mu_converged = n_hit, alpha = alpha, beta_phi = beta_phi,
            beta = beta, U = U, t = t, Q = (0.0, 0.0),
            Nkx = Nkx, Nky = Nky, Lx = Lx, Ly = Ly)
end

end # module MFKSpaceDensityChannel
