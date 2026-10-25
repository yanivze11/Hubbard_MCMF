"""
    magnetic_field_MF.jl

Self-consistent Mean-Field Hubbard model on a 2D square lattice in a magnetic field,
using an "almost symmetric gauge" for the vector potential.

Exploits analytical BdG structure: diagonalizes only the n×n kinetic matrix H₀(q)
and computes quasiparticle energies analytically, avoiding the full 2n×2n BdG matrix.
"""

using LinearAlgebra
using Plots

# ============================================================================
# Phase 1: Lattice Geometry and Momentum Grid Setup
# ============================================================================

"""
    setup_lattice(L::Int, Q::Int)

Set up lattice geometry and Reduced Brillouin Zone for a rectangular lattice
Lx = L, Ly = L+1 with Q magnetic flux quanta.

Returns a NamedTuple with all lattice parameters and the RBZ momentum grid.
"""
function setup_lattice(L::Int, Q::Int)
    Lx = L
    Ly = L + 1
    Nsites = Lx * Ly

    # Reduced fractions Q/Lx → Px/Qx and Q/Ly → Py/Qy
    gx = gcd(Q, Lx)
    Px = Q ÷ gx
    Qx = Lx ÷ gx

    gy = gcd(Q, Ly)
    Py = Q ÷ gy
    Qy = Ly ÷ gy

    # Magnetic basis size
    n = Qx * Qy

    # RBZ momentum grids
    n_qx = Lx ÷ Qx   # number of qx points in RBZ
    n_qy = Ly ÷ Qy   # number of qy points in RBZ

    qx_grid = [(2π / Lx) * ix for ix in 0:(n_qx - 1)]
    qy_grid = [(2π / Ly) * iy for iy in 0:(n_qy - 1)]

    return (Lx=Lx, Ly=Ly, Nsites=Nsites,
            Px=Px, Qx=Qx, Py=Py, Qy=Qy, n=n,
            n_qx=n_qx, n_qy=n_qy,
            qx_grid=qx_grid, qy_grid=qy_grid)
end

# ============================================================================
# Phase 2: Sparse Construction of the Kinetic Matrix H₀(q)
# ============================================================================

"""
    basis_index(mx::Int, my::Int, Qy::Int) -> Int

Map 2D basis index (mx, my) to 1D index (1-based).
mx ∈ [0, Qx-1], my ∈ [0, Qy-1].
"""
@inline function basis_index(mx::Int, my::Int, Qy::Int)
    return mx * Qy + my + 1
end

"""
    build_H0q!(H::Matrix{ComplexF64}, qx::Float64, qy::Float64, lattice; t::Float64=1.0)

Construct the n×n Hermitian kinetic matrix H₀(q) for a given momentum (qx, qy).
Populates only the non-zero nearest-neighbor hopping elements in-place.

Uses the "almost symmetric gauge" Peierls phases.
"""
function build_H0q!(H::Matrix{ComplexF64}, qx::Float64, qy::Float64, lattice; t::Float64=1.0)
    (; Px, Qx, Py, Qy, n) = lattice

    # Zero out matrix
    fill!(H, zero(ComplexF64))

    for mx in 0:(Qx - 1)
        for my in 0:(Qy - 1)
            m = basis_index(mx, my, Qy)

            # --- X-hopping: connects (mx, my) → (mx, (my-1) mod Qy) ---
            my_prime = mod(my - 1, Qy)
            m_prime = basis_index(mx, my_prime, Qy)
            phase_x = -t * cis(qx + 2π * Px * mx / Qx)
            H[m, m_prime] += phase_x
            H[m_prime, m] += conj(phase_x)

            # --- Y-hopping: connects (mx, my) → ((mx-1) mod Qx, my) ---
            mx_prime = mod(mx - 1, Qx)
            m_prime = basis_index(mx_prime, my, Qy)
            phase_y = -t * cis(qy + 2π * Py * my / Qy)
            H[m, m_prime] += phase_y
            H[m_prime, m] += conj(phase_y)
        end
    end

    return H
end

"""
    build_H0q(qx::Float64, qy::Float64, lattice; t::Float64=1.0) -> Matrix{ComplexF64}

Allocating version of build_H0q!.
"""
function build_H0q(qx::Float64, qy::Float64, lattice; t::Float64=1.0)
    H = zeros(ComplexF64, lattice.n, lattice.n)
    build_H0q!(H, qx, qy, lattice; t=t)
    return H
end

# ============================================================================
# Phase 3: Diagonalization and Analytical Energies
# ============================================================================

"""
    compute_all_eigenvalues(lattice; t::Float64=1.0)

Diagonalize H₀(q) for all q in the RBZ. Returns a 3D array of eigenvalues
ε[iq_x, iq_y, λ] where λ ∈ [1, n].

Since H₀(q) does not depend on Δ, this only needs to be called ONCE.
"""
function compute_all_eigenvalues(lattice; t::Float64=1.0)
    (; n, n_qx, n_qy, qx_grid, qy_grid) = lattice

    # Pre-allocate eigenvalue storage
    epsilon = Array{Float64, 3}(undef, n_qx, n_qy, n)

    # Pre-allocate work matrix (reused for each q)
    H_work = zeros(ComplexF64, n, n)

    for (ix, qx) in enumerate(qx_grid)
        for (iy, qy) in enumerate(qy_grid)
            build_H0q!(H_work, qx, qy, lattice; t=t)
            # Hermitian wrapper enables optimized eigensolver (LAPACK dsyev/zheev)
            evals = eigvals(Hermitian(H_work))
            epsilon[ix, iy, :] .= evals
        end
    end

    return epsilon
end

"""
    compute_quasiparticle_energies!(E_q_lambda, epsilon, μ, Δ)

Compute quasiparticle energies analytically:
    E_{qλ} = √((ε_{qλ} - μ)² + |Δ|²)

Updates E_q_lambda in-place.
"""
function compute_quasiparticle_energies!(E_q_lambda::Array{Float64,3},
                                          epsilon::Array{Float64,3},
                                          μ::Float64, Δ::Float64)
    Δ2 = Δ^2
    @inbounds for i in eachindex(epsilon)
        ξ = epsilon[i] - μ
        E_q_lambda[i] = sqrt(ξ^2 + Δ2)
    end
    return E_q_lambda
end

# ============================================================================
# Occupation number
# ============================================================================

"""
    compute_occupation(epsilon, E_q_lambda, μ, β, Nsites) -> Float64

Compute the average occupation per site:
    n = 1 - (1/N) Σ_{q,λ} (ξ_{qλ} / E_{qλ}) tanh(β E_{qλ} / 2)

where ξ_{qλ} = ε_{qλ} - μ.
Should be called after the self-consistency loop has converged.
"""
function compute_occupation(epsilon::Array{Float64,3}, E_q_lambda::Array{Float64,3},
                            μ::Float64, β::Float64, Nsites::Int)
    occ_sum = 0.0
    @inbounds for i in eachindex(epsilon)
        ξ = epsilon[i] - μ
        E = E_q_lambda[i]
        occ_sum += (ξ / E) * tanh(β * E / 2)
    end
    return 1.0 - occ_sum / Nsites
end

# ============================================================================
# Phase 4: The Self-Consistency Loop
# ============================================================================

"""
    self_consistent_delta(lattice, epsilon;
                          U::Float64=4.0, μ::Float64=0.0, T::Float64=0.01,
                          tol::Float64=1e-5, max_iter::Int=10000, α::Float64=0.5,
                          Δ_init::Float64=0.1, verbose::Bool=true)

Run the self-consistency loop for the pairing gap Δ.

The gap equation is:
    Δ_new = (U Δ_old) / (2 N_sites) × Σ_{q,λ} tanh(β E_{qλ}/2) / E_{qλ}

Linear mixing:  Δ_old ← α Δ_new + (1-α) Δ_old

Returns (Δ_converged, E_q_lambda, converged::Bool, iterations::Int).
"""
function self_consistent_delta(lattice, epsilon::Array{Float64,3};
                               U::Float64=4.0, μ::Float64=0.0, T::Float64=0.01,
                               tol::Float64=1e-5, max_iter::Int=10000,
                               α::Float64=0.5, Δ_init::Float64=0.1,
                               verbose::Bool=true)
    β = 1.0 / T
    Nsites = lattice.Nsites

    # Pre-allocate quasiparticle energy array (same shape as epsilon)
    E_q_lambda = similar(epsilon)

    Δ_old = Δ_init
    converged = false
    iter = 0

    for it in 1:max_iter
        iter = it

        # Compute E_{qλ} from current Δ_old
        compute_quasiparticle_energies!(E_q_lambda, epsilon, μ, Δ_old)

        # Accumulate gap sum: Σ tanh(β E / 2) / E
        gap_sum = 0.0
        @inbounds for i in eachindex(E_q_lambda)
            E = E_q_lambda[i]
            gap_sum += tanh(β * E / 2) / E
        end

        # Self-consistency equation
        Δ_new = U * Δ_old / (2 * Nsites) * gap_sum

        # Check convergence
        residual = abs(Δ_new - Δ_old)
        if residual < tol
            converged = true
            # Final update of E_q_lambda with the converged Δ_new
            Δ_old = α * Δ_new + (1 - α) * Δ_old
            compute_quasiparticle_energies!(E_q_lambda, epsilon, μ, Δ_old)
            if verbose
                println("Converged in $it iterations. Δ = $(round(Δ_old; digits=8)), residual = $(round(residual; sigdigits=3))")
            end
            return (Δ=Δ_old, E_q_lambda=E_q_lambda, converged=converged, iterations=iter)
        end

        # Linear mixing
        Δ_old = α * Δ_new + (1 - α) * Δ_old
    end

    # Did not converge — return final state anyway
    compute_quasiparticle_energies!(E_q_lambda, epsilon, μ, Δ_old)
    if verbose
        println("WARNING: Did not converge in $max_iter iterations. Final Δ = $(round(Δ_old; digits=8))")
    end
    return (Δ=Δ_old, E_q_lambda=E_q_lambda, converged=converged, iterations=iter)
end

# ============================================================================
# Main entry point
# ============================================================================

"""
    run_magnetic_field_MF(L, Q; t=1.0, U=4.0, μ=0.0, T=0.01,
                          tol=1e-5, max_iter=10000, α=0.5, Δ_init=0.1, verbose=true)

Full pipeline: sets up lattice, builds & diagonalizes all H₀(q), then runs
the self-consistency loop. Returns a NamedTuple containing:
  - Δ:          converged pairing gap
  - E_q_lambda: quasiparticle energy spectrum array (n_qx × n_qy × n)
  - epsilon:    kinetic eigenvalues (n_qx × n_qy × n)
  - lattice:    lattice geometry parameters
  - converged:  whether the loop converged
  - iterations: number of iterations used
"""
function run_magnetic_field_MF(L::Int, Q::Int;
                               t::Float64=1.0, U::Float64=4.0, μ::Float64=0.0,
                               T::Float64=0.01, tol::Float64=1e-5,
                               max_iter::Int=10000, α::Float64=0.5,
                               Δ_init::Float64=0.1, verbose::Bool=true)
    if verbose
        println("="^60)
        println("Magnetic Field Mean-Field Hubbard Model")
        println("="^60)
    end

    # --- Phase 1: Lattice & Momentum Grid ---
    lattice = setup_lattice(L, Q)
    if verbose
        println("Lattice: Lx=$(lattice.Lx), Ly=$(lattice.Ly), Nsites=$(lattice.Nsites)")
        println("Gauge fractions: Px/Qx = $(lattice.Px)/$(lattice.Qx), Py/Qy = $(lattice.Py)/$(lattice.Qy)")
        println("Magnetic basis size: n = $(lattice.n)")
        println("RBZ grid: $(lattice.n_qx) × $(lattice.n_qy) = $(lattice.n_qx * lattice.n_qy) q-points")
        println("-"^60)
    end

    # --- Phase 2 & 3: Build H₀(q) + Diagonalize (done once) ---
    if verbose
        print("Diagonalizing H₀(q) for all q-points... ")
        t_start = time()
    end
    epsilon = compute_all_eigenvalues(lattice; t=t)
    if verbose
        elapsed = round(time() - t_start; digits=3)
        println("done ($(elapsed)s)")
        println("-"^60)
    end

    # --- Phase 4: Self-consistency loop ---
    result = self_consistent_delta(lattice, epsilon;
                                   U=U, μ=μ, T=T, tol=tol,
                                   max_iter=max_iter, α=α,
                                   Δ_init=Δ_init, verbose=verbose)

    # --- Compute occupation at converged solution ---
    β = 1.0 / T
    n_occ = compute_occupation(epsilon, result.E_q_lambda, μ, β, lattice.Nsites)
    if verbose
        println("Occupation n = $(round(n_occ; digits=6))")
        println("-"^60)
    end

    return (Δ=result.Δ, E_q_lambda=result.E_q_lambda, epsilon=epsilon,
            lattice=lattice, converged=result.converged, iterations=result.iterations,
            n=n_occ, μ=μ)
end


# ============================================================================
# Chemical potential tuning for target occupation
# ============================================================================

"""
    tune_mu_for_target_n(L, Q, target_n;
                         t=1.0, U=4.0, T=0.01,
                         mu_low=-10.0, mu_high=10.0,
                         tol_n=1e-4, tol_delta=1e-5,
                         max_bisection=100, max_iter=10000,
                         α=0.5, Δ_init=0.1, verbose=true)

Bisection search over μ to find the chemical potential that yields
occupation n = target_n. At each μ, runs the full self-consistency loop
for Δ, then computes n.

Returns a NamedTuple with all fields from `run_magnetic_field_MF` plus
`target_n` and the number of bisection steps.
"""
function tune_mu_for_target_n(L::Int, Q::Int, target_n::Float64;
                              t::Float64=1.0, U::Float64=4.0, T::Float64=0.01,
                              mu_low::Float64=-4.0, mu_high::Float64=0.0,
                              tol_n::Float64=1e-4, tol_delta::Float64=1e-5,
                              max_bisection::Int=100, max_iter::Int=10000,
                              α::Float64=0.5, Δ_init::Float64=0.1,
                              verbose::Bool=true)
    if verbose
        println("="^60)
        println("Tuning μ for target n = $target_n")
        println("Bisection range: [$mu_low, $mu_high]")
        println("="^60)
    end

    # --- Setup lattice and diagonalize ONCE (ε_{qλ} don't depend on μ) ---
    lattice = setup_lattice(L, Q)
    epsilon = compute_all_eigenvalues(lattice; t=t)
    β = 1.0 / T

    if verbose
        println("Lattice: Lx=$(lattice.Lx), Ly=$(lattice.Ly), Nsites=$(lattice.Nsites)")
        println("Magnetic basis size: n = $(lattice.n)")
        println("RBZ grid: $(lattice.n_qx) × $(lattice.n_qy) = $(lattice.n_qx * lattice.n_qy) q-points")
        println("-"^60)
    end

    # Helper: run self-consistency at a given μ and return occupation
    function n_at_mu(μ_test::Float64)
        res = self_consistent_delta(lattice, epsilon;
                                    U=U, μ=μ_test, T=T, tol=tol_delta,
                                    max_iter=max_iter, α=α, Δ_init=Δ_init,
                                    verbose=false)
        return compute_occupation(epsilon, res.E_q_lambda, μ_test, β, lattice.Nsites), res
    end

    # Validate bracket: n should be monotonically increasing with μ
    n_low, _ = n_at_mu(mu_low)
    n_high, _ = n_at_mu(mu_high)
    if verbose
        println("n(μ_low=$mu_low) = $(round(n_low; digits=6))")
        println("n(μ_high=$mu_high) = $(round(n_high; digits=6))")
    end

    if !((n_low ≤ target_n ≤ n_high) || (n_high ≤ target_n ≤ n_low))
        @warn "Target n=$target_n is outside the bracket [n($mu_low)=$n_low, n($mu_high)=$n_high]. " *
              "Consider widening mu_low/mu_high."
    end

    # --- Bisection loop ---
    local μ_mid::Float64
    local n_mid::Float64
    local last_result
    bisection_step = 0

    for step in 1:max_bisection
        bisection_step = step
        μ_mid = (mu_low + mu_high) / 2
        n_mid, last_result = n_at_mu(μ_mid)

        if verbose && (step ≤ 5 || step % 10 == 0)
            println("  step $step: μ = $(round(μ_mid; digits=6)), n = $(round(n_mid; digits=6)), " *
                    "Δ = $(round(last_result.Δ; digits=6)), |n - target| = $(round(abs(n_mid - target_n); sigdigits=3))")
        end

        if abs(n_mid - target_n) < tol_n
            if verbose
                println("-"^60)
                println("Converged in $step bisection steps.")
                println("  μ = $(round(μ_mid; digits=8))")
                println("  n = $(round(n_mid; digits=8)) (target: $target_n)")
                println("  Δ = $(round(last_result.Δ; digits=8))")
            end
            return (Δ=last_result.Δ, E_q_lambda=last_result.E_q_lambda, epsilon=epsilon,
                    lattice=lattice, converged=last_result.converged,
                    iterations=last_result.iterations, n=n_mid, μ=μ_mid,
                    target_n=target_n, bisection_steps=step)
        end

        # n increases with μ, so:
        if n_mid < target_n
            mu_low = μ_mid
        else
            mu_high = μ_mid
        end
    end

    @warn "Bisection did not converge in $max_bisection steps. " *
          "Final μ=$μ_mid, n=$n_mid (target=$target_n)"
    return (Δ=last_result.Δ, E_q_lambda=last_result.E_q_lambda, epsilon=epsilon,
            lattice=lattice, converged=last_result.converged,
            iterations=last_result.iterations, n=n_mid, μ=μ_mid,
            target_n=target_n, bisection_steps=max_bisection)
end


# ============================================================================
# Temperature sweep: Δ(T) at fixed Q and target filling
# ============================================================================

"""
    sweep_delta_vs_T(L, Q, T_list;
                     target_n=0.875, t=1.0, U=4.0,
                     mu_low=-4.0, mu_high=0.0,
                     tol_n=1e-4, tol_delta=1e-5,
                     max_bisection=100, max_iter=10000,
                     α=0.5, Δ_init=0.1, verbose=true)

Sweep over temperatures and compute the self-consistent Δ at each T,
re-tuning μ at every temperature to maintain occupation n = target_n.

The eigenvalues ε_{qλ} are diagonalized ONCE (they don't depend on T or μ).

Returns a NamedTuple with:
  - T_list:  the input temperature array
  - Δ_list:  converged Δ at each T
  - μ_list:  tuned μ at each T
  - n_list:  actual occupation at each T
"""
function sweep_delta_vs_T(L::Int, Q::Int, T_list::AbstractVector{Float64};
                          target_n::Float64=0.875,
                          t::Float64=1.0, U::Float64=4.0,
                          mu_low::Float64=-4.0, mu_high::Float64=0.0,
                          tol_n::Float64=1e-4, tol_delta::Float64=1e-5,
                          max_bisection::Int=100, max_iter::Int=10000,
                          α::Float64=0.5, Δ_init::Float64=0.1,
                          verbose::Bool=true)
    nT = length(T_list)
    Δ_list = zeros(Float64, nT)
    μ_list = zeros(Float64, nT)
    n_list = zeros(Float64, nT)

    # --- Setup lattice and diagonalize ONCE ---
    lattice = setup_lattice(L, Q)
    epsilon = compute_all_eigenvalues(lattice; t=t)

    if verbose
        println("="^60)
        println("Δ(T) sweep: L=$L, Q=$Q, target_n=$target_n, U=$U")
        println("Lattice: $(lattice.Lx)×$(lattice.Ly), basis n=$(lattice.n), " *
                "RBZ $(lattice.n_qx)×$(lattice.n_qy)")
        println("Sweeping $nT temperatures: T ∈ [$(minimum(T_list)), $(maximum(T_list))]")
        println("="^60)
    end

    # Track μ from previous T as a warm-start for bisection bracket
    μ_prev = (mu_low + mu_high) / 2

    for (iT, T) in enumerate(T_list)
        β = 1.0 / T

        # Narrow bisection bracket around previous μ (with safety margin)
        ml = max(mu_low, μ_prev - 2.0)
        mh = min(mu_high, μ_prev + 2.0)

        # Inner bisection for μ at this T
        function n_at_mu_inner(μ_test::Float64)
            res = self_consistent_delta(lattice, epsilon;
                                        U=U, μ=μ_test, T=T, tol=tol_delta,
                                        max_iter=max_iter, α=α, Δ_init=Δ_init,
                                        verbose=false)
            n_val = compute_occupation(epsilon, res.E_q_lambda, μ_test, β, lattice.Nsites)
            return n_val, res
        end

        # Validate/widen bracket
        n_lo, _ = n_at_mu_inner(ml)
        n_hi, _ = n_at_mu_inner(mh)
        if !((n_lo ≤ target_n ≤ n_hi) || (n_hi ≤ target_n ≤ n_lo))
            ml, mh = mu_low, mu_high
        end

        local μ_mid, n_mid, last_res
        found = false
        for step in 1:max_bisection
            μ_mid = (ml + mh) / 2
            n_mid, last_res = n_at_mu_inner(μ_mid)
            if abs(n_mid - target_n) < tol_n
                found = true
                break
            end
            if n_mid < target_n
                ml = μ_mid
            else
                mh = μ_mid
            end
        end

        Δ_list[iT] = last_res.Δ
        μ_list[iT] = μ_mid
        n_list[iT] = n_mid
        μ_prev = μ_mid
        # Use previous Δ as warm start for next T
        Δ_init = max(last_res.Δ, 1e-6)

        if verbose
            status = found ? "✓" : "⚠"
            println("  $status T=$(round(T; digits=4)), Δ=$(round(last_res.Δ; digits=6)), " *
                    "μ=$(round(μ_mid; digits=6)), n=$(round(n_mid; digits=6))")
        end
    end

    if verbose
        println("="^60)
        println("Sweep complete.")
    end

    return (T_list=collect(T_list), Δ_list=Δ_list, μ_list=μ_list, n_list=n_list,
            L=L, Q=Q, target_n=target_n, U=U, lattice=lattice, epsilon=epsilon)
end


# ============================================================================
# Hofstadter butterfly: spectrum E vs Q (magnetic flux)
# ============================================================================

"""
    hofstadter_butterfly(L, Q_list;
                         mode=:constant_mu, μ=0.0, target_n=0.875,
                         t=1.0, U=4.0, T=0.01,
                         mu_low=-4.0, mu_high=0.0,
                         tol_n=1e-4, tol_delta=1e-5,
                         max_bisection=60, max_iter=10000,
                         α=0.5, Δ_init=0.1, verbose=true)

Compute the quasiparticle energy spectrum as a function of the magnetic flux
quantum number Q, producing data for a Hofstadter butterfly plot.

**mode** controls how the chemical potential is handled:
  - `:constant_mu`  — use the fixed chemical potential `μ` for all Q.
  - `:constant_n`   — tune μ via bisection at each Q to maintain `n = target_n`.

For each Q, the code:
1. Sets up the lattice and diagonalizes H₀(q) for all q in the RBZ.
2. Runs the Δ self-consistency loop (at the given or tuned μ).
3. Collects all quasiparticle energies E_{qλ} (and optionally ±E for
   the full BdG spectrum).

Returns a NamedTuple with:
  - Q_list:     input flux quanta values
  - phi_list:   Q / (Lx*Ly)  — flux per plaquette in units of Φ₀
  - all_E:      Vector of Vector{Float64}, the spectrum at each Q
  - Δ_list:     converged Δ at each Q
  - μ_list:     chemical potential used at each Q
  - n_list:     occupation at each Q
  - L, U, T:    parameters used
"""
function hofstadter_butterfly(L::Int, Q_list::AbstractVector{Int};
                              mode::Symbol=:constant_mu,
                              μ::Float64=0.0, target_n::Float64=0.875,
                              t::Float64=1.0, U::Float64=4.0, T::Float64=0.01,
                              mu_low::Float64=-4.0, mu_high::Float64=0.0,
                              tol_n::Float64=1e-4, tol_delta::Float64=1e-5,
                              max_bisection::Int=60, max_iter::Int=10000,
                              α::Float64=0.5, Δ_init::Float64=0.1,
                              verbose::Bool=true)
    @assert mode in (:constant_mu, :constant_n) "mode must be :constant_mu or :constant_n"

    Lx = L
    Ly = L + 1
    Nsites = Lx * Ly
    β = 1.0 / T

    nQ = length(Q_list)
    all_E  = Vector{Vector{Float64}}(undef, nQ)
    Δ_list = zeros(Float64, nQ)
    μ_list = zeros(Float64, nQ)
    n_list = zeros(Float64, nQ)
    phi_list = [Q / Nsites for Q in Q_list]

    if verbose
        println("="^60)
        println("Hofstadter Butterfly: L=$L ($Lx×$Ly), U=$U, T=$T")
        println("Mode: $mode" * (mode == :constant_mu ? ", μ=$μ" : ", target_n=$target_n"))
        println("Sweeping $(nQ) values of Q ∈ [$(minimum(Q_list)), $(maximum(Q_list))]")
        println("="^60)
    end

    # Warm-start μ for constant_n mode
    μ_prev = mode == :constant_mu ? μ : (mu_low + mu_high) / 2

    for (iQ, Q) in enumerate(Q_list)
        # Phase 1 & 2-3: lattice + diagonalize for this Q
        lattice = setup_lattice(L, Q)
        epsilon = compute_all_eigenvalues(lattice; t=t)

        local μ_Q::Float64
        local n_Q::Float64
        local res

        if mode == :constant_mu
            # Fixed μ: just run self-consistency
            μ_Q = μ
            res = self_consistent_delta(lattice, epsilon;
                                        U=U, μ=μ_Q, T=T, tol=tol_delta,
                                        max_iter=max_iter, α=α, Δ_init=Δ_init,
                                        verbose=false)
            n_Q = compute_occupation(epsilon, res.E_q_lambda, μ_Q, β, Nsites)

        else  # :constant_n — bisection over μ
            ml = max(mu_low, μ_prev - 2.0)
            mh = min(mu_high, μ_prev + 2.0)

            function _n_at_mu(μ_test)
                r = self_consistent_delta(lattice, epsilon;
                                          U=U, μ=μ_test, T=T, tol=tol_delta,
                                          max_iter=max_iter, α=α, Δ_init=Δ_init,
                                          verbose=false)
                return compute_occupation(epsilon, r.E_q_lambda, μ_test, β, Nsites), r
            end

            # Validate/widen bracket
            n_lo, _ = _n_at_mu(ml)
            n_hi, _ = _n_at_mu(mh)
            if !((n_lo ≤ target_n ≤ n_hi) || (n_hi ≤ target_n ≤ n_lo))
                ml, mh = mu_low, mu_high
            end

            local μ_mid, n_mid, res
            for step in 1:max_bisection
                μ_mid = (ml + mh) / 2
                n_mid, res = _n_at_mu(μ_mid)
                if abs(n_mid - target_n) < tol_n
                    break
                end
                if n_mid < target_n
                    ml = μ_mid
                else
                    mh = μ_mid
                end
            end
            μ_Q = μ_mid
            n_Q = n_mid
            μ_prev = μ_Q
        end

        # Collect the full BdG spectrum: ±E_{qλ}
        E_flat = vec(res.E_q_lambda)
        # if you want only the positive branch, use E_flat directly
        all_E[iQ] = vcat(E_flat, -E_flat)
        # all_E[iQ] = E_flat
        Δ_list[iQ] = res.Δ
        μ_list[iQ] = μ_Q
        n_list[iQ] = n_Q

        # Warm-start Δ for next Q
        Δ_init = max(res.Δ, 1e-6)

        if verbose && (iQ ≤ 5 || iQ % max(1, nQ ÷ 20) == 0 || iQ == nQ)
            println("  Q=$Q (ϕ=$(round(phi_list[iQ]; digits=4))), " *
                    "Δ=$(round(res.Δ; digits=6)), μ=$(round(μ_Q; digits=4)), " *
                    "n=$(round(n_Q; digits=4)), bands=$(length(E_flat))")
        end
    end

    if verbose
        println("="^60)
        println("Butterfly sweep complete.")
    end

    return (Q_list=collect(Q_list), phi_list=phi_list, all_E=all_E,
            Δ_list=Δ_list, μ_list=μ_list, n_list=n_list,
            L=L, U=U, T=T, mode=mode,
            target_n=(mode == :constant_n ? target_n : nothing))
end

"""
    plot_hofstadter_butterfly(result; kwargs...)

Plot the Hofstadter butterfly from the output of `hofstadter_butterfly`.
Plots E vs ϕ/Φ₀ = Q/(Lx·Ly) as a scatter plot.
"""
function plot_hofstadter_butterfly(bf; markersize=0.5, color=:black, kwargs...)
    xs = Float64[]
    ys = Float64[]
    for (i, ϕ) in enumerate(bf.phi_list)
        for E in bf.all_E[i]
            push!(xs, ϕ)
            push!(ys, E)
        end
    end

    mode_str = bf.mode == :constant_mu ? "μ=$(round(bf.μ_list[1]; digits=2))" :
                                          "n=$(bf.target_n)"
    p = scatter(xs, ys;
                markersize=markersize, color=color, markerstrokewidth=0,
                xlabel="ϕ / Φ₀ = Q / N", ylabel="E",
                title="Hofstadter Butterfly (L=$(bf.L), U=$(bf.U), T=$(bf.T), $mode_str)",
                legend=false, kwargs...)
    return p
end


# ============================================================================
# Excitation Density of States (DOS)
# ============================================================================

"""
    lorentzian(ω, ω0, η) -> Float64

Lorentzian broadening of a delta function:
    δ_η(ω - ω₀) = (1/π) η / ((ω - ω₀)² + η²)
"""
@inline function lorentzian(ω::Float64, ω0::Float64, η::Float64)
    return η / (π * ((ω - ω0)^2 + η^2))
end

"""
    compute_dos(epsilon, E_q_lambda, μ, ω_list; η=nothing)

Compute the excitation density of states:

    g(ω) = Σ_{q,λ} [ (1 + ξ_{qλ}/E_{qλ}) δ(ω - E_{qλ})
                    + (1 - ξ_{qλ}/E_{qλ}) δ(ω + E_{qλ}) ]

where ξ_{qλ} = ε_{qλ} - μ, and the delta function is replaced by a
Lorentzian of width η.

If `η` is not provided, it defaults to 0.1 × (4/N) where N is the total
number of bands (approximately one tenth of the uniform energy spacing).

Returns a Vector{Float64} of DOS values g(ω) for each ω in ω_list.
"""
function compute_dos(epsilon::Array{Float64,3}, E_q_lambda::Array{Float64,3},
                     μ::Float64, ω_list::AbstractVector{Float64};
                     η::Union{Float64,Nothing}=nothing)
    N_total = length(epsilon)  # total number of (q,λ) states

    # Default broadening: ~1/10 of uniform level spacing (bandwidth ~ 4t, so 4/N)
    if η === nothing
        η = 2 * 4.0 / N_total
    end

    nω = length(ω_list)
    dos = zeros(Float64, nω)

    # Pre-compute the coherence factors and energies
    @inbounds for i in eachindex(epsilon)
        ξ = epsilon[i] - μ
        E = E_q_lambda[i]
        ratio = ξ / E
        w_plus  = 1.0 + ratio   # weight for δ(ω - E)
        w_minus = 1.0 - ratio   # weight for δ(ω + E)

        for j in 1:nω
            ω = ω_list[j]
            dos[j] += w_plus  * lorentzian(ω,  E, η) +
                      w_minus * lorentzian(ω, -E, η)
        end
    end

    return dos
end

"""
    compute_and_plot_dos(result; ω_range=nothing, nω=500, η=nothing, kwargs...)

Compute and plot the excitation DOS from the output of `run_magnetic_field_MF`
or `tune_mu_for_target_n`.

`result` must have fields: `epsilon`, `E_q_lambda`, `μ`, `lattice`.
"""
function compute_and_plot_dos(result; ω_range::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                              nω::Int=500, η::Union{Float64,Nothing}=nothing,
                              kwargs...)
    E_max = maximum(result.E_q_lambda)
    if ω_range === nothing
        ω_range = (-E_max * 1.2, E_max * 1.2)
    end
    ω_list = collect(range(ω_range[1], ω_range[2]; length=nω))

    dos = compute_dos(result.epsilon, result.E_q_lambda, result.μ, ω_list; η=η)

    Lx = result.lattice.Lx
    Ly = result.lattice.Ly
    Δ_val = hasproperty(result, :Δ) ? round(result.Δ; digits=4) : "?"
    n_val = hasproperty(result, :n) ? round(result.n; digits=4) : "?"

    p = plot(ω_list, dos;
             xlabel="ω", ylabel="g(ω)",
             title="Excitation DOS ($(Lx)×$(Ly), Δ=$Δ_val, n=$n_val)",
             legend=false, linewidth=1.5, kwargs...)
    vline!(p, [0.0]; linestyle=:dash, color=:gray, alpha=0.5)

    return (plot=p, ω_list=ω_list, dos=dos)
end


"""
    dos_heatmap(L, Q_list;
                mode=:constant_mu, μ=0.0, target_n=0.875,
                t=1.0, U=4.0, T=0.01,
                mu_low=-4.0, mu_high=0.0,
                tol_n=1e-4, tol_delta=1e-5,
                max_bisection=60, max_iter=10000,
                α=0.5, Δ_init=0.1,
                ω_range=nothing, nω=300, η=nothing,
                verbose=true)

Compute the excitation DOS g(ω) for each Q in Q_list, and return a 2D matrix
suitable for a heatmap plot (x = Q or ϕ/Φ₀, y = ω, color = DOS intensity).

**mode** controls the chemical potential:
  - `:constant_mu`  — fixed μ for all Q
  - `:constant_n`   — bisection-tune μ at each Q to maintain `n = target_n`

Returns a NamedTuple with:
  - dos_matrix:  nω × nQ matrix of DOS values
  - ω_list:      ω grid (length nω)
  - Q_list, phi_list, Δ_list, μ_list, n_list
"""
function dos_heatmap(L::Int, Q_list::AbstractVector{Int};
                     mode::Symbol=:constant_mu,
                     μ::Float64=0.0, target_n::Float64=0.875,
                     t::Float64=1.0, U::Float64=4.0, T::Float64=0.01,
                     mu_low::Float64=-4.0, mu_high::Float64=0.0,
                     tol_n::Float64=1e-4, tol_delta::Float64=1e-5,
                     max_bisection::Int=60, max_iter::Int=10000,
                     α::Float64=0.5, Δ_init::Float64=0.1,
                     ω_range::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                     nω::Int=300, η::Union{Float64,Nothing}=nothing,
                     verbose::Bool=true)
    @assert mode in (:constant_mu, :constant_n) "mode must be :constant_mu or :constant_n"

    Lx = L
    Ly = L + 1
    Nsites = Lx * Ly
    β = 1.0 / T

    nQ = length(Q_list)
    Δ_list  = zeros(Float64, nQ)
    μ_list  = zeros(Float64, nQ)
    n_list  = zeros(Float64, nQ)
    phi_list = [Q / Nsites for Q in Q_list]

    if verbose
        println("="^60)
        println("DOS Heatmap: L=$L ($Lx×$Ly), U=$U, T=$T")
        println("Mode: $mode" * (mode == :constant_mu ? ", μ=$μ" : ", target_n=$target_n"))
        println("Sweeping $(nQ) values of Q, nω=$nω")
        println("="^60)
    end

    # ---------- First pass: compute spectra and find global E_max ----------
    # Store per-Q results for the DOS computation
    eps_list   = Vector{Array{Float64,3}}(undef, nQ)
    Eql_list   = Vector{Array{Float64,3}}(undef, nQ)
    μ_prev = mode == :constant_mu ? μ : (mu_low + mu_high) / 2
    global_E_max = 0.0

    for (iQ, Q) in enumerate(Q_list)
        lattice = setup_lattice(L, Q)
        epsilon = compute_all_eigenvalues(lattice; t=t)

        local μ_Q::Float64
        local n_Q::Float64
        local res

        if mode == :constant_mu
            μ_Q = μ
            res = self_consistent_delta(lattice, epsilon;
                                        U=U, μ=μ_Q, T=T, tol=tol_delta,
                                        max_iter=max_iter, α=α, Δ_init=Δ_init,
                                        verbose=false)
            n_Q = compute_occupation(epsilon, res.E_q_lambda, μ_Q, β, Nsites)
        else
            ml = max(mu_low, μ_prev - 2.0)
            mh = min(mu_high, μ_prev + 2.0)

            function _n_at_mu_dos(μ_test)
                r = self_consistent_delta(lattice, epsilon;
                                          U=U, μ=μ_test, T=T, tol=tol_delta,
                                          max_iter=max_iter, α=α, Δ_init=Δ_init,
                                          verbose=false)
                return compute_occupation(epsilon, r.E_q_lambda, μ_test, β, Nsites), r
            end

            n_lo, _ = _n_at_mu_dos(ml)
            n_hi, _ = _n_at_mu_dos(mh)
            if !((n_lo ≤ target_n ≤ n_hi) || (n_hi ≤ target_n ≤ n_lo))
                ml, mh = mu_low, mu_high
            end

            local μ_mid, n_mid
            for step in 1:max_bisection
                μ_mid = (ml + mh) / 2
                n_mid, res = _n_at_mu_dos(μ_mid)
                if abs(n_mid - target_n) < tol_n
                    break
                end
                n_mid < target_n ? (ml = μ_mid) : (mh = μ_mid)
            end
            μ_Q = μ_mid
            n_Q = n_mid
            μ_prev = μ_Q
        end

        eps_list[iQ]  = epsilon
        Eql_list[iQ]  = res.E_q_lambda
        Δ_list[iQ]    = res.Δ
        μ_list[iQ]    = μ_Q
        n_list[iQ]    = n_Q
        Δ_init = max(res.Δ, 1e-6)

        E_max_Q = maximum(res.E_q_lambda)
        global_E_max = max(global_E_max, E_max_Q)

        if verbose && (iQ ≤ 3 || iQ % max(1, nQ ÷ 20) == 0 || iQ == nQ)
            println("  Q=$Q, Δ=$(round(res.Δ; digits=5)), μ=$(round(μ_Q; digits=4)), n=$(round(n_Q; digits=4))")
        end
    end

    # ---------- Build ω grid ----------
    if ω_range === nothing
        ω_range = (-global_E_max * 1.2, global_E_max * 1.2)
    end
    ω_list = collect(range(ω_range[1], ω_range[2]; length=nω))

    # ---------- Second pass: compute DOS for each Q ----------
    dos_matrix = zeros(Float64, nω, nQ)

    for iQ in 1:nQ
        dos_matrix[:, iQ] = compute_dos(eps_list[iQ], Eql_list[iQ], μ_list[iQ], ω_list; η=η)
    end

    if verbose
        println("="^60)
        println("DOS heatmap computation complete.")
    end

    return (dos_matrix=dos_matrix, ω_list=ω_list,
            Q_list=collect(Q_list), phi_list=phi_list,
            Δ_list=Δ_list, μ_list=μ_list, n_list=n_list,
            L=L, U=U, T=T, mode=mode,
            target_n=(mode == :constant_n ? target_n : nothing))
end

"""
    plot_dos_heatmap(dh; use_phi=true, kwargs...)

Plot the DOS heatmap from the output of `dos_heatmap`.
  - `use_phi=true`:  x-axis is ϕ/Φ₀ = Q/N
  - `use_phi=false`: x-axis is Q (integer flux quanta)
"""
function plot_dos_heatmap(dh; use_phi::Bool=true, clims::Union{Nothing,Tuple}=nothing, kwargs...)
    x_vals = use_phi ? dh.phi_list : Float64.(dh.Q_list)
    x_label = use_phi ? "ϕ / Φ₀" : "Q"

    mode_str = dh.mode == :constant_mu ? "μ=$(round(dh.μ_list[1]; digits=2))" :
                                          "n=$(dh.target_n)"

    # Transpose: rows=ω, cols=Q → heatmap wants (x, y, z) with z[y, x]
    kw = Dict{Symbol,Any}()
    if clims !== nothing
        kw[:clims] = clims
    end

    p = heatmap(x_vals, dh.ω_list, dh.dos_matrix;
                xlabel=x_label, ylabel="ω",
                title="DOS Heatmap (L=$(dh.L), U=$(dh.U), T=$(dh.T), $mode_str)",
                colorbar_title="g(ω)", color=:inferno,
                kw..., kwargs...)
    hline!(p, [0.0]; linestyle=:dash, color=:white, alpha=0.4)
    return p
end


# ============================================================================
# Zero-field validation: should match mf_analitic.jl results
# ============================================================================

"""
    run_zero_field_validation(L; U=4.0, μ=0.0, T=0.01)

Run the magnetic field code with Q=0 equivalent (Q=Lx*Ly gives trivial case).
For validation, we compare against the known zero-field formula.
"""
function run_zero_field_check(L::Int; U::Float64=4.0, μ::Float64=0.0, T::Float64=0.01,
                              t::Float64=1.0, tol::Float64=1e-6)
    # Zero-field analytic (from mf_analitic.jl logic)
    β = 1.0 / T
    ki = [2π / L * n for n in 0:(L-1)]
    ksi = [-2t * cos(kx) - 2t * cos(ky) - μ for kx in ki for ky in ki]

    delta = 0.1
    for iter in 1:50000
        E_k = [sqrt(ξ^2 + delta^2) for ξ in ksi]
        delta_new = U * delta * sum(tanh(β * E / 2) / (2E) for E in E_k) / L^2
        if abs(delta_new - delta) < tol
            println("Zero-field analytic: Δ = $(round(delta_new; digits=8)) (converged in $iter iterations)")
            return delta_new
        end
        delta = delta_new
    end
    println("Zero-field analytic: did not converge, Δ ≈ $delta")
    return delta
end


# ============================================================================
# Demo / quick test
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n>>> Demo 1: Fixed μ run (L=6, Q=3, U=4.0, μ=-1.5, T=0.05)\n")
    result = run_magnetic_field_MF(6, 3; U=4.0, μ=-1.5, T=0.05)
    println("\n>>> Result: Δ=$(result.Δ), n=$(result.n), converged=$(result.converged)")

    println("\n>>> Demo 2: Tune μ for target n=0.875 (L=6, Q=3, U=4.0, T=0.05)\n")
    result2 = tune_mu_for_target_n(6, 3, 0.875; U=4.0, T=0.05, mu_low=-4.0, mu_high=0.0)
    println("\n>>> Result: μ=$(result2.μ), Δ=$(result2.Δ), n=$(result2.n)")


    T_range = 0.01:0.01:1.0 
    q = 20
    println("\n>>> Demo 3: Δ(T) sweep (L=6, Q=$q, target_n=0.875)\n")
    # T_range = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    sweep = sweep_delta_vs_T(10, q, T_range; target_n=0.875, U=4.0, mu_low=-4.0, mu_high=0.0)
    plot(sweep.T_list, sweep.Δ_list, marker=:circle, xlabel="Temperature T", ylabel="Pairing Gap Δ",
         title="n=$(sweep.target_n), L=$(sweep.lattice.Lx)x$(sweep.lattice.Ly), U=$(sweep.U), Q=$(sweep.Q)->B=$(round(sweep.Q/sweep.lattice.Lx/sweep.lattice.Ly, digits=2))Φ0",  legend=false)


end

begin
    # l = 10

    # bf = hofstadter_butterfly(10, collect(1:div(l*(l+1), 2)); mode=:constant_mu, μ=-2.0, U=0.0, T=0.05)
    # # bf = hofstadter_butterfly(10, collect(1:div(l*(l+1), 2)); mode=:constant_n, target_n=0.875, U=0.0, T=0.05, mu_low=-4.0, mu_high=0.0)
    # plot_hofstadter_butterfly(bf)
end


begin
    # result = tune_mu_for_target_n(10, 5, 0.875; U=4.0, T=0.05)
    # dos = compute_and_plot_dos(result; nω=500)
    # dos.plot  # the Plots.jl plot object
end

begin
    # with constant μ
    l = 25
    u = 0.0

    dh = dos_heatmap(l, collect(1:div(l*(l+1),2)); mode=:constant_mu, μ=-0.0, U=u, T=0.01, nω=300, η= 0.05)
    f1 = plot_dos_heatmap(dh)
    # f2 = plot(dh.phi_list, dh.n_list, marker=:circle, xlabel="ϕ / Φ₀", ylabel="Occupation n",
    #           title="n vs ϕ (L=$(dh.L), U=$(dh.U), T=$(dh.T))", legend=false)
    # plot(f1, f2; layout=(2,1), size=(800, 800))
end

begin
    # with constant n
    l = 20
    u = 0.0
    dh = dos_heatmap(l, collect(1:div(l*(l+1), 2)); mode=:constant_n, target_n=0.875, U=u, T=0.05, nω=300, η= 0.1, mu_low=-4.0, mu_high=0.0)
    fig = plot_dos_heatmap(dh)
    scatter!(fig, dh.phi_list, -dh.μ_list; marker=:circle, color=:white, label="-μ(ϕ)", markersize=4)
end