# dwave-kspaceMF.jl — clean (uniform U) k-space mean field for bond (d-wave) pairing.
#
# Experimental, like all non-s-wave pairing in this code (not benchmarked). Used by
# saddle_point_delta (MC_Hubburd.jl) to seed the initial guess of d-wave runs.
# The s-wave / density-channel mean field is in mf_kspace_density_channel.jl.

using Statistics, Plots

"""
    ksi_k(kx, ky; t=1.0, mu=0.0)

Band energy ξ_k = -2t(cos kx + cos ky) - μ.
"""
function ksi_k(kx::Float64, ky::Float64; t::Float64=1.0, mu::Float64=0.0)
    return -2*t*(cos(kx) + cos(ky)) - mu
end

"""
    Delta_k_dwave(kx, ky; Deltax=1.0, Deltay=-1.0)

Gap of the bond pairing fields: Δ_k = -2(Δ_x cos kx + Δ_y cos ky) (d-wave for Δ_y = -Δ_x).
"""
function Delta_k_dwave(kx::Float64, ky::Float64; Deltax::ComplexF64=1.0, Deltay::ComplexF64=-1.0)
    return -2(Deltax * cos(kx) + Deltay * cos(ky))
end

"""
    E_k(kx, ky; t=1.0, mu=0.0, Deltax=1.0, Deltay=-1.0)

Quasiparticle energy E_k = √(ξ_k² + |Δ_k|²).
"""
function E_k(kx::Float64, ky::Float64; t::Float64=1.0, mu::Float64=0.0, Deltax::ComplexF64=1.0, Deltay::ComplexF64=-1.0)
    ksik = ksi_k(kx, ky; t=t, mu=mu)
    Deltak = Delta_k_dwave(kx, ky; Deltax=Deltax, Deltay=Deltay)
    return sqrt(ksik^2 + abs2(Deltak))
end

"""
    k_vectors(Lx, Ly) -> Vector{Tuple}
    k_vectors(L)

All allowed momenta (2π i/Lx, 2π j/Ly), i, j = 0 … L-1.
"""
function k_vectors(Lx::Int, Ly::Int)
    dkx = 2π / Lx
    dky = 2π / Ly
    kx_list = [dkx * ix for ix in 0:Lx-1]
    ky_list = [dky * iy for iy in 0:Ly-1]
    kvecs = [(kx, ky) for kx in kx_list for ky in ky_list]
    return kvecs
end

# Legacy wrapper for square lattice
k_vectors(L::Int) = k_vectors(L, L)

"""
    dwave_self_consistent_delta(Lx, Ly, Deltax, Deltay; t=1.0, mu=0.0, beta=100.0, Ux=4.0, Uy=4.0)
        -> [Δx_new, Δy_new]
    dwave_self_consistent_delta(L, Deltax, Deltay; kwargs...)

Right-hand side of the d-wave gap equations:
Δ_a,new = -(U_a/N) Σ_k tanh(β E_k/2) Δ_k cos(k_a) / E_k, for a = x, y.
"""
function dwave_self_consistent_delta(Lx::Int64, Ly::Int64, Deltax::ComplexF64, Deltay::ComplexF64; t::Float64=1.0, mu::Float64=0.0, beta::Float64=100.0, Ux::Float64=4.0, Uy::Float64=4.0)
    N = Lx * Ly
    kvecs = k_vectors(Lx, Ly)
    k_sum = zeros(ComplexF64, 2)
    deltas = copy(k_sum)
    for (kx, ky) in kvecs
        Deltak = Delta_k_dwave(kx, ky; Deltax=Deltax, Deltay=Deltay)
        Ek = E_k(kx, ky; t=t, mu=mu, Deltax=Deltax, Deltay=Deltay)
        mul_term = tanh(beta * Ek / 2) * Deltak / Ek
        k_sum[1] += mul_term * cos(kx) # for Deltax
        k_sum[2] += mul_term * cos(ky) # for Deltay
    end
    deltas[1] = - Ux * k_sum[1] / N
    deltas[2] = - Uy * k_sum[2] / N
    return deltas
end

# Legacy wrapper for square lattice
dwave_self_consistent_delta(L::Int64, Deltax::ComplexF64, Deltay::ComplexF64; kwargs...) = dwave_self_consistent_delta(L, L, Deltax, Deltay; kwargs...)


"""
    running_dwave_MF(Lx, Ly, t, mu, beta; Ux=4.0, Uy=4.0, Dx0=0.5, Dy0=-0.5, max_iter=1000,
                     tol=1e-6, a=0.6) -> ([Δx, Δy], residuals, max_residual_per_iteration)
    running_dwave_MF(L, t, mu, beta; kwargs...)

Solve the d-wave gap equations at fixed μ by mixed iteration
(Δ ← a·Δ + (1 - a)·RHS, `dwave_self_consistent_delta`) from (`Dx0`, `Dy0`) until the
largest residual is below `tol`. Prints whether it converged.
"""
function running_dwave_MF(Lx::Int64, Ly::Int64, t::Float64, mu::Float64, beta::Float64; Ux::Float64=4.0, Uy::Float64=4.0, Dx0::ComplexF64=0.5 + 0im, Dy0::ComplexF64=-0.5 + 0im, max_iter::Int64=1000, tol::Float64=1e-6, a::Float64=0.6)
    Deltax = Dx0
    Deltay = Dy0
    eqs = ComplexF64.([Inf, Inf])
    max_eqs = zeros(Float64, max_iter)

    for iter in 1:max_iter
        eq_deltas = dwave_self_consistent_delta(Lx, Ly, Deltax, Deltay; t=t, mu=mu, beta=beta, Ux=Ux, Uy=Uy)
        eqs[1] = Deltax - eq_deltas[1]
        eqs[2] = Deltay - eq_deltas[2]
        maximum_equation = maximum(abs.(eqs))
        max_eqs[iter] = maximum_equation
        if maximum_equation < tol
            println("Converged in $iter iterations.")
            println("Maximal Final Equation Residual: $maximum_equation")
            println("Deltax = $(Deltax), Deltay = $(Deltay)")
            return ([Deltax, Deltay], eqs, max_eqs[1:iter])
        end
        Deltax_new = a * Deltax + (1-a) * eq_deltas[1]
        Deltay_new =  a * Deltay + (1-a) * eq_deltas[2]
        Deltax = Deltax_new
        Deltay = Deltay_new
    end
    println("Did not converge in $max_iter iterations.")
    return [Deltax, Deltay], eqs, max_eqs
end

# Legacy wrapper for square lattice
running_dwave_MF(L::Int64, t::Float64, mu::Float64, beta::Float64; kwargs...) = running_dwave_MF(L, L, t, mu, beta; kwargs...)

"""
    mf_occupation(Lx, Ly, Deltax, Deltay, mu, beta; t=1.0) -> Float64
    mf_occupation(L, Deltax, Deltay, mu, beta; kwargs...)

Mean-field filling n = 1 - (1/N) Σ_k (ξ_k/E_k) tanh(β E_k/2) for given d-wave gaps.
"""
function mf_occupation(Lx::Int64, Ly::Int64, Deltax::ComplexF64, Deltay::ComplexF64, mu::Float64, beta::Float64; t::Float64=1.0)
    N = Lx * Ly
    kvecs = k_vectors(Lx, Ly)
    n_k = 0.0
    for (kx, ky) in kvecs
        ksi = ksi_k(kx, ky; t=t, mu=mu)
        Ek = E_k(kx, ky; t=t, mu=mu, Deltax=Deltax, Deltay=Deltay)
        n_k += (ksi / Ek) * tanh(beta * Ek / 2)
    end
    n = 1.0 - n_k / N
    return n
end

# Legacy wrapper for square lattice
mf_occupation(L::Int64, Deltax::ComplexF64, Deltay::ComplexF64, mu::Float64, beta::Float64; kwargs...) = mf_occupation(L, L, Deltax, Deltay, mu, beta; kwargs...)
