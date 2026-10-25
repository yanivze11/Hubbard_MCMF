# s-wave saddle-point Delta vs lattice size L at fixed T and fixed filling n.
# At each L, bisect mu so that the BCS filling matches target_n, with the
# saddle-point gap equation re-solved at every mu via `saddle_point_delta`.

using Plots
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "MC_Hubburd.jl"))

# ---------------- parameters ----------------
T              = 0.4                    # temperature
U              = 4.0                    # attraction strength
target_n       = 0.5                    # filling per site (n=1 is half-filling)
t_hop          = 1.0
L_list         = [4, 8, 10, 12, 16, 18, 20]
mu_lims        = (-3.0, 1.0)            # search window for mu
n_resolution   = 1e-4                   # |n - target_n| tolerance
mu_tune_iter   = 60                     # bisection budget
d_min          = 1e-4                   # delta scan resolution

beta = 1.0 / T

# ---------------- BCS filling at given (mu, delta) ----------------
function bcs_filling(U_::Float64, mu_::Float64, beta_::Float64, L::Int, t_::Float64, delta_::Float64)
    kx_list = get_kx_list(L)
    ky_list = get_ky_list(L)
    s = 0.0
    @inbounds for kx in kx_list, ky in ky_list
        ksi = -2 * t_ * (cos(kx) + cos(ky)) - mu_
        E = sqrt(ksi^2 + delta_^2)
        if E > 0
            s += (ksi / E) * tanh(0.5 * beta_ * E)
        end
    end
    N = length(kx_list) * length(ky_list)
    return 1.0 - s / N
end

# Saddle-point delta + filling at a given mu.
function delta_and_n_at_mu(U_, mu_, beta_, L, t_; d_min_=1e-4)
    delta_mf, _, _ = saddle_point_delta(U_, mu_, beta_, L; d_min=d_min_, t=t_)
    delta_val = Float64(delta_mf)
    n_val = bcs_filling(U_, mu_, beta_, L, t_, delta_val)
    return delta_val, n_val
end

# Bisect mu so that n(mu) = target_n. Returns (mu_best, delta_best, n_best, converged).
function tune_mu_for_target_n(U_, beta_, L, t_, target, mu_low, mu_high;
                              tol=1e-4, max_iter=60, d_min_=1e-4)
    delta_lo, n_lo = delta_and_n_at_mu(U_, mu_low, beta_, L, t_; d_min_=d_min_)
    delta_hi, n_hi = delta_and_n_at_mu(U_, mu_high, beta_, L, t_; d_min_=d_min_)

    best_mu, best_delta, best_n =
        abs(n_lo - target) <= abs(n_hi - target) ?
            (mu_low, delta_lo, n_lo) :
            (mu_high, delta_hi, n_hi)

    bracketed = (n_lo - target) * (n_hi - target) <= 0
    if !bracketed
        return best_mu, best_delta, best_n, abs(best_n - target) <= tol, false
    end

    for _ in 1:max_iter
        if abs(best_n - target) <= tol
            break
        end
        mu_try = 0.5 * (mu_low + mu_high)
        dn = n_hi - n_lo
        if isfinite(dn) && abs(dn) > 1e-12
            mu_secant = mu_low + (target - n_lo) * (mu_high - mu_low) / dn
            lo, hi = min(mu_low, mu_high), max(mu_low, mu_high)
            if isfinite(mu_secant) && lo <= mu_secant <= hi
                mu_try = mu_secant
            end
        end
        delta_try, n_try = delta_and_n_at_mu(U_, mu_try, beta_, L, t_; d_min_=d_min_)
        if abs(n_try - target) < abs(best_n - target)
            best_mu, best_delta, best_n = mu_try, delta_try, n_try
        end
        if (n_lo - target) * (n_try - target) <= 0
            mu_high, n_hi = mu_try, n_try
        else
            mu_low, n_lo = mu_try, n_try
        end
    end
    return best_mu, best_delta, best_n, abs(best_n - target) <= tol, true
end

# ---------------- sweep ----------------
deltas, mus, ns = Float64[], Float64[], Float64[]

println("Scanning saddle-point Delta at T = $T (beta = $beta), U = $U, target_n = $target_n")
println(rpad("L", 6), rpad("mu", 12), rpad("n", 12), rpad("Delta_MF", 14), "conv")
println("-"^50)
for L in L_list
    mu_b, d_b, n_b, converged, bracketed = tune_mu_for_target_n(
        U, beta, L, t_hop, target_n, mu_lims[1], mu_lims[2];
        tol=n_resolution, max_iter=mu_tune_iter, d_min_=d_min,
    )
    push!(mus, mu_b); push!(deltas, d_b); push!(ns, n_b)
    @printf("%-6d%-12.6f%-12.6f%-14.8f%s%s\n",
        L, mu_b, n_b, d_b, string(converged), bracketed ? "" : " (not-bracketed)")
end

# ---------------- plot ----------------
plt = plot(L_list, deltas;
    xlabel = "L",
    ylabel = "|Delta|_MF",
    marker = :circle,
    lw = 2,
    label = @sprintf("T = %.3g, U = %.3g, n = %.3g", T, U, target_n),
    title = "s-wave saddle-point Delta vs L (constant filling)",
)

plt_n = plot(L_list, ns;
    xlabel = "L", ylabel = "n", marker = :diamond, lw = 2,
    label = "n(L)", title = "Achieved filling vs L")
hline!(plt_n, [target_n]; color = :red, linestyle = :dash, label = "target n")

plt_mu = plot(L_list, mus;
    xlabel = "L", ylabel = "mu", marker = :square, lw = 2,
    label = "mu(L)", title = "Tuned mu vs L")

combined = plot(plt, plt_n, plt_mu; layout = (3, 1), size = (700, 800))

artifact_dir = joinpath(PROJECT_ROOT, "notebooks", "artifacts")
mkpath(artifact_dir)
fig_path = joinpath(artifact_dir,
    @sprintf("swave_saddle_point_L_scan_T%.3g_U%.3g_n%.3g.png", T, U, target_n))
savefig(combined, fig_path)
println("\nSaved figure: ", fig_path)

display(combined)
