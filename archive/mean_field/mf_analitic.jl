using LinearAlgebra
using Plots

function find_self_consistent_delta(L, mu, beta; U=4.0, t=1.0, tol=1e-6, max_iter=10000)
    ki = [2*pi/L * n for n in 0:(L-1)]
    k_list = [[kx, ky] for kx in ki for ky in ki]
    ksi = [-2t*cos(k[1]) - 2t*cos(k[2]) - mu for k in k_list]
    delta = 0.1
    for iter in 1:max_iter
        E_k = [sqrt(ksi_i^2 + delta^2) for ksi_i in ksi]
        delta_new = U * delta * sum(tanh(beta * E / 2) / (2 * E) for E in E_k) / (L^2)
        if abs(delta_new - delta) < tol
            return delta_new
        end
        delta = delta_new
    end
    throw("Did not converge within $max_iter iterations")
end


function occupation_n(L, mu, beta; U=4.0, t=1.0)
    ki = [2*pi/L * n for n in 0:(L-1)]
    k_list = [[kx, ky] for kx in ki for ky in ki]
    ksi = [-2t*cos(k[1]) - 2t*cos(k[2]) - mu for k in k_list]
    delta = find_self_consistent_delta(L, mu, beta; U=U, t=t)
    E_k = [sqrt(ksi_i^2 + delta^2) for ksi_i in ksi]
    n = 0.0
    for i in 1:L^2
        n += ksi[i]/E_k[i] * tanh(beta * E_k[i] / 2)
    end
    n = 1 - n/(L^2)
    return n
end


function tune_mu_for_n(L, target_n, beta; U=4.0, t=1.0, tol=1e-6, max_iter=100)
    mu_low = -U
    mu_high = 0.0
    for iter in 1:max_iter
        mu_mid = (mu_low + mu_high) / 2
        n_mid = occupation_n(L, mu_mid, beta; U=U, t=t)
        if abs(n_mid - target_n) < tol
            return mu_mid
        elseif n_mid < target_n
            mu_low = mu_mid
        else
            mu_high = mu_mid
        end
    end
    throw("Did not converge within $max_iter iterations")
end


function kinetic_energy_Kx(L, mu, beta; U=4.0, t=1.0)
    ki = [2*pi/L * n for n in 0:(L-1)]
    k_list = [[kx, ky] for kx in ki for ky in ki]
    ksi = [-2t*cos(k[1]) - 2t*cos(k[2]) - mu for k in k_list]
    delta = find_self_consistent_delta(L, mu, beta; U=U, t=t)
    E_k = [sqrt(ksi_i^2 + delta^2) for ksi_i in ksi]
    Kx = 0.0
    for i in 1:L^2
        kx = k_list[i][1]
        Kx += cos(kx) * (1 - ksi[i]/E_k[i] * tanh(beta * E_k[i] / 2))
    end
    Kx *= -(2 * t) / (L^2)
    return Kx
end

function Current_Current_correlation_q(L, mu, beta, q::Tuple{Float64, Float64}; U=4.0, t=1.0)
    ki = [2*pi/L * n for n in 0:(L-1)]
    k_list = [[kx, ky] for kx in ki for ky in ki]
    delta = find_self_consistent_delta(L, mu, beta; U=U, t=t)
    Lxx = 0.0
    for i in 1:L^2
        k = k_list[i]
        kpq = [k_list[i][1] + q[1], k_list[i][2] + q[2]]
        ksi_k = -2t*cos(k[1]) - 2t*cos(k[2]) - mu 
        E_k = sqrt(ksi_k^2 + delta^2)
        f_k = (1 + exp(beta * E_k))^-1
        ksi_kpq = -2t*cos(kpq[1]) - 2t*cos(kpq[2]) - mu
        E_kpq = sqrt(ksi_kpq^2 + delta^2)
        f_kpq = (1 + exp(beta * E_kpq))^-1

        # Lxx += (sin(k[1] + q[1]/2))^2 * ((1 + ksi_k*ksi_kpq/(E_k*E_kpq)) * (f_kpq - f_k) / (E_k - E_kpq) + (1 - ksi_k*ksi_kpq/(E_k*E_kpq)) * (1 - f_k - f_kpq) / (E_k + E_kpq))
        Lxx += (sin(k[1] + q[1]/2))^2 * ((1 + (ksi_k*ksi_kpq + delta^2)/(E_k*E_kpq)) * (f_kpq - f_k) / (E_k - E_kpq) + (1 - (ksi_k*ksi_kpq + delta^2)/(E_k*E_kpq)) * (1 - f_k - f_kpq) / (E_k + E_kpq))
        
    end
    Lxx *= 4*t^2/(L^2)
    return Lxx

end

# my_L = 12
# my_n = 0.875
# betas = [0.5:0.05:8.0;]
# Ccorr_qx = [Current_Current_correlation_q(my_L, tune_mu_for_n(my_L, my_n, beta), beta, (2*pi/my_L, 0.0)) for beta in betas]
# CCorr_qy = [Current_Current_correlation_q(my_L, tune_mu_for_n(my_L, my_n, beta), beta, (0.0, 2*pi/my_L)) for beta in betas]
# Kx_vals = [kinetic_energy_Kx(my_L, tune_mu_for_n(my_L, my_n, beta), beta) for beta in betas]

# plot(betas.^-1, Ccorr_qx, xlabel="Temperature", ylabel="Values", label="L_xx(qx)")
# plot!(betas.^-1, CCorr_qy, label="L_xx(qy)")
# plot!(betas.^-1, -Kx_vals, label="-Kx")

# plot(betas.^-1, -CCorr_qy .- Kx_vals, xlabel="Temperature", ylabel="Values", label="ρs = -L_xx(qy) - Kx")
# plot!(betas.^-1, -Ccorr_qx .- Kx_vals, xlabel="Temperature", ylabel="Values", label="ρs = -L_xx(qx) - Kx")

# deltas = [find_self_consistent_delta(my_L, tune_mu_for_n(my_L, my_n, beta), beta) for beta in betas]
# plot(betas.^-1, deltas, xlabel="Temperature", ylabel="Delta", label="Delta vs T")


# my_L = 64
# my_beta = 20.0
# my_n = 0.875
# qys = [2*pi*n/my_L for n in 0:(my_L/2)]
# cc_corr_qy = [Current_Current_correlation_q(my_L, tune_mu_for_n(my_L, my_n, my_beta), my_beta, (0.0, qy)) for qy in qys]
# cc_corr_qx = [Current_Current_correlation_q(my_L, tune_mu_for_n(my_L, my_n, my_beta), my_beta, (qx, 0.0)) for qx in qys]
# kx = kinetic_energy_Kx(my_L, tune_mu_for_n(my_L, my_n, my_beta), my_beta)
# plot(qys, cc_corr_qy, xlabel="q_y", ylabel="L_xx(qy)", label="L_xx vs qy at low T", marker=:circle)
# plot!(qys, cc_corr_qx, xlabel="q_x", ylabel="L_xx(qx)", label="L_xx vs qx at low T", marker=:circle)
# scatter!([0.0], [-kx], label="-Kx", markershape=:star5, markersize=8)