# ============================================================================
# statistics.jl — error analysis of Monte-Carlo time series.
#
#   * blocking and the naive standard error          get_block_means, get_raw_std
#   * integrated autocorrelation time                find_correlation_time, chi_k
#   * mean + error of one observable stream          observable_mean, obs_std
#   * combining independent estimates (MC seeds)     get_x_DL (DerSimonian–Laird)
#   * BKT transition temperature from ρ_s(T)         find_tc, find_tc_fit
#
# The scheme used for every production error bar is obs_std(...; scheme=
# "correlation"): block the stream in blocks of 10 τ and take the standard
# error of the block means. See docs/WORKFLOW.md §6.
#
# Plotting helpers (plot_blocked_std, plot_chi_k) need Plots to be loaded by the
# caller; MC_Hubburd.jl does that.
# ============================================================================
using LsqFit
using Statistics

"""
    get_block_means(data::AbstractArray, block_size; return_nothing=false)
        -> (block_means, var_of_mean)

Cut `data` into consecutive blocks of `block_size` points and return the block
means together with the blocking estimate of the variance of the mean,

    var_of_mean = Σ_b (m_b − x̄)² / ((n_b − 1) n_b),

where `x̄` is the mean of ALL of `data` and `n_b = length(data) ÷ block_size`. A
tail shorter than one block is left out of the blocks. `block_size = 0` is
treated as 1.

With fewer than 2 blocks the variance cannot be estimated: returns `nothing` if
`return_nothing`, else `(data, NaN)`.
"""
function get_block_means(data::AbstractArray, block_size::Int; return_nothing=false)
    data_mean = mean(data)
    n = length(data)
    block_size = block_size == 0 ? 1 : block_size
    num_blocks = div(n, block_size)

    if num_blocks < 2
        if return_nothing
            return nothing
        else
            return data, NaN
        end
    end

    means = zeros(eltype(data), num_blocks)
    for i in 1:num_blocks
        start_idx = (i - 1) * block_size + 1
        end_idx = min(i * block_size, n)
        means[i] = mean(data[start_idx:end_idx])
    end
    blocks_var =  sum((means .- data_mean).^2) / ((num_blocks-1) * num_blocks)

    return means, blocks_var
end

"""
    get_block_means(data::Vector{<:AbstractVector}, block_size; return_nothing=false)
        -> (block_means::Matrix, var_of_mean::Vector)

Blocking of a vector-valued stream (`data[i]` is the vector measured at step i),
component by component. `block_means[j, b]` is block `b` of component `j`.

With fewer than 2 blocks: `nothing` if `return_nothing`, else `(data, nothing)`.
"""
function get_block_means(data::Vector{<:AbstractVector}, block_size::Int; return_nothing=false)
    n = length(data)
    num_blocks = div(n, block_size)
    if num_blocks < 2
        if return_nothing
            return nothing
        else
            return data, nothing
        end
    end
    # Each element of data is a vector, so we process each "observable" separately
    obs_len = length(data[1])
    means = zeros(obs_len, num_blocks)
    blocks_var = zeros(obs_len)
    for j in 1:obs_len
        obs_j = [data[i][j] for i in 1:n]
        block_means_j, block_var_j = get_block_means(obs_j, block_size, return_nothing=return_nothing)
        means[j, :] .= block_means_j
        blocks_var[j] = block_var_j
    end
    return means, blocks_var
end

"""
    get_raw_std(obs; obs_mean=mean(obs), M=length(obs)) -> Float64

Naive standard error of the mean, `sqrt(Σ (x − x̄)² / ((M − 1) M))`. Correct only
for uncorrelated samples; MC streams need `obs_std`.
"""
function get_raw_std(obs::AbstractArray; obs_mean=nothing, M=nothing)
    if isnothing(obs_mean)
        obs_mean = mean(obs)
    end
    if isnothing(M)
        M = length(obs)
    end
    return sqrt(sum((obs .- obs_mean).^2) / ((M - 1) * M))
end

"""
    saturation_model(x, p) = p[1] .- p[2] * exp.(-x ./ p[3])

Model of the blocked error versus block size: it rises from the naive error and
saturates at `p[1]` once the blocks are longer than the correlation time (scale
`p[3]`).
"""
function saturation_model(x::AbstractArray, p::AbstractArray)
    return p[1] .- p[2] * exp.(-x./p[3])
end

"""
    fit_to_saturation(obs, num_blocks_at_max) -> (params, block_sizes, block_std, raw_std)

Blocking analysis: the blocked error `sqrt(var_of_mean)` for block sizes
1, 2, 4, …, up to `length(obs) / num_blocks_at_max`, fitted with
`saturation_model`. Returns the fit parameters `[saturated error, amplitude,
scale]`, the block sizes and errors used, and the naive error. If no block size
gives a finite error, returns `((0, -Inf, -Inf), [], [], raw_std)`.
"""
function fit_to_saturation(obs::AbstractArray, num_blocks_at_max::Int)

    largest_bs = length(obs) / num_blocks_at_max
    largest_i = Int64(floor(log2(largest_bs)))
    block_sizes = [2^i for i in 0:largest_i]
    block_std = [sqrt(get_block_means(obs, bs)[2]) for bs in block_sizes]

    valid_mask = .!(isnan.(block_std) .| isinf.(block_std))
    block_std = block_std[valid_mask]
    if isempty(block_std)
        return (0, -Inf, -Inf), [], [], get_raw_std(obs)
    end
    block_sizes = block_sizes[valid_mask]

    raw_std_val = get_raw_std(obs)
    a_guess = maximum(block_std)
    b_guess = a_guess - raw_std_val
    c_guess = num_blocks_at_max

    fit = LsqFit.curve_fit(saturation_model, block_sizes, block_std, [a_guess, b_guess, c_guess])

    return fit.param, block_sizes, block_std, raw_std_val
end

"""
    get_blocked_std(obs, num_blocks_at_max) -> (error, saturation_block_size)

Error of the mean from the saturation fit of `fit_to_saturation`: the saturated
blocked error and the block size `2^ceil(log2(4 * scale))` at which it is reached.
Falls back to `(raw_std, 1)` if the fit gives a negative amplitude or scale.
Used only by `obs_std(...; scheme="blocking")`.
"""
function get_blocked_std(obs::AbstractArray, num_blocks_at_max::Int)

    params, block_sizes, block_std, raw_std = fit_to_saturation(obs, num_blocks_at_max)
    sat_std = params[1]
    if params[2] < 0 || params[3] < 0
        return raw_std, 1
    end
    sat_bs = 2^ceil(log2(4*params[3]))
    if sat_bs >= maximum(block_sizes)
        if num_blocks_at_max > 16
            # go to higher block sizes to see saturation
            get_blocked_std(obs, Int64(num_blocks_at_max/2))
        else
            # not enough data - need to sample more
            # give back the current estimates
            return sat_std, sat_bs
        end
    end
    return sat_std, sat_bs
end

"""
    plot_blocked_std(obs, num_blocks_at_max)

Plot the blocked error versus block size, the naive error and the saturation fit
of `fit_to_saturation`. Needs Plots.
"""
function plot_blocked_std(obs::AbstractArray, num_blocks_at_max::Int)
    params, block_sizes, block_std, raw_std = fit_to_saturation(obs, num_blocks_at_max)

    fig = plot(block_sizes, block_std, xlabel="Block Size", ylabel="Block std", title="Block std vs Block Size, min#blocks=$(num_blocks_at_max)", linewidth=0, label="Block std", marker=:circle, markersize=6)
    hline!(fig, [raw_std], color=:red, label="Raw std", linestyle=:dash, linewidth=2)
    x_fit = minimum(block_sizes):maximum(block_sizes)
    plot!(x_fit, saturation_model(x_fit, params), label="Fit $(round(params[1], sigdigits=2)) + $(round(params[2], sigdigits=2)) * exp(-x/$(round(params[3], sigdigits=2)))", linestyle=:dashdot, color=:green, linewidth=2, legend=:bottomright)
    display(fig)
end


"""
    observable_mean(stream::AbstractVector{<:Number}, num_measure; std_scheme="correlation")
        -> (mean, error)

Mean and error of the first `num_measure` points of a scalar stream. The error
is `obs_std(...; scheme=std_scheme)`. For a complex stream both refer to `|x|`.
Returns `(NaN, NaN)` with a warning if the stream is shorter than `num_measure`.
"""
function observable_mean(observable::AbstractVector{<:Number}, num_measure::Int; std_scheme::String="correlation")
    if length(observable) < num_measure
        @warn "Not enough measurements to calculate mean."
        return NaN, NaN
    end
    # for complex numbers, this is the mean and error of the absolute value
    if eltype(observable) <: Complex
        observable = abs.(observable)
    end
    obs = observable[1:num_measure]

    mean_value = mean(obs)
    error_value = obs_std(obs, obs_mean=mean_value, M=num_measure, scheme=std_scheme)
    return mean_value, error_value
end

"""
    observable_mean(stream::Vector{<:AbstractVector}, num_measure; std_scheme="correlation")
        -> (means::Vector, errors::Vector)

Vector-valued stream (one vector per measurement, e.g. a q- or ω-grid): the
scalar `observable_mean` of every component. Returns `([NaN], [NaN])` for an
empty stream.
"""
function observable_mean(observable::Array{<:AbstractVector{<:Number}, 1}, num_measure::Int; std_scheme::String="correlation")
    if isempty(observable)
        return [NaN], [NaN]
    end

    # Stack the arrays into a matrix for easier computation
    obs_matrix = hcat(observable[1:num_measure]...)
    obses = size(obs_matrix, 1)
    means = zeros(obses)
    errors = zeros(obses)
    for i in 1:obses
        mean_value, error_value = observable_mean(obs_matrix[i, :], num_measure; std_scheme=std_scheme)
        means[i] = mean_value
        errors[i] = error_value
    end
    return means, errors
end

"""
    observable_mean(stream::Vector{<:Vector{<:AbstractVector}}, num_measure; std_scheme="correlation")
        -> (means::Vector{Vector}, errors::Vector{Vector})

Fixed-δ observables, `stream[measurement][δ index][ω index]`: the scalar
`observable_mean` of every (δ, ω) component, returned as `means[δ][ω]`.
"""
function observable_mean(observable::Array{<:Array{<:AbstractVector{<:Number}, 1}, 1}, num_measure::Int; std_scheme::String="correlation")
    if isempty(observable)
        return Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    end

    means_vec = Vector{Vector{Float64}}()
    errors_vec = Vector{Vector{Float64}}()

    # Get the number of delta values from the first measurement
    if !isempty(observable[1])
        num_deltas = length(observable[1])

        for delta_idx in 1:num_deltas
            # Collect all measurements for this delta index
            delta_measurements = []
            for meas_idx in 1:min(num_measure, length(observable))
                if delta_idx <= length(observable[meas_idx])
                    push!(delta_measurements, observable[meas_idx][delta_idx])
                end
            end

            if !isempty(delta_measurements)
                # Stack into matrix and process each frequency component
                obs_matrix = hcat(delta_measurements...)
                num_freq = size(obs_matrix, 1)
                means = zeros(num_freq)
                errors = zeros(num_freq)

                for freq_idx in 1:num_freq
                    mean_value, error_value = observable_mean(obs_matrix[freq_idx, :], length(delta_measurements); std_scheme=std_scheme)
                    means[freq_idx] = mean_value
                    errors[freq_idx] = error_value
                end

                push!(means_vec, means)
                push!(errors_vec, errors)
            else
                push!(means_vec, Vector{Float64}())
                push!(errors_vec, Vector{Float64}())
            end
        end
    end

    return means_vec, errors_vec
end


"""
    obs_std(obs; obs_mean=mean(obs), M=length(obs), scheme="correlation", tau_mult=10.0)
        -> Float64

Error of the mean of a correlated stream.

- `"raw"`: the naive error `get_raw_std`.
- `"correlation"` (used everywhere in production): τ from
  `find_correlation_time`; block the stream in blocks of `round(tau_mult * τ)`
  points and return `max(naive error, blocked error)`. Falls back to the naive
  error if τ < 1, τ is NaN, or there are fewer than 2 blocks.
- `"blocking"`: the saturation fit of `get_blocked_std` with at least 32 blocks;
  the naive error unless the blocked one is at least 1.5 times larger.

Returns `NaN` if the stream contains NaN or Inf.
"""
function obs_std(obs::AbstractArray; obs_mean=nothing, M=nothing, scheme::String="correlation", tau_mult::Float64=10.0)

    if isnothing(obs_mean)
        obs_mean = mean(obs)
    end
    if isnothing(M)
        M = length(obs)
    end

    if any(isnan, obs) || any(isinf, obs)
        return NaN
    end
    raw_std = get_raw_std(obs; obs_mean=obs_mean, M=M)

    if scheme=="raw"
        return raw_std
    elseif scheme=="correlation"
        # Use autocorrelation time to determine block size
        τ = find_correlation_time(obs)
        if isnan(τ) || τ < 1.0
            return raw_std
        end
        # Block size should be at least 2τ to ensure uncorrelated blocks
        block_size = max(1, round(Int, tau_mult * τ))
        result = get_block_means(obs, block_size)
        if isnothing(result)
            return raw_std  # Not enough data for blocking
        end
        _, blocks_var = result
        if isnothing(blocks_var) || isnan(blocks_var)
            return raw_std
        end
        blocked_std = sqrt(blocks_var)
        # Use blocked std if it's larger than raw (indicates correlations)
        return max(raw_std, blocked_std)
    elseif scheme=="blocking"
        blocked_std = get_blocked_std(obs, 32)[1]
        if (blocked_std < raw_std) || blocked_std/raw_std < 1.5
            return raw_std
        else
            return blocked_std
        end
    end
end


#### Combining independent estimates (MC seeds, disorder samples)

"""
    disorder_variance(data, data_err) -> (mean, total_variance, disorder_variance)

Split the scatter of independent estimates `data` (errors `data_err`) into MC
noise and a genuine sample-to-sample variance:
`disorder_variance = max(0, s² − mean(σ²))` with `s²` the sample variance, and
`total_variance = (disorder_variance + mean(σ²)) / M`, the variance of the plain
mean.
"""
function disorder_variance(data::AbstractArray, data_err::AbstractArray)
    M = length(data)

    data_mean = mean(data)
    disorder_scatter = sum((data .- data_mean).^2) / (M - 1)
    MC_noise = mean(data_err.^2)
    disorder_var = maximum([0.0, disorder_scatter - MC_noise])
    total_variance = (disorder_var + MC_noise)/M

    return data_mean, total_variance, disorder_var
end

"""
    inverse_variance_weighted_mean(data, data_err) -> (weighted_mean, weighted_variance)

Weighted mean of independent estimates with weights `1 / (σᵢ² + disorder_var)`,
`disorder_var` (a variance) from `disorder_variance`, and the variance
`1 / Σ weights`. Note the second return value is a VARIANCE, not an error.

Fixed 2026-10-22: the weights used `disorder_var^2` (a variance squared, wrong
units); Tc error bars aggregated with this before that date are affected.
"""
function inverse_variance_weighted_mean(data::AbstractArray, data_err::AbstractArray)

    data_mean, total_var, disorder_var  = disorder_variance(data, data_err)
    weights =  (data_err.^ 2 .+ disorder_var).^(-1)
    weighted_mean = sum(data .* weights) / sum(weights)
    weighted_variance = 1 / sum(weights)
    return weighted_mean, weighted_variance
end

"""
    get_x_GD(data, data_err) -> (weighted_mean, sum_of_weights)

Fixed-effect (inverse-variance) weighted mean, weights `wᵢ = 1/σᵢ²`.
"""
function get_x_GD(data::AbstractArray, data_err::AbstractArray)
    xgd = 0.0
    w_sum = sum(data_err .^ -2)
    for i in eachindex(data)
        xgd += data[i] * (data_err[i]^-2)
    end
    xgd /= w_sum
    return xgd, w_sum
end

"""
    get_y_DL(data, data_err) -> Float64

DerSimonian–Laird estimate of the between-estimate variance τ²:

    τ² = max(0, (Q − (k − 1)) / (Σw − Σw²/Σw)),   Q = Σ wᵢ (xᵢ − x̄_w)²,

with `wᵢ = 1/σᵢ²` and `x̄_w` from `get_x_GD`. Returns 0 if the denominator is
below 1e-12.
"""
function get_y_DL(data::AbstractArray, data_err::AbstractArray)
    xgd, w_sum = get_x_GD(data, data_err)
    p = length(data)
    ydl = 1 - p
    numi = w_sum - (sum(data_err.^(-4)) / w_sum)

    # Prevent division by zero or near-zero
    if abs(numi) < 1e-12
        return 0.0
    end

    for i in eachindex(data)
        ydl += data_err[i]^(-2) * (data[i] - xgd)^2
    end
    ydl /= numi
    return maximum([ydl, 0.0])
end

"""
    get_x_DL(data, data_err) -> (mean, error)

DerSimonian–Laird random-effects mean of independent estimates: weights
`1 / (σᵢ² + τ²)` with τ² from `get_y_DL`, error `1 / sqrt(Σ weights)`. Estimates
that disagree by more than their errors get a larger combined error. This is how
MC seeds are averaged (`obs_seed_mean2` in sim_num_control.jl).
"""
function get_x_DL(data::AbstractArray, data_err::AbstractArray)
    ydl = get_y_DL(data, data_err)
    ws = (ydl .+ data_err.^2).^(-1)
    xdl = sum(data .* ws) / sum(ws)
    xdl_err = sqrt(1 / sum(ws))
    return xdl, xdl_err
end

"""
    weighted_mean(obs, errs) -> Float64

Inverse-variance weighted mean, weights `1/σᵢ²`.
"""
function weighted_mean(obs::Vector{Float64}, errs::Vector{Float64})
    weights = errs .^ -2
    wmean = sum(obs .* weights) / sum(weights)
    return wmean
end

"""
    weighted_var_unbiased(obs, errs) -> Float64

Weighted sample variance with weights `1/σᵢ²`, bias-corrected with the effective
sample size `n_eff = (Σw)² / Σw²`.
"""
function weighted_var_unbiased(obs::Vector{Float64}, errs::Vector{Float64})
    weights = errs .^ -2
    neff = (sum(weights))^2 / sum(weights .^ 2)
    wmean = sum(obs .* weights) / sum(weights)
    wvar = (sum(weights .* (obs .- wmean).^2) / sum(weights)) * (neff/ (neff - 1))
    return wvar
end

"""
    weighted_std_unbiased(obs, errs) -> Float64

Error of the weighted mean from the weighted scatter:
`sqrt(weighted_var_unbiased / n_eff)`.
"""
function weighted_std_unbiased(obs::Vector{Float64}, errs::Vector{Float64})
    neff = (sum(errs .^ -2))^2 / sum((errs .^ -2) .^ 2)
    return sqrt(weighted_var_unbiased(obs, errs)/neff)
end


## Bootstrap

"""
    BT_std(data, block_size; B=1000) -> Float64

Bootstrap over the block means of `data` (`B` resamples). Not used by the
pipeline; the derived observables use their own block bootstraps
(`BT_stiffness`, `BT_resistivity_tensor` in sim_num_control.jl).
"""
function BT_std(data::AbstractArray, block_size::Int; B=1000)
    n = length(data)
    num_blocks = div(n, block_size)
    means, _ = get_block_means(data, block_size)
    data_mean = mean(means)
    BT_mean = 0.0
    BT2_mean = 0.0
    new_BT_mean = 0.0
    for i in 1:num_blocks
        j = rand(1:num_blocks)
        BT_mean += means[j] / num_blocks
        BT2_mean += means[j]^2 / num_blocks
        new_BT_mean += (means[j] - data_mean).^2 / num_blocks
    end

    boots_means = zeros(B)
    for b in 1:B
        boots_means[b] = mean(rand(means, num_blocks))
    end

    return sqrt((num_blocks/(num_blocks-1)) * std(boots_means)) #???
end


begin # correlation times

    """
        chi_k(obs, k) -> Float64

    Autocovariance of `obs` at lag `k`: `Σ_{i=1}^{n−k} (xᵢ − x̄)(x_{i+k} − x̄) / (n − k)`.
    """
    function chi_k(obs::AbstractVector{<:Number}, k::Int64)
        n = length(obs)
        mod_n = n - k
        mu = mean(obs)
        chi = 0.0
        for i in 1:mod_n
            chi += (obs[i] -  mu) * (obs[i+k] - mu)
        end
        chi = chi/mod_n
        return chi
    end

    """
        plot_chi_k(obs)

    Plot the autocovariance `chi_k(obs, k)` for every lag. Needs Plots.
    """
    function plot_chi_k(obs::AbstractVector{<:Number})
        n = length(obs)
        ks = 0:(n-1)
        chis = [chi_k(obs, k) for k in ks]
        fig = plot(ks, chis, xlabel="k", ylabel="chi(k)", title="Autocorrelation function", linewidth=2)
        return fig
    end

    """
        taux(obs) -> Float64

    `Σ_{k≥1} (1 − k/n) ρ(k)`, with ρ(k) = `chi_k(obs,k)/chi_k(obs,0)`, summed until
    the first negative autocovariance. Note: no leading 1 and no factor 2, so it
    is NOT the τ convention of `find_correlation_time`. Not used by the pipeline.
    """
    function taux(obs::AbstractVector{<:Number})
        n = length(obs)
        chi0 = chi_k(obs, 0)
        taux = 0.0
        for k in 1:(n-1)
            chik = chi_k(obs, k)
            if chik < 0
                break
            end
            taux += (1 - (k/n)) * chik/chi0
        end
        return taux
    end

    """
        taux(obs::Vector{<:AbstractVector}) -> Vector{Float64}

    `taux` of every component of a vector-valued stream.
    """
    function taux(obs::Array{<:AbstractVector{<:Number}, 1})
        obs_data = hcat(obs...)
        return [taux(obs_data[i, :]) for i in 1:size(obs_data, 1)]
    end

    """
        taux_M(obs, M) -> Float64

    Integrated autocorrelation time with a fixed window: `1 + 2 Σ_{k=1}^{M} ρ(k)`
    (`M` is capped at the stream length).
    """
    function taux_M(obs::AbstractVector{<:Number}, M::Int64)
        n = length(obs)
        if M > n
            M = n
        end
        chi0 = chi_k(obs, 0)
        taux = 1.0
        for k in 1:M
            chik = chi_k(obs, k)
            taux += 2 * chik/chi0
        end
        return taux
    end

    """
        Rx(observable, block_size) -> Float64

    `block_size * var(block means) / var(observable)`: for blocks much longer than
    the correlation time this tends to 2τ.
    """
    function Rx(observable::AbstractVector{<:Number}, block_size::Int64)
        blocks, blocks_std = get_block_means(observable, block_size)
        R = block_size * var(blocks) / var(observable)
        return R
    end


    """
        find_correlation_time(stream; C=5, return_nan=false, complex_mode="reim_max") -> Float64

    Integrated autocorrelation time τ of a scalar stream, with automatic
    windowing:

        τ(M) = 1 + 2 Σ_{k=1}^{M} chi_k(k) / chi_k(0),

    increasing M from 1 and stopping at the first M with τ(M) ≤ M / C.

    - Empty stream: `1.0` (`NaN` if `return_nan`).
    - No stopping point found: `1.0` (or `NaN` if τ is NaN and `return_nan`).
    - Complex stream: `complex_mode="reim_max"` (default) returns the larger of
      the real- and imaginary-part τ; `"abs"` uses `|x|`.
    """
    function find_correlation_time(observable::AbstractVector{<:Number}; C=5, return_nan=false, complex_mode::String="reim_max")
        if isempty(observable)
            return return_nan ? NaN : 1.0
        end

        n = length(observable)
        if typeof(observable[1]) <: Complex
            if complex_mode == "abs"
                observable = abs.(observable)
            elseif complex_mode == "reim_max"
                tau_re = find_correlation_time(real.(observable); C=C, return_nan=return_nan, complex_mode=complex_mode)
                tau_im = find_correlation_time(imag.(observable); C=C, return_nan=return_nan, complex_mode=complex_mode)
                if return_nan && isnan(tau_re) && isnan(tau_im)
                    return NaN
                end
                tau_re = isnan(tau_re) ? 1.0 : tau_re
                tau_im = isnan(tau_im) ? 1.0 : tau_im
                return max(tau_re, tau_im)
            else
                error("complex_mode must be 'reim_max' or 'abs' (got $(complex_mode))")
            end
        end
        chi_0 = chi_k(observable, 0)
        tau = 1.0
        M = 1
        while M < n
            tau += 2*chi_k(observable, M)/chi_0

            if tau <= M/C # if tau starts decreasing, stop
                return tau
            end
            M += 1
        end

        if isnan(tau) && return_nan
            return tau
        else
            return 1.0
        end

    end


    """
        find_correlation_time(stream::Vector{<:AbstractVector}; C=5, complex_mode="reim_max")
            -> Vector{Float64}

    τ of every component of a vector-valued stream.
    """
    function find_correlation_time(observable::Array{<:AbstractVector{<:Number}, 1}; C=5, complex_mode::String="reim_max")
        obs_data = hcat(observable...)
        return [find_correlation_time(obs_data[i, :]; C=C, complex_mode=complex_mode) for i in 1:size(obs_data, 1)]
    end

    """
        find_correlation_time(stream::Vector{<:Vector{<:AbstractVector}}; C=5, complex_mode="reim_max")
            -> Vector{Vector{Float64}}

    Fixed-δ observables, `stream[measurement][δ][ω]`: τ of every (δ, ω)
    component, returned as `τ[δ][ω]`.
    """
    function find_correlation_time(observable::Array{<:Array{<:AbstractVector{<:Number}, 1}, 1}; C=5, complex_mode::String="reim_max")
        if isempty(observable)
            return Vector{Vector{Float64}}()
        end

        corr_times_vec = Vector{Vector{Float64}}()

        # Get the number of delta values from the first measurement
        if !isempty(observable[1])
            num_deltas = length(observable[1])

            for delta_idx in 1:num_deltas
                # Collect all measurements for this delta index
                delta_measurements = []
                for meas_idx in 1:length(observable)
                    if delta_idx <= length(observable[meas_idx])
                        push!(delta_measurements, observable[meas_idx][delta_idx])
                    end
                end

                if !isempty(delta_measurements)
                    # Stack into matrix and compute correlation time for each frequency
                    obs_matrix = hcat(delta_measurements...)
                    num_freq = size(obs_matrix, 1)
                    corr_times = zeros(num_freq)

                    for freq_idx in 1:num_freq
                        corr_times[freq_idx] = find_correlation_time(obs_matrix[freq_idx, :]; C=C, complex_mode=complex_mode)
                    end

                    push!(corr_times_vec, corr_times)
                else
                    push!(corr_times_vec, Vector{Float64}())
                end
            end
        end

        return corr_times_vec
    end
end

begin # Finding Tc from stiffness data
    """
        find_tc_fit(sims; lowest_q=false, window=3, order=:auto, prefer_above_below=true)
            -> (Tc, Tc_err, crosses)

    BKT temperature from the crossing of the superfluid stiffness with the
    universal line, d(T) = ρ_s(T) − 2T/π = 0, by a local polynomial fit.

    `sims` is a temperature series in the V6 `get_sims` format (read with
    `get_obs_vs_T`, sim_num_control.jl); `lowest_q=true` uses `stiffness_lowest_q`.

    1. Find adjacent temperatures where d changes sign; prefer the first
       above→below crossing as T increases, else any sign change.
    2. Take ±`window` points around it.
    3. Weighted least-squares fit of d(T) to a polynomial of order `order`
       (`:auto`: 2 if the window has ≥ 5 points, else 1), weights 1/σ².
    4. Solve d(Tc) = 0 and propagate the fit covariance (implicit function
       theorem): σ_Tc² = vᵀ Cov(β) v / d′(Tc)², vₖ = Tcᵏ.

    `Tc`, `Tc_err` are NaN when there is no crossing or the fit fails; `crosses`
    tells whether a sign change was found.
    """
    function find_tc_fit(sims; lowest_q=false, window::Int=3,
                         order::Union{Int, Symbol}=:auto,
                         prefer_above_below::Bool=true)
        Ts, stiff, stiff_err = get_obs_vs_T(sims, lowest_q ? "stiffness_lowest_q" : "stiffness")
        if isempty(Ts) || length(Ts) < 2
            return NaN, NaN, false
        end

        # Filter non-finite measurements and sort by T.
        mask = isfinite.(Ts) .& isfinite.(stiff)
        Ts = Float64.(Ts[mask])
        stiff = Float64.(stiff[mask])
        stiff_err = Float64.(stiff_err[mask])
        if length(Ts) < 2
            return NaN, NaN, false
        end
        ord = sortperm(Ts)
        Ts = Ts[ord]; stiff = stiff[ord]; stiff_err = stiff_err[ord]

        d = stiff .- 2 .* Ts ./ pi
        n = length(Ts)

        # Find crossing pair: prefer above->below, fall back to any sign change.
        cross_i = 0
        if prefer_above_below
            for i in 1:(n-1)
                if d[i] > 0 && d[i+1] < 0
                    cross_i = i
                    break
                end
            end
        end
        if cross_i == 0
            for i in 1:(n-1)
                s1, s2 = sign(d[i]), sign(d[i+1])
                if s1 != 0 && s2 != 0 && s1 != s2
                    cross_i = i
                    break
                end
            end
        end
        if cross_i == 0
            return NaN, NaN, false
        end

        # Local window around the crossing pair.
        lo = max(1, cross_i - window + 1)
        hi = min(n, cross_i + window)
        T_loc = Ts[lo:hi]
        d_loc = d[lo:hi]
        err_loc = stiff_err[lo:hi]
        err_loc = [(isfinite(e) && e > 0) ? e : 1.0 for e in err_loc]
        m = length(T_loc)

        # Polynomial order, clamped by window size.
        p_order = order === :auto ? (m >= 5 ? 2 : 1) : Int(order)
        p_order = clamp(p_order, 1, max(1, m - 1))

        # Weighted least squares: fit d(T) = sum_k beta_k * T^k.
        w = 1.0 ./ (err_loc .^ 2)
        X = [T_loc[i]^k for i in 1:m, k in 0:p_order]
        sw = sqrt.(w)
        Xw = X .* sw
        dw = d_loc .* sw
        A = Xw' * Xw
        b_vec = Xw' * dw
        local beta, covbeta
        try
            beta = A \ b_vec
            covbeta = inv(A)
        catch
            return NaN, NaN, true
        end

        # Solve d(Tc) = 0 inside the local window.
        T_mid = 0.5 * (Ts[cross_i] + Ts[cross_i+1])
        Tlo, Thi = T_loc[1], T_loc[end]
        Tc = NaN
        if p_order == 1
            if beta[2] != 0
                Tc = -beta[1] / beta[2]
            end
        else
            # Use the leading three coefficients (linear/quadratic dominate here).
            a0, a1, a2 = beta[1], beta[2], beta[3]
            if a2 == 0
                a1 != 0 && (Tc = -a0 / a1)
            else
                disc = a1^2 - 4 * a2 * a0
                if disc >= 0
                    sq = sqrt(disc)
                    r1 = (-a1 + sq) / (2 * a2)
                    r2 = (-a1 - sq) / (2 * a2)
                    in_win(r) = (r >= Tlo - 1e-8) && (r <= Thi + 1e-8)
                    if in_win(r1) && !in_win(r2)
                        Tc = r1
                    elseif in_win(r2) && !in_win(r1)
                        Tc = r2
                    else
                        Tc = abs(r1 - T_mid) < abs(r2 - T_mid) ? r1 : r2
                    end
                end
            end
        end

        if !isfinite(Tc)
            return NaN, NaN, true
        end

        # Error via implicit function theorem on d(Tc; beta) = 0.
        d_prime = sum(k * beta[k+1] * Tc^(k-1) for k in 1:p_order)
        if !isfinite(d_prime) || d_prime == 0
            return Tc, NaN, true
        end
        v = [Tc^k for k in 0:p_order]
        var_Tc = (v' * covbeta * v) / d_prime^2
        Tc_err = (isfinite(var_Tc) && var_Tc > 0) ? sqrt(var_Tc) : NaN

        return Tc, Tc_err, true
    end

    """
        find_tc(sims; lowest_q=false) -> (Tc, crosses, closest_index)

    BKT temperature without a fit: the temperature where |ρ_s(T) − 2T/π| is
    smallest, refined by linear interpolation when the sign of ρ_s − 2T/π
    changes next to it (`crosses = true`). `sims` as in `find_tc_fit`. Returns
    `(NaN, false, nothing)` for an empty series.
    """
    function find_tc(sims; lowest_q=false)
    Ts, stiff, stiff_err = get_obs_vs_T(sims, lowest_q ? "stiffness_lowest_q" : "stiffness")

    bkt_line = 2 .* Ts ./ pi
    diffs = abs.(stiff .- bkt_line)
    if isempty(diffs)
        return NaN, false, nothing
    end
    min_index = argmin(diffs)
    # Check for crossing with neighbor
    crosses = false
    lower_idx = nothing
    upper_idx = nothing
    if min_index > 1 && sign(stiff[min_index-1] - bkt_line[min_index-1]) != sign(stiff[min_index] - bkt_line[min_index])
        crosses = true
        lower_idx = min_index - 1
        upper_idx = min_index
    elseif min_index < length(Ts) && sign(stiff[min_index+1] - bkt_line[min_index+1]) != sign(stiff[min_index] - bkt_line[min_index])
        crosses = true
        lower_idx = min_index
        upper_idx = min_index + 1
    end

    Tc = Ts[min_index]
    if crosses
        # Linear interpolation for better Tc estimate
        y1 = stiff[lower_idx]; y2 = stiff[upper_idx]
        b1 = bkt_line[lower_idx]; b2 = bkt_line[upper_idx]
        t1 = Ts[lower_idx]; t2 = Ts[upper_idx]
        denom = (y2 - y1) - (b2 - b1)
        if abs(denom) > 0
            α = (b1 - y1) / denom
            Tc = t1 + (t2 - t1) * α
        end
    end
    return Tc, crosses, min_index
end
end
