l = [2n for n in 1:15]
N = 10000
# l1 = 10
ms_vec = zeros(length(l))
errs_vec = zeros(length(l))
for (i, l1) in enumerate(l)
    n1 = l1^2
    trs = abs.(mean(exp.(im*2pi*rand(n1,N)), dims=1))
    mn = mean(trs)
    ers = std(trs) / sqrt(N)
    ms_vec[i] = mn
    errs_vec[i] = ers
end

plot(l.^1, ms_vec, yerr=errs_vec, xlab="\$L\$", ylab="\$ \\sum_{i=1}^{L^2} e^{2\\pi i \\theta_i}\$", left_margin=5Plots.mm,xlims=(8,30), ylims=(0.02, 0.13))

function rpa_scale(L::Int64; n::Int64=10000)
    # Implementation for RPA scale calculation
    N = L^2

    # trs = abs2.(mean(exp.(im*2pi*rand(N,n)), dims=1))
    # mn = sqrt(mean(trs))
    # ers = 0.5 * mn^(-1/2) * std(trs) / sqrt(n)

    trs = abs.(mean(exp.(im*2pi*rand(N,n)), dims=1))
    mn = mean(trs)
    ers = std(trs) / sqrt(n)
    return mn, ers
end

function rpa_scale_rms(L::Int64; n::Int64=10000)
    # Implementation for RPA scale calculation
    N = L^2

    trs = abs2.(mean(exp.(im*2pi*rand(N,n)), dims=1))
    mn = sqrt(mean(trs))
    ers = 0.5 * mn^(-1/2) * std(trs) / sqrt(n)
    return mn, ers
end

#####
begin
    csv_path = D("Delta_vs_MF_U4.0_n0.5.csv")
    rows = _read_final_plot_csv(csv_path)
    series = ["Delta_c", "Delta_P"]
    by_series = Dict{Tuple{String,Int,Int}, Vector{NamedTuple{(:T,:value,:err)}}}()
    for r in rows
        r.source in series || continue
        key = (r.source, r.L, r.Lc)
        push!(get!(by_series, key, NamedTuple{(:T,:value,:err)}[]),
              (T=r.T, value=r.value, err=r.err))
    end
    isempty(by_series) && error("No matching series in $csv_path (filtered by $series)")
end

begin
    f = plot(xlab="T", ylab="\$\\Delta\$" ,legend=:topleft, size=(800, 600))
    # rpa_type= "RPA_rms"
    rpa_type= "1/L"
    for l in [10, 20, 26]
        srs_nm = ("Delta_c", l, 10)
        srs = by_series[srs_nm]
        ts = [srs[i].T for i in eachindex(srs)]
        ers = [srs[i].err for i in eachindex(srs)]
        vlue = [srs[i].value for i in eachindex(srs)]
        plot!(f, ts, vlue, yerr=ers, label="\$ \\Delta_c\$; L=$l", lw=2, m=:circle)

        srs_nm = ("Delta_P", l, 10)
        srs = by_series[srs_nm]
        ts = [srs[i].T for i in eachindex(srs)]
        ers = [srs[i].err for i in eachindex(srs)]
        vlue = [srs[i].value for i in eachindex(srs)]
        if rpa_type == "RPA_rms"
            scl = rpa_scale_rms(l)
        elseif rpa_type == "RPA"
            scl = rpa_scale(l)
        elseif rpa_type == "1/L"
            scl = (1/l, 0)  # Simplified scaling for 1/L
        end
        scaled_val = vlue * scl[1]
        scaled_err = sqrt.(scl[2]^2 .+ ers.^2)
        plot!(f, ts, scaled_val, yerr=scaled_err, label="\$ \\Delta_P / L\$; L=$l", lw=2, ls=:dash, m=:square)


        u = 4.0
        # ys = 1/(2 * l) * sqrt.(pi*u.*ts)
 
        ys = sqrt.(2pi*u*ts/(8-u)) / l
        # plot!(f, ts, ys, label="Theoretical; L=$l", lw=2, ls=:dot, m=:none)
    end
    plot!(f, ylims=(0.0, 1.0), xlims=(0.0, 1.0))
end

begin
    srs_nm = ("Delta_P", 10, 10)
    srs = by_series[srs_nm]
    ts = [srs[i].T for i in eachindex(srs)]
    ers = [srs[i].err for i in eachindex(srs)]
    vlue = [srs[i].value for i in eachindex(srs)]
    msk = ts.>=1.0
    rvlue = vlue[msk]
    rts = ts[msk]
    plot(rts, rvlue, m=:circle, xscale=:log10, yscale=:log10)
    dfs = diff(log.(rvlue)) ./ diff(log.(rts))
    mdfs = mean(dfs)
    xs = rts
    ys = xs.^mdfs
    plot!(xs, ys)
end