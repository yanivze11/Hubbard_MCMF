# benchmark_MC_stepV2_measurement_scaling.jl — how the time of one MC sweep
# (MC_stepV2!) and of one measurement (make_measurement_K!) grows with the lattice
# size and the TCA cluster size. Options as --key=value (--help lists them); plots
# and tables go to demo_results/benchmark_mc_step_measurement/.
using Statistics
using Printf
using Dates
using Plots

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

Base.@kwdef mutable struct BenchConfig
    L_list::Vector{Int} = [8, 10, 12, 14]
    tca_list::Vector{Int} = [4, 6, 8, 10]
    fixed_tca::Int = 8
    fixed_l::Int = 14
    reps::Int = 30
    warmup_steps::Int = 2
    warmup_measurements::Int = 1
    mem_cap_gb::Float64 = 2.0
    diag_method::String = "K0-TCA"
    pairing::String = "s-wave"
    loop_random::Bool = false
    out_dir::String = joinpath(@__DIR__, "..", "demo_results", "benchmark_mc_step_measurement")
end

parse_int_list(text::AbstractString) = [parse(Int, strip(part)) for part in split(text, ',') if !isempty(strip(part))]

function parse_bool(text::AbstractString)
    val = lowercase(strip(text))
    if val in ("1", "true", "yes", "y", "on")
        return true
    elseif val in ("0", "false", "no", "n", "off")
        return false
    end
    error("Could not parse Bool value from '$text'")
end

function print_help()
    println("Benchmark MC_stepV2! and make_measurement_K! scaling")
    println("Usage:")
    println("  julia test_code/benchmark_MC_stepV2_measurement_scaling.jl [--key=value ...]")
    println("Options:")
    println("  --L-list=8,10,12,14")
    println("  --TCA-list=4,6,8,10")
    println("  --fixed-tca=8")
    println("  --fixed-l=14")
    println("  --reps=30")
    println("  --warmup-steps=2")
    println("  --warmup-measurements=1")
    println("  --mem-cap-gb=2.0")
    println("  --diag-method=K0-TCA")
    println("  --pairing=s-wave")
    println("  --loop-random=false")
    println("  --out-dir=demo_results/benchmark_mc_step_measurement")
    println("  --help")
end

function parse_cli_args(args::Vector{String}=ARGS)
    cfg = BenchConfig()
    for arg in args
        if arg == "--help"
            print_help()
            exit(0)
        elseif startswith(arg, "--L-list=")
            cfg.L_list = parse_int_list(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--TCA-list=")
            cfg.tca_list = parse_int_list(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--fixed-tca=")
            cfg.fixed_tca = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--fixed-l=")
            cfg.fixed_l = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--reps=")
            cfg.reps = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--warmup-steps=")
            cfg.warmup_steps = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--warmup-measurements=")
            cfg.warmup_measurements = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--mem-cap-gb=")
            cfg.mem_cap_gb = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--diag-method=")
            cfg.diag_method = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--pairing=")
            cfg.pairing = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--loop-random=")
            cfg.loop_random = parse_bool(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--out-dir=")
            cfg.out_dir = String(split(arg, "=", limit=2)[2])
        else
            @warn "Ignoring unknown argument" arg
        end
    end

    isempty(cfg.L_list) && error("L-list cannot be empty")
    isempty(cfg.tca_list) && error("TCA-list cannot be empty")
    cfg.reps > 0 || error("reps must be > 0")
    cfg.fixed_l > 0 || error("fixed-l must be > 0")
    cfg.fixed_tca > 0 || error("fixed-tca must be > 0")
    cfg.mem_cap_gb > 0 || error("mem-cap-gb must be > 0")

    return cfg
end

function estimate_data_memory_gb(Lx::Int, Ly::Int; N_SC::Int=1)
    N = Lx * Ly
    M = 2 * N
    G = max(max(Lx, Ly), N_SC)
    bytes_per_big_matrix = M * M * sizeof(ComplexF64)
    approx_num_big_matrices = 3 + 2 * G
    return (approx_num_big_matrices * bytes_per_big_matrix) / 1024^3
end

function get_data_memory_gb(Lx::Int; tcax::Int=10, tcay::Int=10)
    data = make_data(Lx=Lx, Ly=Lx, tcx=tcax, tcy=tcay)
    return Base.summarysize(data) / 1024^3
end

function make_data(; Lx::Int=16, Ly::Int=16, tcx::Int=4, tcy::Int=4,
                   diag_method::String="K0-TCA", pairing::String="s-wave",
                   loop_random::Bool=false, saving_interval::Int=128)
    sim = SimData(
        Lx=Lx,
        Ly=Ly,
        U=1.0,
        t=1.0,
        V=0.0,
        seed=12345,
        mu=0.0,
        beta=10.0,
        num_measure=max(10, saving_interval),
        num_thermal=10,
        measuring_interval=1,
        saving_interval=saving_interval,
        guess="uniform",
        U_disorder="uniform",
        diag_method=diag_method,
        pairing_symmetry=pairing,
        directory=".",
        num_cores=1,
        num_threads=1,
        TCluster_size_x=tcx,
        TCluster_size_y=tcy,
        loop_random=loop_random,
        w_list=collect(-4.0:0.1:4.0),
        w_conductivity_range = collect(range(10^-3, 3, 12)),
        delta_values_fixed = [0.5; 1:10;],
        dw_values = [0.5; 1:10;]
    )
    return Lattice_Data(sim)
end

function summarize_ns(times_ns::Vector{Int64})
    t = Float64.(times_ns)
    std_val = length(t) > 1 ? std(t) : 0.0
    return (
        median_ns=median(t),
        mean_ns=mean(t),
        std_ns=std_val,
        min_ns=minimum(t),
        max_ns=maximum(t),
    )
end

function benchmark_mc_step(data::LatticeData;
                           reps::Int=30,
                           warmup_steps::Int=2,
                           method::String="K0-TCA",
                           pairing::String="s-wave",
                           loop_random::Bool=false)
    for _ in 1:warmup_steps
        MC_stepV2!(data; method=method, pairing=pairing, loop_random=loop_random)
    end

    GC.gc()
    times_ns = Vector{Int64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        MC_stepV2!(data; method=method, pairing=pairing, loop_random=loop_random)
        times_ns[i] = time_ns() - t0
    end
    return summarize_ns(times_ns)
end

function benchmark_measurement_full(data::LatticeData;
                                    reps::Int=30,
                                    warmup_measurements::Int=1,
                                    method::String="K0-TCA",
                                    pairing::String="s-wave",
                                    loop_random::Bool=false)
    saving_interval = data.sim_data.saving_interval
    measurement_ix = 1

    # Emulate normal usage: each measurement follows an MC step.
    for _ in 1:warmup_measurements
        MC_stepV2!(data; method=method, pairing=pairing, loop_random=loop_random)
        invalidate_global_diag!(data)
        make_measurement_K!(data, measurement_ix)
        measurement_ix = measurement_ix < saving_interval ? measurement_ix + 1 : 1
    end

    GC.gc()
    times_ns = Vector{Int64}(undef, reps)
    for i in 1:reps
        MC_stepV2!(data; method=method, pairing=pairing, loop_random=loop_random)
        invalidate_global_diag!(data)
        t0 = time_ns()
        make_measurement_K!(data, measurement_ix)
        times_ns[i] = time_ns() - t0
        measurement_ix = measurement_ix < saving_interval ? measurement_ix + 1 : 1
    end
    return summarize_ns(times_ns)
end

function bench_point(Lx::Int, Ly::Int, tcx::Int, tcy::Int, cfg::BenchConfig, sweep_type::String;
                     benchmark_measurement::Bool=true)
    est_gb = estimate_data_memory_gb(Lx, Ly)
    if est_gb > cfg.mem_cap_gb
        @printf("Skipping point (%s): Lx=%d Ly=%d tcx=%d tcy=%d, estimated memory %.2f GB > cap %.2f GB\n",
                sweep_type, Lx, Ly, tcx, tcy, est_gb, cfg.mem_cap_gb)
        return nothing
    end

    @printf("Running point (%s): Lx=%d Ly=%d tcx=%d tcy=%d, estimated setup memory %.2f GB\n",
            sweep_type, Lx, Ly, tcx, tcy, est_gb)

    save_len = max(128, cfg.reps + cfg.warmup_measurements + 4)

    data_step = make_data(Lx=Lx, Ly=Ly, tcx=tcx, tcy=tcy,
                          diag_method=cfg.diag_method, pairing=cfg.pairing,
                          loop_random=cfg.loop_random, saving_interval=save_len)
    step_stats = benchmark_mc_step(data_step;
                                   reps=cfg.reps,
                                   warmup_steps=cfg.warmup_steps,
                                   method=cfg.diag_method,
                                   pairing=cfg.pairing,
                                   loop_random=cfg.loop_random)
    real_gb = Base.summarysize(data_step) / 1024^3
    @printf("Actual data memory usage: %.2f GB\n", real_gb)
    GC.gc()

    meas_stats = if benchmark_measurement
        data_meas = make_data(Lx=Lx, Ly=Ly, tcx=tcx, tcy=tcy,
                              diag_method=cfg.diag_method, pairing=cfg.pairing,
                              loop_random=cfg.loop_random, saving_interval=save_len)
        benchmark_measurement_full(data_meas;
                                   reps=cfg.reps,
                                   warmup_measurements=cfg.warmup_measurements,
                                   method=cfg.diag_method,
                                   pairing=cfg.pairing,
                                   loop_random=cfg.loop_random)
    else
        (median_ns=NaN, mean_ns=NaN, std_ns=NaN, min_ns=NaN, max_ns=NaN)
    end

    return (
        sweep_type=sweep_type,
        Lx=Lx,
        Ly=Ly,
        N=Lx * Ly,
        tcx=tcx,
        tcy=tcy,
        TN=tcx * tcy,
        reps=cfg.reps,
        est_memory_gb=est_gb,
        init_memory_gb=real_gb,
        step_median_ns=step_stats.median_ns,
        step_mean_ns=step_stats.mean_ns,
        step_std_ns=step_stats.std_ns,
        meas_median_ns=meas_stats.median_ns,
        meas_mean_ns=meas_stats.mean_ns,
        meas_std_ns=meas_stats.std_ns,
    )
end

function collect_rows(points, cfg::BenchConfig; sweep_type::String, benchmark_measurement::Bool=true)
    rows = NamedTuple[]
    for (Lx, Ly, tcx, tcy) in points
        row = bench_point(Lx, Ly, tcx, tcy, cfg, sweep_type; benchmark_measurement=benchmark_measurement)
        isnothing(row) && continue
        push!(rows, row)
        GC.gc()
    end
    return rows
end

function print_rows(title::String, rows)
    println("\n" * "="^126)
    println(title)
    println("="^126)
    println("Lx   Ly   N      tcx  tcy  TN     step_median_ms   step_mean_ms   meas_median_ms   meas_mean_ms")
    println("-"^126)
    for r in rows
        meas_median_str = isfinite(r.meas_median_ns) ? @sprintf("%.4f", r.meas_median_ns / 1e6) : "N/A"
        meas_mean_str = isfinite(r.meas_mean_ns) ? @sprintf("%.4f", r.meas_mean_ns / 1e6) : "N/A"
        @printf("%-4d %-4d %-6d %-4d %-4d %-6d %-16.4f %-14.4f %-16s %-14s\n",
            r.Lx, r.Ly, r.N, r.tcx, r.tcy, r.TN,
            r.step_median_ns / 1e6, r.step_mean_ns / 1e6,
            meas_median_str, meas_mean_str)
    end
end

function fit_power_law(x::Vector{Float64}, y::Vector{Float64})
    valid = findall(i -> x[i] > 0.0 && y[i] > 0.0 && isfinite(x[i]) && isfinite(y[i]), eachindex(x))
    if length(valid) < 2
        return (ok=false, a=NaN, p=NaN, r2=NaN)
    end

    xv = x[valid]
    yv = y[valid]
    lx = log.(xv)
    ly = log.(yv)

    mean_lx = mean(lx)
    mean_ly = mean(ly)
    denom = sum((lx .- mean_lx).^2)
    if denom <= 0.0
        return (ok=false, a=NaN, p=NaN, r2=NaN)
    end

    p = sum((lx .- mean_lx) .* (ly .- mean_ly)) / denom
    log_a = mean_ly - p * mean_lx
    yhat = log_a .+ p .* lx

    ss_tot = sum((ly .- mean_ly).^2)
    ss_res = sum((ly .- yhat).^2)
    r2 = ss_tot > 0 ? (1 - ss_res / ss_tot) : 1.0

    return (ok=true, a=exp(log_a), p=p, r2=r2)
end

function make_runtime_panel(rows, xfield::Symbol, yfield::Symbol;
                            xlabel::String, ylabel::String, title::String)
    x = Float64[getproperty(r, xfield) for r in rows]
    y = Float64[getproperty(r, yfield) / 1e6 for r in rows]

    if isempty(x)
        p = plot(title=title, xlabel=xlabel, ylabel=ylabel)
        annotate!(p, 0.5, 0.5, text("No data", 10, :center))
        return p, (ok=false, a=NaN, p=NaN, r2=NaN)
    end

    perm = sortperm(x)
    x = x[perm]
    y = y[perm]

    p = scatter(x, y,
                xscale=:log10,
                yscale=:log10,
                markersize=5,
                markerstrokewidth=0,
                label="measured",
                xlabel=xlabel,
                ylabel=ylabel,
                title=title)

    fit = fit_power_law(x, y)
    if fit.ok
        xfit = exp.(range(log(minimum(x)), log(maximum(x)), length=200))
        yfit = fit.a .* (xfit .^ fit.p)
        fit_label = @sprintf("fit: t = %.3g*x^%.3f (R2=%.3f)", fit.a, fit.p, fit.r2)
        plot!(p, xfit, yfit, lw=2, ls=:dash, color=:red, label=fit_label)
    end

    return p, fit
end

function make_memory_panel(rows;
                           xlabel::String="L (Lx=Ly)",
                           ylabel::String="memory [GB]",
                           title::String="Initialization memory vs L")
    if isempty(rows)
        p = plot(title=title, xlabel=xlabel, ylabel=ylabel)
        annotate!(p, 0.5, 0.5, text("No data", 10, :center))
        return p, (ok=false, a=NaN, p=NaN, r2=NaN), (ok=false, a=NaN, p=NaN, r2=NaN)
    end

    x = Float64[r.Lx for r in rows]
    y_actual = Float64[r.init_memory_gb for r in rows]
    y_est = Float64[r.est_memory_gb for r in rows]

    perm = sortperm(x)
    x = x[perm]
    y_actual = y_actual[perm]
    y_est = y_est[perm]

    p = scatter(x, y_actual,
                xscale=:log10,
                yscale=:log10,
                markersize=5,
                markerstrokewidth=0,
                label="actual",
                xlabel=xlabel,
                ylabel=ylabel,
                title=title)
    scatter!(p, x, y_est,
             markersize=5,
             markerstrokewidth=0,
             color=:gray40,
             label="estimate")

    fit_actual = fit_power_law(x, y_actual)
    if fit_actual.ok
        xfit = exp.(range(log(minimum(x)), log(maximum(x)), length=200))
        yfit = fit_actual.a .* (xfit .^ fit_actual.p)
        fit_label = @sprintf("actual fit: m = %.3g*L^%.3f (R2=%.3f)", fit_actual.a, fit_actual.p, fit_actual.r2)
        plot!(p, xfit, yfit, lw=2, ls=:dash, color=:red, label=fit_label)
    end

    fit_est = fit_power_law(x, y_est)
    if fit_est.ok
        xfit = exp.(range(log(minimum(x)), log(maximum(x)), length=200))
        yfit = fit_est.a .* (xfit .^ fit_est.p)
        fit_label = @sprintf("estimate fit: m = %.3g*L^%.3f (R2=%.3f)", fit_est.a, fit_est.p, fit_est.r2)
        plot!(p, xfit, yfit, lw=2, ls=:dot, color=:black, label=fit_label)
    end

    return p, fit_actual, fit_est
end

function make_scaling_plot(lattice_rows, cluster_rows, cfg::BenchConfig; save_path::String)
    p1, fit_step_l = make_runtime_panel(lattice_rows, :Lx, :step_median_ns;
                                        xlabel="L (Lx=Ly)",
                                        ylabel="runtime [ms]",
                                        title="MC_stepV2! vs L (fixed TCA=$(cfg.fixed_tca))")

    p2, fit_step_t = make_runtime_panel(cluster_rows, :tcx, :step_median_ns;
                                        xlabel="TCA size (tcx=tcy)",
                                        ylabel="runtime [ms]",
                                        title="MC_stepV2! vs TCA (fixed L=$(cfg.fixed_l))")

    p3, fit_meas_l = make_runtime_panel(lattice_rows, :Lx, :meas_median_ns;
                                        xlabel="L (Lx=Ly)",
                                        ylabel="runtime [ms]",
                                        title="make_measurement_K! vs L (fixed TCA=$(cfg.fixed_tca))")

    p4, fit_mem_l_actual, fit_mem_l_est = make_memory_panel(lattice_rows;
                                                             xlabel="L (Lx=Ly)",
                                                             ylabel="memory [GB]",
                                                             title="Initialization memory vs L (fixed TCA=$(cfg.fixed_tca))")
    fit_meas_t = (ok=false, a=NaN, p=NaN, r2=NaN)

    fig = plot(p1, p2, p3, p4,
               layout=(2, 2),
               size=(1500, 1000),
               plot_title="Runtime scaling (method=$(cfg.diag_method), pairing=$(cfg.pairing), reps=$(cfg.reps))")

    mkpath(dirname(save_path))
    savefig(fig, save_path)

    fits = Dict(
        "mc_step_vs_L" => fit_step_l,
        "mc_step_vs_TCA" => fit_step_t,
        "measurement_vs_L" => fit_meas_l,
        "measurement_vs_TCA" => fit_meas_t,
        "init_memory_vs_L_actual" => fit_mem_l_actual,
        "init_memory_vs_L_estimate" => fit_mem_l_est,
    )

    return fig, fits
end

function print_fit_summary(fits::Dict{String, <:NamedTuple})
    println("\n" * "="^90)
    println("Power-law fit summary (t = a*x^p)")
    println("="^90)
    println("name                          a              p              R2")
    println("-"^90)
    for key in sort(collect(keys(fits)))
        fit = fits[key]
        if fit.ok
            @printf("%-28s %-14.6g %-14.6f %-14.6f\n", key, fit.a, fit.p, fit.r2)
        else
            @printf("%-28s %-14s %-14s %-14s\n", key, "NaN", "NaN", "NaN")
        end
    end
end

function write_rows_csv(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "sweep_type,Lx,Ly,N,tcx,tcy,TN,reps,est_memory_gb,init_memory_gb,step_median_ms,step_mean_ms,step_std_ms,measurement_median_ms,measurement_mean_ms,measurement_std_ms")
        for r in rows
            @printf(io, "%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.sweep_type,
                    r.Lx, r.Ly, r.N,
                    r.tcx, r.tcy, r.TN,
                    r.reps,
                    r.est_memory_gb,
                    r.init_memory_gb,
                    r.step_median_ns / 1e6,
                    r.step_mean_ns / 1e6,
                    r.step_std_ns / 1e6,
                    r.meas_median_ns / 1e6,
                    r.meas_mean_ns / 1e6,
                    r.meas_std_ns / 1e6)
        end
    end
end

function write_fit_csv(path::String, fits::Dict{String, <:NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "name,a,p,r2")
        for key in sort(collect(keys(fits)))
            fit = fits[key]
            if fit.ok
                @printf(io, "%s,%.10g,%.10g,%.10g\n", key, fit.a, fit.p, fit.r2)
            else
                println(io, "$(key),NaN,NaN,NaN")
            end
        end
    end
end

function build_points(cfg::BenchConfig)
    lattice_points = Tuple{Int, Int, Int, Int}[]
    for L in sort(unique(cfg.L_list))
        if cfg.fixed_tca > L
            @warn "Skipping L value because fixed TCA is larger than L" L fixed_tca=cfg.fixed_tca
            continue
        end
        push!(lattice_points, (L, L, cfg.fixed_tca, cfg.fixed_tca))
    end

    cluster_points = Tuple{Int, Int, Int, Int}[]
    for tca in sort(unique(cfg.tca_list))
        if tca > cfg.fixed_l
            @warn "Skipping TCA value because it is larger than fixed L" tca fixed_l=cfg.fixed_l
            continue
        end
        push!(cluster_points, (cfg.fixed_l, cfg.fixed_l, tca, tca))
    end

    isempty(lattice_points) && error("No valid points in L sweep")
    isempty(cluster_points) && error("No valid points in TCA sweep")

    return lattice_points, cluster_points
end

function main()
    cfg = parse_cli_args()
    lattice_points, cluster_points = build_points(cfg)

    println("Running runtime benchmark for MC_stepV2! and make_measurement_K!")
    println("method=$(cfg.diag_method), pairing=$(cfg.pairing), reps=$(cfg.reps)")
    println("L sweep points: $(lattice_points)")
    println("TCA sweep points: $(cluster_points)")

    lattice_rows = collect_rows(lattice_points, cfg; sweep_type="L", benchmark_measurement=true)
    cluster_rows = collect_rows(cluster_points, cfg; sweep_type="TCA", benchmark_measurement=false)

    print_rows("Scaling A: fixed TCA, increasing lattice", lattice_rows)
    print_rows("Scaling B: fixed lattice, increasing TCA", cluster_rows)

    all_rows = vcat(lattice_rows, cluster_rows)
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    out_dir = cfg.out_dir
    rows_csv = joinpath(out_dir, "mc_step_measurement_scaling_rows_$(stamp).csv")
    fits_csv = joinpath(out_dir, "mc_step_measurement_scaling_fits_$(stamp).csv")
    fig_png = joinpath(out_dir, "mc_step_measurement_scaling_plot_$(stamp).png")

    write_rows_csv(rows_csv, all_rows)
    _, fits = make_scaling_plot(lattice_rows, cluster_rows, cfg; save_path=fig_png)
    write_fit_csv(fits_csv, fits)
    print_fit_summary(fits)

    println("\nSaved outputs:")
    println("  Rows CSV: $(rows_csv)")
    println("  Fits CSV: $(fits_csv)")
    println("  Plot PNG: $(fig_png)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
