# benchmark_MCMF_action_TCluster_K0.jl — benchmark of the K0-TCA local action
# (MCMF_action_TCluster_K0): time, allocations and bytes of one call, then its
# scaling with the lattice size at a fixed cluster, and with the cluster size at a
# fixed lattice; optional profile. Prints tables only.
#   julia tests/benchmark_MCMF_action_TCluster_K0.jl
using Statistics
using Printf
using BenchmarkTools
using Profile

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

function make_data(; Lx::Int=16, Ly::Int=16, tcx::Int=4, tcy::Int=4, pairing::String="s-wave")
    sim = SimData(
        Lx=Lx,
        Ly=Ly,
        U=1.0,
        t=1.0,
        V=0.0,
        seed=12345,
        mu=0.0,
        beta=10.0,
        num_measure=10,
        num_thermal=10,
        measuring_interval=1,
        saving_interval=10,
        guess="uniform",
        U_disorder="uniform",
        diag_method="K0-TCA",
        pairing_symmetry=pairing,
        directory=".",
        num_cores=1,
        num_threads=1,
        TCluster_size_x=tcx,
        TCluster_size_y=tcy,
        loop_random=false
    )
    return Lattice_Data(sim)
end

function benchmark_single(data::LatticeData, site::Int; samples::Int=200)
    # Warmup to exclude compilation from timing.
    MCMF_action_TCluster_K0(data, site)

    single_alloc_bytes = @allocated MCMF_action_TCluster_K0(data, site)
    trial = @benchmark MCMF_action_TCluster_K0($data, $site) samples=samples evals=1
    m = median(trial)

    return (
        time_ns=m.time,
        allocs=m.allocs,
        bytes=m.memory,
        single_alloc_bytes=single_alloc_bytes,
    )
end

function bench_point(Lx::Int, Ly::Int, tcx::Int, tcy::Int; samples::Int=100)
    data = make_data(Lx=Lx, Ly=Ly, tcx=tcx, tcy=tcy)
    site = 1
    stats = benchmark_single(data, site; samples=samples)
    return (
        Lx=Lx,
        Ly=Ly,
        N=Lx * Ly,
        tcx=tcx,
        tcy=tcy,
        TN=tcx * tcy,
        time_ns=stats.time_ns,
        allocs=stats.allocs,
        bytes=stats.bytes,
    )
end

function estimate_data_memory_gb(Lx::Int, Ly::Int; N_SC::Int=1)
    # Rough estimate of TemporaryData dominant dense arrays (ComplexF64 matrices).
    N = Lx * Ly
    M = 2 * N
    G = max(max(Lx, Ly), N_SC)
    bytes_per_big_matrix = M * M * sizeof(ComplexF64)
    approx_num_big_matrices = 6 + 6 * G
    return (approx_num_big_matrices * bytes_per_big_matrix) / 1024^3
end

function collect_rows(points; samples::Int=100, mem_cap_gb::Float64=2.0, title::String="")
    rows = NamedTuple[]
    for (Lx, Ly, tcx, tcy) in points
        est_gb = estimate_data_memory_gb(Lx, Ly)
        if est_gb > mem_cap_gb
            @printf("Skipping point (Lx=%d, Ly=%d, tcx=%d, tcy=%d): estimated setup memory %.2f GB > cap %.2f GB\n",
                    Lx, Ly, tcx, tcy, est_gb, mem_cap_gb)
            continue
        end
        @printf("Running %s point (Lx=%d, Ly=%d, tcx=%d, tcy=%d), estimated setup memory %.2f GB\n",
                title, Lx, Ly, tcx, tcy, est_gb)
        push!(rows, bench_point(Lx, Ly, tcx, tcy; samples=samples))
        GC.gc()
    end
    return rows
end

function print_rows(title::String, rows)
    println("\n" * "="^90)
    println(title)
    println("="^90)
    println("Lx   Ly   N      tcx  tcy  TN     median_time_ms   allocs   bytes")
    println("-"^90)
    for r in rows
        t_ms = r.time_ns / 1e6
        @printf("%-4d %-4d %-6d %-4d %-4d %-6d %-16.4f %-8d %-8d\n",
                r.Lx, r.Ly, r.N, r.tcx, r.tcy, r.TN, t_ms, r.allocs, r.bytes)
    end
end

function run_profview(data::LatticeData, site::Int; iters::Int=500)
    println("\nOpening @profview with $(iters) calls to MCMF_action_TCluster_K0...")
    if !isdefined(Main, :ProfileView)
        try
            @eval using ProfileView
        catch
            error("ProfileView is not installed in this environment. Run: import Pkg; Pkg.add(\"ProfileView\")")
        end
    end
    Profile.clear()
    @profile begin
        for _ in 1:iters
            MCMF_action_TCluster_K0(data, site)
        end
    end
    ProfileView.view()
    return nothing
end

function main(; run_profile::Bool=false)

    samples = 200
    mem_cap_gb = 2.0

    println("Running benchmark for MCMF_action_TCluster_K0")
    println("TCA mode: eigenvalues-only diagonalization (jobz='N')")
    println("Full-system diagonalization paths remain unchanged (still use eigenvectors)")
    if "--profview" in ARGS
        run_profile = true
    else
        println("Tip: add --profview to also open @profview flame graph")
    end


    base_data = make_data(Lx=16, Ly=16, tcx=4, tcy=4)
    base_site = 1
    base = benchmark_single(base_data, base_site; samples=samples)

    println("\nSingle-point benchmark (Lx=16, Ly=16, tcx=4, tcy=4):")
    @printf("median time      : %.4f ms\n", base.time_ns / 1e6)
    @printf("median allocs    : %d\n", base.allocs)
    @printf("median bytes     : %d\n", base.bytes)
    @printf("single-call bytes: %d\n", base.single_alloc_bytes)

    # Safe defaults to avoid out-of-memory on dense TemporaryData allocations.
    lattice_points = [(8, 8, 8, 8), (10, 10, 8, 8), (12, 12, 8, 8), (14, 14, 8, 8), (16, 16, 8, 8), (18, 18, 8, 8), (20, 20, 8, 8)]
    cluster_points = [(18, 18, 4, 4), (18, 18, 6, 6), (18, 18, 8, 8), (18, 18, 10, 10), (18, 18, 12, 12), (18, 18, 14, 14), (18, 18, 16, 16)]

    lattice_rows = collect_rows(lattice_points; samples=100, mem_cap_gb=mem_cap_gb, title="lattice-scaling")
    cluster_rows = collect_rows(cluster_points; samples=100, mem_cap_gb=mem_cap_gb, title="cluster-scaling")

    print_rows("Scaling A: fixed TCluster size, increasing lattice", lattice_rows)
    print_rows("Scaling B: fixed lattice size, increasing TCluster size", cluster_rows)

    if run_profile
        run_profview(base_data, base_site; iters=500)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
