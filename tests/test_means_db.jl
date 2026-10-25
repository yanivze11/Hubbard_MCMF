# ============================================================================
# tests/test_means_db.jl — the behaviours means_db.jl is supposed to guarantee.
#
#   julia -O1 tests/test_means_db.jl
#
# Focus is on the two things that are easy to get silently wrong:
#   1. identity stability as parameters get ADDED to the code over time
#   2. merging seeds when the same config is run on more than one cluster
# ============================================================================
using Test, JLD2, Printf, Dates

# Stand-in cluster mounts on two drive letters that are normally unused; no
# built-in mount letters exist, so the mount guard knows only these.
ENV["LANDAU_MOUNT"] = "Q:"; ENV["CURIE_MOUNT"] = "R:"
delete!(ENV, "BACKUP_DRIVE"); delete!(ENV, "MEANS_DB_ROOTS")
include(joinpath(@__DIR__, "..", "means_db.jl"))

const ROOT = joinpath(tempdir(), "means_db_test_" * string(hash(time_ns()) % 100000))
rm(ROOT; recursive=true, force=true)

obs(v; err=0.01) = Dict{String,Any}("value" => v, "error" => err,
                                    "correlation_time" => 1.0,
                                    "num_uncorrelated_measurements" => 100,
                                    "num_measurements" => 20000)

"""A minimal but realistic entry."""
function make_entry(; U=3.0, beta=1.5, Lx=10, Ly=10, TCA=4, V=0.0, B=0.0,
                      n=0.8751, seeds=[111], extra=Dict{String,Any}())
    e = Dict{String,Any}(
        "U" => U, "wanted_n" => 0.875, "Lx" => Lx, "Ly" => Ly, "beta" => beta,
        "diag_method" => "K0-TCA", "TClustersize" => TCA, "V" => V, "B" => B,
        "t" => 1.0, "SC_L" => 1, "impurity" => 0.0, "U_disorder" => "uniform",
        "pairing_symmetry" => "s-wave", "boundary_condition" => "PBC",
        "gauge_choice" => "Mixed", "guess" => "uniform", "max_iter" => 100,
        "n_tol" => 0.001, "num_thermal" => 5000, "num_measure" => 20000,
        "sim_num" => 1, "seeds" => seeds, "start_time" => Dates.now(),
        "convergence" => Dict{String,Any}("best_mu" => -1.23, "converged" => true),
        "means" => Dict{String,Any}("n" => obs(n), "stiffness" => obs(0.05),
                                    "Ps" => obs(12.3), "LDOS" => obs(rand(50))),
    )
    merge!(e, extra)
    return e
end

seed_block(seed, nmeas, val) = Dict{String,Any}(
    "info"  => Dict{String,Any}("seed" => seed, "num_current_measurments" => nmeas,
                                "mu" => -1.2, "status" => "finished",
                                "convergence" => Dict{String,Any}("target_n"=>0.875,
                                    "best_n"=>0.8751, "best_mu"=>-1.2,
                                    "converged"=>true, "n_tol"=>0.001)),
    "means" => Dict{String,Any}("n" => obs(0.875), "stiffness" => obs(val)),
)

@testset "means_db" begin

@testset "identity: stable as parameters are added" begin
    e = make_entry()
    h0 = params_hash(e)

    # 1. an UNREGISTERED parameter must not affect identity
    e2 = make_entry(extra=Dict{String,Any}("some_new_knob" => 42))
    @test params_hash(e2) == h0

    # 2. a REGISTERED parameter sitting at its default must not affect identity —
    #    this is what lets you add a parameter without splitting existing data
    e3 = make_entry(); delete!(e3, "V")          # absent -> default 0.0
    @test params_hash(e3) == h0
    e4 = make_entry(V=0.0)                        # present at default
    @test params_hash(e4) == h0

    # 3. a non-default value MUST change identity
    @test params_hash(make_entry(V=1.0)) != h0
    @test params_hash(make_entry(U=4.0))  != h0
    @test params_hash(make_entry(beta=2.0)) != h0

    # 4. bookkeeping fields must never affect identity
    e5 = make_entry(seeds=[999, 888]); e5["sim_num"] = 77
    e5["start_time"] = Dates.now() + Dates.Hour(5)
    @test params_hash(e5) == h0

    # 5. float noise must not split a config
    @test params_hash(make_entry(U=3.0 + 1e-15)) == h0

    # 6. a missing REQUIRED parameter is an error, not a silent default
    e6 = make_entry(); delete!(e6, "beta")
    @test_throws ErrorException identity_of(e6)
end

@testset "round trip preserves the entry" begin
    e = make_entry()
    p = save_entry(e; root=ROOT, source="unit-test")
    @test isfile(p)
    back = load_entry(p)
    for k in ("U", "beta", "Lx", "diag_method", "TClustersize", "sim_num")
        @test back[k] == e[k]
    end
    @test back["convergence"]["best_mu"] == -1.23         # meta survives
    @test back["means"]["n"]["value"] == e["means"]["n"]["value"]
    @test sort(collect(keys(back["means"]))) == sort(collect(keys(e["means"])))
end

@testset "partial observable reads" begin
    e = make_entry(beta=9.0)
    p = save_entry(e; root=ROOT)
    @test sort(collect(keys(load_entry(p; obs=[:stiffness])["means"]))) == ["stiffness"]
    @test "LDOS" ∉ keys(load_entry(p; obs=:light)["means"])      # heavy excluded
    @test "LDOS" ∈ keys(load_entry(p; obs=:all)["means"])
    @test load_entry(p; obs=[:stiffness])["means"]["stiffness"]["value"] == 0.05
end

@testset "per-seed data round trip" begin
    e = make_entry(beta=3.0)
    sd = Dict("111" => seed_block(111, 20000, 0.051),
              "222" => seed_block(222, 20000, 0.049))
    p = save_entry(e; root=ROOT, seed_data=sd)
    got = read_seeds(p)
    @test sort(collect(keys(got))) == ["111", "222"]
    @test got["111"]["info"]["num_current_measurments"] == 20000
    @test got["111"]["means"]["stiffness"]["value"] == 0.051
    # the fields average_over_seeds/get_obs_dict need must be present
    @test haskey(got["222"]["info"], "mu")
    @test haskey(got["222"]["info"], "convergence")

    # recomputation cache: this is what replaces means_cache.jld2
    @test seed_is_current(p, 111, 20000)
    @test seed_is_current(p, 111, 15000)
    @test !seed_is_current(p, 111, 40000)     # more measurements now -> redo
    @test !seed_is_current(p, 999, 1)         # unknown seed -> redo
end

@testset "merge: same config from two clusters pools seeds" begin
    e = make_entry(beta=4.0)
    p1 = save_entry(e; root=ROOT, seed_data=Dict("111" => seed_block(111, 20000, 0.05)),
                    source="LANDAU")
    # second cluster, same physics, different seed
    p2 = save_entry(make_entry(beta=4.0); root=ROOT,
                    seed_data=Dict("222" => seed_block(222, 20000, 0.06)),
                    source="CURIE")
    @test p1 == p2                                    # same identity -> same file
    got = read_seeds(p2)
    @test sort(collect(keys(got))) == ["111", "222"]  # pooled, not overwritten

    # with an averaging function wired up, the stored mean is recomputed
    MEANS_DB_AVERAGE_FN[] = seeds -> Dict{String,Any}(
        "n" => obs(0.875),
        "stiffness" => obs(sum(d["means"]["stiffness"]["value"] for d in values(seeds)) / length(seeds)))
    p3 = save_entry(make_entry(beta=4.0); root=ROOT,
                    seed_data=Dict("333" => seed_block(333, 20000, 0.07)))
    @test length(read_seeds(p3)) == 3
    @test load_entry(p3; obs=[:stiffness])["means"]["stiffness"]["value"] ≈ 0.06 atol=1e-9
    MEANS_DB_AVERAGE_FN[] = nothing
end

@testset "collision is loud, not silent" begin
    e = make_entry(beta=7.0)
    save_entry(e; root=ROOT)
    # same identity hash, but a parameter that IS recorded differs => must refuse
    bad = make_entry(beta=7.0); bad["cores"] = 4      # in meta, fine
    @test save_entry(bad; root=ROOT) isa String       # no conflict

    worse = make_entry(beta=7.0)
    worse["some_new_knob"] = 42                        # unregistered -> same hash
    save_entry(worse; root=ROOT)
    conflicting = make_entry(beta=7.0)
    conflicting["some_new_knob"] = 99                  # same hash, different value
    @test_throws ErrorException save_entry(conflicting; root=ROOT)
end

@testset "index and query" begin
    for (U, b) in ((5.0, 1.0), (5.0, 2.0), (5.0, 3.0), (6.0, 1.0))
        save_entry(make_entry(U=U, beta=b); root=ROOT)
    end
    rebuild_index(; root=ROOT, verbose=false)
    idx = load_index(; root=ROOT)
    @test idx["n"] == length(scan_files(ROOT))
    @test "V" in idx["columns"]              # queryable for potential disorder
    @test "U" in idx["columns"]

    @test length(find_entries(; root=ROOT, U=5.0)) == 3
    @test length(find_entries(; root=ROOT, U=5.0, beta=2.0)) == 1
    @test length(find_entries(; root=ROOT, U=[5.0, 6.0])) == 4
    @test isempty(find_entries(; root=ROOT, U=999.0))

    si = load_means(; root=ROOT, U=5.0, obs=[:stiffness], verbose=false)
    @test length(si) == 3
    @test all(e -> collect(keys(e["means"])) == ["stiffness"], values(si))
    @test all(e -> haskey(e, "beta"), values(si))    # sim_info shape preserved
end

@testset "conditional identity: irrelevant params cannot split a config" begin
    # diag_method="K0" does no cluster approximation, so a recorded (ignored)
    # TClustersize must not create a second entry.
    k0a = make_entry(TCA=4);  k0a["diag_method"] = "K0"
    k0b = make_entry(TCA=12); k0b["diag_method"] = "K0"
    @test params_hash(k0a) == params_hash(k0b)

    # ...but under TCA it is meaningful and MUST separate them
    tcaa = make_entry(TCA=4);  tcaa["diag_method"] = "K0-TCA"
    tcab = make_entry(TCA=12); tcab["diag_method"] = "K0-TCA"
    @test params_hash(tcaa) != params_hash(tcab)

    # a disorder realization index means nothing with no disorder present
    clean1 = make_entry(); clean1["disorder_seed"] = 1
    clean2 = make_entry(); clean2["disorder_seed"] = 7
    @test params_hash(clean1) == params_hash(clean2)

    # with V != 0 it is the sample label, so it MUST separate realizations
    d1 = make_entry(V=1.0); d1["disorder_seed"] = 1
    d2 = make_entry(V=1.0); d2["disorder_seed"] = 7
    @test params_hash(d1) != params_hash(d2)
    @test params_hash(d1) != params_hash(make_entry(V=1.0))   # vs default seed 0
end

@testset "cluster assignment is pure and stable" begin
    e = make_entry(U=3.0, V=1.0, Lx=4, Ly=4, beta=20.0)
    h = params_hash(e)
    @test assign_cluster(e, 1) == assign_cluster(h, 1)        # entry or hash
    @test assign_cluster(e, 1) == assign_cluster(e, 1)        # deterministic
    @test all(assign_cluster(e, s) in DB_CLUSTERS for s in 1:50)
    # both clusters get used, so seeds of one config really do split
    @test length(unique(assign_cluster(e, s) for s in 1:50)) == 2
end

@testset "first registration into an empty database" begin
    # Regression: the submitter queries for duplicates BEFORE anything exists,
    # which builds an index with zero rows. If that index has no "relpath"
    # column, the very next register_simulation dies with KeyError("relpath").
    fresh = joinpath(ROOT, "fresh")
    mkpath(fresh)

    @test isempty(find_entries(; root=fresh, U=3.0))       # creates an empty index
    idx0 = load_index(; root=fresh)
    @test idx0["n"] == 0
    @test "relpath" in idx0["columns"]                     # must exist even when empty

    e = make_entry(U=3.0, Lx=4, Ly=4, V=1.0, beta=20.0)
    e["disorder_seed"] = 2
    p = register_simulation(e; root=fresh, cluster="landau", seeds=[1])
    @test isfile(p)
    @test length(find_entries(; root=fresh, status=nothing)) == 1

    # and a second one appends cleanly onto the now non-empty index
    e2 = make_entry(U=3.0, Lx=4, Ly=4, V=1.0, beta=10.0)
    e2["disorder_seed"] = 2
    register_simulation(e2; root=fresh, cluster="landau", seeds=[1])
    @test length(find_entries(; root=fresh, status=nothing)) == 2
    @test load_index(; root=fresh)["n"] == length(scan_files(fresh))
end

@testset "no index rebuild over a cluster mount" begin
    # A missing index on a cluster mount must error, not walk the cluster's storage over SFTP.
    @test _on_cluster_mount(raw"Q:\data\MC_Hubbard_V7\means_db")
    @test _on_cluster_mount("r:/data/MC_Hubbard_V7/means_db")
    @test !_on_cluster_mount(raw"Y:\data\MC_Hubbard_V7\means_db")   # not a configured mount
    @test _on_cluster_mount(raw"\\RaiDrive-someuser\SFTP\data\MC_Hubbard_V7\means_db")
    @test !_on_cluster_mount(raw"C:\Users\x\means_db")
    @test !_on_cluster_mount("/home/data/someuser/MC_Hubbard_V7/means_db")
    @test !_on_cluster_mount(ROOT)

    gone = raw"R:\__means_db_test_no_such_root__"         # one stat if R: is mounted
    @test_throws MountIndexMissing load_index(; root=gone)
    # find_all skips a root that is not a directory; for one that is, the error must
    # propagate rather than become a "skipping root" warning.
    @test occursin("NOT rebuilt", sprint(showerror, MountIndexMissing(gone)))
end

@testset "register at submission, fill in later" begin
    sub = joinpath(ROOT, "sub")
    e = make_entry(U=3.0, Lx=4, Ly=4, V=1.0, beta=20.0)
    e["disorder_seed"] = 1
    p = register_simulation(e; root=sub, cluster="landau", seeds=[101, 102])
    @test isfile(p)

    # visible in the index BEFORE anything has run
    idx = load_index(; root=sub)
    @test idx["n"] == 1
    @test find_entries(; root=sub, status=DB_STATUS_SUBMITTED) == [relpath(p, sub)]
    @test isempty(find_entries(; root=sub, status=DB_STATUS_DONE))

    # a stub has no means, and load_means must not return it by default
    @test isempty(load_means(; root=sub, U=3.0, verbose=false))
    @test length(load_means(; root=sub, U=3.0, status=nothing, verbose=false)) == 1

    # registering again is harmless (idempotent submission)
    @test register_simulation(e; root=sub, seeds=[103]) == p
    @test load_index(; root=sub)["n"] == 1

    # now the means arrive at the SAME path and the status flips
    @test save_entry(e; root=sub, seed_data=Dict("101" => seed_block(101, 20000, 0.05))) == p
    append_index!(sub, [p])
    @test find_entries(; root=sub, status=DB_STATUS_DONE) == [relpath(p, sub)]
    @test length(load_means(; root=sub, U=3.0, verbose=false)) == 1
    @test length(read_seeds(p)) == 1
end

@testset "incremental index == full rebuild" begin
    inc = joinpath(ROOT, "inc")
    ps = String[]
    for b in (1.0, 2.0, 3.0)
        push!(ps, save_entry(make_entry(U=7.0, beta=b); root=inc, source="x"))
    end
    full = rebuild_index(; root=inc, verbose=false)
    # rebuild from scratch, then add one more incrementally
    p4 = save_entry(make_entry(U=7.0, beta=4.0); root=inc)
    appended = append_index!(inc, [p4])
    rebuilt = rebuild_index(; root=inc, verbose=false)
    @test appended["n"] == rebuilt["n"] == 4
    @test Set(String.(appended["relpath"])) == Set(String.(rebuilt["relpath"]))
    for c in ("U", "beta", "status")
        @test Set(appended[c]) == Set(rebuilt[c])
    end
end

@testset "federation and the query API" begin
    # Simulate the real layout: two clusters, each owning some disorder samples
    # of the SAME physical points.
    landau = joinpath(ROOT, "fed", "landau")
    curie  = joinpath(ROOT, "fed", "curie")
    ROOTS  = [landau, curie]

    # stiffness(beta, dseed) -- deterministic so the averages are checkable
    stiff(beta, d) = 0.10 + 0.01 * beta + 0.001 * d

    for beta in (20.0, 10.0, 5.0), d in 1:4
        e = make_entry(U=3.0, Lx=4, Ly=4, V=1.0, beta=beta)
        e["disorder_seed"] = d
        e["diag_method"]   = "K0"
        e["means"] = Dict{String,Any}("n" => obs(0.875),
                                      "stiffness" => obs(stiff(beta, d); err=0.002))
        # odd samples to curie, even to landau — as assign_cluster would
        root, who = isodd(d) ? (curie, "curie") : (landau, "landau")
        save_entry(e; root=root, source="test", cluster=who,
                   seed_data=Dict("1" => seed_block(1, 20000, stiff(beta, d))))
    end
    for r in ROOTS; rebuild_index(; root=r, verbose=false); end

    @testset "find_all spans both roots" begin
        hits = find_all(; roots=ROOTS, U=3.0, V=1.0)
        @test length(hits) == 12
        @test length(unique(h -> h["root"], hits)) == 2
        @test length(find_all(; roots=ROOTS, U=3.0, V=1.0, beta=20.0)) == 4
        # a missing root is skipped, not an error
        @test length(find_all(; roots=[landau, joinpath(ROOT, "nope")], U=3.0)) == 6
    end

    @testset "disorder average over samples" begin
        r = get_observable(; roots=ROOTS, U=3.0, Lx=4, V=1.0, beta=20.0, obs=:stiffness)
        @test r !== nothing
        @test r.n_samples == 4
        expected = sum(stiff(20.0, d) for d in 1:4) / 4
        @test r.value ≈ expected atol=1e-12
        @test r.std > 0                       # samples genuinely differ
        @test r.error ≈ r.std / sqrt(4) atol=1e-12
        @test sort(r.disorder_seeds) == [1, 2, 3, 4]
        @test sort(unique(r.clusters)) == ["curie", "landau"]
    end

    @testset "seeds=true gives the per-sample breakdown" begin
        r = get_observable(; roots=ROOTS, U=3.0, Lx=4, V=1.0, beta=20.0,
                             obs=:stiffness, seeds=true)
        @test r.seeds == [1, 2, 3, 4]
        @test r.values ≈ [stiff(20.0, d) for d in 1:4]
        @test length(r.errors) == 4
        @test length(r.clusters) == 4
        @test all(m -> m == ["1"], r.mc_seeds)     # one MC seed per sample
    end

    @testset "pinning one disorder sample" begin
        r = get_observable(; roots=ROOTS, U=3.0, Lx=4, V=1.0, beta=20.0,
                             disorder_seed=3, obs=:stiffness)
        @test r.n_samples == 1
        @test r.value ≈ stiff(20.0, 3)
        @test r.clusters == ["curie"]              # odd seeds went to curie
    end

    @testset "reading one MC seed" begin
        r = get_observable(; roots=ROOTS, U=3.0, Lx=4, V=1.0, beta=10.0,
                             disorder_seed=2, obs=:stiffness, mc_seed=1)
        @test r !== nothing
        @test r.value ≈ stiff(10.0, 2)
    end

    @testset "curve vs beta" begin
        c = get_curve(; roots=ROOTS, U=3.0, Lx=4, V=1.0, obs=:stiffness, x=:beta)
        @test c.x == [5.0, 10.0, 20.0]             # sorted
        @test all(c.n_samples .== 4)               # 4 disorder samples per point
        @test c.value ≈ [sum(stiff(b, d) for d in 1:4)/4 for b in (5.0, 10.0, 20.0)]
        @test all(c.error .> 0)
    end

    @testset "unpinned selectors are surfaced, not hidden" begin
        # Same U/L/V but a different filling: averaging these together is almost
        # certainly a mistake, so it must be visible in the result.
        e = make_entry(U=3.0, Lx=4, Ly=4, V=1.0, beta=20.0)
        e["disorder_seed"] = 1
        e["wanted_n"]      = 0.5          # <- the odd one out
        e["diag_method"]   = "K0"
        e["means"] = Dict{String,Any}("n" => obs(0.5),
                                      "stiffness" => obs(0.99; err=0.002))
        save_entry(e; root=landau, cluster="landau", source="test")
        rebuild_index(; root=landau, verbose=false)

        r = get_observable(; roots=ROOTS, U=3.0, Lx=4, V=1.0, beta=20.0, obs=:stiffness)
        @test "wanted_n" in r.varied         # the mistake is reported
        @test r.n_samples == 5

        # pinning it removes the ambiguity
        r2 = get_observable(; roots=ROOTS, U=3.0, Lx=4, V=1.0, beta=20.0,
                              wanted_n=0.875, obs=:stiffness)
        @test isempty(r2.varied)
        @test r2.n_samples == 4
    end

    @testset "misses return nothing, not an error" begin
        @test get_observable(; roots=ROOTS, U=999.0, obs=:stiffness) === nothing
        @test get_observable(; roots=ROOTS, U=3.0, beta=20.0, obs=:no_such_obs) === nothing
    end

    @testset "submitted stubs are excluded from queries" begin
        e = make_entry(U=3.0, Lx=4, Ly=4, V=1.0, beta=20.0)
        e["disorder_seed"] = 9
        register_simulation(e; root=landau, cluster="landau", seeds=[1])
        # still 4 samples: the unrun one has no means and must not join the average
        # (wanted_n pinned so the n=0.5 entry from the previous testset is excluded)
        r = get_observable(; roots=ROOTS, U=3.0, Lx=4, V=1.0, beta=20.0,
                             wanted_n=0.875, obs=:stiffness)
        @test r.n_samples == 4
        @test 9 ∉ r.disorder_seeds
        @test length(find_all(; roots=ROOTS, U=3.0, V=1.0, beta=20.0,
                                wanted_n=0.875, status=nothing)) == 5
    end
end

@testset "full-cluster K0-TCA is treated as K0 in get_curve" begin
    # A K0-TCA run whose cluster spans the whole lattice does NO cluster
    # approximation — it is a full K0 diagonalization. get_curve must merge it
    # with a genuine K0 run of the same physics rather than split off a spurious
    # method/cluster variant; a partial-cluster K0-TCA stays a real variant.
    r1 = joinpath(ROOT, "tca", "a"); r2 = joinpath(ROOT, "tca", "b")
    RS = [r1, r2]
    L  = 4
    for beta in (5.0, 10.0)
        k0 = make_entry(U=7.0, Lx=L, Ly=L, V=0.0, beta=beta, TCA=2)  # K0: cluster meaningless
        k0["diag_method"] = "K0"
        k0["means"] = Dict{String,Any}("n" => obs(0.875), "stiffness" => obs(0.3; err=0.001))
        save_entry(k0; root=r1, cluster="a", source="test",
                   seed_data=Dict("1" => seed_block(1, 20000, 0.3)))
        full = make_entry(U=7.0, Lx=L, Ly=L, V=0.0, beta=beta, TCA=L)  # full-cluster TCA
        full["diag_method"] = "K0-TCA"
        full["means"] = Dict{String,Any}("n" => obs(0.875), "stiffness" => obs(0.3; err=0.001))
        save_entry(full; root=r2, cluster="b", source="test",
                   seed_data=Dict("1" => seed_block(1, 20000, 0.3)))
    end
    for r in RS; rebuild_index(; root=r, verbose=false); end

    c = get_curve(; roots=RS, U=7.0, Lx=L, V=0.0, obs=:stiffness, x=:beta)
    @test c.x == [5.0, 10.0]
    @test all(c.n_samples .== 2)        # K0 and full-cluster K0-TCA merged into one point
    @test isempty(c.variants)           # not split on diag_method / TClustersize
    @test c.value ≈ [0.3, 0.3] atol=1e-12

    # a partial-cluster K0-TCA (TClustersize < L) is genuinely different physics
    part = make_entry(U=7.0, Lx=L, Ly=L, V=0.0, beta=5.0, TCA=2)
    part["diag_method"] = "K0-TCA"
    part["means"] = Dict{String,Any}("n" => obs(0.875), "stiffness" => obs(0.9; err=0.001))
    save_entry(part; root=r1, cluster="a", source="test",
               seed_data=Dict("1" => seed_block(1, 20000, 0.9)))
    rebuild_index(; root=r1, verbose=false)

    c2 = get_curve(; roots=RS, U=7.0, Lx=L, V=0.0, obs=:stiffness, x=:beta)
    @test !isempty(c2.variants)         # the TCA=2 run is still surfaced as a variant
end

@testset "atomic write leaves no partial file" begin
    e = make_entry(beta=8.0)
    p = save_entry(e; root=ROOT)
    @test isempty(filter(f -> endswith(f, ".tmp"), readdir(dirname(p))))
end

end # testset

rm(ROOT; recursive=true, force=true)
println("\nAll means_db tests passed.")
