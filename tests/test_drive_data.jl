# ============================================================================
# tests/test_drive_data.jl — reading the archive from the backup drive.
#
#   julia tests/test_drive_data.jl                                  (fake drive, ~30 s)
#   DRIVE_DATA_REAL='<drive>/Yaniv-Data' julia tests/test_drive_data.jl  (+ opens ~6 real files)
#
# Builds a fake `Yaniv-Data` drive in a temp folder: one V7 means_db per cluster
# (written with save_entry OFF the drive, then copied on), fake slims, a V6
# campaign and _records. Checks:
#   * BACKUP_DRIVE alone points MEANS_DB_ROOTS at the drive;
#   * LANDAU_ROOT / CURIE_ROOT read one cluster from elsewhere (e.g. its mount);
#   * means_db rows -> seeds -> slims (find_slims, seed_slims, slim_path, read_slim);
#   * V6 helpers (v6_campaigns, v6_seeds, load_v6_sim_info);
#   * drive_check / drive_summary;
#   * the read-only guard: no means_db write on the drive, no index rebuilt there,
#     and the drive's files are unchanged afterwards.
# The optional real-drive part only reads.
# ============================================================================
using Test, JLD2, Dates

const TMP   = mktempdir()
const DRIVE = joinpath(TMP, "Yaniv-Data")
const BUILD = joinpath(TMP, "build")                  # off the drive: save_entry writes here
const PFX   = "/home/data/testuser/data/MC_Hubbard_V7/data/Camp__01-01-2026"

ENV["BACKUP_DRIVE"] = DRIVE
delete!(ENV, "MEANS_DB_ROOTS")
ENV["MEANS_DB_ROOT"] = joinpath(BUILD, "default_root")
include(joinpath(@__DIR__, "..", "drive_data.jl"))

obs(v) = Dict{String,Any}("value" => v, "error" => 0.01, "correlation_time" => 1.0,
                          "num_measurements" => 2000)
function make_entry(; U, beta=2.0)
    Dict{String,Any}(
        "U" => U, "wanted_n" => 0.875, "Lx" => 8, "Ly" => 8, "beta" => beta,
        "diag_method" => "K0-TCA", "TClustersize" => 4, "V" => 0.0, "B" => 0.0,
        "t" => 1.0, "SC_L" => 1, "impurity" => 0.0, "U_disorder" => "uniform",
        "pairing_symmetry" => "s-wave", "boundary_condition" => "PBC",
        "gauge_choice" => "Mixed", "guess" => "uniform", "sim_num" => 1,
        "means" => Dict{String,Any}("n" => obs(0.875), "stiffness" => obs(0.05)))
end
seed(unit, s) = Dict{String,Any}(
    "info"  => Dict{String,Any}("num_current_measurments" => 2000, "mu" => -1.0,
                                "status" => "done", "dir_path" => "$PFX/$unit/seed_$s"),
    "means" => Dict{String,Any}("n" => obs(0.875), "stiffness" => obs(0.05 + unit / 1000)))

function write_slim(path; stiff, src="means_db")
    mkpath(dirname(path))
    jldopen(path, "w") do f
        f["slim_format"] = "slim_v1"
        f["sim_data"] = Dict{String,Any}("U" => 3.0, "Lx" => 8, "mu" => -1.0)
        f["meta/num_measurements"] = 2000; f["meta/num_bins"] = 100; f["meta/bin_size"] = 20
        f["meta/means_source"] = src
        f["means/stiffness"] = obs(stiff); f["means/n"] = obs(0.875)
        f["binned/n"] = fill(0.875, 100); f["binned/DdD"] = fill(0.1, 100)
        f["meanobs/LDOS"] = zeros(4)
    end
end

# --- build the fake drive ----------------------------------------------------
for (c, U, units) in (("curie", 3.0, (5, 6)), ("landau", 4.0, (7, 8)))
    b = joinpath(BUILD, c)
    save_entry(make_entry(U=U); root=b, cluster=c, source="test",
               seed_data=Dict(string(u) => seed(u, 1) for u in units))
    rebuild_index(; root=b, verbose=false)
    dest = joinpath(DRIVE, c, "MC_Hubbard_V7", "means_db")
    mkpath(dirname(dest)); cp(b, dest)
end
# slims: every seed except landau unit 8 (a "held-back" seed)
for (c, u) in (("curie", 5), ("curie", 6), ("landau", 7))
    write_slim(joinpath(DRIVE, c, "MC_Hubbard_V7", "data", "Camp__01-01-2026", string(u), "seed_1",
                        "results_slim.jld2"); stiff=0.05 + u / 1000)
end
# V6 campaign
v6c = joinpath(DRIVE, "landau", "MC_Hubbard_V6", "data", "Old__01-01-2026")
mkpath(joinpath(v6c, "num_control"))
jldsave(joinpath(v6c, "num_control", "sim_info.jld2");
        var"1" = Dict{String,Any}("U" => 4.0, "beta" => 10.0, "means" => Dict("stiffness" => obs(0.1))))
for u in ("2", "10")
    sd = joinpath(v6c, u, "seed_777")
    write_slim(joinpath(sd, "results_slim.jld2"); stiff=0.1, src="means_cache")
    jldsave(joinpath(sd, "means_cache.jld2"); means=Dict("stiffness" => obs(0.1)))
end
mkpath(joinpath(v6c, "3", "seed_777"))                  # a unit without a slim
mkpath(joinpath(DRIVE, "landau", "MC_Hubbard_V6", "data", "Unused__01-01-2026", "num_control"))
# _records: one raw chunk (unit 8 held back) and a means_db chunk (skipped)
rc = joinpath(DRIVE, "_records", "landau", "landau__Camp__01-01-2026__c01")
mkpath(rc)
write(joinpath(rc, "meta.tsv"),
      "R\t100\t1\tMC_Hubbard_V7/data/Camp__01-01-2026/7/seed_1/results.jld2\n" *
      "C\t10\t1\tMC_Hubbard_V7/data/Camp__01-01-2026/7/sim.jld2\n" *
      "R\t100\t1\tMC_Hubbard_V7/data/Camp__01-01-2026/8/seed_1/results.jld2\n")
write(joinpath(rc, "slim_problems.txt"),
      "/tmp/x/extract/MC_Hubbard_V7/data/Camp__01-01-2026/8/seed_1/results.jld2\tmissing_slim\tno results_slim.jld2\n")
mkpath(joinpath(DRIVE, "_records", "landau", "landau__means_db__c01"))
write(joinpath(DRIVE, "_records", "landau", "landau__means_db__c01", "meta.tsv"),
      "R\t1\t1\tMC_Hubbard_V7/data/Nope/1/seed_1/results.jld2\n")

"Size + mtime of every file under the drive, to prove the guard tests wrote nothing."
snapshot() = Dict(joinpath(d, f) => (filesize(joinpath(d, f)), mtime(joinpath(d, f)))
                  for (d, _, fs) in walkdir(DRIVE) for f in fs)
before = snapshot()

@testset "drive_data" begin

@testset "one setting: BACKUP_DRIVE" begin
    @test drive_root() == DRIVE
    @test MEANS_DB_ROOTS == drive_means_db_roots(DRIVE)
    @test endswith(replace(MEANS_DB_ROOTS[1], '\\' => '/'), "landau/MC_Hubbard_V7/means_db")
    @test length(find_all(; U=3.0)) == 1 && length(find_all(; U=4.0)) == 1
end

@testset "V7: means_db rows -> slims" begin
    @test slim_path("$PFX/5/seed_1"; cluster="curie", drive=DRIVE) ==
          normpath(joinpath(DRIVE, "curie", "MC_Hubbard_V7", "data", "Camp__01-01-2026", "5", "seed_1", "results_slim.jld2"))
    @test slim_path(raw"Z:\data\MC_Hubbard_V7\data\Camp__01-01-2026\5\seed_1\results.jld2"; cluster="curie", drive=DRIVE) ==
          slim_path("$PFX/5/seed_1"; cluster="curie", drive=DRIVE)
    @test_throws ErrorException slim_path("$PFX/5/seed_1"; cluster="mars", drive=DRIVE)

    s = find_slims(; U=3.0)
    @test length(s) == 2 && all(x -> x.cluster == "curie" && x.exists, s)
    @test sort([x.seed for x in s]) == ["5", "6"]
    held = find_slims(; U=4.0)
    @test sort([(x.seed, x.exists) for x in held]) == [("7", true), ("8", false)]

    r = read_slim(s[1].slim)
    @test r.format == "slim_v1" && r.meta["bin_size"] == 20 && r.sim_data["mu"] == -1.0
    @test Set(keys(r.means)) == Set(["n", "stiffness"]) && haskey(r.meanobs, "LDOS")
    # slim means == the seed's means_db means (what the real slim guarantees)
    db = read_seeds(s[1].db_path)[s[1].seed]["means"]["stiffness"]["value"]
    @test r.means["stiffness"]["value"] == db
    r2 = read_slim(s[1].slim; obs=:n)
    @test collect(keys(r2.means)) == ["n"] && collect(keys(r2.binned)) == ["n"] && isempty(r2.meanobs)
    @test slim_obs(s[1].slim).binned == ["DdD", "n"]
end

@testset "V6" begin
    cs = v6_campaigns(; cluster="landau")
    @test [c.campaign for c in cs] == ["Old__01-01-2026", "Unused__01-01-2026"]
    @test !isempty(cs[1].sim_info) && isempty(cs[2].sim_info)
    si = load_v6_sim_info("landau", "Old__01-01-2026")
    @test si["1"]["means"]["stiffness"]["value"] == 0.1
    @test_throws ErrorException load_v6_sim_info("landau", "Unused__01-01-2026")
    sd = v6_seeds("landau", "Old__01-01-2026")
    @test [x.unit for x in sd] == ["2", "10"]                   # numeric order, unit 3 has no slim
    @test all(x -> !isempty(x.means_cache), sd)
    @test read_slim(sd[1].slim).meta["means_source"] == "means_cache"
    @test slim_path("/home/data/testuser/data/MC_Hubbard_V6/data/Old__01-01-2026/2/seed_777";
                    cluster="landau", drive=DRIVE) == normpath(sd[1].slim)
end

@testset "drive_check / drive_summary" begin
    h = drive_check(; io=devnull)
    @test length(h) == 1
    @test h[1].cluster == "landau" && h[1].seed_dir == "MC_Hubbard_V7/data/Camp__01-01-2026/8/seed_1"
    @test occursin("missing_slim", h[1].reason)
    rows = drive_summary(; io=devnull)
    @test sum(r.slims for r in rows) == 5
    @test any(r -> r.kind == "means_db" && r.cluster == "curie" && r.files >= 2, rows)
end

@testset "per cluster: LANDAU_ROOT / CURIE_ROOT (e.g. a cluster mount)" begin
    # a stand-in for Curie's mount: the same layout as <drive>/curie, somewhere else
    croot = joinpath(TMP, "curie_mount", "data")
    mkpath(dirname(croot)); cp(joinpath(DRIVE, "curie"), croot)
    lroot = joinpath(TMP, "landau_copy")
    cp(joinpath(DRIVE, "landau"), lroot)
    @test cluster_root("curie") == joinpath(DRIVE, "curie")           # no *_ROOT: the drive
    @test cluster_means_db_roots() == drive_means_db_roots(DRIVE)
    @test_throws ErrorException cluster_root("mars")
    withenv("CURIE_ROOT" => croot) do
        @test cluster_root("curie") == croot                          # wins over the drive
        @test cluster_root("landau") == joinpath(DRIVE, "landau")     # the other cluster: unchanged
        roots = cluster_means_db_roots()
        @test roots == [joinpath(DRIVE, "landau", "MC_Hubbard_V7", "means_db"),
                        joinpath(croot, "MC_Hubbard_V7", "means_db")]
        s = find_slims(; roots=roots, U=3.0)
        @test length(s) == 2 && all(x -> x.exists && startswith(x.slim, normpath(croot)), s)
        @test all(x -> startswith(x.db_path, croot), s)
        # drive= still forces the drive
        @test startswith(slim_path("$PFX/5/seed_1"; cluster="curie", drive=DRIVE), normpath(DRIVE))
        @test drive_data_dir("curie") == joinpath(croot, "MC_Hubbard_V7", "data")
    end
    withenv("LANDAU_ROOT" => lroot) do
        cs = v6_campaigns(; cluster="landau")
        @test all(c -> startswith(c.dir, lroot), cs) && length(cs) == 2
        @test load_v6_sim_info("landau", "Old__01-01-2026")["1"]["means"]["stiffness"]["value"] == 0.1
        @test all(x -> startswith(x.slim, lroot), v6_seeds("landau", "Old__01-01-2026"))
    end
    # nothing set: no built-in location, and a clear error from the helpers
    withenv("BACKUP_DRIVE" => nothing, "LANDAU_ROOT" => nothing, "CURIE_ROOT" => nothing,
            "LANDAU_MOUNT" => nothing, "CURIE_MOUNT" => nothing) do
        @test cluster_root("landau") == "" && isempty(cluster_means_db_roots())
        @test_throws ErrorException slim_path("$PFX/5/seed_1"; cluster="curie")
        @test_throws ErrorException v6_campaigns(; cluster="landau")
        # a mount given as a bare drive letter becomes <letter>:\data
        withenv("LANDAU_MOUNT" => "Q:") do
            Sys.iswindows() && @test cluster_root("landau") == raw"Q:\data"
        end
        # (Landau pointed at an empty temp folder: a test never touches the real Y:)
        withenv("CURIE_MOUNT" => joinpath(TMP, "curie_mount"), "LANDAU_MOUNT" => joinpath(TMP, "nowhere")) do
            @test cluster_root("curie") == croot
            @test length(find_slims(; roots=cluster_means_db_roots(), U=3.0)) == 2
        end
    end
    rm(joinpath(TMP, "curie_mount"); recursive=true); rm(lroot; recursive=true)
end

@testset "read-only guard" begin
    croot = drive_means_db_roots(DRIVE)[2]
    @test _on_backup_drive(croot) && !_on_backup_drive(BUILD)
    # the folder name alone protects it, even without BACKUP_DRIVE
    withenv("BACKUP_DRIVE" => nothing) do
        @test _on_backup_drive(croot)
        @test !_on_backup_drive(BUILD)
    end
    @test_throws DriveReadOnly save_entry(make_entry(U=5.0); root=croot)
    @test_throws DriveReadOnly register_simulation(make_entry(U=5.0); root=croot)
    @test_throws DriveReadOnly rebuild_index(; root=croot, verbose=false)
    p = find_all(; U=3.0)[1]["path"]
    @test_throws DriveReadOnly set_status!(p, "running"; root=croot)
    @test_throws DriveReadOnly append_index!(croot, [p])
    @test_throws DriveReadOnly _assert_writable(joinpath(v6c), "update_means")
    # a drive root with no index: the query stops, nothing is rebuilt there
    noidx = joinpath(DRIVE, "landau", "MC_Hubbard_V7", "means_db_copy")
    cp(joinpath(BUILD, "landau"), noidx); rm(joinpath(noidx, "index.jld2")); rm(joinpath(noidx, "index.csv"))
    @test_throws DriveReadOnly find_all(; roots=[noidx], U=4.0)
    @test !isfile(joinpath(noidx, "index.jld2"))
    rm(noidx; recursive=true)
    # nothing on the drive changed
    @test snapshot() == before
    # off the drive, writing still works
    @test isfile(save_entry(make_entry(U=5.0); root=joinpath(BUILD, "curie")))
end

real = get(ENV, "DRIVE_DATA_REAL", "")
if !isempty(real)
    @testset "real drive (read-only): $real" begin
        roots = drive_means_db_roots(real)
        @test all(isdir, roots) && all(r -> isfile(joinpath(r, "index.jld2")), roots)
        @test _on_backup_drive(roots[1])
        sl = find_slims(; roots=roots, drive=real, U=4.0, Lx=10, beta=0.5, TClustersize=4, wanted_n=0.9)
        @test !isempty(sl) && all(x -> x.exists, sl)
        r = read_slim(sl[1].slim; obs=[:n, :stiffness])
        @test r.format == "slim_v1" && r.meta["means_source"] == "means_db"
        @test r.means["n"]["value"] == read_seeds(sl[1].db_path)[sl[1].seed]["means"]["n"]["value"]
        v6 = v6_seeds("landau", "TCA_test__26-04-2026"; drive=real)
        @test length(v6) == 55
        @test read_slim(v6[1].slim; obs=:stiffness).meta["means_source"] == "means_cache"
        si = load_v6_sim_info("landau", "TCA_test__26-04-2026"; drive=real)
        @test count(k -> k != "updated_at", keys(si)) == 55
        cs = v6_campaigns(; drive=real)
        @test count(c -> c.cluster == "curie", cs) == 12
    end
end

end
rm(TMP; recursive=true, force=true)
