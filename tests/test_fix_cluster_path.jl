# ============================================================================
# tests/test_fix_cluster_path.jl — fix_cluster_path maps a cluster path to the
# backup drive (first, when BACKUP_DRIVE is set) or to the right cluster's
# mount, never to the other cluster.
#
#   julia tests/test_fix_cluster_path.jl        (laptop; ~1 min to load the engine)
#
# Builds fake "mounts" and a fake backup drive in a temp folder. Both clusters
# hold the SAME campaign/unit path, as happens for real.
# ============================================================================
using Test

const TMP = mktempdir()
const UNIT = "data/MC_Hubbard_V7/data/Camp__01-01-2026/12/seed_1"
for d in ("landau_mount", "curie_mount", "drive/curie/MC_Hubbard_V7/data/Camp__01-01-2026/12/seed_1")
    mkpath(joinpath(TMP, d))
end
mkpath(joinpath(TMP, "landau_mount", UNIT)); mkpath(joinpath(TMP, "curie_mount", UNIT))
mkpath(joinpath(TMP, "landau_mount", "data/MC_Hubbard_V7/data/OnlyLandau/1/seed_1"))
# a seed only on Curie's mount, not on the drive (like a held-back raw seed)
const HELD = "data/MC_Hubbard_V7/data/Camp__01-01-2026/13/seed_1"
mkpath(joinpath(TMP, "curie_mount", HELD))

# Landau's own folder (LANDAU_ROOT, e.g. a local copy): has unit 14, which the mount has too
const ROOTED = "MC_Hubbard_V7/data/Camp__01-01-2026/14/seed_1"
mkpath(joinpath(TMP, "landau_root", ROOTED)); mkpath(joinpath(TMP, "landau_mount", "data", ROOTED))

# The settings are read when MC_Hubburd.jl is loaded.
ENV["LANDAU_ROOT"] = joinpath(TMP, "landau_root")
ENV["CLUSTER_DATA_PREFIX"] = "/home/data/testuser"
ENV["LANDAU_MOUNT"] = joinpath(TMP, "landau_mount")
ENV["CURIE_MOUNT"]  = joinpath(TMP, "curie_mount")
ENV["BACKUP_DRIVE"] = joinpath(TMP, "drive")
include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

same(a, b) = normpath(a) == normpath(b)
cl = "/home/data/testuser/" * UNIT

@testset "fix_cluster_path" begin
    drive_unit = joinpath(TMP, "drive", "curie", "MC_Hubbard_V7/data/Camp__01-01-2026/12/seed_1")
    # the cluster decides, even though both mounts have the folder
    @test same(fix_cluster_path(cl; cluster="landau"), joinpath(TMP, "landau_mount", UNIT))
    # BACKUP_DRIVE set: the drive wins over a mount that also has the folder
    @test same(fix_cluster_path(cl; cluster="curie"), drive_unit)
    # unknown cluster + folder on both: Landau (old behaviour), with a warning
    @test_logs (:warn, r"several clusters") match_mode=:any fix_cluster_path(cl)
    # only one cluster has it: no ambiguity
    only = "/home/data/testuser/data/MC_Hubbard_V7/data/OnlyLandau/1/seed_1"
    @test same(fix_cluster_path(only), joinpath(TMP, "landau_mount", "data/MC_Hubbard_V7/data/OnlyLandau/1/seed_1"))
    @test fix_cluster_path(only; cluster="curie") == only          # not on Curie: unchanged
    # not on the drive: the mount is the fallback
    @test same(fix_cluster_path("/home/data/testuser/" * HELD; cluster="curie"),
               joinpath(TMP, "curie_mount", HELD))
    # backup drive still found once the mount is gone
    rm(joinpath(TMP, "curie_mount", UNIT); recursive=true)
    @test same(fix_cluster_path(cl; cluster="curie"), drive_unit)
    # an existing folder is returned as it is
    @test fix_cluster_path(TMP) == TMP
    # candidates, in order: drive, then mount
    c = cluster_path_candidates(cl, "curie")
    @test first.(c) == ["curie", "curie"]
    @test occursin("drive", c[1][2]) && occursin("curie_mount", c[2][2])
    @test isempty(cluster_path_candidates("/somewhere/else", "curie"))
    # a path stored under another user's prefix is still recognised (by its data/MC_Hubbard_Vx part)
    other = "/home/data/someoneelse/" * UNIT
    @test same(fix_cluster_path(other; cluster="curie"), drive_unit)
    @test !isempty(cluster_path_candidates("/home/data/testuser/notes/x", "curie"))   # prefix match: mount only
    @test isempty(cluster_path_candidates("/var/tmp/other/x", "curie"))
    # LANDAU_ROOT: tried before the drive and the mount
    rooted = "/home/data/testuser/data/" * ROOTED
    @test same(fix_cluster_path(rooted; cluster="landau"), joinpath(TMP, "landau_root", ROOTED))
    cl_l = cluster_path_candidates(rooted, "landau")
    @test occursin("landau_root", cl_l[1][2]) && occursin("drive", cl_l[2][2]) && occursin("landau_mount", cl_l[3][2])
    # not in LANDAU_ROOT: the mount is still found
    @test same(fix_cluster_path(cl; cluster="landau"), joinpath(TMP, "landau_mount", UNIT))
end
