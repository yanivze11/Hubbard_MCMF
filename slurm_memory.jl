# ============================================================================
# slurm_memory.jl — how much memory to request for one simulation job.
#
# Base only (no packages), so the submit scripts can include it cheaply.
#
# Mirrors the dominant terms of MC_Hubburd.estimate_memory_bytes: the
# G_matrix_vec_x/_y buffers, 2*(max(Lx,Ly)+1) complex 2N x 2N matrices
# (4.1 GB at L=32), plus about five more 2N x 2N matrices for the lattice. On top
# of that it adds 25% and 1.5 GB for the Julia runtime. Results: 4 GB up to
# L~24, 5 GB at L=28 (estimate 3.92), 8 GB at L=32 (estimate 6.44).
# The field histograms (0.58 MB at 1000 + 1500 x 50 bins) and snapshots (nothing
# kept between writes) are far inside that margin, so this ignores them.
#
# A flat 4 GB per job was too little: every Landau L=32 job of the 2026-09-17
# campaign was OOM-killed (Landau runs 1 CPU per job). The run itself now checks
# its budget at startup (check_memory_budget) and errors out instead.
# ============================================================================

"""
    job_mem_gb(Lx, Ly) -> Int

Memory (GB) one simulation job needs: 1.25 × (current-vertex buffers + lattice
matrices) + 1.5 GB runtime, at least 4 GB.
"""
function job_mem_gb(Lx::Integer, Ly::Integer)
    N = Lx * Ly
    mat = (2N)^2 * 16 / 2^30                     # one complex 2N x 2N matrix, GiB
    G = 2 * (max(Lx, Ly) + 1) * mat
    lattice = 5 * mat
    return max(4, ceil(Int, 1.25 * (G + lattice) + 1.5))
end

"""
    mem_per_cpu_gb(Lx, Ly, cpus) -> Int

`--mem-per-cpu` (GB) for a job with `cpus` CPUs: `job_mem_gb` split over the CPUs
(SLURM gives the job that times its CPU count), at least 4 GB.
"""
mem_per_cpu_gb(Lx::Integer, Ly::Integer, cpus::Integer) = max(4, cld(job_mem_gb(Lx, Ly), cpus))
