# ============================================================================
# bench_profile_collect.jl — summarises the runtime-profile benchmark.
#
#   julia bench_profile_collect.jl <benchmark folder> [<another folder> ...]
#
#   Reads every */profile_results.json under the given folders (one per cluster,
#   e.g. Profile_landau__.. and Profile_curie__..). It prints the tables and
#   writes profile_summary.md and profile_summary.csv into the FIRST folder.
#   Only the small JSON files are read, so this is light enough to point at both
#   clusters' folders from one machine. It needs only the JSON package (no
#   MC_Hubburd).
# ============================================================================
using JSON, Printf

isempty(ARGS) && error("usage: julia bench_profile_collect.jl <folder> [<folder> ...]")

function load_all(folders)
    rows = Any[]
    for f in folders, d in sort(readdir(f; join=true))
        p = joinpath(d, "profile_results.json")
        isfile(p) || continue
        r = JSON.parsefile(p)
        r["_dir"] = d
        push!(rows, r)
    end
    return rows
end

const R = load_all(ARGS)
isempty(R) && error("no profile_results.json found under $(join(ARGS, ", "))")
key(r) = (r["meta"]["cluster"], r["meta"]["L"], r["meta"]["cores"])
sort!(R; by=key)

out = IOBuffer()
emit(s="") = (println(s); println(out, s))
fmt(x) = x isa Number ? @sprintf("%.3g", x) : string(x)

emit("# Runtime profile — $(join(sort(unique(r["meta"]["cluster"] for r in R)), " + "))")
emit()
emit("Jobs: $(length(R)).  Unfinished sections are simply missing from the tables.")
emit()

# ── nodes ───────────────────────────────────────────────────────────────────
emit("## Nodes")
emit()
emit("| cluster | job | host | CPU | BLAS thr | Julia thr | load at start |")
emit("|---|---|---|---|---|---|---|")
for r in R
    m = r["meta"]
    emit("| $(m["cluster"]) | $(basename(r["_dir"])) | $(m["hostname"]) | $(m["cpu_model"]) | $(m["blas_threads"]) | $(m["julia_threads"]) | $(join(round.(Float64.(m["loadavg_start"]), digits=1), " ")) |")
end
emit()

# ── profile ─────────────────────────────────────────────────────────────────
P = [r for r in R if haskey(r, "profile")]
csv = ["cluster,L,cores,t_sweep,t_sweep_phi,t_cluster_action_ms,t_measurement,measurement_over_interval,projected_days,projected_days_fast,top_part,top_part_share"]
if !isempty(P)
    emit("## MC sweep vs measurement (current code)")
    emit()
    emit("`fast` = measurement with the prototype kernels swapped in (from the proto section of the same job, all q kept).")
    emit()
    emit("| cluster | L | cores | sweep [s] | sweep, phi on [s] | cluster action [ms] | 25 sweeps [s] | measurement [s] | meas / interval | meas fast [s] | days / point | days fast | top kernels |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in P
        m = r["meta"]; p = r["profile"]
        parts = Dict(k => Float64(v) for (k, v) in p["parts"])
        tot = sum(values(parts))
        top = sort(collect(parts); by=last, rev=true)[1:min(3, length(parts))]
        tops = join([@sprintf("%s %.0f%%", k, 100v / tot) for (k, v) in top], "; ")
        mi = p["measuring_interval"]
        # projected measurement with the prototype kernels (same job, same node)
        tfast = NaN
        if haskey(r, "proto")
            pr = r["proto"]
            gbest = min(pr["G(q) per-q ZGEMM"]["fast"], pr["G(q) row-block"]["fast"])
            saved = sum(pr[k]["ref"] for k in ("G(q) row-block", "L_ab(q) x4", "fixed-delta L(w)", "DdD", "A(k,w)+BZ+LDOS"))
            spent = gbest + sum(pr[k]["fast"] for k in ("L_ab(q) x4", "fixed-delta L(w)", "DdD", "A(k,w)+BZ+LDOS"))
            tfast = max(p["t_measurement"] - saved + spent, 0.0)
        end
        dfast = isnan(tfast) ? NaN : (p["num_thermal"] * p["t_sweep"] + p["num_measure"] * (mi * p["t_sweep"] + tfast)) / 86400
        emit(@sprintf("| %s | %d | %d | %.3f | %.3f | %.3f | %.1f | %.1f | %.2f | %s | %.1f | %s | %s |",
                      m["cluster"], m["L"], m["cores"], p["t_sweep"], p["t_sweep_phi"], 1e3 * p["t_cluster_action"],
                      mi * p["t_sweep"], p["t_measurement"], p["t_measurement"] / (mi * p["t_sweep"]),
                      isnan(tfast) ? "–" : @sprintf("%.1f", tfast), p["projected_days"],
                      isnan(dfast) ? "–" : @sprintf("%.1f", dfast), tops))
        push!(csv, join([m["cluster"], m["L"], m["cores"], p["t_sweep"], p["t_sweep_phi"], 1e3 * p["t_cluster_action"],
                         p["t_measurement"], p["t_measurement"] / (mi * p["t_sweep"]), p["projected_days"],
                         isnan(dfast) ? "" : dfast, "\"$(top[1][1])\"", top[1][2] / tot], ","))
    end
    emit()
end

# ── proto ───────────────────────────────────────────────────────────────────
Q = [r for r in R if haskey(r, "proto")]
if !isempty(Q)
    emit("## Prototype kernels vs current (speed-up, max relative difference)")
    emit()
    ks = ["G(q) per-q ZGEMM", "G(q) row-block", "L_ab(q) x4", "fixed-delta L(w)", "DdD", "A(k,w)+BZ+LDOS"]
    emit("| cluster | L | cores | " * join(ks, " | ") * " | work |")
    emit("|---|---|---|" * repeat("---|", length(ks) + 1))
    for r in Q
        m = r["meta"]; pr = r["proto"]
        cells = [haskey(pr, k) ? @sprintf("%.3g→%.3g s (x%.1f, %.0e%s)", pr[k]["ref"], pr[k]["fast"], pr[k]["speedup"], pr[k]["relerr"],
                                          haskey(pr[k], "alloc_MB_per_call") ? @sprintf(", %.2g MB", pr[k]["alloc_MB_per_call"]) : "") : "–" for k in ks]
        wk = haskey(pr, "work") ? @sprintf("%.0f MB", pr["work"]["MB"]) : "–"
        emit("| $(m["cluster"]) | $(m["L"]) | $(m["cores"]) | " * join(cells, " | ") * " | $wk |")
    end
    emit()
    emit("Each cell: current → prototype time (speed-up, max relative difference, MB allocated per prototype call). Last column: preallocated work, built once per run.")
    emit()
end

# ── occ ─────────────────────────────────────────────────────────────────────
O = [r for r in R if haskey(r, "occ")]
if !isempty(O)
    emit("## Occupation-convergence phase: one measurement")
    emit()
    emit("| cluster | L | cores | make_measurement_K! [s] | diag + n only [s] | ratio | sweeps between [s] | measurement share now | with n only |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for r in O
        m = r["meta"]; o = r["occ"]; s = o["measuring_interval_occ"] * o["t_sweep"]
        emit(@sprintf("| %s | %d | %d | %.2f | %.3f | %.0f | %.2f | %.0f%% | %.0f%% |", m["cluster"], m["L"], m["cores"],
                      o["t_measurement"], o["t_n_only"], o["t_measurement"] / o["t_n_only"], s,
                      100o["t_measurement"] / (o["t_measurement"] + s), 100o["t_n_only"] / (o["t_n_only"] + s)))
    end
    emit()
end

# ── threads ─────────────────────────────────────────────────────────────────
T = [r for r in R if haskey(r, "threads")]
if !isempty(T)
    emit("## BLAS threads")
    emit()
    emit("| cluster | L | job cores | BLAS threads | TCA cluster action [ms] | full diag [s] |")
    emit("|---|---|---|---|---|---|")
    for r in T, row in r["threads"]["rows"]
        m = r["meta"]
        emit(@sprintf("| %s | %d | %d | %d | %.3f | %.3f |", m["cluster"], m["L"], m["cores"], row["blas_threads"],
                      1e3 * row["t_cluster_action"], row["t_full_diag"]))
    end
    emit()
end

# ── accept ──────────────────────────────────────────────────────────────────
A = [r for r in R if haskey(r, "accept")]
if !isempty(A)
    emit("## Metropolis acceptance (K0-TCA)")
    emit()
    emit("| cluster | L | U | beta | beta_phi | step std | acc Delta | acc phi | <abs Delta> |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for r in A, row in r["accept"]["points"]
        a = row["acceptance"]
        emit(@sprintf("| %s | %d | %.1f | %.2f | %.2f | %.3f | %.1f%% | %s | %.3f |", r["meta"]["cluster"], r["accept"]["L"],
                      row["U"], row["beta"], row["beta_phi"], row["step_std"], 100a["Delta_s"],
                      haskey(a, "phi") ? @sprintf("%.1f%%", 100a["phi"]) : "–", row["mean_abs_Delta"]))
    end
    emit()
end

first_dir = ARGS[1]
write(joinpath(first_dir, "profile_summary.md"), String(take!(out)))
write(joinpath(first_dir, "profile_summary.csv"), join(csv, "\n") * "\n")
println("\nwrote ", joinpath(first_dir, "profile_summary.md"), " and profile_summary.csv")
