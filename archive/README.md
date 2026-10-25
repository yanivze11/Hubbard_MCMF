# archive/

Code that is no longer part of the V7 pipeline, kept for reference. Nothing in
the active pipeline includes these files. They may need small fixes to run again
(paths, settings now in `cluster_config.jl`).

| Folder | Contents |
|---|---|
| `old_drivers/` | V6 submit drivers (`fh_pre_run*.jl`, `fh_run*.jl`) and the V6 num_control (`pre_run_sim_num.jl`, `run_sim_num_control*.jl`). Replaced by `fh_pre_run_V.jl` and `pre_run_num_control_V.jl` / `run_num_control_V.jl`, which use `means_db`. Includes the U = 0 (NIL) drivers. |
| `batch_reports/` | V6 pipeline that ran `simulation_statistics_report.jl` for every simulation of a campaign as a SLURM array. |
| `v6_tools/` | one-off audit of the V6 `sim_info.jld2` files into a CSV / Markdown summary. |
| `means_db_migration/` | one-time conversion of the V6 `means_store.jld2` into the per-simulation `means_db` layout, and its validation. |
| `benchmarks/` | runtime and autocorrelation-time benchmarks (`benchmark_runtime_tau/`), the October 2026 runtime profile (`profile_2026-10/`), cluster storage utilities (`cluster_utils/`). |
| `mean_field/` | real-space BdG and homogeneous k-space mean-field solvers, analytic BCS, L-scans. |
| `magnetic_field/` | magnetic-field mean-field study and its driver. |
| `gaussian_model/` | Gaussian-Δ model of the density of states. |
| `scratch/` | throw-away tests and trials. |

Files here reach the top-level code with `joinpath(@__DIR__, "..", "..", "X.jl")`
(three `..` from `benchmarks/profile_2026-10/`).
