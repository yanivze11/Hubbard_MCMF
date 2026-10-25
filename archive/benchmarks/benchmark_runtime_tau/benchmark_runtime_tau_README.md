Runtime and Autocorrelation Benchmark

Purpose
- Submit one cluster job per benchmark point.
- Run the normal MC simulation path using running_simulation_once.
- Estimate:
  - Effective MC step time
  - Worst-case autocorrelation time across selected observable families
  - Steps and wall-time per uncorrelated measurement

Files
- benchmark_runtime_tau/benchmark_runtime_tau_config.json
  Config file with defaults and per-point overrides.
- benchmark_runtime_tau/benchmark_runtime_tau_pre_run.jl
  Generates per-point job folders, writes sim.jld2 manifests, writes slurm files, and submits jobs.
- benchmark_runtime_tau/benchmark_runtime_tau_run.jl
  Worker script run by each SLURM job. Executes one point and writes benchmark_summary outputs.
  By default, it skips get_means/tau and leaves tau recomputation to the collector.
- benchmark_runtime_tau/benchmark_runtime_tau_collect.jl
  Aggregates all point summaries into CSV, JLD2, JSON, and sim_info-like JLD2.
  By default, recomputes get_means + tau_worst from simulation data in each point folder.

What is measured
- wall_time_sec: Full simulation wall time for running_simulation_once.
- mean_step_time_sec: Legacy step-time metric (collector may derive from active compute time).
- mean_mc_step_time_sec: Mean time per MC_stepV2! call from explicit timed counters.
- mean_mc_step_time_source: Provenance for mean_mc_step_time_sec.
- mean_measurement_time_sec: Mean time per make_measurement_K! call (save/checkpoint excluded).
- mean_measurement_time_source: Provenance for mean_measurement_time_sec.
- tau_worst: Max tau among matched observable keys.
- steps_per_uncorr: tau_worst * measuring_interval.
- time_per_uncorr_sec: mean_mc_step_time_sec * steps_per_uncorr when available, otherwise legacy mean_step_time_sec.
- indep_samples_per_hour: 3600 / time_per_uncorr_sec.

Timing provenance notes
- Collector precedence for explicit timing means:
  1) timed counters in loaded current_state
  2) timed counters in sim.jld2 seed tracking
  3) existing benchmark_summary values
  4) fallback division from active or wall timing
- Worker writes timed accumulator fields for transparency:
  - timed_mc_step_total_sec, timed_mc_step_count
  - timed_measurement_total_sec, timed_measurement_count

Observable families for worst-case tau
Configured by tau_observable_patterns in benchmark_runtime_tau/benchmark_runtime_tau_config.json.
Current defaults target:
- n
- Kx
- Delta_MC_abs
- K0_action (MCMF action proxy)
- Lxx and related q-family keys
- conductivity_regular_xx_fixed and family
- Drude_weight_fixed
- conductivity_dc_xx and conductivity_dc_yy
- drude_dc_xx and drude_dc family

Cluster workflow
1. Submit the pre-run through your wrapper:
  julia pre_pre_run.jl benchmark_runtime_tau/benchmark_runtime_tau_pre_run.jl benchmark_runtime_tau/benchmark_runtime_tau_config.json

2. Each benchmark point creates a numbered folder under output_base_dir/folder_name:
   - sim.jld2 (point manifest)
   - slurm.txt
   - job logs
   - benchmark_summary.jld2 and benchmark_summary.json after completion

3. Aggregate when jobs are done:
  julia benchmark_runtime_tau/benchmark_runtime_tau_collect.jl <V6_DATA_ROOT>/benchmark_runtime_tau/runtime_tau_28-03-2026

   Optional flags:
  - --no-recompute-tau
    Use existing benchmark_summary tau values instead of recomputing from results.
  - --no-write-point-summaries
    Do not overwrite per-point benchmark_summary outputs.

Output files from aggregation
- benchmark_runtime_tau_aggregate.jld2
- benchmark_runtime_tau_aggregate.csv
- benchmark_runtime_tau_aggregate.json
- benchmark_runtime_tau_sim_info.jld2

Tau warning behavior
- The collector writes summary["tau_warning"] when tau_worst=1.0 due to missing/invalid matched correlation_time values.
- This distinguishes true tau≈1 behavior from fallback-to-1 due to no usable matched keys.

Notes
- One seed is used per point (as requested).
- Cores are taken from CPUS_PER_TASK exported by pre_pre_run.jl (cluster-specific), and applied to all points in the batch.
- This benchmark excludes occupation-convergence timing and runs only the MC simulation stage.
- You can enable worker-side tau as an override by setting compute_tau_in_worker=true in defaults or per-point config.
