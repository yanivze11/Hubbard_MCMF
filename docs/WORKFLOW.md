# Habburd_MC_V7 — workflow guide

How to run a simulation campaign on a SLURM cluster, how its raw Monte-Carlo
data is turned into means and error bars, how to read the results, and how to
read the archived data on the backup drive. The physics and a single run on your
own computer are in the [README](../README.md); what is specific to the FHRC
clusters (Landau, Curie: storage rules, scratch-mode jobs, preemption) is in
[FHRC_CLUSTERS.md](FHRC_CLUSTERS.md).

> **The pipeline in one line**
> `pre_pre_run.jl` → submit driver (`fh_pre_run_V.jl`) → one SLURM job per
> simulation (`fh_run_n.jl`) → `results.jld2` → num_control
> (`run_num_control_V.jl`) → **`means_db`** → queries (`get_curve`, `get_observable`, …).

Contents

1. [What the code computes](#1-what-the-code-computes)
2. [One-time setup](#2-one-time-setup)
3. [Where things live](#3-where-things-live)
4. [Submitting a campaign (pre_pre_run → pre_run)](#4-submitting-a-campaign)
5. [One simulation job (run)](#5-one-simulation-job-run)
6. [Means and error bars (num_control)](#6-means-and-error-bars-num_control)
7. [Reading results: `means_db`](#7-reading-results-means_db)
8. [The backup drive](#8-the-backup-drive)
9. [Extending the code](#9-extending-the-code)
10. [Troubleshooting](#10-troubleshooting)
11. [Appendix A — parameters of one simulation (`sim.jld2`)](#appendix-a--parameters-of-one-simulation-simjld2)
12. [Appendix B — file index](#appendix-b--file-index)

---

## 1. What the code computes

Monte-Carlo mean-field (MCMF) simulation of the attractive-*U* Hubbard model on a
2D square lattice; the model, the method and the observables are described in
the [README](../README.md#the-physics).

A **simulation** is one point in parameter space (U, n, L, T, disorder, …). It
can have several **MC seeds** (independent Markov chains) and, for disordered
systems, several **disorder seeds** (independent disorder samples; each sample
is its own simulation). For every measurement the code records ~50 observables
(density, pairing correlations, superfluid stiffness ingredients, conductivities,
DOS, …); num_control turns these streams into means with error bars.

## 2. One-time setup

### 2.1 Settings for your user — `cluster_config.jl`

All user- and cluster-specific values are in **`cluster_config.jl`**:

| Setting | What it is | Env override |
|---|---|---|
| `CLUSTER_USER` | your user name on the clusters (default: `$USER`); the two folders below are built from it | `CLUSTER_USER` |
| `V7_DATA_ROOT` | data tree on the cluster (default `/home/data/<user>/data/MC_Hubbard_V7`) | `V7_DATA_ROOT` |
| `V6_DATA_ROOT` | the old V6 tree, used only by `archive/` (default next to `V7_DATA_ROOT`) | `V6_DATA_ROOT` |
| `SCRATCH_ROOT` | node-local scratch of the jobs (default `/home/scr/<user>`) | `SCRATCH_ROOT` |
| `JULIA_BIN` | Julia executable the jobs run (default: the Julia you submit with) | `JULIA_BIN` |
| `JOB_ACCOUNT` | SLURM account of the simulation jobs | `JOB_ACCOUNT` |
| `PRIORITY_ACCOUNT`, `PRIORITY_PARTITION` | the group queue, used by the pre_run and num_control jobs | same names |
| `CLUSTER_SETTINGS` | per cluster: depot, partitions, cores per job, excluded nodes | `JULIA_DEPOT_PATH`, `PARTITIONS`, `CPUS_PER_TASK`, `EXCLUDE_NODES` |

Check these (the accounts and partitions above all) before the first submission.
Nothing else in the submit scripts is user-specific, and no script has a path of
its own.

On your own computer, the locations are environment variables read by
`means_db.jl` (§7.1); none has a built-in default: `BACKUP_DRIVE` (the backup
drive's `Yaniv-Data` folder; with it set, the code reads the archived data there,
§8.2), `LANDAU_ROOT` / `CURIE_ROOT` (read one cluster from another folder, §8.2),
`LANDAU_MOUNT` / `CURIE_MOUNT` (where each cluster's `/home/data/<user>` is mounted),
`MEANS_DB_ROOTS` (where to read means from; overrides all of them) and `MEANS_DB_ROOT`
(the default database for writing; if unset, a `means_db` folder next to the code). The old V6 store `means_store.jl` reads `V6_CLUSTER_RESULTS`
(the folder holding the V6 `sim_info` files) and `MEANS_STORE_DIR` (default
`<V6_CLUSTER_RESULTS>/means_store`); it has no default and says so if they are
unset. Set them before `include`, e.g. in `~/.julia/config/startup.jl`.

### 2.2 Julia and packages

The packages are listed in `Project.toml`. On a laptop:
`julia --project=. -e 'using Pkg; Pkg.instantiate()'`. `Manifest.toml` records
the exact versions of the Curie depot (Julia 1.12.1, 2026-10-22), i.e. what
produced the V7 data. With another Julia version, `Pkg.instantiate()` may warn;
`Pkg.resolve()` then picks versions that fit your Julia. On a cluster each
cluster needs its own depot, and the jobs use its default environment; see
[FHRC_CLUSTERS.md §3](FHRC_CLUSTERS.md#3-one-time-setup).

**Jupyter notebooks.** IJulia's default kernel starts Julia with `--project=@.`,
so from `notebooks/` it uses this repository's environment. If that environment
is not instantiated, the kernel fails to start ("Package Preferences ... is
required but does not seem to be installed"). Either instantiate it, or add a
kernel on an environment that has the packages and choose it in the notebook:
`using IJulia; installkernel("Julia 1.11 default env", "--project=@v1.11")`.

### 2.3 Code and cluster rules

One git checkout of the code on the cluster, updated with `git pull`. The FHRC
clusters have strict rules on how jobs use the shared storage; the code follows
them (scratch-mode jobs, no polling of the data disk, night-time transfers).
Read [FHRC_CLUSTERS.md §2](FHRC_CLUSTERS.md#2-the-rules) before the first
submission.

## 3. Where things live

```
/usr/people/<group>/<user>/Projects/MC_Habburd_V7/   code (one checkout, both clusters)

<V7_DATA_ROOT> = /home/data/<user>/data/MC_Hubbard_V7/   (one per cluster, same path string)
├── data/<campaign>/                 one folder per campaign, e.g. HighT__13-10-2026
│   └── <unit>/                      one folder per simulation ("work unit"): 1, 2, 3, ...
│       ├── sim.jld2                 parameters + per-seed tracking
│       ├── slurm.txt                the job script (resubmit with `sbatch slurm.txt`)
│       ├── <jobid>_1.out / .error   job stdout / stderr
│       └── seed_<s>/
│           ├── results.jld2         raw MC measurements (large: ~100 MB–1 GB)
│           ├── checkpoint.jld2      last field configuration, for resuming
│           ├── simulation_log.txt   progress log ("Measurments taken : X/Y")
│           └── field_snapshots.jld2 periodic raw-field snapshots (if enabled)
├── means_db/                        the means: one small file per simulation + an index
│   ├── index.jld2, index.csv
│   └── U<U>/L<Lx>x<Ly>/U<U>_n<n>_L<Lx>x<Ly>_TCA<c>_b<beta>_<hash>.jld2
└── sim_logs/                        logs of the pre_run and num_control jobs
```

In an archived campaign (every campaign up to October 2026, §8) `results.jld2`
and `checkpoint.jld2` are gone: each seed folder holds a small
`results_slim.jld2` instead, and the seed cannot be resumed.

The two clusters (Landau, Curie) have **separate** data trees. A campaign is
submitted on both. A fixed rule splits its simulations between them, so they never
duplicate work, and the queries read both trees together.

## 4. Submitting a campaign

### 4.1 Edit the sweep

Copy `fh_pre_run_V.jl` (the template driver) or edit it. Everything you choose is
in the block under `# The sweep`:

```julia
const Ls             = [(8, 8), (10, 10)]   # (Lx, Ly)
const Us             = [2.0, 4.0, 6.0]
const Vs             = [0.0]                # potential-disorder strength
const BETAS          = ...                  # inverse temperatures
const WANTED_NS      = [0.6, 0.7, 0.8, 0.9] # target densities
const DISORDER_SEEDS = [1]                  # which disorder samples
const SCHEMES        = [(name="baseline", alpha=1.0, beta_phi=0.0, Qx=0.0, Qy=0.0), ...]
const MC_SEED        = 1
const DIAG_METHOD    = "K0"
const num_measure    = 20000
...
```

and the campaign folder name: `FOLDER_NAME` (env var) or the `folder_name`
default in the file, e.g. `HighT__13-10-2026`. Appendix A lists every parameter.

> **Statistics come from disorder samples.** Each `(parameters, disorder_seed)`
> is a separate simulation. For disordered systems add `DISORDER_SEEDS`; more MC
> seeds on the same sample add much less. At `V = 0` (clean system) every
> disorder seed gives the same system, so only the first one is used.

> **Never change a value "a little".** Every parameter in `IDENTITY_PARAMS`
> (means_db.jl) is part of the simulation's identity hash. `beta=16.7` and
> `beta=16.67` are different simulations and are never averaged together.

### 4.2 Preview on the laptop (submits nothing)

```bash
PREVIEW=1 julia fh_pre_run_V.jl
```

or, in VS Code, run the block at the end of the file (Alt+Enter). It prints how
many simulations the sweep has and how they split between Landau and Curie. It
needs no cluster and no database.

### 4.3 Submit, on each cluster

```bash
cd <code checkout> && git pull
julia pre_pre_run.jl fh_pre_run_V.jl                          # submits
DRY_RUN=1 julia pre_pre_run.jl fh_pre_run_V.jl                # shows what it would submit
```

Run the same command on Landau **and** on Curie.

What happens:

1. **`pre_pre_run.jl`** (login node, seconds) records what was submitted in
   `sim_logs/fh_pre_run_V_<timestamp>/` and submits a small **pre_run job** that
   exports the cluster's settings and runs the driver
   ([FHRC_CLUSTERS.md §4](FHRC_CLUSTERS.md#4-submitting-a-campaign)).
2. **The driver** (`fh_pre_run_V.jl`, inside the pre_run job) loops over the sweep.
   For each simulation it:
   - computes the identity hash (`params_hash`) and the owning cluster
     (`assign_cluster(hash, mc_seed)`, a pure function, so both clusters agree
     without talking to each other); it skips simulations the other cluster owns;
   - skips simulations already registered in this cluster's `means_db`
     (`is_registered`), so **re-running the driver is safe** and submits only
     what is missing;
   - creates `data/<campaign>/<unit>/`, writes `sim.jld2` and `slurm.txt`, and
     `sbatch`es the job;
   - only after `sbatch` succeeded, **registers** the simulation in `means_db`
     with status `submitted` (`register_simulation`). The index is updated once,
     at the end.

The memory of each job is `slurm_memory.jl`'s estimate for its lattice size
(4 GB up to L≈24, 8 GB at L=32). The job script itself depends on `JOB_MODE`
(§5.1).

### 4.4 Other drivers

| Driver | Use |
|---|---|
| `fh_pre_run_V.jl` | template: U/L/V/β/n sweep × decoupling schemes |
| `fh_pre_run_mu_disorder.jl` | μ–U disorder campaign (`mu_disorder`, `mu_disorder_f`) |
| `fh_pre_run_tarat_rerun.jl` | adds MC seeds 2–4 to existing simulations (same identity, so they pool with seed 1) |
| `fh_resubmit_campaign.jl` | resubmits every **unfinished** simulation of a campaign; each job resumes from its checkpoint. `julia pre_pre_run.jl fh_resubmit_campaign.jl --folder=<campaign> [--dry-run]` |
| `fh_pre_run_continue_bad.jl` | finds points whose statistics look wrong (too few uncorrelated samples, S(0) rising with T, outlier in L) and continues those chains |
| `tests_pre_run.jl` | runs test files from `tests/` as SLURM jobs |

### 4.5 Watching the jobs

```bash
squeue -u $USER
tail <V7_DATA_ROOT>/data/<campaign>/<unit>/seed_1/simulation_log.txt
```

In Julia, `db_status()` shows how many simulations each cluster has in each status.

## 5. One simulation job (run)

### 5.1 Job modes

`slurm.txt` runs in one of two modes, chosen at submission by `JOB_MODE`
(`slurm_job_mode.jl`):

- **`scratch`** (default): the job runs on the node's local disk and copies its
  files back to the data tree once per hour and at the end (`scratch_job.sh`).
  It survives preemption: on a requeue it resumes from the last copy.
- **`legacy`**: the job runs directly in its unit folder on the data disk.

The mode is written into `slurm.txt`, so a resubmitted unit keeps its mode.
Details, and why: [FHRC_CLUSTERS.md §5–6](FHRC_CLUSTERS.md#5-how-a-simulation-job-runs-scratch-mode).

### 5.2 What `fh_run_n.jl` does

`fh_run_n.jl --sim_path=<unit>` reads `sim.jld2` and calls
`running_simulation_wanted_occupation` (`occupation_convergence.jl`):

1. **Find μ for the wanted density** (`converge_to_occupation!`). At each trial μ
   a short MC run (`num_thermal_occ`, `num_measure_occ`, only ⟨n⟩ measured) is
   extended until it has `num_uncorrelated_measurments` uncorrelated ⟨n⟩ samples.
   The starting bracket is mean-field μ ± 0.75 (`mf_starting_mu`); it is widened
   by 2 until it brackets `wanted_n`, then narrowed by false-position
   interpolation until |⟨n⟩ − wanted_n| < `n_tol` (at most `max_iter` steps).
2. **Thermalize** at that μ until `num_thermal` MC steps are done (`Thermalize!`).
   The last configuration of the μ search is the starting point, and its MC steps
   count toward `num_thermal`.
3. **Measure** `num_measure` times (`MeasurmentSteps!`, `make_measurement_K!`).
   The interval between measurements is the larger of `measuring_interval` and
   the interval the μ search ended with (the search lengthens it until its ⟨n⟩
   samples are uncorrelated). One MC step = one sweep of N single-site updates
   per field.
4. **Save** every `saving_interval` measurements (500 in `fh_run_n.jl`) or every
   hour (`dump_time`), whichever comes first (`save_progress`): the new
   measurements are appended to `results.jld2` as one chunk, the field
   configuration goes to `checkpoint.jld2`, the per-seed tracking in `sim.jld2`
   is updated, and a progress line goes to `simulation_log.txt`.

**Resuming.** If `checkpoint.jld2` exists, the job continues the same Markov chain
from it (thermalization and measurements so far are kept). This is what makes
SLURM requeues (preemption) harmless. A seed whose raw data was slimmed (§8)
can never be resumed (`assert_not_slimmed`).

## 6. Means and error bars (num_control)

### 6.1 Running it

```bash
FOLDER_NAME=<campaign> julia pre_run_num_control_V.jl      # on each cluster
```

This submits one job (`run_num_control_V.jl`) in the group queue. By default
(`LOOP_MODE=0`) it makes **one pass** and exits: run it once the campaign has
finished (and again if seeds are added). `LOOP_MODE=1` is a **monitor**: one
pass every `LOOP_INTERVAL_SEC` (default 3600 s) for up to `LOOP_MAX_HOURS`
(168 h), stopping once every seed reports `finished`; each pass recomputes only
seeds with new measurements. On the FHRC clusters this polling breaks the
storage rules, so do not use it there
([FHRC_CLUSTERS.md §7](FHRC_CLUSTERS.md#7-num_control-on-the-cluster)). Follow
the job with `tail <V7_DATA_ROOT>/sim_logs/num_control_<cluster>_<timestamp>/num_control.out`.

| Env var | Default | Meaning |
|---|---|---|
| `FOLDER_NAME` | — | the campaign |
| `LOOP_MODE` | `0` | `0` = one pass, then exit; `1` = monitor loop |
| `LOOP_INTERVAL_SEC` | `3600` | seconds between passes |
| `LOOP_MAX_HOURS` | `168` | time budget |
| `NUMCTL_MEM` | `8GB` | memory per CPU; raise it for large L |
| `NUMCTL_TIME` | `7-00:00:00` | walltime |
| `NUMCTL_PARTITION` | the group queue | another partition if it rejects the walltime |
| `DRY_RUN` | `0` | `1` = print the job script, submit nothing |

### 6.2 One pass, step by step (`update_means` in `sim_num_control.jl`)

1. `get_sim_info(campaign)` reads every `<unit>/sim.jld2`.
2. `merge_sims` joins units with identical parameters, so seeds of one simulation
   that ran in different unit folders are averaged together.
3. `get_seeds_means` computes **per-seed means**: seeds already current in
   `means_db` (same measurement count) are read back, not recomputed; otherwise
   `load_results` loads `results.jld2` and `get_means` computes every observable.
4. `average_over_seeds` combines the MC seeds of each simulation.
5. `export_sim_info!` writes one `means_db` file per simulation and sets its
   status to `done`.

### 6.3 The error bar of one seed (`statistics.jl`)

Successive MC measurements are correlated, so the naive standard error
σ/√N is too small. For every observable stream `x₁…x_N`:

1. **Integrated autocorrelation time** τ (`find_correlation_time`):
   τ = 1 + 2 Σₖ C(k)/C(0), summed up to the first M with τ ≤ M/5 (automatic
   windowing, `C = 5`). C(k) is the autocovariance at lag k (`chi_k`).
2. **Blocking** (`obs_std`, scheme `"correlation"`): the stream is cut into blocks
   of `10 τ` measurements (`tau_mult = 10`); the block means are close to
   independent. Error = standard error of the block means.
3. The reported error is `max(naive error, blocked error)`. If τ < 1 or there
   are fewer than 2 blocks, the naive error is used.

Each means entry is a Dict:

| Key | Meaning |
|---|---|
| `value` | mean over the N measurements |
| `error` | error bar from step 2–3 |
| `correlation_time` | τ (a vector for vector observables) |
| `num_uncorrelated_measurements` | ⌊N / τ⌋ |
| `num_measurements` | N |

Complex observables: the mean and error are of `|x|`; τ is the larger of the
real- and imaginary-part times.

### 6.4 Derived observables

Some quantities are nonlinear functions of several streams. They are computed
from **block means** with a common block size
(`max(N/32, τ_a, τ_b)`) so that correlations between the streams are kept:

| Observable | Formula | Error |
|---|---|---|
| `stiffness` | ρ_s = −¼ (⟨K_x⟩ + Λ_xx(q_y→0)), with Λ_xx(q_y→0) = 2 Λ_xx[1] − Λ_xx[2] from the first two entries of the measured `Lxx_qy` list (`extrapolate_Lxx0`) | block bootstrap, B = 1000 (`BT_stiffness`) |
| `stiffness_lowest_q` | same, with Λ_xx[1] only | block bootstrap |
| `stiffness_q0` | −¼ (⟨K_x⟩ + Λ_xx(q=0, ω=0)) | block bootstrap (`BT_stiffness_scalar`) |
| `stiffness_y` | as `stiffness`, y direction | block bootstrap |
| `Ps` | 2⟨DdD⟩ − ⟨n⟩ + 1 | from block variances and the DdD–n covariance |
| `S_Delta_Q0` | ⟨\|Δ_mean\|²⟩, pairing-field structure factor at q = 0 | as §6.3 |
| `drude_dc_*`, `drude_direct_*` | D = −π (⟨K⟩ + ⟨Λ_D⟩) (diagonal), −π⟨Λ_D⟩ (off-diagonal) | block covariance |
| `resistivity_dc_tensor`, `resistivity_tensor_fixed` | inverse of the 2×2 conductivity tensor | block bootstrap (`BT_resistivity_tensor`) |
| `conductivity_regular_*_fixed` | Im Λ(ω)/ω at each fixed broadening δ | as §6.3, per ω and δ |
| `*_extrap` | δ → 0 extrapolation of the fixed-δ observables | `extrapolate_delta_to_zero`: weighted linear fits on windows of δ, combined with DerSimonian–Laird |

### 6.5 Averaging MC seeds (`average_over_seeds`)

The seeds of one simulation are combined with the **DerSimonian–Laird**
random-effects estimator (`obs_seed_mean2(...; method="DL")`, `get_x_DL`):

- fixed-effect weights wᵢ = 1/σᵢ², weighted mean x̄_w (`get_x_GD`);
- between-seed variance τ² = max(0, (Q − (k−1)) / (Σw − Σw²/Σw)), with
  Q = Σ wᵢ (xᵢ − x̄_w)² (`get_y_DL`);
- final weights 1/(σᵢ² + τ²), mean and error 1/√Σ(weights).

So seeds that disagree by more than their error bars get a larger combined
error. If DL fails (all errors NaN, any error zero, NaN result) it falls back to
the plain mean and std/√k. One seed → its own value and error.

### 6.6 Averaging disorder samples (in the queries)

Disorder samples are separate simulations, so they are averaged **when you read**
(§7.3), with plain statistics over the samples:

- `value` = mean over samples
- `std` = sample-to-sample standard deviation (physical; it changes with L)
- `error` = std / √N_samples (uncertainty of the disorder average); with a
  single sample, that sample's MC error
- `stat_error` = √(Σ σᵢ²) / N, the propagated MC error

## 7. Reading results: `means_db`

### 7.1 Setup

```julia
include("means_db.jl")          # light: JLD2 only, loads in seconds
```

`means_db.jl` reads the databases listed in **`MEANS_DB_ROOTS`** (separator `;`).
Set it **before** the `include`:

```julia
ENV["MEANS_DB_ROOTS"] = "<Landau's means_db>;<Curie's means_db>"
include("means_db.jl")
```

Usually you set where each cluster's data is instead, and `MEANS_DB_ROOTS` follows:
each cluster's root is `<cluster folder>/MC_Hubbard_V7/means_db`, where the cluster
folder is `LANDAU_ROOT` / `CURIE_ROOT` if set, else the backup drive's if
`BACKUP_DRIVE` is set (§8.2), else `<LANDAU_MOUNT>/data` / `<CURIE_MOUNT>/data` if
the cluster is mounted (SFTP mounts of `/home/data/<user>`,
[FHRC_CLUSTERS.md §8](FHRC_CLUSTERS.md#8-reading-the-clusters-from-a-laptop)).
A cluster with none of these set is left out; with nothing set, the queries warn
that there is no database. Roots that do not exist are skipped.

Two things need the full engine instead (`include("sim_num_control.jl")`, which
loads MKL/Plots and takes about a minute): `get_stream`, and `get_mu` for old
simulations that did not store μ. They open the run's own folder, found from the
cluster path stored in the record (`seeds/<s>/info["dir_path"]`). Both clusters
use the same path strings, so `fix_cluster_path(path; cluster)` maps it to
**that cluster's** copy on this machine. Set these before the `include`:

| Env var | Default | Meaning |
|---|---|---|
| `LANDAU_ROOT`, `CURIE_ROOT` | (none) | a folder holding that cluster's `MC_Hubbard_V7/` and `MC_Hubbard_V6/` (e.g. a mount's `data` folder); looked up first (§8.2) |
| `BACKUP_DRIVE` | (none) | the drive's `Yaniv-Data` folder (§8.2); looked up **before** the mounts |
| `LANDAU_MOUNT`, `CURIE_MOUNT` | (none) | where each cluster's `/home/data/<user>` is mounted on this machine (e.g. a drive letter) |
| `CLUSTER_DATA_PREFIX` | two levels above `V7_DATA_ROOT` | the clusters' `/home/data/<user>`; only needed on a cluster, to map a mount path back. Stored paths are recognised by their `data/MC_Hubbard_Vx/` part, whoever ran them |

### 7.2 What is there

```julia
db_status()                         # per root: number of simulations by status (instant)
db_summary(; root=MEANS_DB_ROOTS[1])   # every queryable parameter and its values
rows = find_all(; U=4.0, Lx=20)     # matching index rows (Dicts) across all roots
```

### 7.3 Queries

```julia
# observable vs temperature (or any parameter: x=:U, x=:Lx, ...)
c = get_curve(; obs=:stiffness, x=:beta, U=3.0, Lx=8, V=1.0, wanted_n=0.875,
                alpha=1.0, beta_phi=0.0)
c.x, c.value, c.error, c.std, c.n_samples

# one point, disorder-averaged
r = get_observable(; obs=:stiffness, U=3.0, Lx=8, V=1.0, wanted_n=0.875, beta=20.0)
r.value, r.error, r.std, r.stat_error, r.n_samples, r.disorder_seeds, r.clusters

# the samples behind that point
b = get_observable(; obs=:stiffness, U=3.0, Lx=8, V=1.0, wanted_n=0.875, beta=20.0, seeds=true)
b.seeds, b.values, b.errors

# frequency-resolved observables (DOS, LDOS, conductivity_regular_*_fixed, ...)
s = get_spectrum(; obs=:conductivity_regular_xx_fixed, U=3.0, Lx=8, V=1.0,
                   wanted_n=0.875, beta=50.0)
s.w, s.deltas, s.value[i]           # i-th broadening δ

get_mu(; U=4.0, Lx=14, wanted_n=0.9, beta=10.0)        # the converged μ (an output)
get_param(:w_conductivity_range; U=3.0, Lx=8, beta=50.0)   # an array parameter

# raw MC time series of one chain (reads results.jld2; needs sim_num_control.jl)
st = get_stream(; obs=:DdD, U=3.0, Lx=12, beta=20.0, wanted_n=0.875)
```

`notebooks/read_archive.ipynb` has worked examples.

> **Pin every parameter you are not sweeping.** Anything left out is averaged
> over: leave out `wanted_n` and n = 0.5 is averaged with n = 0.875. The query
> warns and lists the variants it mixed (`r.varied`, `c.variants`);
> `strict=true` makes it an error. The exception is `disorder_seed`: leave it out
> to average the disorder samples, pin it to get one sample.

Selectors take a value, a vector of values (any of them), or `n=…` with
tolerance `n_tol` for the *measured* density. Only `done` simulations are returned
unless you pass `status=…`.

### 7.4 Inside one means_db file

```
params                  full parameter Dict
identity                the parameters that were hashed into the file name
meta                    bookkeeping (sim_num, convergence, ...)
summary                 status (submitted/running/done), cluster, n_measured, observable list
means/<obs>             seed-averaged means, one JLD2 dataset per observable
seeds/<s>/info          per seed: num_current_measurments, mu, convergence, status, dir_path
seeds/<s>/means/<obs>   per-seed means
```

One dataset per observable means a query reads only the observable it asks for.
The index (`index.jld2`/`index.csv`) is only a cache: `rebuild_index(; root)`
regenerates it from the files.

**Identity.** Only the parameters in `IDENTITY_PARAMS` are hashed, and a parameter
equal to its registered default is left out of the hash. So adding a new
parameter (with a default meaning "as before") does not change any existing hash.
`CONDITIONAL_IDENTITY` drops parameters that cannot matter (e.g. `TClustersize`
for a non-TCA method, `disorder_seed` for a clean system).

## 8. The backup drive

The raw `results.jld2` files are too large to keep. In October 2026 all of them
were reduced: every V7 campaign and the V6 campaigns used in the thesis. Each raw
file was replaced by a small **`results_slim.jld2`** (`slim_results.jl`, checked
by `verify_slims.jl`). The slims and the `means_db` of both clusters were copied
to an external drive, and the raw files, the checkpoints and the transfer stages
were then deleted from the clusters (V7 data: Curie 3.2 TB → 143 GB, Landau
3.4 TB → 125 GB). The transfer tools are specific to the FHRC clusters and are not
in the public repository; [FHRC_CLUSTERS.md §9](FHRC_CLUSTERS.md#9-big-transfers-and-deletes)
says how it was done.

| Where | What is there |
|---|---|
| **Dror's backup drive**, folder `Yaniv-Data` (an external disk labelled "Yosef: Curie") | the archive: slims, both `means_db`, the V6 `sim_info` files, the transfer records |
| each cluster, `/home/data/<user>/data/MC_Hubbard_V7` | its `means_db`, every unit's `sim.jld2`, `slurm.txt` and logs, and the slims in the seed folders |

So **the drive is the archive**. A slimmed seed can never be resumed
(`assert_not_slimmed`); more statistics for a parameter set means a new run with
another MC seed. A campaign submitted later goes through the same reduction once it
is finished.

### 8.1 Layout

```
Yaniv-Data/
├── curie/
│   ├── MC_Hubbard_V7/
│   │   ├── means_db/                     copy of Curie's means_db (index + one file per simulation)
│   │   └── data/<campaign>/<unit>/seed_<s>/results_slim.jld2
│   │                                     (only the slims; sim.jld2 stays on the cluster —
│   │                                      the slim holds the run's sim_data, means_db its params)
│   └── MC_Hubbard_V6/data/<campaign>/
│       ├── num_control/sim_info.jld2     V6 means (old, single-file format)
│       └── <unit>/sim.jld2, <unit>/seed_<s>/results_slim.jld2, means_cache.jld2
├── landau/                               same structure
└── _records/
    ├── <cluster>/<chunk>/                checksums, file lists (meta.tsv), slim_problems.txt
    └── *_dr_state_2026-10-25.tar.gz      the transfer tools' state (logs, chunk records); not needed for reading
```

Paths below the cluster folder are the same as on the cluster
(`/home/data/<user>/data/...`).

What is on it (checked 2026-10-25):

| Cluster | V7 slims | V7 means_db | V6 data |
|---|---|---|---|
| curie | 18.0 GB, 4,713 slims, 10 campaigns | 24.3 GB, 4,876 simulations | 22.5 GB, 12 campaigns (5,504 slims) |
| landau | 16.7 GB, 4,319 slims, 9 campaigns | 22.6 GB, 4,366 simulations | 15.5 GB, 24 campaigns (3,617 slims) |

- **V7:** every campaign is slimmed; `HighT__13-10-2026` ran on Curie only.
- **V7 seeds without a slim:** 11 seeds could not be slimmed. Their means_db
  records stay as they are; their raw files were not kept.
  - Curie: Mu_disorder__30-08-2026 units 1143, 1145; Phi_channel__18-08-2026 units 273, 285.
  - Landau: Phi_channel__08-08-2026 unit 217; Phi_channel__17-09-2026 units 599, 647;
    Phi_channel__18-08-2026 units 202, 229, 248, 293.

  `_records/<cluster>/<chunk>/slim_problems.txt` says why for each one.
- **V6:** the 15 campaigns used in the thesis have slims and `means_cache.jld2`.
  `mf_kspace_sweep_big__07-04-2026` has slims on Curie; on Landau it has only
  `sim_info.jld2` + `means_cache.jld2` (there were no raw results to slim). Landau
  also holds `num_control/sim_info.jld2` + `sim.jld2` (no slims) of 16 campaigns
  that were not used and were deleted on the cluster (NIL*, Mixed_gauge*,
  Mag__06-01, Mag_L12_U3, Mag_disorder, Disorder_small).
- **Not on the drive:** the V7 units' `sim.jld2` (each slim holds the run's
  `sim_data`, and means_db holds the parameters) and all checkpoints.

### 8.2 One setting: `BACKUP_DRIVE`

Set the drive's `Yaniv-Data` folder **before** including any of the code, once per
machine in `~/.julia/config/startup.jl` or in the first notebook cell:

```julia
ENV["BACKUP_DRIVE"] = "<where the drive is mounted>/Yaniv-Data"   # e.g. a drive letter on Windows
include("drive_data.jl")                      # loads means_db.jl too (JLD2 only, seconds)
```

With `BACKUP_DRIVE` set:
- **means_db queries read the drive.** `MEANS_DB_ROOTS` defaults to
  `<drive>/landau/MC_Hubbard_V7/means_db` and `<drive>/curie/MC_Hubbard_V7/means_db`.
  An explicit `ENV["MEANS_DB_ROOTS"]` still wins, and so does a cluster's own
  `LANDAU_ROOT` / `CURIE_ROOT` (below).
- **The engine looks on the drive first.** With `sim_num_control.jl` loaded,
  `fix_cluster_path` (used by `get_mu` when it falls back to a run's `sim_data`)
  looks on the drive before the mounts. A seed on the drive is never read over
  the network.
- **The drive is read-only to the code.** `save_entry`, `register_simulation`,
  `set_status!`, `rebuild_index`, `append_index!` and `update_means` throw
  `DriveReadOnly` for any path under the drive. Any path with a folder named
  `Yaniv-Data` in it counts too, even if `BACKUP_DRIVE` was not set. If a
  `means_db` `index.jld2` were missing on the drive, a query stops with that error
  instead of rebuilding the index there. To change anything, copy the folder to a
  local disk first.

The drive is a plain local disk (NTFS over USB), not a network mount, so reading
it is fast: a query over a whole campaign is fine on a laptop.

**Reading a cluster instead.** Each cluster keeps the same layout under
`/home/data/<user>/data` (means_db, slims, V6 `sim_info`), so any cluster can be
read from somewhere else. Set its folder before the `include`:

```julia
ENV["BACKUP_DRIVE"] = "<drive>/Yaniv-Data"   # Landau from the drive
ENV["CURIE_ROOT"]   = "<Curie's mount>/data" # Curie from its cluster mount
include("drive_data.jl")
cluster_root("curie")                        # where Curie is read from
```

Per cluster the first that applies wins: `LANDAU_ROOT` / `CURIE_ROOT`, then
`<BACKUP_DRIVE>/<cluster>`, then `<LANDAU_MOUNT>/data` / `<CURIE_MOUNT>/data`; with
none of them, that cluster is not read (there are no built-in locations). The queries,
`find_slims`, the V6 helpers and `fix_cluster_path` all follow it; `drive=` on a
helper forces the drive. `drive_check` and `drive_summary` read the drive only.
Over a cluster mount, keep to small reads (FHRC_CLUSTERS.md §8).

`notebooks/read_archive.ipynb` goes through all of §8.2–8.4 on one parameter set
and plots the V6 and V7 stiffness together.

### 8.3 V7: means and slims

**Means.** Every §7 query works unchanged on the drive (`get_curve`,
`get_observable`, `get_spectrum`, `find_all`, `db_status`, …):

```julia
db_status()
c = get_curve(; obs=:stiffness, x=:beta, U=4.0, Lx=20, wanted_n=0.875, V=0.0,
                alpha=1.0, beta_phi=0.0)
```

The exception is `get_stream`: the raw streams are gone, so it refuses slimmed
seeds. Use the binned streams of the slims instead.

**Slims**, found with the same selectors:

```julia
sl = find_slims(; U=4.0, Lx=10, beta=0.5, TClustersize=4, wanted_n=0.9)
# -> [(cluster, seed, dir_path, slim, exists, db_path), ...]   exists=false: no slim
s = read_slim(sl[1].slim; obs=[:n, :stiffness])   # obs=nothing reads everything
s.means["stiffness"]["value"], s.means["stiffness"]["error"]   # exact, == means_db
s.binned["n"]                     # meta["num_bins"] bin means of the stream
s.meta["bin_size"], s.meta["num_measurements"], s.sim_data["mu"]
slim_obs(sl[1].slim)              # names only, no data read
```

- `seed_slims(row)` does the same for one `find_all` row.
- `slim_path(dir_path; cluster)` maps one stored cluster path
  (`seeds/<s>/info["dir_path"]` of a means_db record, or a path on a cluster mount) to its slim.

**Inside a slim.** Each slim keeps the exact means (copied from means_db, not recomputed) and the
measurement stream averaged into 100 equal bins, one dataset per observable:

```
slim_format            "slim_v1"
sim_data               the run's parameters (Dict)
meta/num_measurements  N (full stream length)
meta/num_bins, meta/bin_size, meta/dropped_remainder
meta/means_source      "means_db" (V7) or "means_cache" (V6)
meta/tau_max           largest autocorrelation time of any observable
meta/bins_decorrelated bin_size >= 2*tau_max
meta/heavy_obs, meta/heavy_bins, meta/heavy_bin_size
means/<obs>            Dict(value, error, correlation_time, ...)  — exact
binned/<obs>           Vector of num_bins bin means
meanobs/BZ, A, LDOS    observables stored as one mean array
```

Large per-measurement observables (`meta/heavy_obs`, e.g. DOS, LDOS,
`Lxx_w_fixed`) keep fewer bins (`meta/heavy_bins`); each heavy bin is a whole
number of fine bins.

Without the helpers, a slim is a plain JLD2 file:

```julia
using JLD2
slim = sl[1].slim                       # or any path to a results_slim.jld2
jldopen(slim) do f
    N   = f["meta/num_measurements"]
    m   = f["means/stiffness"]          # Dict: m["value"], m["error"]
    dd  = f["binned/DdD"]               # 100 bin means of the DdD stream
    bs  = f["meta/bin_size"]            # measurements per bin
end
```

Linear functions of the stream can be re-derived from the bins. Error bars and
nonlinear observables must be taken from `means/`: re-deriving them from 100 bins
is only approximate (and only meaningful when `meta/bins_decorrelated` is true).

### 8.4 V6 data

V6 is the earlier version of this code (thesis data up to mid-2026). Its means are
one file per campaign, `num_control/sim_info.jld2`: a Dict
`sim_num => Dict(parameters..., "means" => Dict(obs => Dict(value, error, ...)),
"simulations" => per-seed data)`.

```julia
v6_campaigns()                                          # (cluster, campaign, dir, sim_info)
si = load_v6_sim_info("landau", "TCA_test__26-04-2026") # the campaign's sim_info Dict
for (k, sim) in si
    k == "updated_at" && continue
    println(k, "  U=", sim["U"], "  beta=", sim["beta"], "  rho_s=", sim["means"]["stiffness"]["value"])
end
v6_seeds("landau", "TCA_test__26-04-2026")              # (unit, seed, dir, slim, means_cache)
read_slim(v6_seeds("landau", "TCA_test__26-04-2026")[1].slim)   # meta["means_source"] == "means_cache"
```

`load_v6_sim_info` returns what `load_sim_info` (sim_num_control.jl) returns. The
V6 slims have the same format as §8.3, with `meta/means_source = "means_cache"`:
their means are the thesis-time values, copied verbatim.

> **The drive is not the thesis input for V6.** The drive has each campaign's
> *last* `sim_info.jld2` from the cluster. For the clean-system campaigns
> (FullSys_TCA ×2, Full_Sys_TCA, Full_TCA_Bigger) it is byte-identical to the
> copies the thesis figures were made from. The disorder campaigns (Disorder_*,
> DisorderII, Disorder_U, both clusters) were rewritten on the cluster on
> 2026-05-17 and 05-26, after those copies were taken, so they differ slightly.
> The thesis figures come from the earlier copies (via `means_store.jl`).

The lookup helpers in sim_num_control.jl (`get_sims`, `get_obs_vs_T`, ...) work on
such a `sim_info`. Two known quirks, kept so old notebooks give the same results:
- `disorder_type` means different things. In `get_sims` and `group_by_X` it keeps
  only points with `U_disorder == disorder_type`. In `get_sims_disorder`,
  `get_obs_vs_T(sim_info, ...)` and `get_mu_vs_T(sim_info, ...)` it is the *name of
  the parameter to group by* (e.g. `"impurity"`), and `V` picks the group by value.
- The default `diag_method` differs: `"exact"` in `group_by_X`, `group_disorder` and
  `get_obs_vs_T(sim_info, ...)`, `"K0"` in `get_sims` and `get_sims_disorder`,
  `"K0-TCA"` in `get_mu_vs_T(sim_info, ...)`. Pass it explicitly.

### 8.5 Checking the drive

```julia
drive_check()     # every archived raw seed vs. its slim; lists the seeds without one (~20 s)
drive_summary()   # GB / files / slims per cluster, version, campaign (stats ~70k files)
```

`drive_check` reads `_records/<cluster>/<chunk>/meta.tsv` and `slim_problems.txt`
and only checks that the files exist. `julia tests/test_drive_data.jl` tests the
helpers on a fake drive in a temporary folder; with `DRIVE_DATA_REAL` set to the
drive's `Yaniv-Data` folder it also opens about six real files, read-only.

## 9. Extending the code

**A new simulation parameter.**
1. Add the field to `SimData` (MC_Hubburd.jl) with a default that reproduces the
   old behaviour, and pass it through `running_simulation_wanted_occupation`.
2. Read it in `fh_run_n.jl` with `get(sim_data_dict, "<name>", <old default>)`, so
   old `sim.jld2` files still run.
3. If it changes the physics, add it to `IDENTITY_PARAMS` in means_db.jl **with
   the same default**. Existing hashes stay valid. If it only matters under some
   condition, add a rule to `CONDITIONAL_IDENTITY`. If it is bookkeeping only, add
   it to `NON_IDENTITY_FIELDS`.
4. Write it into `build_params` of your driver.

**A new observable.**
1. Add a field to `MeasureData` and initialize it in `Initial_measurments`.
2. Fill it in `make_measurement_K!`.
3. Add its symbol to `observable_list` in `get_means` (sim_num_control.jl), or
   compute it there if it is derived from other streams (§6.4).

Run the tests in `tests/` through `tests_pre_run.jl` before a campaign.

### Ideas not yet implemented

Open projects for whoever continues this code.

**Parallel TCA cluster updates.** In `K0-TCA`, one MC step updates the N sites one
after another (`MC_stepV2!`), and each update diagonalizes the Lc×Lc cluster around
its site (`localized_update`). This is serial, but many updates do not affect each
other: if site j lies outside the cluster of site i and i outside the cluster of j,
the two accept/reject decisions are independent. So the lattice could be split into
groups of sites that are far enough apart (more than about Lc/2 in x or y, like a
checkerboard with a larger cell). All sites of one group would be updated in
parallel threads. The step would wait until the group is done, then move to the next
group, so no site is updated while a cluster that contains it is being used.
What it needs:
- a fixed group order instead of the random site choice (`loop_random`);
- a separate random-number stream per thread, so runs stay reproducible;
- `localized_update`/`revert_pairing_field!` made safe to run at the same time on
  different sites (today they change the field in place and share work buffers);
- a check that the result is the same, statistically, as the serial sweep.

## 10. Troubleshooting

Cluster problems (depots, partitions, memory, scratch mode) are in
[FHRC_CLUSTERS.md §11](FHRC_CLUSTERS.md#11-troubleshooting).

| Symptom | Cause / fix |
|---|---|
| a query returns nothing | read its warning: it names the selector that matched nothing |
| a query mixes variants | pin the parameters it lists (§7.3) |
| means are older than the data | run num_control again (`LOOP_MODE=0` for one pass) |
| `MountIndexMissing` | the cluster mount's listing is stale; wait and retry. The index is never rebuilt over a mount |
| `EOFError` reading a mount | flaky network read; `means_db.jl` retries and falls back to a local copy |
| `DriveReadOnly` | something tried to write on the backup drive (index rebuild, `update_means`, …). The drive is read-only: copy the folder to a local disk and point the code there (§8.2) |
| `get_stream` refuses a seed | the seed was slimmed and its raw stream is gone; use `read_slim(...).binned` (§8.3) |
| "replacing module" warnings | `MC_Hubburd.jl` was included twice (e.g. `occupation_convergence.jl` and then `sim_num_control.jl`); harmless |

## Appendix A — parameters of one simulation (`sim.jld2`)

Written by the driver's `build_params`, read by `fh_run_n.jl`. **Id** = part of the
identity hash (default in brackets; a parameter at its default is not hashed).

| Key | Id | Meaning |
|---|---|---|
| `Lx`, `Ly` | ✔ | lattice size |
| `U` | ✔ | attractive interaction strength |
| `t` | ✔ [1.0] | nearest-neighbour hopping |
| `wanted_n` | ✔ | target density; μ is tuned to reach it (§5.2) |
| `beta` | ✔ | inverse temperature 1/T |
| `V` | ✔ [0.0] | potential disorder, Vᵢ ~ Uniform[−V/2, V/2] (`make_V_matrix`) |
| `disorder_seed` | ✔ [0] | which disorder sample; 0 = derived from the MC seed (old runs) |
| `seeds` | | MC seeds of this simulation (one per job) |
| `diag_method` | ✔ | `"K0"`: full BdG diagonalization for every update. `"K0-TCA"`: travelling cluster approximation, ΔS from a `TClustersize`² cluster around the updated site (falls back to `"K0"` if `TClustersize ≥ L`). `"exact"`, `"TCluster"`: the abandoned supercell method; do not use (`"TCluster"` stops with an error pointing to `"K0-TCA"`) |
| `TClustersize` | ✔ [0] | linear size of the travelling cluster (`"K0-TCA"` only) |
| `SC_L` | ✔ [1] | supercells: the lattice is repeated `SC_L` × `SC_L` times (`L·SC_L` sites per direction), with a supercell wavevector K. Abandoned (very low acceptance; many observables undefined): keep 1 |
| `B` | ✔ [0.0] | orbital magnetic field along z, in units of the flux through one plaquette; enters as Peierls phases (`neighbor_matrix`). With periodic boundaries the total flux is quantized: `assert_flux_quantization!` rounds B so that B·Ly (`"LandauX"`), B·Lx (`"LandauY"`) or B·Lx·Ly (`"Mixed"`) is a multiple of 2π |
| `gauge_choice` | ✔ ["Mixed"] | `"LandauX"`, `"LandauY"`, `"Mixed"` |
| `boundary_condition` | ✔ ["PBC"] | boundary conditions (only PBC is implemented) |
| `pairing_symmetry` | ✔ ["s-wave"] | `"s-wave"` (validated); `"d-wave"`, `"d-p-wave"`, `"mayr-d-wave"`, `"s+d-wave"` are experimental, never benchmarked (a warning is printed) |
| `guess` | ✔ ["uniform"] | initial pairing field: `"uniform"` or `"rand"` |
| `U_disorder` | ✔ ["uniform"] | site-dependent U_i (`make_U_params`, whose docstring has the formula of each model). The thesis uses `"impurity-gaussian-kernel"` (Ch. 6: Gaussian bumps of width σ around a fraction p of the sites, mean Ū = pU). `"puddles"` is the model of Sulangi et al. 2025 (background V0 plus Yukawa bumps). Also `"rand"`, `"ON-OFF"`, `"gaussian"`, `"on-off-puddles"`. Every map is drawn from the disorder seed; negative U_i are set to 0, so no site is repulsive |
| `U_puddle_params` | ✔ | shape parameters (V0, VI, λ, r_z): all four for `"puddles"`; entry 3 is σ for `"impurity-gaussian-kernel"` and the puddle radius for `"on-off-puddles"` |
| `impurity` | ✔ [0.0] | disorder strength; its meaning depends on the model (fraction p of centers, fraction of impurity sites, width of the distribution, covered fraction; see `make_U_params`) |
| `smooth_U` | ✔ | (smooth?, σ): Gaussian smoothing of the U map, `"gaussian"` model only |
| `alpha` | ✔ [1.0] | fraction of U in the pairing channel (penalty \|Δ\|²/(αU)) |
| `beta_phi` | ✔ [0.0] | fraction of U in the density channel (penalty φ²/(2β_φU)); 0 = channel off |
| `Qx`, `Qy` | ✔ [0.0] | ordering vector of the density coupling: φ couples as cos(Q·r)φᵢnᵢ |
| `mu_disorder` | ✔ ["none"] | μ–U disorder: `"none"`, `"interaction-only"`, `"coupled"`, `"potential-only"` |
| `mu_disorder_f` | ✔ [0.0] | fraction of sites with Uᵢ = 0 |
| `num_thermal` | ✔ [5000] | thermalization MC steps |
| `num_measure` | ✔ [20000] | number of measurements |
| `measuring_interval` | | MC steps between two measurements |
| `num_thermal_occ`, `num_measure_occ`, `measuring_interval_occ` | ✔ | the same, for each step of the μ search |
| `n_tol` | ✔ [0.001] | tolerance on ⟨n⟩ − wanted_n |
| `max_iter` | ✔ [100] | maximum μ-search steps |
| `num_uncorrelated_measurments` | | uncorrelated ⟨n⟩ samples collected at each trial μ of the μ search |
| `w_list` | ✔ | energies of the DOS/LDOS |
| `w_conductivity_range` | ✔ | frequencies of the optical conductivity |
| `delta_fixed_values` | ✔ | fixed broadenings δ of the conductivity (then extrapolated to δ→0) |
| `delta_values_stiffness` | ✔ | removed feature (stiffness at fixed broadenings); the drivers write `[]`. Kept so old data still loads |
| `dw_values` | ✔ | Δω windows of the DC conductivity |
| `cores` | | cores of the job (`CPUS_PER_TASK`) |
| `snapshot_interval` | | raw-field snapshot every this many measurements (0 = off) |
| `phi_hist_*`, `Delta_hist_*` | | field histograms: ranges and bin counts (0 bins = off) |

## Appendix B — file index

| File | Role |
|---|---|
| `examples/run_one_simulation.jl` | one small simulation on your computer, end to end |
| `examples/readme_stiffness_plot.jl` | the README figure (ρ_s vs T for several U) from the archived means |
| `notebooks/read_archive.ipynb` | worked example: `means_db` queries, slims, V6 + V7 from the backup drive or the clusters |
| `docs/FHRC_CLUSTERS.md` | the FHRC clusters: rules, scratch mode, preemption |
| `cluster_config.jl` | user/cluster settings |
| `pre_pre_run.jl` | runs any script as a SLURM pre_run job |
| `fh_pre_run_*.jl`, `fh_resubmit_campaign.jl` | submit drivers (§4.4) |
| `slurm_job_mode.jl`, `slurm_memory.jl`, `scratch_job.sh`, `scratch_fix_paths.jl` | job scripts, memory, scratch mode |
| `fh_run_n.jl` | one simulation job |
| `occupation_convergence.jl` | μ search + the run loop |
| `MC_Hubburd.jl` | the engine: lattice, Hamiltonian, MC updates, measurements, saving |
| `mf_kspace_density_channel.jl`, `dwave-kspaceMF.jl` | mean-field solvers (starting μ, normalisation) |
| `statistics.jl` | autocorrelation, blocking, seed averaging |
| `sim_num_control.jl` | num_control: per-seed means (`get_means`), seed averaging, export |
| `pre_run_num_control_V.jl`, `run_num_control_V.jl` | submit / run num_control |
| `means_db.jl` | the means database and all queries |
| `means_store.jl` | the old V6 means store (thesis figures from the V6 `sim_info` files) |
| `slim_results.jl`, `verify_slims.jl` | the slim file format: write and check `results_slim.jld2` |
| `drive_data.jl` | read the archive on the backup drive: slims, V6 `sim_info`, checks (§8) |
| `data_summery.jl`, `plotting_helpers*.jl`, `disorder_p_helpers.jl`, `simulation_obs_inspection.jl`, `simulation_statistics_report.jl` | analysis and plotting helpers (V6-style `sim_info` dicts) |
| `tests/`, `tests_pre_run.jl` | tests |
| `archive/` | superseded scripts, kept for reference |
