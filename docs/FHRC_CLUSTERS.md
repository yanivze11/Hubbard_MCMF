# Running on the FHRC clusters (Landau, Curie)

How the code runs on the two SLURM clusters of the Fritz Haber Research Center
(Hebrew University), and the storage rules it has to follow. The general
pipeline (drivers, statistics, `means_db`) is in [WORKFLOW.md](WORKFLOW.md);
this file is what is specific to these clusters.

Contents

1. [The clusters and their disks](#1-the-clusters-and-their-disks)
2. [The rules](#2-the-rules)
3. [One-time setup](#3-one-time-setup)
4. [Submitting a campaign](#4-submitting-a-campaign)
5. [How a simulation job runs: scratch mode](#5-how-a-simulation-job-runs-scratch-mode)
6. [Preemption, requeue, resubmission](#6-preemption-requeue-resubmission)
7. [num_control on the cluster](#7-num_control-on-the-cluster)
8. [Reading the clusters from a laptop](#8-reading-the-clusters-from-a-laptop)
9. [Big transfers and deletes](#9-big-transfers-and-deletes)
10. [Checking a job](#10-checking-a-job)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. The clusters and their disks

| | Landau | Curie |
|---|---|---|
| Partitions for the simulations | `allq,infiniq` | `allq` (no `infiniq`) |
| Cores per simulation job | 1 | 2 |
| Excluded nodes | `landau59` (faulty), `landau61` (very slow) | `curie25`, `curie31` |
| Julia depot | `~/.julia_landau` | `~/.julia_curie` |

These values are in `CLUSTER_SETTINGS` in [`cluster_config.jl`](../cluster_config.jl).

| Disk | Where | Use |
|---|---|---|
| `/usr/people/<group>/<user>` | central NetApp, shared by **both** clusters, NFS on every node, heavily loaded | code checkout, Julia binary, depots. **Never written by a running job.** |
| `/home/data/<user>` | a disk array on each cluster's **master** (the login host), NFS to its ~100 nodes; **not backed up** | the data tree `V7_DATA_ROOT`, logs. Every file operation from a compute node is an NFS round-trip to the master. |
| `/home/scr/<user>` | **local** disk of each compute node (~1 TB), invisible to other nodes | where a running job writes |

`/home/data` has the same path on both clusters but different contents: each
cluster has its own data. A campaign is submitted on both, and `assign_cluster`
splits its simulations between them.

The login host of each cluster is its storage master: commands typed there
(`ls`, `du`, `tar`, …) touch the disk directly, without NFS.

## 2. The rules

These come from the FHRC staff (meeting 2026-10-19, "Cluster Etiquette" email
2026-10-20, earlier admin replies). Breaking them overloads the master; in
October 2026 Landau's master crashed from `/home/data` overload.

**Storage**
1. The cost is per **file operation** (open, create, list, delete), not per byte.
   Hundreds of jobs touching `/home/data` all land on the master.
2. A running job writes to node-local `/home/scr`, and copies its results to
   `/home/data` in few, large operations (§5).
3. Never write a small file every step to `/home/data`; never run a loop that
   polls `/home/data`.
4. At most a few thousand files per directory; combine small outputs.
5. Input read many times: copy it to `/home/scr` once.
6. No `du`/`find` over `/home/data` from compute nodes. Disk usage, on the
   master: `nice -n 19 ionice -c3 du -h --max-depth=2 /home/data/$USER | sort -h`.
7. Delete from the master, with one process (never parallel jobs). Before deleting
   hundreds of GB or millions of files, **tell the admins** so they can schedule it.
8. No symlinks between `/usr/people` and `/home/data`; no job output
   (`--output`, `--error`, any write) to `/usr/people`.

**Jobs**
9. Heavy work runs in SLURM jobs, not on the login node. Even a quick check is
   submitted, through `pre_pre_run.jl` (§4).
10. Partition lists have no spaces (`allq,infiniq`): after a malformed line,
    sbatch silently ignores every later `#SBATCH` line (`--exclude`, `--chdir`, …).

**Transfers and data reduction**
11. Large transfers (hundreds of GB) only before 08:00 or after 18:00. Our tools
    also count the weekend, Friday and Saturday, as off-peak (§9).
12. Data leaves the cluster as tars of 100–200 GB, to a university workstation
    first (never straight to an external drive), is verified there, and only then
    deleted on the cluster.
13. Slimming in SLURM jobs is allowed only in the approved form: a few jobs at a
    time, preferably 18:00–08:00, through `/home/scr` (≤ 1 TB per node), scratch
    removed even when a job fails.

## 3. One-time setup

```bash
# on a login node
cd /usr/people/<group>/<user>/Projects
git clone <repository> MC_Habburd_V7
```

Check `cluster_config.jl`: the data and scratch folders default to
`/home/data/<user>/data/MC_Hubbard_V7` and `/home/scr/<user>` (from `$USER`),
`JULIA_BIN` to the Julia you submit with; set the depots and the accounts for your
group. Update the checkout with `git pull`; never copy files into it by hand.

**Packages.** The jobs run `julia <script>` in the depot's default environment.
Install the packages of `Project.toml` there, once per cluster (each cluster
compiles for its own CPUs, so each needs its own depot):

```bash
export JULIA_DEPOT_PATH=~/.julia_landau        # ~/.julia_curie on Curie
julia -e 'import Pkg, TOML; Pkg.add(collect(keys(TOML.parsefile("Project.toml")["deps"])))'
```

Compilation is heavy; if the login node limits it, run the same line inside a
small SLURM job.

`Manifest.toml` in the repository records the package versions of the Curie
depot (Julia 1.12.1) at the time the V7 data were produced. It was made on the
login node with `julia --project=. -e 'using Pkg; Pkg.offline(true); Pkg.resolve()'`,
which only reads the installed packages. Pkg rewrites `Project.toml` in its own
format when it does this; restore it with `git checkout -- Project.toml`.

## 4. Submitting a campaign

Edit the sweep of a driver (WORKFLOW.md §4.1), then on **each** cluster:

```bash
cd /usr/people/<group>/<user>/Projects/MC_Habburd_V7 && git pull
DRY_RUN=1 julia pre_pre_run.jl fh_pre_run_V.jl     # what would be submitted
julia pre_pre_run.jl fh_pre_run_V.jl               # submit
```

`pre_pre_run.jl` runs on the login node for a second. It:

1. creates `<V7_DATA_ROOT>/sim_logs/fh_pre_run_V_<timestamp>/` with a copy of
   the driver (`config_snapshot.jl`) and `run_info.txt`;
2. detects the cluster from the hostname;
3. writes and sbatches `submit_me.txt`, the **pre_run job**, on the group queue
   (`PRIORITY_ACCOUNT` / `PRIORITY_PARTITION`; `--prerun-queue=general` uses the
   normal partitions instead). It exports the cluster settings (`JULIA_DEPOT_PATH`,
   `PARTITIONS`, `CPUS_PER_TASK`, `EXCLUDE_NODES`, `MEANS_DB_CLUSTER`,
   `V7_DATA_ROOT`, `JOB_MODE`) and runs the driver.

The driver then runs on a compute node and submits one job per simulation it
owns (WORKFLOW.md §4.3). Its output: `pre_run.out` / `pre_run.error` in that log
folder.

Any script runs the same way: `julia pre_pre_run.jl <script.jl> [args...]`
(arguments are passed on). `tests_pre_run.jl` submits the test files.

**Memory.** Each simulation job asks for `slurm_memory.jl`'s estimate for its
lattice size (4 GB up to L ≈ 24, 5 GB at L = 28, 8 GB at L = 32). At start the
job checks it fits (`check_memory_budget`) and stops early if not.

## 5. How a simulation job runs: scratch mode

`JOB_MODE` (env, default `scratch`, chosen at submission and written into each
`slurm.txt`) selects how a job touches the disks. `JOB_MODE=legacy` is the old
way back: the job runs directly in its `/home/data` folder.

In **scratch mode** the job runs `scratch_job.sh`, which follows rules 1–5:

1. **Stage-in.** Copy the unit folder (`sim.jld2`; after a requeue also
   `seed_<s>/checkpoint.jld2`, `results.jld2`, logs) to
   `/home/scr/<user>/mcjob_<jobid>/<unit>/`. Leftover scratch of dead jobs on the
   node (heartbeat older than 3 h) is removed first.
2. **Run** `fh_run_n.jl` there. Julia's output goes to `stdout.log` in scratch,
   not to the SLURM `.out` file on `/home/data`.
3. **Copy back once per hour** (`SYNC_INTERVAL`, default 3600 s), in ONE pass:
   every file that changed is copied to a hidden temporary name next to its target,
   then renamed into place, so `/home/data` always holds a complete earlier copy.
   A JLD2 file is copied only when Julia has not written it for 10 s
   (`QUIET_SEC`) and it did not change during the copy. Renames go
   `results.jld2` first and `checkpoint.jld2` last: a kill between two renames
   leaves an older checkpoint with newer results, which the resume handles.
4. **Finish.** A last pass, then `scratch_fix_paths.jl` rewrites the scratch paths
   recorded inside `sim.jld2` to the `/home/data` ones, and scratch is deleted.

While the job runs, the copied-back `sim.jld2` is named **`sim.scratch.jld2`**:
it records scratch paths, which num_control and the slim step must not see. The
unit's `sim.jld2` stays the submitter's version until step 4.

If the node has no usable `/home/scr`, the job runs in place like legacy mode
and says so in its `.out` file.

The hourly copy is the job's only regular write to `/home/data`, so a job
killed without warning loses at most the last hour.

## 6. Preemption, requeue, resubmission

`allq`/`infiniq` jobs are **preempted** by higher-priority jobs (PreemptMode
REQUEUE, no grace time: SIGTERM, then SIGKILL 30 s later). The group queues are
never preempted, but the simulations run on `allq`/`infiniq`. Preemption is
frequent, so every simulation job must survive it:

- On **SIGTERM** `scratch_job.sh` stops Julia, copies back the files that were
  complete before the signal (best effort, 30 s), removes scratch and exits.
- SLURM **requeues** the job (`--requeue`), possibly on another node. Its
  stage-in starts from the last copy on `/home/data`, and `fh_run_n.jl`
  **resumes** from `checkpoint.jld2` (the same Markov chain; measurements kept).
- **`--signal=B:USR1@900`** triggers an extra copy 15 min before a time limit.
- A job that ended for good (time limit, node failure) is resubmitted with
  ```bash
  julia pre_pre_run.jl fh_resubmit_campaign.jl --folder=<campaign> --dry-run
  julia pre_pre_run.jl fh_resubmit_campaign.jl --folder=<campaign>
  ```
  which re-sbatches the `slurm.txt` of every unfinished unit not in `squeue`.
  Each resumes from its checkpoint, in the mode it was submitted with.

Never switch a unit that already ran in one mode to the other by hand: a
scratch unit holds its live state in `sim.scratch.jld2`.

## 7. num_control on the cluster

```bash
FOLDER_NAME=<campaign> julia pre_run_num_control_V.jl
```

submits one job on the group queue (`run_num_control_V.jl`) that computes the
means of the campaign once and writes them into this cluster's `means_db`
(WORKFLOW.md §6). Run it on both clusters, after the campaign has finished.

> **Do not use `LOOP_MODE=1` here.** It turns the job into a monitor that
> re-scans the campaign every hour for up to a week, so the means stay fresh
> while jobs run. A loop polling `/home/data` is exactly what rule 3 forbids.
> (It was the default until 2026-10-22.)

`NUMCTL_MEM` (default `8GB` per CPU) must cover the largest `results.jld2` of
the campaign (an L = 20 file is ~370 MB in memory, more for larger L).

## 8. Reading the clusters from a laptop

The original setup mounted `/home/data/<user>` of both clusters over SFTP with
**RaiDrive** (one drive letter per cluster) and read `means_db` there. Tell the
code where each mount is before the `include`, e.g. in
`~/.julia/config/startup.jl`: `ENV["LANDAU_MOUNT"]` and `ENV["CURIE_MOUNT"]`
(the drive letters), or `LANDAU_ROOT` / `CURIE_ROOT`
([WORKFLOW.md §7.1](WORKFLOW.md#71-setup)).

Every campaign up to October 2026 is archived on the backup drive (§9), with a
full copy of both clusters' `means_db`. For those campaigns, read the drive
instead ([WORKFLOW.md §8.2](WORKFLOW.md#82-one-setting-backup_drive)): it is a
local disk and has none of the problems below. The mounts are only needed for
campaigns run after that.

- Keep RaiDrive at version **2025.7.16**: later versions break `realpath` on the
  mount and no file can be opened.
- Interactive queries (`get_curve` over one sweep) are fine: each opens a few
  small files on the master. Bulk reads over the mount (thousands of files)
  crashed the laptop twice; run those on the cluster.
- The index of a mounted `means_db` is never rebuilt from the laptop
  (`MountIndexMissing`): that would open every file over SFTP and write on the
  master. Rebuild it on the cluster.

## 9. Big transfers and deletes

Moving or deleting the raw data follows rules 7 and 11–13. The group's
data-reduction tools (specific to these clusters, and kept out of the public
repository) reduced all the data in October 2026, in two ways:

- **Tar flow** (V6, 2026-10-19/20). Chunks of ≤ 100 GB (at most 150) were tarred
  on the master at night and fetched to a university workstation. There each
  chunk was verified (sha256 and extraction in one read), slimmed with
  `slim_results.jl`, checked with `verify_slims.jl` and copied to the backup
  drive. The slims were then uploaded back next to the raw files, and only then
  were the raw files deleted.
- **Slim jobs** (V7, 2026-10-22 → 10-23, rule 13). No raw data left the cluster.
  One array job on the group queue, at most 4 tasks at a time, copied each chunk
  to `/home/scr` in one tar stream, slimmed and verified it there, and wrote back
  one tar of slims. The workstation fetched that tar, checked it and copied it to
  the drive. The slims were then installed on the cluster and the raw files
  deleted.

**Off-peak window.** The tools treat 18:00–07:50 on weekdays and the whole weekend
(Friday and Saturday, so Thursday 18:00 → Sunday 07:50) as off-peak. Anything over
5 GB, and every bulk delete, starts only in that window; `--wait` waits for it.
A delete of ≥ 100 k files or ≥ 200 GB in one command needs a note of which admin
agreed.

**Afterwards** the V6 and V7 checkpoints were deleted (V7 on 2026-10-25, about
41 GB per cluster), as well as the V6 campaigns not used in the thesis and the
transfer stages on the clusters. The V7 data went from 3.2 TB to 143 GB on Curie
and from 3.4 TB to 125 GB on Landau. The archive is the backup drive
([WORKFLOW.md §8](WORKFLOW.md#8-the-backup-drive)).

## 10. Checking a job

```bash
squeue -u $USER
scontrol show job <jobid>        # WorkDir, ExcNodeList, Partition must match slurm.txt
tail <unit>/<jobid>_1.out        # scratch mode: start / copy / end lines
```

In scratch mode the progress log (`simulation_log.txt`) and Julia's output
(`stdout.log`, `scratch.log`) reach `/home/data` with each hourly copy. If
`ExcNodeList=(null)` although `slurm.txt` has `--exclude`, an earlier `#SBATCH`
line is malformed (rule 10).

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `JULIA_DEPOT_PATH is unset` | submit through `pre_pre_run.jl`, which exports it |
| jobs cannot find packages, or crash on CPU instructions | the other cluster's depot; each cluster has its own |
| `invalid partition specified` | Curie has no `infiniq` |
| a job is OOM-killed | `slurm_memory.jl` sets memory by L; `fh_resubmit_campaign.jl` raises the request of old units |
| `sim.scratch.jld2` left in a unit after the job ended | `scratch_fix_paths.jl` failed; run `julia scratch_fix_paths.jl <unit>` |
| the unit's `.out` says "running IN PLACE" | no `/home/scr` on that node; the job ran in legacy mode |
| means are older than the data | run num_control again (`LOOP_MODE=0`) |
