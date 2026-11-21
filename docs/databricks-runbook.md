# Running the benchmarks on Databricks

For the standard cluster: a 32 GB driver, 10 workers of 16 GB, Databricks Runtime 17.3 LTS (Apache
Spark 4.0.0). Do the stages in order. Each one is small, and each tells you whether the next is
worth starting. Nothing here has been run on Databricks yet, so expect the first pass to find
problems; send the error text back and it gets fixed.

## 0. Before you start

| Setting | Value | Why |
|---|---|---|
| Workers | 10. Fixed if you can: clear Enable autoscaling, or set the minimum equal to the maximum. Otherwise see "If the cluster autoscales" | A timing on a cluster that changes size measures the autoscaler as much as the code |
| Access mode | Dedicated (single user) | pprof_spark's Python wrappers need py4j |
| Runtime | 17.3 LTS | The target of this project (D-03) |
| Photon | As the cluster has it | The cluster as it is is the target. With Photon on, executors have a heap of 2.2 GB; records say so |

### If the cluster autoscales

Workers that arrive after a fit has started come without its cached data, and Databricks' optimized
autoscaling can also remove workers while a job is running. Either makes a timing unrepresentative.

1. Preferred: fix the size for the test window (10 workers, autoscaling off, or minimum = maximum).
   A compute policy may not let you; then:
2. Set the widget `hold_executors` to 10 in `02_run`. Before each Spark fit the runner submits rounds
   of sleeping tasks until 10 executors are attached, for at most 15 minutes, and only then starts
   the clock. It cannot stop the autoscaler from removing workers later.
3. Either way, every record says what the fit actually had: `environment.during` holds the smallest
   and largest executor count, the time-averaged cores and each change, and the report's
   `share_of_cores` is executor run time over wall time and average attached cores. A run whose
   count moved gets a caveat. Such a run is still a useful diagnostic (it shows directly how much of
   a slow fit is the cluster arriving late) but it is not a baseline timing.

None of this has been tried on a real autoscaling cluster yet (OI-02).

### Which driver

The plans are for the standard cluster with the 32 GB driver. The cluster with the 128 GB driver is
for the single-machine packages at sizes the small driver cannot hold: attach `02_run` to it with
plan `standard`, `adapters` = `pprof_py,pprof_r,lab_numpy`, `max_rows` = 5000000, and the same
`data_root` and `results_root`. The datasets are shared, so those runs are compared with the Spark
runs on identical tables. Nothing else depends on the larger driver.

### Code and folders

Add https://github.com/taoxu2016/statcompute as a Git folder in the workspace. Choose two folders:

| Widget | What | Example |
|---|---|---|
| `data_root` | Where Spark writes the datasets | A Volume, or `dbfs:/tmp/statcompute/data` |
| `results_root` | Where records are written with ordinary file calls | `/Volumes/...`, `/Workspace/Users/<you>/statcompute_results`, or `dbfs:/tmp/statcompute/results` |

The notebooks are in `notebooks/databricks/`. They put the repository on the Python path themselves.

**After pulling a new version of the repository, run the notebook again: nothing needs restarting.**
The notebooks import the package afresh on every run, and from version 0.18.0 the package's code
travels to the executors inside each task, so the executors always run what the driver runs. Before
that version the package went to the executors once as an archive, and a changed package needed a
restart of the cluster.

## Stage A. Environment check (two minutes)

Run `00_environment`. It fits nothing. It prints the cluster's facts, lists problems, and shows
which implementations are available. **Send back its printed output.** It is the first evidence of
what the cluster really is, and it tells me whether task metrics can be read.

## Stage B. Smoke plan (about ten minutes)

1. `01_generate` with `plan` = `smoke`.
2. `02_run` with `plan` = `smoke`.
3. `03_report`.

The data are tiny (20,000 rows), so timings mean nothing. The point is that every code path works
on Databricks: writing and reading datasets, the prototypes, the records. Expect `lab_numpy` and
`lab_spark` to say `ok` and the references you have not set up yet to say `unavailable`.

## Stage C. The references (each is optional; a missing one is recorded as unavailable)

| Reference | What to do | Note |
|---|---|---|
| pprof_py | Set the widget `install_pprof_py` to `yes`. The setup then installs numba, fast_poibin and pprof_py at its pinned commit on the driver, for that notebook | Needs outbound access to PyPI and github.com. Without numba pprof_py runs its pure-Python kernels; a record says so under `caveats` |
| pprof_spark | Build current main with pprof_spark's own `00_build` notebook, attach the three JARs to the cluster as its guide says, and restart. Run its `01_parity` notebook once on this runtime. Then set the widget `pprof_spark_build` to the build folder (the one holding `manifest.json`) | Building does not change that repository. Its JAR targets Spark 4.1; whether it loads on 17.3 is the first thing to learn (OI-03) |
| pprof (R) | In a `%sh` cell: `Rscript -e 'install.packages("remotes", repos = "https://cloud.r-project.org"); remotes::install_github("UM-KevinHe/pprof@d780cf50e11e4e5ff9008f14732e35cf632762f0", upgrade = "never")'` | Untested on Databricks (OI-11); it compiles C++ on the driver. Leave it for later if it resists |

Run `00_environment` again afterwards: its table shows what is now available and at which commit.

## Stage D. Parity plan (numerical agreement)

`01_generate`, `02_run` and `03_report` with `plan` = `parity`: 1 million rows, 8,000 providers, 50
features, both designs, every implementation at its tightest tolerance. The report's second table
shows each implementation's differences from pprof_py. This is the functional baseline: do the
packages agree on this cluster.

## Stage E. The baseline (roadmap step 0)

1. `01_generate` with `plan` = `standard` and `max_rows` = 5000000: four datasets of about 7 GB each
   before compression.
2. `02_run` with the same settings. Two repeats per implementation and dataset. The Spark
   implementations run first, back to back; pprof_spark runs twice, from the stored table and with
   its input cached first (`pprof_spark_cached`).
3. `03_report`. Compare the column `compare_s`.

At 180 features the single-machine packages do not fit a 32 GB driver next to Spark's own JVM, and
the memory guard skips them. To compare them at 5 million rows, run plan `width` (50, 100 and 180
features): they run at the widths that fit.

How long this takes is not known. Per pass, pprof_spark's Cox kernel needs about 260 CPU-seconds at
this size by the feasibility study's rates, so a healthy fit takes minutes; the earlier runs were
far from healthy, and finding out why is the purpose of this stage. A fit is cancelled after two
hours.

## Stage F. Scale

Repeat stage E with `max_rows` = 20000000, then 50000000, then 100000000.

| Rows | Table, uncompressed | Per worker if cached evenly on 10 | Expectation |
|---|---:|---:|---|
| 5 million | 7 GB | 0.7 GB | Fits in memory everywhere |
| 20 million | 29 GB | 2.9 GB | Cached working sets fit in memory |
| 50 million | 73 GB | 7.3 GB | Cached working sets exceed executor storage memory and go to disk; the records' spill and GC figures show the cost |
| 100 million | 146 GB | 14.6 GB | Runs from local disk if at all; this is the "where feasible" end of the plan |

These are arithmetic, not measurements. The two largest sizes are there to find the limits, and a
run that fails or times out is a result.


## What comes back

Copy `results_root/runs/` and `results_root/reports/` into `results/<date>-databricks/` in the
repository and push. Records hold timings, metrics, coefficients of the synthetic models and
environment facts (node types, runtime, Spark settings). They are scrubbed when written: the
cluster identifier is hashed, and e-mail addresses and user folders are removed.

## Reading a failure

| What you see | Meaning |
|---|---|
| Status `unavailable` | The implementation is not installed in this notebook's session; the note says what is missing |
| Status `skipped_memory` | The fit would not fit the driver; `memory_guard` has the two numbers |
| Status `failed` | The fit raised; the message and the end of the trace are in the record |
| Status `timeout` | Cancelled after the plan's `timeout_s` |
| Status `running` | The process died during that fit, usually the driver out of memory |
| An entry under `caveats` | The run completed but its timing is not a fair one; the entry says why |
| "the statcompute package has changed since it was sent to the executors" | A version before 0.18.0 (or the archive asked for with `STATCOMPUTE_SHIP=archive`) on a cluster that had run another version: restart the cluster, or move to 0.18.0, which does not need it |
