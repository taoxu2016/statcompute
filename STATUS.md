# Status

Updated 2026-11-23, version 0.25.0 (round 27: the Cox kernel adds up the same bits in a third of the memory and, in
the sandbox, 56 to 66 percent of the time).

## What exists

| Part | State | Evidence |
|---|---|---|
| Adapter contract and registry | Done | `tests/test_runner.py` |
| Synthetic data: simple and correlated designs, Cox and logistic outcomes | Done | `tests/test_synthetic.py`: identical checksums with 1 and 7 partitions; margins as specified |
| Lab kernels: weighted-Gram Cox (Breslow), profile-likelihood logistic | Done | `tests/test_kernels.py`: against explicit risk sets and a joint dummy-variable fit |
| `lab_numpy`, `lab_spark` prototypes | Done | `tests/test_references.py`: within T_COEF, T_VAR and T_LOGLIK of pprof_py 0.7.0 on five small datasets |
| `pprof_py` adapter | Done | Same tests |
| `pprof_r` adapter (the R package, through Rscript) | Done | Same file: logistic within the three classes of pprof_py; Cox within 1e-6 standard errors, the limit its fixed stopping rule guarantees |
| `pprof_spark` and `pprof_spark_cached` adapters | Done | Ran on Databricks on 2026-11-16 and agreed with pprof_py to 1e-15; the cached variant so far against a mock only |
| Runner, metrics, records, report, command line | Done | `tests/test_runner.py`; plans `smoke` and `laptop` run end to end |
| Round 2 safeguards: `running` checkpoint, scrubbed records, caveats, task metrics from the status store when the REST API is closed | Done | Three tests in `tests/test_runner.py` |
| Autoscaling support: `hold_executors` warm-up, executors sampled during each fit, caveats | Done locally | Three tests in `tests/test_runner.py`; untried on a real autoscaling cluster (OI-02) |
| Databricks notebooks | Run once by the maintainer | 2026-11-16 |
| Engine: `CoxPH` with strata or without (Breslow and Efron ties, entry times, weights, offsets) and `LogisticFE` (providers at the bound, provider effects with standard errors) | Specification 0.6, with the baseline hazard, robust variance, martingale residuals and standardized ratios for Cox, and standardized ratios with the score, Wald and exact provider tests for the logistic model. Version 0.9.0 ran on the cluster on 2026-11-18; this version has not | `docs/spec/engine.md`; the tests in `tests/test_engine_kernels.py`, `tests/test_engine_spark.py`, `tests/test_engine_time.py` and `tests/test_engine_outputs.py`, against explicit risk sets and pprof_py |

All of the above ran in one sandbox: Python 3.12, PySpark 4.0.1 in local mode, R 4.3.3, one core;
pprof_py 0.7.0 and pprof 2.0.12 installed from exported copies of their pinned commits. 26 tests
pass in about 80 seconds (29 since round 3, 66 since round 5, 96 since round 6, 102 since round 8, 103 since round 9, 104 since round 10, in about three minutes).
Rounds 12 to 15 ran in another sandbox of the same kind without R, with 4 GB of memory: 114 tests since round 13, 115
since round 14 117 since round 15 and 119 since round 16, of which 116 pass and the 3 that need R are skipped, in about five minutes.
Round 17 added none; round 18 has 120, of which 117 pass; round 19 has 122, of which 119 pass; round 20 has 125, of which 122 pass. Round 21 added none; round 22 has 126, of which 123 pass; round 23 has 130, of which 127 pass; round 24 has 132, of which 129 pass; round 25 has 134, of which 131 pass; round 26 adds none; round 27 has 143, of which 140 pass.

## First cluster run, 2026-11-16

Plan `compare_5m` (the maintainer's): 5 million rows, 50 and 100 features, on Runtime 17.3 with Photon, autoscaling
from 1 to 10 workers. Analysis in `docs/reports/2026-11-16-databricks-compare-5m.md`. In short:

- All four implementations that ran agree to rounding, pprof_spark included, whose Spark 4.1 JAR loads on 17.3.
- pprof_spark on 10 executors is 1.2 to 1.8 times faster than one pprof_py thread for Cox and 4 to 6 times slower
  for the logistic model. Its time does not follow the number of executors, and most of its CPU is outside its kernels.
- The Spark prototype is clearly faster for the logistic model. For Cox it is faster when both start from the stored
  table, and not when pprof_spark's input is cached first.
- The cluster's size was not controlled: 5 of 16 Spark runs had all 10 executors throughout.

## The engine on the cluster, 2026-11-18

The maintainer ran version 0.9.0 with plans `compare_5m` and `unstratified_5m` (5 million rows; Runtime 17.3 with
Photon; autoscaling from 1 to 10 workers). The numbers are the maintainer's; the records are not in the repository yet.

| Model, features | Whole call, s | Of which Newton's method, s |
|---|---|---|
| Cox with strata, 50 | 26.8 | 14.9 |
| Cox with strata, 100 | 24.1 | 13.9 |
| Logistic, 50 | 21.2 | 8.3 |
| Logistic, 100 | 22.1 | 9.5 |
| Cox without strata, 50 | 45.5 | 19.6 |
| Cox without strata, 100 | 60.6 | 34.2 |

- About half of every call was preparation. Without strata the fits used 8 to 10 of 40 cores.
- Without strata the engine and pprof_py differed by 2.5e-9 standard errors at 50 features and by 4.3e-5 at 100, the
  same at pprof_py's default and tightest tolerance. The cause is pprof_py's stopping rule, not the engine: its last
  Newton step is halved when rounding makes the log-likelihood come out lower (`docs/spec/engine.md`, section 5.3;
  OI-24). Shown with pprof_py itself in the sandbox at 100,000 rows; for the cluster's fit it is an inference from the
  regenerated data, where the engine's third step cut to 1/64 leaves the reported 5.54e-8 and 4.34e-5, to the three
  digits reported.

## Round 12, 2026-11-19: less preparation

What changed, all in `statcompute/engine`:

1. **Sorted once.** Spark sorts every partition in the shuffle that builds the working set; the kernels no longer
   sort, factorize or copy their rows in every pass (OI-21, closed).
2. **One job fewer.** A light job reads the columns other than the features: it counts the rows and refuses bad ones.
   The separate validation job is gone: the first pass over the working set builds it and counts rows with missing
   features. The row count is still taken, in that light job, because the number of partitions is still
   ceil(rows / `rows_per_partition`); that is what keeps every fit with strata identical to version 0.9.0.
3. **Without strata.** The same number of partitions as with strata instead of the input's own, the rows placed by row
   identifier and sorted by it; the distinct times come from the light job. `partitions="input"` keeps the old layout.
   With `row_id` such a fit now returns the same bits whatever the input's partitions; it costs a shuffle and a sort
   that this layout did not have (D-10, proposed).

What was measured, all in the sandbox (one core; PySpark 4.0.1 in local mode with two threads):

| Measurement | Result |
|---|---|
| Version 0.9.0 against this one, ten cases with strata or providers | Coefficients, covariance, log-likelihood and every output of the fit identical bit for bit; two sums in batch order differ by 1e-15 |
| Python side of one pass over 50,000 rows, outside Spark, 50 and 100 features | Cox with Breslow ties 218 to 177 ms and 406 to 375 ms; with Efron ties 280 to 243 and 563 to 493; logistic 57 to 57 and 218 to 153 |
| Whole call, 400,000 rows by 50 features, median of three: before, after | Cox with strata 13.7 s, 13.0 s; logistic 15.7, 15.6; Cox without strata 13.1 (the input's 2 partitions), 18.2 with the new default of 8 partitions, 12.8 with 2 partitions placed by row identifier, 11.6 with `partitions="input"` |
| Whole call, 400,000 rows by 100 features, median of three: before, after | Cox with strata 25.3 s, 21.0 s; logistic 27.8, 23.0; without strata not run at this size |
| Spark jobs and tasks of one fit, 400,000 by 50: before, after | Cox with strata 10 jobs and 78 tasks, 8 and 57; logistic 12 and 98, 10 and 77; Cox without strata 13 and 28, 12 and 97 (10 and 21 with `partitions="input"`) |

These times say little about the cluster. In this sandbox a Python task costs about 0.1 second whatever it does (OI-25),
so whole calls mostly count tasks: that is why the new default without strata is slower here, on one core where more
partitions cannot run side by side. What the changes do to preparation and to the use of 40 cores has to be measured
there. Two other layouts of the working set were tried and dropped because they moved the rows to Python no faster
here: one flat column per feature, and one packed block per partition.

## Round 13, 2026-11-20: what the repository is for

Written while round 12 was running on the cluster, without its result. Nothing in this round touches the engine's
passes.

The maintainer's decisions of 2026-11-20:

- **D-12.** pprof_py remains the statistical specification, but not unquestionable computational ground truth.
  statcompute validates independently and reports potential issues; changes of methodology are reviewed explicitly.
- **D-13.** Algorithms flow from statcompute to both pprof_py and pprof_spark; validation and benchmark evidence flow
  back. statcompute is to be a reusable platform for computational research and development.
- **D-14.** Bitwise reproducibility is not an absolute requirement: strict numerical tolerances, stable
  provider-level flags and auditable execution configurations may be sufficient.

What follows from them in this round:

| Part | State |
|---|---|
| Roles of the packages, and a last stage "handed over" for an algorithm | `docs/architecture.md`; the stage is proposed (D-17), with nothing handed over yet |
| Register of issues found in the packages this project reads | `docs/reference-issues.md`: eight entries, none reported yet. Four rest on the comparison of 2026-11-17, whose probes are not in this repository |
| Stable provider-level flags | Proposed as D-15, in `docs/spec/reproducibility.md`. `statcompute.validation.flags` compares two provider tables by the rule; the engine's logistic records carry a flag summary and the report says whether every run flagged the same providers. Nothing passes or fails by it |
| Auditable execution configuration | Proposed as D-16, same document. The harness now also records the Arrow batch size and the driver's BLAS. Missing: the libraries on the executors, and a record on fits made outside the harness (OI-29) |

Measured in the sandbox (probe A-10, `docs/research/flag-margins.md`; logistic model, 800,000 rows, 4,000 providers,
10 features): the provider test statistics differ by 5e-15 between two layouts of the engine, by 6e-13 between the
engine and pprof_py at its tightest tolerance, and by 9e-9 between pprof_py at 1e-5 and at 1e-12. No flag differs.
The provider nearest a cut-off is 1.5e-4 away for a z statistic and 3.7e-5 for the exact p-value.

## Round 12 on the cluster, and round 14 (2026-11-20)

The maintainer ran version 0.10.0 with both plans, two runs per configuration, and sent the two reports. The analysis
is `docs/reports/2026-11-20-databricks-round-12.md`. In short:

- **Confirmed:** one job fewer per fit (101 tasks fewer with strata or providers); without strata 14 to 18 effective
  cores instead of 7 to 12.
- **Not shown:** that any call is faster. Six of the eight calls with strata or providers took 49 to 76 seconds, against
  medians of 19 to 34 before, all of it outside the fit phase; the fit phase per evaluation was no shorter. Seven of
  the eight runs carry a caveat about the cluster's size, against two of twenty-two before, so the versions were not
  measured under the same conditions.
- **The stalls** of 30 seconds and more outside the fit phase are older than round 12 and larger than anything it
  changed (OI-31). What they are is not established.
- **Not compared:** round 12's estimates. The report compared each adapter's first run, a run of 0.9.0.

Round 14 changes the harness, not the engine:

| Part | State |
|---|---|
| The report reads a results folder version by version | Every run shows the version that made it; a table of medians by version, with and without the runs that carry a caveat |
| Where the time went | For every run of a Spark implementation: phases, the first evaluation beside a later one, the slowest task with its stage and that stage's median, spilled bytes, the executors seen during the fit. From the records that exist |
| Agreement of every run | One row per adapter and version, with the number of runs and the run furthest from the reference; before, only an adapter's first run was compared |
| The slowest task, recorded | From 0.12.0: its executor, how long that executor had been attached, the task's duration, garbage collection, wait for shuffle data and locality. By both routes to Spark's metrics; tried in local mode only |

## The second reading, and round 15 (2026-11-20)

The maintainer rendered the two results folders again with the report of round 14. Section 5 of
`docs/reports/2026-11-20-databricks-round-12.md` has the reading. In short:

- **Round 12's estimates agree with pprof_py on the cluster** exactly as those of 0.8.0 did: 8.0e-13 and 7.3e-13
  standard errors for Cox with strata at 50 and 100 features, 1.9e-10 for the logistic model. Without strata the known
  2.5e-9 and 4.3e-5, which are pprof_py's stopping rule.
- **The stall is located.** In all six stalled calls of round 12 the slowest task of the run is in the light job, the
  first job that reads the stored table: one task of 17 to 21 seconds in a stage whose median task takes 0.1 to 0.4.
  Earlier versions, `lab_spark` and pprof_spark have the same task wherever they first read the table. It is a property
  of the cluster and the stored table, not of a version (OI-31). The earlier labels were wrong in one respect: the runs
  called 0.9.0 were of 0.8.0 with strata, and of 0.8.0 and 0.9.0 without.
- **Net of the stalls round 12 is not faster.** Later evaluations take what evaluations took before. Without strata
  the new layout used more cores for the same seconds (OI-33).

Round 15 changes the harness only:

| Part | State |
|---|---|
| A read of the stored table before every timed Spark fit | Plan option `warm_input`: `"all"` or `"footers"`. Off unless a plan sets it; the record keeps the mode and the seconds. Proposed as D-18; untried on a cluster |
| The report | Every later evaluation's seconds instead of their median, which hid one slow evaluation; how much the slowest task read; whether the input was read first |
| Plans for the next runs | `compare_5m_warm`, `unstratified_5m_warm` and `unstratified_5m_warm_input`: the Spark implementations only, four repeats, with `warm_input`; the last with the rows left in the input's partitions |

## Round 16 (2026-11-20): the reference's halved steps, counted by the harness

The question left from round 12: was the difference of 4.34e-5 standard errors from pprof_py at 100 features without
strata a last Newton step halved six times, as the regenerated data say it should be (OI-24)? Instead of a separate
script the harness now answers it. The pprof_py adapter counts the calls a Cox fit makes to the package's own
likelihood, by a wrapper around that function while the fit runs; the package is not changed. A fit of n iterations
makes n + 2 calls when no step is halved, and one more per halved step. The report has a section for it, and plan
`reference_stop_5m` runs pprof_py alone on the two datasets.

Checked in the sandbox through the harness, on the two datasets of probe A-8 (100,000 rows by 100 features): seed 10
gives 19 calls and 14 halved steps, beside a difference of 3.49e-5 standard errors from the engine at both
tolerances; seed 79 gives 5 calls, none halved, and 6e-12.

## Round 17 (2026-11-21): decisions, and the warm runs

**Approved by the maintainer on 2026-11-21:** D-08 (the engine is Python on Spark), D-09 (agreement with pprof_py for
Cox allows for the Newton step that remains at its estimate), D-10 (the layout without strata), D-11 (what "prepare"
and "fit" mean), D-15 (stable provider-level flags), D-16 (the execution record), D-17 (the stage "handed over" and
the register of issues). Open: D-18, and the two proposals of this round, D-19 and D-20.

**The four reports of 2026-11-21**, read in `docs/reports/2026-11-21-databricks-warm-runs.md`:

- **pprof_py's halved steps are confirmed on the cluster.** At 100 features without strata its fit of 3 iterations made
  11 calls to its likelihood: 6 steps halved, as the regenerated data said. At 50 features 5 calls, none halved.
- **Reading the stored table before the clock takes the 17 to 21 second stall out of the timed call.** The light job
  took 0.6 to 1.8 seconds in all 24 Cox calls. What is left inside is the first Python task of a new executor, 6.5 to
  10.5 seconds in half of the calls (D-20).
- **Without strata the input's partitions beat the default of round 12 in every part of the call:** 17.7 and 18.5
  seconds at best against 29 and 37. D-10 was approved the same day and named this run as the test of its default,
  so D-19 proposes the input's partitions as the default; the code is unchanged until the maintainer rules.
- **Side by side, with the table read first,** the engine's whole calls are 1.4 to 2.5 times shorter than pprof_spark's
  by medians, in different sessions and against a build that is not pinned (OI-36), and 2.5 to 4 seconds longer than
  its own prototype's with strata (OI-35).
- **The same providers were flagged in all four runs** of both logistic datasets, by all three tests.

Round 17 changes little code: the two flag tolerances are named classes (T_FLAG_Z, T_FLAG_P), and an agreement row
says how many steps the reference halved.

## Round 18 (2026-11-21): the three decisions implemented

The maintainer approved D-18, D-19 and D-20 on 2026-11-21, and will benchmark pprof_spark at its current version
from now on, not at the old build.

| Part | State |
|---|---|
| Without strata, the input's partitions by default (D-19) | `partitions` unset or `"input"`: the rows stay as stored, no shuffle, no sort. A number, or `"rows"`, places them by row identifier and sorts them, which repeats bit for bit: the default of rounds 12 to 17. The configuration that is now the default is the one that ran on the cluster as `partitions="input"`: 17.7 and 18.5 seconds at best |
| The Python workers started before the clock (D-20) | Part of `warm_input`: after the read of the stored table, one short job whose tasks import NumPy, pandas, pyarrow and SciPy on every executor. Its seconds are in the record and the report (`python_started`). Untried on a cluster |
| The record of a fit (D-16) | Every model carries `execution`: versions, the driver's libraries and BLAS, the Arrow batch size, the layout. The executors' libraries come from the job that starts their Python workers, in the harness, or from `executor_libraries(spark)` |
| Plans | `unstratified_5m_warm` now names the layout by row identifier (`"engine_partitions": "rows"`), so that it keeps its meaning; `unstratified_5m_warm_input` names the input's |
| pprof_spark's pin | `5db844e` (round 176), the head of its repository on 2026-11-21. Its capability notes were read 25 commits earlier and are due for another reading (OI-37) |

Not done in this round: the Newton step that remains at pprof_py's estimate, in the harness (D-09, OI-34).

## Round 19 (2026-11-21): one session with every implementation

The maintainer rebuilt pprof_spark at its current version and ran `compare_5m_warm` with version 0.16.0: the engine,
its prototype, pprof_spark and `pprof_spark_cached` in one session, four repeats each
(`docs/reports/2026-11-21-databricks-warm-runs.md`, section 8).

| Model, features | engine: median, best (s) | pprof_spark | pprof_spark over engine, by medians |
|---|---|---|---:|
| Cox with strata, 50 | 19.7, 18.6 | 34.5, 25.4 | 1.8 |
| Cox with strata, 100 | 19.8, 16.6 | 50.4, 49.5 | 2.5 |
| Logistic, 50 | 15.0, 13.4 | 47.7, 34.8 | 3.2 |
| Logistic, 100 | 19.0, 16.9 | 86.5, 68.8 | 4.6 |

- **The build that ran was not the pinned one**, so every pprof_spark record carries that caveat. Round 19 pins the
  head of pprof_spark's repository as it is now (`b92919c`, round 177); the report lists every build's commit and judges
  the pin on the day it is made, so rendering it again settles this without a new fit (OI-36).
- **The Python start before the clock halves what a new executor costs a call, and does not remove it:** the slowest
  task was 4.0 to 7.3 seconds in 6 of 16 calls, against 6.5 to 9.5 in 7 of 16 before (OI-31).
- **The engine is still behind its own prototype for Cox with strata**, 19.7 against 14.1 seconds at 50 features. Part
  of it is found: the engine's kernel centres the features and takes 27 to 39 percent longer (probe A-12). And a pass
  costs more for its tasks than for its rows: in the sandbox, receiving the rows is half of a pass or more and hardly
  depends on how much is received (probe A-11; `docs/research/pass-cost.md`).

What round 19 adds for the next run: adapters `engine_125k` and `engine_250k`, the engine at 125,000 and 250,000 rows per
partition, and plan `engine_partitions_5m_warm`, which runs them beside the default in one session (OI-25).

RI-1, the report for pprof_py's maintainers drafted in `docs/reference-reports/`, was checked against pprof_py's
repository on 2026-11-21: its head is still the commit the draft cites, and the loop is unchanged.

## Round 20 (2026-11-21): the pin confirmed; no more restarts for a changed package

**The comparison with pprof_spark stands.** The report rendered with round 19 names the builds: the session of version
0.16.0 ran pprof_spark at `b92919c`, the pinned commit, and the earlier session at `ada822d` (round 163). In one session,
by medians of four, the engine's whole calls are 1.8 to 4.6 times shorter than pprof_spark's; 1.7 to 4.5 times leaving
out the runs that carry a caveat (`docs/reports/2026-11-21-databricks-warm-runs.md`, sections 8 and 9; OI-36 closed).

**The plan for the size of partitions measured nothing.** It was started with version 0.17.0 on a cluster that had run
0.16.0, and every fit was refused within three seconds: the executors still had the earlier version's archive. The
assistant's instructions for that run had left out the restart.

**Round 20 removes the need for it.** The package's code now travels inside the function of every task: PySpark's
pickler is told to send this package by value (`statcompute/engine/dist.py`). The executors need no copy of the
package and cannot run an older one than the driver; after pulling a new version it is enough to run the notebook
again. Tested in the sandbox with executors whose own copy of the package is stale and refuses to import: the fits
run by value and fail by reference (`tests/test_ship.py`). A task's function weighs 13 to 28 kB that way, against
3 kB, and carries the kernels' functions but no module object, whose loader and path would belong to the driver's
machine. The archive is kept as the fallback, with its guard. Every model's `execution` says how its code was sent.

## Round 21 (2026-11-21): the cluster's answers on round 20 and on the size of partitions

The maintainer ran `engine_partitions_5m_warm` with version 0.18.0 on the cluster as it stood, without a restart
(`docs/reports/2026-11-21-databricks-partition-size.md`).

- **The package sent by value works on Databricks.** The cluster still held the archive of 0.16.0 and had refused every
  fit of 0.17.0; all 48 fits of 0.18.0 completed. A changed package needs no restart any more (OI-38 closed).
- **One task per core is the fastest of the three sizes tried.** Later evaluations, medians at a steady ten executors:

| Model, features | 100 tasks per pass (the default) | 40 tasks | 20 tasks |
|---|---:|---:|---:|
| Cox with strata, 50 | 2.5 s | 1.8 s | 2.2 s |
| Cox with strata, 100 | 3.6 s | 3.0 s | 3.6 s |
| Logistic, 50 | 1.6 s | 1.3 s | 1.5 s |
| Logistic, 100 | 2.4 s | 1.9 s | 2.2 s |

  Whole calls by medians of four: 12.8, 21.1, 15.5 and 17.8 seconds with 40 tasks, against 20.1, 22.4, 17.8 and 24.1
  with 100. The flags are the same with all three.
- **D-21 proposes the consequence:** one partition per core by default, more only when a partition would exceed 200 MB
  of features; an explicit `rows_per_partition` or `partitions` keeps the layout independent of the cluster. Not
  implemented until the maintainer rules.
- **What is left of an evaluation:** a second or more that moved with neither the tasks nor their rows (OI-40).

Round 21 changes one thing in the code: the report's list of builds says how each record's code reached the executors.

## Round 22 (2026-11-21): D-21 approved and implemented

The maintainer approved D-21 on 2026-11-21.

| Part | State |
|---|---|
| The default number of partitions with strata or providers | One for every core the cluster has when the fit starts; more only when a partition would otherwise hold over 200 MB of features, and then as many as that limit asks for. At 5 million rows on 40 cores: 40, which is what ran on the cluster as `engine_125k` |
| A layout that does not depend on the cluster | As before, by `partitions` or `rows_per_partition`. With `row_id` such fits repeat bit for bit on any cluster; with the default they do between fits on the same number of cores, and agree to rounding otherwise |
| What a fit records | The rule that gave the number of partitions ("one per core", "by size", "rows per partition", "given") and the cores, in the model's settings and its `execution` |
| Adapters and plans | `engine_50k` is the default of versions up to 0.19.0, kept for comparison. `engine_partitions_5m_warm` runs the default beside 50,000, 125,000 and 250,000 rows per partition. In `compare_5m_warm` the prototype now has 125,000 rows per partition, so that it too has one task per core at 5 million rows |
| Without strata | Unchanged: the input's partitions (D-19). Whether a task per core pays there too is OI-39 |

Not run on a cluster as the default. Untried there as well: partitions of 200 MB on every core of an executor at
once, which larger data will give (OI-41).

## Round 23 (2026-11-22): the default on the cluster, and a harness that sees the cluster it measures on

The maintainer ran `compare_5m_warm` with version 0.20.0: the engine with the default of D-21, its prototype at one
task per core, pprof_spark and `pprof_spark_cached`, 64 fits in one session
(`docs/reports/2026-11-22-databricks-default-partitions.md`). The report arrived without its middle, so the task
counts and the caveats of this session have not been read (OI-44); everything below is from the 64 rows that survived.

- **The default gave one partition per core at the start in all 16 engine fits**: 40 on ten executors, 24 in the three
  fits that ran on six.
- **An evaluation takes what it took with 40 tasks before.** Later evaluations at a steady ten executors, medians: 2.0,
  3.2, 1.3 and 1.9 seconds (Cox at 50 and 100 features, logistic at 50 and 100), 11 to 21 percent below the 100 tasks
  of the earlier default.
- **Whole calls were not shorter than in the earlier sessions**: medians of 18.4, 22.0, 17.4 and 22.0 seconds, best
  calls of 13.3, 18.5, 12.9 and 17.2. Two sessions with the same layout differ by up to 5.6 seconds in the median, so
  whole calls are compared inside a session only (OI-32).
- **Against pprof_spark** the engine's whole call is 1.9 to 4.5 times shorter by the medians, 2.0 to 4.4 by the best.
- **Against its prototype, on equal terms,** the engine is 3.5 to 5.5 seconds behind by the best call. For the logistic
  model an evaluation costs the same in both, and the engine's call is longer mostly by the pass for the provider
  effects, which the prototype does not have. For Cox the engine's evaluation is 0.3 and 0.6 seconds longer, about a
  fifth (OI-35).
- **Seven of the 64 fits ran on six executors and nothing flagged them.** The harness had counted ten; the cluster
  lost four in the seconds before the fit. The fit after such a one starts on four new executors and takes roughly 4
  to 7 seconds longer; the engine, which runs first on every dataset, met that in 6 of its 12 calls at a steady ten
  (OI-43, OI-31).
- **The same providers are flagged** in all 12 runs of each logistic dataset, at 100, 40 and 24 partitions.

What round 23 changes, none of it in the engine's fits:

| Part | State |
|---|---|
| The cluster's size before a fit | Counted again after the reads, just before the fit, and recorded. A short count is flagged, and so is a fit that ran on fewer executors than its plan holds; the report gives that caveat to the earlier records too, from what they measured. A plan can ask again (`hold_again`). Whether every plan should is decision D-22, proposed |
| The order of a plan's runs | With `interleave` the adapters take turns, repeat by repeat |
| The report | For every Spark fit the working set's partitions, the rule that gave them, the cores at the start and what the warm-up saw. `versions` narrows a report to one session. Notebook `03_report` shows the whole text in a box to copy and no longer leaves an empty file when rendering fails |
| Without strata (OI-39) | Adapters `engine_cores` and `engine_cores_hash`: one partition per core, the rows placed by row identifier and sorted, or by a hash and not sorted. Plan `unstratified_cores_5m_warm` runs them beside the default in one session. In the sandbox the three agree with pprof_py within the tolerance classes. Not run on a cluster |
| Handoff candidates | `docs/handover/README.md`: four for pprof_py and five for pprof_spark, each with its evidence and what is missing |

Tried in the sandbox only (one core, local mode, mocked counts of executors): the count before the fit, asking again,
the interleaved order. The box in `03_report` has not been seen on Databricks.

## Round 24 (2026-11-22): the whole report, three layouts without strata, and D-22

The maintainer applied round 23, rendered the session of `compare_5m_warm` again, whole, with pprof_spark as the
reference, ran `unstratified_cores_5m_warm` with version 0.21.0, and approved D-22.

**The session of the default, read whole** (`docs/reports/2026-11-22-databricks-default-partitions.md`, section 9; OI-44
closed):

- The tasks per fit are what the rule gives: 236 and 268 for Cox, 316 and 348 for the logistic model at 40 partitions;
  156, 188 and 204 at 24.
- The engine's coefficients agree with pprof_spark's, the pinned build, within 6.9e-13 standard errors for Cox and
  2.1e-10 for the logistic model in every run, those at 24 partitions included: at most 5e-4 of T_COEF.
- Before each of the eight fits that started on six executors the warm-up had reported "10 of 10 in 0 s". Getting four
  executors back took 44 to 127 seconds, 14 times in the session.
- Without the 13 fits that carry a caveat the engine's whole call is 1.7 to 4.7 times shorter than pprof_spark's and
  4.2 to 7.9 seconds longer than its prototype's, by medians.

**Without strata, three layouts in one session** (`docs/reports/2026-11-22-databricks-unstratified-layouts.md`; OI-39):

| Features | | The input's partitions (default) | One per core, sorted | One per core, by a hash |
|---|---|---:|---:|---:|
| 50 | A later evaluation, median | 3.9 s | 2.8 s | 2.7 s |
| 50 | Whole call, undisturbed | 17.4 s | 18.3 s | 14.6, 16.1, 14.6 s |
| 100 | A later evaluation, median | 5.35 s | 3.8 s | 3.95 s |
| 100 | Whole call, undisturbed | 26.5, 24.5, 23.2 s | 21.4, 21.9 s | none |

- One task per core makes an evaluation 26 to 31 percent shorter, at 100 features too, where the input already has 33
  partitions.
- The sort costs about 2 seconds when the working set is built and gives nothing back. Placing the rows by a hash
  builds as fast as leaving them where they are.
- At 100 features the hash layout had no undisturbed call, and the default was slower than on 2026-11-21 with the same
  layout: 5.0 to 6.0 seconds an evaluation against 3.8 to 4.2 (OI-46).
- **D-23 proposes the consequence:** without strata, one partition per core placed by a hash, by default. Not
  implemented until the maintainer rules.

**D-22, approved 2026-11-22, and its first run.** In `unstratified_cores_5m_warm`, which already asked again, one fit
still ran on six executors: the count taken after the reads said ten. Round 24 gives the rule to every plan that holds
the cluster and changes the count: a pause of three seconds after the reads, then the lowest of three counts (the
scheduler's block managers, the executors listed as active, the scheduler's cores over the cores of one executor),
all three recorded. Five other fits of that session lost executors during the fit, four of them in its first
evaluation; nothing before a fit can prevent that (OI-43).

Tried in the sandbox only: the pause and the three counts, with the counts mocked.

## Round 25 (2026-11-22): D-23 implemented, and the probe of what a pass costs

The maintainer applied round 24 and approved D-23. No cluster run since.

| Part | State |
|---|---|
| The default without strata (D-23) | `CoxPH` without strata and without `partitions` spreads the rows over one partition per core, placed by a hash of the row and not sorted, with a row identifier or without. `partitions="input"` gives the former default; a number or `"rows"` the layout placed by row identifier and sorted, as before. Specification 0.9. The model's settings and its `execution` say which rule gave the number, and the cores |
| Adapters and plans | `engine` is the default; `engine_input` the input's partitions; `engine_cores` the sorted layout. `unstratified_cores_5m_warm` runs these three in turn. The plans that name a layout keep it |
| The probe (OI-40, OI-35) | `statcompute/lab/passes.py`, adapter `engine_passes`, plan `pass_cost_5m_warm`: on the engine's own working set, eight kinds of pass, each five times in turn, the tasks timing themselves: a job of idle JVM tasks; Python tasks that receive a byte a row; that receive and stack the rows; that also run the engine's kernel with BLAS at one thread, or without that limit, or the prototype's Cox kernel; the engine's whole evaluation; and the kernel on every second partition. The report has a section for it. A probe's record carries no coefficients, so nothing is compared with it and its seconds are not among the medians |

In the sandbox (one core, local mode): the default without strata, the input's partitions and the sorted layout agree
within 1e-9 standard errors, and with pprof_py within the tolerance classes; the probe's evaluation returns the engine's
log-likelihood bit for bit, which is the test that it measures the engine's working set and not another.

## Round 26 (2026-11-23): what an evaluation is made of, and the two decisions on the cluster

The maintainer applied round 25 and ran both plans with version 0.23.0 in one session: `pass_cost_5m_warm` twice (16
probes) and `unstratified_cores_5m_warm` (24 fits). Both reports arrived whole. Round 26 records them and changes no
code.

**An evaluation, taken apart** (`docs/reports/2026-11-23-databricks-pass-cost.md`; ten probes at a steady ten
executors; OI-40 and OI-35 answered):

| Part of an evaluation, seconds | Cox, 50 | Cox, 100 | Logistic, 50 | Logistic, 100 |
|---|---:|---:|---:|---:|
| A job of Python tasks, whatever they do | 0.48 | 0.43 | 0.41 | 0.45 |
| Bringing the rows to Python | 0.48 | 0.78 | 0.45 | 0.77 |
| The kernel | 0.79 | 1.63 | 0.31 | 0.45 |
| The evaluation | 1.74 | 2.86 | 1.14 | 1.78 |

- The Spark job is 0.1 seconds of the first row; the rest of it is starting tasks in Python.
- Returning the sums and adding them costs nothing that shows.
- The engine's Cox kernel takes a quarter more than the prototype's inside a task (0.67 to 0.70 seconds against 0.54
  to 0.56 at 50 features; 1.41 to 1.50 against 1.13 to 1.16 at 100): the centring. That is most of what separates the
  two implementations' Cox evaluations.
- The limit of one BLAS thread changes nothing, and neither do two tasks on one physical core: closed.

**The default without strata** (D-23; `docs/reports/2026-11-23-databricks-unstratified-default.md`; OI-39 closed):
later evaluations a third shorter than the input's partitions at 50 features (2.7 against 4.05 seconds) and 6 percent
shorter at 100 (3.65 against 3.9); whole calls of 15.8 against 18.3 seconds at 50 features, 18.9 to 19.3 at 100. The 26
to 29 percent at 100 features on which D-23 was proposed did not return: the input's partitions were slow that day and
not since (OI-46, closed). `DECISIONS.md` says so under D-23.

**The count before a fit** (D-22 as round 24 amended it): in 40 runs none started on fewer than ten executors, and
eleven times the harness asked again. A loss during a run remains: 6 of 16 probes, 3 of 24 fits. The cluster gave up
four executors 70 to 90 seconds after the harness had last kept every core busy, eight times in a row (OI-43).

What follows for the main goal is in two new items: the Cox kernel (OI-47: 0.2 to 0.4 seconds of an evaluation are
centring, and the kernel makes five arrays the size of the features in every evaluation) and the rows (OI-48: 0.45 to
0.8 seconds of every evaluation bring rows to Python that have not changed).

## Round 27 (2026-11-23): the Cox kernel, the same sums in less memory

The maintainer chose the Cox kernel (OI-47) as the next work. No cluster run since round 26.

The probe had shown the kernel to be half of a Cox evaluation. Reading it for where the time could go: in every call
it made eleven arrays the size of the rows (the stratum means spread over the rows, the centred rows, the rows times
the risk scores, their running sums, the sums and the means at the event times and their copies, the rows times the
expected events), 5.3 times the rows at its peak. None of that is arithmetic.

| Part | State |
|---|---|
| `statcompute.engine.kernels.cox_partial` | Makes three such arrays and reuses them. Centres the rows stratum by stratum where strata are large (at once where they are small), and in the caller's own rows when the caller allows it (`overwrite`), which the Spark tasks do. Every floating-point operation is the one it was, in the same order |
| Evidence that the sums did not change | The kernel of version 0.24.0 is kept, word for word, in `statcompute/lab/cox_kernel_0_24.py`. Every output of the two is identical bit for bit: 6,720 comparisons in the sandbox over shapes, tie methods, weights, offsets, entry times, baseline, rows and robust, and 8 test cases in `tests/test_engine_kernels.py`. The tests against explicit risk sets and against pprof_py pass unchanged |
| Time and memory, sandbox | One partition of the cluster's shape (125,000 rows, 200 strata), one core, best of seven: 158 ms against 280 at 50 features, 348 against 526 at 100. Peak memory of a call 2.0 and 1.8 times the rows, against 5.4 and 5.3. The prototype's kernel, which does not centre: 226 and 393 ms |
| For the cluster | The probe has the kind `kernel_before`, the frozen kernel inside the same tasks, and a plan can name the kinds it runs (`pass_kinds`). Plan `cox_kernel_5m_warm`: the two Cox datasets, three repeats, five kinds |

Not done: the two matrix products in their symmetric form (5 to 9 percent of a call more in the sandbox) would change
the last bits of the information; nothing in this round changes any bit. The logistic kernel and the kernels of the
fit without strata were not looked at.

## What has not happened

- The kernel of version 0.25.0 has not run on a cluster: its time inside a task there is not known.
- Nothing has been done yet about the rows, which are a quarter to a half of an evaluation (OI-48).
- The layouts without strata were not compared in their coefficients on the cluster: the plan has no reference.
- No run at 180 features or above 5 million rows; the maintainer has set those aside for now.
- No run of pprof (R) on Databricks.

## Next

The repository is at https://github.com/taoxu2016/statcompute and decisions D-01 to D-07 are approved.

The maintainer's direction of 2026-11-16: work within autoscaling, no larger sizes for now, and the main goal first, which is the faster implementation.

1. Run `cox_kernel_5m_warm` on the cluster: 6 probes, about a minute each (OI-47). No restart is needed after pulling.
2. The rows (OI-48): prior art and a sandbox trial of not sending them again; then a proposal for a decision.
3. Send RI-1 to pprof_py's maintainers (D-12); the text is in `docs/reference-reports/`.
4. Read pprof_spark's Cox and logistic fits again at the pinned commit (OI-37).
5. The remaining Newton step in the harness (D-09, OI-34); the outputs the engine still lacks (OI-20); the open limits
   of D-21 (OI-41, OI-42).

Prior-art search done on 2026-11-17 (`docs/research/prior-art-time-grid.md`): no new estimator or identity; the algebra is
coxdev's; one arrangement for distributed data was not found elsewhere. The publication plan changes accordingly.
