# Benchmark protocol

What is run, under which conditions, what is measured, and what may be concluded.

## 1. Conditions

| Condition | Requirement | Checked by |
|---|---|---|
| Cluster size | Fixed for the whole plan where that is possible. On an autoscaling cluster the option `hold_executors` asks for full size before each Spark fit | Notebook `00_environment`; every record samples the executors attached during its fit (`environment.during`) and sets `cluster_changed` and a caveat when the count moved |
| Standard cluster | As it is: 32 GB driver; up to 10 workers of 16 GB; Databricks Runtime 17.3 LTS with Photon, which leaves the executors a heap of 2.2 GB | Recorded in every record |
| Access mode | Dedicated (single user): pprof_spark's Python wrappers need py4j | `00_environment` |
| Data | The stored synthetic table named in the record, with its checksum | Manifest |
| Reference versions | The commits in `REFERENCES.lock`; a record says whether the session's build matches | `adapter.matches_pin` |
| Repeats | Two per configuration in the scale plans; the first includes JVM warm-up | Plan |

On the standard cluster the first read of a stored file by an executor stalls one task for 17 to
21 seconds, in whichever job reads the table first (`docs/reports/2026-11-20-databricks-round-12.md`,
section 5), and a new executor's first Python task takes 6.5 to 10.5 seconds. Timings that compare
implementations or versions are therefore taken with both out of the way (decisions D-18 and D-20,
approved 2026-11-21): a plan sets the option `warm_input`, and before every timed Spark fit the
stored table is read once (`"all"` reads every column, `"footers"` only counts the rows) and one
short Python task runs on every executor. The record keeps the mode, the seconds of the read, the
seconds of the Python start and the libraries the executors reported; the report shows the two
times. The plans named `..._warm` set it. A plan without it measures a first run on whatever
executors are attached, which is what the earlier results folders hold.

A run on a cluster that changed size is kept, flagged under `caveats`, and left out of timing
comparisons. The same list names anything else that would distort a timing: pprof_py running
without numba, a pprof_spark build other than the pinned commit, pprof_spark's JAR on a Spark line
it was not built for.

## 2. Datasets

`statcompute/testkit/synthetic.py`. Deterministic in (row identifier, seed, column), so independent
of the cluster and of partitioning. Written once; every implementation reads the same table.

| Design | Features | Purpose |
|---|---|---|
| simple | Independent: half indicators with prevalence 0.3, half uniform on a grid in [-1, 1] | Comparable with the earlier notebook runs; well conditioned |
| correlated | One shared factor with loading 0.5; indicators with prevalences 0.30, 0.10, 0.03 and 0.01; continuous features with standard deviations 0.1, 1, 10 and 50 | Closer to risk-adjustment designs; badly scaled on purpose |

Providers have equal sizes. Cox times are whole days over five years, so ties occur. Neither
design is calibrated to production data (OI-09).

## 3. Models and what "the same model" means

A plan asks every implementation for the same model through `ModelSpec`. Where an implementation
cannot fit it, the adapter says `unsupported`; where it fits a variant, `deviation`. Stopping rules
differ between correct implementations, so a plan chooses one of two modes.

| Mode | pprof (R) | pprof_py | pprof_spark | Lab prototypes |
|---|---|---|---|---|
| Cox, `default` | survival::coxph's defaults, which `fit_cox()` does not expose: relative change in the log partial likelihood below 1e-9; at most 20 iterations | The same rule and values | Its engine defaults: the same formula, tested on the full step first | Next Newton step below 1e-8 standard errors |
| Cox, `tight` | The same: it cannot be tightened | eps 1e-13; at most 50 iterations | eps 1e-13; at most 50 iterations | 1e-10 standard errors |
| Logistic, `default` | Largest change in beta below 1e-5; bound 10; backtracking; one thread | The same rule at 1e-8 | Its engine defaults, which are pprof_py's | Next Newton step below 1e-8 standard errors |
| Logistic, `tight` | tol 1e-12 | tol 1e-12 | tol 1e-12 | 1e-10 standard errors |

Timings use `default`: what a user gets from each package, different tolerances included. Numerical
comparisons use `tight`: converged estimates without stopping rules in the way. For both packages
that screen providers, the adapters turn screening off, so that every implementation fits every
provider.

## 4. Measurements

| Quantity | Source | Caveat |
|---|---|---|
| Wall time of the call | Driver clock around the adapter's `fit` | Includes the adapter's own data preparation; phases are reported separately where the adapter can separate them |
| The time to compare, `compare_s` | For a Spark implementation the whole call from the stored table; for a single-machine one its fit phase, with the data already in memory | pprof_spark's fit includes its preparation and lab_spark's does not, so their fit phases must not be compared |
| Phases | Inside the adapter: load, prepare, fit | Reference packages do not expose their internal phases |
| Tasks, executor run time, executor CPU time, GC time, shuffle and spill bytes | Spark UI REST API for the run's job group, or the status store behind it through py4j where the API is closed; `spark.source` says which | Executor CPU time excludes Python workers; for `lab_spark` use run time |
| Effective cores | Executor run time (or CPU time) divided by wall time | A measure of how much of the cluster a fit used |
| Spread of task times | Median, 95th percentile and maximum for the fifteen heaviest stages | The ratio of maximum to median shows stragglers |
| The slowest task | For the stage with the longest task: its executor, how long that executor had been attached when the task started, the task's duration, garbage collection, wait for shuffle data and locality (from version 0.12.0) | One task per run. It says what kind of stall a run had, not how many |
| Driver peak memory and CPU | Sampling the driver process, alone and with its children (an Rscript run; the JVM of a local session) | For single-machine implementations this is the whole cost; pprof (R) also reports R's own peak |
| Executor peak heap | REST API | Peak since the executor started, not per run (OI-10) |
| Iterations, convergence | The implementation's own report | Reported, never compared |
| Likelihood calls of pprof_py's Cox fit | A wrapper around the package's own likelihood function while the fit runs (from version 0.14.0); the package is not changed | Iterations + 2 when no step is halved; each call beyond that is a halved step, which can leave the estimate short of the root (`docs/spec/engine.md`, section 5.3) |
| Recovery of the truth | z = (estimate - generating coefficient) / standard error; largest absolute z | Above 4.5 deserves a look |

## 5. Comparisons

`statcompute/validation/compare.py` compares every completed run with the reference's first
completed run on the same dataset and model: coefficient differences in absolute terms and in
standard errors, relative differences of standard errors and of the log-likelihood, each also as a
multiple of its tolerance class. There is one row per adapter and version, with the number of runs
compared and the run furthest from the reference; until version 0.11.0 only the first run of an
adapter was compared, so a later round's estimates went unseen in a folder that held an earlier
round's. A row is `like_for_like` only if both adapters reported `exact`.

A results folder usually holds runs of several rounds. Every record carries the version of this
repository that made it, and the report reads the folder version by version: the runs with their
version, medians by version (with and without the runs that carry a caveat), and for every run of a
Spark implementation where its time went: phases, the first evaluation beside a later one, and the
slowest task with its stage.

Three things are never concluded from this harness:

- that an implementation is wrong because it differs from another that fits a different model;
- that a timing on a cluster that changed size, or on another runtime, describes the standard cluster;
- that a prototype's timing is the engine's.

## 6. Failures

A fit that raises, times out, is unsupported, or would not fit the driver is recorded with that
status and the reason. A record is built from nothing for each run, so it cannot carry values from
an earlier fit. It is first written with status `running`, just before the fit starts, and replaced
when the fit ends: a record still at `running` means the process died during that fit.

The memory guard compares the adapter's estimate with 40 percent of the driver node's memory (the
Spark driver's JVM shares the node), or with `driver_memory_bytes` in the plan's options, and
records both numbers.

Records are scrubbed when written, because the repository is public: the cluster identifier is
hashed, and e-mail addresses and user folders are removed from messages and traces.

pprof_spark reads its input several times while it prepares a fit, and its own notebooks cache the
input first. The plans therefore run it twice: `pprof_spark` is given the stored table, as a
production job would be, and `pprof_spark_cached` caches the table first and counts that time.

A plan runs every Spark implementation on every dataset first, then the single-machine ones. A
single-machine fit leaves the executors idle, and an autoscaling cluster shrinks under it.

## 7. Plans

| Plan | Content | Use |
|---|---|---|
| `smoke` | 20,000 rows, three datasets | Seconds; CI |
| `laptop` | 200,000 rows, 400 providers, 20 and 50 features | A few minutes on one machine |
| `parity` | 1 million rows, 8,000 providers, 50 features, both designs, tight tolerances | Numerical agreement at a size every implementation can run |
| `width` | 5 million rows, 8,000 providers, 50, 100 and 180 features | Cost against width |
| `standard` | 5, 20, 50 and 100 million rows, 8,000 providers, 180 features; the correlated design at 5 and 20 million | The scale test |
| `compare_5m` | 5 million rows, 8,000 providers, 50 and 100 features; Cox with strata and the logistic model; every implementation; two repeats | The maintainer's comparison on the standard cluster |
| `unstratified_5m` | The two Cox datasets of `compare_5m`, without strata; the engine, pprof_spark and pprof_py | The fit without strata on the cluster |
| `reference_stop_5m` | pprof_py alone on the two Cox datasets of `unstratified_5m`, two repeats | How many steps its Newton loop halved (OI-24). Needs no executors; about 18 GB on the driver at 100 features |
| `engine_partitions_5m_warm` | The engine at its default (one partition per core, D-21) and at 50,000, 125,000 and 250,000 rows per partition (adapters `engine`, `engine_50k`, `engine_125k`, `engine_250k`) on the datasets of `compare_5m`, four repeats, with `warm_input` | What a task costs on the cluster apart from its rows (OI-25). Ran on 2026-11-21: `docs/reports/2026-11-21-databricks-partition-size.md` |
| `compare_5m_warm` | `compare_5m` for the Spark implementations only, four repeats, with `warm_input` | Timings clear of an executor's first read of a file (D-18) |
| `unstratified_5m_warm`, `unstratified_5m_warm_input` | `unstratified_5m` for the engine only, four repeats, with `warm_input`; the second leaves the rows in the input's partitions | The two layouts without strata side by side (D-10) |

Single-machine implementations are skipped by the memory guard where they cannot fit. At 180
features pprof_py is at its ceiling near 5 million rows on a 32 GB driver, and pprof (R), which
holds more copies, below that; the 128 GB driver may be used to extend those comparisons, and
nothing else depends on it.

## 8. Adding an implementation

Write an adapter (about a hundred lines), register it, add its name to a plan. Add its row to
`docs/capabilities.md` from a reading of its source, and its commit to `REFERENCES.lock` if it is a
reference.
