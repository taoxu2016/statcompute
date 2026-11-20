# Status

Updated 2026-11-20, version 0.11.0 (round 13: the maintainer's decisions on what this repository is for, and a
proposal for what stands in place of bitwise reproducibility).

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
Rounds 12 and 13 ran in another sandbox of the same kind without R, with 4 GB of memory: 114 tests since round 13, of which
111 pass and the 3 that need R are skipped, in about five minutes.

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

## What has not happened

- No run of this version of the engine on Databricks; version 0.9.0 ran on 2026-11-18.
- No run at 180 features or above 5 million rows; the maintainer has set those aside for now.
- No run of pprof (R) on Databricks.

## Next

The repository is at https://github.com/taoxu2016/statcompute and decisions D-01 to D-07 are approved.

The maintainer's direction of 2026-11-16: work within autoscaling, no larger sizes for now, and the main goal first, which is the faster implementation.

1. Rerun plans `compare_5m` and `unstratified_5m` with this version (both hold 10 executors). Compare whole calls:
   for Cox "prepare" now contains the first evaluation (D-11), and `evaluation_seconds` in the records has each one.
   A third run of `unstratified_5m` with `"engine_partitions": "input"` in its options shows what the new layout
   without strata buys or costs.
2. On the cluster, count pprof_py's likelihood calls in the 100-feature fit without strata (OI-24).
3. Decisions D-08 to D-11 and D-15 to D-17 are proposed and open. With this round applied, the logistic records of the
   rerun also say whether every run flagged the same providers.
4. The outputs the engine still lacks (OI-20), in the order the maintainer needs them.

Prior-art search done on 2026-11-17 (`docs/research/prior-art-time-grid.md`): no new estimator or identity; the algebra is
coxdev's; one arrangement for distributed data was not found elsewhere. The publication plan changes accordingly.
