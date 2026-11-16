# First cluster run: compare_5m on Databricks, 2026-11-16

Sources: the two tables the maintainer exported from notebook `03_report`
(`results/2026-11-16-databricks/`); the maintainer's description of the cluster; and, as a
cross-check, pprof_spark's own results page at its commit `7a14624`
(`docs/parity/scale/perf-results.md`), read only. The JSON records of these runs were not
available. Statements that need them say so.

## 1. Conditions

| Item | Value |
|---|---|
| Platform | Databricks on AWS, classic compute, dedicated access mode |
| Runtime | 17.3 LTS with Photon: Apache Spark 4.0.0, Scala 2.13, Python 3.12.3 |
| Driver | m5dn.2xlarge: 8 vCPUs, 32 GB, of which 28.3 GB visible to the driver process |
| Workers | m5dn.xlarge: 4 vCPUs, 16 GB each; autoscaling from 1 to 10; `hold_executors` = 10 |
| Executor heap | 2,218 MB. With Photon on this is a quarter of the 8,874 MB that Databricks' formula gives a 16 GiB worker without it [derived] |
| Spark settings | 200 shuffle partitions; adaptive execution on; ANSI on; driver result limit 8 GB |
| pprof_spark | Build `d0b9bd5` (round 151), built for Spark 4.1 and run on 4.0 |
| pprof_py | `94b5476`, with numba 0.59.1 |
| Data | Synthetic: 5 million rows, 8,000 providers, simple design, 50 and 100 features |
| Runs | Two per implementation and dataset: 30 completed, 1 skipped by the memory guard, 1 unavailable |

On the same day pprof_spark's own project decided (its D-76) to measure on this cluster as it is:
Photon on, autoscaling, heaps of 2.2 GB. This report takes the same view. Those are the production
conditions, not defects of the run.

## 2. Numerical agreement

Largest difference from pprof_py over the three datasets pprof_py ran on:

| Implementation | Coefficients | The same, in standard errors | Standard errors, relative | Log-likelihood, relative |
|---|---:|---:|---:|---:|
| pprof_spark | 9.0e-16 | 7.0e-13 | 3.8e-15 | 2.5e-15 |
| lab_spark | 3.0e-13 | 1.9e-10 | 6.4e-14 | 2.5e-15 |
| lab_numpy | 3.0e-13 | 1.9e-10 | 6.4e-14 | 2.6e-15 |

- **The four implementations return the same estimates at 5 million rows.** Every row is
  like-for-like, and every multiple of a tolerance class is below 0.0004.
- **The truth is recovered.** The largest |z| against the generating coefficients is 3.20, 2.87,
  1.88 and 3.22 on the four datasets, the same for every implementation.
- **pprof_spark's JAR, built for Spark 4.1, loads on Runtime 17.3** and reproduces pprof_py to
  rounding on these two models. That answers OI-03 for them.
- Logistic at 100 features has no comparison: the memory guard skipped pprof_py there.

## 3. Timings

Seconds to compare: for a Spark implementation the whole call from the stored table; for a
single-machine one the fit, with the data already in memory. Executors attached during the fit are
in brackets.

| Model | Features | pprof_py | lab_numpy | pprof_spark | lab_spark |
|---|---:|---|---|---|---|
| Cox | 50 | 87.6, 82.9 | 55.1, 60.3 | 96.7 (4), 70.9 (10) | 40.9 (6 to 10), 57.4 (10) |
| Cox | 100 | 202.3, 200.7 | 95.2, 85.4 | 144.4 (4 to 6), 114.1 (10) | 55.1 (3 to 10), 63.5 (4) |
| Logistic | 50 | 25.0, 23.4 | 21.5, 22.5 | 112.4 (2), 149.3 (9 to 10) | 26.7 (8), 127.6 (8 to 10) |
| Logistic | 100 | skipped | 49.1, 36.6 | 157.4 (5), 106.6 (5) | 56.5 (10), 16.4 (10) |

Inside lab_spark's calls, preparation took 7 to 41 s and the Newton passes 10 to 38 s, and once
93 s. pprof_spark's preparation is inside its fit and is not reported separately.

## 4. What the timings show

**4.1 The observation that started the project is reproduced, with records.** With all 10
executors attached, pprof_spark's Cox call took 71 s at 50 features and 114 s at 100. One pprof_py
thread fitted the same models in 83 to 88 s and 201 to 202 s. For the logistic model pprof_spark
was slower than that one thread: 112 to 149 s against 23 to 25 s at 50 features.

**4.2 pprof_spark's time does not follow the number of executors.**

| Fit | Executors | Seconds | Effective cores |
|---|---|---:|---:|
| Logistic, 50 features | 2 | 112 | 5.9 |
| Logistic, 50 features | 9 to 10 | 149 | 8.2 |
| Cox, 50 features | 4 | 97 | 9.5 |
| Cox, 50 features | 10 | 71 | 10.6 |
| Cox, 100 features | 4 to 6 | 144 | 13.4 |
| Cox, 100 features | 10 | 114 | 19.8 |

Effective cores are executor run time over wall time. With 40 cores attached the fits used 8 to 20
of them on average. At this size the fit is not limited by executor cores.

**4.3 Most of pprof_spark's CPU is outside its kernels** [derived].

| Fit | Executor CPU measured, s | Kernel passes at the feasibility study's rates, s | Ratio |
|---|---:|---:|---:|
| Cox, 50 features | 507 to 518 | 99 | 5.1 to 5.2 |
| Cox, 100 features | 1,593 to 1,614 | 352 | 4.5 to 4.6 |
| Logistic, 50 features | 522 to 693 | 98 | 5.3 to 7.1 |
| Logistic, 100 features | 1,366 to 1,448 | 335 | 4.1 to 4.3 |

The kernel figures scale the study's one-thread rates to these widths and iteration counts, and a
vCPU here is a hyperthread, so the ratios are rough. pprof_spark's own results page reaches the same
conclusion from its own harness at 180 features with a cached input: the kernels account for about a
third of the measured CPU. This is the roadmap's F-2: the cost that matters is the whole job, and
the study's table covered the passes only.

**4.4 This harness made pprof_spark read its input from storage each time, which its own notebooks
do not.** They cache the input first. pprof_spark's results page records, for the same build on the
same kind of cluster that day:

| Fit, 5 million rows, 100 features | pprof_spark's notebook, input cached | This run, from the stored table |
|---|---:|---:|
| Cox | 44 s (24 cores) | 98 and 142 s inside the call |
| Logistic | 98 s (24 cores) | 105 and 142 s inside the call |

So for Cox about half of what was measured here is the price of fitting from the stored table.
Neither way is wrong: a production job starts from a stored table, and caching 180 features of 20
million rows does not fit heaps of 2.2 GB. But a comparison has to show both. Round 4 adds the
adapter `pprof_spark_cached`, which caches first and counts the caching in its time.

**4.5 The prototypes.** Three statements, in falling order of strength.

- *Logistic:* at 100 features lab_spark's whole call took 16 to 57 s where pprof_spark took 107 to
  157 s here and 98 s in its own notebook; at 50 features 27 s (and 128 s once) against 112 to 149 s.
  On the driver alone, lab_numpy fitted in 22 s and 37 to 49 s, level with pprof_py and faster than
  the cluster runs of pprof_spark.
- *Cox, both from the stored table:* lab_spark's whole call took 41 to 63 s where pprof_spark's
  took 71 to 144 s, in two cases with fewer executors.
- *Cox against pprof_spark at its best:* with its input cached, pprof_spark fitted 100 features in
  44 s, the caching not counted. lab_spark's 55 to 63 s is not faster than that. At this width the
  prototype shows no clear gain for Cox.

**4.6 lab_spark has stragglers.** In three of its eight runs one task took 190 to 215 times the
median of its stage, and one logistic run took 128 s instead of 27 s. One of the three ran while
executors were arriving; the other two had 10 throughout. The records would say which stage.

## 5. What the timings do not show

- **The cluster's size was not controlled.** Of the 16 Spark runs, 5 had all 10 executors
  throughout, 7 started with fewer (2 to 8) although full size had been requested, and 5 saw the
  count change during the fit (one run is in both groups).
- **Two repeats, and other notebooks may have shared the cluster.** The spread between repeats is
  up to a factor of five.
- **Cores are not equal.** lab_numpy can use all 8 driver cores through BLAS; pprof_py uses one.
- **Widths of 50 and 100, not 180,** where the weighted-Gram form should matter most.
- **The prototypes do less.** Breslow ties, right-censored rows, no weights, no providers at the
  bound; and their results are not bitwise reproducible.

## 6. Defects this run found in the harness

| Defect | Evidence | Action in round 4 |
|---|---|---|
| `hold_executors` did not get the cluster to 10 before 7 of 16 Spark fits | The caveats column | Cause unknown without the records. The warm-up now prints its progress and stores the executor count after each round |
| Single-machine fits ran between Spark fits, and the cluster shrank under them | 1 to 4 executors during most lab_numpy and pprof_py runs | Every Spark implementation now runs first, back to back |
| `fit_s` was not comparable: pprof_spark's includes its preparation, lab_spark's does not | Section 3 | A column `compare_s` |
| pprof_spark was only given the stored table | Section 4.4 | Adapter `pprof_spark_cached` |
| The driver memory sampled for a run included what earlier fits had left | lab_numpy: 8.8 to 14.3 GB for a 4 GB matrix | Memory at the start of each run is recorded, and freed memory is returned between runs |
| Loading data for the single-machine packages took 35 to 85 s | Wall minus fit | Not counted in `compare_s`; unchanged |

## 7. Next

1. The JSON records of this run into `results/2026-11-16-databricks/runs/`.
2. Plan `width` (50, 100 and 180 features) with round 4 and `hold_executors` = 10.
3. The single-machine packages on the 128 GB driver for the same plan.
