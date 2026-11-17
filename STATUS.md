# Status

Updated 2026-11-17, version 0.6.0 (round 6: Cox regression without strata).

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
| Engine: `CoxPH` with strata or without (Breslow and Efron ties, entry times, weights, offsets) and `LogisticFE` (providers at the bound, provider effects with standard errors) | Version 0.2; not yet run on a cluster | `docs/spec/engine.md`; 67 tests in `tests/test_engine_kernels.py`, `tests/test_engine_spark.py` and `tests/test_engine_time.py` against explicit risk sets and pprof_py |

All of the above ran in one sandbox: Python 3.12, PySpark 4.0.1 in local mode, R 4.3.3, one core;
pprof_py 0.7.0 and pprof 2.0.12 installed from exported copies of their pinned commits. 26 tests
pass in about 80 seconds (29 since round 3, 66 since round 5, 96 since round 6, in about three minutes).

## First cluster run, 2026-11-16

Plan `compare_5m` (the maintainer's): 5 million rows, 50 and 100 features, on Runtime 17.3 with Photon, autoscaling
from 1 to 10 workers. Analysis in `docs/reports/2026-11-16-databricks-compare-5m.md`. In short:

- All four implementations that ran agree to rounding, pprof_spark included, whose Spark 4.1 JAR loads on 17.3.
- pprof_spark on 10 executors is 1.2 to 1.8 times faster than one pprof_py thread for Cox and 4 to 6 times slower
  for the logistic model. Its time does not follow the number of executors, and most of its CPU is outside its kernels.
- The Spark prototype is clearly faster for the logistic model. For Cox it is faster when both start from the stored
  table, and not when pprof_spark's input is cached first.
- The cluster's size was not controlled: 5 of 16 Spark runs had all 10 executors throughout.

## What has not happened

- No run of the engine on Databricks.
- No run at 180 features or above 5 million rows; the maintainer has set those aside for now.
- No run of pprof (R) on Databricks.

## Next

The repository is at https://github.com/taoxu2016/statcompute and decisions D-01 to D-07 are approved.

The maintainer's direction of 2026-11-16: work within autoscaling, no larger sizes for now, and the main goal first, which is the faster implementation.

1. Plans `compare_5m` and `unstratified_5m`: the engine's first cluster runs, with and without strata.
2. The outputs the engine still lacks (OI-20), in the order the maintainer needs them.

Prior-art search done on 2026-11-17 (`docs/research/prior-art-time-grid.md`): no new estimator or identity; the algebra is
coxdev's; one arrangement for distributed data was not found elsewhere. The publication plan changes accordingly.
