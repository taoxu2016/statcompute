# Status

Updated 2026-11-14, version 0.1.0 (the first round, in two sessions).

## What exists

| Part | State | Evidence |
|---|---|---|
| Adapter contract and registry | Done | `tests/test_runner.py` |
| Synthetic data: simple and correlated designs, Cox and logistic outcomes | Done | `tests/test_synthetic.py`: identical checksums with 1 and 7 partitions; margins as specified |
| Lab kernels: weighted-Gram Cox (Breslow), profile-likelihood logistic | Done | `tests/test_kernels.py`: against explicit risk sets and a joint dummy-variable fit |
| `lab_numpy`, `lab_spark` prototypes | Done | `tests/test_references.py`: within T_COEF, T_VAR and T_LOGLIK of pprof_py 0.7.0 on five small datasets |
| `pprof_py` adapter | Done | Same tests |
| `pprof_r` adapter (the R package, through Rscript) | Done | Same file: logistic within the three classes of pprof_py; Cox within 1e-6 standard errors, the limit its fixed stopping rule guarantees |
| `pprof_spark` adapter | Written; run against a mock only | `tests/test_runner.py`; no JARs in the sandbox (OI-04) |
| Runner, metrics, records, report, command line | Done | `tests/test_runner.py`; plans `smoke` and `laptop` run end to end |
| Databricks notebooks | Written; not run | They parse; no cluster was available |
| New engine | Not started | `engine/README.md` |

All of the above ran in one sandbox: Python 3.12, PySpark 4.0.1 in local mode, R 4.3.3, one core;
pprof_py 0.7.0 and pprof 2.0.12 installed from exported copies of their pinned commits. 23 tests
pass in about 95 seconds.

## What has not happened

- No run on Databricks. Every statement about the standard cluster is a plan.
- No run of pprof_spark from this harness.
- No engine code.

## Next

1. The maintainer creates the repository from the delivered bundle and confirms the cluster (OI-02).
2. `00_environment`, then `01_generate` and `02_run` at 5 million rows: the baseline (roadmap step 0).
3. After the baseline: decision D-04, then the engine's first step.
