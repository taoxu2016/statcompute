# Running the benchmarks on Databricks

For the standard cluster: a 32 GB driver, 10 workers of 16 GB, Databricks Runtime 17.3 LTS
(Apache Spark 4.0.0). The worker count and size are to be confirmed (OI-02).

## 1. Cluster

| Setting | Value | Why |
|---|---|---|
| Workers | A fixed number; autoscaling off | Timings are meaningless on a cluster that changes size |
| Access mode | Dedicated (single user) | pprof_spark's Python wrappers need py4j |
| Runtime | 17.3 LTS | The target of this project (D-03) |
| Photon | Off, or noted | pprof_spark runs typed Dataset code, which Photon does not execute |

## 2. Code

1. Create the repository on GitHub from the delivered archive and add it as a Git folder in the workspace.
2. The notebooks put the repository on the Python path themselves (`_setup`). Nothing to install for
   the harness and the prototypes: NumPy, pandas, PyArrow and psutil are in the runtime, and
   `lab_spark` ships the package to the executors.

## 3. The references (each is optional; a missing one is recorded as unavailable)

| Reference | How to make it available | Note |
|---|---|---|
| pprof_py | In a notebook cell before `%run ./_setup`: `%pip install --no-deps git+https://github.com/UM-KevinHe/pprof_py@94b5476398da262987b97ac6fec3bbb91b048731 numba fast_poibin` | Installs from its repository at the pinned commit; changes nothing there |
| pprof_spark | Its own guide: build with its `00_build` notebook, attach the three JARs to the cluster, `%pip install --no-deps` its wheel | Its JAR targets Spark 4.1 and Runtime 18; run its `01_parity` notebook on 17.3 first (OI-03) |
| pprof (R) | In a `%sh` cell, so that the driver's own `Rscript` finds it: `Rscript -e 'install.packages("remotes", repos = "https://cloud.r-project.org"); remotes::install_github("UM-KevinHe/pprof@d780cf50e11e4e5ff9008f14732e35cf632762f0", upgrade = "never")'` | Untested on Databricks (OI-11): it needs a compiler and CRAN on the driver. It runs on the driver only, on one thread |

## 4. Order of work

| Step | Notebook | Widgets | Expect |
|---|---|---|---|
| 1 | `00_environment` | none | The cluster facts, problems if any, and which implementations are available |
| 2 | `01_generate` | `data_root`, plan `standard`, `max_rows` 5000000 | Four datasets (two models, two designs) of about 7 GB each before compression |
| 3 | `02_run` | the same, plus `results_root` | One record per implementation and dataset, two repeats |
| 4 | `03_report` | `results_root` | Tables, and a Markdown report under `results_root/reports/` |
| 5 | Repeat 2 to 4 with `max_rows` 20000000, then 50000000, then 100000000 | | See section 5 |

`data_root` can be a Volume or a `dbfs:/` path. `results_root` must be reachable with ordinary file
calls: `/Volumes/...`, `/Workspace/...` or `dbfs:/...`.

## 5. Sizes on this cluster

Rows of 180 features are 1,464 bytes as doubles.

| Rows | Table, uncompressed | Per worker if cached evenly on 10 | Expectation |
|---|---:|---:|---|
| 5 million | 7 GB | 0.7 GB | Fits in memory everywhere; pprof_py is at its ceiling on the 32 GB driver |
| 20 million | 29 GB | 2.9 GB | Spark implementations only; cached working sets fit in memory |
| 50 million | 73 GB | 7.3 GB | Cached working sets exceed executor storage memory and go to disk; the records' spill and GC figures show the cost |
| 100 million | 146 GB | 14.6 GB | Runs from local disk if at all; this is the "where feasible" end of the plan |

These are arithmetic, not measurements. The two largest sizes are there to find the limits, and a
run that fails or times out (two hours per fit by default) is a result.

The 128 GB driver may be used to run pprof_py or pprof (R) at sizes the 32 GB driver cannot hold,
as extra references. Nothing in the plans depends on it.

## 6. What comes back

The `runs/` folder and the report under `results_root`. They contain run records of synthetic
data: timings, metrics, coefficients of the synthetic models and environment facts (node types,
runtime, Spark settings). Copy them into `results/<date>-<place>/` in the repository.
