# Open items

| ID | Item | Next step |
|---|---|---|
| OI-01 | Closed 2026-11-14. It concerned a package named spark-survival, which does not exist: the three references are pprof (R), pprof_py and pprof_spark. The placeholder adapter was removed and an adapter for pprof (R) added | None |
| OI-02 | The maintainer confirmed on 2026-11-14 that the cluster has 10 workers of 16 GB (it used to have 30). Still to see on the cluster itself: that the number is fixed (no autoscaling), the driver's size and the runtime | Notebook `00_environment` prints them |
| OI-03 | pprof_spark's JAR is built for Spark 4.1.0 and Runtime 18. Whether it loads and passes its own parity notebook on Runtime 17.3 (Spark 4.0.0) is untested. A build against 4.0 would be a change in that repository, so it is the maintainer's to decide | Run pprof_spark's `01_parity` on the standard cluster first |
| OI-04 | The pprof_spark adapter has run only against a mock: the sandbox cannot build its JARs (no access to Maven Central) | First real run is `02_run` on the cluster |
| OI-05 | Task metrics come from the Spark UI's REST API on the driver. Whether Databricks exposes it at `uiWebUrl` on this cluster is unverified | Notebook `00_environment` reports it; fallback is a JVM listener in `engine/` |
| OI-06 | `lab_spark` holds one partition of rows in a Python worker: 100,000 rows of 180 features are about 144 MB, several times that at peak | Tune `rows_per_partition` on the cluster |
| OI-07 | The prototypes implement Breslow ties and right-censored rows only; no entry times, weights or offsets; no providers at the bound in the logistic model | Roadmap steps 3 and 5 |
| OI-08 | The engine is not started | After the baseline and D-04 |
| OI-09 | Aggregate facts about the production designs (share and prevalence of indicator columns, rows with entry times, event times per stratum) are unknown, so the synthetic designs are guesses | Roadmap decision 4; then a third design |
| OI-10 | Peak executor memory from the REST API is the peak since the executor started, not the peak of one run | Restart the cluster between plans, or add a listener |
| OI-12 | D-02 is approved (MIT), but no LICENSE file exists: the copyright holder has not been named. Until it exists the public repository is "all rights reserved" by default | The maintainer names the holder |
| OI-13 | A single-machine fit that exhausts the driver kills the notebook's Python process. The runner now leaves a record with status `running` for such a fit, and the memory guard allows a fit 40 percent of the driver node (the Spark driver's JVM shares it). Both the factor and the adapters' memory estimates are guesses | Compare `memory_guard` with the measured peaks in the first records |
| OI-11 | pprof (R) on Databricks is untested: the adapter needs `Rscript` on the driver's PATH and the package installed where that Rscript finds it. Its driver-memory factor (five copies of the data) is a guess | `docs/databricks-runbook.md`, section 3; the records report R's own peak |
