# Open items

| ID | Item | Next step |
|---|---|---|
| OI-01 | Closed 2026-11-14. It concerned a package named spark-survival, which does not exist: the three references are pprof (R), pprof_py and pprof_spark. The placeholder adapter was removed and an adapter for pprof (R) added | None |
| OI-02 | The cluster has 10 workers of 16 GB (it used to have 30) and it autoscales; whether its size can be fixed is being explored (maintainer, 2026-11-14). Drivers of 32 GB and 128 GB are both available. The runner can ask for full size before each Spark fit and records the executors attached during it; neither has been tried on a real autoscaling cluster | Stage A's output, then stage B with `hold_executors` if the size cannot be fixed |
| OI-03 | Closed 2026-11-16 for the two models: pprof_spark's Spark 4.1 JAR (build `d0b9bd5`) loads on Runtime 17.3 and matches pprof_py to 1e-15 on three datasets of 5 million rows. pprof_spark's own project has since decided to support Spark 4.0 (its D-76) | None |
| OI-04 | Closed 2026-11-16: the adapter ran on the cluster. `pprof_spark_cached`, added in round 4, has run against a mock only | Its first run is the next plan |
| OI-05 | Closed 2026-11-16: the first cluster run's records carry task metrics. Which of the two sources supplied them is in the JSON records, not yet seen | None |
| OI-06 | `lab_spark` holds one partition of rows in a Python worker: 100,000 rows of 180 features are about 144 MB, several times that at peak | Tune `rows_per_partition` on the cluster |
| OI-07 | The prototypes implement Breslow ties and right-censored rows only; no entry times, weights or offsets; no providers at the bound in the logistic model | Roadmap steps 3 and 5 |
| OI-08 | The engine is not started | After the baseline and D-04 |
| OI-09 | Aggregate facts about the production designs (share and prevalence of indicator columns, rows with entry times, event times per stratum) are unknown, so the synthetic designs are guesses | Roadmap decision 4; then a third design |
| OI-10 | Peak executor memory from the REST API is the peak since the executor started, not the peak of one run | Restart the cluster between plans, or add a listener |
| OI-12 | Closed 2026-11-14: LICENSE added (MIT, copyright Tao Xu) | None |
| OI-13 | A single-machine fit that exhausts the driver kills the notebook's Python process. The runner now leaves a record with status `running` for such a fit, and the memory guard allows a fit 40 percent of the driver node (the Spark driver's JVM shares it). Both the factor and the adapters' memory estimates are guesses | Compare `memory_guard` with the measured peaks in the first records |
| OI-14 | `hold_executors` = 10 did not bring the cluster to 10 executors before 7 of the 16 Spark fits of the first run. Whether it waited its full 15 minutes each time, and what the autoscaler did meanwhile, is in the JSON records | The records; the warm-up now prints and stores its progress |
| OI-15 | lab_spark had one task at 190 to 215 times its stage's median in three of eight runs | The records name the stage |
| OI-16 | On this cluster Photon leaves the executors a heap of 2.2 GB. pprof_spark's own project measures on the cluster as it is (its D-76), and so does this one; a run without Photon would be a diagnostic, not the baseline | None now |
| OI-11 | pprof (R) on Databricks is untested: the adapter needs `Rscript` on the driver's PATH and the package installed where that Rscript finds it. Its driver-memory factor (five copies of the data) is a guess | `docs/databricks-runbook.md`, section 3; the records report R's own peak |
