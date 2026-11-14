# Decisions

Proposed by the assistant; taken by the maintainer. Numbers are never reused.

| ID | Decision | Status |
|---|---|---|
| D-01 | The repository is named `statcompute`. It holds the research workspace, the benchmark of all implementations and the new engine. Alternatives considered: `statcompute-lab` (suggests experiments only) and `pprof-compute` (ties the name to one family of models) | Proposed 2026-11-14 |
| D-02 | Licence: MIT, as the reference packages. No LICENSE file is added until the maintainer names the copyright holder | Proposed |
| D-03 | Target runtime: Databricks Runtime 17.3 LTS, which is Apache Spark 4.0.0 with Scala 2.13. pprof_spark targets Spark 4.1.0 and Runtime 18 | Proposed |
| D-04 | The engine's language and determinism are decided after the baseline, not before. Two candidates are measured side by side in step 0: JVM kernels with bitwise partition invariance (pprof_spark's rule NN-4, the feasibility study's route), and NumPy kernels inside Spark partitions (`lab_spark`), which are simpler and use BLAS but are not bitwise reproducible | Proposed |
| D-05 | Prototypes stop when the next Newton step is below 1e-8 standard errors (1e-10 for "tight"): the tolerance class T_CONV of the research roadmap (F-10). The bound is provisional until calibrated | Proposed |
| D-06 | Numerical comparisons take pprof_py at its tightest tolerance as the reference | Proposed |
| D-07 | The repository stores run records and reports of synthetic data only, as JSON and Markdown | Proposed |
