# Decisions

Proposed by the assistant; taken by the maintainer. Numbers are never reused.

| ID | Decision | Status |
|---|---|---|
| D-01 | The repository is named `statcompute`. It holds the research workspace, the benchmark of all implementations and the new engine | Approved 2026-11-14; pushed to https://github.com/taoxu2016/statcompute |
| D-02 | Licence: MIT, as the reference packages | Approved 2026-11-14. The LICENSE file waits for the name of the copyright holder (OI-12) |
| D-03 | Target runtime: Databricks Runtime 17.3 LTS, which is Apache Spark 4.0.0 with Scala 2.13. pprof_spark targets Spark 4.1.0 and Runtime 18 | Approved 2026-11-14 |
| D-04 | The engine's language and determinism are decided after the baseline, not before. Two candidates are measured side by side in step 0: JVM kernels with bitwise partition invariance (pprof_spark's rule NN-4, the feasibility study's route), and NumPy kernels inside Spark partitions (`lab_spark`), which are simpler and use BLAS but are not bitwise reproducible | Approved 2026-11-14; the choice itself is still open |
| D-05 | Prototypes stop when the next Newton step is below 1e-8 standard errors (1e-10 for "tight"): the tolerance class T_CONV of the research roadmap (F-10). The bound is provisional until calibrated | Approved 2026-11-14 |
| D-06 | Numerical comparisons take pprof_py at its tightest tolerance as the reference | Approved 2026-11-14 |
| D-07 | The repository stores run records and reports of synthetic data only, as JSON and Markdown. Since the repository is public, records are scrubbed when written: the cluster identifier is hashed, and e-mail addresses and user folders are removed from messages and traces | Approved 2026-11-14; the scrubbing was added in round 2 |

The research roadmap has decisions of its own (its section 9: adopting track A with its amendments,
the definition of the estimate at the bound, the production design facts, the first second-wave
project). Those are not covered by the approval above and stay open.
