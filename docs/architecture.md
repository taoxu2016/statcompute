# Architecture

## Purpose of the repository

One place, independent of the existing packages, in which to (1) try new algorithms, (2) measure
every implementation under the same conditions, and (3) prove the algorithms at scale in an engine
of its own. The existing packages stay untouched. What is validated here is meant to serve both
pprof_py and pprof_spark: a reusable platform for computational research and development (D-13).

| Package | Role (D-12, D-13; the maintainer's, 2026-11-20) |
|---|---|
| pprof_py | The statistical specification: the estimators' definitions and conventions, the provider-profiling layer, the fixtures and R comparisons that serve as acceptance tests. Not unquestionable computational ground truth: its results are validated here like any other, and what is found is reported (`docs/reference-issues.md`). A change of methodology is reviewed explicitly, by its owners |
| statcompute | Algorithms and evidence: probes, prototypes, an engine that runs the algorithms on the standard cluster, the benchmark of every implementation, and the record of what was tried |
| pprof_spark | The Spark-native implementation in Scala, whose results are bitwise equal across partitionings. A receiver of validated algorithms, and one of the implementations measured here |
| pprof (R) | The published R package, a reference for parity |

Algorithms flow from this repository to pprof_py and pprof_spark; validation and benchmark evidence
flow back. Nothing here writes to either package: a hand-over is a proposal that its maintainers
apply.

## Areas and their dependencies

```
            testkit            specifications, synthetic data, tolerance classes
           /   |   \
  references  lab   (engine adapter, later)      each registers adapters with statcompute/adapters.py
           \   |   /
           validation          fit-to-fit comparison
               |
             bench             plans, runner, metrics, records, reports
               |
        notebooks/databricks   thin entry points; no logic of their own
```

| Rule | Reason |
|---|---|
| `testkit` imports nothing else from this package | Everything depends on it |
| Adapters import the wrapped package inside their methods, never at module level | Listing adapters must work where a package is absent |
| `bench` never imports a reference package or a prototype directly; it goes through the registry | A new implementation needs no change to the harness |
| Notebooks contain no statistics and no measurement code | The same functions run locally, in CI and on a cluster |
| No source of a reference package is stored here | The references are read-only; their licences and histories stay with them |
| The engine is the Python package `statcompute.engine` (D-08) and imports nothing from `lab`, `references` or `bench` | It is the product; the rest measures and checks it |

## The adapter contract

`statcompute/adapters.py`. An adapter answers three questions the same way for every
implementation.

| Method | Answer |
|---|---|
| `available(spark)` | Can it run in this session; if not, why |
| `support(model, data)` | `exact`, `deviation` or `unsupported`, with notes. This is the statement about statistics: `exact` means the same estimand, likelihood and tie rule |
| `fit(spark, dataset, model)` | A `FitResult`: coefficients, standard errors, log-likelihood, iterations, the options actually passed, and the seconds spent in each phase |

An adapter also states where its arithmetic runs (`kind`: driver or spark), what it is (`role`:
reference, prototype or engine) and how much driver memory a fit needs (`driver_bytes`), which the
runner checks before starting.

Adapters today: `engine`, `pprof_r`, `pprof_py`, `pprof_spark`, `pprof_spark_cached` (the same package
with its input cached first), `lab_numpy`, `lab_spark`.

## How an algorithm moves

| Stage | Where | Leaves the stage when |
|---|---|---|
| Probe | `statcompute/lab/probes/` | A single file with fixed seeds shows the identity or the count, and its output is filed in `docs/research/` |
| Prototype | `statcompute/lab/` with an adapter of role "prototype" | It agrees with pprof_py within the tolerance classes on the validation tests, and the harness has timed it |
| Specification | `docs/spec/` | The maintainer approves the estimator's definition, edge cases and validation plan |
| Engine | `statcompute/engine/` with an adapter of role "engine" | It passes the same validation tests and meets its target on the standard cluster |
| Handed over (D-17) | `docs/handover/`, one folder per algorithm; none yet | Three things exist and have been offered to the maintainers of pprof_py and pprof_spark: a written specification, a reference kernel that needs no Spark, and fixtures with expected values. What they found when they applied it has come back as evidence |

A rejected candidate keeps its probe and its numbers. The register of hypotheses in the research
roadmap (its section 8.4) records the status of each.

## Equivalence

The four classes of the feasibility study apply: exact with the same iterates, exact with other
iterates, approximate, another estimator. `testkit/tolerances.py` holds the tolerance classes.
Prototypes stop on a rule that does not depend on the path: the next Newton step, in standard
errors (T_CONV).

Bitwise reproducibility is not an absolute requirement (D-14). `docs/spec/reproducibility.md`
proposes what stands in its place: the tolerance classes, stable provider-level flags and an
execution record that can be audited.

## Results and reports

`bench` writes one JSON record per run, built from nothing each time. `results/` keeps records
from synthetic data by place and day; `docs/reports/` keeps the reports rendered from them.
