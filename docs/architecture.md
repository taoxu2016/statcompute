# Architecture

## Purpose of the repository

One place, independent of the existing packages, in which to (1) try new algorithms, (2) measure
every implementation under the same conditions, and (3) build the next Spark implementation. The
existing packages stay untouched and serve as references.

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
| The engine lives in `engine/` under the Scala package `statcompute.engine` | It must load next to pprof_spark's `pprof.spark` JARs on one cluster |

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

Adapters today: `pprof_r`, `pprof_py`, `pprof_spark`, `lab_numpy`, `lab_spark`.

## How an algorithm moves

| Stage | Where | Leaves the stage when |
|---|---|---|
| Probe | `statcompute/lab/probes/` | A single file with fixed seeds shows the identity or the count, and its output is filed in `docs/research/` |
| Prototype | `statcompute/lab/` with an adapter of role "prototype" | It agrees with pprof_py within the tolerance classes on the validation tests, and the harness has timed it |
| Specification | `docs/spec/` | The maintainer approves the estimator's definition, edge cases and validation plan |
| Engine | `engine/` with an adapter of role "engine" | It passes the same validation tests and meets its target on the standard cluster |

A rejected candidate keeps its probe and its numbers. The register of hypotheses in the research
roadmap (its section 8.4) records the status of each.

## Equivalence

The four classes of the feasibility study apply: exact with the same iterates, exact with other
iterates, approximate, another estimator. `testkit/tolerances.py` holds the tolerance classes.
Prototypes stop on a rule that does not depend on the path: the next Newton step, in standard
errors (T_CONV).

## Results and reports

`bench` writes one JSON record per run, built from nothing each time. `results/` keeps records
from synthetic data by place and day; `docs/reports/` keeps the reports rendered from them.
