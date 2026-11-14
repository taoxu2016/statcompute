# engine: the new Spark implementation

Not started. This folder will hold the Scala build of the engine that replaces pprof_spark's
kernels and layouts. It starts after the baseline on the standard cluster (roadmap step 0) and
decision D-04.

## Fixed points

| Point | Value |
|---|---|
| Target | Databricks Runtime 17.3 LTS: Apache Spark 4.0.0, Scala 2.13, JDK 17 (D-03) |
| Names | Scala package `statcompute.engine`, artifacts `statcompute-engine_2.13`. Never `pprof.spark`: both engines must load on one cluster for side-by-side runs |
| Entry to the harness | An adapter with role "engine", registered like the others in `statcompute/adapters.py` |
| Source of truth for results | pprof_py at its tightest tolerance (D-06), then pprof_spark's fixtures where the same function exists |
| Reuse | pprof_spark's design (blocks, ordered reduction, specifications) is the starting point by reference. Code is written here; nothing is copied without the maintainer's decision, although both are MIT |

## Order of work

The order of the research roadmap, each step entering through the harness with a measured gain:

1. Reduction on the executors in fixed groups.
2. Chunked accumulation, shared by every model family.
3. The weighted-Gram form of the Cox information, with per-stratum centring, Efron ties and entry times.
4. Per-time aggregates for unstratified fits and oversized strata.
5. Profile-likelihood Newton for the logistic model, with providers at the bound.

`statcompute/lab` already holds NumPy prototypes of steps 3 (Breslow, right-censored) and 5
(without bounded providers). They show that the algorithms return pprof_py's estimates; they are
not the engine.
