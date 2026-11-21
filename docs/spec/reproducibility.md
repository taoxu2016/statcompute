# Reproducibility without bitwise identity

Decision D-14 (the maintainer's, 2026-11-20): bitwise reproducibility is not an absolute requirement; agreement
within strict numerical tolerances, together with stable provider-level flags and auditable execution
configurations, may be sufficient. This document proposes what the three conditions mean and says what exists
today. Sections 2 and 3 are decisions D-15 and D-16, approved on 2026-11-21. Section 3 is not implemented in full
(OI-29). Evidence is from the assistant's sandbox unless a cluster is named.

## 1. Strict numerical tolerances: in place

`statcompute/testkit/tolerances.py` holds the classes; the harness reports every comparison as a multiple of them.

| Class | Quantity | Bound |
|---|---|---|
| T_FN | The objective at fixed parameters | 1e-12, relative |
| T_COEF | Converged coefficients | 1e-10 + 1e-8 times the largest coefficient |
| T_VAR | Standard errors | 1e-7, relative |
| T_LOGLIK | The log-likelihood at the estimate | 1e-9, relative |
| T_CONV | The Newton step that remains at the estimate | 1e-8 standard errors (1e-10 for "tight") |

One amendment is decision D-09, approved 2026-11-21: for Cox a difference from pprof_py is judged after allowing for the Newton step
that remains at pprof_py's estimate, because its stopping rule can leave one (`docs/reference-issues.md`, RI-1).

## 2. Stable provider-level flags (D-15)

A flag is a cut-off on a continuous statistic: |z| against the normal quantile for the score and Wald tests, the
p-value against 1 - level for the exact test. Two computations of one fit can therefore flag a provider differently
only when the cut-off lies between its two statistics. No tolerance on the statistics can rule that out; it can
only say when it is acceptable.

**Rule.** Between two computations of the same fit on the same data, a provider's flags may differ only if its two
statistics are within T_FLAG of each other: 1e-7 for a z statistic (class T_FLAG_Z), 1e-9 for a p-value (T_FLAG_P). Such a provider is
borderline, and it is counted and reported. A flag that differs with statistics further apart is a numerical
disagreement, to be explained like a coefficient outside T_COEF.

**What is reported.** `statcompute.validation.compare_flags` takes two provider tables and returns, per test: the
flags that differ, how many of them are borderline, the largest difference in the statistic, the smallest distance
of any statistic from its cut-off, and whether the pair is stable. `flag_summary` reduces one table to the counts, a
digest of the flags and that smallest distance; the engine's logistic records carry it, and the benchmark report
says per dataset whether every run flagged the same providers.

**Evidence** (probe A-10, `docs/research/flag-margins.md`: 800,000 rows, 4,000 providers, 10 features).

| Pair | Largest difference in a statistic | Flags that differ |
|---|---|---|
| The engine with 16 partitions and with 40 | 5e-15 | 0 |
| The engine and pprof_py at its tightest tolerance | 6e-13 | 0 |
| pprof_py at 1e-5, the default of pprof in R, and at 1e-12 | 9e-9 (Wald) | 0 |

The provider nearest a cut-off was 1.5e-4 away on the z scale and 3.7e-5 on the p scale. By the counts near the
cut-offs, about one fit in 5,000 to 7,000 at this size would have a provider within 1e-7 of a z cut-off.

On the cluster (2026-11-21, 5 million rows, 8,000 providers, four runs of each logistic dataset at 6 to 10 executors):
every run flagged the same providers in the same direction by all three tests. The nearest provider to a cut-off was
6.2e-5 away for a z statistic and 4.7e-7 for the exact p-value, closer than the sandbox's dataset suggested.

**Not covered.** Cox models: the engine has no provider tests for them yet. Flags across implementations on the
cluster: the reference adapters return no provider tables (OI-28). Whether a production report should mark
borderline providers: that is methodology, for its owners.

## 3. Auditable execution configuration (D-16)

What must be kept with a fit so that someone else can say what produced its digits, and repeat it.

| Item | In a run record of the harness today | Missing |
|---|---|---|
| The data | Dataset identifier, its specification, rows, and the manifest's checksum | Nothing for synthetic data. For other tables: a checksum of the caller's choosing |
| The package | Adapter name, version, and for the references the commit and whether it is the pinned one | The engine's commit: the version only |
| The estimator's settings | Every option passed (`fit.settings`): ties, strata, tolerance, the row identifier | Nothing |
| The layout | Partitions, rows per partition, how rows are placed, passes per evaluation; from round 13 the Arrow batch size | Nothing |
| Spark | Version, application, Databricks runtime and node types, 18 settings, the executors attached before, during and after the fit | Nothing |
| Numerical libraries on the driver | Python, NumPy, pandas, pyarrow, SciPy; from round 13 the BLAS library, its version and threading | Nothing |
| Numerical libraries on the executors | Nothing | Python, NumPy and BLAS where the kernels run: this needs a pass over the executors (OI-29) |
| Convergence | Iterations, evaluations, the last Newton step in standard errors, each evaluation's seconds | Nothing |
| The outcome | Coefficients, standard errors, log-likelihood; from round 13 the flag summary | Provider tables themselves, which stay out of records |

A fit made outside the harness carries its settings and its partitions and nothing else. The proposal is that the
engine attach the whole record to the model it returns, so that the audit trail does not depend on the harness.

## 4. What still repeats bit for bit

With a row identifier, two fits of the same data with the same settings return identical bits, with strata and
without, however the input is partitioned (`docs/spec/engine.md`, section 4, with the conditions). That stays
tested. D-14 means it is a property the engine has where it is cheap, not one a design must pay for.
