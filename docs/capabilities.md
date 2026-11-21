# Capability matrix

What each implementation supports for the two case studies, read from its source at the commit in
`statcompute/references/REFERENCES.lock` as it stood on 2026-11-16 (pprof `d780cf5`, pprof_py `94b5476`, pprof_spark
`e5c9484`, whose code for these two models is unchanged at the build `d0b9bd5` pinned then). Since 2026-11-21 the lock
pins pprof_spark at `5db844e` (round 176), which lays out and stores its blocks differently; its column below has not been
read again at that commit (OPEN_ITEMS OI-37). "Read" means read in the source or its documentation; the last row of each table says
what has actually been run here. A benchmark row is like-for-like only where the entries of the two
implementations agree for the model that was run.

## Cox proportional hazards

| Feature | pprof (R) | pprof_py | pprof_spark | Lab prototypes (this repository) |
|---|---|---|---|---|
| Entry point | `pp_data()` then `fit_cox()`, which fits through `survival::coxph` | `CoxPH().fit()` | `CoxPH.fit` (Scala), `pprof_spark.CoxPH` (Python) | `lab_numpy`, `lab_spark` |
| Where it runs | One machine, one thread | One machine, NumPy and numba, one thread | Spark executors; blocks of whole strata | `lab_numpy`: one machine. `lab_spark`: NumPy inside Spark partitions |
| Ties | Breslow (default), Efron | Breslow (default), Efron; exact not implemented | Breslow (default), Efron | Breslow only |
| Strata | Provider (default) or none | Yes | Yes | Yes; `lab_spark` requires them |
| Unstratified fit | Yes | Yes, in driver memory | Yes, as one task per pass, up to `maxStratumBytes` (4 GiB) | `lab_numpy` only |
| Entry times | Yes (`start`) | Yes | Yes | No |
| Case weights, offsets | Yes | Yes | Yes | No |
| Robust (clustered) variance | Yes, with a patient column | Yes | Yes | No |
| Stopping rule | survival::coxph's defaults; `fit_cox()` does not expose them: relative change in the log partial likelihood below 1e-9, at most 20 iterations | The same rule, eps and iterations adjustable | The same formula, tested on the full step first; adjustable | Next Newton step below 1e-8 standard errors |
| Information matrix | survival's C code | Running risk-set sums with second moments per event time | The same, with compensated sums | Weighted Gram matrix minus one rank-one term per event time |
| Bitwise reproducible across partitionings | Not applicable | Not applicable | Yes, by design | No: BLAS |
| Binding limit | Driver memory: a data frame and its copies | Driver memory: X and a centred copy | CPU per pass; driver budget for block partials | `lab_numpy`: driver memory. `lab_spark`: one partition per Python worker |
| Run in this repository | Yes: `tests/test_references.py` | Yes | Yes, on Databricks on 2026-11-16: within 1e-15 of pprof_py at 5 million rows | Yes |

## Logistic regression with provider fixed effects

| Feature | pprof (R) | pprof_py | pprof_spark | Lab prototypes |
|---|---|---|---|---|
| Entry point | `pp_data()` then `fit_logistic_fe()` | `LogisticFixedEffectModel().fit()` | `LogisticFE.fit`, `pprof_spark.LogisticFixedEffect` | `lab_numpy`, `lab_spark` |
| Algorithm | SerBIN (default) or BAN; optional Firth penalty | SerBIN (default) or BAN | SerBIN | Profile-likelihood Newton: each provider's intercept solved in place |
| Passes over the data per Newton step | In memory; OpenMP threads optional | In memory | Two, plus two variance passes after the fit | One |
| Providers without events or with only events | Kept within the median plus or minus the bound (10) | The same | The same | Refused: not prototyped |
| Small providers | Left out below `cutoff` (default 10); the adapter sets 0 | Screening optional; the adapter turns data preparation off | Screening optional; off by default in the adapter's call | Not applicable |
| Binomial trials | Yes (`n`) | A trials column is accepted; the response is still validated as 0/1 (its README) | A trials column is accepted | No |
| Default stopping rule | Largest change in beta below 1e-5; four other rules selectable | Largest change in beta below 1e-8 | pprof_py's defaults | Next Newton step below 1e-8 standard errors |
| Variance of beta | Inverse of the reduced information | The same | The same | The same |
| Provider tests, standardized measures | Yes | Yes | Yes | No |
| Binding limit | Driver memory | Driver memory: X, a stacked copy and q*X | CPU per pass; provider effects on the driver | As for Cox |
| Run in this repository | Yes | Yes | Yes, on Databricks on 2026-11-16 | Yes |

## The engine

`statcompute.engine` is not a reference and is not in the tables above. `docs/spec/engine.md` lists
what it implements: Cox regression with or without strata, Breslow or Efron ties, entry times,
weights and offsets; logistic regression with provider fixed effects and providers at the bound.
The same document lists its outputs beyond the fit.

## Consequences for the benchmarks

- The common ground of all implementations is **Cox with Breslow ties, right-censored rows,
  stratified by provider, no weights or offsets**, and **logistic fixed effects with 0/1 outcomes,
  every provider kept, and no provider at the bound**. The plans use exactly that.
- Entry times, Efron ties, weights, offsets and bounded providers can be compared among the three
  packages only. A plan that asks for them marks the prototypes `unsupported`.
- The three packages have different default tolerances for the logistic model (1e-5 in R, 1e-8 in
  the other two), and R's Cox tolerance cannot be changed. Timings in `default` mode therefore
  compare what each package does by default, and iteration counts are never a criterion.
- pprof (R) and pprof_py run on one machine. They enter the scale plans only where they fit the
  driver; the runner's memory guard decides and records it.
