# The engine: specification, version 0.1

`statcompute.engine` is the new implementation of the two case studies. This document says what
it estimates, how it computes, what has been verified and what is missing. Evidence is from the
assistant's sandbox (Python 3.12, NumPy, PySpark 4.0.1 in local mode) unless a cluster is named.

## 1. Scope

| Model | Entry point | Options |
|---|---|---|
| Cox proportional hazards, stratified | `CoxPH(time, event, features, strata)` | `ties` "breslow" or "efron"; `entry` (entry times); `weights`; `offset`; `row_id` |
| Logistic regression with one fixed intercept per provider | `LogisticFE(outcome, features, provider)` | `bound` (default 10); `row_id`; `provider_effects` |

Both return coefficients, their covariance (the inverse information), the log-likelihood, the
iteration count and the convergence measure. `LogisticFE` also returns every provider's intercept
with its standard error. `statcompute.engine.local` fits the same models from arrays on one machine
with the same kernels.

## 2. Cox regression

**Estimator.** The maximum partial likelihood estimate, as in `survival::coxph` and pprof_py. A row
is at risk at time t when entry < t <= time. Write r_i = w_i exp(x_i'beta + offset_i). At an event
time t of a stratum with k tied deaths of total weight d, S0 and S1 are the sums of r_i and r_i x_i
over the risk set, and S0D and S1D the same sums over the tied deaths. Efron's method takes the
steps j = 0, ..., k-1 with A_j = S0 - (j/k) S0D and xbar_j = (S1 - (j/k) S1D) / A_j, each with weight
d/k; Breslow's is the single step A = S0, xbar = S1/S0, with weight d.

**Information without second moments.** The classical information needs the second moment of x over
every risk set. The engine uses the identity

    information = X' diag(E) X  -  sum over steps of  weight * xbar xbar',

where E_i is row i's expected number of events:

    E_i = r_i [c(time_i) - c(entry_i)] - [i died in a tie] r_i cD(time_i),
    c(t)  = sum over event times s <= t of (d/k) sum_j 1/A_j(s),
    cD(t) = (d/k) sum_j (j/k) / A_j(t).

One weighted Gram matrix replaces a second moment per event time. This is the roadmap's O-2 with
amendment F-5 (Efron ties and entry times).

**Centring within strata** (F-4). The partial likelihood does not change when a covariate is shifted
by a constant per stratum. The kernel subtracts each stratum's mean first, which removes the
cancellation the identity otherwise suffers when stratum means are far apart.

## 3. Logistic regression with provider fixed effects

**Estimator.** The estimate pprof_py's SerBIN converges to. For a provider with events and
non-events, its intercept solves its own score equation. A provider without events, or with only
events, has no finite intercept; it is held at the median intercept minus or plus `bound`. The
coefficients solve the reduced score equations, to which the held providers contribute with their
intercepts fixed. This is the definition amendment F-6 of the roadmap asks the methodology owner to
confirm; the engine reproduces pprof_py's number and does not decide the question.

**Computation.** Profile-likelihood Newton (O-4): for the current coefficients each provider's
intercept is solved in place, and one pass returns the profile gradient and its negative Hessian,
the Schur complement X'QX - sum_j h_j xbar_j xbar_j'. The rows of held providers are collected to
the driver once and evaluated there, because their intercepts depend on the median over all
providers.

## 4. Layout, passes and reproducibility

- **One shuffle.** Rows are repartitioned by stratum or provider, features packed into one array
  column, and the table cached. The number of partitions is ceil(rows / `rows_per_partition`), so it
  does not depend on how many executors happen to be attached.
- **One job per evaluation.** Each partition goes to a Python worker as Arrow batches; a NumPy
  kernel returns its sums; the driver adds them in partition order. A Cox fit of three iterations is
  a row count, one validation job and four passes. A logistic fit adds one scan and one pass for
  provider effects.
- **Memory.** The kernels run in Python workers, outside the executors' JVM heap. One partition of
  50,000 rows and 180 features is 72 MB as a matrix, several times that at the peak of a pass.
- **Stopping.** Newton's method stops when the next step is below `tol` standard errors in every
  coefficient (class T_CONV, default 1e-8), and reports that number.
- **Reproducibility.** With `row_id`, two fits of the same data with the same settings in the same
  session return identical bits (tested). Across layouts, or BLAS builds, results agree to rounding
  (1e-9 standard errors, tested), not bit for bit. This is weaker than pprof_spark's guarantee.

## 5. Verification

| Claim | Test | Result |
|---|---|---|
| The Cox kernel equals explicit risk sets | `test_engine_kernels.py`: 16 combinations of ties, weights, offset, entry | Log-likelihood within 1e-12, score 1e-11, information 1e-10, relative |
| Centring removes the cancellation | Same file: stratum means 1,000 standard deviations apart | Information within 1e-9 of the unshifted one |
| Cox fits equal pprof_py's | Same file: 10 combinations | The estimates differ by exactly the Newton step that remains at pprof_py's estimate (2e-8 and 7e-8 standard errors in the two plain cases), to 1e-9; standard errors within 1e-7; log-likelihood within 1e-9 |
| Logistic fits equal pprof_py's, with providers at the bound | Same file | Coefficients within 4e-15 standard errors; every intercept, the held ones included, within 4e-16 |
| Spark equals one machine | `test_engine_spark.py`: every Cox option, both tie methods; logistic with held providers | Within 1e-9 standard errors |
| Provider standard errors equal pprof_py's | Same file | Within 1e-8, relative |
| Bad data are refused | Same file | Missing values, events not 0 or 1, entry not before exit |

66 tests pass. Nothing of the engine has run on Databricks yet.

## 6. Not implemented

| Missing | Note |
|---|---|
| Unstratified Cox | Needs the per-time layout (O-3): two passes per evaluation, no shuffle. It is the next algorithm |
| A provider with events and non-events outside the bound | Refused with a message. Needs a second pass per evaluation |
| More than 200,000 rows in providers without events or with only events | Refused |
| Robust variance, baseline hazard, residuals, prediction | Cox |
| Provider tests, standardized measures, binomial trials | Logistic |
| Sorting once instead of in every pass | A pass factorizes and sorts its partition each time; storing the order at preparation would remove that |
