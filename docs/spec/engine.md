# The engine: specification, version 0.3

`statcompute.engine` is the new implementation of the two case studies. This document says what
it estimates, how it computes, what has been verified and what is missing. Evidence is from the
assistant's sandbox (Python 3.12, NumPy, PySpark 4.0.1 in local mode) unless a cluster is named.

## 1. Scope

| Model | Entry point | Options |
|---|---|---|
| Cox proportional hazards, with strata or without | `CoxPH(time, event, features, strata)`; `strata=None` for a fit without strata | `ties` "breslow" or "efron"; `entry` (entry times); `weights`; `offset`; `row_id`; `max_times` |
| Logistic regression with one fixed intercept per provider | `LogisticFE(outcome, features, provider)` | `bound` (default 10); `row_id`; `provider_effects` |

Both return coefficients, their covariance (the inverse information), the log-likelihood, the
iteration count and the convergence measure, and add a linear-predictor column to a table on
request. Section 3.1 lists what `LogisticFE` returns per provider; `CoxPH(..., baseline=True)` also
returns the baseline hazard (section 2.3). `statcompute.engine.local` fits the same models from arrays on one machine
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

### 2.1 With strata: whole strata in a partition

Each partition holds whole strata, the kernel sorts them by time, and one pass returns the
log-likelihood, the score and the information (section 4).

### 2.2 Without strata: the time grid

Without strata every risk set spans the whole table, so no partition can hold one. The engine then
uses the fact that times take few distinct values (days over some years: 1,825). It needs at most
`max_times` distinct values, 20,000 by default, and neither shuffles nor sorts the rows.

*Pass 1.* Every batch of rows adds (r_i, r_i x_i) to a table at the row's exit time; with Efron
ties the same for the rows that died; with entry times the same at the row's entry time. That is
p + 1 numbers per time and table. The driver adds the partitions' tables and takes suffix sums over
the grid:

    (S0, S1)(t) = sum over times >= t of the exit table  -  sum over times >= t of the entry table.

From S0, S1 and the death table it has the log-likelihood, the score, the hazard increment at every
time and the matrix of rank-one terms. For Efron's steps it does not form a vector per step: with
a_j = 1/A_j and b_j = (j/k)/A_j, xbar_j = a_j S1 - b_j S1D, so

    sum_j xbar_j xbar_j' = (sum a_j^2) S1 S1' - (sum a_j b_j)(S1 S1D' + S1D S1') + (sum b_j^2) S1D S1D',

three numbers per time.

*Pass 2.* Every batch forms X' diag(E) X, reading E_i off the cumulative hazard at the row's exit
and entry times. The information is the sum over batches minus the matrix from pass 1.

| Quantity | Value |
|---|---|
| Passes per evaluation | 2 |
| Shuffles, sorts | None; rows stay in the partitions and batches they come in |
| To the driver per partition, pass 1 | (p + 1) numbers per distinct time, times 1 to 3 tables: 1.5 MB at 100 features and 1,825 days, Breslow, no entry times |
| Memory in a Python worker | One Arrow batch at a time (10,000 rows by default), whatever the partition's size |
| Centring | At the overall means: there is one stratum |

*What is new and what is not* (search of 2026-11-17, `docs/research/prior-art-time-grid.md`). The
identity is old, and the coxdev package of Taylor, Hastie and Narasimhan publishes it as a
matrix-free operator with Efron ties, start times, strata and weights: the engine's kernels are an
independent implementation of that algebra. Aggregating per distinct event time across workers is
what WebDISCO and H2O-3 do, with second moments. The one part not found in any source is the
second pass that replaces those second moments by a single weighted Gram matrix, so that
O(Kp + p^2) numbers travel instead of O(Kp^2). Nothing else is claimed.

### 2.3 Baseline hazard

On request, the cumulative baseline hazard at every event time, per stratum when there are strata:
Breslow's estimate, or Efron's with Efron ties, at x = 0 and offset 0, with exp(-hazard) beside it.
This is the table pprof_py and R's `basehaz(centered = FALSE)` report when there is no offset; with
an offset those two multiply by exp(mean offset) and the engine does not. With strata it takes one
more pass; without, it comes from the last evaluation.

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

### 3.1 Per provider

`provider_effects` has one row per provider, from one pass after the fit. Definitions are pprof_py's.

| Column | Definition |
|---|---|
| `gamma`, `bounded` | The intercept; whether it is held at the bound |
| `standard_error` | sqrt(1/h_j + xbar_j' S^-1 xbar_j): pprof_py's `variances_["gamma"]` |
| `standard_error_case_mix` | sqrt(1/h_j + (xbar_j - xbar)' S^-1 (xbar_j - xbar)), xbar the mean covariate row: pprof_py's `gamma_case_mix`, which does not depend on the covariates' origin |
| `rows`, `events`, `expected` | Rows, observed events, and events expected if the provider had the median effect |
| `indirect_ratio`, `indirect_rate` | events / expected; the ratio times the overall event rate in percent, kept within 0 and 100 |
| `score_z`, `score_p`, `score_flag` | (events - expected) / sqrt(variance under the median effect); two-sided p-value; 1, -1 or 0 at `level` (default 0.95) |
| `wald_z`, `wald_p`, `wald_flag` | (gamma - median) / `standard_error_case_mix`, and the same |

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
| The time grid equals explicit risk sets | `test_engine_time.py`: the same 16 combinations, rows cut into chunks | Log-likelihood within 1e-12, score 1e-11, information 1e-10, relative |
| The time grid gives the same fit as the stratum kernel run as one stratum | Same file: both tie methods, with and without weights, offset and entry | Coefficients within 1e-10 standard errors (measured: 6e-14) |
| Unstratified fits equal pprof_py's | Same file: 4 combinations | As for the stratified fits: equal up to the Newton step pprof_py leaves |
| Unstratified Cox on Spark equals one machine, and does not depend on partitions or batch sizes | Same file | Within 1e-9 standard errors with 7 partitions and batches of 700 rows |

| The baseline hazard equals pprof_py's | `test_engine_outputs.py`: both tie methods, with strata and without, with weights | Same rows; hazard within 1e-8 relative (measured: 8e-10 with strata, 5e-14 without) |
| Standardized ratios and the score and Wald tests equal pprof_py's | Same file, with one provider at the bound | Ratios, rates and expected counts within 1e-9; score statistics within 1e-9; Wald within 1e-7; identical flags |

102 tests pass. Nothing of the engine has run on Databricks yet.

One observation about the reference, not the engine: with its default tolerance pprof_py's Efron fit
can stop 5e-5 standard errors short of the root (seen at 200,000 rows without strata). Comparisons
with it therefore use its tightest tolerance.

## 6. Not implemented

| Missing | Note |
|---|---|
| A provider with events and non-events outside the bound | Refused with a message. Needs a second pass per evaluation |
| More than 200,000 rows in providers without events or with only events | Refused |
| Times with more than `max_times` distinct values, without strata | Refused. Continuous times need a sorted layout |
| A few large strata | The time grid handles one stratum; a grid per stratum is the natural extension |
| Identical bits across runs, without strata | The sums follow the input's partitions and batches, so runs agree to rounding |
| Robust variance; martingale and score residuals per row; survival curves for new rows; standardized measures and provider tests | Cox |
| The exact (Poisson-binomial) provider test; direct standardization; confidence intervals for the ratios; binomial trials | Logistic |
| Sorting once instead of in every pass | A pass factorizes and sorts its partition each time; storing the order at preparation would remove that |
