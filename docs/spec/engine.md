# The engine: specification, version 0.6

`statcompute.engine` is the new implementation of the two case studies. This document says what
it estimates, how it computes, what has been verified and what is missing. Evidence is from the
assistant's sandbox (Python 3.12, NumPy, PySpark 4.0.1 in local mode) unless a cluster is named.

## 1. Scope

| Model | Entry point | Options |
|---|---|---|
| Cox proportional hazards, with strata or without | `CoxPH(time, event, features, strata)`; `strata=None` for a fit without strata | `ties` "breslow" or "efron"; `entry` (entry times); `weights`; `offset`; `row_id`; `max_times`; `rows_per_partition`, `partitions` |
| Logistic regression with one fixed intercept per provider | `LogisticFE(outcome, features, provider)` | `bound` (default 10); `row_id`; `provider_effects`; `rows_per_partition`, `partitions` |

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

Each partition holds whole strata, sorted once when the working set is built: by stratum, time
descending and row identifier. One pass returns the log-likelihood, the score and the information
(section 4); it does not sort.

### 2.2 Without strata: the time grid

Without strata every risk set spans the whole table, so no partition can hold one. The engine then
uses the fact that times take few distinct values (days over some years: 1,825). It needs at most
`max_times` distinct values, 20,000 by default. No pass needs the rows in any order; section 4 says
where they are put and why.

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
| Shuffles, sorts | One of each, when the working set is built (section 4); none with `partitions="input"` |
| To the driver per partition, pass 1 | (p + 1) numbers per distinct time, times 1 to 3 tables: 1.5 MB at 100 features and 1,825 days, Breslow, no entry times |
| Memory in a Python worker | One Arrow batch at a time (10,000 rows by default), whatever the partition's size |
| Centring | At the overall means: there is one stratum. The first pass runs at beta = 0 before the means are known; it returns the feature sums, and the driver centres its tables afterwards, which is exact there because the risk scores at beta = 0 do not depend on the centring |

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

### 2.4 Robust variance, residuals and standardized ratios (with strata)

Each is on request and takes one or two more passes after the fit. Definitions are pprof_py's.

| Request | Result | Definition |
|---|---|---|
| `robust=True` | `robust_covariance`, `robust_standard_errors` | The sandwich I^-1 (sum over rows of u_i u_i') I^-1, each row its own cluster. u_i is the row's score residual: its events times (x_i - xbar at its time), minus its risk score times the integral of (x_i - xbar(t)) against the hazard over its time at risk. Breslow ties only |
| `residuals=True` (needs `row_id`) | `residuals`: row_id, martingale | The event indicator minus the row's expected events. Brought to the driver as one table |
| `standardized=True` | `standardized`: per stratum, observed, expected, indirect_ratio, person_time | He and Schaubel's two-stage measure. The national baseline hazard is Breslow's over the whole table at the stratified estimate; a provider's expected events are its rows' risk scores times that hazard over their time at risk; the ratio is observed over expected. Expected events add up to the observed. Breslow ties only; times with at most `max_times` values |

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
| `exact_p`, `exact_flag` | The exact test, pprof_py's default: the observed count against the Poisson-binomial distribution of the rows' probabilities under the median effect. Two-sided mid-p value: twice the smaller tail, each tail counting half the probability of the observed count. The flag has the sign of events - expected. Cost is quadratic in a provider's rows |
| `direct_expected`, `direct_ratio`, `direct_rate` | Only with `LogisticFE(..., direct=True)`: events expected over the whole table if every row had this provider's effect; that over all observed events; that over all rows, in percent. One more pass, with rows times providers evaluations |

## 4. Layout, jobs and reproducibility

- **The working set.** The rows are shuffled once: by stratum or provider when there is one, by row
  identifier without strata (by a hash of the whole row when no `row_id` is given). Spark then sorts
  every partition into the order the kernel works in: stratum, time descending, row identifier for
  Cox with strata; provider, row identifier for the logistic model; row identifier without strata.
  The features are packed into one array column and the table is cached. So the order is computed
  once and stored, and no pass sorts, factorizes or copies its rows again.
- **Partitions.** ceil(rows / `rows_per_partition`) in all three cases, 50,000 rows each by default,
  unless `partitions` is given. The number depends on the data and the settings only: not on the
  executors attached, and not on the files the input is stored in. Without strata
  `partitions="input"` leaves the rows in the input's own partitions (no shuffle, no sort): the
  layout before version 0.6, kept for comparison.
- **Jobs.** A light job reads every column except the features. It counts the rows, refuses rows
  that cannot be fitted and, without strata, collects the distinct times in the same pass. The
  working set is then built by the first pass over it, in that pass's job: the first evaluation for
  Cox, the scan of the providers for the logistic model. That pass counts the rows with a missing
  feature, and the fit is refused if there are any. A Cox fit of three iterations is the light job
  and four passes (eight passes without strata); a logistic fit is the light job, the scan, four
  passes and one more for the provider effects. Up to version 0.5 a fit with strata or providers
  started with a row count and then a validation job that built the cache; a fit without strata
  with that validation job and another job for the distinct times.
- **Times reported.** `seconds["prepare"]` ends when the working set exists. For Cox that includes
  the first evaluation, because it is the pass that builds the working set; `seconds["fit"]` is the
  evaluations after it. For the logistic model the scan builds it, and "fit" is every evaluation.
  `evaluation_seconds` has each evaluation's time, the first included.
- **Memory.** The kernels run in Python workers, outside the executors' JVM heap. One partition of
  50,000 rows and 180 features is 72 MB as a matrix, several times that at the peak of a pass.
  Without strata a worker holds one Arrow batch at a time, and every partition returns its pass-1
  tables to the driver: 100 partitions of 1.5 MB at 5 million rows, 100 features and 1,825 days.
- **Stopping.** Newton's method stops when the next step is below `tol` standard errors in every
  coefficient (class T_CONV, default 1e-8), and reports that number.
- **Reproducibility.** With `row_id`, two fits of the same data with the same settings return
  identical bits, with strata and without, however the input is partitioned (tested for Cox, also
  from a repartitioned input). What must stay the same: `rows_per_partition` or `partitions`; without
  strata also the Arrow batch size, since the sums run batch by batch, and tables small enough to be
  added on the driver; NumPy and its BLAS. The layout does not depend on the number of executors,
  so neither should the bits; that has been tested in local mode only. Without `row_id`, with
  `partitions="input"`, across layouts or across BLAS builds, results agree to rounding (1e-9
  standard errors, tested), not bit for bit. This is weaker than pprof_spark's guarantee.
- **Refusals.** Rows are refused by name and count, as a `ValueError`: by the light job for a missing
  time, event, stratum, weight or offset, an event that is not 0 or 1, an entry time not before the
  exit; by the pass that builds the working set for a missing feature. A table that gives another
  number of rows to that pass than to the light job stops the fit with a `RuntimeError`: it is not
  a fixed table.

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
| Rows given in the kernel's order need no sort | `test_engine_kernels.py`: both tie methods with weights, offset and entry; baseline, expected events, robust middle | Identical bits to the kernel's own sort of the same rows. Strata in another arrangement and with text labels: within 1e-11 |
| Fits repeat bit for bit with `row_id`, with strata and without, also from a repartitioned input | `test_engine_spark.py`, `test_engine_time.py` | Identical coefficients, covariance and log-likelihood. Rows placed by a hash, or left in the input's partitions: within 1e-9 standard errors |
| Tables centred after the first pass equal tables accumulated centred | `test_engine_time.py`: both tie methods, with weights, offset and entry | Score and rank-one matrix within 1e-9, cumulative hazard within 1e-12, relative |
| Missing features are refused by the pass that builds the working set | Both files: Cox with and without strata, logistic; null and NaN | The same message and count as before |
| Tables too large for the driver are added on the executors | `test_engine_time.py`, with the limit set to zero | Within 1e-9 standard errors |

| The baseline hazard equals pprof_py's | `test_engine_outputs.py`: both tie methods, with strata and without, with weights | Same rows; hazard within 1e-8 relative (measured: 8e-10 with strata, 5e-14 without) |
| Standardized ratios and the score and Wald tests equal pprof_py's | Same file, with one provider at the bound | Ratios, rates and expected counts within 1e-9; score statistics within 1e-9; Wald within 1e-7; identical flags |
| Cox robust variance, martingale residuals and standardized ratios equal pprof_py's | Same file: six strata, Breslow | Robust covariance within 1e-6 relative; residuals within 1e-7; expected events within 1e-6 and ratios within 1e-7 |
| The exact test and direct standardization equal pprof_py's | Same file | p-values within 1e-12 in absolute terms, identical flags; direct ratios and rates within 1e-10. p-values far below 1e-15 are not compared in relative terms |

The repository has 111 tests. In the sandbox of 2026-11-19 (no R installed) 108 pass and the 3
that need R are skipped.

### 5.1 Version 0.6 against version 0.5

The sorted working set must not change a fit. Version 0.9.0 of the repository (commit `387fc16`) and
this version were run on the same ten cases in the sandbox: Cox with strata on 6,000 rows with every
option, both tie methods, numeric and text strata, and all outputs; the logistic model on 8,000 rows
with providers at the bound and direct standardization; Cox (Breslow and Efron) and the logistic
model on 400,000 rows by 50 features and 200,000 by 100.

| Quantity | Result |
|---|---|
| Coefficients, covariance, log-likelihood, iterations, of all ten | Identical bits |
| Baseline hazard, robust covariance, martingale residuals, provider effects with tests | Identical bits |
| Expected events of the Cox standardized ratio; direct standardization | Differ by 1e-15, relative: these sums run over the rows in batch order, which is now sorted |

This is a comparison of two versions, not a test in the repository. Fits without strata changed
layout and agree with version 0.5 to rounding only; their tests against explicit risk sets, the
stratum kernel and pprof_py are those of the table above.

### 5.2 On Databricks

The maintainer ran version 0.9.0 on 2026-11-18: plans `compare_5m` and `unstratified_5m`, 5 million
rows, Runtime 17.3 LTS with Photon, autoscaling from 1 to 10 workers. Medians as reported (the
records are not in the repository yet):

| Model, features | Whole call, s | Of which Newton's method, s |
|---|---|---|
| Cox with strata, 50 | 26.8 | 14.9 |
| Cox with strata, 100 | 24.1 | 13.9 |
| Logistic, 50 | 21.2 | 8.3 |
| Logistic, 100 | 22.1 | 9.5 |
| Cox without strata, 50 | 45.5 | 19.6 |
| Cox without strata, 100 | 60.6 | 34.2 |

Without strata the fits used 8 to 10 of 40 cores. Their task counts, 190 and 366, fit an input of
17 and 33 partitions, which is the number of files the generator writes at these sizes, and the
layout of version 0.5 kept the input's partitions. Version 0.6 has not run on a cluster.

### 5.3 The reference's stopping rule

Without strata the engine and pprof_py differed on the cluster by 2.5e-9 standard errors at 50
features and by 4.3e-5 at 100, with pprof_py at its default tolerance and at its tightest alike.
The cause is in pprof_py's Newton loop (`algorithms/survival/optimization.py`), not in the engine.
The loop halves a step whenever the new log-likelihood is lower than the old one, and tests
convergence afterwards. At the last iteration the true gain is as small as the rounding of the
log-likelihood, so whether the computed value is lower is chance. When it is, the step is halved
until it is not; the change is then far below any `eps`, the fit reports convergence, and the
estimate is left a fraction 1 - 2^-h of the last Newton step short, h being the number of halvings.

The programs are probes A-8 and A-9 in `statcompute/lab/probes/`; their output is in
`docs/research/reference-stopping-rule.md`.

| Evidence (sandbox, 2026-11-19) | Result |
|---|---|
| pprof_py 0.7.0 on the benchmark's design, 100,000 rows by 100 features, 12 seeds; its likelihood calls counted by a wrapper in the calling process | 11 seeds: no halving, within 9e-12 standard errors of the engine. Seed 10: 3 iterations, converged, 18 calls instead of 4; the third step halved 14 times; 3.495e-5 standard errors from the engine. Identical at `eps` 1e-9 and 1e-13 |
| The same case, pprof_py's own kernel | Newton step remaining at its estimate: 3.495e-5 standard errors; at the engine's: 6e-12 |
| The cluster's two datasets regenerated (identifiers `...7e0c0d91fc` and `...c3994f7499`; 2,590,775 and 2,899,213 events), Newton's method with the engine's kernels | Steps of 66.3, 0.699 and 1.263e-4 standard errors at 50 features; 50.2, 0.423 and 4.414e-5 at 100; then 4e-12 and 5e-13, so three iterations, as on the cluster. The third step raises the log-likelihood by 1.4e-7 and 1.6e-8, where neighbouring doubles are 7.5e-9 apart |
| The score at every iterate, recomputed in 80-bit arithmetic by other code on uncentred covariates | Within 3e-11 standard errors of the engine's at 50 features, 8e-12 at 100 |
| The third step at 100 features, cut to 2^-6 | Leaves 5.54e-8 in absolute terms and 4.34e-5 standard errors: the two numbers the cluster reported. Five halvings would leave 4.28e-5, seven 4.38e-5 |

The six halvings on the cluster are an inference from those two numbers. pprof_py could not be run
at that size in the sandbox, and its rounding depends on the cluster's processor and BLAS; counting
its likelihood calls there would settle it (OI-24). At 50 features the third step was taken whole:
2.5e-9 is not a fraction 1 - 2^-h of 1.263e-4.

What follows for comparisons: pprof_py's tightest tolerance does not prevent this, so decision D-06
does not suffice as worded (D-09 is proposed). The tests compare as before: the two estimates must
differ by the Newton step that remains at pprof_py's estimate, and that step may be up to 1e-5
standard errors. On the cluster's 100-feature data it was 4.3e-5, above that bound.

The 80-bit recomputation also shows a property of the engine. The gap of 3e-11 standard errors is the
same at every iterate: it comes from the score's first term, the sum over the deaths of the centred
covariates. For covariates with few distinct values the rounding of that sum does not average out:
over the 50 chunks of the 50-feature data it adds up to 2.776e-8, seven times what independent
errors would give, and the whole gap at beta = 0 is 2.777e-8; without that term 1e-13 standard
errors remain. So the engine's root is the exact root to 3e-11 standard errors here, not to the
1e-12 its own stopping measure shows. That is far below every tolerance class and grows with the
rows (OI-27).

## 6. Not implemented

| Missing | Note |
|---|---|
| A provider with events and non-events outside the bound | Refused with a message. Needs a second pass per evaluation |
| More than 200,000 rows in providers without events or with only events | Refused |
| Times with more than `max_times` distinct values, without strata | Refused. Continuous times need a sorted layout |
| A few large strata | The time grid handles one stratum; a grid per stratum is the natural extension |
| Identical bits across runs without strata and without `row_id` | The rows are placed by a hash but not sorted, so runs agree to rounding |
| For fits without strata: robust variance, residuals, standardized measures. For all fits: robust variance with Efron ties or with clusters of several rows; direct standardization; provider tests; survival curves for new rows | Cox |
| Confidence intervals for effects and ratios; funnel limits; binomial trials; a faster exact test for very large providers | Logistic |
