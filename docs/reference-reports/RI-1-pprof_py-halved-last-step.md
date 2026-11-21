# RI-1, for pprof_py's maintainers: a Cox fit can report convergence with its last Newton step halved by rounding

A draft to send; the maintainer of statcompute decides whether and where. Register entry: RI-1 in
`docs/reference-issues.md`. Nothing in pprof_py was changed to find this.

## Summary

`CoxPH.fit` can stop with `converged_ = True` while its estimate is up to one whole Newton step short
of the maximum. It happens at the last iteration, when the gain of the step is as small as the
rounding error of the log-likelihood. Tightening `eps` does not prevent it. The amounts are small,
3e-5 to 5e-5 standard errors in the cases seen, but they are far above what a converged Newton
iteration leaves, and they show as a disagreement when the fit is compared with another exact
implementation.

## Where

`pprof_py/algorithms/survival/optimization.py`, `newton_raphson`, at commit `94b5476` (lines 57 to 81):

1. A full step is tried and the log-likelihood evaluated there (line 65).
2. While the new log-likelihood is lower than the old one, the step is halved and tried again, up
   to `max_step_halving` times (lines 68 to 72).
3. Only then is the relative change compared with `eps` (lines 75 to 80).

At the last iteration the true gain of the full step is of the order of the rounding error of the
log-likelihood. Whether the computed value comes out lower is then chance. If it does, step 2 halves
the step until it does not; the change that step 3 sees is then far below any `eps`, and the fit
reports convergence at the old estimate plus 2^-h of the step, h being the number of halvings.

## How to see it without changing the package

Count the calls `CoxPH.fit` makes to `cox_partial_likelihood`. A fit of n iterations makes n + 2
(the start, one per iteration, the null model); every call beyond that is a halved step.
statcompute's probe does this with a wrapper around the function in the calling process:

    python statcompute/lab/probes/check_reference_stop.py 100000 100 DATA 79 1 2 3 4 5 6 7 8 9 10 11

## What was seen

| Where | Data | Result |
|---|---|---|
| A sandbox, pprof_py 0.7.0 with numba 0.68.1, NumPy 2.5.4 | Synthetic, 100,000 rows by 100 features, no strata, Breslow ties, 12 seeds | 11 seeds: no step halved. One seed: 3 iterations, `converged_` true, 18 calls instead of 4 in the loop: the third step halved 14 times. The same at `eps` 1e-9 and 1e-13 |
| The same fit, pprof_py's own likelihood, score and information | | The Newton step that remains at its estimate is 3.5e-5 standard errors; at the estimate of an independent implementation it is 6e-12 |
| Databricks Runtime 17.3, on the driver | Synthetic, 5 million rows by 100 features, no strata | 3 iterations, `converged_` true, 11 calls: the third step halved 6 times. The estimate is 4.34e-5 standard errors from the maximum, at `eps` 1e-9 and 1e-13 alike. At 50 features: 5 calls, none halved, and 2.5e-9 |

On the regenerated 5 million rows the third Newton step is 4.414e-5 standard errors long and raises
the log-likelihood by 1.6e-8, where neighbouring doubles are 7.5e-9 apart. Cut to 2^-6 it leaves
4.34e-5, which is the difference observed.

## What it is not

Not an error in the likelihood, the score or the information: evaluated at the other estimate,
pprof_py's own functions give a remaining step of 6e-12 standard errors. Only the acceptance of the
last step is affected.

## Two possible remedies

- Test convergence on the full step first, and look for a decrease only if that test fails. This is
  the order pprof_spark's loop uses. Whether survival's `coxfit6.c` does the same was not checked
  here and may be worth comparing.
- Or accept a decrease that is within rounding, for example
  `new_loglik < loglik - 1e-12 * abs(loglik)` as the condition for halving.

Either keeps the safeguard for steps that really overshoot.

## Evidence in statcompute

`statcompute/lab/probes/check_reference_stop.py` and `newton_steps_exact.py`, with their output in
`docs/research/reference-stopping-rule.md`; the cluster's count in
`docs/reports/2026-11-21-databricks-warm-runs.md`, section 1; the reasoning in
`docs/spec/engine.md`, section 5.3.
