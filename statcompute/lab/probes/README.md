# Probes for the research roadmap (2026-11-13)

Synthetic data only. Each probe is a single file and prints its result. `bash run_all.sh` runs them all and
writes one log per probe under `out/` (about two minutes on one core). Requirements: Python 3 with NumPy, and a
JDK 17 or later for `GramBench.java` (run by the `java` source launcher; no build step).

| Probe | File | Question |
|---|---|---|
| A-1 | `check_cox_gram.py` | Does the weighted-Gram form equal the classical Cox information with Efron ties, entry times, weights, offsets and strata? |
| A-2 | `check_tilt.py` | Do exponentially tilted Poisson-binomial tails equal recomputed ones? |
| A-3 | `check_twosum.py` | Is the branch-free two-sum's error term Neumaier's, bit for bit? |
| A-4 | `GramBench.java` | What do the two-sum, 64-row chunks and index lists gain in a JVM Gram kernel? Arguments: rows, indicator prevalence |
| A-5 | `iter_cox.py`, `iter_cox2.py`, `iter_cox3.py` | Cox iterations with an exact score and reused, lagged or BFGS-updated information; a prior-period start |
| A-6 | `iter_logit.py` | Joint Newton by block elimination against profile-likelihood Newton for the logistic model with provider effects |
| A-7 | `check_cox_centering.py` | Cancellation in the Cox information when stratum means are spread, with overall and per-stratum centring |

Notes: `iter_cox2.py` and `iter_cox3.py` reuse definitions from `iter_cox.py` and must run from this directory.
`check_cox_centering.py` imports from `check_cox_gram.py`. The timings in A-2 and A-4 vary from run to run; every
other figure is fixed by the seeds.
