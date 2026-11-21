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
| A-8 | `check_reference_stop.py` | Can pprof_py's Cox estimate stop short of the root whatever its tolerance? Its likelihood calls are counted from outside and its own kernel is asked for the step that remains. Arguments: rows, features, data folder, seeds |
| A-10 | `check_flag_margins.py` | What does "stable provider-level flags" need? Differences in the provider test statistics between layouts, implementations and tolerances, against the distance of providers from the cut-offs. Arguments: rows, providers, features, data folder |
| A-11 | `packed_working_set.py` | What a pass costs before its kernel runs: a pass that only receives the rows against one with the Cox kernel, over the usual working set and over the same rows packed into blocks of bytes. Arguments: rows, features, data folder, and optionally partitions and rows per block |
| A-12 | `kernel_against_prototype.py` | One sorted partition of 50,000 rows through the engine's Cox kernel and through the prototype's. Argument: features |
| A-9 | `newton_steps_exact.py` | The Newton steps of a fit without strata on a regenerated benchmark dataset, the score recomputed in 80-bit arithmetic, and what a last step cut short would leave. Arguments: rows, providers, features, work folder, and optionally a reported difference (absolute, in standard errors) |

Notes: `iter_cox2.py` and `iter_cox3.py` reuse definitions from `iter_cox.py` and must run from this directory.
A-8 to A-12 (rounds 12, 13 and 19) are not part of `run_all.sh`: they need this repository on the Python path, all but
A-12 PySpark, A-8 and A-10 also pprof_py with numba, and they take minutes. Their output is in
`docs/research/reference-stopping-rule.md`, `docs/research/flag-margins.md` and `docs/research/pass-cost.md`.
`check_cox_centering.py` imports from `check_cox_gram.py`. The timings in A-2 and A-4 vary from run to run; every
other figure is fixed by the seeds.
