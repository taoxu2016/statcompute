# Provider flags: how far statistics move, and how close providers sit to a cut-off

Output of probe A-10 (`statcompute/lab/probes/check_flag_margins.py`), as the program printed it on 2026-11-20 in
the assistant's sandbox; only Spark's start-up warnings are left out. Synthetic data only. The proposal it supports
is in `docs/spec/reproducibility.md`, section 2 (decision D-15).

A flag is a cut-off on a continuous statistic, so two computations of one fit can disagree on a provider's flag only
if that provider's statistic is closer to the cut-off than the two computations are to each other. The probe
measures both distances for the logistic model with provider effects: 800,000 rows, 4,000 providers, 10 features,
level 0.95.

## Environment

```text
Python 3.12.3
numpy 2.5.4 | pyspark 4.0.1 | pyarrow 26.0.1 | pandas 3.0.6 | numba 0.68.1 | pprof_py 0.7.0 from an export of commit 94b5476
openjdk version "21.0.13.0.3" 2026-10-24
model name	: Intel(R) Xeon(R) Processor @ 2.10GHz
cores: 1; memory: 3992 MB
```

## A-10. Differences between computations, and distances from the cut-offs

Program: `check_flag_margins.py 800000 4000 10 DATA`

```text
logistic_fe-n800000-m4000-p10-simple-s79-913a7d44f6: 4000 providers, 0 of them held at the bound; engine 3 iterations, next step 4.5e-11 se
  score: 889 flagged above, 791 below; distance of the statistic from its cut-off: smallest 4.2e-04, 1.0e-03, 1.6e-03; providers within 1e-7: 0, 1e-5: 0, 1e-3: 1, 1e-2: 14
  Wald: 888 flagged above, 787 below; distance of the statistic from its cut-off: smallest 1.5e-04, 7.9e-04, 1.0e-03; providers within 1e-7: 0, 1e-5: 0, 1e-3: 3, 1e-2: 18
  exact: 878 flagged above, 802 below; distance of the statistic from its cut-off: smallest 3.7e-05, 1.6e-04, 2.0e-04; providers within 1e-7: 0, 1e-5: 0, 1e-3: 19, 1e-2: 183
  engine with 16 partitions against 40: largest difference in the statistic: score 4.4e-15 (0 flags differ); Wald 4.9e-15 (0 flags differ); exact 1.4e-15 (0 flags differ)
  engine against pprof_py at tol 1e-12: coefficients 4.5e-11 se apart
  engine against pprof_py at tol 1e-12: largest difference in the statistic: score 6.0e-13 (0 flags differ); Wald 5.8e-13 (0 flags differ); exact 4.4e-13 (0 flags differ)
  pprof_py at tol 1e-08 against itself at 1e-12: coefficients 0.0e+00 se apart
  pprof_py at tol 1e-08 against itself at 1e-12: largest difference in the statistic: score 0.0e+00 (0 flags differ); Wald 0.0e+00 (0 flags differ); exact 0.0e+00 (0 flags differ)
  pprof_py at tol 1e-05 against itself at 1e-12: coefficients 2.7e-11 se apart
  pprof_py at tol 1e-05 against itself at 1e-12: largest difference in the statistic: score 3.7e-13 (0 flags differ); Wald 8.8e-09 (0 flags differ); exact 2.8e-13 (0 flags differ)
```

## Reading

- Between two layouts of the engine the statistics differ by 5e-15 at most; between the engine and pprof_py at its
  tightest tolerance by 6e-13; between pprof_py at 1e-5, the default of pprof in R, and at 1e-12 by 9e-9 (the Wald
  statistic). No flag differs in any of these.
- The provider nearest a cut-off is 1.5e-4 away on the z scale and 3.7e-5 on the scale of the exact p-value: four
  orders of magnitude more than the largest difference seen.
- How often would a provider lie within the proposed tolerance of a cut-off? Scaling the counts within 1e-2 down
  linearly: 14 and 18 providers within 1e-2 give 1.4e-4 and 1.8e-4 within 1e-7 for the score and Wald tests, and 183
  within 1e-2 give 1.8e-5 within 1e-9 for the exact test. So about one fit in 5,000 to 7,000 has a borderline
  provider for a z test at this size, and one in 55,000 for the exact test; with 8,000 providers, twice as often.
  This is arithmetic from one synthetic dataset, not a measurement of production data.

What the probe does not show: Cox models, for which the engine has no provider tests yet; data in which many
providers have the same statistic, as can happen with small providers and the exact test; and the cluster.
