# Evidence for the research roadmap of 2026-11-13

Output of the sandbox probes cited in `statcompute-research-roadmap.md` as [probe A-n], exactly as the programs printed it on 2026-11-13. Source: `roadmap-probes.tar.gz`. Synthetic data only; one core; no engine code and no cluster. In A-4, V1 is the per-element Neumaier update, V2 the branch-free two-sum, V3 plain updates in 64-row chunks folded by the two-sum, and V4 the same with indicator columns as index lists; factors are relative to V1, and the runs at p = 20 last about 5 ms and are too short to rely on.

## Environment

Program: `run_all.sh`

```text
Python 3.12.3
numpy 2.5.4
openjdk version "21.0.12.1" 2026-08-18
model name	: Intel(R) Xeon(R) Processor @ 2.30GHz
cores: 1
```

## A-1. Weighted-Gram form of the Cox information

Program: `check_cox_gram.py`

```text
Breslow: max|I_gram - I_classical|/max|I| = 1.24e-15; max|U_gram - U|/max|U| = 0.00e+00; max|I_gram - FD|/max|I| = 8.07e-12; sum(E) = 739.0000000000 vs weighted events = 739.0000000000; min E = 4.544e-03
Efron: max|I_gram - I_classical|/max|I| = 7.82e-16; max|U_gram - U|/max|U| = 0.00e+00; max|I_gram - FD|/max|I| = 1.28e-11; sum(E) = 739.0000000000 vs weighted events = 739.0000000000; min E = 4.544e-03
```

## A-2. Exponential tilting of Poisson-binomial tails

Program: `check_tilt.py`

```text
g = -0.400: direct 1.000000000000002e+00  tilted 9.999999999999730e-01
g = +0.200: direct 6.092552288532727e-01  tilted 6.092552288532729e-01
g = +0.800: direct 1.009438670377147e-55  tilted 1.009438670377140e-55
n = 4000, observed o = 958; worst relative difference over 25 shifts: 6.63e-14
40 evaluations: re-running the recursion 0.440 s; one recursion + 40 tilts 0.012 s (NumPy, one core)
```

## A-3. Two-sum against Neumaier's update

Program: `check_twosum.py`

```text
wide exponents        : 4,000,000 pairs, bitwise mismatches in the error term: 0
similar magnitudes    : 4,000,000 pairs, bitwise mismatches in the error term: 0
near cancellation     : 4,000,000 pairs, bitwise mismatches in the error term: 0
subnormals and zeros  : 4,000,000 pairs, bitwise mismatches in the error term: 0
running state identical: True | total pairs 16,000,000 | mismatches 0
```

## A-4. JVM Gram kernels, indicator prevalence 0.30

Program: `GramBench.java 20000 0.30`

```text
p= 20  indicator density 0.30 | V1 Neumaier  0.010s (2.36 ns/elem) | V2 TwoSum  0.005s x2.0 bit-identical=true | V3 chunked  0.003s x3.5 maxrel 1.6e-16 | V4 sparse-aware  0.002s x4.6 maxrel 1.6e-16
p= 50  indicator density 0.30 | V1 Neumaier  0.035s (1.38 ns/elem) | V2 TwoSum  0.028s x1.2 bit-identical=true | V3 chunked  0.015s x2.4 maxrel 1.6e-16 | V4 sparse-aware  0.007s x5.3 maxrel 1.6e-16
p=100  indicator density 0.30 | V1 Neumaier  0.127s (1.26 ns/elem) | V2 TwoSum  0.098s x1.3 bit-identical=true | V3 chunked  0.051s x2.5 maxrel 1.6e-16 | V4 sparse-aware  0.023s x5.5 maxrel 1.6e-16
```

## A-4. JVM Gram kernels, indicator prevalence 0.05

Program: `GramBench.java 20000 0.05`

```text
p= 20  indicator density 0.05 | V1 Neumaier  0.006s (1.39 ns/elem) | V2 TwoSum  0.005s x1.2 bit-identical=true | V3 chunked  0.003s x2.1 maxrel 1.6e-16 | V4 sparse-aware  0.002s x3.6 maxrel 1.6e-16
p= 50  indicator density 0.05 | V1 Neumaier  0.034s (1.35 ns/elem) | V2 TwoSum  0.028s x1.2 bit-identical=true | V3 chunked  0.015s x2.4 maxrel 1.6e-16 | V4 sparse-aware  0.005s x7.3 maxrel 1.6e-16
p=100  indicator density 0.05 | V1 Neumaier  0.125s (1.24 ns/elem) | V2 TwoSum  0.098s x1.3 bit-identical=true | V3 chunked  0.051s x2.5 maxrel 1.6e-16 | V4 sparse-aware  0.016s x7.6 maxrel 1.6e-16
```

## A-5a. Cox iterations stopped by the relative change in the log-likelihood

Program: `iter_cox.py`

```text
n = 200,000, p = 20, events = 56,975, distinct event times = 1775
variant                            iters  O(np^2) evals  max|score|  max|beta - ref|
Newton (exact curvature each step)     3              3    7.77e-07         5.99e-11
lagged-hazard Gram                     4              4    2.64e-03         5.39e-08
chord, curvature frozen after 2        3              2    9.19e-04         7.04e-08
chord, curvature frozen after 1        4              1    2.60e-01         2.03e-05

n = 200,000, p = 60, events = 57,367, distinct event times = 1776
variant                            iters  O(np^2) evals  max|score|  max|beta - ref|
Newton (exact curvature each step)     3              3    5.04e-07         1.05e-11
lagged-hazard Gram                     4              4    3.31e-03         6.80e-08
chord, curvature frozen after 2        3              2    8.12e-04         1.78e-08
chord, curvature frozen after 1        4              1    7.41e-02         2.50e-06
```

## A-5b. Cox iterations stopped by a common Newton-decrement rule

Program: `iter_cox2.py`

```text
n = 200,000, p = 20, effect scale 0.5, events = 57,266, max|beta| = 0.34
variant                              steps  O(np^2) evals  O(np) passes  max|score|  max|beta - ref|
Newton, exact curvature each step        3              4             4    1.31e-07         9.57e-12
lagged-hazard Gram (one pass/step)       5              6             6    9.52e-08         1.98e-12
chord, curvature frozen after 3          3              3             4    1.31e-07         9.57e-12
chord, curvature frozen after 2          3              2             4    3.68e-04         1.80e-08
chord, curvature frozen after 1          7              1             8    1.82e-04         1.41e-08

n = 200,000, p = 60, effect scale 0.5, events = 57,757, max|beta| = 0.20
variant                              steps  O(np^2) evals  O(np) passes  max|score|  max|beta - ref|
Newton, exact curvature each step        3              4             4    5.92e-07         1.35e-11
lagged-hazard Gram (one pass/step)       5              6             6    2.55e-07         5.17e-12
chord, curvature frozen after 3          3              3             4    5.92e-07         1.35e-11
chord, curvature frozen after 2          4              2             5    2.55e-06         8.36e-11
chord, curvature frozen after 1          6              1             7    4.46e-04         3.52e-08

n = 200,000, p = 60, effect scale 2.0, events = 79,461, max|beta| = 0.75
variant                              steps  O(np^2) evals  O(np) passes  max|score|  max|beta - ref|
Newton, exact curvature each step        5              6             6    2.64e-06         1.68e-10
lagged-hazard Gram (one pass/step)      13             14            14    5.10e-05         3.25e-09
chord, curvature frozen after 3          9              3            10    1.35e-04         8.42e-09
chord, curvature frozen after 2         19              2            20    3.31e-04         2.03e-08
chord, curvature frozen after 1         48              1            49    5.31e-04         3.42e-08
```

## A-5c. BFGS-updated information and a prior-period start

Program: `iter_cox3.py`

```text
n = 200,000, p = 60, effect scale 0.5
variant                                      steps  O(np^2) evals  O(np) passes  max|beta - ref|
Newton                                           3              4             4         1.59e-11
exact curvature x3, then BFGS updates            3              3             4         1.59e-11
exact curvature x2, then BFGS updates            3              2             4         1.57e-08
exact curvature x1, then BFGS updates            5              1             6         5.49e-08
warm start from last period: Newton              2              3             3         6.72e-11
warm start: exact curvature x1, then BFGS        2              1             3         5.86e-08
   distance of the warm start from the solution: max|beta_prev - ref| = 0.010

n = 200,000, p = 60, effect scale 2.0
variant                                      steps  O(np^2) evals  O(np) passes  max|beta - ref|
Newton                                           5              6             6         1.39e-10
exact curvature x3, then BFGS updates            7              3             8         6.26e-10
exact curvature x2, then BFGS updates            8              2             9         3.14e-09
exact curvature x1, then BFGS updates            9              1            10         4.03e-09
warm start from last period: Newton              2              3             3         3.44e-11
warm start: exact curvature x1, then BFGS        3              1             4         8.47e-11
   distance of the warm start from the solution: max|beta_prev - ref| = 0.012
```

## A-6. Joint and profile-likelihood Newton, logistic model

Program: `iter_logit.py`

```text
n = 207,330, providers = 1,994, p = 20, event rate = 0.303
variant                                        steps  O(np^2) evals  data passes  max|beta - joint|  max|gamma - joint|
joint Newton, block elimination (SerBIN step)      6              7           13                  -                   -
profile Newton                                     4              5            5           1.97e-12            1.38e-10
profile, Schur complement x2 then BFGS             5              2            6           2.74e-10            4.73e-10
profile, Schur complement x1 then BFGS             6              1            7           7.15e-09            2.00e-08
```

## A-7. Spread of stratum means and centring, Cox information

Program: `check_cox_centering.py`

```text
 spread of stratum means / within-stratum SD   (a) sweep, overall   (b) Gram, overall  (c) Gram, per stratum
                                           0              3.9e-16             9.1e-16                2.3e-15
                                           1              5.3e-16             2.0e-15                5.3e-16
                                          10              1.0e-15             1.2e-13                3.3e-15
                                         100              2.0e-13             2.4e-12                3.6e-15
                                        1000              5.0e-11             8.1e-10                3.4e-15
```
