# pprof_py's stopping rule and the Newton steps on the cluster's data

Output of probes A-8 and A-9 (`statcompute/lab/probes/`), as the programs printed it on 2026-11-19 in the
assistant's sandbox; only Spark's start-up warnings are left out. Synthetic data only. What the output means is
in `docs/spec/engine.md`, section 5.3.

The question came from the cluster runs of 2026-11-18: without strata the engine and pprof_py differed by
2.52e-9 standard errors at 50 features and by 4.34e-5 at 100 (5.54e-8 in absolute terms), the same with pprof_py at
its default tolerance and at its tightest.

## Environment

```text
Python 3.12.3
numpy 2.5.4 | pyspark 4.0.1 | pyarrow 26.0.1 | pandas 3.0.6 | numba 0.68.1 | pprof_py 0.7.0 from an export of commit 94b5476
openjdk version "21.0.13.0.3" 2026-10-24
model name	: Intel(R) Xeon(R) Processor @ 2.10GHz
cores: 1; memory: 3992 MB
```

## A-8. pprof_py's likelihood calls, and the step that remains at its estimate

Program: `check_reference_stop.py 100000 100 DATA 79 1 2 3 4 5 6 7 8 9 10 11`

A fit of three iterations makes four calls in its loop when no step is halved. In seed 10 the third step was
halved 14 times: the loop made 18 calls, moved 2^-14 of the step and reported convergence, at both tolerances.

```text
cox-n100000-m160-p100-simple-s79-875718eb53: 57977 events; engine: 3 iterations, next step 2.7e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 3.877e-05 se
      difference from the engine 6.074e-12 se; step remaining at pprof_py's estimate 2.856e-12 se by its own kernel, 5.597e-12 by the engine's; at the engine's estimate 6.5e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 3.877e-05 se
      difference from the engine 6.074e-12 se; step remaining at pprof_py's estimate 2.856e-12 se by its own kernel, 5.597e-12 by the engine's; at the engine's estimate 6.5e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s1-ff16494719: 58527 events; engine: 3 iterations, next step 1.2e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 2.502e-05 se
      difference from the engine 6.553e-12 se; step remaining at pprof_py's estimate 1.271e-12 se by its own kernel, 5.916e-12 by the engine's; at the engine's estimate 6.5e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 2.502e-05 se
      difference from the engine 6.553e-12 se; step remaining at pprof_py's estimate 1.271e-12 se by its own kernel, 5.916e-12 by the engine's; at the engine's estimate 6.5e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s2-19f8695667: 57691 events; engine: 3 iterations, next step 1.5e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 2.848e-05 se
      difference from the engine 7.773e-12 se; step remaining at pprof_py's estimate 1.600e-12 se by its own kernel, 6.426e-12 by the engine's; at the engine's estimate 8.2e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 2.848e-05 se
      difference from the engine 7.773e-12 se; step remaining at pprof_py's estimate 1.600e-12 se by its own kernel, 6.426e-12 by the engine's; at the engine's estimate 8.2e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s3-56ac4fc29f: 58315 events; engine: 3 iterations, next step 2.0e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 3.251e-05 se
      difference from the engine 7.040e-12 se; step remaining at pprof_py's estimate 2.022e-12 se by its own kernel, 5.865e-12 by the engine's; at the engine's estimate 7.3e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 3.251e-05 se
      difference from the engine 7.040e-12 se; step remaining at pprof_py's estimate 2.022e-12 se by its own kernel, 5.865e-12 by the engine's; at the engine's estimate 7.3e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s4-b91d9a0f83: 58450 events; engine: 3 iterations, next step 3.6e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 4.394e-05 se
      difference from the engine 6.356e-12 se; step remaining at pprof_py's estimate 3.721e-12 se by its own kernel, 8.548e-12 by the engine's; at the engine's estimate 6.6e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 4.394e-05 se
      difference from the engine 6.356e-12 se; step remaining at pprof_py's estimate 3.721e-12 se by its own kernel, 8.548e-12 by the engine's; at the engine's estimate 6.6e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s5-9bf654dbe3: 57050 events; engine: 3 iterations, next step 4.0e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 4.589e-05 se
      difference from the engine 8.675e-12 se; step remaining at pprof_py's estimate 3.831e-12 se by its own kernel, 7.074e-12 by the engine's; at the engine's estimate 9.1e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 4.589e-05 se
      difference from the engine 8.675e-12 se; step remaining at pprof_py's estimate 3.831e-12 se by its own kernel, 7.074e-12 by the engine's; at the engine's estimate 9.1e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s6-ab08a6503e: 57705 events; engine: 3 iterations, next step 6.6e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 6.106e-05 se
      difference from the engine 7.234e-12 se; step remaining at pprof_py's estimate 6.438e-12 se by its own kernel, 6.732e-12 by the engine's; at the engine's estimate 1.1e-11 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 6.106e-05 se
      difference from the engine 7.234e-12 se; step remaining at pprof_py's estimate 6.438e-12 se by its own kernel, 6.732e-12 by the engine's; at the engine's estimate 1.1e-11 by pprof_py's kernel
cox-n100000-m160-p100-simple-s7-cce6464d1a: 58056 events; engine: 3 iterations, next step 5.2e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 5.132e-05 se
      difference from the engine 7.146e-12 se; step remaining at pprof_py's estimate 5.450e-12 se by its own kernel, 7.097e-12 by the engine's; at the engine's estimate 7.1e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 5.132e-05 se
      difference from the engine 7.146e-12 se; step remaining at pprof_py's estimate 5.450e-12 se by its own kernel, 7.097e-12 by the engine's; at the engine's estimate 7.1e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s8-93bcc9d0c8: 57524 events; engine: 3 iterations, next step 9.6e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 7.000e-05 se
      difference from the engine 6.926e-12 se; step remaining at pprof_py's estimate 9.531e-12 se by its own kernel, 1.108e-11 by the engine's; at the engine's estimate 7.8e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 7.000e-05 se
      difference from the engine 6.926e-12 se; step remaining at pprof_py's estimate 9.531e-12 se by its own kernel, 1.108e-11 by the engine's; at the engine's estimate 7.8e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s9-7fbfd34424: 58682 events; engine: 3 iterations, next step 1.1e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 2.501e-05 se
      difference from the engine 6.613e-12 se; step remaining at pprof_py's estimate 1.809e-12 se by its own kernel, 5.544e-12 by the engine's; at the engine's estimate 6.2e-12 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 2.501e-05 se
      difference from the engine 6.613e-12 se; step remaining at pprof_py's estimate 1.809e-12 se by its own kernel, 5.544e-12 by the engine's; at the engine's estimate 6.2e-12 by pprof_py's kernel
cox-n100000-m160-p100-simple-s10-4e5b9ed132: 57404 events; engine: 3 iterations, next step 2.4e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 18 likelihood calls in the loop (4 without halving), last move 6.1e-05 of its Newton step of 3.495e-05 se
      difference from the engine 3.495e-05 se; step remaining at pprof_py's estimate 3.495e-05 se by its own kernel, 3.495e-05 by the engine's; at the engine's estimate 6.4e-12 by pprof_py's kernel
      log-likelihood at each call minus the last: -1.76e+03 -0.132 -1.05e-09 -3.61e-09 -5.7e-09 -1.51e-09 -5.7e-09 -8.15e-09 -3.49e-09 -4.66e-09 -6.29e-09 -1.28e-09 -7.33e-09 -5.12e-09 -6.98e-09 -3.26e-09 -6.05e-09 0
  pprof_py eps=1e-13: 3 iterations, converged=True, 18 likelihood calls in the loop (4 without halving), last move 6.1e-05 of its Newton step of 3.495e-05 se
      difference from the engine 3.495e-05 se; step remaining at pprof_py's estimate 3.495e-05 se by its own kernel, 3.495e-05 by the engine's; at the engine's estimate 6.4e-12 by pprof_py's kernel
      log-likelihood at each call minus the last: -1.76e+03 -0.132 -1.05e-09 -3.61e-09 -5.7e-09 -1.51e-09 -5.7e-09 -8.15e-09 -3.49e-09 -4.66e-09 -6.29e-09 -1.28e-09 -7.33e-09 -5.12e-09 -6.98e-09 -3.26e-09 -6.05e-09 0
cox-n100000-m160-p100-simple-s11-0aca4096ca: 57211 events; engine: 3 iterations, next step 9.2e-12 se
  pprof_py eps=1e-09: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 7.163e-05 se
      difference from the engine 7.033e-12 se; step remaining at pprof_py's estimate 9.243e-12 se by its own kernel, 8.115e-12 by the engine's; at the engine's estimate 1.1e-11 by pprof_py's kernel
  pprof_py eps=1e-13: 3 iterations, converged=True, 4 likelihood calls in the loop (4 without halving), last move 1 of its Newton step of 7.163e-05 se
      difference from the engine 7.033e-12 se; step remaining at pprof_py's estimate 9.243e-12 se by its own kernel, 8.115e-12 by the engine's; at the engine's estimate 1.1e-11 by pprof_py's kernel
12 datasets, 2 tolerances each: a step was halved in 2 of 24 fits
```

## A-9. The cluster's dataset of 100 features, regenerated

Program: `newton_steps_exact.py 5000000 8000 100 WORK 5.54e-08 4.34e-05`

The dataset identifier is the one in the cluster's report. Its manifest there should show the same events and
checksum. The third step, cut to 2^-6, leaves the two numbers the cluster reported. Both datasets of 5 million
rows had been regenerated earlier the same day, by the same generator and into the same files, so these two runs
found them and skipped step 1; step 1 as written was run on a dataset of 200,000 rows by 20 features.

```text
cox-n5000000-m8000-p100-simple-s79-c3994f7499: 5000000 rows, 2899213 events, 1825 distinct times; manifest: events 2899213, checksum 6760246443547002433
engine kernels: 3 iterations, 4 evaluations, converged=True (69 s)
  iterate 0: log-likelihood -42455746.49101926; next Newton step 5.0185e+01 se by the engine, 5.0185e+01 by the 80-bit score (5.2053e-02 absolute); the two scores differ by 7.60e-12 se as a step
  iterate 1: log-likelihood -42366441.65071454; next Newton step 4.2291e-01 se by the engine, 4.2291e-01 by the 80-bit score (5.3942e-04 absolute); the two scores differ by 7.54e-12 se as a step
  iterate 2: log-likelihood -42366439.07724804; next Newton step 4.4139e-05 se by the engine, 4.4139e-05 by the 80-bit score (5.6305e-08 absolute); the two scores differ by 7.56e-12 se as a step
  iterate 3: log-likelihood -42366439.07724803; next Newton step 4.5515e-13 se by the engine, 7.6173e-12 by the 80-bit score (9.7132e-15 absolute); the two scores differ by 7.52e-12 se as a step
gain in log-likelihood of each step: 8.93e+04, 2.573, 1.619e-08; neighbouring doubles at the log-likelihood are 7.45e-09 apart
last step: 4.4139e-05 se, 5.6305e-08 absolute; reported difference 4.34e-05 se, 5.54e-08 absolute
  cut to 2^-1: leaves 2.2069e-05 se, 2.8153e-08 absolute
  cut to 2^-2: leaves 3.3104e-05 se, 4.2229e-08 absolute
  cut to 2^-3: leaves 3.8621e-05 se, 4.9267e-08 absolute
  cut to 2^-4: leaves 4.1380e-05 se, 5.2786e-08 absolute
  cut to 2^-5: leaves 4.2759e-05 se, 5.4546e-08 absolute
  cut to 2^-6: leaves 4.3449e-05 se, 5.5426e-08 absolute
  cut to 2^-7: leaves 4.3794e-05 se, 5.5865e-08 absolute
  cut to 2^-8: leaves 4.3966e-05 se, 5.6085e-08 absolute
```

## A-9. The cluster's dataset of 50 features, regenerated

Program: `newton_steps_exact.py 5000000 8000 50 WORK 2.69e-12 2.52e-09`

Here the reported difference is no fraction 1 - 2^-h of the third step: that step was taken whole.

```text
cox-n5000000-m8000-p50-simple-s79-7e0c0d91fc: 5000000 rows, 2590775 events, 1825 distinct times; manifest: events 2590775, checksum 8985229526020740072
engine kernels: 3 iterations, 4 evaluations, converged=True (35 s)
  iterate 0: log-likelihood -38029153.84373859; next Newton step 6.6312e+01 se by the engine, 6.6312e+01 by the 80-bit score (7.3457e-02 absolute); the two scores differ by 2.96e-11 se as a step
  iterate 1: log-likelihood -37948143.43114262; next Newton step 6.9902e-01 se by the engine, 6.9902e-01 by the 80-bit score (9.4027e-04 absolute); the two scores differ by 2.96e-11 se as a step
  iterate 2: log-likelihood -37948138.295134254; next Newton step 1.2629e-04 se by the engine, 1.2629e-04 by the 80-bit score (1.6991e-07 absolute); the two scores differ by 2.96e-11 se as a step
  iterate 3: log-likelihood -37948138.29513412; next Newton step 4.1323e-12 se by the engine, 2.9713e-11 by the 80-bit score (3.1740e-14 absolute); the two scores differ by 2.96e-11 se as a step
gain in log-likelihood of each step: 8.101e+04, 5.136, 1.391e-07; neighbouring doubles at the log-likelihood are 7.45e-09 apart
last step: 1.2629e-04 se, 1.6991e-07 absolute; reported difference 2.52e-09 se, 2.69e-12 absolute
  cut to 2^-1: leaves 6.3144e-05 se, 8.4954e-08 absolute
  cut to 2^-2: leaves 9.4716e-05 se, 1.2743e-07 absolute
  cut to 2^-3: leaves 1.1050e-04 se, 1.4867e-07 absolute
  cut to 2^-4: leaves 1.1840e-04 se, 1.5929e-07 absolute
  cut to 2^-5: leaves 1.2234e-04 se, 1.6460e-07 absolute
  cut to 2^-6: leaves 1.2431e-04 se, 1.6725e-07 absolute
  cut to 2^-7: leaves 1.2530e-04 se, 1.6858e-07 absolute
  cut to 2^-8: leaves 1.2579e-04 se, 1.6924e-07 absolute
```

## What the output does not show

pprof_py was not run on the two datasets of 5 million rows: its fit needs about 18 GB at 100 features and the
sandbox has 4. That its third step was halved six times on the cluster is therefore an inference from the two
reported numbers. Its rounding depends on the processor and the BLAS, so the sandbox could not have reproduced
that fit's halvings in any case; counting its likelihood calls on the cluster would (OPEN_ITEMS.md, OI-24).
