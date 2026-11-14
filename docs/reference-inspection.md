# Reference inspection: baseline functionality and performance evidence

State on 2026-11-14. This is the record of step 1 ("inspect the existing repositories") and of what
step 2 ("establish baseline functionality and performance") can and cannot say yet.

## 1. What was inspected

| Reference | Commit | How | Result |
|---|---|---|---|
| pprof (R) 2.0.12 | `d780cf5` | Clone read for the two models; installed from an exported copy under R 4.3.3 and run | Runs through its adapter; agrees with pprof_py (section 3.3) |
| pprof_py 0.7.0 | `94b5476` | Clone read in full for the two models; installed from an exported copy and run | Used as the numerical reference |
| pprof_spark, round 146 | `e5c9484` | Clone read: engine, layouts, guards, Python wrappers, tutorials, state files; full history fetched to read round 81 | Not buildable in the sandbox (no Maven Central); adapter written from the wrapper's signatures and run against a mock |

No file in any clone was changed. Both installable references were installed from `git archive`
output, not from the clones.

## 2. Baseline functionality

`docs/capabilities.md` holds the matrix. In short: the three packages implement the same two
models. pprof (R) fits the Cox model through `survival::coxph` and has its own SerBIN and BAN code
for the logistic model; pprof_py reimplements both; pprof_spark reimplements pprof_py on Spark and
was validated against it fixture by fixture in its own repository. They differ in where they run,
in default tolerances, and in what limits them.

## 3. Performance evidence so far

None of this is a baseline on the standard cluster. It is what exists.

**3.1 Kernel rates on one thread (the feasibility study, 2026-11-13).** At 180 features a Cox
evaluation costs 51 microseconds per row in pprof_spark and 17 in pprof_py; a logistic iteration 33
and 2.3. The study's prototypes of the recommended kernels run at 5.4 and 4.85.

**3.2 Six recorded cluster runs (2026-10-21 and 2026-11-13).** Build `b670dbb` (round 81, 68
commits behind main), Databricks Runtime 16.4 with Photon (Spark 3.5.2), m5dn.xlarge workers, 2 to
10 of them, a different number in every row. Three of the six rows are not completed fits. The
three that are used the equivalent of 2 to 6 cores: a Cox fit of 10 million rows and 100 features
took 165 s where one pprof_py thread would need about 210 s. The research roadmap (F-1) has the
analysis. These runs do not describe current main, Runtime 17.3 or a fixed cluster.

**3.3 This harness in the sandbox (2026-11-14).** One core, PySpark 4.0.1 in local mode, R 4.3.3,
200,000 rows, 400 providers, each implementation at its tightest tolerance, one run each. Seconds
for the fit phase, with iterations in brackets:

| Model | Features | pprof (R) | pprof_py | lab_numpy | lab_spark |
|---|---:|---:|---:|---:|---:|
| Cox, stratified, Breslow | 20 | 3.09 (3) | 1.10 (4) | 0.51 (3) | 5.61 (3) |
| Cox, stratified, Breslow | 50 | 5.46 (3) | 3.01 (3) | 1.26 (3) | 4.57 (3) |
| Logistic fixed effects | 20 | 7.58 (5) | 0.55 (5) | 0.19 (3) | 2.67 (3) |
| Logistic fixed effects | 50 | 4.06 (5) | 1.01 (5) | 0.45 (3) | 3.23 (3) |

Against pprof_py, largest differences over the four datasets:

| Implementation | Coefficients | The same, in standard errors | Standard errors, relative |
|---|---:|---:|---:|
| pprof (R) | 1.8e-13 | 2.6e-11 | 3.3e-14 |
| lab_numpy | 6.1e-13 | 7.6e-11 | 1.2e-13 |
| lab_spark | 6.1e-13 | 7.6e-11 | 1.2e-13 |

The records are in `results/2026-11-14-sandbox/` and the rendered report in
`docs/reports/2026-11-14-sandbox.md`.

What this shows: the harness runs all three kinds of implementation, the three installed ones agree
to rounding, and the recommended algorithms return the validated estimates. What it does not show:
anything about a cluster, or stable timings. These are single runs on a shared core (R's logistic
fit took longer at 20 features than at 50, which is noise). `lab_spark`'s times at this size are
Spark's fixed costs.

## 4. What the baseline still needs

1. The standard cluster confirmed and of fixed size (OI-02).
2. Plan `standard` at 5 million rows for `pprof_spark`, `lab_spark` and, where the memory guard
   allows, `pprof_py` and `pprof_r`; then 20 million and upwards for the Spark implementations.
3. pprof_spark's own parity notebook on Runtime 17.3, because its JAR targets Spark 4.1 (OI-03).
4. pprof (R) installed where the driver's `Rscript` finds it (OI-11).
