# The warm runs on the cluster, 2026-11-21

Sources: four reports the maintainer rendered with notebook `03_report` and pasted on 2026-11-21,
one per plan. The JSON records are not in the repository. The engine's rows were typed from the
reports to make the tables below.

| Plan | What ran | Version |
|---|---|---|
| `reference_stop_5m` | pprof_py alone on the two Cox datasets, without strata | 0.14.0 |
| `compare_5m_warm` | The engine and `lab_spark`, four repeats; pprof_spark and `pprof_spark_cached` added in a later session, four repeats | 0.13.0; 0.14.0 for pprof_spark |
| `unstratified_5m_warm` | The engine without strata, rows placed by row identifier in 100 partitions, four repeats | 0.13.0 |
| `unstratified_5m_warm_input` | The same with the rows left in the input's partitions, after a restart | 0.14.0 |

The engine's code is the same in 0.13.0 and 0.14.0. Conditions as before: Runtime 17.3 LTS with
Photon, autoscaling from 1 to 10 workers of 4 vCPUs, 5 million rows, 8,000 providers, the simple
design. The three warm plans read the stored table once before every timed call (`warm_input`,
decision D-18) and ask for ten executors first.

## 1. pprof_py's halved steps: confirmed

| Features | Completed runs | Iterations | Calls to its likelihood | Steps halved |
|---|---:|---:|---:|---:|
| 50 | 4 | 3 | 5 | 0 |
| 100 | 1 | 3 | 11 | 6 |

This is what the regenerated data said it should be (`docs/spec/engine.md`, section 5.3): at 100
features pprof_py's third Newton step was halved six times, which leaves its estimate 4.34e-5
standard errors from the root with `converged` true. The inference of round 12 is now a count made
on the cluster. Two other attempts at 100 features ended with the driver's process dead; the one
that completed peaked at 22.1 GB on a driver of 28.3.

## 2. Reading the stored table first takes the first kind of stall out of the timed call

- **The light job no longer stalls.** It took 0.6 to 1.8 seconds in all 24 Cox calls. In the six
  stalled calls of round 12 it had taken 19 to 29.
- **The stall is in the read before the clock.** That read took 11 to 37 seconds before 17 of the
  engine's 32 calls and 2 to 4 seconds before the other 15.
- **Why so often.** The cluster replaces its executors all the time. The executor of a run's
  slowest task carries an identifier, and the identifiers reached 54 during `compare_5m_warm` and 74
  during `unstratified_5m_warm`, on a cluster of at most ten.

So the reading of round 15 holds in what matters: the 17 to 21 seconds belong to an executor's first
contact with the stored table, not to an implementation, and reading the table before the clock
moves them out. Whether it is the files or the access to storage that is new to the executor is not
settled: one call still spent 13.8 seconds in the job of one task that runs before the phases,
on an executor attached 97 seconds earlier, right after a read of the whole table that took 2.

## 3. What still stalls inside a timed call

- **The first Python task of a new executor.** In 16 of the 32 calls the slowest task took 6.5 to
  10.5 seconds, in the pass that builds the working set, where the median task takes 0.4 to 1.3
  seconds with strata. In 12 of the 16 the executor had been attached 44 to 128 seconds before.
  The first evaluation of a Cox fit then takes 10 to 17 seconds instead of 5 to 9. D-20 proposes to
  start the Python workers before the clock as well.
- **A cluster that changes size during the call**, three calls, each from 10 executors to 6: a pass
  for the provider effects of 37 seconds, an evaluation of 37.7 seconds, and a task of 15.8 seconds
  in the read of the build.
- **The job of one task before the phases**, once, as above.

## 4. Without strata: the input's partitions against 100 partitions by row identifier

Calls at a steady ten executors, in seconds.

| Features | Layout | Calls | Whole call | First evaluation | Later evaluations | Effective cores |
|---|---|---:|---|---|---|---|
| 50 | The input's 17 partitions | 3 | 17.7, 24.3, 28.9 | 4.6, 10.0, 12.6 | 3.6 to 4.0 | 8 to 9 |
| 50 | 100 by row identifier | 2 | 29.0, 33.2 | 12.7, 16.4 | 4.3 to 4.7 | 16 to 17 |
| 100 | The input's 33 partitions | 3 | 18.5, 18.7, 26.7 | 4.8, 5.3, 12.2 | 3.8 to 4.2 | 17 to 19 |
| 100 | 100 by row identifier | 2 | 37.1, 41.1 | 15.6, 17.2 | 6.0 to 6.8 | 17 to 18 |

- **The input's partitions are faster in every part.** A later evaluation costs 0.6 seconds less at
  50 features and 2.3 less at 100; the first evaluation, which builds the working set, costs 5 seconds
  where it meets no new executor, against 13 to 17 for the other layout in calls that all did.
- **Where the difference per evaluation comes from** was foreseen as OI-26 and is not measured
  apart: with 100 partitions every evaluation brings 100 pass-1 tables to the driver, 147 MB at
  100 features against 48 MB from 33 partitions, and runs 200 tasks instead of 66.
- **The best configuration without strata seen so far** is the flow of round 12 with the input's
  partitions: 17.7 and 18.5 seconds. Before round 12 the best calls were 19.8 and 30.3.
- **Limits of the comparison.** The two layouts ran in two sessions with a restart between; two and
  three calls each at a steady ten; all four calls of the layout by row identifier met a new
  executor in their first evaluation. The later evaluations do not depend on that, and they differ
  in every call.

The run that decision D-10 named as the test of its default has gone against the default. D-19
proposes the consequence.

## 5. The implementations side by side, with the table read first

Whole calls in seconds: median of four, and the best.

| Model, features | engine | `lab_spark` | pprof_spark | `pprof_spark_cached` |
|---|---|---|---|---|
| Cox with strata, 50 | 27.6, 17.9 | 12.2, 11.2 | 37.5, 25.2 | 31.6, 25.4 |
| Cox with strata, 100 | 26.0, 20.8 | 25.6, 16.8 | 58.9, 55.3 | 60.7, 58.1 |
| Logistic, 50 | 15.9, 14.4 | 12.0, 10.2 | 39.0, 32.4 | 55.8, 54.1 |
| Logistic, 100 | 32.1, 19.9 | 22.4, 15.8 | 65.0, 63.5 | 71.5, 70.5 |

How far this table can be read:

- **Against pprof_spark** the engine's medians are 1.4 to 2.5 times shorter and its best calls 1.4
  to 3.2 times. The two ran in different sessions. Every run of pprof_spark carries the caveat that
  the installed build is not the pinned one, and it is not the build of 2026-11-18 either: a fit now
  runs 3,800 to 15,600 tasks, against 1,500 to 3,700 then.
- **Against the prototype** the engine looks slower, and part of that is the order of the plan. The
  engine runs first after every change of dataset and meets the new executors; the prototype follows
  on a cluster the engine has warmed. In calls that met none, with ten executors: Cox at 100
  features 20.7 seconds against 16.8; the logistic model at 50 features 14.4 to 16.4 against 10.2
  to 10.9, of which about 3.5 seconds are the engine's pass for the provider effects and tests,
  which the prototype does not have. So with strata the engine is some 2.5 to 4 seconds behind its
  own prototype (OI-35).
- **Against the engine before round 12**, in calls that met no stall: Cox with strata at 100
  features 20.7 seconds against 18.7; the logistic model at 100 features 19.9 against 21.0 to 23.4,
  at 50 features 14.4 to 16.4 against 15.5 to 17.1. One call here and there. Round 12 saved a job
  and, for the logistic model, a second or two; for Cox with strata it did not make the call
  shorter.

## 6. Provider flags across runs

The engine's logistic records carry a summary of the flags (decision D-15). In the four runs of each
dataset, at 6 to 10 executors, the same providers were flagged in the same direction by all three
tests.

| Features | Score test: flagged above, below | Wald | Exact | Nearest provider to a cut-off: score, Wald (z), exact (p) |
|---|---|---|---|---|
| 50 | 2,770 and 2,596 | 2,770 and 2,592 | 2,765 and 2,600 | 6.2e-5, 4.4e-4, 9.0e-5 |
| 100 | 2,780 and 2,640 | 2,779 and 2,639 | 2,776 and 2,642 | 1.2e-3, 2.8e-4, 4.7e-7 |

With a row identifier the engine's layout does not depend on the cluster, so equal flags are what
identical bits would give; the summary cannot tell the two apart. The nearest provider to a cut-off
is 4.7e-7 away for the exact test at 100 features, where T_FLAG_P is 1e-9.

## 7. What follows

- RI-1 of `docs/reference-issues.md` is confirmed on the cluster and can be reported to pprof_py's
  maintainers with probe A-8 and this count.
- D-18 has done what it was proposed for and is still to be ruled on; D-20 proposes its extension to
  the Python workers.
- D-19: without strata, the input's partitions by default.
- For a comparison with pprof_spark that can be quoted: its build pinned, and all implementations
  in one session. Section 8 has the session; the pin is to be confirmed.

## 8. One session with every implementation (version 0.16.0)

The maintainer rebuilt pprof_spark at its current version and ran `compare_5m_warm` once more on
2026-11-21, with version 0.16.0: the engine, `lab_spark`, pprof_spark and `pprof_spark_cached` in one
session, four repeats each, the stored table read and the Python workers started before every timed
call. The engine's code for these two models is that of section 5.

Whole calls in seconds: median of four, and the best.

| Model, features | engine | `lab_spark` | pprof_spark | `pprof_spark_cached` | pprof_spark over engine: medians, best |
|---|---|---|---|---|---|
| Cox with strata, 50 | 19.7, 18.6 | 14.1, 10.7 | 34.5, 25.4 | 36.3, 29.1 | 1.8, 1.4 |
| Cox with strata, 100 | 19.8, 16.6 | 18.9, 15.4 | 50.4, 49.5 | 56.9, 55.7 | 2.5, 3.0 |
| Logistic, 50 | 15.0, 13.4 | 9.6, 9.4 | 47.7, 34.8 | 55.0, 53.0 | 3.2, 2.6 |
| Logistic, 100 | 19.0, 16.9 | 16.0, 12.6 | 86.5, 68.8 | 69.5, 66.9 | 4.6, 4.1 |

- **Against pprof_spark, in one session**, the engine's whole calls are 1.8 to 4.6 times shorter by
  medians and 1.4 to 4.1 times by the best call. All of pprof_spark's 32 records carry the caveat
  that their build is not the pinned one: the lock named the head of its repository of some hours
  earlier. The records hold the commit of the build; the reports sent do not show it. From round 19
  the report lists it and judges the pin on the day it is made.
- **Against the prototype** the engine is 0.9 to 5.6 seconds behind by medians. For the logistic
  model 3.2 to 4.6 seconds of that are the engine's pass for the provider effects and their tests,
  which the prototype does not compute. For Cox it is the engine: its later evaluations take 2.0 to
  2.8 seconds at 50 features and 3.0 to 4.0 at 100 at ten executors, the prototype's 1.8 to 2.0 and
  2.7 to 2.9 on average in its calls without a stall. Probe A-12 finds 27 to 39 percent more time
  in the engine's kernel, which centres the features (`docs/research/pass-cost.md`); that is part
  of it (OI-35).
- **What the Python start changed.** It took 1 to 2 seconds before 12 of the engine's 16 calls and
  4 to 10 before the other 4. Inside the timed calls the slowest task was 4.0 to 7.3 seconds in 6 of
  16, each on an executor attached 45 to 114 seconds before, and 1.2 to 2.1 in the other 10. Without
  it (section 3) it was 6.5 to 9.5 seconds in 7 of 16, and 13.8 once. So a new executor still costs
  its first pass some seconds, about half of what it did. The engine's medians went from 27.6, 26.0,
  15.9 and 32.1 seconds to 19.7, 19.8, 15.0 and 19.0 between the two sessions with the same code;
  how much of that is the Python start and how much the session cannot be told apart.
- **Flags.** The same providers flagged in all eight runs of each logistic dataset, the four of
  section 6 and these four.

## 9. Which builds of pprof_spark ran

The report of `compare_5m_warm` rendered with round 19 lists the builds:

| Records | Session | Commit of pprof_spark | The pinned build |
|---|---|---|---|
| 16 of pprof_spark, 16 of `pprof_spark_cached` | With version 0.14.0 (section 5) | `ada822d` (round 163, 2026-11-18) | No |
| 16 and 16 | With version 0.16.0 (section 8) | `b92919c` (round 177, 2026-11-21) | Yes |

So the comparison of section 8 is with pprof_spark at the build the lock pins, in one session, and
the caveat about the pin is gone from those 32 records. Left out the runs that still carry a caveat
(the cluster changed size during 5 of the 64), the medians are 19.7, 19.8, 15.7 and 19.0 seconds for
the engine and 34.2, 50.4, 47.7 and 84.8 for pprof_spark: 1.7 to 4.5 times. What the comparison
covers: synthetic data of 5 million rows and 8,000 providers, 50 and 100 features, the simple design,
four repeats, an autoscaling cluster of ten workers of 4 vCPUs, the stored table read and the Python
workers started before each call. pprof_spark's code has not been read again at this commit (OI-37).

The same day the plan `engine_partitions_5m_warm` was started with version 0.17.0 on a cluster that
had run 0.16.0. All 12 first fits stopped within three seconds with "the statcompute package has
changed since it was sent to the executors": the archive of the earlier version was on the
executors. Nothing was measured. Round 20 sends the package's code inside each task, so that a
changed package needs no restart.

