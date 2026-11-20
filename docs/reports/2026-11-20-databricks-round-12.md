# Round 12 on the cluster: compare_5m and unstratified_5m, 2026-11-20

Sources: the two reports the maintainer rendered with notebook `03_report` after running version
0.10.0 (round 12) on the standard cluster, as pasted on 2026-11-20; and the same two folders
rendered again that day with the report of round 14, which names each run's version, shows where
its time went and compares every run with pprof_py (section 5). Each report covers every record in
its results folder, so runs of earlier versions and of 0.10.0 stand side by side, with the
prototypes' and pprof_spark's runs since 2026-11-16. The JSON records were not available. The
engine's rows were typed from the reports into the table of section 2.

Sections 1 to 4 were written from the first pair of reports, which did not name versions: runs were
assigned by their task counts, and the earlier runs were called 0.9.0. The second pair shows them to
be of 0.8.0 with strata and providers, and of 0.8.0 and 0.9.0 without strata. Those two versions
have the same code for these fits, and the sections below now say "before round 12".

Conditions as in `docs/reports/2026-11-16-databricks-compare-5m.md`: Runtime 17.3 LTS with Photon,
autoscaling from 1 to 10 workers of 4 vCPUs, 5 million rows, 8,000 providers, the simple design.

## 1. What these reports could not show

- **Whether round 12's estimates agree with pprof_py.** The agreement tables compared the first
  completed run of each adapter, which for the engine is a run of an earlier version. No number in
  them comes from round 12. In the sandbox, fits with strata and logistic fits of 0.10.0 are
  identical bit for bit to 0.9.0's (`docs/spec/engine.md`, section 5.1). Section 5 has the
  comparison on the cluster.
- **Where in a call the time went.** The phases, each evaluation's seconds and the heaviest stages
  are in the records, not in the report.
- **Anything about pprof_spark in that session**: it was not installed there
  (`No module named 'pprof_spark'`).

Round 14 changes the report so that the same records answer the first two points (section 4).

## 2. The engine's runs by version

"Per evaluation" is the fit phase over the evaluations it holds: four before round 12 and for the
logistic model, three for Cox in round 12, whose first evaluation builds the working set and is
counted under "prepare". "Whole minus fit" therefore contains that first evaluation for Cox in round
12, about one "per evaluation" more than before; for the logistic model it contains the pass for the
provider effects in both. A stall is counted when whole minus fit is 30 seconds or more.

| Model, features | Runs of | Runs | Whole call, s: median (range) | Per evaluation, s: median (range) | Whole minus fit, s: median (range) | Stalls | Runs with a caveat | Runs with a task 50 times its stage's median |
|---|---|---:|---|---|---|---:|---:|---:|
| Cox with strata, 50 | before round 12 | 8 | 34.3 (12.9 to 96.3) | 3.7 (2.1 to 12.1) | 19.9 (4.3 to 48.0) | 3 | 2 | 2 |
| Cox with strata, 50 | round 12 | 2 | 59.1, 75.9 | 3.1, 3.8 | 49.7, 64.4 | 2 | 2 | 1 |
| Cox with strata, 100 | before round 12 | 4 | 21.9 (18.7 to 29.5) | 3.7 (3.3 to 4.0) | 6.4 (5.5 to 14.8) | 0 | 0 | 0 |
| Cox with strata, 100 | round 12 | 2 | 49.3, 59.2 | 5.6, 4.4 | 32.6, 46.1 | 2 | 1 | 2 |
| Logistic, 50 | before round 12 | 6 | 18.8 (15.5 to 28.2) | 1.8 (1.6 to 2.4) | 11.2 (8.9 to 20.2) | 0 | 0 | 0 |
| Logistic, 50 | round 12 | 2 | 24.1, 58.9 | 2.3, 2.3 | 14.9, 49.7 | 1 | 2 | 1 |
| Logistic, 100 | before round 12 | 4 | 22.0 (21.0 to 23.4) | 2.4 (2.3 to 2.5) | 12.5 (11.6 to 13.4) | 0 | 0 | 0 |
| Logistic, 100 | round 12 | 2 | 28.3, 67.5 | 3.2, 3.5 | 15.4, 53.5 | 1 | 2 | 1 |
| Cox without strata, 50 | before round 12 | 10 | 58.0 (19.8 to 139) | 5.9 (3.6 to 23.6) | 32.9 (5.1 to 50.1) | 6 | 4 | 0 |
| Cox without strata, 50 | round 12 | 2 | 129, 29.9 | 4.8, 3.8 | 114.7, 18.6 | 1 | 1 | 1 |
| Cox without strata, 100 | before round 12 | 8 | 58.5 (30.3 to 61.2) | 8.2 (5.9 to 10.7) | 21.1 (6.8 to 31.1) | 1 | 2 | 0 |
| Cox without strata, 100 | round 12 | 2 | 30.8, 68.7 | 5.8, 18.9 | 13.3, 11.9 | 0 | 1 | 0 |

The median of 26.8 seconds for Cox with strata at 50 features in the task of round 12 is the median
of the six of these eight runs that carry no caveat.

## 3. Findings

1. **One job fewer per fit, confirmed.** With strata or providers a fit of 0.10.0 ran 101 tasks
   fewer than one of 0.8.0: 536 against 637, 736 against 837, 568 against 669, 768 against 869.
2. **Round 12 did not make these calls faster.** With strata or providers, six of the eight calls
   of 0.10.0 took 49 to 76 seconds, against medians of 19 to 34 seconds before. The time is outside
   the fit phase in every one of them.
3. **The fit phase is no shorter.** Per evaluation it is the same for Cox with strata at 50 features
   and longer elsewhere: 4.4 and 5.6 seconds against 3.3 to 4.0 for Cox at 100 features, 2.3 against
   1.8 and 3.2 to 3.5 against 2.4 for the logistic model. Seven of those eight runs carry a caveat
   about the cluster's size, against two of twenty-two before, so the two versions were not
   measured under the same conditions. The sandbox had the Python side of a pass 8 to 30 percent
   shorter; the cluster does not show it and does not refute it.
4. **The stalls are older than round 12 and larger than anything it changed.** Earlier versions
   had them in three of eight Cox fits with strata at 50 features and in six of ten without strata. In the
   session of round 12 they were more frequent: six of eight runs with strata or providers, against
   three of twenty-two. Whether the working set of round 12 is more exposed to them, or that
   session's cluster was, these reports cannot say. They may not all be one thing. In five of the
   six stalled runs of 0.10.0 one task ran 58 to 252 times its stage's median, and the prototype
   `lab_spark` has ratios above 100 in six of its runs, with whole calls of 30 to 128 seconds. Of
   the three earlier stalls with strata one had such a task; the six without strata had none above
   13, so there every task of a stage was slow.
5. **Without strata the new layout uses more of the cluster and is not clearly faster.** Its two
   calls on a steady cluster took 29.9 and 30.8 seconds with 14 and 18 effective cores, against 7
   to 12 before. The medians before were 58 seconds; the best calls 19.8 and 30.3. Per
   evaluation 3.8 and 5.8 seconds against 3.6 and 5.9 at best before. In the call where the cluster
   changed size the fit wrote 1.73 GB of shuffle data instead of 1.12: with the new layout the
   partitions of a lost executor are shuffled again, which the old layout did not need.
6. **Preparation without a stall is not shorter either.** The two calls of 0.10.0 without a stall,
   both logistic, spent 14.9 and 15.4 seconds outside the fit phase, against medians of 11.2 and
   12.5 before. Two runs, both on a cluster that changed size.

What a stall is has not been established. Candidates, none tested: an executor that joined just
before or during the fit runs its first tasks slowly; an executor that left took cached partitions
and shuffle files with it; one read of the stored table is slow; garbage collection in a heap of
2.2 GB. The records name the stage of the slowest task; from version 0.12.0 they also name its
executor, how long that executor had been attached, and the task's garbage collection and wait for
shuffle data.

## 4. What follows

- The report of round 14 reads a results folder version by version: medians by version; for every
  Spark run its phases, its first and its later evaluations, and the slowest task with its stage;
  and the agreement of every run, not of the first. Running `03_report` again on the two existing
  folders gives those sections for the runs above, without a new run on the cluster.
- The next runs on the cluster record the slowest task in full. That should say which of the four
  candidates a stall is.
- Until the stalls are understood, differences of a few seconds between versions cannot be read
  from two runs per configuration on this cluster.

## 5. Second reading: the report of round 14 on the same records

**The estimates of round 12 agree with pprof_py as the earlier ones did.** Every run is now
compared, and the run furthest from pprof_py is shown.

| Model, features | 0.8.0 or 0.9.0, in standard errors | 0.10.0, in standard errors | Runs of 0.10.0 |
|---|---|---|---:|
| Cox with strata, 50 | 8.02e-13 | 8.02e-13 | 2 |
| Cox with strata, 100 | 7.26e-13 | 7.26e-13 | 2 |
| Logistic, 50 | 1.86e-10 | 1.86e-10 | 2 |
| Cox without strata, 50 | 2.52e-09 | 2.52e-09 | 2 |
| Cox without strata, 100 | 4.34e-05 | 4.34e-05 | 2 |

The last row is pprof_py's stopping rule (`docs/spec/engine.md`, section 5.3). Logistic at 100
features has no comparison: the memory guard skips pprof_py there.

**The stall is in the first job that reads the stored table.** For round 12's eight calls with
strata or providers:

| Model, features | Whole call | Outside the phases | Prepare | First evaluation | Later evaluations (median) | Slowest task | Its stage: tasks, median task | Executors seen |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Cox, 50 | 75.9 | 22.0 | 42.4 | 17.5 | 3.7 | 20.5 | `spark.py:509`: 17, 2.15 | 5 |
| Cox, 50 | 59.1 | 17.5 | 32.3 | 11.8 | 3.1 | 18.1 | `spark.py:509`: 17, 0.14 | 10, 9 |
| Logistic, 50 | 24.1 | 2.3 | 6.5 | 2.7 | 2.1 | 1.6 | a stage of one task | 10, 6 |
| Logistic, 50 | 58.9 | 1.5 | 42.9 | 2.5 | 2.2 | 18.8 | `spark.py:509`: 17, 0.14 | 9, 10 |
| Cox, 100 | 49.3 | 0.9 | 31.7 | 12.3 | 5.4 | 17.3 | `spark.py:509`: 47, 0.28 | 10, 6 |
| Cox, 100 | 59.2 | 0.9 | 45.1 | 16.4 | 4.3 | 21.2 | `spark.py:509`: 33, 0.08 | 10 |
| Logistic, 100 | 28.3 | 0.8 | 8.5 | 3.3 | 3.1 | 1.7 | `spark.py:934`: 100, 1.10 | 10, 6 |
| Logistic, 100 | 67.5 | 8.8 | 38.6 | 3.3 | 3.4 | 21.0 | `spark.py:509`: 33, 0.36 | 9 |

Line 509 of `spark.py` in version 0.10.0 is the action of `_check`: the light job. Its stage has one
task per file of the stored table, 17 at 50 features and 33 at 100. In all six stalled calls the
slowest task of the whole run is one of those: 17 to 21 seconds, in a stage whose median task takes
0.1 to 0.4 seconds. For Cox the light job is "prepare" less the first evaluation: 24.9, 20.5, 19.4
and 28.7 seconds here, and 0.8 to 2.7 seconds in the three calls without strata that did not stall
there.

**The same task is in the earlier runs and in the other implementations, wherever a job first reads
the table.**

| Where | Seen in | Slowest task, s | Stage |
|---|---|---|---|
| A Photon scan of the stored table in 0.8.0: its row count or the read of its validation job | Cox with strata, 50 features: three calls | 12.3, 15.0, 17.9 | 17 tasks, median 0.1 to 1.5 |
| A job of one task before the engine's phases, probably the read of the table's schema, which is where `spark.read.parquet` runs | 0.8.0: three calls; `lab_spark`; pprof_spark | 16.9, 17.1, 7.7; 17.0; 16.7 | One task. In calls without a stall it takes 1.5 to 2 seconds |
| The validation job of 0.8.0 and 0.9.0 without strata | Twelve calls | 10 to 20 | `spark.py:376` or `:451`: 17 or 33 tasks. In three calls the median task took 14 to 17 seconds: every file was read slowly |
| `lab_spark` and pprof_spark | Many calls | 12 to 29 | Photon scans of 17 or 33 tasks |

**A second kind, about 80 seconds, sits in a pass over the working set.** Three calls: 0.10.0 without
strata (80.9 seconds in the pass that builds the working set, the executors going from 10 to 9),
0.9.0 without strata (77.9, from 4 to 7) and `lab_spark` on 2026-11-16 (83.9, from 10 to 8). The
cluster changed size during each.

**Reading.** A task that reads a file of the stored table for the first time on its executor takes
17 to 21 seconds, whatever it reads from the file: three columns in the light job, the footer in the
row count, the schema in the job of one task. That is the size of every stall of the first kind. An
executor that has just joined has read nothing, and in the session of round 12 the cluster went
from 10 executors to 6 during three of the eight calls and was brought back before the next. This
fits a local cache of the stored files on each executor, empty on a new one; it has not been shown.
The records name neither the executor of the slow task nor what it read. Records made from
version 0.12.0 on do.

**Net of the stalls round 12 is not faster either.** Later evaluations take what evaluations took
before: 3.1 to 3.7 seconds against a median of 3.7 for Cox at 50 features, 4.3 to 5.4 against 3.7
at 100, and for the logistic model 2.1 to 2.2 against 1.8 and 3.1 to 3.4 against 2.4. Without strata the
one call of 0.10.0 on a steady cluster at 100 features took 30.8 seconds: 1.2 for the light job, 10.1
for the first evaluation, 5.9 for each later one. The best call of 0.9.0 took 30.3: 5.2 to prepare
and 5.9 per evaluation. The new layout there used 17.6 effective cores where the old one used 11.7,
for the same seconds.

## 6. What follows from the second reading

- Stalls of the first kind are a property of the cluster and of the stored table, not of an
  implementation or a version. They hit whichever job reads the table first. They cannot be
  optimized away in the engine, and comparisons of versions have to be made clear of them.
- The harness of round 15 can read the stored table once before every timed Spark fit (plan option
  `warm_input`, off unless a plan sets it). A run with it and the records of 0.12.0 or later will
  say whether the reading above is right: the slow task's executor, how long it had been attached,
  and how much it read.
- Without strata, placing the rows in 100 partitions has not paid on this cluster: the same seconds
  per evaluation, more cores, and a new shuffle when an executor is lost. A run of
  `unstratified_5m` with `"engine_partitions": "input"` beside one without would settle whether the
  default of round 12 should stay (D-10).
