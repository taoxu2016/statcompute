# One partition per core as the default, on the cluster (2026-11-22)

Source: the report of plan `compare_5m_warm` that the maintainer rendered on 2026-11-22 at 01:14 UTC
with version 0.20.0. It is the first cluster run of the default of decision D-21, and the first in
which the engine and its prototype `lab_spark` both have one task per core: 64 fits, the four
implementations on four datasets of 5 million rows, four repeats each. Conditions as in
`2026-11-21-databricks-warm-runs.md`: Runtime 17.3 LTS with Photon, autoscaling from 1 to 10 workers
of 4 vCPUs, 8,000 providers, the simple design, the stored table read and the Python workers started
before every timed call.

**The report reached the repository without its middle.** It came as the output of a notebook cell
that printed it, and such a cell keeps the first and the last 25,600 characters of a long text. What
survived: the environment, the list of implementations, the first 111 rows of the table of runs (the
sessions of versions 0.13.0 and 0.14.0 and part of 0.16.0), the last 94 rows of the table of where
the time went, which hold all 64 runs of this session, the provider flags and the adapters' notes.
What is missing for this session: its rows in the table of runs, so the number of tasks, the
iterations, the effective cores and the caveats of each fit, and the table of medians. Everything
below was computed by script from the surviving rows, whose times are rounded to a tenth of a
second. The JSON records are not in the repository. Round 23 gives the report a way out that keeps
it whole, and section 7 says what rendering it again will add.

## 1. What ran

| Implementation | Version or commit | Records in the folder | Of this session |
|---|---|---|---|
| `engine` | 0.20.0, its code sent by value | 16 | 16 |
| `lab_spark` | 0.20.0 | 16 | 16 |
| `pprof_spark` | `b92919c`, the pinned build | 32 | 16 |
| `pprof_spark_cached` | `b92919c`, the pinned build | 32 | 16 |

The prototype's records carry no execution record, so the report cannot say how its code reached
the executors; it calls the function the engine calls. The 32 records of each pprof_spark adapter at
`b92919c` are the 16 of the 0.16.0 session and the 16 of this one; the 16 at `ada822d` are the 0.14.0
session's.

## 2. The default gave one partition per core at the start

The surviving part shows tasks per stage in one place only, the stage of each call's slowest task.
For the engine that stage was a pass over the working set in all 16 calls:

| Executors during the fit | Engine calls | Tasks in the stage |
|---|---|---|
| 10 from start to end | 12 | 40 |
| 10 at the start, 6 at the end | 1 | 40 |
| 6 from start to end | 3 | 24 |

So the number followed the cores at the start in every call. The three fits on six executors had
24 partitions for their 24 cores: one task per core for the cluster they had. What OI-42 warns of, a
fit that starts small on a cluster that then grows, did not happen to the engine in this session.
The opposite did once: logistic regression at 100 features started on ten executors with 40
partitions and lost four of them, and its evaluations took 1.9, 5.6 and 3.1 seconds.

The prototype's slowest task was in a stage of 40 tasks in 13 of its 16 calls and in the 17-task
stage that reads the input in the other three. It ran at 125,000 rows per partition, as the plan
says; the widget of `02_run` did not override it.

**Not checked: the tasks per fit.** They are in the missing rows. The rows of the earlier sessions
that survived confirm the rule the counts follow, with N the input's partitions (17 and 33), P the
working set's and three iterations: the engine at P = 100 ran 536 and 568 tasks per Cox fit, which
is 2N + 2 + 5P, and 736 and 768 per logistic fit, 2N + 2 + 7P; the prototype 619 and 635, N + 2 + 6P,
and 638 and 670, 2N + 4 + 6P. For this session that gives 236 and 268 (Cox) and 316 and 348
(logistic) at 40 partitions, and 156, 188, 204 and 236 at 24.

## 3. Evaluations and whole calls against the earlier sessions

Later evaluations of the engine, in seconds, in the calls at a steady ten executors (nine
evaluations each): the median, and the range. The two columns on the right are the 0.18.0 session
(`2026-11-21-databricks-partition-size.md`, section 2).

| Model, features | 0.20.0, the default | 0.18.0, 40 tasks | 0.18.0, 100 tasks |
|---|---|---|---|
| Cox with strata, 50 | 2.0 (1.8 to 3.0) | 1.8 (1.7 to 2.2) | 2.5 (2.2 to 2.6) |
| Cox with strata, 100 | 3.2 (2.9 to 3.7) | 3.0 (2.9 to 3.9) | 3.6 (3.2 to 3.8) |
| Logistic, 50 | 1.3 (1.1 to 1.5) | 1.3 (1.1 to 1.4) | 1.6 (1.5 to 1.9) |
| Logistic, 100 | 1.9 (1.7 to 2.3) | 1.9 (1.8 to 2.0) | 2.4 (2.0 to 2.9) |

Whole calls of the engine, in seconds: the median of four, and the best.

| Model, features | 0.20.0, the default | 0.18.0, 40 tasks | 0.16.0, 100 tasks |
|---|---|---|---|
| Cox with strata, 50 | 18.4, 13.3 | 12.8, 10.7 | 19.7, 18.6 |
| Cox with strata, 100 | 22.0, 18.5 | 21.1, 17.9 | 19.8, 16.6 |
| Logistic, 50 | 17.4, 12.9 | 15.5, 13.2 | 15.0, 13.4 |
| Logistic, 100 | 22.0, 17.2 | 17.8, 17.1 | 19.0, 16.9 |

- **An evaluation takes what it took with 40 tasks before**, 11 to 21 percent less than the 100
  tasks of the earlier default in the 0.18.0 session.
- **Whole calls were not shorter than in the earlier sessions.** Against the 100 tasks of the
  0.16.0 session the median is 1.3 seconds shorter for Cox at 50 features and 2.2 to 3.0 seconds
  longer in the other three; the best call is 5.3 seconds shorter for Cox at 50 features and within
  2 seconds otherwise. Against the same layout in the 0.18.0 session the medians are 0.9 to 5.6
  seconds longer.
- **Two sessions with the same layout differ by as much as two layouts in different sessions.**
  A whole call compared across sessions does not measure a layout. What supports D-21 is the
  comparison inside one session (the 0.18.0 one) and the evaluations above.

Where the session's time went: 4 of the engine's 16 calls did not have ten executors throughout
(section 5), and 6 of the other 12 ran on executors new to the table. Before those six the read
that precedes the clock took 9 to 37 seconds instead of 2 to 3, and the calls took longer:

| Model, features | After a read of 2 to 3 s | After a read of 9 to 37 s |
|---|---|---|
| Cox with strata, 50 | 13.3 | 20.8, 18.2 |
| Cox with strata, 100 | 19.3, 18.5 | 24.6 |
| Logistic, 50 | 12.9 | 16.7, 19.1 |
| Logistic, 100 | 19.6, 17.2 | 24.4 |

The first evaluation of Cox, the pass that builds the working set, took 4.8 seconds after a short
read and 9.1 and 11.0 after a long one at 50 features, and 6.9 and 7.6 against 12.2 at 100. With one
task per core this no longer shows as one task far above the rest. The whole pass is slow: its
median task took 3.1 to 6.2 seconds against 1.4 to 2.9, and in no engine call was the slowest task
as much as twice its stage's median. The reads before the clock (D-18, D-20) take the first
contact with the table and the start of Python out of the timing; a call after a long read still
took roughly 4 to 7 seconds more (OI-31). Five of the six long reads followed a loss of executors
or the start of the session; the sixth, of 9 seconds, was the first read of a new dataset. The
engine is the first to run on every dataset and the others follow it, so it meets this more often:
6 of its 12 calls at a steady ten, against 2 of the prototype's 15.

## 4. The engine against its prototype, and against pprof_spark

Whole calls of this session, in seconds: the median of four, and the best.

| Model, features | engine | lab_spark | pprof_spark | pprof_spark_cached |
|---|---|---|---|---|
| Cox with strata, 50 | 18.4, 13.3 | 10.4, 9.8 | 35.1, 26.5 | 29.8, 26.5 |
| Cox with strata, 100 | 22.0, 18.5 | 15.1, 14.9 | 69.5, 57.6 | 62.4, 60.8 |
| Logistic, 50 | 17.4, 12.9 | 10.9, 8.5 | 50.7, 36.4 | 59.2, 55.9 |
| Logistic, 100 | 22.0, 17.2 | 12.4, 11.7 | 99.0, 76.1 | 69.3, 68.5 |

**Against pprof_spark the engine's whole call is 1.9 to 4.5 times shorter by the medians and 2.0 to
4.4 times by the best call** (0.16.0 session: 1.8 to 4.6 and 1.4 to 4.1). Restricted to the calls at
a steady ten executors, of which pprof_spark has two or three per dataset, 1.7 to 4.6 times.

**Against its prototype, on equal terms, the engine is 3.5 to 5.5 seconds behind by the best call**
and 6.5 to 9.6 by the medians, which carry the new executors of section 3. One evaluation at a
steady ten executors, in seconds, the median and the range. For the engine's Cox fits these are the
later evaluations; for its logistic fits the mean of the four evaluations of a call; for the
prototype, which does not time its evaluations, the fit phase divided by its four evaluations.

| Model, features | engine | lab_spark |
|---|---|---|
| Cox with strata, 50 | 2.0 (1.8 to 3.0) | 1.7 (1.6 to 1.9) |
| Cox with strata, 100 | 3.2 (2.9 to 3.7) | 2.6 (2.5 to 3.5) |
| Logistic, 50 | 1.3 (1.3 to 1.5) | 1.25 (1.2 to 2.1) |
| Logistic, 100 | 1.9 (1.85 to 2.2) | 1.8 (1.75 to 1.9) |

- **Logistic: an evaluation costs the same in both**, within a tenth of a second. The engine's
  call is longer by the pass that computes the provider effects and their tests, 3.0 to 3.8
  seconds at ten executors, which the prototype does not have. In the best calls the remainder is
  1.4 and 1.7 seconds: 0.8 and 1.2 before the first evaluation, 0.7 and 0.5 in the four
  evaluations.
- **Cox: the engine's evaluation is 0.3 seconds longer at 50 features and 0.6 at 100**, about a
  fifth. Probe A-12 found the engine's kernel 27 to 39 percent slower on one partition because it
  centres the features, which is 0.07 to 0.1 seconds at this size of partition in the sandbox: a
  part of the difference, not all of it. The other candidate grows with the features as the
  difference does: the engine limits BLAS to one thread in each worker and the prototype does not.
  Neither was measured on the cluster.
- **Cox, the rest of the call.** In the engine's best call at 50 features (13.3 seconds against
  the prototype's median of 10.4) the three later evaluations account for 1.0 second of the
  difference. The other 1.9 are before them: 0.6 for the light job that counts and checks the
  rows, which the prototype does not have, and 1.3 in opening the table, building the working set
  and the first evaluation, which the two report in different phases: 6.4 seconds in the engine
  (1.6 outside its phases, where it opens the table, and a first evaluation of 4.8) against 5.1
  in the prototype (a preparation of 3.4 and one evaluation of 1.7). The part outside the
  engine's phases is 0.7 to 0.8 seconds in 9 of its 16 calls and 1.0 to 1.6 in the other 7; what
  makes the difference was not looked at.

## 5. The cluster: fits on six executors that nothing flagged

51 of the 64 fits had ten executors from start to end. Of the other 13:

| What the cluster did | Fits | Which |
|---|---|---|
| Six executors from start to end | 7 | engine 3, pprof_spark 4 |
| Lost four executors during the fit | 5 | engine 1, lab_spark 1, pprof_spark 1, pprof_spark_cached 2 |
| Grew from six to eight during the fit | 1 | pprof_spark |

The seven on six executors are the finding. Before every fit the harness asks for ten executors
and waits until it counts them (`hold_executors`), then reads the table and starts Python, 3 to 4
seconds, and then starts the fit. These seven ran on six executors from their first sampled moment
to their last, so the cluster lost four between the count and the fit. The harness flags a fit
when its warm-up did not reach the number or when the count moves during the fit. Neither is the
case here, so by its own rules these seven carry no caveat, and they are in the medians "without
caveat". The surviving rows of the earlier sessions show the same: eleven fits on six executors
from start to end in the sessions of 0.13.0, 0.14.0 and 0.16.0, none with a caveat about the
cluster. It is always six. Whether the count was late or the loss came just after it is not in
the report; the records hold what each warm-up saw.

After each of the seven, the next fit was preceded by a read of 13 to 28 seconds, the mark of new
executors: the harness, counting six by then, asked for the four and got them. That is the long
read and the slow first pass of section 3. So one loss costs two fits: one on six executors, the
next on new ones.

Round 23 changes the harness for this: it counts the executors again after the reads, just before
the fit, records the count and flags the fit when it is short; a report rendered from now on gives
the caveat to the earlier records as well, from what they measured; and a plan can have the harness
ask again when the count is short (`hold_again`). Whether every plan should ask again is proposed
as decision D-22.

pprof_spark's slowest task was 3.3 to 4.6 seconds in 8 of its 16 calls, in stages of 515 and 992
tasks whose median is a tenth of a second; five of the eight came after a read of 12 to 28
seconds, which is the mark of new executors. Its kernels run in the JVM, and nothing the harness
warms covers them, as before.

## 6. The provider flags

For both logistic datasets the engine flagged the same providers the same way in all 12 runs the
folder holds, by all three tests: the four of this session, at 40 partitions and once at 24, and
the eight of the sessions of 0.13.0 and 0.16.0 at 100. At 50 features 2,770 above and 2,596 below
by the score test; at 100 features 2,780 and 2,640. The smallest distance of a provider from a
cut-off in any run is 6.2e-5 and 1.2e-3 in |z| by the score test, and 9.0e-5 and 4.7e-7 in the
p-value of the exact test.

The report compares coefficients with pprof_py only, and pprof_py is not in this plan, so this
session's estimates were compared with nothing. The flags are the only check of the fits at 40 and
at 24 partitions on the cluster so far.

## 7. To complete this reading, without a new run

Render the report of `compare_5m_warm` again with round 23's notebook: `versions` = `0.20.0` and
`reference` = `pprof_spark`. The report of one session is about a third of the size of this
folder's, and the notebook's last cell holds the whole text to copy. It adds:

- the tasks of every fit, and with them the check of section 2;
- the partitions, the rule that gave them and the cores at the start of every engine fit, which
  the records carry and the report now prints;
- what each warm-up saw before the seven fits on six executors, and the new caveat on them;
- the medians without those fits;
- the coefficients of the engine and the prototype against pprof_spark's. An earlier build of
  pprof_spark was within 7e-13 standard errors of pprof_py on three of these datasets
  (`2026-11-16-databricks-compare-5m.md`, section 2).

## 8. Limits

One session, one size of data, one cluster, four repeats, and the numbers of sections 3 to 5 from
rows rounded to a tenth of a second. The prototype's evaluations are an average that includes its
first one. Nothing was measured inside a task, so the difference between the two Cox kernels on
the cluster is inferred from whole evaluations. The adapters ran one after another on each dataset,
the engine first; `interleave`, new in round 23, lets a plan have them take turns.
