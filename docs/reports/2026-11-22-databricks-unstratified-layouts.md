# Without strata: three layouts of the rows on the cluster (2026-11-22)

Source: the report of plan `unstratified_cores_5m_warm` that the maintainer ran and rendered on
2026-11-22 with version 0.21.0, whole. The JSON records are not in the repository; the 24 rows were
typed from the report's tables and the numbers below computed from them by script. It is the first
cluster run of version 0.21.0. Conditions as in `2026-11-21-databricks-warm-runs.md`: Runtime 17.3
LTS with Photon, autoscaling from 1 to 10 workers of 4 vCPUs, 5 million rows, the simple design,
the stored table read and the Python workers started before every timed call.

Cox without strata is fitted on a grid of the distinct times, two passes per evaluation. The plan
runs the engine three ways in one session, four repeats each, the three taking turns:

| Adapter | Partitions of the working set | How the rows get there |
|---|---|---|
| `engine` | The input's: 17 at 50 features, 33 at 100 | Left where they are; no shuffle. The default (decision D-19) |
| `engine_cores` | One per core: 40 on ten executors | Placed by row identifier and sorted by it |
| `engine_cores_hash` | One per core | Placed by a hash of the row; not sorted |

All 24 fits completed in three iterations. Tasks per fit: 155 and 299 with the input's partitions,
396 and 428 with 40.

## 1. A task per core makes an evaluation a quarter to a third shorter

Later evaluations, in seconds, in the calls at a steady ten executors: the median, and the range.

| Features | The input's partitions | One per core, sorted | One per core, by a hash |
|---|---|---|---|
| 50 | 3.9 (3.6 to 4.1) | 2.8 (2.6 to 3.3) | 2.7 (2.5 to 3.0) |
| 100 | 5.35 (5.0 to 6.0) | 3.8 (3.6 to 4.4) | 3.95 (3.7 to 4.2) |

- **26 to 31 percent less with one partition per core**, at both widths. The fits used 15 to 21
  effective cores, against 9 with the input's 17 partitions and 13 to 15 with its 33.
- **The sort gives nothing back.** Sorted or not, an evaluation takes the same: 2.8 against 2.7
  seconds, 3.8 against 3.95.
- **At 100 features the input has 33 partitions, almost one per core, and is still a quarter
  slower.** So the gain there is not a matter of 33 tasks against 40. The input's tasks ran on 13
  to 15 cores at a time. A likely reason, not measured: the cached partitions lie where the first
  read put them, several on one executor, and tasks wait for it; a shuffle spreads them evenly.

## 2. The hash builds as fast as leaving the rows where they are; the sort costs two seconds

The first evaluation, the pass that builds the working set, in seconds, in the calls at ten
executors that did not follow a long read or a wait for executors:

| Features | The input's partitions | One per core, sorted | One per core, by a hash |
|---|---|---|---|
| 50 | 4.9 | 6.9 | 4.8, 5.3, 4.8 |
| 100 | 7.2, 6.5, 6.2 | 7.8, 8.8 | none; 6.9 in a call that lost executors later |

Shuffling 5 million rows to 40 partitions by a hash cost nothing that shows against reading them
in place. Sorting them by the row identifier as well cost 2 seconds at 50 features and 1 to 2 at
100.

## 3. Whole calls

In seconds: the median of four and the best, then the calls at ten executors that met no new
executors.

| Features | The input's partitions | One per core, sorted | One per core, by a hash |
|---|---|---|---|
| 50, median and best | 23.5, 17.4 | 20.3, 18.3 | 15.4, 14.6 |
| 50, undisturbed calls | 17.4 | 18.3 | 14.6, 16.1, 14.6 |
| 100, median and best | 25.5, 23.2 | 26.6, 21.4 | 27.3, 26.3 |
| 100, undisturbed calls | 26.5, 24.5, 23.2 | 21.4, 21.9 | none |

- **At 50 features the hash layout is the fastest by every count**: 14.6 seconds against 17.4 for
  the default and 18.3 with the sort.
- **At 100 features the hash layout had no undisturbed call.** Two of its four lost executors
  during the fit and the other two followed a wait for new ones, with first evaluations of 12.5
  and 12.9 seconds. Its parts say what an undisturbed call would be: a first evaluation near 7
  seconds and three later ones of 3.95, against 6.2 to 7.2 and 5.35 for the default. That is an
  inference from one session, not a timing.
- Sorted, one per core beats the default at 100 features, 21.4 and 21.9 against 23.2 to 26.5, and
  not at 50, where it is 18.3 against 17.4: the sort's 2 seconds cancel what three shorter
  evaluations save, and that call had a second more outside the engine's phases (OI-45; 7 of the
  24 calls had it).

## 4. The default at 100 features was slower than on 2026-11-21

With the same layout, the input's 33 partitions, the session of version 0.14.0 had later
evaluations of 3.8 to 4.2 seconds on 17 to 19 effective cores and whole calls of 18.5 and 18.7
seconds, with a third of 26.7 that had a stall (`2026-11-21-databricks-warm-runs.md`, section 4).
This session: 5.0 to 6.0 seconds on 13 to 15 cores, and 23.2 to 26.5, with a fourth of 29.7 after a
wait for executors. At 50 features the two sessions agree: 3.6 to 4.0 then, 3.6 to 4.1 now. Between the two versions this path changed in how its code reaches the executors (by value,
since 0.18.0) and in what it records, not in its passes, and the same changes left the 50-feature
timings where they were. What differs is not known (OI-46). If the placement of the cached
partitions is the reason, it is one more argument for a layout that does not depend on where a
read happened to put the rows.

## 5. The cluster, and decision D-22 on its first run

| What the cluster did | Fits |
|---|---|
| Ten executors from start to end | 18 |
| Six from start to end | 1 |
| Lost four executors during the fit | 5 |

- **The fit on six executors was not caught by the new count.** The plan asks again when the
  executors are short just before a fit (`hold_again`). The warm-up reported "10 of 10 in 0 s", the
  count after the reads said ten as well, and the fit ran on six from its first moment and counted
  24 cores when it chose its partitions. So the loss came in the second after that count, or the
  scheduler's list of block managers was late. The fit carries the caveat for a fit on fewer
  executors than the plan holds, which is judged from what was sampled during it.
- **Four of the five losses during a fit came during its first evaluation**, which then took 2 to
  4 seconds longer, and every later evaluation was slow. The fifth came during the second
  evaluation.
- Asking again did work once, at the start of the session: 1,040 seconds to reach ten executors,
  asked twice. After a loss, getting the executors back took 121 to 226 seconds, six times.
- Taking turns spread the six disturbed fits over the three adapters, one, two and three, where
  one after another they would have fallen on whichever was running.

Round 24 amends the count accordingly (decision D-22, approved 2026-11-22): a pause of three
seconds after the reads, then the lowest of three counts: the scheduler's block managers, the
executors listed as active, and the scheduler's cores over the cores of one executor. All three are
recorded. It can catch a loss that comes in those seconds; it does nothing for one that comes
during the fit.

## 6. What follows

Without strata, one partition per core placed by a hash is the layout to prefer on this evidence:
evaluations as short as with the sort, the cheapest build, the shortest whole calls at 50 features. It is
proposed as decision D-23. What it costs is in the proposal: the layout then depends on the size of
the cluster when the fit starts, as with strata (D-21), and fits repeat to rounding, as they
already do with the default.

## 7. Limits

One session, one size of data, one cluster, four repeats; at 100 features two undisturbed calls for
one layout and none for another. The coefficients of the three layouts were not compared on the
cluster: the plan has no reference in it and the report compares with a reference only. In the
sandbox the three agree with pprof_py within the tolerance classes (`tests/test_runner.py`). The
reason for section 4 is a guess.
