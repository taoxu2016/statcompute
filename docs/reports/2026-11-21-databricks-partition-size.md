# The size of partitions, and the package sent by value, on the cluster (2026-11-21)

Source: the report of plan `engine_partitions_5m_warm` that the maintainer rendered and pasted on
2026-11-21, version 0.18.0. The JSON records are not in the repository; the rows were typed from the
report to make the tables. Conditions as in `2026-11-21-databricks-warm-runs.md`: Runtime 17.3 LTS
with Photon, autoscaling from 1 to 10 workers of 4 vCPUs, 5 million rows, 8,000 providers, the simple
design, the stored table read and the Python workers started before every timed call.

The plan runs the engine three ways in one session, four repeats each: 50,000 rows per partition
(the default; 100 tasks per pass at this size), 125,000 (40 tasks, one for each of the cluster's 40
cores) and 250,000 (20 tasks).

## 1. The package sent by value works on Databricks

The cluster had run version 0.16.0, whose archive was still on its executors, and had refused all
twelve first fits of version 0.17.0 for that reason (`2026-11-21-databricks-warm-runs.md`, section
9). It was not restarted. With version 0.18.0 all 48 fits completed on it. So a changed package no
longer needs a restart of the cluster (OI-38 closed). The executors' identifiers run from 155 to
205 in these records: it is the application of the earlier sessions.

## 2. One task per core is the fastest of the three

Later evaluations, in seconds, in the calls at a steady ten executors (nine evaluations each):
the median, and the range.

| Model, features | 100 tasks per pass | 40 tasks | 20 tasks |
|---|---|---|---|
| Cox with strata, 50 | 2.5 (2.2 to 2.6) | 1.8 (1.7 to 2.2) | 2.2 (1.9 to 2.6) |
| Cox with strata, 100 | 3.6 (3.2 to 3.8) | 3.0 (2.9 to 3.9) | 3.6 (3.4 to 4.0) |
| Logistic, 50 | 1.6 (1.5 to 1.9) | 1.3 (1.1 to 1.4) | 1.5 (1.3 to 1.6) |
| Logistic, 100 | 2.4 (2.0 to 2.9) | 1.9 (1.8 to 2.0) | 2.2 (2.1 to 2.5) |

Whole calls, in seconds: the median of four, and the best.

| Model, features | 100 tasks per pass | 40 tasks | 20 tasks |
|---|---|---|---|
| Cox with strata, 50 | 20.1, 14.4 | 12.8, 10.7 | 14.0, 11.5 |
| Cox with strata, 100 | 22.4, 19.5 | 21.1, 17.9 | 21.8, 20.1 |
| Logistic, 50 | 17.8, 13.6 | 15.5, 13.2 | 14.8, 13.4 |
| Logistic, 100 | 24.1, 19.6 | 17.8, 17.1 | 22.3, 19.6 |

- **With 40 tasks an evaluation takes 17 to 28 percent less than with 100.** The pass for the
  provider effects follows: 2.9 to 3.7 seconds against 3.4 to 5.0.
- **With 20 tasks it takes as long as with 100, or up to 12 percent less.** Half the cores have no
  task then: in the calls at a steady ten executors without a stall the fits used 11 to 15
  effective cores, against 21 to 27 with 40 tasks.
- **Whole calls move the same way with more noise.** 16 of the 48 calls had a task of 4 seconds or
  more, 13 of them on an executor attached less than 130 seconds before; in 12 the cluster changed
  size or had fewer than ten executors.
- **The flags do not depend on the layout**: for both logistic datasets the same numbers of
  providers flagged, and the same nearest distances to the cut-offs, with all three sizes and in
  all four runs of each.

This is what probe A-11 suggested (`docs/research/pass-cost.md`): a pass costs for its tasks as
well as for its rows, so the tasks should be as few as keeps every core busy.

## 3. What an evaluation still costs

With one task per core an evaluation takes 1.3 to 3.0 seconds. Going from 40 tasks of 125,000 rows
to 20 of 250,000 changes it by 0.2 to 0.6 seconds, which bounds what the rows of a task cost;
going from 100 tasks to 40 saves 0.3 to 0.7. The rest, a second or more of every evaluation, moved
with neither. It was not measured apart: the candidates are what a Spark job costs on this
platform whatever it does, the transfer of the results, and the two tasks an executor of two
physical cores runs on each of them.

## 4. Limits

One size of data, one cluster, four repeats; the three sizes were not interleaved within a dataset
but ran one after another. Nothing here says where between 20 and 100 tasks the best number lies
other than that 40 beat both, nor what it is on a cluster of another size or with data that do
not fit one partition per core. Without strata the engine uses the input's partitions, 17 and 33
here, and was not part of this run (OI-39).
