# What a pass costs before its kernel runs

Probes A-11 and A-12 (`statcompute/lab/probes/packed_working_set.py`, `kernel_against_prototype.py`),
run on 2026-11-21 in the assistant's sandbox: one core, PySpark 4.0.1 in local mode, synthetic data.
One core means Spark's JVM and the Python worker share it, so these are indications for the
cluster, not predictions. They were run to explain two things seen there
(`docs/reports/2026-11-21-databricks-warm-runs.md`, sections 5 and 8): the engine's evaluations take
longer than its prototype's, and a task of a pass takes about a second where its kernel takes a
fraction of that.

## A-12: the engine's Cox kernel against the prototype's

One partition of the cluster's shape: 50,000 rows, 80 strata, already sorted. Best of seven.

| Features | Prototype's kernel | Engine's kernel | Ratio |
|---:|---:|---:|---:|
| 50 | 70.3 ms | 98.1 ms | 1.39 |
| 100 | 142.6 ms | 180.8 ms | 1.27 |

The sums are the same: log-likelihood identical, score within 3e-10, information within 8e-14
relative. The engine's kernel centres the features within each stratum before it forms any product
(`check_cox_centering.py` is why), which is one more pass over the rows and one more copy of them;
the prototype's does not. Over 100 partitions on 40 cores this is about a tenth of a second per
evaluation: part of the difference seen on the cluster, not all of it.

Not measured here, and a candidate for the rest: the engine limits BLAS to one thread in each
worker and the prototype does not, so on executors that are not fully busy the prototype's matrix
products can use idle cores.

## A-11: bringing the rows to Python, and whether packing them helps

400,000 rows in 8 partitions, whole strata per partition, sorted, cached. Two working sets from the
same rows: the usual one, with the features as a column of arrays; and one packed into 49 blocks of
about 8,192 rows, each block one row whose columns are the block's values as bytes. Two passes over
each: one that only receives the rows and counts them, one that runs the engine's Cox kernel. Best
of three, in seconds.

| Features | Pass | Column of arrays | Packed into blocks |
|---:|---|---:|---:|
| 50 | Only receives the rows | 1.74 | 1.63 |
| 50 | With the Cox kernel | 2.46 | 2.18 |
| 100 | Only receives the rows | 1.95 | 2.37 |
| 100 | With the Cox kernel | 3.84 | 3.66 |

The sums agree to 1e-14 relative.

- **Receiving the rows is half of a pass or more**: 1.7 to 2.0 seconds of 2.5 to 3.8.
- **It hardly depends on how much is received**: twice the features cost 0.2 seconds more. So most
  of it is a cost per task, about 0.2 seconds here for each of the 8, and not a cost per byte.
- **Packing the rows into bytes does not remove it**: within 0.4 seconds either way.

What follows for the engine: fewer, larger tasks per pass should cost less than many small ones, as
long as a pass still has a task for every core. At 5 million rows the default of 50,000 rows per
partition gives 100 tasks for 40 cores. Plan `engine_partitions_5m_warm` runs the engine at 50,000,
125,000 and 250,000 rows per partition side by side on the cluster (OI-25). The unstratified runs
of 2026-11-21 point the same way: 17 and 33 tasks per pass were faster than 100.

What does not follow: that a packed working set is worth building. It may still be on a cluster,
where the two processes do not share a core, but nothing here supports it.
