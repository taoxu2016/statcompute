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

## What the cluster said

Plan `engine_partitions_5m_warm` ran on 2026-11-21
(`docs/reports/2026-11-21-databricks-partition-size.md`). At 5 million rows on 40 cores, later
evaluations took 17 to 28 percent less with 40 tasks per pass than with 100, and with 20 tasks about
as long as with 100: fewer, larger tasks cost less until cores go idle. With one task per core a
second or more of every evaluation still moved with neither the number of tasks nor their rows
(OI-40).

## The probe for the cluster (round 25)

The sandbox has one core; what an evaluation costs around its tasks, and what four workers on two
physical cores do to a kernel, can only be measured on the cluster. `statcompute/lab/passes.py`
does it there: adapter `engine_passes`, plan `pass_cost_5m_warm`. It builds the working set the
engine builds (a test asks the probe's evaluation and the engine's for the same bits) and runs
eight kinds of pass over it, five times each, the kinds taking turns:

| Kind | What it adds to the kind before | What it answers |
|---|---|---|
| `job` | As many JVM tasks as partitions, doing nothing | What a job costs on this platform whatever it does |
| `idle` | Python tasks sent one byte a row | What starting a task in a Python worker adds |
| `rows` | The rows received and stacked, as in the engine | What bringing the rows to Python costs |
| `kernel` | The engine's kernel, BLAS at one thread | The kernel inside a task, as the engine runs it |
| `kernel_before` | The same rows through the Cox kernel of version 0.24.0 (from round 27) | What reusing the kernel's arrays saved (OI-47) |
| `kernel_free` | The same without the limit | Whether one BLAS thread is the right limit on two physical cores (OI-35) |
| `kernel_plain` | The prototype's Cox kernel on the same rows | What centring the features costs on the cluster (OI-35) |
| `evaluation` | The engine's own pass, results collected and summed | What returning the sums and adding them costs beyond `kernel` |
| `half` | `kernel` on every second partition | What two tasks on one physical core cost each other |

The tasks time themselves, so a pass's seconds less its slowest task's are what the job costs
outside its tasks. Read with these in mind: the passes run after a first evaluation has built and
cached the working set, so nothing here is a first pass; the probe evaluates at the coefficients
the data were generated with; `half` leaves the idle tasks of the other partitions in the job, so
its seconds are a job of all the tasks while its kernels have the cores to themselves.

In the sandbox (one core, local mode, 6,000 rows) the probe's evaluation returns the engine's
log-likelihood bit for bit and the prototype's kernel the same to 1e-9. No timing from the sandbox
is meant.

## What the cluster answered (2026-11-23)

`docs/reports/2026-11-23-databricks-pass-cost.md` has the numbers: ten probes at a steady ten
executors, 40 tasks of 125,000 rows. Against what this note said from the sandbox:

- **A-11 said receiving the rows hardly depends on how much is received. On the cluster it does.** A
  task sent one byte a row takes 0.1 seconds; with the rows it takes 0.5 seconds at 50 features and
  0.8 at 100. The cost per task that A-11 saw is there as well, 0.3 seconds a pass for starting
  tasks in Python, but it is the smaller part. On one core the JVM and the Python worker take turns,
  which may be what hid it.
- **A-12's quarter is on the cluster too**: the engine's Cox kernel 0.67 to 0.70 and 1.41 to 1.50
  seconds a task, the prototype's 0.54 to 0.56 and 1.13 to 1.16. Both take about three times what
  they took here for the same rows.
- **The two candidates this note left open are closed**: the limit on BLAS changes nothing, and
  with half the tasks a kernel is 4 to 8 percent shorter, so the workers do not slow each other.
- **The Spark job itself is a tenth of a second**, and returning the sums costs nothing that shows.

So an evaluation is a job of Python tasks (0.4 to 0.5 seconds), the rows (0.45 to 0.8) and the
kernel (0.3 to 1.6), and the two that can be reduced without changing the platform are the kernel
(OI-47) and the rows (OI-48).

## The Cox kernel with the same sums in less memory (round 27, OI-47)

The cluster said the Cox kernel is half of a Cox evaluation and that a quarter of it is centring.
Read with that in mind, the kernel of version 0.24.0 makes eleven arrays the size of the rows, or of
the event times by the features, in every call: the stratum means spread over the rows, the centred
rows, the rows times the risk scores, their running sums, three for the sums at the event times,
three for the means at the event times and their weighted copy, and the rows times the expected
events. Version 0.25.0 makes three and reuses them, with every floating-point operation unchanged
and in the same order (`docs/spec/engine.md`, section 2).

One partition of the cluster's shape, 125,000 rows in 200 strata, sorted; Breslow ties; best of
seven calls on the sandbox's one core, and the peak of the memory NumPy asked for during a call:

| Features | Kernel | Milliseconds | Of 0.24.0 | Peak, times the rows |
|---:|---|---:|---:|---:|
| 50 | Version 0.24.0 | 280 | 1.00 | 5.4 |
| 50 | Version 0.25.0, centring in a copy | 173 | 0.62 | 3.0 |
| 50 | Version 0.25.0, in the task's own rows | 158 | 0.56 | 2.0 |
| 50 | The prototype's, which does not centre | 226 | 0.81 | 3.3 |
| 100 | Version 0.24.0 | 526 | 1.00 | 5.3 |
| 100 | Version 0.25.0, centring in a copy | 397 | 0.75 | 2.8 |
| 100 | Version 0.25.0, in the task's own rows | 348 | 0.66 | 1.8 |
| 100 | The prototype's, which does not centre | 393 | 0.75 | 3.2 |

The outputs of the two versions are identical bit for bit, here and in 6,720 comparisons over
shapes, tie methods and options. The kernel that centres is now faster than the prototype's that
does not.

What is left in a call at 100 features, by measuring the pieces apart on the same core: the two
matrix products take 93 of the 348 milliseconds, and a pass over an array the size of the rows
takes 17 when the array exists and 28 when it has to be made. Forming the two products in their
symmetric form would take 33 milliseconds off (8 at 50 features), and would change the last bits of
the information; not done. On the cluster the kernels took about 2.5 times what they take here, so
what this is worth there is for the probe to say: plan `cox_kernel_5m_warm` runs both versions and
the prototype's inside the same tasks.

**On the cluster (2026-11-23).** Inside the same tasks, 125,000 rows each
(`docs/reports/2026-11-23-databricks-cox-kernel.md`): the kernel of version 0.24.0 took 0.72 to 0.75
seconds at 50 features and 1.54 to 1.60 at 100; the kernel of version 0.25.0 takes 0.43 and 0.90 to
0.94, which is 57 to 60 percent; the prototype's 0.56 and 1.20. The sandbox's 56 and 66 percent
were a fair indication at 50 features and too cautious at 100, where the kernel before suffered
most in its slowest tasks. A Cox evaluation is 0.33 to 0.39 and 0.87 to 0.98 seconds shorter.

