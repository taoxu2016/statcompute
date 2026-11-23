# What an evaluation costs, part by part, on the cluster (2026-11-23)

Source: the report of plan `pass_cost_5m_warm`, which the maintainer ran twice with version 0.23.0
and rendered whole; it reached the repository on 2026-11-23 (UTC). 16 runs of the probe
`engine_passes`, four on each dataset of 5 million rows. The JSON records are not in the
repository; the numbers below were read from the report's table "What a pass costs". Conditions as
before: Runtime 17.3 LTS with Photon, autoscaling from 1 to 10 workers of 4 vCPUs, 8,000 providers,
the simple design. The working set is the engine's own for a model with strata or providers: 40
partitions of 125,000 rows, one per core.

The probe builds that working set and runs eight kinds of pass over it, five times each, the kinds
taking turns (`docs/research/pass-cost.md` says what each kind is). Ten of the 16 probes had ten
executors from start to end, two or three on every dataset. Everything below is from those ten;
a number is the median of a kind's five passes, and a range runs over the probes of a dataset.

## 1. An evaluation, taken apart

Seconds of one pass as the driver saw it. Each row adds one thing to the row above.

| Kind of pass | Cox, 50 | Cox, 100 | Logistic, 50 | Logistic, 100 |
|---|---|---|---|---|
| 40 tasks in the JVM that do nothing (`job`) | 0.11 to 0.19 | 0.10 to 0.11 | 0.11 to 0.12 | 0.10 to 0.12 |
| Python tasks sent one byte a row (`idle`) | 0.47 to 0.49 | 0.40 to 0.47 | 0.41 to 0.42 | 0.42 to 0.47 |
| The rows received and stacked (`rows`) | 0.93 to 0.98 | 1.21 | 0.84 to 0.89 | 1.18 to 1.27 |
| And the engine's kernel (`kernel`) | 1.71 to 1.78 | 2.75 to 2.93 | 1.14 to 1.19 | 1.62 to 1.72 |
| The engine's own evaluation (`evaluation`) | 1.70 to 1.78 | 2.75 to 2.97 | 1.12 to 1.15 | 1.75 to 1.80 |

By differences of the means, in seconds and as shares of the evaluation:

| Part | Cox, 50 | Cox, 100 | Logistic, 50 | Logistic, 100 |
|---|---|---|---|---|
| A job with Python tasks, whatever they do | 0.48 (27%) | 0.43 (15%) | 0.41 (35%) | 0.45 (27%) |
| Bringing the rows to Python | 0.48 (27%) | 0.78 (27%) | 0.45 (38%) | 0.77 (46%) |
| The kernel | 0.79 (45%) | 1.63 (57%) | 0.31 (26%) | 0.45 (27%) |

- **The Spark job is not the cost.** Forty tasks that do nothing come back in a tenth of a
  second. With Python tasks that are sent almost nothing the pass takes 0.4 to 0.5 seconds: 0.3
  more for starting the tasks in Python.
- **Bringing the rows to Python is a quarter to a half of an evaluation**: 0.45 to 0.5 seconds at
  50 features, 0.8 at 100. It grows with what is sent, which the sandbox did not show (probe A-11).
- **The kernel is half of a Cox evaluation and a quarter of a logistic one.**
- **Returning the sums and adding them costs nothing that shows**: the engine's evaluation takes
  what the `kernel` pass takes, whose tasks return five numbers.

## 2. Inside a task

The tasks time themselves. Seconds of the median task, and of the slowest, in the `kernel` pass:

| | Cox, 50 | Cox, 100 | Logistic, 50 | Logistic, 100 |
|---|---|---|---|---|
| Receiving and stacking 125,000 rows | 0.52 to 0.53 | 0.78 to 0.82 | 0.48 to 0.49 | 0.75 to 0.77 |
| The kernel | 0.67 to 0.70 | 1.41 to 1.50 | 0.24 | 0.48 to 0.50 |
| The median task | 1.22 to 1.23 | 2.21 to 2.32 | 0.72 to 0.73 | 1.24 to 1.27 |
| The slowest task | 1.47 to 1.50 | 2.54 to 2.69 | 0.90 to 0.92 | 1.45 to 1.52 |

A pass takes its slowest task and 0.2 to 0.3 seconds around it; that margin is the same in every
kind with Python tasks, whatever the tasks do. The slowest task is 0.2 to 0.4 seconds longer than
the median one: a fifth of a pass is waiting for the largest partition. A task sent one byte a row
takes 0.08 to 0.11 seconds, so 0.4 to 0.7 seconds of a task are the rows themselves: read from the
cache, written as Arrow, read by Python and stacked.

## 3. The two Cox kernels, and the limit on BLAS (OI-35)

Seconds in the kernel, median task, on the same partitions in the same probe:

| Kernel | 50 features | 100 features |
|---|---|---|
| The engine's, BLAS held to one thread, as the engine runs it | 0.67 to 0.70 | 1.41 to 1.50 |
| The engine's without the limit | 0.69 to 0.70 | 1.42 to 1.51 |
| The prototype's | 0.54 to 0.56 | 1.13 to 1.16 |

- **The limit on BLAS changes nothing**, for Cox or for the logistic model (0.24 against 0.23 to
  0.24 seconds; 0.48 to 0.50 against 0.48). That candidate is closed.
- **The engine's kernel takes a quarter more than the prototype's**: 0.14 seconds a task at 50
  features and 0.3 at 100, which made a pass 0.2 and 0.4 seconds longer. The difference between
  the two implementations' evaluations on 2026-11-22 was 0.3 and 0.6 seconds, so the kernel is
  most of it. Probe A-12 had found 27 to 39 percent in the sandbox and named the reason: the
  engine's kernel centres the features within each stratum, a pass over the rows and two copies
  of them, before it forms any product.
- The kernels take about three times what they took in the sandbox for the same number of rows
  (probe A-12). The probe does not say why.

## 4. Two tasks on one physical core are not the reason

`half` runs the kernel on every second partition: 20 tasks for 40 virtual cores, one for each
physical core. The kernel then took 0.62 to 0.67 seconds at 50 features and 1.34 to 1.41 at 100
for Cox, 4 to 8 percent less than with all 40 tasks, and 0.20 to 0.21 and 0.44 to 0.45 for the
logistic model, against 0.24 and 0.48 to 0.50. So the workers do not slow each other much: a kernel takes this long on these
machines whether its neighbour is busy or not. The `half` pass as a whole is longer than the
`kernel` pass, because the probe's filter stands between the cache and the tasks and slows the
rows by 0.15 to 0.35 seconds; that is the probe's doing and says nothing about the engine.

## 5. The cluster during the probes, and decision D-22

- Before 13 of the 16 probes the harness found the cluster short of ten executors and waited: 45
  to 181 seconds, and 945 and 882 at the start of the two runs. Six of those times the first
  count said ten and the count after the reads, as round 24 amended it, did not: "asked 2 times".
  **No probe started on fewer than ten executors.**
- Six probes lost executors on the way, five of them four at once. A probe lasts 43 to 150
  seconds, which is long enough for the autoscaler: in the fits without strata of the same session
  the cluster gave up four executors 70 to 90 seconds after the harness had last kept every core
  busy, every time (`2026-11-23-databricks-unstratified-default.md`, section 4).
- A pass during which executors went took 11 to 36 seconds instead of 1 to 3, three times.

## 6. What follows

What an evaluation is made of says where a faster one can come from, and how much there is:

| Part | Seconds of an evaluation | What could reduce it | State |
|---|---|---|---|
| The Cox kernel | 0.8 at 50 features, 1.6 at 100 | Centring that costs less: the prototype shows 0.2 and 0.4 seconds of it. Fewer temporaries in the kernel. A compiled kernel | Not started. The first two can be measured in the sandbox and confirmed with this probe (OI-47) |
| Bringing the rows to Python | 0.45 to 0.8 | Not sending them again: the stacked arrays kept on the executor between evaluations | An idea. Prior art to be searched and a sandbox trial made before anything is built (OI-48) |
| Starting the Python tasks | 0.3 | Tasks that live for the whole fit | Not worth its machinery alone; it would come with the row above |
| Waiting for the largest partition | 0.2 to 0.4 | Partitions of more equal size | Not looked at |
| The Spark job | 0.1 | Nothing | |

The logistic kernel is small (0.3 and 0.45 seconds); for that model the rows and the start of the
tasks are three quarters of an evaluation.

## 7. Limits

One cluster, one size of data, one layout, ten undisturbed probes of five passes a kind. The
probe evaluates at the coefficients the data were generated with, on a working set already built
and cached, so nothing here is a first pass. The parts are differences between kinds of pass, each
a median of five, not measurements of the parts themselves; the tasks' own timings (section 2)
agree with them. The engine's evaluations inside a fit were 0.1 to 0.35 seconds longer on
2026-11-22 than the probe's here; the two were not run in one session.
