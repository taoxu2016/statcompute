# The Cox kernel that reuses its arrays, on the cluster (2026-11-23)

Source: the report of plan `cox_kernel_5m_warm` run with version 0.25.0, rendered whole; it reached
the repository on 2026-11-23. Six runs of the probe `engine_passes`, three on each Cox dataset of 5
million rows. The JSON records are not in the repository; the numbers were read from the report's
table "What a pass costs". Conditions as before: Runtime 17.3 LTS with Photon, autoscaling from 1 to
10 workers of 4 vCPUs, 8,000 providers, the simple design.

Version 0.25.0 changed how the engine's Cox kernel uses memory and nothing in what it adds up
(`docs/research/pass-cost.md`). Each probe ran, five times in turn over the same cached working
set: the rows received and stacked (`rows`); the kernel as it was up to version 0.24.0
(`kernel_before`); the kernel as it is (`kernel`); the prototype's, which does not centre
(`kernel_plain`); and the engine's own evaluation. Four probes had ten executors from start to end,
two on each dataset: 40 tasks of 125,000 rows. The tables are from those four.

## 1. Inside a task

Seconds in the kernel, median task, and the slowest task's:

| Features | Kernel before | Kernel now | The prototype's | Now, of before |
|---|---|---|---|---|
| 50, median | 0.72 to 0.75 | 0.43 | 0.56 | 57 to 60% |
| 50, slowest | 0.90 to 0.99 | 0.54 to 0.56 | 0.66 to 0.69 | |
| 100, median | 1.54 to 1.60 | 0.90 to 0.94 | 1.20 to 1.21 | 58 to 59% |
| 100, slowest | 2.13 to 2.39 | 1.12 to 1.15 | 1.45 to 1.46 | |

- **The kernel takes 57 to 60 percent of what it took**: 0.3 seconds less in a task at 50 features,
  0.65 less at 100. The sandbox had said 56 and 66 percent.
- **The slowest task gained most at 100 features**: its kernel took 2.1 to 2.4 seconds before and
  1.1 to 1.2 now. Before, the slowest kernel was 40 to 50 percent longer than the median one; now
  20 to 30.
- **The kernel that centres is now a quarter faster than the prototype's that does not**, where it
  was a quarter slower.

The two other probes say the same. On eight to nine executors, with 32 tasks of 156,250 rows:
0.97 seconds before, 0.58 now, 0.72 for the prototype's. In the probe that lost four executors
during its last pass: 1.55, 0.87 and 1.17.

## 2. A pass, and the evaluation

Seconds of a pass as the driver saw it, median of five:

| Features | With the kernel before | With the kernel now | With the prototype's | The engine's evaluation |
|---|---|---|---|---|
| 50 | 1.70 to 1.81 | 1.43 to 1.54 | 1.46 to 1.57 | 1.31 to 1.48 |
| 100 | 3.13 to 3.36 | 2.20 to 2.33 | 2.55 | 2.26 to 2.38 |

- **An evaluation is 0.33 to 0.39 seconds shorter at 50 features and 0.87 to 0.98 at 100**, probe
  by probe, the engine's evaluation against the pass with the kernel before: 18 to 23 percent and
  28 to 29 percent. At 100 features that is more than the median kernel's 0.65 seconds, because a
  pass waits for its slowest task.
- With the kernel before, the pass was 0.2 seconds longer than with the prototype's at 50 features
  and 0.6 to 0.8 at 100; it is now as long at 50 and 0.2 to 0.35 shorter at 100. What separated the
  two implementations' Cox evaluations (OI-35) is gone.
- What a Cox evaluation is made of now, by the parts of
  `2026-11-23-databricks-pass-cost.md`: 0.45 seconds for a job of Python tasks, 0.45 and 0.8 for
  the rows, and about 0.45 and 1.0 for the kernel.

## 3. Not in this report

- **Whether the two kernels added up the same bits on the cluster.** They do in the sandbox, in
  every output. The records hold each kernel's log-likelihood and the report did not print them;
  from version 0.26.0 it does (`log_likelihood_vs_kernel`), so rendering this session again shows
  it without a new run.
- **A fit.** The probe times passes. The engine's fits with this kernel have not run on a cluster;
  their coefficients have not been set beside a reference's there.
- The memory of a task. In the sandbox the kernel's peak fell from 5.3 times the rows to 1.8.

## 4. The cluster

Before every probe the harness waited for executors, 452 to 1,101 seconds, where it had waited
about two minutes in the sessions before: 73 minutes of waiting for six probes that took seven.
Once the second ask ran into its limit of 15 minutes with eight executors, and that probe ran on
eight and nine, flagged. Why the cluster was slow to give executors that day is not in the report.

For this plan the waiting bought nothing: a kernel's seconds inside a task do not depend on how
many executors there are (0.37 seconds per 100,000 rows on eight to nine executors, 0.34 on ten).
A plan that only times kernels should not hold the cluster.

## 5. Limits

One session; two undisturbed probes of five passes on each dataset. The comparison is inside the
tasks and inside one session, on the same partitions, so it does not rest on the cluster's state.
The engine's evaluations inside a fit are a little longer than the probe's (0.1 to 0.35 seconds on
2026-11-22), and a whole call has four of them, so what a fit gains is to be measured by fits.
