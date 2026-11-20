# Round 12 on the cluster: compare_5m and unstratified_5m, 2026-11-20

Sources: the two reports the maintainer rendered with notebook `03_report` after running version
0.10.0 (round 12) on the standard cluster, as pasted on 2026-11-20. Each report covers every record
in its results folder, so runs of version 0.9.0 and of 0.10.0 stand side by side, with the
prototypes' and pprof_spark's runs since 2026-11-16. The reports did not name a run's version. Runs
are assigned here by their task counts: round 12 removed one job of 101 tasks from every fit with
strata or providers, and changed the layout without strata. The JSON records were not available;
statements that need them say so. The engine's rows were typed from the reports into the table of
section 2.

Conditions as in `docs/reports/2026-11-16-databricks-compare-5m.md`: Runtime 17.3 LTS with Photon,
autoscaling from 1 to 10 workers of 4 vCPUs, 5 million rows, 8,000 providers, the simple design.

## 1. What these reports could not show

- **Whether round 12's estimates agree with pprof_py.** The agreement tables compared the first
  completed run of each adapter, which for the engine is a run of version 0.9.0. No number in them
  comes from round 12. In the sandbox, fits with strata and logistic fits of 0.10.0 are identical
  bit for bit to 0.9.0's (`docs/spec/engine.md`, section 5.1); on the cluster nothing was compared.
- **Where in a call the time went.** The phases, each evaluation's seconds and the heaviest stages
  are in the records, not in the report.
- **Anything about pprof_spark in that session**: it was not installed there
  (`No module named 'pprof_spark'`).

Round 14 changes the report so that the same records answer the first two points (section 4).

## 2. The engine's runs by version

"Per evaluation" is the fit phase over the evaluations it holds: four for 0.9.0 and for the logistic
model, three for Cox in 0.10.0, whose first evaluation builds the working set and is counted under
"prepare". "Whole minus fit" therefore contains that first evaluation for Cox in 0.10.0, about one
"per evaluation" more than in 0.9.0; for the logistic model it contains the pass for the provider
effects in both versions. A stall is counted when whole minus fit is 30 seconds or more.

| Model, features | Version | Runs | Whole call, s: median (range) | Per evaluation, s: median (range) | Whole minus fit, s: median (range) | Stalls | Runs with a caveat | Runs with a task 50 times its stage's median |
|---|---|---:|---|---|---|---:|---:|---:|
| Cox with strata, 50 | 0.9.0 | 8 | 34.3 (12.9 to 96.3) | 3.7 (2.1 to 12.1) | 19.9 (4.3 to 48.0) | 3 | 2 | 2 |
| Cox with strata, 50 | 0.10.0 | 2 | 59.1, 75.9 | 3.1, 3.8 | 49.7, 64.4 | 2 | 2 | 1 |
| Cox with strata, 100 | 0.9.0 | 4 | 21.9 (18.7 to 29.5) | 3.7 (3.3 to 4.0) | 6.4 (5.5 to 14.8) | 0 | 0 | 0 |
| Cox with strata, 100 | 0.10.0 | 2 | 49.3, 59.2 | 5.6, 4.4 | 32.6, 46.1 | 2 | 1 | 2 |
| Logistic, 50 | 0.9.0 | 6 | 18.8 (15.5 to 28.2) | 1.8 (1.6 to 2.4) | 11.2 (8.9 to 20.2) | 0 | 0 | 0 |
| Logistic, 50 | 0.10.0 | 2 | 24.1, 58.9 | 2.3, 2.3 | 14.9, 49.7 | 1 | 2 | 1 |
| Logistic, 100 | 0.9.0 | 4 | 22.0 (21.0 to 23.4) | 2.4 (2.3 to 2.5) | 12.5 (11.6 to 13.4) | 0 | 0 | 0 |
| Logistic, 100 | 0.10.0 | 2 | 28.3, 67.5 | 3.2, 3.5 | 15.4, 53.5 | 1 | 2 | 1 |
| Cox without strata, 50 | 0.9.0 | 10 | 58.0 (19.8 to 139) | 5.9 (3.6 to 23.6) | 32.9 (5.1 to 50.1) | 6 | 4 | 0 |
| Cox without strata, 50 | 0.10.0 | 2 | 129, 29.9 | 4.8, 3.8 | 114.7, 18.6 | 1 | 1 | 1 |
| Cox without strata, 100 | 0.9.0 | 8 | 58.5 (30.3 to 61.2) | 8.2 (5.9 to 10.7) | 21.1 (6.8 to 31.1) | 1 | 2 | 0 |
| Cox without strata, 100 | 0.10.0 | 2 | 30.8, 68.7 | 5.8, 18.9 | 13.3, 11.9 | 0 | 1 | 0 |

The median of 26.8 seconds for Cox with strata at 50 features in the task of round 12 was over the
six runs of 0.9.0 that existed then; two more have been added since.

## 3. Findings

1. **One job fewer per fit, confirmed.** With strata or providers a fit of 0.10.0 ran 101 tasks
   fewer than one of 0.9.0: 536 against 637, 736 against 837, 568 against 669, 768 against 869.
2. **Round 12 did not make these calls faster.** With strata or providers, six of the eight calls
   of 0.10.0 took 49 to 76 seconds, against medians of 19 to 34 seconds before. The time is outside
   the fit phase in every one of them.
3. **The fit phase is no shorter.** Per evaluation it is the same for Cox with strata at 50 features
   and longer elsewhere: 4.4 and 5.6 seconds against 3.3 to 4.0 for Cox at 100 features, 2.3 against
   1.8 and 3.2 to 3.5 against 2.4 for the logistic model. Seven of those eight runs carry a caveat
   about the cluster's size, against two of twenty-two before, so the two versions were not
   measured under the same conditions. The sandbox had the Python side of a pass 8 to 30 percent
   shorter; the cluster does not show it and does not refute it.
4. **The stalls are older than round 12 and larger than anything it changed.** 0.9.0 had them in
   three of eight Cox fits with strata at 50 features and in six of ten without strata. In the
   session of round 12 they were more frequent: six of eight runs with strata or providers, against
   three of twenty-two. Whether the working set of round 12 is more exposed to them, or that
   session's cluster was, these reports cannot say. They may not all be one thing. In five of the
   six stalled runs of 0.10.0 one task ran 58 to 252 times its stage's median, and the prototype
   `lab_spark` has ratios above 100 in six of its runs, with whole calls of 30 to 128 seconds. Of
   0.9.0's three stalls with strata one had such a task; its six without strata had none above 13,
   so there every task of a stage was slow.
5. **Without strata the new layout uses more of the cluster and is not clearly faster.** Its two
   calls on a steady cluster took 29.9 and 30.8 seconds with 14 and 18 effective cores, against 7
   to 12 before. The medians of 0.9.0 were 58 seconds; its best calls were 19.8 and 30.3. Per
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
