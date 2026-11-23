# The default without strata on the cluster (2026-11-23)

Source: the report of plan `unstratified_cores_5m_warm` run with version 0.23.0, rendered whole; it
reached the repository on 2026-11-23 (UTC). It is the first cluster run of the default of decision
D-23, and it was run in the same session as the probes of
`2026-11-23-databricks-pass-cost.md`. 24 fits of Cox without strata at 5 million rows, three
layouts taking turns, four repeats: the engine's default, one partition per core with the rows
placed by a hash (`engine`); the input's own partitions, the default before (`engine_input`); one
per core with the rows placed by row identifier and sorted (`engine_cores`). All 24 completed in
three iterations. The default laid out 40 partitions, "one per core", on 40 cores in all eight of
its fits, and ran 396 and 428 tasks a fit, as the adapter that had its layout did on 2026-11-22.

## 1. Evaluations

Later evaluations, in seconds, in the calls at a steady ten executors: the median, and the range.

| Features | The default, by a hash | The input's partitions | One per core, sorted |
|---|---|---|---|
| 50 | 2.7 (2.4 to 3.0) | 4.05 (3.8 to 4.5) | 2.6 (2.4 to 3.0) |
| 100 | 3.65 (3.5 to 4.2) | 3.9 (3.7 to 4.2) | 3.6 (3.5 to 4.0) |

- **At 50 features the default's evaluation is a third shorter than the input's partitions'**, as
  on 2026-11-22 (2.7 against 3.9 then).
- **At 100 features it is 6 percent shorter, not the 26 to 29 percent of 2026-11-22.** The input's
  33 partitions took 3.7 to 4.2 seconds an evaluation in this session, as on 2026-11-21 (3.8 to
  4.2), and not the 5.0 to 6.0 of the session D-23 was proposed on. That session's input layout
  was slow for a reason that did not return (OI-46, closed as a difference between sessions), and
  the proposal's "26 to 31 percent at both widths" overstated the case at 100 features.

## 2. Whole calls

In seconds. "Undisturbed" are the calls at ten executors before which the harness neither waited
for executors nor read the table for more than three seconds.

| Features | | The default, by a hash | The input's partitions | One per core, sorted |
|---|---|---|---|---|
| 50 | Median of four, best | 21.3, 15.8 | 22.1, 18.3 | 26.9, 14.4 |
| 50 | Undisturbed | 15.8 | 18.3 | 14.4, 15.9 |
| 100 | Median of four, best | 19.2, 18.9 | 24.1, 23.0 | 20.4, 20.1 |
| 100 | Undisturbed | 21.2, 19.3, 19.1, 18.9 | none | 20.2, 20.5, 20.1 |

- **At 50 features**: 15.8 seconds for the default against 18.3 for the input's partitions, one
  undisturbed call each.
- **At 100 features the input's partitions had no undisturbed call**: before three of its four
  fits the harness waited two minutes for executors, and the fourth lost four during the fit. Its
  first evaluations took 9.5 to 10.3 seconds on new executors. From its evaluations alone an
  undisturbed call would be within a second of the default's, and its best calls on 2026-11-21
  were 18.5 and 18.7 seconds: as fast as the default's 18.9 to 19.3 here. **At 100 features, where
  the input happens to have almost one partition per core, D-23 gains little.**
- **The sort**: first evaluations of 7.5 to 7.6 seconds against 6.1 to 6.3 at 100 features, so 1.3
  seconds, and whole calls a second longer. At 50 features the two undisturbed sorted calls (14.4
  and 15.9) straddle the default's one.

## 3. What this means for D-23

The default does what it was meant to: one partition per core whatever the input's files are, no
sort, evaluations as short as any layout's. Its measured gain is a third of an evaluation where the
input has far fewer partitions than the cluster has cores (17 for 40), and a few percent where it
has nearly as many (33 for 40). It was never slower. The statement of its evidence in
`DECISIONS.md` is corrected accordingly.

## 4. The cluster, and decision D-22

- **No fit started on fewer than ten executors.** Before 9 of the 24 fits the harness waited for
  executors, 121 to 172 seconds (754 at the start); five of those times the first count said ten
  and the count after the reads did not ("asked 2 times"). On 2026-11-22, before the count was
  amended, one fit in 24 of this plan and eight in 64 of another had started on six.
- **Three fits lost four executors during the fit.** One evaluation then took 41.3 seconds and one
  first evaluation 36.7: a loss in the middle of a pass costs half a minute, not a slower pass.
- **The cluster gave up four executors 70 to 90 seconds after the harness had last kept every
  core busy, every time**: a wait for executors, three fits, a loss, eight times in a row. The
  fits use half the cores or fewer, and the autoscaler takes that for an idle cluster. With three
  adapters taking turns the loss came round to the same one: six of the eight fits on the input's
  partitions came right after a wait.
- One call spent 14.2 seconds in a single task outside the engine's phases, in the job that opens
  the stored table (`SAMSnapshotHelper.scala:43`): 37.9 seconds for a call whose phases took 23
  (OI-45). It carries no caveat.

## 5. Limits

One session, four repeats, and at 50 features one undisturbed call for two of the three layouts.
The coefficients were not compared on the cluster: the plan has no reference. In the sandbox the
three layouts agree within 1e-9 standard errors and with pprof_py within the tolerance classes.
