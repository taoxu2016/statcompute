"""A Markdown report from run records: what ran where, timings, and numerical agreement."""
from __future__ import annotations

from typing import Any, Dict, List

from ..validation import compare_records, flags_across_records
from .. import references
from .results import pass_costs, halved_steps, implementations, medians, summary, to_markdown, where_time_went

TIMING = ["model", "rows", "features", "design", "adapter", "role", "version", "status", "compare_s", "wall_s", "fit_s", "iterations",
          "executors", "tasks", "effective_cores", "share_of_cores", "gc_s", "shuffle_gb", "worst_straggler_ratio",
          "driver_peak_rss_gb", "cluster_changed", "caveats", "note"]
AGREEMENT = ["dataset", "model", "adapter", "version", "runs", "reference", "reference_halved_steps", "like_for_like",
             "max_abs_coefficient_difference",
             "max_coefficient_difference_in_se", "max_rel_se_difference", "rel_log_likelihood_difference",
             "coefficients_over_T_COEF", "iterations"]


def render(records: List[Dict[str, Any]], reference: str = "pprof_py", title: str = "Benchmark report") -> str:
    if not records:
        return f"# {title}\n\n_no records_\n"
    static = next((r["environment"]["static"] for r in records if r.get("environment")), {})
    dbx = static.get("databricks") or {}
    lock = references.lock()
    parts = [f"# {title}\n", f"Records: {len(records)}; plans: {sorted({r['plan'] for r in records})}.\n",
             "## Environment\n",
             to_markdown([{"spark": static.get("spark_version"), "runtime": dbx.get("runtime"),
                           "worker_type": dbx.get("worker_type"), "driver_type": dbx.get("driver_type"),
                           "scaling": dbx.get("scaling_type"), "photon": dbx.get("photon"),
                           "driver_memory_gb": None if not static.get("driver_memory_bytes")
                           else round(static["driver_memory_bytes"] / 1e9, 1)}]),
             "## Implementations\n",
             "Every build that ran, with the commit it reported and the commit `REFERENCES.lock` pins today. The caveat about "
             "the pinned build, below, is judged against today's lock, so pinning a build clears it for the runs of that build. "
             "`code_sent` says how the engine's code reached the executors: by value, inside each task, or as an archive.\n",
             to_markdown(implementations(records, lock)),
             "## Runs\n",
             "`compare_s` is the number to compare: for a Spark implementation the whole call from the stored table (its own "
             "preparation included, whichever phase it reports it under); for a single-machine one the fit alone, with the data "
             "already in memory. A run with an entry under `caveats` is not a fair timing. A status of `running` means the "
             "process died during that fit, typically because the driver ran out of memory.\n",
             to_markdown(summary(records, lock), TIMING),
             "## Medians by version\n",
             "Completed runs by model, size, adapter and the version of this repository that made them. A results folder "
             "that has collected several rounds is read here round by round. `without_caveat` leaves out the runs "
             "whose timing the harness itself calls unfair.\n",
             to_markdown(medians(records, lock)),
             "## Where the time went\n",
             "Completed runs of the Spark implementations. `other_s` is the part of the call outside the reported phases. "
             "For the engine's Cox fits from version 0.10.0 on, `prepare_s` contains the first evaluation, which builds the "
             "working set, and `fit_s` is the evaluations after it. The slowest task is the longest single task among the "
             "run's heaviest stages: far above `stage_median_task_s`, it is a stall, and `stage_tasks` says where "
             "(the input's partitions, or the working set's). A stage is named by the line of the action that ran it, "
             "in the source of the run's version. `partitions` is the working set's; `partitions_from` is the rule that "
             "gave the number and `cores_at_start` the cores the cluster had when the fit counted them (the engine, from "
             "version 0.20.0): by default a fit with strata or providers has one partition per core at that moment "
             "(decision D-21), so one that started on a smaller cluster shows fewer partitions here. `executors_held` is "
             "what the warm-up that asks for the plan's executors reported before the fit; where it says the full number "
             "and `executors` says fewer, the cluster lost them in the seconds between, and the run carries a caveat.\n",
             to_markdown(where_time_went(records)),
             *([] if not pass_costs(records) else [
                 "## What a pass costs\n",
                 "The probe `engine_passes` builds the engine's working set and runs a ladder of passes over it, each kind "
                 "several times, the kinds taking turns. Each kind adds one thing to the kind before it: `job` is as many "
                 "tasks as there are partitions, in the JVM, that do nothing; `idle` is Python tasks that are sent one byte a "
                 "row; `rows` receives the rows and stacks them as the engine does; `kernel` also runs the engine's kernel, "
                 "BLAS held to one thread as the engine holds it; `kernel_before` runs the engine's Cox kernel as it was up to "
                 "version 0.24.0; `kernel_free` runs the engine's without that limit; `kernel_plain` runs "
                 "the prototype's Cox kernel on the same rows; `evaluation` is the engine's own pass, its results collected "
                 "and summed; `half` is `kernel` on every second partition, one task for every two cores. `seconds` are the "
                 "passes as the driver saw them. The other columns come from the tasks, each the median over the passes: "
                 "`task_s` and `slowest_task_s` are a task's own seconds from its first line to its last, `receive_s` the "
                 "part until its rows were stacked, `kernel_s` and `slowest_kernel_s` the part in the kernel. "
                 "`around_tasks_s` is a pass's seconds less its slowest task's: what the job costs outside its tasks. "
                 "`log_likelihood_vs_kernel` sets the log-likelihood a kind's last pass added up beside the `kernel` "
                 "kind's: the same bits, or their relative difference.\n",
                 to_markdown(pass_costs(records))]),
             f"## Numerical agreement with {reference}\n",
             "Only rows with `like_for_like` true compare implementations that fit exactly the same model. Every run is "
             "compared; a row shows, for one adapter and version, the run furthest from the reference. "
             "Iterations are listed as [adapter, reference] and are not a criterion. Where `reference_halved_steps` is not "
             "zero the reference may have stopped short of the root, and a difference above T_COEF is not the adapter's "
             "(decision D-09).\n",
             to_markdown(compare_records(records, reference), AGREEMENT)]
    halved = halved_steps(records)
    if halved:
        parts += ["## Halved steps in the reference's Newton loop\n",
                  "pprof_py's Cox fit, watched from outside: a fit of n iterations calls its likelihood n + 2 times when no "
                  "step is halved (the start, each iteration, the null model), and once more for every halved step. A step "
                  "halved at the last iteration leaves the estimate short of the root by 1 - 2^-h of that step, h being the "
                  "number of halvings, with `converged` true (`docs/spec/engine.md`, section 5.3).\n",
                  to_markdown(halved)]
    flags = flags_across_records(records)
    if flags:
        parts += ["## Provider flags across runs\n",
                  "For each test: how many providers the first run flagged, whether every run flagged the same providers "
                  "the same way, and the smallest distance of a provider's statistic from the cut-off in any run "
                  "(|z| from the normal quantile; the exact p-value from 1 - level). A flag can change only for a provider "
                  "closer to the cut-off than the computations are to each other.\n",
                  to_markdown(flags)]
    notes = sorted({(r["adapter"]["name"], r["model"]["label"], note) for r in records
                    for note in ((r.get("support") or {}).get("notes") or [])})
    if notes:
        parts += ["## Notes from the adapters\n",
                  to_markdown([{"adapter": a, "model": m, "note": n} for a, m, n in notes])]
    return "\n".join(parts)
