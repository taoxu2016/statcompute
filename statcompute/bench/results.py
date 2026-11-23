"""Read run records back and flatten them for tables."""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
from typing import Any, Dict, Iterable, List, Optional

from .environment import short_of
from .runner import local_path


def load_records(results_root: str, plan: Optional[str] = None) -> List[Dict[str, Any]]:
    pattern = os.path.join(local_path(results_root), "runs", plan or "*", "*.json")
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            records.append(json.load(f))
    return records


def _gb(value) -> Optional[float]:
    return None if value is None else round(value / 1e9, 3)


NOT_PINNED = "the installed build is not the commit pinned in REFERENCES.lock"


def _pin(record: Dict[str, Any], lock: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The lock's entry for a record's implementation: its own name, or the name it is a variant of."""
    names = (lock or {}).get("references") or {}
    name = (record.get("adapter") or {}).get("name") or ""
    keys = [key for key in names if name == key or name.startswith(key + "_")]
    return names[max(keys, key=len)] if keys else None


def _matches(record: Dict[str, Any], lock: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Whether the build a record ran is the one the lock pins now. None when either is unknown."""
    found, pinned = (record.get("adapter") or {}).get("commit"), (_pin(record, lock) or {}).get("commit")
    return None if not (found and pinned) else bool(pinned.startswith(found) or found.startswith(pinned))


def caveats(record: Dict[str, Any], lock: Optional[Dict[str, Any]] = None) -> List[str]:
    """A record's caveats. With a lock, the one about the pinned build is judged against that lock,
    not against the lock of the day of the run: pinning a build afterwards clears it for the runs of
    that build, and pinning another raises it. The caveat for a fit that ran on fewer executors than
    its plan holds is judged here too, from what the record measured, so that the records written
    before version 0.21.0 carry it as the later ones do."""
    kept = [c for c in (record.get("caveats") or []) if lock is None or c != NOT_PINNED]
    short = short_of(record.get("warmup"), (record.get("environment") or {}).get("during"))
    if short and short not in kept:
        kept.append(short)
    if lock is not None and _matches(record, lock) is False:
        kept.append(NOT_PINNED)
    return kept


def of_versions(records: Iterable[Dict[str, Any]], versions: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    """The records made with the given versions of this repository; all of them when none is given.
    A results folder keeps every session of a plan; the report of one session is a fraction of
    the size, and it is the session that a reading is about."""
    wanted = {str(v).strip() for v in (versions or []) if str(v).strip()}
    return [r for r in records if not wanted or str(r.get("statcompute")) in wanted]


def implementations(records: Iterable[Dict[str, Any]], lock: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every build that ran: adapter, its own version, the commit it reported, the commit the lock
    pins and whether they are the same, how the engine's code reached the executors (by value, or as
    an archive; from version 0.18.0), and the number of records."""
    groups: Dict[Any, int] = {}
    last: Dict[Any, Dict[str, Any]] = {}
    sent: Dict[Any, set] = {}
    for r in records:
        a = r.get("adapter") or {}
        if r.get("status") in ("unavailable", None):
            continue
        key = (a.get("name"), str(a.get("version")), a.get("commit"))
        groups[key] = groups.get(key, 0) + 1
        last[key] = r
        how = ((((r.get("fit") or {}).get("extras") or {}).get("execution")) or {}).get("code_sent")
        if how:
            sent.setdefault(key, set()).add(how)
    rows = []
    for (name, version, commit), count in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), str(item[0][2]))):
        pinned = (_pin(last[(name, version, commit)], lock) or {}).get("commit")
        same = _matches(last[(name, version, commit)], lock)
        rows.append({"adapter": name, "role": (last[(name, version, commit)].get("adapter") or {}).get("role"), "version": version,
                     "commit": commit, "pinned_commit": pinned, "is_the_pinned_build": None if same is None else ("yes" if same else "no"),
                     "code_sent": ", ".join(sorted(sent.get((name, version, commit), ()))) or None, "records": count})
    return rows


def summary(records: Iterable[Dict[str, Any]], lock: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """One flat row per run: what ran, whether it completed, and what it took."""
    rows = []
    for r in records:
        spec, timing = r["dataset"]["spec"], r.get("timing") or {}
        spark, process, fit = r.get("spark") or {}, r.get("process") or {}, r.get("fit") or {}
        env = r.get("environment") or {}
        after = env.get("after") or {}
        during = env.get("during") or {}
        low, high = during.get("executors_min"), during.get("executors_max")
        rows.append({
            "model": r["model"]["label"], "rows": spec["rows"], "providers": spec["providers"],
            "features": spec["features"], "design": spec["design"],
            "adapter": r["adapter"]["name"], "role": r["adapter"].get("role"), "repeat": r.get("repeat"),
            "version": r.get("statcompute"),          # of this repository when the run was made
            "status": r["status"],
            "compare_s": None if not timing else round(
                timing["wall_s"] if r["adapter"].get("kind") == "spark" else (timing.get("phases") or {}).get("fit", timing["wall_s"]), 2),
            "wall_s": None if not timing else round(timing["wall_s"], 2),
            "fit_s": None if "fit" not in (timing.get("phases") or {}) else round(timing["phases"]["fit"], 2),
            "iterations": fit.get("iterations"), "converged": fit.get("converged"),
            "max_abs_z": None if not r.get("truth") or r["truth"]["max_abs_z"] is None else round(r["truth"]["max_abs_z"], 2),
            "executors": after.get("executors") if low == high or low is None else f"{low}-{high}",
            "cores": after.get("cores"),
            "share_of_cores": None if spark.get("share_of_attached_cores") is None
            else round(spark["share_of_attached_cores"], 2),
            "cluster_changed": env.get("cluster_changed"),
            "tasks": spark.get("tasks"),
            "effective_cores": None if spark.get("effective_cores_by_run_time") is None
            else round(spark["effective_cores_by_run_time"], 2),
            "executor_cpu_s": None if spark.get("executor_cpu_s") is None else round(spark["executor_cpu_s"], 1),
            "gc_s": None if spark.get("gc_s") is None else round(spark["gc_s"], 1),
            "shuffle_gb": _gb(spark.get("shuffle_write_bytes")),
            "worst_straggler_ratio": None if spark.get("worst_straggler_ratio") is None
            else round(spark["worst_straggler_ratio"], 1),
            "driver_peak_rss_gb": _gb(process.get("peak_rss_with_children_bytes") or process.get("peak_rss_bytes")),
            "caveats": "; ".join(caveats(r, lock)),
            "note": "" if not r.get("error") else r["error"]["message"][:160],
        })
    return rows


def medians(records: Iterable[Dict[str, Any]], lock: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Completed runs by model, size, adapter and version of this repository: how many, how many of
    them without a caveat, and the median and the smallest time to compare, over all of them and
    over those without a caveat. The version separates the rounds of the engine in one results folder."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in summary(records, lock):
        if row["status"] == "ok" and row["compare_s"] is not None and row["role"] != "probe":    # a probe's seconds are not a fit's
            groups.setdefault((row["model"], row["rows"], row["features"], row["adapter"], row["version"]), []).append(row)
    def order(key):                                   # versions as numbers, so that 0.10.0 follows 0.9.0
        model, rows, features, adapter, version = key
        return model, rows, features, adapter, [int(part) if part.isdigit() else 0 for part in str(version or "").split(".")]

    out = []
    for (model, rows, features, adapter, version), runs in sorted(groups.items(), key=lambda item: order(item[0])):
        clean = [r for r in runs if not r["caveats"]]
        fits = [r["fit_s"] for r in runs if r["fit_s"] is not None]
        out.append({"model": model, "rows": rows, "features": features, "adapter": adapter, "version": version,
                    "runs": len(runs), "median_compare_s": statistics.median(r["compare_s"] for r in runs),
                    "min_compare_s": min(r["compare_s"] for r in runs),
                    "median_fit_s": statistics.median(fits) if fits else None,
                    "runs_without_caveat": len(clean),
                    "median_compare_s_without_caveat": statistics.median(r["compare_s"] for r in clean) if clean else None})
    return out


def halved_steps(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cox fits whose records carry the number of calls the implementation made to its own likelihood
    (pprof_py's, from version 0.14.0 of this repository): per dataset and model, the runs, the
    iterations reported, the calls made and the steps halved, each as the values seen over the runs."""
    def seen(values):
        distinct = sorted({v for v in values if v is not None})
        return None if not distinct else (distinct[0] if len(distinct) == 1 else ", ".join(str(v) for v in distinct))

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        fit = r.get("fit") or {}
        if r.get("status") == "ok" and (fit.get("extras") or {}).get("likelihood_calls") is not None:
            groups.setdefault((r["dataset"]["id"], r["model"]["label"], r["adapter"]["name"]), []).append(fit)
    return [{"dataset": dataset, "model": model, "adapter": adapter, "runs": len(fits),
             "iterations": seen(f.get("iterations") for f in fits), "converged": seen(f.get("converged") for f in fits),
             "likelihood_calls": seen(f["extras"]["likelihood_calls"] for f in fits),
             "halved_steps": seen(f["extras"].get("halved_steps") for f in fits)}
            for (dataset, model, adapter), fits in sorted(groups.items())]


def _site(name: str) -> str:
    """A stage's name without the folders of its call site."""
    return re.sub(r"\S*/", "", name or "")[:60]


def _sources(counts: Optional[Dict[str, Any]]) -> str:
    """The counts of executors taken just before a fit, when they do not agree: the scheduler's
    block managers, the executors the status store lists as active, and the scheduler's cores
    over the cores of one executor."""
    seen = {name: (counts or {}).get(name) for name in ("scheduler", "active", "by_cores")}
    if len({value for value in seen.values() if value is not None}) < 2:
        return ""
    return " (" + ", ".join(f"{name.replace('_', ' ')} {value}" for name, value in seen.items() if value is not None) + ")"


def where_time_went(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For every completed run of a Spark implementation: its phases, its first evaluation beside a
    later one where the adapter reports them, and the slowest single task among its heaviest stages,
    with that stage's name, number of tasks and median task. A stall shows as one task far above its
    stage's median; the number of tasks says which stage it was (the input's partitions, or the
    working set's). `other_s` is the part of the call outside the reported phases. Records made
    from version 0.12.0 on also say which executor ran that task, how long it had been attached
    when the task started, how much it read from storage, and how much of it was garbage
    collection or waiting for shuffle data. `input_warmed` says whether the plan had the stored
    table read once before the timed call, and `python_started` what starting the executors'
    Python workers took there (from version 0.16.0). `partitions` is the working set's number of
    partitions, `partitions_from` the rule that gave it (with strata or providers: "one per core",
    "by size", "rows per partition" or "given"; without strata how the rows were placed) and
    `cores_at_start` the cores the cluster had when the fit counted them (from version 0.20.0):
    with the default of decision D-21 a fit that starts on a smaller cluster has fewer partitions,
    and this is where it shows. `executors_held` is what the warm-up that asks for the plan's
    executors reported just before: how many it saw, of how many, after how long, and from
    version 0.21.0 how many were counted after the reads, just before the fit, where that differs;
    from version 0.22.0 that count is the lowest of three, which are listed where they disagree."""
    out = []
    for r in records:
        timing, spark = r.get("timing") or {}, r.get("spark") or {}
        if r.get("status") != "ok" or r["adapter"].get("kind") != "spark" or not timing:
            continue
        phases = timing.get("phases") or {}
        during = (r.get("environment") or {}).get("during") or {}
        extras = (r.get("fit") or {}).get("extras") or {}
        execution = extras.get("execution") or {}
        seconds = extras.get("evaluation_seconds") or []
        held = r.get("warmup") or {}
        stages = [s for s in spark.get("heaviest_stages") or [] if s.get("max_task_s") is not None]
        slow = max(stages, key=lambda s: s["max_task_s"], default=None)
        low, high = during.get("executors_min"), during.get("executors_max")
        seen = [str(c[1]) for c in during.get("changes") or []]
        task = spark.get("slowest_task") or {}
        warmed = r.get("warm_input") or {}
        row = {"model": r["model"]["label"], "features": r["dataset"]["spec"]["features"], "adapter": r["adapter"]["name"],
               "version": r.get("statcompute"),
               "partitions": extras.get("partitions") or execution.get("partitions"),
               "partitions_from": execution.get("partitions_from") or execution.get("rows_placed"),
               "cores_at_start": execution.get("cores"),
               "wall_s": round(timing["wall_s"], 1),
               "prepare_s": None if "prepare" not in phases else round(phases["prepare"], 1),
               "fit_s": None if "fit" not in phases else round(phases["fit"], 1),
               "other_s": round(timing["wall_s"] - sum(phases.values()), 1) if phases else None,
               "first_evaluation_s": round(seconds[0], 1) if seconds else None,
               "later_evaluations_s": " ".join(f"{s:.1f}" for s in seconds[1:]) or None,
               "slowest_task_s": None if slow is None else round(slow["max_task_s"], 1),
               "its_stage": None if slow is None else f"{slow.get('stage')}: {_site(slow.get('name'))}",
               "stage_tasks": None if slow is None else slow.get("tasks"),
               "stage_median_task_s": None if slow is None else round(slow["median_task_s"], 2),
               "on_executor": task.get("executor"),
               "executor_attached_s_before": None if task.get("executor_attached_s_before") is None
               else round(task["executor_attached_s_before"], 1),
               "task_gc_s": None if task.get("gc_s") is None else round(task["gc_s"], 1),
               "task_fetch_wait_s": None if task.get("fetch_wait_s") is None else round(task["fetch_wait_s"], 1),
               "task_locality": task.get("locality"),
               "spilled_gb": _gb(spark.get("disk_spilled_bytes")),
               "task_input_gb": _gb(task.get("input_bytes")),
               "executors": low if low == high else f"{low}-{high}",
               "executors_seen": ">".join(seen[:12]) + ("..." if len(seen) > 12 else ""),
               "executors_held": None if held.get("reached") is None
               else (f"{held['reached']} of {held.get('target')} in {held.get('seconds', 0):.0f} s"
                     + (f", asked {held['asked']} times" if (held.get("asked") or 1) > 1 else "")
                     + (f", {held['before_fit']} before the fit" if held.get("before_fit") not in (None, held["reached"]) else "")
                     + _sources(held.get("counts"))),
               "input_warmed": None if not warmed else (f"{warmed.get('mode')}: failed" if "error" in warmed
                                                        else f"{warmed.get('mode')} in {warmed.get('seconds', 0):.0f} s"),
               "python_started": None if warmed.get("python_seconds") is None else f"in {warmed['python_seconds']:.0f} s"}
        out.append(row)
    return out


def pass_costs(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The records of the probe of what a pass costs (`statcompute.lab.passes`), one row for every
    kind of pass in every record: the seconds of each pass as the driver saw it and their median;
    and from the tasks themselves, each the median over the passes: the median and the largest of
    a task's own seconds, the median of those it spent receiving and stacking its rows, and the
    median and the largest of those in the kernel. `around_tasks_s` is the median over the passes
    of a pass's seconds less its slowest task's: what the job costs outside its tasks.
    `log_likelihood_vs_kernel` sets the log-likelihood a kind's last pass added up beside that of
    the `kernel` kind: "the same bits", or their relative difference. It is how a cluster says
    whether two kernels add up the same sums."""
    middle = lambda values: None if not values else round(statistics.median(values), 2)   # noqa: E731
    out = []
    for r in records:
        extras = (r.get("fit") or {}).get("extras") or {}
        if r.get("status") != "ok" or not extras.get("ladder"):        # "ladder" is the probe's; the prototypes count their "passes"
            continue
        during = (r.get("environment") or {}).get("during") or {}
        low, high = during.get("executors_min"), during.get("executors_max")
        values = dict(extras.get("kernel_values") or {}, evaluation=extras.get("value"))
        mark = values.get("kernel")

        def beside(kind):
            if kind == "kernel" or mark is None or values.get(kind) is None:
                return None
            if values[kind] == mark:
                return "the same bits"
            return f"{abs(values[kind] - mark) / abs(mark):.1e}" if mark else None

        for kind in extras.get("pass_kinds") or list(extras["ladder"]):
            found = extras["ladder"][kind]
            seconds, slowest = found.get("seconds") or [], found.get("task_max_s") or []
            out.append({"model": r["model"]["label"], "features": r["dataset"]["spec"]["features"], "version": r.get("statcompute"),
                        "partitions": extras.get("partitions"), "executors": low if low == high else f"{low}-{high}",
                        "kind": kind, "passes": len(seconds), "tasks": middle(found.get("tasks")),
                        "seconds": " ".join(f"{s:.2f}" for s in seconds), "median_s": middle(seconds),
                        "task_s": middle(found.get("task_median_s")), "slowest_task_s": middle(slowest),
                        "receive_s": middle(found.get("receive_median_s")), "kernel_s": middle(found.get("kernel_median_s")),
                        "slowest_kernel_s": middle(found.get("kernel_max_s")),
                        "around_tasks_s": middle([s - t for s, t in zip(seconds, slowest)]) if len(slowest) == len(seconds) else None,
                        "log_likelihood_vs_kernel": beside(kind),
                        "cores": "-".join(str(c) for c in sorted(set(found.get("cores") or []))[::max(1, len(set(found.get("cores") or [])) - 1)])})
    return out


def to_markdown(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return "_no rows_\n"
    columns = columns or list(rows[0])
    cell = lambda v: "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)).replace("|", "/")  # noqa: E731
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"
