"""Run a plan: every dataset, every model, every adapter, and one JSON record per run.

A record is built from scratch for each run and written whether the fit succeeded or not. Nothing
is carried from one run to the next, so a failed fit can never be recorded with the values of an
earlier one.
"""
from __future__ import annotations

import copy
import datetime
import gc
import hashlib
import json
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from .. import __version__, adapters, references
from ..testkit import DatasetSpec, ModelSpec, materialize
from . import environment
from .metrics import ExecutorSampler, ProcessSampler, SparkMetrics

SCHEMA_VERSION = 1
PLANS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plans")


def load_plan(name_or_path: str) -> Dict[str, Any]:
    """A plan by name (statcompute/bench/plans/<name>.json) or by path."""
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(PLANS, f"{name_or_path}.json")
    with open(path) as f:
        plan = json.load(f)
    for key in ("name", "datasets", "models", "adapters"):
        if key not in plan:
            raise ValueError(f"the plan {path} lacks '{key}'")
    return plan


def local_path(path: str) -> str:
    """Results are written with ordinary file calls: on Databricks use /Volumes/..., /Workspace/...
    or a dbfs:/ path, which is reached through /dbfs."""
    return "/dbfs/" + path[len("dbfs:/"):].lstrip("/") if path.startswith("dbfs:/") else path


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _truth(spec: DatasetSpec, fit) -> Optional[Dict[str, Any]]:
    if not fit.standard_errors:
        return None
    z = [(b - t) / s for b, t, s in zip(fit.coefficients, spec.true_beta(), fit.standard_errors) if s and s > 0]
    worst = max((abs(v) for v in z), default=None)
    return {"max_abs_z": worst, "recovered": None if worst is None else bool(worst <= 4.5)}


def run_one(spark, plan_name: str, adapter_name: str, dataset, model: ModelSpec, repeat: int,
            options: Dict[str, Any], checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """One fit, one record. `checkpoint` is called with the record, status "running", just before
    the fit starts: if the process dies during the fit, that is what remains."""
    spec = dataset.spec
    run_id = uuid.uuid4().hex[:16]
    record: Dict[str, Any] = {
        "schema": SCHEMA_VERSION, "run_id": run_id, "plan": plan_name, "repeat": repeat,
        "statcompute": __version__, "started_at": _now(), "finished_at": None,
        "adapter": {"name": adapter_name}, "dataset": {"id": dataset.id, "spec": spec.to_dict(),
                                                       "rows": dataset.rows, "events": dataset.manifest.get("events"),
                                                       "checksum": dataset.manifest.get("checksum")},
        "model": dict(model.to_dict(), label=model.label()),
        "support": None, "status": None, "error": None, "caveats": [], "memory_guard": None, "warmup": None,
        "warm_input": None,
        "environment": None, "timing": None, "spark": None, "process": None, "fit": None, "truth": None,
    }

    def done(status: str, **more) -> Dict[str, Any]:
        record.update(status=status, finished_at=_now(), **more)
        return record

    adapter = adapters.get(adapter_name)
    adapter.options = dict(options)
    record["adapter"] = adapter.describe(spark)
    pinned = (references.lock().get("references") or {}).get(getattr(adapter, "pin", None) or adapter_name)
    if pinned:
        found = record["adapter"].get("commit")
        record["adapter"]["pinned"] = {k: pinned.get(k) for k in ("commit", "version")}
        record["adapter"]["matches_pin"] = None if not (found and pinned.get("commit")) else (
            pinned["commit"].startswith(found) or found.startswith(pinned["commit"]))
    available = adapter.available(spark)
    if not available.runnable:
        return done("unavailable", support=asdict(available))
    support = adapter.support(model, spec)
    record["support"] = asdict(support)
    if not support.runnable:
        return done("unsupported")
    need = adapter.driver_bytes(model, spec)
    info = environment.static(spark)
    # The driver node also holds the Spark driver's JVM, so a single-machine fit gets a fraction of it.
    budget = options.get("driver_memory_bytes") or (
        (info.get("driver_memory_bytes") or 0) * float(options.get("driver_memory_fraction", 0.4)))
    if need:
        record["memory_guard"] = {"need_bytes": int(need), "budget_bytes": int(budget or 0)}
    if need and budget and need > budget:
        return done("skipped_memory", error={"type": "MemoryGuard",
                                             "message": f"needs about {need} bytes on the driver; the budget is {int(budget)}"})

    sc = spark.sparkContext
    group = f"statcompute-{run_id}"
    metrics = SparkMetrics(spark)
    held = bool(options.get("hold_executors")) and adapter.kind == "spark"
    asked = 0
    while True:
        if held:
            # an autoscaling cluster: ask for its full size before the clock starts
            warm = environment.hold_at(spark, int(options["hold_executors"]), float(options.get("hold_timeout_s", 900)))
            earlier = record["warmup"] or {}
            record["warmup"] = dict(warm, seconds=warm.get("seconds", 0.0) + earlier.get("seconds", 0.0),
                                    rounds=warm.get("rounds", 0) + earlier.get("rounds", 0))
            asked += 1
        if options.get("warm_input") and adapter.kind == "spark":
            # every executor reads the stored table once before the clock starts (OI-31); off unless the plan asks
            try:
                record["warm_input"] = environment.warm_input(spark, dataset, str(options["warm_input"]))
            except Exception as failure:
                record["warm_input"] = {"mode": str(options["warm_input"]), "error": f"{type(failure).__name__}: {failure}"[:300]}
        if not held or record["warmup"].get("skipped"):
            break
        # The cluster can lose executors in the seconds after the warm-up saw them all (7 of the 64 runs of
        # 2026-11-22 then ran on six of ten). So they are counted again here, after the reads and just before
        # the fit, and with the option `hold_again` the harness asks again, that many times at most
        # (decision D-22). The loss comes about a second after the reads end, and a count taken at once has
        # missed it: `hold_settle_s` waits that long first, and the count is the lowest of those the harness has.
        if float(options.get("hold_settle_s", 0)) > 0:
            time.sleep(float(options["hold_settle_s"]))
        counts = environment.counted(spark)
        record["warmup"].update(before_fit=counts["executors"], counts=counts, asked=asked)
        now = counts["executors"]
        if now is None or now >= int(options["hold_executors"]) or asked > int(options.get("hold_again", 0)):
            break
    if adapter.kind == "spark":
        try:
            record["dataset"]["input_partitions"] = int(dataset.frame(spark).rdd.getNumPartitions())
        except Exception:
            record["dataset"]["input_partitions"] = None
    before = environment.moving(spark)
    timed_out = threading.Event()
    timer = None
    if options.get("timeout_s") and adapter.kind == "spark":
        def cancel():
            timed_out.set()
            sc.cancelJobGroup(group)
        timer = threading.Timer(float(options["timeout_s"]), cancel)
        timer.daemon = True
        timer.start()
    sc.setJobGroup(group, f"{plan_name}: {adapter_name} on {dataset.id} ({model.label()})")
    record["environment"] = {"static": info, "before": before, "after": None, "cluster_changed": None}
    if checkpoint is not None:
        checkpoint(dict(record, status="running"))
    fit, error = None, None
    started = time.perf_counter()
    watch = ExecutorSampler(lambda: environment.attached(spark), float(options.get("executor_sample_s", 5)))
    with ProcessSampler() as sampler, watch:
        try:
            fit = adapter.fit(spark, dataset, model)
        except BaseException as failure:  # a failed fit is a result; record it and go on
            if isinstance(failure, KeyboardInterrupt):
                raise
            error = {"type": type(failure).__name__, "message": str(failure)[:4000],
                     "trace": traceback.format_exc()[-4000:]}
    wall = time.perf_counter() - started
    if timer is not None:
        timer.cancel()
    try:
        sc.setLocalProperty("spark.jobGroup.id", None)
    except Exception:
        pass
    after = environment.moving(spark)
    time.sleep(0.5)                      # let the listener bus deliver the last stage's completion
    spark_metrics = metrics.collect(group)
    during = watch.result() or {}
    moved = (before.get("executors") != after.get("executors")
             or during.get("executors_min") != during.get("executors_max"))
    if spark_metrics is not None and wall > 0:
        cores = after.get("cores") or before.get("cores")
        if during.get("cores_time_average"):
            spark_metrics["share_of_attached_cores"] = (
                spark_metrics["executor_run_s"] / wall / during["cores_time_average"])
        spark_metrics["effective_cores_by_run_time"] = spark_metrics["executor_run_s"] / wall
        spark_metrics["effective_cores_by_cpu_time"] = spark_metrics["executor_cpu_s"] / wall
        spark_metrics["cpu_utilization"] = None if not cores else spark_metrics["executor_cpu_s"] / (wall * cores)
    record.update(
        environment={"static": info, "before": before, "after": after, "during": during, "cluster_changed": moved},
        timing={"wall_s": wall, "phases": {} if fit is None else fit.phases},
        spark=spark_metrics, process=sampler.result(wall),
    )
    if options.get("clear_cache", True):
        try:
            spark.catalog.clearCache()
        except Exception:
            pass
    gc.collect()
    try:                                  # hand freed memory back to the system, so the next run's peak is its own
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    caveats = [] if fit is None else list(fit.caveats)
    if moved:
        low, high = during.get("executors_min"), during.get("executors_max")
        caveats.append("the cluster changed size during the fit" if low == high or low is None
                       else f"executors ranged from {low} to {high} during the fit")
    warm = record.get("warmup") or {}
    at_start = warm.get("before_fit") if warm.get("before_fit") is not None else warm.get("reached")
    if at_start is not None and at_start < warm["target"]:
        caveats.append(f"the cluster had {at_start} of {warm['target']} executors when the fit started")
    short = environment.short_of(warm, during)
    if short:
        caveats.append(short)
    if record["adapter"].get("matches_pin") is False:
        caveats.append("the installed build is not the commit pinned in REFERENCES.lock")
    record["caveats"] = caveats
    if error is not None:
        return done("timeout" if timed_out.is_set() else "failed", error=error)
    record["fit"] = {k: v for k, v in asdict(fit).items() if k not in ("phases", "caveats")}
    record["truth"] = _truth(spec, fit)
    return done("ok")


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_USER_PATH = re.compile(r"/(Users|Repos|home)/[^/\s'\"\\]+")


def scrub(record: Dict[str, Any]) -> str:
    """The record as JSON text with nothing that names a person or a workspace: the cluster
    identifier becomes a short hash, and e-mail addresses and user folders in any message or trace
    are replaced. Records are meant to be committed to a public repository."""
    clean = copy.deepcopy(record)
    dbx = ((clean.get("environment") or {}).get("static") or {}).get("databricks") or {}
    if dbx.get("cluster_id"):
        dbx["cluster_id"] = "sha256:" + hashlib.sha256(str(dbx["cluster_id"]).encode()).hexdigest()[:12]
    clean["scrubbed"] = True
    text = json.dumps(clean, indent=1, default=str)
    return _USER_PATH.sub(r"/\1/<user>", _EMAIL.sub("<user>", text))


def write_record(results_root: str, record: Dict[str, Any]) -> str:
    """Write the record, replacing an earlier version of the same run (its "running" checkpoint)."""
    folder = os.path.join(local_path(results_root), "runs", record["plan"])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{record['started_at'][:19].replace(':', '')}-{record['adapter']['name']}-{record['run_id']}.json")
    with open(path, "w") as f:           # one sequential write: works on Volumes and workspace files
        f.write(scrub(record))
    return path


def run_plan(spark, plan: Dict[str, Any], data_root: str, results_root: str,
             only_adapters: Optional[Iterable[str]] = None, max_rows: Optional[int] = None,
             families: Optional[Iterable[str]] = None,
             on_record: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
    """Materialize each dataset of the plan if needed, then run every model on every adapter.
    `only_adapters`, `max_rows` and `families` narrow a plan without editing it."""
    options = dict(plan.get("options", {}))
    chosen = [a for a in plan["adapters"] if only_adapters is None or a in set(only_adapters)]
    # Two passes unless the plan says otherwise: every Spark implementation on every dataset first,
    # then the single-machine ones. A single-machine fit leaves the executors idle for minutes, and
    # an autoscaling cluster shrinks under it; run in between, it left the next Spark fit short.
    kinds = {name: adapters.get(name).kind for name in chosen}
    passes = ([[a for a in chosen if kinds[a] == "spark"], [a for a in chosen if kinds[a] != "spark"]]
              if options.get("spark_first", True) else [chosen])
    work = []
    for group in passes:
        for entry in plan["datasets"]:
            spec = DatasetSpec.from_dict(entry)
            if (max_rows and spec.rows > max_rows) or (families and spec.family not in set(families)):
                continue
            for model_entry in plan["models"].get(spec.family, []):
                work.append((spec, ModelSpec.from_dict(model_entry), group))
    records, stored = [], {}
    for spec, model, group in work:
        if group:
            if spec.dataset_id() not in stored:
                stored[spec.dataset_id()] = materialize(spark, spec, data_root)
            dataset = stored[spec.dataset_id()]
            # By default an adapter's repeats follow one another. With the option `interleave` the adapters
            # take turns, repeat by repeat, so that what the cluster does over a session (new executors after a
            # change of dataset, a loss of executors) falls on all of them and not on the one that happens to go first.
            repeats = range(int(plan.get("repeats", 1)))
            turns = ([(a, k) for k in repeats for a in group] if options.get("interleave")
                     else [(a, k) for a in group for k in repeats])
            stopped = set()
            for adapter_name, repeat in turns:
                if adapter_name in stopped:
                    continue  # a repeat of a run that did not complete adds nothing
                record = run_one(spark, plan["name"], adapter_name, dataset, model, repeat, options,
                                 checkpoint=lambda partial: write_record(results_root, partial))
                record["file"] = write_record(results_root, record)
                records.append(record)
                if on_record:
                    on_record(record)
                if record["status"] != "ok":
                    stopped.add(adapter_name)
    return records
