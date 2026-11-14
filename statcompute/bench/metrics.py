"""Measurements taken around one fit.

SparkMetrics     reads the Spark UI's REST API for the jobs of one job group: tasks, executor run
                 time and CPU time, GC time, shuffle and spill bytes, and the spread of task times
ProcessSampler   samples the driver process: peak resident memory and CPU time

Both degrade to "unavailable" rather than fail: a benchmark record without Spark metrics is still a
valid timing.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional


class SparkMetrics:
    """Two sources for the same numbers: the UI's REST API and, where that is not reachable from
    the driver, the status store behind it, read through py4j. Records say which was used."""

    STAGE_FIELDS = ("numCompleteTasks", "executorRunTime", "executorCpuTime", "jvmGcTime", "shuffleWriteBytes",
                    "shuffleReadBytes", "memoryBytesSpilled", "diskBytesSpilled", "resultSize", "peakExecutionMemory")

    def __init__(self, spark, timeout: float = 15.0):
        sc = spark.sparkContext
        self._sc = sc
        self.base = None
        url = getattr(sc, "uiWebUrl", None)
        if url:
            self.base = f"{url.rstrip('/')}/api/v1/applications/{sc.applicationId}"
        self.timeout = timeout

    # ---- REST ---------------------------------------------------------------------------------
    def _get(self, path: str) -> Optional[Any]:
        if not self.base:
            return None
        try:
            with urllib.request.urlopen(f"{self.base}/{path}", timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

    # ---- status store, through py4j ---------------------------------------------------------------
    def _store(self):
        return self._sc._jsc.sc().statusStore()

    def _store_stages(self, group: str):
        """(number of jobs in the group, their completed stages as dictionaries), or None."""
        try:
            store, gateway = self._store(), self._sc._gateway
            jobs = store.jobsList(None)
            ids, count = set(), 0
            for i in range(jobs.size()):
                job = jobs.apply(i)
                label = job.jobGroup()
                if label.isDefined() and str(label.get()) == group:
                    count += 1
                    stage_ids = job.stageIds()
                    ids.update(int(stage_ids.apply(k)) for k in range(stage_ids.size()))
            stages = store.stageList(None, False, False, gateway.new_array(gateway.jvm.double, 0), None)
            done = []
            for i in range(stages.size()):
                stage = stages.apply(i)
                if int(stage.stageId()) in ids and str(stage.status().toString()) == "COMPLETE":
                    row = {"stageId": int(stage.stageId()), "attemptId": int(stage.attemptId())}
                    row.update({name: int(getattr(stage, name)()) for name in self.STAGE_FIELDS})
                    done.append(row)
            return count, done
        except Exception:
            return None

    def _store_task_times(self, stage_id: int, attempt_id: int) -> Optional[List[float]]:
        try:
            gateway = self._sc._gateway
            quantiles = gateway.new_array(gateway.jvm.double, 3)
            quantiles[0], quantiles[1], quantiles[2] = 0.5, 0.95, 1.0
            summary = self._store().taskSummary(stage_id, attempt_id, quantiles)
            if not summary.isDefined():
                return None
            run = summary.get().executorRunTime()
            return [float(run.apply(i)) for i in range(3)]
        except Exception:
            return None

    # ---- public -------------------------------------------------------------------------------
    def executors(self) -> Optional[Dict[str, Any]]:
        rows = self._get("executors")
        if rows is None:
            try:
                listed = self._store().executorList(True)
                rows = []
                for i in range(listed.size()):
                    e = listed.apply(i)
                    peak = e.peakMemoryMetrics()
                    rows.append({"id": str(e.id()), "isActive": bool(e.isActive()), "totalCores": int(e.totalCores()),
                                 "maxMemory": int(e.maxMemory()),
                                 "peakMemoryMetrics": {"JVMHeapMemory": int(peak.get().getMetricValue("JVMHeapMemory"))}
                                 if peak.isDefined() else {}})
            except Exception:
                return None
        workers = [r for r in rows if r.get("id") != "driver" and r.get("isActive", True)]
        peaks = [r.get("peakMemoryMetrics", {}).get("JVMHeapMemory") for r in workers]
        peaks = [v for v in peaks if v is not None]
        return {
            "executors": len(workers),
            "cores": sum(int(r.get("totalCores", 0)) for r in (workers or rows)),
            "max_memory_bytes": sum(int(r.get("maxMemory", 0)) for r in workers),
            "peak_jvm_heap_bytes_max": max(peaks) if peaks else None,
        }

    def collect(self, group: str) -> Optional[Dict[str, Any]]:
        """Totals over the completed stages of the jobs in `group`; None if neither source answers."""
        jobs, stages, source = self._get("jobs"), self._get("stages"), "rest"
        if jobs is not None and stages is not None:
            mine = [j for j in jobs if j.get("jobGroup") == group]
            ids = {s for j in mine for s in j.get("stageIds", [])}
            count, done = len(mine), [s for s in stages if s.get("stageId") in ids and s.get("status") == "COMPLETE"]
        else:
            found, source = self._store_stages(group), "status_store"
            if found is None:
                return None
            count, done = found
        total = lambda key: sum(float(s.get(key, 0) or 0) for s in done)  # noqa: E731
        out: Dict[str, Any] = {
            "source": source,
            "jobs": count,
            "stages": len(done),
            "tasks": int(total("numCompleteTasks")),
            "executor_run_s": total("executorRunTime") / 1e3,
            "executor_cpu_s": total("executorCpuTime") / 1e9,
            "gc_s": total("jvmGcTime") / 1e3,
            "shuffle_write_bytes": int(total("shuffleWriteBytes")),
            "shuffle_read_bytes": int(total("shuffleReadBytes")),
            "memory_spilled_bytes": int(total("memoryBytesSpilled")),
            "disk_spilled_bytes": int(total("diskBytesSpilled")),
            "result_bytes": int(total("resultSize")),
            "peak_execution_memory_bytes": int(max((s.get("peakExecutionMemory", 0) or 0) for s in done)) if done else 0,
        }
        heavy = sorted(done, key=lambda s: -(s.get("executorRunTime", 0) or 0))[:10]
        spreads: List[Dict[str, Any]] = []
        for s in heavy:
            attempt = s.get("attemptId", 0)
            if source == "rest":
                run = (self._get(f"stages/{s['stageId']}/{attempt}/taskSummary?quantiles=0.5,0.95,1.0") or {}).get("executorRunTime")
            else:
                run = self._store_task_times(s["stageId"], attempt)
            if run and len(run) == 3 and run[0] > 0:
                spreads.append({"stage": s["stageId"], "tasks": s.get("numCompleteTasks"),
                                "median_task_s": run[0] / 1e3, "p95_task_s": run[1] / 1e3,
                                "max_task_s": run[2] / 1e3, "straggler_ratio": run[2] / run[0]})
        out["heaviest_stages"] = spreads
        out["worst_straggler_ratio"] = max((s["straggler_ratio"] for s in spreads), default=None)
        return out


class ProcessSampler:
    """Peak resident memory and CPU time between enter and exit: of this process, and of this
    process with its children (an Rscript run, or the JVM of a local Spark session)."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_rss = 0
        self.peak_tree = 0
        self.cpu_s = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
        except Exception:
            self._process = None

    def _cpu(self) -> Optional[float]:
        if self._process is None:
            return None
        t = self._process.cpu_times()
        return t.user + t.system + getattr(t, "children_user", 0.0) + getattr(t, "children_system", 0.0)

    def _loop(self):
        while not self._stop.is_set():
            try:
                own = self._process.memory_info().rss
                self.peak_rss = max(self.peak_rss, own)
                tree = own
                for child in self._process.children(recursive=True):
                    try:
                        tree += child.memory_info().rss
                    except Exception:
                        pass
                self.peak_tree = max(self.peak_tree, tree)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._start_cpu = self._cpu()
        if self._process is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        end = self._cpu()
        if end is not None and self._start_cpu is not None:
            self.cpu_s = end - self._start_cpu
        return False

    def result(self, wall_s: float) -> Optional[Dict[str, Any]]:
        if self._process is None:
            return None
        cores = os.cpu_count() or 1
        return {"peak_rss_bytes": int(self.peak_rss), "peak_rss_with_children_bytes": int(self.peak_tree),
                "cpu_s": self.cpu_s, "cores": cores,
                "cpu_utilization": None if not self.cpu_s or wall_s <= 0 else self.cpu_s / (wall_s * cores)}


class ExecutorSampler:
    """Executors and cores attached between enter and exit, sampled on a thread. On an autoscaling
    cluster this is what says how much of the cluster a fit actually had. `probe` returns
    (executors, cores)."""

    def __init__(self, probe, interval: float = 5.0):
        self._probe, self.interval = probe, interval
        self._samples: List[Any] = []
        self.changes: List[Any] = []
        self._stop = threading.Event()
        self._thread = None

    def _take(self):
        try:
            executors, cores = self._probe()
        except Exception:
            return
        at = time.perf_counter() - self._started
        self._samples.append((at, executors, cores))
        if (not self.changes or self.changes[-1][1:] != [executors, cores]) and len(self.changes) < 200:
            self.changes.append([round(at, 1), executors, cores])

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._take()

    def __enter__(self):
        self._started = time.perf_counter()
        self._take()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._take()
        return False

    def result(self) -> Optional[Dict[str, Any]]:
        if not self._samples:
            return None
        executors = [s[1] for s in self._samples if s[1] is not None]
        cores = [s[2] for s in self._samples if s[2] is not None]
        span = self._samples[-1][0] - self._samples[0][0]
        average = None
        if cores:
            pairs = [(a, b) for a, b in zip(self._samples, self._samples[1:]) if a[2] is not None]
            average = (sum(a[2] * (b[0] - a[0]) for a, b in pairs) / span) if span > 0 and pairs else float(cores[0])
        return {
            "samples": len(self._samples),
            "executors_min": min(executors) if executors else None,
            "executors_max": max(executors) if executors else None,
            "cores_min": min(cores) if cores else None,
            "cores_max": max(cores) if cores else None,
            "cores_time_average": average,
            "changes": self.changes,           # [seconds into the fit, executors, cores] at each change
        }
