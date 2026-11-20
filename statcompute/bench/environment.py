"""A record of where a benchmark ran: runtime, cluster shape and the settings that change results
or timings. Two snapshots, before and after a fit, show whether the cluster changed size under it."""
from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict, Optional

from .metrics import SparkMetrics

SPARK_KEYS = [
    "spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled", "spark.sql.adaptive.coalescePartitions.enabled",
    "spark.sql.optimizer.canChangeCachedPlanOutputPartitioning", "spark.sql.codegen.maxFields",
    "spark.sql.ansi.enabled", "spark.executor.memory", "spark.executor.cores", "spark.driver.memory",
    "spark.driver.maxResultSize", "spark.default.parallelism", "spark.master",
    "spark.memory.offHeap.enabled", "spark.memory.offHeap.size", "spark.sql.files.maxPartitionBytes",
    "spark.databricks.photon.enabled", "spark.locality.wait",
    "spark.sql.execution.arrow.maxRecordsPerBatch",       # fixes the batches, and so the sums, of an engine fit without strata
]
DATABRICKS_TAGS = {
    "runtime": "spark.databricks.clusterUsageTags.sparkVersion",
    "cluster_id": "spark.databricks.clusterUsageTags.clusterId",
    "worker_type": "spark.databricks.clusterUsageTags.clusterNodeType",
    "driver_type": "spark.databricks.clusterUsageTags.driverNodeType",
    "workers_configured": "spark.databricks.clusterUsageTags.clusterWorkers",
    "min_workers": "spark.databricks.clusterUsageTags.clusterMinWorkers",
    "max_workers": "spark.databricks.clusterUsageTags.clusterMaxWorkers",
    "scaling_type": "spark.databricks.clusterUsageTags.clusterScalingType",
    "photon": "spark.databricks.photon.enabled",
}


def conf(spark, key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return spark.conf.get(key)
    except Exception:
        try:
            return spark.sparkContext.getConf().get(key, default)
        except Exception:
            return default


def executors(spark) -> Dict[str, Any]:
    """Active executors and their cores, from the UI's REST API or, failing that, from the scheduler."""
    found = SparkMetrics(spark).executors()
    if found is not None:
        return found
    try:
        count = int(spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()) - 1
    except Exception:
        count = None
    return {"executors": count, "cores": None, "max_memory_bytes": None, "peak_jvm_heap_bytes_max": None}


def static(spark) -> Dict[str, Any]:
    """What does not change during a session."""
    info: Dict[str, Any] = {
        "spark_version": spark.version,
        "application_id": spark.sparkContext.applicationId,
        "python": platform.python_version(),
        "driver_cores": os.cpu_count(),
        "databricks": {name: conf(spark, key) for name, key in DATABRICKS_TAGS.items()},
        "spark_conf": {key: conf(spark, key) for key in SPARK_KEYS},
    }
    try:
        import psutil
        info["driver_memory_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        info["driver_memory_bytes"] = None
    for module in ("numpy", "pandas", "pyarrow", "scipy"):
        try:
            info[module] = __import__(module).__version__
        except Exception:
            info[module] = None
    try:                                              # the BLAS behind NumPy on the driver; the executors' is not recorded yet (OI-29)
        from threadpoolctl import threadpool_info
        info["blas"] = [{k: lib.get(k) for k in ("user_api", "internal_api", "version", "threading_layer", "architecture", "num_threads")}
                        for lib in threadpool_info()]
    except Exception:
        info["blas"] = None
    return info


def moving(spark) -> Dict[str, Any]:
    """What can change under a run: executors, cores and the default parallelism."""
    out = executors(spark)
    out["default_parallelism"] = spark.sparkContext.defaultParallelism
    return out


def attached(spark):
    """(executors, cores) attached right now, from the scheduler: cheap enough to sample."""
    sc = spark.sparkContext
    try:
        executors = int(sc._jsc.sc().getExecutorMemoryStatus().size()) - 1
    except Exception:
        executors = None
    return executors, sc.defaultParallelism


def hold_at(spark, executors: int, timeout_s: float = 900.0, task_seconds: float = 15.0,
            allow_local: bool = False, log=print) -> Dict[str, Any]:
    """Ask an autoscaling cluster for `executors` workers and wait until they are attached.

    The autoscaler adds workers when tasks are waiting, so this submits rounds of sleeping tasks,
    many more than there are cores, until the count is reached or the time is up. It cannot stop the
    autoscaler from removing workers later; the runner samples the count during every fit for that.
    """
    sc = spark.sparkContext
    out = {"target": int(executors), "reached": None, "seconds": 0.0, "rounds": 0, "seen": []}
    if sc.master.startswith("local") and not allow_local:
        return dict(out, skipped="local session")
    started = time.perf_counter()
    current = attached(spark)[0]
    out["seen"].append([0.0, current])
    while (current is None or current < executors) and time.perf_counter() - started < timeout_s:
        tasks = max(8 * int(executors), 2 * sc.defaultParallelism)
        sc.setJobGroup("statcompute-warmup", f"asking the autoscaler for {executors} executors")
        sc.parallelize(range(tasks), tasks).foreach(lambda _, s=task_seconds: __import__("time").sleep(s))
        out["rounds"] += 1
        current = attached(spark)[0]
        out["seen"].append([round(time.perf_counter() - started, 1), current])     # executors after each round
        if log is not None:
            log(f"warm-up: {current} of {executors} executors after {time.perf_counter() - started:.0f} s")
    return dict(out, reached=current, seconds=time.perf_counter() - started)
