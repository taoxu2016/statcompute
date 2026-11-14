"""A record of where a benchmark ran: runtime, cluster shape and the settings that change results
or timings. Two snapshots, before and after a fit, show whether the cluster changed size under it."""
from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional

from .metrics import SparkMetrics

SPARK_KEYS = [
    "spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled", "spark.sql.adaptive.coalescePartitions.enabled",
    "spark.sql.optimizer.canChangeCachedPlanOutputPartitioning", "spark.sql.codegen.maxFields",
    "spark.sql.ansi.enabled", "spark.executor.memory", "spark.executor.cores", "spark.driver.memory",
    "spark.driver.maxResultSize", "spark.default.parallelism", "spark.master",
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
    return info


def moving(spark) -> Dict[str, Any]:
    """What can change under a run: executors, cores and the default parallelism."""
    out = executors(spark)
    out["default_parallelism"] = spark.sparkContext.defaultParallelism
    return out
