"""What produced a fit's digits (decision D-16): the versions, the layout and the settings.

`execution_record` is taken on the driver when a fit is made and kept with the model it returns, so
that the audit trail does not depend on the benchmark harness. `executor_libraries` asks the
executors, where the kernels run, for their Python, NumPy and BLAS: one short job, which also
starts their Python workers. The harness runs it before a timed fit (decision D-20); a fit made
directly may call it afterwards and put the answer in `model.execution["executors"]`.
"""
from __future__ import annotations

import sys
import time as clock
from typing import Any, Dict, List, Optional


def _libraries() -> Dict[str, Any]:
    """This process's Python and numerical libraries. Importing them is the point as much as their
    versions: on an executor that has run no Python task yet it takes seconds."""
    out: Dict[str, Any] = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "pyarrow", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    try:
        from threadpoolctl import threadpool_info

        out["blas"] = sorted({f"{lib.get('internal_api')} {lib.get('version')} {lib.get('threading_layer') or ''}".strip()
                              for lib in threadpool_info() if lib.get("user_api") == "blas"})
    except Exception:
        out["blas"] = None
    return out


def execution_record(spark, **layout: Any) -> Dict[str, Any]:
    """The driver's side of the record: package and Spark versions, the libraries here, the Arrow
    batch size (which fixes the batches, and with them the sums, of a fit without strata), the
    layout the caller names and how the package's code was sent to the executors (`code_sent`: by
    value, so that they ran the driver's code, or as an archive). `executors` is filled by
    `executor_libraries` when someone asks."""
    from .. import __version__

    try:
        batch = int(spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch"))
    except Exception:
        batch = None
    return {"statcompute": __version__, "spark": spark.version, "driver": _libraries(), "executors": None,
            "arrow_batch_rows": batch, **layout}


def executor_libraries(spark, tasks: Optional[int] = None, hold_seconds: float = 0.2) -> List[Dict[str, Any]]:
    """The distinct sets of libraries on the executors, each with the number of tasks that reported
    it. More tasks than there are cores are sent and each holds its core for a moment, so that every
    executor takes some; a task imports NumPy, pandas, pyarrow and SciPy, which starts the Python
    worker that the first pass of a fit would otherwise have to wait for."""
    sc = spark.sparkContext
    tasks = int(tasks or max(8, 2 * sc.defaultParallelism))

    def report(_):                                           # defined here: it travels by value, no package needed there
        import json
        import sys
        import time

        out = {"python": sys.version.split()[0]}
        for name in ("numpy", "pandas", "pyarrow", "scipy"):
            try:
                out[name] = __import__(name).__version__
            except Exception:
                out[name] = None
        try:
            from threadpoolctl import threadpool_info

            out["blas"] = sorted({f"{lib.get('internal_api')} {lib.get('version')} {lib.get('threading_layer') or ''}".strip()
                                  for lib in threadpool_info() if lib.get("user_api") == "blas"})
        except Exception:
            out["blas"] = None
        time.sleep(hold_seconds)
        return json.dumps(out, sort_keys=True)

    import json

    counts: Dict[str, int] = {}
    for text in sc.parallelize(range(tasks), tasks).map(report).collect():
        counts[text] = counts.get(text, 0) + 1
    return [dict(json.loads(text), tasks=count) for text, count in sorted(counts.items())]


def timed_executor_libraries(spark) -> Dict[str, Any]:
    """`executor_libraries` with the seconds it took: the cost of starting the Python workers."""
    started = clock.perf_counter()
    found = executor_libraries(spark)
    return {"seconds": clock.perf_counter() - started, "libraries": found}
