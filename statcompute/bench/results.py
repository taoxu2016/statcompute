"""Read run records back and flatten them for tables."""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional

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


def summary(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One flat row per run: what ran, whether it completed, and what it took."""
    rows = []
    for r in records:
        spec, timing = r["dataset"]["spec"], r.get("timing") or {}
        spark, process, fit = r.get("spark") or {}, r.get("process") or {}, r.get("fit") or {}
        env = r.get("environment") or {}
        after = env.get("after") or {}
        rows.append({
            "model": r["model"]["label"], "rows": spec["rows"], "providers": spec["providers"],
            "features": spec["features"], "design": spec["design"],
            "adapter": r["adapter"]["name"], "role": r["adapter"].get("role"), "repeat": r.get("repeat"),
            "status": r["status"], "wall_s": None if not timing else round(timing["wall_s"], 2),
            "fit_s": None if "fit" not in (timing.get("phases") or {}) else round(timing["phases"]["fit"], 2),
            "iterations": fit.get("iterations"), "converged": fit.get("converged"),
            "max_abs_z": None if not r.get("truth") or r["truth"]["max_abs_z"] is None else round(r["truth"]["max_abs_z"], 2),
            "executors": after.get("executors"), "cores": after.get("cores"),
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
            "note": "" if not r.get("error") else r["error"]["message"][:160],
        })
    return rows


def to_markdown(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return "_no rows_\n"
    columns = columns or list(rows[0])
    cell = lambda v: "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)).replace("|", "/")  # noqa: E731
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"
