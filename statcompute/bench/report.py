"""A Markdown report from run records: what ran where, timings, and numerical agreement."""
from __future__ import annotations

from typing import Any, Dict, List

from ..validation import compare_records
from .results import summary, to_markdown

TIMING = ["model", "rows", "features", "design", "adapter", "role", "status", "wall_s", "fit_s", "iterations",
          "executors", "tasks", "effective_cores", "gc_s", "shuffle_gb", "worst_straggler_ratio",
          "driver_peak_rss_gb", "cluster_changed", "note"]
AGREEMENT = ["dataset", "model", "adapter", "reference", "like_for_like", "max_abs_coefficient_difference",
             "max_coefficient_difference_in_se", "max_rel_se_difference", "rel_log_likelihood_difference",
             "coefficients_over_T_COEF", "iterations"]


def render(records: List[Dict[str, Any]], reference: str = "pprof_py", title: str = "Benchmark report") -> str:
    if not records:
        return f"# {title}\n\n_no records_\n"
    static = next((r["environment"]["static"] for r in records if r.get("environment")), {})
    dbx = static.get("databricks") or {}
    parts = [f"# {title}\n", f"Records: {len(records)}; plans: {sorted({r['plan'] for r in records})}.\n",
             "## Environment\n",
             to_markdown([{"spark": static.get("spark_version"), "runtime": dbx.get("runtime"),
                           "worker_type": dbx.get("worker_type"), "driver_type": dbx.get("driver_type"),
                           "scaling": dbx.get("scaling_type"), "photon": dbx.get("photon"),
                           "driver_memory_gb": None if not static.get("driver_memory_bytes")
                           else round(static["driver_memory_bytes"] / 1e9, 1)}]),
             "## Runs\n",
             "Timings from runs with `cluster_changed` true are not comparable: the cluster changed size under them.\n",
             to_markdown(summary(records), TIMING),
             f"## Numerical agreement with {reference}\n",
             "Only rows with `like_for_like` true compare implementations that fit exactly the same model. "
             "Iterations are listed as [adapter, reference] and are not a criterion.\n",
             to_markdown(compare_records(records, reference), AGREEMENT)]
    notes = sorted({(r["adapter"]["name"], r["model"]["label"], note) for r in records
                    for note in ((r.get("support") or {}).get("notes") or [])})
    if notes:
        parts += ["## Notes from the adapters\n",
                  to_markdown([{"adapter": a, "model": m, "note": n} for a, m, n in notes])]
    return "\n".join(parts)
