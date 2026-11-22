"""Compare two fits of the same model on the same dataset.

Differences are reported three ways: in absolute terms, in units of the reference's standard
errors (the unit that matters statistically), and as a multiple of the tolerance classes of
testkit/tolerances.py (at most 1 passes). Iteration counts are reported and never compared: two
correct implementations may take different paths.

A comparison is like-for-like only when both implementations said they fit the requested model
exactly. Anything else is listed with the reasons and must not be read as an error of either.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..testkit.tolerances import excess


def compare_fits(fit: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    names = reference["feature_names"]
    position = {name: i for i, name in enumerate(fit["feature_names"])}
    if set(position) != set(names):
        raise ValueError("the two fits do not have the same features")
    b = [fit["coefficients"][position[n]] for n in names]
    b0 = reference["coefficients"]
    out: Dict[str, Any] = {
        "features": len(names),
        "max_abs_coefficient_difference": max(abs(x - y) for x, y in zip(b, b0)),
        "coefficients_over_T_COEF": excess(b, b0, "T_COEF"),
        "iterations": [fit.get("iterations"), reference.get("iterations")],
    }
    se0 = reference.get("standard_errors")
    if se0:
        out["max_coefficient_difference_in_se"] = max(abs(x - y) / s for x, y, s in zip(b, b0, se0) if s > 0)
    if fit.get("standard_errors") and se0:
        se = [fit["standard_errors"][position[n]] for n in names]
        out["max_rel_se_difference"] = max(abs(x - y) / y for x, y in zip(se, se0) if y > 0)
        out["standard_errors_over_T_VAR"] = excess(se, se0, "T_VAR")
    l, l0 = fit.get("log_likelihood"), reference.get("log_likelihood")
    if l is not None and l0 is not None:
        out["rel_log_likelihood_difference"] = abs(l - l0) / abs(l0) if l0 else abs(l - l0)
        out["log_likelihood_over_T_LOGLIK"] = excess([l], [l0], "T_LOGLIK")
    return out


def _version(record: Dict[str, Any]) -> Optional[str]:
    """What tells two builds of an implementation apart: for code of this repository its version
    when the run was made, for a reference package its own commit or version."""
    adapter = record["adapter"]
    if adapter.get("role") in ("engine", "prototype"):
        return record.get("statcompute")
    return adapter.get("commit") or adapter.get("version")


def compare_records(records: Iterable[Dict[str, Any]], reference: str = "pprof_py") -> List[Dict[str, Any]]:
    """Every completed run against the reference adapter's first completed run on the same dataset
    and model, one row per adapter and version: the number of runs compared, and the comparison of
    the run that is furthest from the reference, so that no run of a version is left unseen. Runs
    without a completed reference run are left out, and so are the records of a probe, which
    carry no coefficients."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("status") == "ok" and (r.get("fit") or {}).get("coefficients"):
            groups.setdefault((r["dataset"]["id"], r["model"]["label"]), []).append(r)
    out = []
    for (dataset, model), runs in sorted(groups.items()):
        base: Optional[Dict[str, Any]] = next((r for r in runs if r["adapter"]["name"] == reference), None)
        if base is None:
            continue
        builds: Dict[Any, List[Dict[str, Any]]] = {}
        for r in runs:
            if r["adapter"]["name"] != reference:
                builds.setdefault((r["adapter"]["name"], _version(r)), []).append(r)
        for (name, version), same in builds.items():
            compared = [(compare_fits(r["fit"], base["fit"]), r) for r in same]
            worst, r = max(compared, key=lambda pair: pair[0]["max_abs_coefficient_difference"])
            exact = r["support"]["status"] == "exact" and base["support"]["status"] == "exact"
            out.append({"dataset": dataset, "model": model, "adapter": name, "version": version, "runs": len(same),
                        "reference": reference, "like_for_like": exact,
                        # decision D-09: a reference that halved a step may have stopped short of the root
                        "reference_halved_steps": (base["fit"].get("extras") or {}).get("halved_steps"),
                        "notes": {name: r["support"]["notes"], reference: base["support"]["notes"]},
                        **worst})
    return out
