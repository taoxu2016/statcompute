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


def compare_records(records: Iterable[Dict[str, Any]], reference: str = "pprof_py") -> List[Dict[str, Any]]:
    """Every completed run against the reference adapter's first completed run on the same dataset
    and model. Runs without a completed reference run are left out."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("status") == "ok":
            groups.setdefault((r["dataset"]["id"], r["model"]["label"]), []).append(r)
    out = []
    for (dataset, model), runs in sorted(groups.items()):
        base: Optional[Dict[str, Any]] = next((r for r in runs if r["adapter"]["name"] == reference), None)
        if base is None:
            continue
        seen = set()
        for r in runs:
            name = r["adapter"]["name"]
            if name == reference or name in seen:
                continue
            seen.add(name)
            exact = r["support"]["status"] == "exact" and base["support"]["status"] == "exact"
            out.append({"dataset": dataset, "model": model, "adapter": name, "reference": reference,
                        "like_for_like": exact,
                        "notes": {name: r["support"]["notes"], reference: base["support"]["notes"]},
                        **compare_fits(r["fit"], base["fit"])})
    return out
