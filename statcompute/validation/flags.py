"""Provider-level flags: do two computations of the same fit flag the same providers?

A flag is a cut-off on a continuous statistic: |z| against a normal quantile for the score and Wald
tests, a p-value against 1 - level for the exact test. Two computations can give one provider
different flags only when the cut-off lies between its two statistics, so both are then closer to the
cut-off than they are apart.

The rule proposed in DECISIONS.md (D-15): a flag that differs is acceptable when the two statistics
are within the tolerance of each other; the provider is then borderline, and is reported. Any other
difference in a flag is a numerical disagreement between the two computations. The tolerances
below are provisional until the maintainer has ruled.

Tables are the engine's `provider_effects`: one row per provider with the columns named in TESTS.
"""
from __future__ import annotations

import hashlib
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# test: (column of the statistic, column of the flag, scale of the statistic)
TESTS = {"score": ("score_z", "score_flag", "z"), "wald": ("wald_z", "wald_flag", "z"),
         "exact": ("exact_p", "exact_flag", "p")}
T_FLAG = {"z": 1e-7, "p": 1e-9}        # how far apart two computations of one statistic may be


def distance_from_cutoff(values, scale: str, level: float = 0.95) -> np.ndarray:
    """How far each statistic is from the cut-off that decides its flag."""
    values = np.asarray(values, dtype=float)
    if scale == "z":
        return np.abs(np.abs(values) - NormalDist().inv_cdf(0.5 + level / 2.0))
    return np.abs(values - (1.0 - level))


def _present(table, test: str) -> bool:
    statistic, flag, _ = TESTS[test]
    return statistic in table.columns and flag in table.columns


def flag_summary(table, key: str = "provider", level: float = 0.95,
                 tolerance: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
    """What a run record keeps of a table of provider tests, per test: how many providers are flagged
    above and below, a digest of the flags in provider order (equal digests: the same providers
    flagged the same way), the smallest distance of a statistic from its cut-off, and how many
    providers lie within the tolerance of it. Enough to compare runs without storing the table."""
    tolerance = dict(T_FLAG, **(tolerance or {}))
    table = table.sort_values(key)
    out: Dict[str, Dict[str, Any]] = {}
    for test, (statistic, flag, scale) in TESTS.items():
        if not _present(table, test):
            continue
        flags = table[flag].to_numpy(dtype=np.int8)
        distance = distance_from_cutoff(table[statistic].to_numpy(dtype=float), scale, level)
        distance = distance[np.isfinite(distance)]
        out[test] = {"above": int(np.sum(flags > 0)), "below": int(np.sum(flags < 0)),
                     "digest": hashlib.sha256(flags.tobytes()).hexdigest()[:16],
                     "smallest_distance_from_cutoff": float(distance.min()) if len(distance) else None,
                     "within_tolerance_of_cutoff": int(np.sum(distance <= tolerance[scale]))}
    return out


def compare_flags(table, reference, key: str = "provider", level: float = 0.95,
                  tolerance: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
    """Two tables of provider tests for the same providers, per test that both hold:

    flags_that_differ                the providers flagged differently
    of_them_borderline               whose two statistics are within the tolerance of each other
    of_them_not                      the others: a numerical disagreement
    largest_difference_in_statistic  over all providers
    smallest_distance_from_cutoff    over both tables
    within_tolerance_of_cutoff       providers whose flag could differ and still be acceptable
    stable                           no flag differs beyond the tolerance
    """
    tolerance = dict(T_FLAG, **(tolerance or {}))
    a, b = table.sort_values(key).reset_index(drop=True), reference.sort_values(key).reset_index(drop=True)
    if len(a) != len(b) or not np.array_equal(a[key].to_numpy(), b[key].to_numpy()):
        raise ValueError("the two tables do not hold the same providers")
    out: Dict[str, Dict[str, Any]] = {}
    for test, (statistic, flag, scale) in TESTS.items():
        if not (_present(a, test) and _present(b, test)):
            continue
        x, y = a[statistic].to_numpy(dtype=float), b[statistic].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            apart = np.where((x == y) | (np.isnan(x) & np.isnan(y)), 0.0, np.abs(x - y))
        apart = np.where(np.isnan(apart), np.inf, apart)                 # a number beside none is a difference
        differ = a[flag].to_numpy() != b[flag].to_numpy()
        borderline = differ & (apart <= tolerance[scale])
        nearest = np.fmin(distance_from_cutoff(x, scale, level), distance_from_cutoff(y, scale, level))
        nearest = nearest[np.isfinite(nearest)]
        out[test] = {"providers": int(len(a)), "flags_that_differ": int(differ.sum()),
                     "of_them_borderline": int(borderline.sum()), "of_them_not": int((differ & ~borderline).sum()),
                     "largest_difference_in_statistic": float(apart.max()) if len(apart) else 0.0,
                     "smallest_distance_from_cutoff": float(nearest.min()) if len(nearest) else None,
                     "within_tolerance_of_cutoff": int(np.sum(nearest <= tolerance[scale])),
                     "stable": not bool((differ & ~borderline).any())}
    return out


def flags_across_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For every dataset, model and adapter whose completed runs carry a flag summary: the number of
    runs and, per test, whether all of them flagged the same providers (equal digests), with the
    smallest distance from the cut-off seen in any run."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        flags = ((r.get("fit") or {}).get("extras") or {}).get("flags")
        if r.get("status") == "ok" and flags:
            groups.setdefault((r["dataset"]["id"], r["model"]["label"], r["adapter"]["name"]), []).append(flags)
    out = []
    for (dataset, model, adapter), runs in sorted(groups.items()):
        row: Dict[str, Any] = {"dataset": dataset, "model": model, "adapter": adapter, "runs": len(runs)}
        for test in TESTS:
            seen = [run[test] for run in runs if test in run]
            if not seen:
                continue
            distances = [s["smallest_distance_from_cutoff"] for s in seen if s["smallest_distance_from_cutoff"] is not None]
            row[f"{test}_flagged"] = f"{seen[0]['above']} above, {seen[0]['below']} below"
            row[f"{test}_same_in_all_runs"] = len({s["digest"] for s in seen}) == 1
            row[f"{test}_smallest_distance_from_cutoff"] = min(distances) if distances else None
        out.append(row)
    return out
