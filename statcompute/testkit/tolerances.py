"""Tolerance classes for numerical comparisons, by name. T_FN, T_COEF and T_VAR follow pprof_spark's
project context (its section 8.4), so that results read the same way there. T_CONV is the class
proposed in the research roadmap (F-10): the size of one more Newton step, in standard errors.
T_FLAG_Z and T_FLAG_P are decision D-15: how far apart two computations of one provider's test
statistic may be for a difference in its flag to count as borderline and not as a disagreement.

A scalar passes when |a - e| <= atol + rtol * |e|; element i of a vector passes when
|a_i - e_i| <= atol + rtol * max(|e_i|, s), with s the largest magnitude in the expected vector.
"""
from __future__ import annotations

from typing import Dict, Sequence

TOLERANCES: Dict[str, Dict[str, float]] = {
    "T_FN": {"atol": 0.0, "rtol": 1e-12},      # objective at fixed parameters
    "T_COEF": {"atol": 1e-10, "rtol": 1e-8},   # converged coefficients
    "T_VAR": {"atol": 0.0, "rtol": 1e-7},      # standard errors
    "T_LOGLIK": {"atol": 0.0, "rtol": 1e-9},   # log-likelihood at the converged estimate
    "T_CONV": {"atol": 1e-8, "rtol": 0.0},     # Newton step at the estimate, in standard errors
    "T_FLAG_Z": {"atol": 1e-7, "rtol": 0.0},   # a provider's z statistic (score, Wald), between two computations
    "T_FLAG_P": {"atol": 1e-9, "rtol": 0.0},   # a provider's p-value (exact test), between two computations
}


def excess(actual: Sequence[float], expected: Sequence[float], name: str) -> float:
    """Largest |a - e| / (atol + rtol * max(|e|, s)) over the elements; at most 1 means it passes."""
    tol = TOLERANCES[name]
    scale = max((abs(e) for e in expected), default=0.0)
    worst = 0.0
    for a, e in zip(actual, expected):
        allowed = tol["atol"] + tol["rtol"] * max(abs(e), scale)
        diff = abs(a - e)
        worst = max(worst, diff / allowed if allowed > 0 else (0.0 if diff == 0 else float("inf")))
    return worst
