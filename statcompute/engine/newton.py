"""Newton's method with step halving for a concave objective given by (value, gradient, negative
Hessian). It stops on a measure that does not depend on the path taken: the largest component of
the next Newton step, in standard errors (tolerance class T_CONV of the roadmap, F-10)."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

Evaluate = Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]]


def maximize(evaluate: Evaluate, p: int, tol: float = 1e-8, max_iter: int = 50) -> Dict[str, object]:
    beta = np.zeros(p)
    value, gradient, information = evaluate(beta)
    evaluations, iterations, converged = 1, 0, False
    while True:
        covariance = np.linalg.inv(information)
        step = covariance @ gradient
        step_in_se = float(np.max(np.abs(step) / np.sqrt(np.diag(covariance))))
        if step_in_se <= tol:
            converged = True
            break
        if iterations >= max_iter:
            break
        length = 1.0
        while True:
            trial = evaluate(beta + length * step)
            evaluations += 1
            if trial[0] >= value - 1e-12 * abs(value) or length < 1e-6:
                break
            length *= 0.5
        beta = beta + length * step
        value, gradient, information = trial
        iterations += 1
    return {
        "beta": beta,
        "covariance": covariance,
        "se": np.sqrt(np.diag(covariance)),
        "loglik": float(value),
        "iterations": iterations,
        "evaluations": evaluations,
        "converged": converged,
        "newton_step_in_se": step_in_se,
    }
