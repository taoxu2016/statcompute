"""NumPy kernels for the two case studies. Each returns sums that add up over disjoint sets of
whole groups (strata or providers), so the same function serves one machine and a Spark partition.

cox_partial               Breslow partial likelihood; information by the weighted-Gram identity
                          I = X' diag(E) X - sum_t d_t xbar(t) xbar(t)',  E_i = expected events of row i
logistic_profile_partial  profile likelihood of beta: each provider's intercept solved in place
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class DegenerateProviders(ValueError):
    """A provider has no events or only events: its intercept has no finite solution."""


def _expit(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def cox_partial(X: np.ndarray, time: np.ndarray, event: np.ndarray, strata: np.ndarray, beta: np.ndarray,
                presorted: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    """Log partial likelihood, score and information (Breslow ties, right-censored rows) of the
    rows given, which must hold whole strata. `presorted` promises rows ordered by stratum and then
    by time descending. Second moments per event time are never formed."""
    n, p = X.shape
    if not presorted:
        order = np.lexsort((-time, strata))
        X, time, event, strata = X[order], time[order], event[order], strata[order]
    d = event.astype(np.float64)
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = strata[1:] != strata[:-1]
    start = np.flatnonzero(first)
    sid = np.cumsum(first) - 1
    eta = X @ beta
    eta = eta - np.maximum.reduceat(eta, start)[sid]      # a shift per stratum changes nothing, and avoids overflow
    rw = np.exp(eta)
    last = np.empty(n, dtype=bool)                        # last row of each (stratum, time) group
    last[-1] = True
    last[:-1] = first[1:] | (time[1:] != time[:-1])
    ends = np.flatnonzero(last)
    grp = np.cumsum(np.concatenate(([0], last[:-1])))     # tie-group index of each row
    gsid = sid[ends]
    c0 = np.cumsum(rw, dtype=np.longdouble)
    base0 = np.concatenate((np.zeros(1, dtype=np.longdouble), c0[start[1:] - 1]))
    S0 = (c0[ends] - base0[gsid]).astype(np.float64)      # risk-set sum at each group's time
    C1 = np.cumsum(rw[:, None] * X, axis=0)
    base1 = np.vstack((np.zeros((1, p)), C1[start[1:] - 1]))
    S1 = C1[ends] - base1[gsid]
    dg = np.bincount(grp, weights=d, minlength=len(ends))
    has = dg > 0
    xbar = S1[has] / S0[has, None]
    loglik = float(d @ eta - dg[has] @ np.log(S0[has]))
    score = d @ X - dg[has] @ xbar
    h = np.zeros(len(ends))
    h[has] = dg[has] / S0[has]                            # hazard increment at each event time
    suffix = np.cumsum(h[::-1])[::-1]
    gfirst = np.empty(len(ends), dtype=bool)
    gfirst[0] = True
    gfirst[1:] = gsid[1:] != gsid[:-1]
    gstart = np.flatnonzero(gfirst)
    beyond = np.concatenate((suffix, [0.0]))[np.concatenate((gstart[1:], [len(ends)]))]
    H = suffix - beyond[gsid]                             # cumulative hazard up to each group's time
    E = rw * H[grp]                                       # expected events of each row
    information = (X * E[:, None]).T @ X - (xbar * dg[has, None]).T @ xbar
    return loglik, score, information


def logistic_profile_partial(X: np.ndarray, y: np.ndarray, provider: np.ndarray, beta: np.ndarray,
                             presorted: bool = False, inner_tol: float = 1e-13, max_inner: int = 100
                             ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Profile log-likelihood of beta, its gradient and its negative Hessian (the Schur complement)
    over whole providers, with every provider's intercept solved by Newton's method in place.
    Returns (loglik, gradient, schur, gamma); gamma is in order of first appearance after sorting."""
    n, p = X.shape
    if not presorted:
        order = np.argsort(provider, kind="stable")
        X, y, provider = X[order], y[order], provider[order]
    y = y.astype(np.float64)
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = provider[1:] != provider[:-1]
    start = np.flatnonzero(first)
    codes = np.cumsum(first) - 1
    size = np.diff(np.concatenate((start, [n]))).astype(np.float64)
    events = np.add.reduceat(y, start)
    if np.any(events <= 0) or np.any(events >= size):
        raise DegenerateProviders("a provider has no events or only events; bounded intercepts are not prototyped")
    xb = X @ beta
    gamma = np.log(events / (size - events)) - np.add.reduceat(xb, start) / size
    for _ in range(max_inner):
        pi = _expit(gamma[codes] + xb)
        step = (events - np.add.reduceat(pi, start)) / np.add.reduceat(pi * (1.0 - pi), start)
        gamma += np.clip(step, -2.0, 2.0)
        if np.max(np.abs(step)) < inner_tol:
            break
    lin = gamma[codes] + xb
    pi = _expit(lin)
    q = pi * (1.0 - pi)
    residual = y - pi
    loglik = float(y @ lin - np.logaddexp(0.0, lin).sum())
    g = np.add.reduceat(residual, start)
    h = np.add.reduceat(q, start)
    xbar = np.add.reduceat(q[:, None] * X, start, axis=0) / h[:, None]
    gradient = residual @ X - g @ xbar
    schur = (X * q[:, None]).T @ X - (xbar * h[:, None]).T @ xbar
    return loglik, gradient, schur, gamma


def whole_group_blocks(groups_sorted: np.ndarray, block_rows: int):
    """Index ranges that cover `groups_sorted` (sorted by group) and cut only between groups."""
    n = len(groups_sorted)
    bounds = np.concatenate((np.flatnonzero(groups_sorted[1:] != groups_sorted[:-1]) + 1, [n]))
    blocks, begin = [], 0
    for b in bounds:
        if b - begin >= block_rows or b == n:
            blocks.append((begin, int(b)))
            begin = int(b)
    return blocks
