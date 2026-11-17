"""The engine's two models fitted from arrays on one machine. The kernels and Newton's method are
the ones the Spark estimators use; only the loop over blocks differs. Used by the tests as the
reference for the Spark path, and usable on its own for data that fit in memory."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import kernels
from .newton import maximize


def cox_fit(X: np.ndarray, time: np.ndarray, event: np.ndarray, strata: Optional[np.ndarray] = None, *,
            ties: str = "breslow", weights: Optional[np.ndarray] = None, offset: Optional[np.ndarray] = None,
            entry: Optional[np.ndarray] = None, tol: float = 1e-8, max_iterations: int = 50,
            block_rows: int = 50_000) -> Dict[str, object]:
    n, p = X.shape
    codes = np.zeros(n, dtype=np.int64) if strata is None else np.unique(strata, return_inverse=True)[1]
    order = np.argsort(codes, kind="stable")
    take = lambda v: None if v is None else np.asarray(v)[order]  # noqa: E731
    X, time, event, codes = X[order], np.asarray(time, dtype=float)[order], np.asarray(event)[order], codes[order]
    weights, offset, entry = take(weights), take(offset), take(entry)
    blocks = kernels.whole_group_blocks(codes, block_rows)
    totals = {}

    def evaluate(beta):
        value, score, information = 0.0, np.zeros(p), np.zeros((p, p))
        events = deaths = 0
        for a, b in blocks:
            part = kernels.cox_partial(
                X[a:b], time[a:b], event[a:b], codes[a:b], beta, ties=ties,
                weights=None if weights is None else weights[a:b], offset=None if offset is None else offset[a:b],
                entry=None if entry is None else entry[a:b])
            value, score, information = value + part["loglik"], score + part["score"], information + part["information"]
            events, deaths = events + part["events"], deaths + part["deaths"]
        totals.update(events=events, deaths=deaths)
        return value, score, information

    fit = maximize(evaluate, p, tol, max_iterations)
    return dict(fit, rows=n, **totals)


def logistic_fit(X: np.ndarray, y: np.ndarray, provider: np.ndarray, *, bound: float = 10.0, tol: float = 1e-8,
                 max_iterations: int = 50, block_rows: int = 50_000) -> Dict[str, object]:
    """Logistic regression with one intercept per provider. Providers without events, or with only
    events, are held at the median intercept minus or plus `bound`, which is the estimate pprof_py's
    SerBIN converges to (docs/spec/engine.md, section 3)."""
    n, p = X.shape
    ids, codes = np.unique(provider, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    X, y, codes = X[order], np.asarray(y, dtype=float)[order], codes[order]
    events = np.bincount(codes, weights=y, minlength=len(ids))
    rows = np.bincount(codes, minlength=len(ids))
    degenerate = (events <= 0) | (events >= rows)
    inside = ~degenerate[codes]
    Xi, yi, ci = X[inside], y[inside], codes[inside]
    Xd, yd, cd = X[~inside], y[~inside], codes[~inside]
    blocks = kernels.whole_group_blocks(ci, block_rows) if len(ci) else []
    low = events[degenerate] <= 0                              # no events: held at the lower bound
    state = {}

    def evaluate(beta):
        value, gradient, schur, free = 0.0, np.zeros(p), np.zeros((p, p)), []
        for a, b in blocks:
            part = kernels.logistic_partial(Xi[a:b], yi[a:b], ci[a:b], beta)
            value, gradient, schur = value + part["loglik"], gradient + part["gradient"], schur + part["schur"]
            free.append(part["gamma"])
        free = np.concatenate(free) if free else np.zeros(0)
        held = bounded_gamma(free, int(low.sum()), int((~low).sum()), bound)
        gamma = np.empty(len(ids))
        gamma[~degenerate] = free
        if degenerate.any():
            gamma[degenerate] = np.where(low, held["lower"], held["upper"])
            part = kernels.logistic_partial(Xd, yd, cd, beta, gamma=gamma[degenerate])
            value, gradient, schur = value + part["loglik"], gradient + part["gradient"], schur + part["schur"]
        state.update(gamma=gamma, median=held["median"])
        return value, gradient, schur

    fit = maximize(evaluate, p, tol, max_iterations)
    return dict(fit, rows=n, providers=ids, gamma=state["gamma"], bounded=degenerate, median_gamma=state["median"])


def bounded_gamma(free: np.ndarray, low: int, high: int, bound: float) -> Dict[str, float]:
    """The bounds for this evaluation. `free` holds the intercepts of the providers that have one;
    `low` providers have no events and `high` have only events. The median is taken over all
    providers, the degenerate ones lying below and above everything else."""
    everything = np.concatenate((free, np.full(low, -np.inf), np.full(high, np.inf)))
    median = float(np.median(everything)) if len(everything) else 0.0
    if not np.isfinite(median):
        raise NotImplementedError("half of the providers or more have no events or only events")
    if len(free) and np.max(np.abs(free - median)) > bound:
        raise NotImplementedError("a provider with events and non-events lies outside the bound; "
                                  "that case is not implemented (OPEN_ITEMS.md)")
    return {"median": median, "lower": median - bound, "upper": median + bound}
