"""Kernels of the engine: NumPy on whole groups (strata or providers).

Every function returns sums that add over disjoint sets of whole groups, so one function serves a
Spark partition and a single machine.

cox_partial        Partial likelihood, score and information of a Cox model with Breslow or Efron
                   ties, case weights, offsets, entry times and strata. The information is formed as
                       X' diag(E) X  -  sum over tie steps of  wbar * xbar xbar',
                   where E_i is row i's expected number of events (docs/spec/engine.md). No second
                   moment per event time is ever formed. Covariates are centred within each stratum
                   first: that changes nothing statistically and removes a cancellation.
logistic_partial   Profile likelihood of beta in a logistic model with one intercept per provider:
                   every provider's intercept is solved in place, or held at a given value.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class DegenerateProviders(ValueError):
    """A provider has no events or only events, so its intercept has no finite solution."""


def _starts(sorted_values: np.ndarray) -> np.ndarray:
    first = np.empty(len(sorted_values), dtype=bool)
    first[0] = True
    first[1:] = sorted_values[1:] != sorted_values[:-1]
    return first


def cox_partial(X: np.ndarray, time: np.ndarray, event: np.ndarray, strata: np.ndarray, beta: np.ndarray, *,
                ties: str = "breslow", weights: Optional[np.ndarray] = None, offset: Optional[np.ndarray] = None,
                entry: Optional[np.ndarray] = None, order: Optional[np.ndarray] = None,
                baseline: bool = False, robust: bool = False, rows: bool = False) -> Dict[str, object]:
    """Sums for the rows given, which must hold whole strata.

    With `baseline` the result also holds, for every event time of every stratum, the increment of
    the baseline hazard at x = 0 and offset 0 (Breslow's estimate, or Efron's). With `rows` it holds
    each row's expected events, in the order the rows were given (the martingale residual is the
    event minus this). With `robust` it holds the sum over rows of the outer products of their score
    residuals, the middle of the sandwich variance (Breslow ties only).

    time, event    exit time and event indicator (1 = event) of each row
    strata         integer stratum code of each row
    weights        case weights (default 1); offset: added to the linear predictor (default 0)
    entry          entry times: a row is at risk at t when entry < t <= time. NaN or -inf: at risk from the start
    order          optional tie-breaker (for example a row identifier) that fixes the order of rows
                   with equal stratum and time, and with it the last bits of the sums

    Returns loglik, score, information, events (weighted) and deaths (count).
    """
    if ties not in ("breslow", "efron"):
        raise ValueError("ties must be 'breslow' or 'efron'")
    n, p = X.shape
    keys = (-time, strata) if order is None else (order, -time, strata)
    perm = np.lexsort(keys)                               # stratum ascending, then time descending
    X, t, s = X[perm], time[perm], strata[perm]
    d = event[perm].astype(np.float64)
    w = np.ones(n) if weights is None else weights[perm].astype(np.float64)

    first = _starts(s)
    start = np.flatnonzero(first)
    sid = np.cumsum(first) - 1
    size = np.diff(np.concatenate((start, [n])))
    centre = np.add.reduceat(X, start, axis=0) / size[:, None]
    Xc = X - centre[sid]                                  # centred within the stratum
    eta = Xc @ beta
    if offset is not None:
        eta = eta + offset[perm]
    shift = np.maximum.reduceat(eta, start)
    eta = eta - shift[sid]                                # a constant per stratum changes nothing; avoids overflow
    r = w * np.exp(eta)                                   # weighted risk score
    RX = r[:, None] * Xc

    last = np.empty(n, dtype=bool)                        # last row of each (stratum, time) group
    last[-1] = True
    last[:-1] = first[1:] | (t[1:] != t[:-1])
    ends = np.flatnonzero(last)
    group_first_row = np.concatenate(([0], ends[:-1] + 1))
    grp = np.cumsum(np.concatenate(([0], last[:-1])))     # group of each row
    gsid = sid[ends]
    G = len(ends)

    c0 = np.cumsum(r, dtype=np.longdouble)
    base0 = np.concatenate((np.zeros(1, dtype=np.longdouble), c0[start[1:] - 1]))
    S0 = (c0[ends] - base0[gsid]).astype(np.float64)      # sum over rows with time >= t
    C1 = np.cumsum(RX, axis=0)
    base1 = np.vstack((np.zeros((1, p)), C1[start[1:] - 1]))
    S1 = C1[ends] - base1[gsid]

    truncated = False
    if entry is not None:
        e = entry[perm].astype(np.float64)
        finite = np.isfinite(e)
        truncated = bool(finite.any())
    if truncated:
        # Times become integer codes, so that "entry >= t" is decided on the original values.
        values = np.unique(np.concatenate((t, e[finite])))
        V = len(values)
        tcode = np.searchsorted(values, t[ends])
        ecode = np.full(n, -1, dtype=np.int64)
        ecode[finite] = np.searchsorted(values, e[finite])
        by_entry = np.lexsort((-ecode, s))                # stratum ascending, entry descending, none last
        ekey = sid[by_entry].astype(np.int64) * (V + 2) + (V - ecode[by_entry])
        gkey = gsid.astype(np.int64) * (V + 2) + (V - tcode)
        ce0 = np.cumsum(r[by_entry], dtype=np.longdouble)
        ebase0 = np.concatenate((np.zeros(1, dtype=np.longdouble), ce0[start[1:] - 1]))
        CE1 = np.cumsum(RX[by_entry], axis=0)
        ebase1 = np.vstack((np.zeros((1, p)), CE1[start[1:] - 1]))
        k = np.searchsorted(ekey, gkey, side="right")     # rows up to here have entry >= the group's time
        some = k > start[gsid]
        at = np.maximum(k - 1, 0)
        S0 = S0 - np.where(some, (ce0[at] - ebase0[gsid]).astype(np.float64), 0.0)
        S1 = S1 - np.where(some[:, None], CE1[at] - ebase1[gsid], 0.0)

    dw = d * w
    weighted_deaths = np.add.reduceat(dw, group_first_row)
    deaths = np.add.reduceat(d, group_first_row)
    has = deaths > 0
    loglik = float(dw @ eta)
    score = dw @ Xc
    h = np.zeros(G)                                       # hazard increment at each group's time
    h_tied = None
    if ties == "breslow" or not has.any():
        xbar = S1[has] / S0[has, None]
        loglik -= float(weighted_deaths[has] @ np.log(S0[has]))
        score = score - weighted_deaths[has] @ xbar
        h[has] = weighted_deaths[has] / S0[has]
        M = (xbar * weighted_deaths[has, None]).T @ xbar
    else:
        gi = np.flatnonzero(has)
        k_tied = deaths[gi].astype(np.int64)
        step_first = np.cumsum(k_tied) - k_tied
        idx = np.repeat(np.arange(len(gi)), k_tied)       # the tie group of each Efron step
        frac = (np.arange(int(k_tied.sum())) - step_first[idx]) / k_tied[idx]
        dead = np.flatnonzero(d > 0)                      # rows that died, in group order
        dead_first = np.flatnonzero(_starts(grp[dead]))
        S0D = np.add.reduceat(r[dead], dead_first)        # the same sums over the tied deaths only
        S1D = np.add.reduceat(RX[dead], dead_first, axis=0)
        wbar = weighted_deaths[gi] / k_tied               # mean weight of the tied deaths
        A = S0[gi][idx] - frac * S0D[idx]
        xbar = (S1[gi][idx] - frac[:, None] * S1D[idx]) / A[:, None]
        wstep = wbar[idx]
        loglik -= float(wstep @ np.log(A))
        score = score - wstep @ xbar
        M = (xbar * wstep[:, None]).T @ xbar
        h[gi] = wbar * np.add.reduceat(1.0 / A, step_first)
        h_tied = np.zeros(G)
        h_tied[gi] = wbar * np.add.reduceat(frac / A, step_first)

    suffix = np.concatenate((np.cumsum(h[::-1])[::-1], [0.0]))
    gstart = np.flatnonzero(_starts(gsid))
    gnext = np.concatenate((gstart[1:], [G]))
    beyond = suffix[gnext]                                # per stratum: what lies in later strata
    H = suffix[:-1] - beyond[gsid]                        # cumulative hazard up to each group's time
    exposure = H[grp]
    if truncated:
        qrow = sid.astype(np.int64) * (V + 2) + (V - ecode)
        pos = np.searchsorted(gkey, qrow, side="left")    # first group at or before the entry time
        exposure = exposure - (suffix[pos] - beyond[sid])
    E = r * exposure                                      # expected events of each row
    if h_tied is not None:
        E = E - d * r * h_tied[grp]
    information = (Xc * E[:, None]).T @ Xc - M
    out = {"loglik": loglik, "score": score, "information": information,
           "events": float(weighted_deaths.sum()), "deaths": int(deaths.sum())}
    if rows:
        expected = np.empty(n)
        expected[perm] = E
        out["expected"] = expected
    if robust:
        if h_tied is not None:
            raise NotImplementedError("the robust variance is implemented for Breslow ties only")
        xbar_all = np.zeros((G, p))
        xbar_all[has] = xbar
        sx = np.vstack((np.cumsum((h[:, None] * xbar_all)[::-1], axis=0)[::-1], np.zeros((1, p))))
        beyond_x = sx[gnext]
        integral = (sx[:-1] - beyond_x[gsid])[grp]          # sum over event times up to the row's exit of h * xbar
        if truncated:
            integral = integral - (sx[pos] - beyond_x[sid])
        U = dw[:, None] * (Xc - xbar_all[grp]) - r[:, None] * (Xc * exposure[:, None] - integral)
        out["meat"] = U.T @ U
    if baseline:
        # the kernel's risk scores are exp(x'b - centre'b - shift); undo both to report the hazard at x = 0
        back = np.exp(-(centre @ beta + shift))
        out["baseline"] = {"stratum": gsid[has], "time": t[ends][has], "increment": h[has] * back[gsid[has]]}
    return out


def _expit(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def logistic_partial(X: np.ndarray, y: np.ndarray, provider: np.ndarray, beta: np.ndarray, *,
                     gamma: Optional[np.ndarray] = None, inner_tol: float = 1e-13, max_inner: int = 100
                     ) -> Dict[str, object]:
    """Sums for whole providers, rows sorted by provider code.

    With gamma None every provider's intercept is solved by Newton's method in place (the profile
    likelihood); a provider without events, or with only events, raises DegenerateProviders. With
    gamma given (one value per provider, in code order) the intercepts are held there: this is how
    providers at the bound are evaluated.

    Returns loglik, gradient (the reduced score of beta), schur (the reduced information), gamma,
    and per provider rows, events and the weight h = sum of p(1 - p).
    """
    n, p = X.shape
    y = y.astype(np.float64)
    start = np.flatnonzero(_starts(provider))
    codes = np.cumsum(_starts(provider)) - 1
    size = np.diff(np.concatenate((start, [n]))).astype(np.float64)
    events = np.add.reduceat(y, start)
    xb = X @ beta
    if gamma is None:
        if np.any(events <= 0) or np.any(events >= size):
            raise DegenerateProviders("a provider has no events or only events")
        gamma = np.log(events / (size - events)) - np.add.reduceat(xb, start) / size
        for _ in range(max_inner):
            pi = _expit(gamma[codes] + xb)
            step = (events - np.add.reduceat(pi, start)) / np.add.reduceat(pi * (1.0 - pi), start)
            gamma = gamma + np.clip(step, -2.0, 2.0)
            if np.max(np.abs(step)) < inner_tol:
                break
    else:
        gamma = np.asarray(gamma, dtype=np.float64)
    lin = gamma[codes] + xb
    pi = _expit(lin)
    q = pi * (1.0 - pi)
    residual = y - pi
    g = np.add.reduceat(residual, start)                  # each provider's own score; zero when solved
    h = np.add.reduceat(q, start)
    xbar = np.add.reduceat(q[:, None] * X, start, axis=0) / h[:, None]
    return {"loglik": float(y @ lin - np.logaddexp(0.0, lin).sum()),
            "gradient": residual @ X - g @ xbar,
            "schur": (X * q[:, None]).T @ X - (xbar * h[:, None]).T @ xbar,
            "gamma": gamma, "rows": size, "events": events, "h": h, "xbar": xbar}


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
