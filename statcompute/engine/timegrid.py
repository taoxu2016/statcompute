"""Cox regression on a time grid: the layout for fits without strata.

When rows are not tied to a stratum that fits in one partition, the risk sets span the whole table.
This layout needs the rows in no order. Each evaluation takes two passes over the rows in
whatever partitions and batches they come:

  pass 1   every batch adds its rows' risk scores r and r*x to a table indexed by time (and, where
           needed, the same for the tied deaths and for entry times). The driver adds the tables,
           takes suffix sums over the grid and has every risk-set sum S0(t), S1(t). From them it
           computes the log-likelihood, the score, the hazard increment at each time and the term
           M = sum over tie steps of weight * xbar xbar'.
  pass 2   every batch forms X' diag(E) X, with each row's expected events E read off the
           cumulative hazard at its exit and entry times. The information is the sum minus M.

Only first moments are kept per time: (p + 1) numbers, where the classical computation needs
p(p + 1)/2 more. The times must take a limited number of distinct values (days, say).
docs/spec/engine.md, section 2.2, has the algebra.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def layout(K: int, p: int, efron: bool, entry: bool) -> Dict[str, slice]:
    """Where each piece lies in the flat vector a pass-1 partition returns."""
    width, at, out = K * (p + 1), 0, {}
    for name, size in (("exit", width), ("dead", width if efron else 0), ("entry", width if entry else 0),
                       ("weighted_deaths", K), ("deaths", K), ("loglik", 1), ("score", p), ("rows", 1), ("eta_max", 1)):
        out[name] = slice(at, at + size)
        at += size
    out["size"] = at
    return out


def new_tables(K: int, p: int, efron: bool, entry: bool) -> np.ndarray:
    tables = np.zeros(layout(K, p, efron, entry)["size"])
    tables[-1] = -np.inf
    return tables


def _scatter(table: np.ndarray, codes: np.ndarray, r: np.ndarray, RX: np.ndarray) -> None:
    """table[code] += (r, r*x) for every row; table is [K, p + 1]."""
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    first = np.empty(len(sc), dtype=bool)
    first[0] = True
    first[1:] = sc[1:] != sc[:-1]
    start = np.flatnonzero(first)
    table[sc[start], 0] += np.add.reduceat(r[order], start)
    table[sc[start], 1:] += np.add.reduceat(RX[order], start, axis=0)


def _risk(X, beta, mu, shift, weights, offset):
    Xc = X - mu
    eta = Xc @ beta - shift
    if offset is not None:
        eta = eta + offset
    r = np.exp(eta) if weights is None else weights * np.exp(eta)
    return Xc, eta, r


def accumulate(tables: np.ndarray, grid: np.ndarray, X: np.ndarray, time: np.ndarray, event: np.ndarray,
               beta: np.ndarray, mu: np.ndarray, shift: float, *, efron: bool, weights: Optional[np.ndarray] = None,
               offset: Optional[np.ndarray] = None, entry: Optional[np.ndarray] = None) -> None:
    """Pass 1 for one batch of rows: add its contributions to `tables` in place."""
    K, p = len(grid), X.shape[1]
    where = layout(K, p, efron, entry is not None)
    Xc, eta, r = _risk(X, beta, mu, shift, weights, offset)
    RX = r[:, None] * Xc
    code = np.searchsorted(grid, time)
    d = event.astype(np.float64)
    dw = d if weights is None else d * weights
    _scatter(tables[where["exit"]].reshape(K, p + 1), code, r, RX)
    tables[where["weighted_deaths"]] += np.bincount(code, weights=dw, minlength=K)
    tables[where["deaths"]] += np.bincount(code, weights=d, minlength=K)
    tables[where["loglik"]] += dw @ eta
    tables[where["score"]] += dw @ Xc
    tables[where["rows"]] += len(time)
    tables[-1] = max(tables[-1], float(eta.max()) + shift)
    if efron:
        dead = np.flatnonzero(d > 0)
        if len(dead):
            _scatter(tables[where["dead"]].reshape(K, p + 1), code[dead], r[dead], RX[dead])
    if entry is not None:
        inside = np.flatnonzero(np.isfinite(entry))
        if len(inside):
            _scatter(tables[where["entry"]].reshape(K, p + 1), np.searchsorted(grid, entry[inside]), r[inside], RX[inside])


def add(total: Optional[np.ndarray], part: np.ndarray) -> np.ndarray:
    """Sum of two pass-1 results (the last element is a maximum, not a sum)."""
    if total is None:
        return part.copy()
    top = max(total[-1], part[-1])
    total += part
    total[-1] = top
    return total


def centre(tables: np.ndarray, K: int, p: int, mu: np.ndarray, *, efron: bool, entry: bool) -> None:
    """Turn summed pass-1 tables that were accumulated at beta = 0 with mu = 0 into the tables that
    mu would have given, in place. At beta = 0 the risk scores do not depend on the centring, so
    each sum of r*x only loses mu times its sum of r, and the score mu times the weighted events.
    This lets the first pass over the rows also yield the feature means it should be centred at."""
    where = layout(K, p, efron, entry)
    for name in ("exit", "dead", "entry"):
        if where[name].stop > where[name].start:
            table = tables[where[name]].reshape(K, p + 1)
            table[:, 1:] -= table[:, :1] * mu
    tables[where["score"]] -= tables[where["weighted_deaths"]].sum() * mu


def _suffix(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values[::-1], axis=0, dtype=np.longdouble)[::-1].astype(np.float64)


def reduce(tables: np.ndarray, K: int, p: int, *, efron: bool, entry: bool, max_steps: int = 2_000_000) -> Dict[str, object]:
    """From the summed tables: log-likelihood, score, the matrix M, the cumulative hazard at each
    grid time and, for Efron, the correction for tied deaths."""
    where = layout(K, p, efron, entry)
    exits = tables[where["exit"]].reshape(K, p + 1)
    S = _suffix(exits)                                     # sums over rows with time >= t
    if entry:
        S = S - _suffix(tables[where["entry"]].reshape(K, p + 1))      # minus rows that have not entered by t
    S0, S1 = S[:, 0], S[:, 1:]
    weighted, deaths = tables[where["weighted_deaths"]], tables[where["deaths"]]
    at = np.flatnonzero(deaths > 0)                        # the event times
    if not np.all(np.isfinite(S0[at])) or np.any(S0[at] <= 0):
        raise FloatingPointError("a risk-set sum is not finite and positive: the linear predictor overflowed")
    loglik, score = float(tables[where["loglik"]][0]), tables[where["score"]].copy()
    h, tied = np.zeros(K), np.zeros(K)
    if not efron:
        xbar = S1[at] / S0[at, None]
        loglik -= float(weighted[at] @ np.log(S0[at]))
        score -= weighted[at] @ xbar
        M = (xbar * weighted[at, None]).T @ xbar
        h[at] = weighted[at] / S0[at]
    else:
        D = tables[where["dead"]].reshape(K, p + 1)
        D0, D1, S1e = D[at, 0], D[at, 1:], S1[at]
        k = deaths[at].astype(np.int64)
        wbar = weighted[at] / k
        sums = np.zeros((6, len(at)))                      # over each time's steps: a, b, a*a, a*b, b*b, log A
        begin = 0
        while begin < len(at):                             # in chunks, so the expanded steps stay small
            end = begin + max(1, int(np.searchsorted(np.cumsum(k[begin:]), max_steps, side="right")))
            kk = k[begin:end]
            first = np.cumsum(kk) - kk
            rep = np.repeat(np.arange(len(kk)), kk)
            frac = (np.arange(int(kk.sum())) - first[rep]) / kk[rep]
            A = S0[at[begin:end]][rep] - frac * D0[begin:end][rep]
            a, b = 1.0 / A, frac / A
            for row, value in enumerate((a, b, a * a, a * b, b * b, np.log(A))):
                sums[row, begin:end] = np.add.reduceat(value, first)
            begin = end
        sa, sb, saa, sab, sbb, slog = sums
        loglik -= float(wbar @ slog)
        score -= (wbar * sa) @ S1e - (wbar * sb) @ D1
        # sum over steps of xbar xbar' with xbar = a*S1 - b*D1: three weights per time instead of one matrix per step
        cross = (S1e * (wbar * sab)[:, None]).T @ D1
        M = (S1e * (wbar * saa)[:, None]).T @ S1e - cross - cross.T + (D1 * (wbar * sbb)[:, None]).T @ D1
        h[at] = wbar * sa
        tied[at] = wbar * sb
    return {"loglik": loglik, "score": score, "M": M, "cumulative": np.cumsum(h), "tied": tied,
            "events": float(weighted.sum()), "deaths": int(round(deaths.sum())), "rows": int(round(tables[where["rows"]][0])),
            "eta_max": float(tables[-1])}


def gram(G: np.ndarray, grid: np.ndarray, cumulative: np.ndarray, tied: np.ndarray, X: np.ndarray, time: np.ndarray,
         event: np.ndarray, beta: np.ndarray, mu: np.ndarray, shift: float, *, weights: Optional[np.ndarray] = None,
         offset: Optional[np.ndarray] = None, entry: Optional[np.ndarray] = None) -> None:
    """Pass 2 for one batch of rows: G += X' diag(E) X, in place."""
    Xc, _, r = _risk(X, beta, mu, shift, weights, offset)
    code = np.searchsorted(grid, time)
    exposure = cumulative[code]
    if entry is not None:
        inside = np.isfinite(entry)
        exposure = exposure - np.where(inside, cumulative[np.searchsorted(grid, np.where(inside, entry, grid[0]))], 0.0)
    E = r * exposure - event * r * tied[code]
    G += (Xc * E[:, None]).T @ Xc


def evaluate_arrays(X: np.ndarray, time: np.ndarray, event: np.ndarray, beta: np.ndarray, *, ties: str = "breslow",
                    weights: Optional[np.ndarray] = None, offset: Optional[np.ndarray] = None,
                    entry: Optional[np.ndarray] = None, grid: Optional[np.ndarray] = None, mu: Optional[np.ndarray] = None,
                    shift: float = 0.0, chunk_rows: int = 100_000) -> Dict[str, object]:
    """Both passes over arrays in memory, chunk by chunk: the single-machine form of the layout."""
    n, p = X.shape
    efron, has_entry = ties == "efron", entry is not None and bool(np.isfinite(entry).any())
    entry = entry if has_entry else None
    if grid is None:
        grid = np.unique(time if entry is None else np.concatenate((time, entry[np.isfinite(entry)])))
    mu = X.mean(axis=0) if mu is None else mu
    K = len(grid)
    pick = lambda v, a, b: None if v is None else v[a:b]  # noqa: E731
    tables = new_tables(K, p, efron, has_entry)
    for a in range(0, n, chunk_rows):
        b = min(n, a + chunk_rows)
        accumulate(tables, grid, X[a:b], time[a:b], event[a:b], beta, mu, shift, efron=efron,
                   weights=pick(weights, a, b), offset=pick(offset, a, b), entry=pick(entry, a, b))
    out = reduce(tables, K, p, efron=efron, entry=has_entry)
    G = np.zeros((p, p))
    for a in range(0, n, chunk_rows):
        b = min(n, a + chunk_rows)
        gram(G, grid, out["cumulative"], out["tied"], X[a:b], time[a:b], event[a:b], beta, mu, shift,
             weights=pick(weights, a, b), offset=pick(offset, a, b), entry=pick(entry, a, b))
    out["information"] = G - out["M"]
    return out
