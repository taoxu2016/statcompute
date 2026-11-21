"""The engine's two estimators on Spark.

Layout. The rows are shuffled once so that every partition holds whole strata (Cox) or whole
providers (logistic), with the features packed into one array column; Spark sorts each partition
into the order the kernels work in, and that table is cached. So the order is computed once, when
the working set is built, and no pass sorts or copies the rows again. With strata or providers the
number of partitions depends only on the number of rows and `rows_per_partition`, never on the size
of the cluster, so the same data give the same partitions on 2 executors and on 10.

Without strata no pass needs the rows in any order, and by default they stay in the input's own
partitions: no shuffle and no sort (decision D-19). On request they are placed by row identifier
in a chosen number of partitions and sorted by it, which costs a shuffle and repeats bit for bit.

Jobs. One light job reads the columns other than the features: it counts the rows and refuses data
that cannot be fitted. The working set is then built by the first pass that needs it, in the same
job; that pass also looks for missing feature values. Nothing else scans the input.

Passes. Every evaluation of the likelihood is one Spark job (two without strata): each partition's
rows go to a Python worker as Arrow batches, a NumPy kernel returns that partition's sums, and the
driver adds them in partition order. Newton's method runs on the driver. The kernels use memory
outside the JVM heap.
"""
from __future__ import annotations

import contextlib
import math
import time as clock
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import kernels as _kernels
from . import timegrid as _timegrid
from .dist import functions_of, ship_package
from .local import bounded_gamma
from .newton import maximize
from .record import execution_record

# The kernels as namespaces of functions: what a task carries to an executor holds no module object (see dist.py).
kernels, timegrid = functions_of(_kernels), functions_of(_timegrid)

ROWS_PER_PARTITION = 50_000
MAX_HELD_ROWS = 200_000
MAX_TIMES = 20_000
DRIVER_TABLES = 768 * 2 ** 20       # pass-1 tables of the time grid beyond this many bytes are added on the executors first
PARTIAL = "ll double, u binary, i binary, n long, e double, g binary, bad long"
SCAN = "n long, providers long, low long, high long, held long, e double, bad long, m binary"
TABLES = "t binary, m binary, bad long"


# ---------------------------------------------------------------- inside the Python workers
def _one_thread():
    """One BLAS thread per worker: an executor already runs one worker per core."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except Exception:
        return contextlib.nullcontext()


def _stack(batches, p: int) -> Optional[Dict[str, Any]]:
    """The rows of one partition as arrays: "x" is float64 [n, p]; the other columns are vectors."""
    xs, columns, key_type = [], {}, None
    for batch in batches:
        if batch.num_rows == 0:
            continue
        if key_type is None and "s" in batch.schema.names:
            key_type = batch.schema.field("s").type
        xs.append(batch.column("x").flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, p))
        for name in batch.schema.names:
            if name != "x":
                columns.setdefault(name, []).append(batch.column(name).to_numpy(zero_copy_only=False))
    if not xs:
        return None
    out = {name: (np.concatenate(parts) if len(parts) > 1 else parts[0]) for name, parts in columns.items()}
    out["x"] = np.concatenate(xs) if len(xs) > 1 else xs[0]
    out["key_type"] = key_type
    return out


def _missing(X: np.ndarray) -> int:
    """How many rows have a missing feature (null or NaN). A NaN anywhere makes the sum NaN, so
    clean data cost one pass and no temporary."""
    if not np.isnan(X.sum()):
        return 0
    return int(np.isnan(X).any(axis=1).sum())


def _runs(keys: np.ndarray):
    """For keys that arrive sorted (equal keys together): the code 0, 1, ... of each row and the
    distinct keys. What a factorization would give, without hashing or sorting."""
    first = kernels._starts(keys)
    return np.cumsum(first) - 1, np.asarray(keys)[first]


def _partial(pa, loglik, vector, matrix, rows, events, gamma=b"", bad=0):
    return pa.RecordBatch.from_arrays(
        [pa.array([float(loglik)], pa.float64()), pa.array([vector.tobytes()], pa.binary()),
         pa.array([np.ascontiguousarray(matrix).tobytes()], pa.binary()), pa.array([int(rows)], pa.int64()),
         pa.array([float(events)], pa.float64()), pa.array([gamma], pa.binary()), pa.array([int(bad)], pa.int64())],
        names=["ll", "u", "i", "n", "e", "g", "bad"])


def _cox_partition(p: int, beta: List[float], ties: str, validate: bool = False):
    """One evaluation over a partition, whose rows arrive sorted by stratum, time descending and
    row identifier. With `validate` (the pass that builds the working set) rows with a missing
    feature are counted first, and a partition that has any returns only that count."""
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        rows = len(cols["t"])
        bad = _missing(cols["x"]) if validate else 0
        if bad:
            yield _partial(pa, 0.0, np.zeros(p), np.zeros((p, p)), rows, 0.0, bad=bad)
            return
        with _one_thread():
            out = kernels.cox_partial(cols["x"], cols["t"], cols["d"], _runs(cols["s"])[0], np.asarray(beta, dtype=np.float64),
                                      ties=ties, weights=cols.get("w"), offset=cols.get("o"), entry=cols.get("e"),
                                      presorted=True)
        yield _partial(pa, out["loglik"], out["score"], out["information"], rows, out["events"])
    return run


def _batch(batch, p: int) -> Dict[str, Any]:
    """One Arrow batch as arrays, for the kernels that take rows in any grouping."""
    out = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in batch.schema.names if name != "x"}
    out["x"] = batch.column("x").flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, p)
    return out


def _tables(pa, tables: np.ndarray, sums: Optional[np.ndarray], bad: int):
    return pa.RecordBatch.from_arrays(
        [pa.array([tables.tobytes()], pa.binary()), pa.array([b"" if sums is None else sums.tobytes()], pa.binary()),
         pa.array([int(bad)], pa.int64())], names=["t", "m", "bad"])


def _time_pass1(p: int, beta: bytes, mu: bytes, shift: float, grid: bytes, efron: bool, has_entry: bool,
                first: bool = False):
    """Pass 1 of the time grid: the partition's batches, one after another, into one set of tables.
    With `first` (the pass that builds the working set, at beta = 0 and without centring) it also
    returns the sum of every feature, from which the driver takes the means, and counts the rows
    with a missing feature; such a batch adds nothing to the tables."""
    def run(batches):
        import pyarrow as pa

        b, m, g = (np.frombuffer(v, dtype=np.float64) for v in (beta, mu, grid))
        tables = timegrid.new_tables(len(g), p, efron, has_entry)
        sums, bad, seen = (np.zeros(p) if first else None), 0, False
        for batch in batches:
            if batch.num_rows == 0:
                continue
            cols, seen = _batch(batch, p), True
            if first:
                missing = _missing(cols["x"])
                if missing:
                    bad += missing
                    continue
                sums += cols["x"].sum(axis=0)
            with _one_thread():
                timegrid.accumulate(tables, g, cols["x"], cols["t"], cols["d"], b, m, shift, efron=efron,
                                    weights=cols.get("w"), offset=cols.get("o"), entry=cols.get("e") if has_entry else None)
        if seen:
            yield _tables(pa, tables, sums, bad)
    return run


def _time_sum():
    """Add several pass-1 results on an executor, when there are too many to bring to the driver."""
    def run(batches):
        import pyarrow as pa

        total, sums, bad = None, None, 0
        for batch in batches:
            for value, extra, count in zip(batch.column("t").to_pylist(), batch.column("m").to_pylist(),
                                           batch.column("bad").to_pylist()):
                total = timegrid.add(total, np.frombuffer(value, dtype=np.float64))
                bad += count
                if extra:
                    sums = np.frombuffer(extra, dtype=np.float64) + (0.0 if sums is None else sums)
        if total is not None:
            yield _tables(pa, total, sums, bad)
    return run


def _time_pass2(p: int, beta: bytes, mu: bytes, shift: float, grid: bytes, cumulative: bytes, tied: bytes, has_entry: bool):
    """Pass 2 of the time grid: X' diag(E) X over the partition's batches."""
    def run(batches):
        import pyarrow as pa

        b, m, g, c, t = (np.frombuffer(v, dtype=np.float64) for v in (beta, mu, grid, cumulative, tied))
        G, rows = np.zeros((p, p)), 0
        for batch in batches:
            if batch.num_rows == 0:
                continue
            cols = _batch(batch, p)
            rows += batch.num_rows
            with _one_thread():
                timegrid.gram(G, g, c, t, cols["x"], cols["t"], cols["d"], b, m, shift, weights=cols.get("w"),
                              offset=cols.get("o"), entry=cols.get("e") if has_entry else None)
        if rows:
            yield pa.RecordBatch.from_arrays([pa.array([G.tobytes()], pa.binary()), pa.array([rows], pa.int64())],
                                             names=["g", "n"])
    return run


def _cox_baseline(p: int, beta: List[float], ties: str):
    """Every event time of every stratum with the increment of the baseline hazard there."""
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        codes, values = _runs(cols["s"])
        with _one_thread():
            out = kernels.cox_partial(cols["x"], cols["t"], cols["d"], codes, np.asarray(beta, dtype=np.float64),
                                      ties=ties, weights=cols.get("w"), offset=cols.get("o"), entry=cols.get("e"),
                                      baseline=True, presorted=True)["baseline"]
        yield pa.RecordBatch.from_arrays(
            [pa.array(np.asarray(values)[out["stratum"]], type=cols["key_type"]), pa.array(out["time"], pa.float64()),
             pa.array(out["increment"], pa.float64())], names=["s", "time", "increment"])
    return run


def _cox_extra(p: int, beta: List[float], ties: str, what: str):
    """After the fit, one pass for the robust variance's middle ("meat") or the martingale residuals ("rows")."""
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        with _one_thread():
            out = kernels.cox_partial(cols["x"], cols["t"], cols["d"], _runs(cols["s"])[0], np.asarray(beta, dtype=np.float64),
                                      ties=ties, weights=cols.get("w"), offset=cols.get("o"), entry=cols.get("e"),
                                      robust=what == "meat", rows=what == "rows", presorted=True)
        if what == "meat":
            yield pa.RecordBatch.from_arrays([pa.array([np.ascontiguousarray(out["meat"]).tobytes()], pa.binary())], names=["m"])
        else:
            yield pa.RecordBatch.from_arrays([pa.array(cols["k"], pa.float64()),
                                              pa.array(cols["d"] - out["expected"], pa.float64())], names=["row_id", "martingale"])
    return run


def _national_tables(p: int, beta: bytes, grid: bytes, has_entry: bool):
    """For the national baseline at fixed coefficients: per distinct time, the risk scores of the rows that exit
    there, of the rows that enter there, and the weighted events."""
    def run(batches):
        import pyarrow as pa

        b, g = np.frombuffer(beta, dtype=np.float64), np.frombuffer(grid, dtype=np.float64)
        K = len(g)
        tables, seen = np.zeros((3, K)), False
        for batch in batches:
            if batch.num_rows == 0:
                continue
            cols, seen = _batch(batch, p), True
            r = np.exp(cols["x"] @ b + (cols["o"] if "o" in cols else 0.0)) * (cols["w"] if "w" in cols else 1.0)
            code = np.searchsorted(g, cols["t"])
            tables[0] += np.bincount(code, weights=r, minlength=K)
            tables[2] += np.bincount(code, weights=cols["d"] * (cols["w"] if "w" in cols else 1.0), minlength=K)
            if has_entry:
                inside = np.isfinite(cols["e"])
                tables[1] += np.bincount(np.searchsorted(g, cols["e"][inside]), weights=r[inside], minlength=K)
        if seen:
            yield pa.RecordBatch.from_arrays([pa.array([tables.tobytes()], pa.binary())], names=["t"])
    return run


def _provider_expected(p: int, beta: bytes, grid: bytes, cumulative: bytes, has_entry: bool):
    """Per provider (stratum): observed events, events expected at the national baseline, and person-time."""
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        b, g, c = (np.frombuffer(v, dtype=np.float64) for v in (beta, grid, cumulative))
        w = cols["w"] if "w" in cols else 1.0
        r = np.exp(cols["x"] @ b + (cols["o"] if "o" in cols else 0.0)) * w
        exposure, start_time = c[np.searchsorted(g, cols["t"])], np.zeros(len(r))
        if has_entry:
            inside = np.isfinite(cols["e"])
            exposure = exposure - np.where(inside, c[np.searchsorted(g, np.where(inside, cols["e"], g[0]))], 0.0)
            start_time = np.where(inside, cols["e"], 0.0)
        codes, values = _runs(cols["s"])
        m = len(values)
        yield pa.RecordBatch.from_arrays(
            [pa.array(np.asarray(values), type=cols["key_type"]),
             pa.array(np.bincount(codes, weights=cols["d"] * w, minlength=m), pa.float64()),
             pa.array(np.bincount(codes, weights=r * exposure, minlength=m), pa.float64()),
             pa.array(np.bincount(codes, weights=cols["t"] - start_time, minlength=m), pa.float64())],
            names=["s", "observed", "expected", "person_time"])
    return run


def _hazard_table(table, by: Optional[str]):
    """Increments to the table pprof_py and R report: cumulative hazard and exp(-hazard), by stratum and time."""
    keys = ([by] if by else []) + ["time"]
    table = table.sort_values(keys).reset_index(drop=True)
    table["hazard"] = table.groupby(by)["increment"].cumsum() if by else table["increment"].cumsum()
    table["survival"] = np.exp(-table["hazard"])
    return table[keys + ["hazard", "survival"]]


def _normal_tests(z: np.ndarray, level: float):
    """Two-sided p-values and flags (1 above, -1 below, 0 neither) for standard normal statistics."""
    from statistics import NormalDist

    critical = NormalDist().inv_cdf(0.5 + level / 2.0)
    p = np.array([math.erfc(abs(v) / math.sqrt(2.0)) for v in z])
    return p, np.where(z > critical, 1, np.where(z < -critical, -1, 0))


def _measures(effects, level: float):
    """Indirect standardization and the score and Wald tests of each provider against the median
    effect, from the per-provider sums. Definitions follow pprof_py (docs/spec/engine.md, section 3)."""
    observed, expected = effects["events"].to_numpy(dtype=float), effects["expected"].to_numpy(dtype=float)
    effects["indirect_ratio"] = observed / expected
    effects["indirect_rate"] = np.clip(effects["indirect_ratio"] * 100.0 * observed.sum() / effects["rows"].sum(), 0.0, 100.0)
    variance = effects.pop("variance_null").to_numpy(dtype=float)
    score = np.where(variance >= 1e-14, (observed - expected) / np.sqrt(np.where(variance >= 1e-14, variance, 1.0)), 0.0)
    wald = (effects["gamma"].to_numpy() - effects.attrs["median"]) / effects["standard_error_case_mix"].to_numpy()
    for name, z in (("score", score), ("wald", wald)):
        effects[name + "_z"] = z
        effects[name + "_p"], effects[name + "_flag"] = _normal_tests(z, level)
    exact = effects.pop("exact_p").to_numpy(dtype=float)
    effects["exact_p"] = exact                       # last, beside its flag
    effects["exact_flag"] = np.where(exact < 1.0 - level, np.sign(observed - expected), 0).astype(int)
    return effects


def _providers(cols):
    """The partition's rows, which arrive sorted by provider (and row identifier), with each
    provider's rows and events and whether it can be solved."""
    c, values = _runs(cols["s"])
    X, y = cols["x"], cols["y"]
    events, rows = np.bincount(c, weights=y, minlength=len(values)), np.bincount(c, minlength=len(values))
    return X, y, c, values, events, rows, (events > 0) & (events < rows)


def _kept(X, y, c, keep):
    """The rows of the providers that can be solved; no copy when that is all of them."""
    return (X, y, c) if keep.all() else (X[keep], y[keep], c[keep])


def _logistic_scan(p: int):
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        X, y, _, _, events, rows, solvable = _providers(cols)
        yield pa.RecordBatch.from_arrays(
            [pa.array([len(y)], pa.int64()), pa.array([len(rows)], pa.int64()),
             pa.array([int(np.sum(events <= 0))], pa.int64()), pa.array([int(np.sum(events >= rows))], pa.int64()),
             pa.array([int(rows[~solvable].sum())], pa.int64()), pa.array([float(y.sum())], pa.float64()),
             pa.array([_missing(X)], pa.int64()), pa.array([X.sum(axis=0).tobytes()], pa.binary())],
            names=["n", "providers", "low", "high", "held", "e", "bad", "m"])
    return run


def _logistic_held(p: int):
    """The rows of the providers that cannot be solved, to be evaluated on the driver at the bound."""
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        X, y, c, values, _, _, solvable = _providers(cols)
        held = ~solvable[c]
        if not held.any():
            return
        X, y, c = X[held], y[held], c[held]
        offsets = pa.array(np.arange(0, (len(y) + 1) * p, p, dtype=np.int32))
        yield pa.RecordBatch.from_arrays(
            [pa.array(values[c], type=cols["key_type"]), pa.ListArray.from_arrays(offsets, pa.array(X.ravel())),
             pa.array(y, pa.float64())], names=["s", "x", "y"])
    return run


def _logistic_partition(p: int, beta: List[float]):
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        X, y, c, _, _, _, solvable = _providers(cols)
        keep = solvable[c]
        total, events = len(y), float(y.sum())
        if not keep.any():
            yield _partial(pa, 0.0, np.zeros(p), np.zeros((p, p)), total, events)
            return
        with _one_thread():
            out = kernels.logistic_partial(*_kept(X, y, c, keep), np.asarray(beta, dtype=np.float64))
        yield _partial(pa, out["loglik"], out["gradient"], out["schur"], total, events, out["gamma"].tobytes())
    return run


def _mid_p(p0: np.ndarray, observed: float) -> float:
    """Two-sided mid-p value of the observed count under the Poisson-binomial distribution of its
    rows' null probabilities: twice the smaller tail, each tail counting half of P(S = observed).
    The recursion adds positive terms only; its cost is quadratic in the provider's rows."""
    pmf = np.zeros(len(p0) + 1)
    pmf[0] = 1.0
    for k, q in enumerate(p0):
        pmf[1:k + 2] = pmf[1:k + 2] * (1.0 - q) + pmf[:k + 1] * q
        pmf[0] *= 1.0 - q
    o = int(round(observed))
    half = 0.5 * pmf[o]
    return float(min(1.0, 2.0 * min(pmf[:o + 1].sum() - half, pmf[o:].sum() - half)))


def _null_sums(X, y, c, beta, median):
    """Per provider (rows sorted by provider), at the median effect: expected events, their variance,
    and the exact two-sided mid-p value of the observed count."""
    p0 = kernels._expit(median + X @ beta)
    start = np.flatnonzero(kernels._starts(c))
    stop = np.concatenate((start[1:], [len(c)]))
    exact = np.array([_mid_p(p0[a:b], y[a:b].sum()) for a, b in zip(start, stop)])
    return np.add.reduceat(p0, start), np.add.reduceat(p0 * (1.0 - p0), start), exact


def _logistic_direct(p: int, beta: bytes, gamma: bytes, chunk: int = 256):
    """Direct standardization: for every provider's effect, the events expected over this partition's
    rows if all of them had that effect. One number per provider; rows times providers evaluations."""
    def run(batches):
        import pyarrow as pa

        b, g = np.frombuffer(beta, dtype=np.float64), np.frombuffer(gamma, dtype=np.float64)
        total, seen = np.zeros(len(g)), False
        for batch in batches:
            if batch.num_rows == 0:
                continue
            seen = True
            xb = _batch(batch, p)["x"] @ b
            with _one_thread():
                for a in range(0, len(xb), chunk):
                    total += kernels._expit(xb[a:a + chunk, None] + g[None, :]).sum(axis=0)
        if seen:
            yield pa.RecordBatch.from_arrays([pa.array([total.tobytes()], pa.binary())], names=["t"])
    return run


def _case_mix(h, xbar, S, mean):
    """pprof_py's variance for comparing a provider with a reference: 1/h + (xbar_j - xbar)' S^-1 (xbar_j - xbar),
    xbar being the mean covariate row. Unlike the plain variance it does not depend on the covariates' origin."""
    centred = xbar - mean
    return 1.0 / h + np.einsum("ij,jk,ik->i", centred, S, centred)


def _logistic_effects(p: int, beta: List[float], inverse: bytes, median: float, mean: bytes):
    """Each solvable provider's intercept with its two standard errors, and its expected events and
    their variance under the median effect."""
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        X, y, c, values, _, _, solvable = _providers(cols)
        keep = solvable[c]
        if not keep.any():
            return
        with _one_thread():
            X, y, c = _kept(X, y, c, keep)
            out = kernels.logistic_partial(X, y, c, np.asarray(beta, dtype=np.float64))
            S = np.frombuffer(inverse, dtype=np.float64).reshape(p, p)
            variance = 1.0 / out["h"] + np.einsum("ij,jk,ik->i", out["xbar"], S, out["xbar"])
            expected, null, exact = _null_sums(X, y, c, np.asarray(beta, dtype=np.float64), median)
        yield pa.RecordBatch.from_arrays(
            [pa.array(values[np.flatnonzero(solvable)], type=cols["key_type"]), pa.array(out["gamma"], pa.float64()),
             pa.array(np.sqrt(variance), pa.float64()),
             pa.array(np.sqrt(_case_mix(out["h"], out["xbar"], S, np.frombuffer(mean, dtype=np.float64))), pa.float64()),
             pa.array(out["rows"].astype(np.int64)), pa.array(out["events"], pa.float64()),
             pa.array(expected, pa.float64()), pa.array(null, pa.float64()), pa.array(exact, pa.float64())],
            names=["s", "gamma", "standard_error", "standard_error_case_mix", "rows", "events", "expected",
                   "variance_null", "exact_p"])
    return run


# ---------------------------------------------------------------- on the driver
def _sum(parts, p: int):
    value = math.fsum(r["ll"] for r in parts)
    vector = np.sum([np.frombuffer(r["u"], dtype=np.float64) for r in parts], axis=0)
    matrix = np.sum([np.frombuffer(r["i"], dtype=np.float64) for r in parts], axis=0).reshape(p, p)
    return value, vector, matrix


def _table(features: Sequence[str], beta: np.ndarray, se: np.ndarray):
    import pandas as pd

    z = beta / se
    return pd.DataFrame({"feature": list(features), "estimate": beta, "standard_error": se, "z": z,
                         "p_value": [math.erfc(abs(v) / math.sqrt(2.0)) for v in z]})


def _refuse(bad: Dict[str, int]) -> None:
    bad = {name: int(count) for name, count in bad.items() if count}
    if bad:
        raise ValueError(f"the data cannot be fitted: rows by problem {bad}")


def _check(frame, conditions: Dict[str, Any]) -> int:
    """The light job: count the rows and refuse the data if any row meets one of the named
    conditions. `frame` holds the columns other than the features, so only those are read."""
    from pyspark.sql import functions as F

    row = frame.agg(F.count("*").alias("n"), *[F.sum(F.when(c, 1).otherwise(0)).alias(name)
                                               for name, c in conditions.items()]).first()
    _refuse({name: row[name] for name in conditions})
    return int(row["n"])


def _built(rows: int, seen: int, missing: int) -> None:
    """After the pass that built the working set: refuse missing feature values, and a table that
    did not give the rows that were validated."""
    _refuse({"missing value": missing})
    if seen != rows:
        raise RuntimeError(f"the input gave {rows} rows when it was validated and {seen} when the working set was "
                           "built: it is not a fixed table (store or cache it first)")


def _clocked(evaluate, seconds: List[float]):
    """`evaluate` with the time of every call appended to `seconds`."""
    def run(*args, **kwargs):
        started = clock.perf_counter()
        try:
            return evaluate(*args, **kwargs)
        finally:
            seconds.append(clock.perf_counter() - started)
    return run


@dataclass
class CoxModel:
    features: List[str]
    coefficients: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    iterations: int
    evaluations: int
    converged: bool
    newton_step_in_se: float
    rows: int
    events: float
    partitions: int
    settings: Dict[str, Any]
    seconds: Dict[str, float] = field(default_factory=dict)
    baseline_hazard: Any = None       # pandas: [stratum,] time, hazard (cumulative, at x = 0), survival; on request
    robust_covariance: Any = None     # the sandwich variance, each row its own cluster; on request
    residuals: Any = None             # pandas: row_id, martingale; on request
    standardized: Any = None          # pandas, per stratum: observed, expected at the national baseline, indirect_ratio
    evaluation_seconds: List[float] = field(default_factory=list)   # each evaluation of the likelihood, in order
    execution: Dict[str, Any] = field(default_factory=dict)         # what produced the digits: versions, libraries, layout

    @property
    def robust_standard_errors(self):
        return None if self.robust_covariance is None else np.sqrt(np.diag(self.robust_covariance))

    @property
    def standard_errors(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance))

    def summary(self):
        return _table(self.features, self.coefficients, self.standard_errors)

    def linear_predictor(self, frame, name: str = "linear_predictor"):
        """`frame` with one more column: the features times the coefficients (no provider effect, no offset)."""
        from pyspark.sql import functions as F

        terms = [F.col(c).cast("double") * float(b) for c, b in zip(self.features, self.coefficients)]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return frame.withColumn(name, total)


@dataclass
class LogisticModel(CoxModel):
    providers: int = 0
    bounded_low: int = 0
    bounded_high: int = 0
    median_gamma: float = float("nan")
    provider_effects: Any = None      # pandas, one row per provider: effect, standard errors, standardized ratio, tests


class CoxPH:
    """Cox proportional hazards regression.

    time, event     columns of the exit time and the event indicator (0 or 1)
    features        covariate columns
    strata          column whose values define the strata, for example the provider; None for a fit
                    without strata, which uses the time grid: two passes per evaluation, and times
                    that take at most `max_times` distinct values
    ties            "breslow" or "efron"
    entry           optional column of entry times; a row is at risk at t when entry < t <= time
    weights, offset optional columns of case weights and of an offset on the linear predictor
    row_id          optional numeric column, different in every row, that fixes the order of the
                    rows and with it the last bits of the result
    tol             the fit stops when the next Newton step is below tol standard errors
    rows_per_partition, partitions
                    with strata the working set has `partitions` partitions, by default the number
                    of rows over `rows_per_partition`, rounded up. Without strata the rows stay in
                    the input's own partitions by default (also partitions="input"): no shuffle, as
                    many tasks per pass as the input has partitions, and results that repeat to
                    rounding. A number, or "rows" for the number of rows over `rows_per_partition`,
                    places them by row identifier in that many partitions and sorts them: a
                    shuffle, and with `row_id` the same bits in every fit.

    `execution` of the result is the record of what produced it (decision D-16): versions, the
    libraries on the driver, the Arrow batch size and the layout.

    `seconds` of the result: "prepare" ends when the working set is built, which the first
    evaluation does (it is in "prepare"); "fit" is the evaluations after it. `evaluation_seconds`
    has every evaluation.
    """

    def __init__(self, time: str, event: str, features: Sequence[str], strata: str, *, ties: str = "breslow",
                 entry: Optional[str] = None, weights: Optional[str] = None, offset: Optional[str] = None,
                 row_id: Optional[str] = None, tol: float = 1e-8, max_iterations: int = 50,
                 rows_per_partition: int = ROWS_PER_PARTITION, partitions: Any = None,
                 max_times: int = MAX_TIMES, baseline: bool = False, robust: bool = False, residuals: bool = False,
                 standardized: bool = False):
        if ties not in ("breslow", "efron"):
            raise ValueError("ties must be 'breslow' or 'efron'")
        if (robust or residuals or standardized) and not strata:
            raise NotImplementedError("without strata the robust variance, residuals and standardized measures are "
                                      "not implemented yet (OPEN_ITEMS.md, OI-20)")
        if (robust or standardized) and ties != "breslow":
            raise NotImplementedError("the robust variance and the standardized measures are implemented for Breslow ties only")
        if residuals and not row_id:
            raise ValueError("residuals need row_id, to say which row each belongs to")
        if isinstance(partitions, str) and (partitions not in ("input", "rows") or strata):
            raise ValueError('partitions must be a number, or "input" or "rows" for a fit without strata')
        self.max_times, self.baseline = max_times, baseline
        self.robust, self.want_residuals, self.want_standardized = robust, residuals, standardized
        self.time, self.event, self.features, self.strata = time, event, list(features), strata
        self.ties, self.entry, self.weights, self.offset, self.row_id = ties, entry, weights, offset, row_id
        self.tol, self.max_iterations = tol, max_iterations
        self.rows_per_partition, self.partitions = rows_per_partition, partitions

    def _columns(self, stratified: bool):
        """The working set's columns; those among them that the light job reads (all but the
        features and the row identifier); and the conditions it refuses, on those columns."""
        from pyspark.sql import functions as F

        scalars = [F.col(self.time).cast("double").alias("t"), F.col(self.event).cast("double").alias("d")]
        if stratified:
            scalars.insert(0, F.col(self.strata).alias("s"))
        for name, alias in ((self.weights, "w"), (self.offset, "o"), (self.entry, "e")):
            if name:
                scalars.append(F.col(name).cast("double").alias(alias))
        columns = scalars + [F.array(*[F.col(c).cast("double") for c in self.features]).alias("x")]
        if self.row_id:
            columns.append(F.col(self.row_id).cast("double").alias("k"))
        conditions = {
            "missing value": F.col("t").isNull() | F.col("d").isNull(),
            "event not 0 or 1": ~F.col("d").isin(0.0, 1.0),
        }
        if stratified:
            conditions["missing stratum"] = F.col("s").isNull()
        if self.entry:
            conditions["entry not before exit"] = F.col("e") >= F.col("t")
        if self.weights:
            conditions["weight missing or not positive"] = F.col("w").isNull() | (F.col("w") <= 0)
        if self.offset:
            conditions["offset missing"] = F.col("o").isNull()
        return columns, scalars, conditions

    def _settings(self, **more):
        return dict({"ties": self.ties, "strata": self.strata, "entry": self.entry, "weights": self.weights,
                     "offset": self.offset, "row_id": self.row_id, "tol_step_in_se": self.tol}, **more)

    def fit(self, frame) -> CoxModel:
        return self._fit_strata(frame) if self.strata else self._fit_time(frame)

    def _grid(self, scalars, conditions: Dict[str, Any], has_entry: bool):
        """The light job without strata: the row count, the refusals and the distinct times, from one
        grouping by time of the columns other than the features. Returns (rows, grid)."""
        from pyspark.sql import functions as F

        names = [f"c{i}" for i in range(len(conditions))]
        times = scalars.select(F.col("t").alias("v"), F.lit(1).alias("n"),
                               *[F.when(c, 1).otherwise(0).alias(name) for name, c in zip(names, conditions.values())])
        if has_entry:
            times = times.unionByName(scalars.where(F.col("e").isNotNull()).select(
                F.col("e").alias("v"), F.lit(0).alias("n"), *[F.lit(0).alias(name) for name in names]))
        groups = times.groupBy("v").agg(F.sum("n").alias("n"), *[F.sum(name).alias(name) for name in names]) \
            .limit(self.max_times + 2).collect()
        _refuse({problem: sum(r[name] for r in groups) for problem, name in zip(conditions, names)})
        values = [r["v"] for r in groups if r["v"] is not None]
        if len(values) > self.max_times:
            raise NotImplementedError(
                f"the times take more than {self.max_times} distinct values; the time grid needs a limited number "
                "(round them, raise max_times, or wait for the sorted layout: OPEN_ITEMS.md)")
        return sum(int(r["n"]) for r in groups), np.sort(np.asarray(values, dtype=np.float64))

    def _fit_time(self, frame) -> CoxModel:
        """Without strata: the time grid. No pass needs the rows in any order, so by default they
        stay where the input has them; on request they are spread over partitions by row identifier
        and sorted by it, so that every fit sums in the same order."""
        from pyspark.sql import functions as F

        shipped = ship_package(frame.sparkSession)
        started = clock.perf_counter()
        p, efron, has_entry = len(self.features), self.ties == "efron", bool(self.entry)
        columns, scalars, conditions = self._columns(stratified=False)
        rows, grid = self._grid(frame.select(*scalars), conditions, has_entry)
        selected = frame.select(*columns)
        as_stored = self.partitions in (None, "input")
        if as_stored:
            work = selected.persist()
            partitions = work.rdd.getNumPartitions()
        else:
            partitions = (max(1, math.ceil(rows / self.rows_per_partition)) if self.partitions == "rows"
                          else int(self.partitions))
            if self.row_id:
                work = selected.repartition(partitions, "k").sortWithinPartitions("k").persist()
            else:
                work = selected.repartition(partitions, F.xxhash64(*selected.columns)).persist()
        try:
            K = len(grid)
            where = timegrid.layout(K, p, efron, has_entry)
            size = 8 * where["size"]
            state, seconds = {"shift": 0.0, "mu": None}, []

            def evaluate(beta):
                vector, shift, first = np.asarray(beta, dtype=np.float64).tobytes(), state["shift"], state["mu"] is None
                if first and np.any(beta):
                    raise RuntimeError("the pass that builds the working set must be the evaluation at zero")
                mu = np.zeros(p) if first else state["mu"]
                tables = work.mapInArrow(_time_pass1(p, vector, mu.tobytes(), shift, grid.tobytes(), efron, has_entry, first),
                                         TABLES)
                if partitions * size > DRIVER_TABLES:          # too much for the driver at once: add on the executors
                    tables = tables.repartition(max(1, partitions * size // (128 * 2 ** 20))).mapInArrow(_time_sum(), TABLES)
                total, sums, missing = None, np.zeros(p), 0
                for r in tables.collect():
                    total = timegrid.add(total, np.frombuffer(r["t"], dtype=np.float64))
                    missing += r["bad"]
                    if first and r["m"]:
                        sums += np.frombuffer(r["m"], dtype=np.float64)
                if first:
                    _built(rows, 0 if total is None else int(round(total[where["rows"]][0])), missing)
                    mu = state["mu"] = sums / rows
                    timegrid.centre(total, K, p, mu, efron=efron, entry=has_entry)
                out = timegrid.reduce(total, K, p, efron=efron, entry=has_entry)
                if out["rows"] != rows:
                    raise RuntimeError("a pass did not see every row")
                parts = work.mapInArrow(
                    _time_pass2(p, vector, mu.tobytes(), shift, grid.tobytes(), out["cumulative"].tobytes(),
                                out["tied"].tobytes(), has_entry), "g binary, n long").collect()
                if sum(r["n"] for r in parts) != rows:
                    raise RuntimeError("a pass did not see every row")
                G = np.sum([np.frombuffer(r["g"], dtype=np.float64) for r in parts], axis=0).reshape(p, p)
                state.update(shift=out["eta_max"], events=out["events"], last=(out["cumulative"], shift))
                return out["loglik"], out["score"], G - out["M"]

            evaluate = _clocked(evaluate, seconds)
            first = evaluate(np.zeros(p))                     # builds the working set
            prepared = clock.perf_counter()
            fit = maximize(evaluate, p, self.tol, self.max_iterations, start=first)
            done = clock.perf_counter()
        finally:
            work.unpersist()
        mu, hazard = state["mu"], None
        placed = "as in the input" if as_stored else ("by row_id" if self.row_id else "by a hash of the row")
        if self.baseline:
            import pandas as pd

            cumulative, shift = state["last"]             # from the last evaluation, which was at the estimate
            increment = np.diff(np.concatenate(([0.0], cumulative))) * math.exp(-(float(mu @ fit["beta"]) + shift))
            at = increment > 0
            hazard = _hazard_table(pd.DataFrame({"time": grid[at], "increment": increment[at]}), None)
        return CoxModel(self.features, fit["beta"], fit["covariance"], fit["loglik"], fit["iterations"],
                        fit["evaluations"], fit["converged"], fit["newton_step_in_se"], rows, state["events"], partitions,
                        self._settings(layout="time grid", distinct_times=K, passes_per_evaluation=2,
                                       rows_per_partition=self.rows_per_partition if self.partitions == "rows" else None,
                                       rows_placed=placed),
                        {"prepare": prepared - started, "fit": done - prepared}, hazard, evaluation_seconds=seconds,
                        execution=execution_record(frame.sparkSession, layout="time grid", partitions=partitions,
                                                   rows_placed=placed, row_id=self.row_id, code_sent=shipped))

    def _fit_strata(self, frame) -> CoxModel:
        from pyspark.sql import functions as F

        spark = frame.sparkSession
        shipped = ship_package(spark)
        started = clock.perf_counter()
        p = len(self.features)
        columns, scalars, conditions = self._columns(stratified=True)
        rows = _check(frame.select(*scalars), conditions)
        partitions = self.partitions or max(1, math.ceil(rows / self.rows_per_partition))
        order = ["s", F.desc("t")] + (["k"] if self.row_id else [])        # the kernel's order, computed once
        work = frame.select(*columns).repartition(partitions, "s").sortWithinPartitions(*order).persist()
        try:
            state, seconds = {}, []

            def evaluate(beta, validate=False):
                parts = work.mapInArrow(_cox_partition(p, [float(b) for b in beta], self.ties, validate), PARTIAL).collect()
                seen = sum(r["n"] for r in parts)
                if validate:
                    _built(rows, seen, sum(r["bad"] for r in parts))
                if seen != rows:
                    raise RuntimeError("a pass did not see every row")
                state["events"] = math.fsum(r["e"] for r in parts)
                return _sum(parts, p)

            evaluate = _clocked(evaluate, seconds)
            first = evaluate(np.zeros(p), validate=True)      # builds the working set
            prepared = clock.perf_counter()
            fit = maximize(evaluate, p, self.tol, self.max_iterations, start=first)
            done = clock.perf_counter()
            hazard = None
            if self.baseline:
                key_type = frame.schema[self.strata].dataType.simpleString()
                hazard = _hazard_table(work.mapInArrow(
                    _cox_baseline(p, [float(b) for b in fit["beta"]], self.ties),
                    f"s {key_type}, time double, increment double").toPandas(), "s").rename(columns={"s": self.strata})
            extra, vector = {}, [float(b) for b in fit["beta"]]
            if self.robust:
                parts = work.mapInArrow(_cox_extra(p, vector, self.ties, "meat"), "m binary").collect()
                meat = np.sum([np.frombuffer(r["m"], dtype=np.float64) for r in parts], axis=0).reshape(p, p)
                extra["robust_covariance"] = fit["covariance"] @ meat @ fit["covariance"]
            if self.want_residuals:
                extra["residuals"] = work.mapInArrow(_cox_extra(p, vector, self.ties, "rows"),
                                                    "row_id double, martingale double").toPandas().sort_values("row_id").reset_index(drop=True)
            if self.want_standardized:
                extra["standardized"] = self._standardized(work, frame, p, np.asarray(fit["beta"], dtype=np.float64))
        finally:
            work.unpersist()
        return CoxModel(self.features, fit["beta"], fit["covariance"], fit["loglik"], fit["iterations"], fit["evaluations"], fit["converged"],
                        fit["newton_step_in_se"], rows, state["events"], partitions,
                        self._settings(layout="whole strata", rows_per_partition=self.rows_per_partition,
                                       passes_per_evaluation=1),
                        {"prepare": prepared - started, "fit": done - prepared}, hazard, evaluation_seconds=seconds,
                        execution=execution_record(spark, layout="whole strata", partitions=partitions,
                                                   rows_per_partition=self.rows_per_partition, row_id=self.row_id,
                                                   code_sent=shipped), **extra)

    def _standardized(self, work, frame, p: int, beta: np.ndarray):
        """He and Schaubel's two-stage measure, as pprof_py computes it for a provider-stratified fit: the national
        baseline hazard is Breslow's over the whole table at the stratified estimate, and a provider's expected
        events are its rows' risk scores times that hazard over their time at risk."""
        from pyspark.sql import functions as F

        has_entry = bool(self.entry)
        times = work.select(F.col("t").alias("v"))
        if has_entry:
            times = times.union(work.where(F.col("e").isNotNull()).select(F.col("e").alias("v")))
        values = [r["v"] for r in times.distinct().limit(self.max_times + 1).collect()]
        if len(values) > self.max_times:
            raise NotImplementedError(f"the times take more than {self.max_times} distinct values")
        grid = np.sort(np.asarray(values, dtype=np.float64))
        parts = work.mapInArrow(_national_tables(p, beta.tobytes(), grid.tobytes(), has_entry), "t binary").collect()
        exits, entries, events = np.sum([np.frombuffer(r["t"], dtype=np.float64) for r in parts], axis=0).reshape(3, len(grid))
        at_risk = timegrid._suffix(exits) - timegrid._suffix(entries)
        hazard = np.where(events > 0, events / np.where(at_risk > 0, at_risk, 1.0), 0.0)
        key_type = frame.schema[self.strata].dataType.simpleString()
        table = work.mapInArrow(
            _provider_expected(p, beta.tobytes(), grid.tobytes(), np.cumsum(hazard).tobytes(), has_entry),
            f"s {key_type}, observed double, expected double, person_time double").toPandas()
        table = table.sort_values("s").reset_index(drop=True).rename(columns={"s": self.strata})
        table.insert(3, "indirect_ratio", table["observed"] / table["expected"])
        return table


class LogisticFE:
    """Logistic regression with one fixed intercept per provider.

    Providers without events, or with only events, are held at the median intercept minus or plus
    `bound`: the estimate pprof_py's SerBIN converges to. Their rows are evaluated on the driver,
    so they must be few (at most MAX_HELD_ROWS rows).

    `seconds` of the result: "prepare" ends when the working set is built and scanned, "fit" is
    Newton's method, "provider_effects" the pass after it.
    """

    def __init__(self, outcome: str, features: Sequence[str], provider: str, *, bound: float = 10.0,
                 row_id: Optional[str] = None, tol: float = 1e-8, max_iterations: int = 50,
                 provider_effects: bool = True, level: float = 0.95, direct: bool = False,
                 rows_per_partition: int = ROWS_PER_PARTITION, partitions: Optional[int] = None):
        self.level, self.direct = level, direct
        self.outcome, self.features, self.provider = outcome, list(features), provider
        self.bound, self.row_id, self.tol, self.max_iterations = bound, row_id, tol, max_iterations
        self.provider_effects = provider_effects
        self.rows_per_partition, self.partitions = rows_per_partition, partitions

    def fit(self, frame) -> LogisticModel:
        import pandas as pd
        from pyspark.sql import functions as F

        spark = frame.sparkSession
        shipped = ship_package(spark)
        started = clock.perf_counter()
        p = len(self.features)
        key_type = frame.schema[self.provider].dataType.simpleString()
        scalars = [F.col(self.provider).alias("s"), F.col(self.outcome).cast("double").alias("y")]
        columns = scalars + [F.array(*[F.col(c).cast("double") for c in self.features]).alias("x")]
        if self.row_id:
            columns.append(F.col(self.row_id).cast("double").alias("k"))
        rows = _check(frame.select(*scalars), {"missing value": F.col("s").isNull() | F.col("y").isNull(),
                                               "outcome not 0 or 1": ~F.col("y").isin(0.0, 1.0)})
        partitions = self.partitions or max(1, math.ceil(rows / self.rows_per_partition))
        keys = ["s"] + (["k"] if self.row_id else [])                      # the kernel's order, computed once
        work = frame.select(*columns).repartition(partitions, "s").sortWithinPartitions(*keys).persist()
        try:
            scan = work.mapInArrow(_logistic_scan(p), SCAN).collect()      # builds the working set
            _built(rows, sum(r["n"] for r in scan), sum(r["bad"] for r in scan))
            mean = np.sum([np.frombuffer(r["m"], dtype=np.float64) for r in scan], axis=0) / rows
            providers, low, high = (sum(r[k] for r in scan) for k in ("providers", "low", "high"))
            held_rows = sum(r["held"] for r in scan)
            Xd = yd = cd = held_ids = held_low = None
            if held_rows:
                if held_rows > MAX_HELD_ROWS:
                    raise NotImplementedError(f"{held_rows} rows belong to providers without events or with only "
                                              f"events; at most {MAX_HELD_ROWS} are handled (OPEN_ITEMS.md)")
                held = work.mapInArrow(_logistic_held(p), f"s {key_type}, x array<double>, y double").toPandas()
                held_ids, cd = np.unique(held["s"].to_numpy(), return_inverse=True)
                order = np.argsort(cd, kind="stable")
                Xd = np.vstack(held["x"].to_numpy())[order]
                yd, cd = held["y"].to_numpy(dtype=float)[order], cd[order]
                held_low = np.bincount(cd, weights=yd, minlength=len(held_ids)) <= 0
            prepared = clock.perf_counter()
            state, evaluation_seconds = {}, []

            def evaluate(beta):
                parts = work.mapInArrow(_logistic_partition(p, [float(b) for b in beta]), PARTIAL).collect()
                if sum(r["n"] for r in parts) != rows:
                    raise RuntimeError("a pass did not see every row")
                value, gradient, schur = _sum(parts, p)
                free = np.concatenate([np.frombuffer(r["g"], dtype=np.float64) for r in parts])
                bounds = bounded_gamma(free, int(low), int(high), self.bound)
                if held_rows:
                    gamma = np.where(held_low, bounds["lower"], bounds["upper"])
                    part = kernels.logistic_partial(Xd, yd, cd, beta, gamma=gamma)
                    value, gradient, schur = value + part["loglik"], gradient + part["gradient"], schur + part["schur"]
                    state["held"] = (gamma, part)
                state.update(median=bounds["median"], events=math.fsum(r["e"] for r in parts))
                return value, gradient, schur

            fit = maximize(_clocked(evaluate, evaluation_seconds), p, self.tol, self.max_iterations)
            fitted = clock.perf_counter()
            effects = None
            if self.provider_effects:
                inverse = np.ascontiguousarray(fit["covariance"]).tobytes()
                effects = work.mapInArrow(
                    _logistic_effects(p, [float(b) for b in fit["beta"]], inverse, state["median"], mean.tobytes()),
                    f"s {key_type}, gamma double, standard_error double, standard_error_case_mix double, rows long, "
                    "events double, expected double, variance_null double, exact_p double").toPandas()
                effects["bounded"] = False
                if held_rows:
                    gamma, part = state["held"]
                    variance = 1.0 / part["h"] + np.einsum("ij,jk,ik->i", part["xbar"], fit["covariance"], part["xbar"])
                    expected, null, exact = _null_sums(Xd, yd, cd, fit["beta"], state["median"])
                    effects = pd.concat([effects, pd.DataFrame({
                        "s": held_ids, "gamma": gamma, "standard_error": np.sqrt(variance),
                        "standard_error_case_mix": np.sqrt(_case_mix(part["h"], part["xbar"], fit["covariance"], mean)),
                        "rows": part["rows"].astype(np.int64), "events": part["events"], "expected": expected,
                        "variance_null": null, "exact_p": exact, "bounded": True})], ignore_index=True)
                effects.attrs["median"] = state["median"]
                effects = _measures(effects, self.level)
                effects = effects.rename(columns={"s": self.provider}).sort_values(self.provider).reset_index(drop=True)
                if self.direct:
                    parts = work.mapInArrow(
                        _logistic_direct(p, np.asarray(fit["beta"], dtype=np.float64).tobytes(),
                                         effects["gamma"].to_numpy(dtype=np.float64).tobytes()), "t binary").collect()
                    total = np.sum([np.frombuffer(r["t"], dtype=np.float64) for r in parts], axis=0)
                    effects["direct_expected"] = total
                    effects["direct_ratio"] = total / effects["events"].sum()
                    effects["direct_rate"] = 100.0 * total / rows
            done = clock.perf_counter()
        finally:
            work.unpersist()
        seconds = {"prepare": prepared - started, "fit": fitted - prepared}
        if self.provider_effects:
            seconds["provider_effects"] = done - fitted
        return LogisticModel(
            self.features, fit["beta"], fit["covariance"], fit["loglik"], fit["iterations"], fit["evaluations"],
            fit["converged"], fit["newton_step_in_se"], rows, state["events"], partitions,
            {"bound": self.bound, "provider": self.provider, "row_id": self.row_id, "tol_step_in_se": self.tol,
             "rows_per_partition": self.rows_per_partition},
            seconds, evaluation_seconds=evaluation_seconds,
            execution=execution_record(spark, layout="whole providers", partitions=partitions,
                                       rows_per_partition=self.rows_per_partition, row_id=self.row_id, code_sent=shipped),
            providers=int(providers), bounded_low=int(low),
            bounded_high=int(high), median_gamma=state["median"], provider_effects=effects)
