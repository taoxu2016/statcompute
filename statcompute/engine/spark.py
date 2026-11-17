"""The engine's two estimators on Spark.

Layout. The rows are shuffled once so that every partition holds whole strata (Cox) or whole
providers (logistic), with the features packed into one array column, and that table is cached.
The number of partitions depends only on the number of rows and `rows_per_partition`, never on the
size of the cluster, so the same data give the same partitions on 2 executors and on 10.

Passes. Every evaluation of the likelihood is one Spark job: each partition's rows go to a Python
worker as Arrow batches, a NumPy kernel returns that partition's sums, and the driver adds them in
partition order. Newton's method runs on the driver. The kernels use memory outside the JVM heap.
"""
from __future__ import annotations

import contextlib
import math
import time as clock
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import kernels, timegrid
from .dist import ship_package
from .local import bounded_gamma
from .newton import maximize

ROWS_PER_PARTITION = 50_000
MAX_HELD_ROWS = 200_000
MAX_TIMES = 20_000
PARTIAL = "ll double, u binary, i binary, n long, e double, g binary"
SCAN = "n long, providers long, low long, high long, held long, e double"


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


def _partial(pa, loglik, vector, matrix, rows, events, gamma=b""):
    return pa.RecordBatch.from_arrays(
        [pa.array([float(loglik)], pa.float64()), pa.array([vector.tobytes()], pa.binary()),
         pa.array([np.ascontiguousarray(matrix).tobytes()], pa.binary()), pa.array([int(rows)], pa.int64()),
         pa.array([float(events)], pa.float64()), pa.array([gamma], pa.binary())],
        names=["ll", "u", "i", "n", "e", "g"])


def _cox_partition(p: int, beta: List[float], ties: str):
    def run(batches):
        import pandas as pd
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        codes = pd.factorize(cols["s"], sort=True)[0]
        with _one_thread():
            out = kernels.cox_partial(cols["x"], cols["t"], cols["d"], codes, np.asarray(beta, dtype=np.float64),
                                      ties=ties, weights=cols.get("w"), offset=cols.get("o"), entry=cols.get("e"),
                                      order=cols.get("k"))
        yield _partial(pa, out["loglik"], out["score"], out["information"], len(codes), out["events"])
    return run


def _batch(batch, p: int) -> Dict[str, Any]:
    """One Arrow batch as arrays, for the kernels that take rows in any grouping."""
    out = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in batch.schema.names if name != "x"}
    out["x"] = batch.column("x").flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, p)
    return out


def _time_pass1(p: int, beta: bytes, mu: bytes, shift: float, grid: bytes, efron: bool, has_entry: bool):
    """Pass 1 of the time grid: the partition's batches, one after another, into one set of tables."""
    def run(batches):
        import pyarrow as pa

        b, m, g = (np.frombuffer(v, dtype=np.float64) for v in (beta, mu, grid))
        tables = timegrid.new_tables(len(g), p, efron, has_entry)
        seen = False
        for batch in batches:
            if batch.num_rows == 0:
                continue
            cols, seen = _batch(batch, p), True
            with _one_thread():
                timegrid.accumulate(tables, g, cols["x"], cols["t"], cols["d"], b, m, shift, efron=efron,
                                    weights=cols.get("w"), offset=cols.get("o"), entry=cols.get("e") if has_entry else None)
        if seen:
            yield pa.RecordBatch.from_arrays([pa.array([tables.tobytes()], pa.binary())], names=["t"])
    return run


def _time_sum():
    """Add several pass-1 results on an executor, when there are too many to bring to the driver."""
    def run(batches):
        import pyarrow as pa

        total = None
        for batch in batches:
            for value in batch.column("t").to_pylist():
                total = timegrid.add(total, np.frombuffer(value, dtype=np.float64))
        if total is not None:
            yield pa.RecordBatch.from_arrays([pa.array([total.tobytes()], pa.binary())], names=["t"])
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


def _providers(cols):
    """Rows sorted by provider, with each provider's rows and events and whether it can be solved."""
    import pandas as pd

    codes, values = pd.factorize(cols["s"], sort=True)
    order = np.lexsort((cols["k"], codes)) if "k" in cols else np.argsort(codes, kind="stable")
    X, y, c = cols["x"][order], cols["y"][order], codes[order]
    events, rows = np.bincount(c, weights=y, minlength=len(values)), np.bincount(c, minlength=len(values))
    return X, y, c, np.asarray(values), events, rows, (events > 0) & (events < rows)


def _logistic_scan(p: int):
    def run(batches):
        import pyarrow as pa

        cols = _stack(batches, p)
        if cols is None:
            return
        _, y, _, _, events, rows, solvable = _providers(cols)
        yield pa.RecordBatch.from_arrays(
            [pa.array([len(y)], pa.int64()), pa.array([len(rows)], pa.int64()),
             pa.array([int(np.sum(events <= 0))], pa.int64()), pa.array([int(np.sum(events >= rows))], pa.int64()),
             pa.array([int(rows[~solvable].sum())], pa.int64()), pa.array([float(y.sum())], pa.float64())],
            names=["n", "providers", "low", "high", "held", "e"])
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
            out = kernels.logistic_partial(X[keep], y[keep], c[keep], np.asarray(beta, dtype=np.float64))
        yield _partial(pa, out["loglik"], out["gradient"], out["schur"], total, events, out["gamma"].tobytes())
    return run


def _logistic_effects(p: int, beta: List[float], inverse: bytes):
    """Each solvable provider's intercept and its standard error, 1/h + xbar' S^-1 xbar."""
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
            out = kernels.logistic_partial(X[keep], y[keep], c[keep], np.asarray(beta, dtype=np.float64))
            S = np.frombuffer(inverse, dtype=np.float64).reshape(p, p)
            variance = 1.0 / out["h"] + np.einsum("ij,jk,ik->i", out["xbar"], S, out["xbar"])
        yield pa.RecordBatch.from_arrays(
            [pa.array(values[np.flatnonzero(solvable)], type=cols["key_type"]), pa.array(out["gamma"], pa.float64()),
             pa.array(np.sqrt(variance), pa.float64()), pa.array(out["rows"].astype(np.int64)),
             pa.array(out["events"], pa.float64())], names=["s", "gamma", "standard_error", "rows", "events"])
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


def _check(frame, conditions: Dict[str, Any], extra: Sequence[Any] = ()):
    """Count the rows and refuse the data if any row meets one of the named conditions. The same
    job materializes the cached table and computes any `extra` aggregates. Returns (rows, row)."""
    from pyspark.sql import functions as F

    row = frame.agg(F.count("*").alias("n"), *[F.sum(F.when(c, 1).otherwise(0)).alias(name)
                                               for name, c in conditions.items()], *extra).first()
    bad = {name: int(row[name]) for name in conditions if row[name]}
    if bad:
        raise ValueError(f"the data cannot be fitted: rows by problem {bad}")
    return int(row["n"]), row


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

    @property
    def standard_errors(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance))

    def summary(self):
        return _table(self.features, self.coefficients, self.standard_errors)


@dataclass
class LogisticModel(CoxModel):
    providers: int = 0
    bounded_low: int = 0
    bounded_high: int = 0
    median_gamma: float = float("nan")
    provider_effects: Any = None      # pandas: provider, gamma, standard_error, rows, events, bounded


class CoxPH:
    """Cox proportional hazards regression.

    time, event     columns of the exit time and the event indicator (0 or 1)
    features        covariate columns
    strata          column whose values define the strata, for example the provider; None for a fit
                    without strata, which uses the time grid: no shuffle, two passes per evaluation,
                    and times that take at most `max_times` distinct values
    ties            "breslow" or "efron"
    entry           optional column of entry times; a row is at risk at t when entry < t <= time
    weights, offset optional columns of case weights and of an offset on the linear predictor
    row_id          optional numeric column that fixes the order of tied rows, and with it the last
                    bits of the result
    tol             the fit stops when the next Newton step is below tol standard errors
    """

    def __init__(self, time: str, event: str, features: Sequence[str], strata: str, *, ties: str = "breslow",
                 entry: Optional[str] = None, weights: Optional[str] = None, offset: Optional[str] = None,
                 row_id: Optional[str] = None, tol: float = 1e-8, max_iterations: int = 50,
                 rows_per_partition: int = ROWS_PER_PARTITION, partitions: Optional[int] = None,
                 max_times: int = MAX_TIMES):
        if ties not in ("breslow", "efron"):
            raise ValueError("ties must be 'breslow' or 'efron'")
        self.max_times = max_times
        self.time, self.event, self.features, self.strata = time, event, list(features), strata
        self.ties, self.entry, self.weights, self.offset, self.row_id = ties, entry, weights, offset, row_id
        self.tol, self.max_iterations = tol, max_iterations
        self.rows_per_partition, self.partitions = rows_per_partition, partitions

    def _columns(self, stratified: bool):
        from pyspark.sql import functions as F

        columns = [F.array(*[F.col(c).cast("double") for c in self.features]).alias("x"),
                   F.col(self.time).cast("double").alias("t"), F.col(self.event).cast("double").alias("d")]
        if stratified:
            columns.insert(0, F.col(self.strata).alias("s"))
        for name, alias in ((self.weights, "w"), (self.offset, "o"), (self.entry, "e"),
                            (self.row_id if stratified else None, "k")):
            if name:
                columns.append(F.col(name).cast("double").alias(alias))
        conditions = {
            "missing value": F.col("t").isNull() | F.col("d").isNull() | F.exists("x", lambda v: v.isNull() | F.isnan(v)),
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
        return columns, conditions

    def _settings(self, **more):
        return dict({"ties": self.ties, "strata": self.strata, "entry": self.entry, "weights": self.weights,
                     "offset": self.offset, "row_id": self.row_id, "tol_step_in_se": self.tol}, **more)

    def fit(self, frame) -> CoxModel:
        return self._fit_strata(frame) if self.strata else self._fit_time(frame)

    def _fit_time(self, frame) -> CoxModel:
        """Without strata: the time grid. The rows stay where they are; nothing is shuffled or sorted."""
        from pyspark.sql import functions as F

        ship_package(frame.sparkSession)
        started = clock.perf_counter()
        p, efron, has_entry = len(self.features), self.ties == "efron", bool(self.entry)
        columns, conditions = self._columns(stratified=False)
        selected = frame.select(*columns)
        work = (selected.repartition(self.partitions) if self.partitions else selected).persist()
        try:
            rows, row = _check(work, conditions, [F.avg(F.col("x")[i]).alias(f"m{i}") for i in range(p)])
            mu = np.array([row[f"m{i}"] for i in range(p)], dtype=np.float64)
            times = work.select(F.col("t").alias("v"))
            if has_entry:
                times = times.union(work.where(F.col("e").isNotNull()).select(F.col("e").alias("v")))
            values = [r["v"] for r in times.distinct().limit(self.max_times + 1).collect()]
            if len(values) > self.max_times:
                raise NotImplementedError(
                    f"the times take more than {self.max_times} distinct values; the time grid needs a limited number "
                    "(round them, raise max_times, or wait for the sorted layout: OPEN_ITEMS.md)")
            grid = np.sort(np.asarray(values, dtype=np.float64))
            K, partitions = len(grid), work.rdd.getNumPartitions()
            size = 8 * timegrid.layout(K, p, efron, has_entry)["size"]
            prepared = clock.perf_counter()
            state = {"shift": 0.0}

            def evaluate(beta):
                vector, shift = np.asarray(beta, dtype=np.float64).tobytes(), state["shift"]
                tables = work.mapInArrow(_time_pass1(p, vector, mu.tobytes(), shift, grid.tobytes(), efron, has_entry),
                                         "t binary")
                if partitions * size > 768 * 2 ** 20:          # too much for the driver at once: add on the executors
                    tables = tables.repartition(max(1, partitions * size // (128 * 2 ** 20))).mapInArrow(_time_sum(), "t binary")
                total = None
                for r in tables.collect():
                    total = timegrid.add(total, np.frombuffer(r["t"], dtype=np.float64))
                out = timegrid.reduce(total, K, p, efron=efron, entry=has_entry)
                if out["rows"] != rows:
                    raise RuntimeError("a pass did not see every row")
                parts = work.mapInArrow(
                    _time_pass2(p, vector, mu.tobytes(), shift, grid.tobytes(), out["cumulative"].tobytes(),
                                out["tied"].tobytes(), has_entry), "g binary, n long").collect()
                if sum(r["n"] for r in parts) != rows:
                    raise RuntimeError("a pass did not see every row")
                G = np.sum([np.frombuffer(r["g"], dtype=np.float64) for r in parts], axis=0).reshape(p, p)
                state.update(shift=out["eta_max"], events=out["events"])
                return out["loglik"], out["score"], G - out["M"]

            fit = maximize(evaluate, p, self.tol, self.max_iterations)
            done = clock.perf_counter()
        finally:
            work.unpersist()
        return CoxModel(self.features, fit["beta"], fit["covariance"], fit["loglik"], fit["iterations"],
                        fit["evaluations"], fit["converged"], fit["newton_step_in_se"], rows, state["events"], partitions,
                        self._settings(layout="time grid", distinct_times=K, passes_per_evaluation=2),
                        {"prepare": prepared - started, "fit": done - prepared})

    def _fit_strata(self, frame) -> CoxModel:
        spark = frame.sparkSession
        ship_package(spark)
        started = clock.perf_counter()
        p = len(self.features)
        columns, conditions = self._columns(stratified=True)
        partitions = self.partitions or max(1, math.ceil(frame.count() / self.rows_per_partition))
        work = frame.select(*columns).repartition(partitions, "s").persist()
        try:
            rows, _ = _check(work, conditions)
            prepared = clock.perf_counter()
            state = {}

            def evaluate(beta):
                parts = work.mapInArrow(_cox_partition(p, [float(b) for b in beta], self.ties), PARTIAL).collect()
                if sum(r["n"] for r in parts) != rows:
                    raise RuntimeError("a pass did not see every row")
                state["events"] = math.fsum(r["e"] for r in parts)
                return _sum(parts, p)

            fit = maximize(evaluate, p, self.tol, self.max_iterations)
            done = clock.perf_counter()
        finally:
            work.unpersist()
        return CoxModel(self.features, fit["beta"], fit["covariance"], fit["loglik"], fit["iterations"], fit["evaluations"], fit["converged"],
                        fit["newton_step_in_se"], rows, state["events"], partitions,
                        self._settings(layout="whole strata", rows_per_partition=self.rows_per_partition,
                                       passes_per_evaluation=1),
                        {"prepare": prepared - started, "fit": done - prepared})


class LogisticFE:
    """Logistic regression with one fixed intercept per provider.

    Providers without events, or with only events, are held at the median intercept minus or plus
    `bound`: the estimate pprof_py's SerBIN converges to. Their rows are evaluated on the driver,
    so they must be few (at most MAX_HELD_ROWS rows).
    """

    def __init__(self, outcome: str, features: Sequence[str], provider: str, *, bound: float = 10.0,
                 row_id: Optional[str] = None, tol: float = 1e-8, max_iterations: int = 50,
                 provider_effects: bool = True, rows_per_partition: int = ROWS_PER_PARTITION,
                 partitions: Optional[int] = None):
        self.outcome, self.features, self.provider = outcome, list(features), provider
        self.bound, self.row_id, self.tol, self.max_iterations = bound, row_id, tol, max_iterations
        self.provider_effects = provider_effects
        self.rows_per_partition, self.partitions = rows_per_partition, partitions

    def fit(self, frame) -> LogisticModel:
        import pandas as pd
        from pyspark.sql import functions as F

        spark = frame.sparkSession
        ship_package(spark)
        started = clock.perf_counter()
        p = len(self.features)
        key_type = frame.schema[self.provider].dataType.simpleString()
        columns = [F.col(self.provider).alias("s"),
                   F.array(*[F.col(c).cast("double") for c in self.features]).alias("x"),
                   F.col(self.outcome).cast("double").alias("y")]
        if self.row_id:
            columns.append(F.col(self.row_id).cast("double").alias("k"))
        partitions = self.partitions or max(1, math.ceil(frame.count() / self.rows_per_partition))
        work = frame.select(*columns).repartition(partitions, "s").persist()
        try:
            rows, _ = _check(work, {
                "missing value": F.col("s").isNull() | F.col("y").isNull() | F.exists("x", lambda v: v.isNull() | F.isnan(v)),
                "outcome not 0 or 1": ~F.col("y").isin(0.0, 1.0)})
            scan = work.mapInArrow(_logistic_scan(p), SCAN).collect()
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
            state = {}

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

            fit = maximize(evaluate, p, self.tol, self.max_iterations)
            fitted = clock.perf_counter()
            effects = None
            if self.provider_effects:
                inverse = np.ascontiguousarray(fit["covariance"]).tobytes()
                effects = work.mapInArrow(
                    _logistic_effects(p, [float(b) for b in fit["beta"]], inverse),
                    f"s {key_type}, gamma double, standard_error double, rows long, events double").toPandas()
                effects["bounded"] = False
                if held_rows:
                    gamma, part = state["held"]
                    variance = 1.0 / part["h"] + np.einsum("ij,jk,ik->i", part["xbar"], fit["covariance"], part["xbar"])
                    effects = pd.concat([effects, pd.DataFrame({
                        "s": held_ids, "gamma": gamma, "standard_error": np.sqrt(variance),
                        "rows": part["rows"].astype(np.int64), "events": part["events"], "bounded": True})],
                        ignore_index=True)
                effects = effects.rename(columns={"s": self.provider}).sort_values(self.provider).reset_index(drop=True)
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
            seconds, providers=int(providers), bounded_low=int(low), bounded_high=int(high),
            median_gamma=state["median"], provider_effects=effects)
