"""Adapters for the lab prototypes.

lab_numpy   one machine: the kernels of `kernels.py` over blocks of whole groups, Newton on top
lab_spark   the same kernels inside Spark partitions that hold whole providers (mapInArrow), with
            the partition sums added on the driver: one pass over the data per evaluation

Both are prototypes of the algorithms the roadmap recommends (the weighted-Gram form of the Cox
information; profile-likelihood Newton for the logistic model). They are not the engine: they use
the BLAS under NumPy, so their results are not bitwise reproducible across machines.
"""
from __future__ import annotations

import math
import time as clock
from typing import Dict, List

import numpy as np

from .. import __version__
from ..adapters import EXACT, UNSUPPORTED, Adapter, FitResult, Support, register
from ..testkit.localdata import footprint_bytes, load_arrays
from ..testkit.specs import COX
from . import kernels
from .dist import ship_package
from .newton import maximize

PARTIAL_SCHEMA = "ll double, u binary, i binary, n long"


def _tolerance(model) -> float:
    return 1e-10 if model.convergence == "tight" else 1e-8


def _support(model, spark_layout: bool) -> Support:
    if model.family == COX and model.ties != "breslow":
        return Support(UNSUPPORTED, ["the prototype implements Breslow ties only"])
    if model.family == COX and spark_layout and model.strata is None:
        return Support(UNSUPPORTED, ["an unstratified fit needs the per-time layout (roadmap O-3), "
                                     "which is not prototyped: one stratum cannot span partitions here"])
    return Support(EXACT, ["stops when the next Newton step is below the tolerance in standard errors (T_CONV)"])


def _result(names: List[str], fit: Dict[str, object], tol: float, phases: Dict[str, float], extras=None) -> FitResult:
    out = dict(extras or {})
    out.update({"evaluations": fit["evaluations"], "newton_step_in_se": fit["newton_step_in_se"]})
    return FitResult(names, [float(v) for v in fit["beta"]], [float(v) for v in fit["se"]], fit["loglik"],
                     int(fit["iterations"]), bool(fit["converged"]), {"tol_step_in_se": tol}, phases, out)


class LabNumpy(Adapter):
    name, kind, role = "lab_numpy", "driver", "prototype"
    block_rows = 50_000

    def describe(self, spark=None):
        return {"name": self.name, "kind": self.kind, "role": self.role, "version": __version__,
                "numpy": np.__version__, "algorithm": "weighted-Gram Cox; profile-likelihood logistic"}

    def support(self, model, data) -> Support:
        return _support(model, spark_layout=False)

    def driver_bytes(self, model, data) -> int:
        return footprint_bytes(data.rows, data.features, 1.2) + footprint_bytes(self.block_rows, data.features, 4.0)

    def fit(self, spark, dataset, model) -> FitResult:
        spec = dataset.spec
        names, p, tol = spec.feature_names(), spec.features, _tolerance(model)
        started = clock.perf_counter()
        others = ["provider", "time", "event"] if model.family == COX else ["provider", "y"]
        X, cols = load_arrays(spark, dataset, names, others)
        groups = cols["provider"].astype(np.int64)
        if model.family == COX and model.strata is None:
            groups = np.zeros(len(groups), dtype=np.int64)
        if np.any(groups[1:] < groups[:-1]):
            order = np.argsort(groups, kind="stable")
            X, groups = X[order], groups[order]
            cols = {k: v[order] for k, v in cols.items()}
        blocks = kernels.whole_group_blocks(groups, self.block_rows)
        loaded = clock.perf_counter()

        def evaluate(beta):
            value, gradient, information = 0.0, np.zeros(p), np.zeros((p, p))
            for a, b in blocks:
                if model.family == COX:
                    part = kernels.cox_partial(X[a:b], cols["time"][a:b], cols["event"][a:b], groups[a:b], beta)
                else:
                    part = kernels.logistic_profile_partial(X[a:b], cols["y"][a:b], groups[a:b], beta, presorted=True)
                value, gradient, information = value + part[0], gradient + part[1], information + part[2]
            return value, gradient, information

        fit = maximize(evaluate, p, tol, model.max_iterations or 50)
        phases = {"load": loaded - started, "fit": clock.perf_counter() - loaded}
        return _result(names, fit, tol, phases, {"blocks": len(blocks)})


def partition_kernel(family: str, p: int, beta: List[float]):
    """The function mapInArrow runs on each partition: stack the partition's rows, call the kernel,
    return one row of sums. Partitions hold whole providers, sorted as the kernels expect."""
    def run(batches):
        import pyarrow as pa

        xs, columns = [], {}
        for batch in batches:
            if batch.num_rows == 0:
                continue
            xs.append(batch.column("x").flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, p))
            for name in batch.schema.names:
                if name != "x":
                    columns.setdefault(name, []).append(batch.column(name).to_numpy(zero_copy_only=False))
        if not xs:
            return
        X = np.concatenate(xs) if len(xs) > 1 else xs[0]
        cols = {k: np.concatenate(v) for k, v in columns.items()}
        vector = np.asarray(beta, dtype=np.float64)
        if family == COX:
            value, gradient, information = kernels.cox_partial(
                X, cols["time"], cols["event"], cols["provider"], vector, presorted=True)
        else:
            value, gradient, information, _ = kernels.logistic_profile_partial(
                X, cols["y"], cols["provider"], vector, presorted=True)
        yield pa.RecordBatch.from_arrays(
            [pa.array([value], pa.float64()), pa.array([gradient.tobytes()], pa.binary()),
             pa.array([np.ascontiguousarray(information).tobytes()], pa.binary()), pa.array([X.shape[0]], pa.int64())],
            names=["ll", "u", "i", "n"])
    return run


class LabSpark(Adapter):
    name, kind, role = "lab_spark", "spark", "prototype"
    rows_per_partition = 100_000

    def describe(self, spark=None):
        return {"name": self.name, "kind": self.kind, "role": self.role, "version": __version__,
                "algorithm": "weighted-Gram Cox and profile-likelihood logistic, NumPy kernels per partition",
                "passes_per_evaluation": 1}

    def support(self, model, data) -> Support:
        return _support(model, spark_layout=True)

    def fit(self, spark, dataset, model) -> FitResult:
        from pyspark.sql import functions as F

        ship_package(spark)
        spec = dataset.spec
        names, p, tol = spec.feature_names(), spec.features, _tolerance(model)
        started = clock.perf_counter()
        frame = dataset.frame(spark)
        partitions = min(20_000, max(spark.sparkContext.defaultParallelism,
                                     math.ceil(dataset.rows / self.rows_per_partition)))
        x = F.array(*[F.col(c) for c in names]).alias("x")
        if model.family == COX:
            work = (frame.select("provider", x, "time", "event").repartition(partitions, "provider")
                    .sortWithinPartitions(F.col("provider"), F.col("time").desc()))
        else:
            degenerate = (frame.groupBy("provider").agg(F.sum("y").alias("o"), F.count("*").alias("n"))
                          .where((F.col("o") == 0) | (F.col("o") == F.col("n"))).count())
            if degenerate:
                raise kernels.DegenerateProviders(
                    f"{degenerate} providers have no events or only events; bounded intercepts are not prototyped")
            work = (frame.select("provider", x, F.col("y").cast("double").alias("y"))
                    .repartition(partitions, "provider").sortWithinPartitions("provider"))
        work = work.persist()
        try:
            rows = work.count()
            prepared = clock.perf_counter()

            def evaluate(beta):
                parts = work.mapInArrow(partition_kernel(model.family, p, [float(b) for b in beta]),
                                        PARTIAL_SCHEMA).collect()
                if sum(r["n"] for r in parts) != rows:
                    raise RuntimeError("a pass did not see every row")
                value = math.fsum(r["ll"] for r in parts)
                gradient = np.sum([np.frombuffer(r["u"], dtype=np.float64) for r in parts], axis=0)
                information = np.sum([np.frombuffer(r["i"], dtype=np.float64) for r in parts], axis=0).reshape(p, p)
                return value, gradient, information

            fit = maximize(evaluate, p, tol, model.max_iterations or 50)
            phases = {"prepare": prepared - started, "fit": clock.perf_counter() - prepared}
        finally:
            work.unpersist()
        return _result(names, fit, tol, phases, {"partitions": partitions, "passes": fit["evaluations"]})


register("lab_numpy", LabNumpy)
register("lab_spark", LabSpark)
