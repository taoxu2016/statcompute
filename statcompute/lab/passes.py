"""What a pass over the engine's working set costs, part by part (OPEN_ITEMS OI-40), and what the
Cox kernels cost inside a task (OI-35). A probe: it fits nothing.

With one task per core an evaluation at 5 million rows takes 1.3 to 3.2 seconds on the cluster, and
a second or more of that moved neither with the number of tasks nor with the rows of a task
(docs/reports/2026-11-21-databricks-partition-size.md). This module takes an evaluation apart. It
builds the working set as the engine builds it for a model with strata or providers (the same
columns, the same partitions, the same order) and runs a ladder of passes over it. Every pass is
one Spark job whose tasks send one small row each to the driver, as the tasks of an evaluation do,
and each kind adds one thing to the kind before it:

    job           as many tasks as the working set has partitions, in the JVM, that do nothing
    idle          Python tasks that are sent one byte for every row and do nothing with it
    rows          Python tasks that receive the rows and stack them into arrays, as the engine's do
    kernel        the same, and the engine's kernel, BLAS held to one thread as the engine holds it
    kernel_before the same rows through the engine's Cox kernel as it was up to version 0.24.0 (Cox only)
    kernel_free   the engine's kernel without the limit on BLAS
    kernel_plain  the same rows through the prototype's Cox kernel, as the prototype runs it (Cox only)
    evaluation    the engine's own pass: its partition function, its results collected and summed
    half          `kernel` on every second partition: half the tasks, one for every two cores

The kinds take turns, repeat by repeat, so that what the cluster does during the probe falls on all
of them. The tasks of the Python kinds time themselves: from their first line to their last, of
that the part until the rows are stacked, and the part in the kernel. A pass's seconds less its
slowest task's are what the job costs around its tasks.

The working set is built by this module's own lines, not by the engine's: `tests/test_passes.py`
holds the two together by asking for the same bits from the probe's evaluation and the engine's.
"""
from __future__ import annotations

import contextlib
import statistics
import time as clock
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import __version__
from ..adapters import EXACT, UNSUPPORTED, Adapter, FitResult, Support, register
from ..engine.spark import (PARTIAL, CoxPH, _cox_partition, _kept, _logistic_partition, _one_thread, _providers, _runs,
                            _stack, _sum, kernels, working_partitions)
from ..testkit.specs import COX
from . import kernels as _plain
from .cox_kernel_0_24 import cox_partial_0_24
from .dist import functions_of, ship_package

plain = functions_of(_plain)           # a namespace of functions: no module object travels with a task (engine/dist.py)

KINDS = ("job", "idle", "rows", "kernel", "kernel_before", "kernel_free", "kernel_plain", "evaluation", "half")
COX_ONLY = ("kernel_before", "kernel_plain")
TIMED = "n long, task_s double, receive_s double, kernel_s double, value double"


def _row(pa, rows: int, task: float, receive: float, kernel: float, value: float):
    return pa.RecordBatch.from_arrays(
        [pa.array([int(rows)], pa.int64()), pa.array([float(task)], pa.float64()), pa.array([float(receive)], pa.float64()),
         pa.array([float(kernel)], pa.float64()), pa.array([float(value)], pa.float64())],
        names=["n", "task_s", "receive_s", "kernel_s", "value"])


def _idle():
    """A Python task that reads what it is sent, one byte a row, and does nothing with it."""
    def run(batches):
        import time

        import pyarrow as pa

        started, rows = time.perf_counter(), 0
        for batch in batches:
            rows += batch.num_rows
        spent = time.perf_counter() - started
        yield _row(pa, rows, spent, spent, 0.0, 0.0)
    return run


def _timed(family: str, p: int, beta: List[float], ties: str, kernel: Optional[str]):
    """A Python task that receives and stacks its partition's rows as the engine's tasks do and
    then runs `kernel` on them: None for no kernel, "engine" for the engine's with BLAS at one
    thread, "before" for the engine's Cox kernel of version 0.24.0, "free" for the engine's
    without the limit, "plain" for the prototype's Cox kernel. It
    returns its rows, its seconds from first line to last, those until the rows were stacked,
    those in the kernel, and the log-likelihood the kernel gave."""
    def run(batches):
        import time

        import pyarrow as pa

        started = time.perf_counter()
        cols = _stack(batches, p)
        if cols is None:
            return
        received, value, vector = time.perf_counter(), 0.0, np.asarray(beta, dtype=np.float64)
        if kernel:
            with (_one_thread() if kernel in ("engine", "before") else contextlib.nullcontext()):
                if family == COX and kernel == "plain":
                    value = plain.cox_partial(cols["x"], cols["t"], cols["d"], _runs(cols["s"])[0], vector, presorted=True)[0]
                elif family == COX and kernel == "before":
                    value = cox_partial_0_24(cols["x"], cols["t"], cols["d"], _runs(cols["s"])[0], vector, ties=ties,
                                             presorted=True)["loglik"]
                elif family == COX:                        # as the engine's task calls it: in the task's own rows
                    value = kernels.cox_partial(cols["x"], cols["t"], cols["d"], _runs(cols["s"])[0], vector, ties=ties,
                                                presorted=True, overwrite=True)["loglik"]
                else:
                    X, y, c, _, _, _, solvable = _providers(cols)
                    keep = solvable[c]
                    value = kernels.logistic_partial(*_kept(X, y, c, keep), vector)["loglik"] if keep.any() else 0.0
        done = time.perf_counter()
        yield _row(pa, len(cols["x"]), done - started, received - started, done - received, value)
    return run


def working_set(spark, frame, family: str, features: Sequence[str], rows: int, *, ties: str = "breslow",
                row_id: Optional[str] = "id", time: str = "time", event: str = "event", provider: str = "provider",
                outcome: str = "y"):
    """The engine's working set for a model with strata or providers, not yet built: the frame to
    persist, its partitions, the rule that gave them and the cores at this moment. Line for line
    what `CoxPH._fit_strata` and `LogisticFE.fit` select, place and sort."""
    from pyspark.sql import functions as F

    p = len(features)
    if family == COX:
        columns, _, _ = CoxPH(time, event, list(features), provider, ties=ties, row_id=row_id)._columns(stratified=True)
        order = ["s", F.desc("t")] + (["k"] if row_id else [])
    else:
        columns = [F.col(provider).alias("s"), F.col(outcome).cast("double").alias("y"),
                   F.array(*[F.col(c).cast("double") for c in features]).alias("x")]
        if row_id:
            columns.append(F.col(row_id).cast("double").alias("k"))
        order = ["s"] + (["k"] if row_id else [])
    partitions, rule, cores = working_partitions(spark, rows, p)
    return frame.select(*columns).repartition(partitions, "s").sortWithinPartitions(*order), partitions, rule, cores


def measure(spark, frame, family: str, features: Sequence[str], rows: int, beta: Sequence[float], *, repeats: int = 5,
            ties: str = "breslow", row_id: Optional[str] = "id", kinds: Sequence[str] = KINDS, **names) -> Dict[str, Any]:
    """Build the working set and run every kind of pass `repeats` times, the kinds taking turns.

    Returns the layout, the seconds of the pass that built the working set (an evaluation, as in
    the engine), the log-likelihood the evaluations gave at `beta`, and for every kind, pass by
    pass: the seconds of the job as the driver saw it, the tasks that returned a row, the cores
    the cluster had, and from the tasks themselves the median and the largest of their own
    seconds, the median of those spent receiving and stacking, and the median and the largest of
    those in the kernel."""
    from pyspark.sql import functions as F

    p, vector = len(features), [float(b) for b in beta]
    kinds = [kind for kind in kinds if kind not in COX_ONLY or family == COX]
    frame, partitions, rule, cores = working_set(spark, frame, family, features, rows, ties=ties, row_id=row_id, **names)
    work = frame.persist()
    state: Dict[str, Any] = {}

    def evaluation():
        function = _cox_partition(p, vector, ties) if family == COX else _logistic_partition(p, vector)
        parts = work.mapInArrow(function, PARTIAL).collect()
        if sum(r["n"] for r in parts) != rows:
            raise RuntimeError("a pass did not see every row")
        state["value"] = _sum(parts, p)[0]

    passes = {
        "job": lambda: spark.range(0, partitions, 1, partitions).collect() and None,
        "idle": lambda: work.select(F.lit(0).cast("byte").alias("b")).mapInArrow(_idle(), TIMED).collect(),
        "rows": lambda: work.mapInArrow(_timed(family, p, vector, ties, None), TIMED).collect(),
        "kernel": lambda: work.mapInArrow(_timed(family, p, vector, ties, "engine"), TIMED).collect(),
        "kernel_before": lambda: work.mapInArrow(_timed(family, p, vector, ties, "before"), TIMED).collect(),
        "kernel_free": lambda: work.mapInArrow(_timed(family, p, vector, ties, "free"), TIMED).collect(),
        "kernel_plain": lambda: work.mapInArrow(_timed(family, p, vector, ties, "plain"), TIMED).collect(),
        "evaluation": evaluation,
        "half": lambda: work.where(F.spark_partition_id() % 2 == 0).mapInArrow(_timed(family, p, vector, ties, "engine"), TIMED).collect(),
    }
    fields = ("seconds", "tasks", "cores", "task_median_s", "task_max_s", "receive_median_s", "kernel_median_s", "kernel_max_s")
    found = {kind: {name: [] for name in fields} for kind in kinds}
    try:
        started = clock.perf_counter()
        evaluation()                                           # builds the working set, as the engine's first evaluation does
        built = clock.perf_counter() - started
        for _ in range(int(repeats)):
            for kind in kinds:
                started = clock.perf_counter()
                returned = passes[kind]()
                spent, entry = clock.perf_counter() - started, found[kind]
                entry["seconds"].append(round(spent, 3))
                entry["cores"].append(int(spark.sparkContext.defaultParallelism))
                if returned:
                    if kind not in ("idle", "half") and sum(r["n"] for r in returned) != rows:
                        raise RuntimeError("a pass did not see every row")
                    middle = lambda name: round(statistics.median(r[name] for r in returned), 4)   # noqa: E731
                    entry["tasks"].append(len(returned))
                    entry["task_median_s"].append(middle("task_s"))
                    entry["task_max_s"].append(round(max(r["task_s"] for r in returned), 4))
                    entry["receive_median_s"].append(middle("receive_s"))
                    entry["kernel_median_s"].append(middle("kernel_s"))
                    entry["kernel_max_s"].append(round(max(r["kernel_s"] for r in returned), 4))
                    if kind.startswith("kernel"):
                        state[kind] = sum(r["value"] for r in returned)
    finally:
        work.unpersist()
    return {"partitions": partitions, "partitions_from": rule, "cores": cores, "build_s": built, "repeats": int(repeats),
            "value": state.get("value"), "kernel_values": {k: v for k, v in state.items() if k.startswith("kernel")},
            "kinds": list(kinds), "passes": found}


class EnginePasses(Adapter):
    """The probe as an adapter, so that a plan can run it on the cluster like a fit: it holds and
    warms the cluster as for one, and its record carries what the passes cost in `extras`. It
    evaluates at the coefficients the data were generated with, where the kernels do the work of a
    late evaluation. It returns no coefficients: nothing is compared with it. A plan can name the
    kinds of pass to run (`pass_kinds`) and how often (`pass_repeats`)."""
    name, kind, role = "engine_passes", "spark", "probe"

    def describe(self, spark=None):
        return {"name": self.name, "kind": self.kind, "role": self.role, "version": __version__,
                "algorithm": "a ladder of passes over the engine's working set; no fit"}

    def support(self, model, data) -> Support:
        if model.family == COX and model.strata is None:
            return Support(UNSUPPORTED, ["without strata an evaluation is the time grid's two passes, which this probe does not take apart"])
        return Support(EXACT, ["a probe: it times the parts of a pass over the engine's working set and fits nothing"])

    def fit(self, spark, dataset, model) -> FitResult:
        ship_package(spark)
        spec = dataset.spec
        names = spec.feature_names()
        started = clock.perf_counter()
        out = measure(spark, dataset.frame(spark), model.family, names, dataset.rows, spec.true_beta(),
                      repeats=int(self.options.get("pass_repeats", 5)), ties=(model.ties or "breslow"), row_id="id",
                      kinds=tuple(self.options.get("pass_kinds") or KINDS))
        spent = clock.perf_counter() - started
        execution = {"layout": "whole strata", "partitions": out["partitions"], "partitions_from": out["partitions_from"],
                     "cores": out["cores"], "row_id": "id"}
        return FitResult(names, [], phases={"prepare": out["build_s"], "fit": spent - out["build_s"]},
                         extras={"ladder": out["passes"], "pass_kinds": out["kinds"], "pass_repeats": out["repeats"],
                                 "partitions": out["partitions"], "value": out["value"], "kernel_values": out["kernel_values"],
                                 "execution": execution})


register("engine_passes", EnginePasses)
