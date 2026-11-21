"""Probe A-11: what a pass costs before the kernel runs, and whether packing the working set removes it.

On the cluster a task of a pass over 50,000 rows by 100 features takes about one second, of which
the NumPy kernel is a fraction; the pass for the provider effects, whose kernel is light, takes as
long (docs/reports/2026-11-21-databricks-warm-runs.md). The suspicion: the time goes into bringing
the rows from Spark's cache to Python, where a column of arrays is written to Arrow element by
element. A working set packed into blocks (a few rows per partition, each holding thousands of
source rows as bytes) would cross as bytes.

This probe builds both working sets from the same rows, with whole strata per partition and sorted,
and times (a) a pass that only receives the rows and counts them, (b) a pass that runs the Cox
kernel. Run: python packed_working_set.py ROWS FEATURES DATA_ROOT [PARTITIONS] [BLOCK_ROWS]
"""
from __future__ import annotations

import sys
import time

import numpy as np


def main(rows: int, features: int, root: str, partitions: int = 8, block_rows: int = 8192) -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from statcompute.engine import kernels
    from statcompute.testkit import DatasetSpec, materialize

    spark = (SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false").config("spark.driver.memory", "2g")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    dataset = materialize(spark, DatasetSpec("cox", rows, max(8, rows // 625), features), root)
    names, p = dataset.spec.feature_names(), features
    frame = dataset.frame(spark)
    wide = (frame.select(F.col("provider").alias("s"), F.col("time").cast("double").alias("t"), F.col("event").cast("double").alias("d"),
                         F.array(*[F.col(c).cast("double") for c in names]).alias("x"))
            .repartition(partitions, "s").sortWithinPartitions(F.col("s"), F.col("t").desc()).persist())
    n = wide.count()

    def pack(batches):
        """Whole strata stay together: a block ends at a stratum's end once it has block_rows rows."""
        import pyarrow as pa

        xs, ss, ts, ds = [], [], [], []
        for batch in batches:
            if batch.num_rows:
                xs.append(batch.column("x").flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, p))
                ss.append(batch.column("s").to_numpy(zero_copy_only=False)); ts.append(batch.column("t").to_numpy(zero_copy_only=False))
                ds.append(batch.column("d").to_numpy(zero_copy_only=False))
        if not xs:
            return
        X, s, t, d = np.concatenate(xs), np.concatenate(ss).astype(np.int64), np.concatenate(ts), np.concatenate(ds)
        ends = np.flatnonzero(np.concatenate((s[1:] != s[:-1], [True]))) + 1          # where a stratum ends
        cut, start = [], 0
        for end in ends:
            if end - start >= block_rows:
                cut.append((start, int(end))); start = int(end)
        if start < len(s):
            cut.append((start, len(s)))
        for a, b in cut:
            yield pa.RecordBatch.from_arrays(
                [pa.array([b - a], pa.int64()), pa.array([s[a:b].tobytes()], pa.large_binary()), pa.array([t[a:b].tobytes()], pa.large_binary()),
                 pa.array([d[a:b].tobytes()], pa.large_binary()), pa.array([np.ascontiguousarray(X[a:b]).tobytes()], pa.large_binary())],
                names=["n", "s", "t", "d", "x"])

    packed = wide.mapInArrow(pack, "n long, s binary, t binary, d binary, x binary").persist()
    blocks = packed.count()

    def receive_wide(batches):
        import pyarrow as pa
        seen = 0
        for batch in batches:
            if batch.num_rows:
                batch.column("x").flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, p)
                seen += batch.num_rows
        yield pa.RecordBatch.from_arrays([pa.array([seen], pa.int64())], names=["n"])

    def receive_packed(batches):
        import pyarrow as pa
        seen = 0
        for batch in batches:
            for i in range(batch.num_rows):
                np.frombuffer(batch.column("x")[i].as_buffer(), dtype=np.float64)
                seen += batch.column("n")[i].as_py()
        yield pa.RecordBatch.from_arrays([pa.array([seen], pa.int64())], names=["n"])

    beta = [0.01] * p

    def kernel_wide(batches):
        import pyarrow as pa
        xs, ss, ts, ds = [], [], [], []
        for batch in batches:
            if batch.num_rows:
                xs.append(batch.column("x").flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, p))
                ss.append(batch.column("s").to_numpy(zero_copy_only=False)); ts.append(batch.column("t").to_numpy(zero_copy_only=False))
                ds.append(batch.column("d").to_numpy(zero_copy_only=False))
        if not xs:
            return
        out = kernels.cox_partial(np.concatenate(xs), np.concatenate(ts), np.concatenate(ds), np.concatenate(ss), np.asarray(beta), presorted=True)
        yield pa.RecordBatch.from_arrays([pa.array([out["loglik"]], pa.float64()), pa.array([out["score"].tobytes()], pa.binary())], names=["ll", "u"])

    def kernel_packed(batches):
        import pyarrow as pa
        ll, u = 0.0, np.zeros(p)
        for batch in batches:
            for i in range(batch.num_rows):
                m = batch.column("n")[i].as_py()
                X = np.frombuffer(batch.column("x")[i].as_buffer(), dtype=np.float64).reshape(m, p)
                s = np.frombuffer(batch.column("s")[i].as_buffer(), dtype=np.int64)
                t = np.frombuffer(batch.column("t")[i].as_buffer(), dtype=np.float64)
                d = np.frombuffer(batch.column("d")[i].as_buffer(), dtype=np.float64)
                out = kernels.cox_partial(X, t, d, s, np.asarray(beta), presorted=True)
                ll += out["loglik"]; u += out["score"]
        yield pa.RecordBatch.from_arrays([pa.array([ll], pa.float64()), pa.array([u.tobytes()], pa.binary())], names=["ll", "u"])

    def timed(work, function, schema, repeat=3):
        best, result = None, None
        for _ in range(repeat):
            started = time.perf_counter()
            result = work.mapInArrow(function, schema).collect()
            seconds = time.perf_counter() - started
            best = seconds if best is None else min(best, seconds)
        return best, result

    print(f"{n} rows by {p} features in {partitions} partitions; packed into {blocks} blocks of about {block_rows} rows")
    a, got = timed(wide, receive_wide, "n long")
    b, got_packed = timed(packed, receive_packed, "n long")
    assert sum(r["n"] for r in got) == sum(r["n"] for r in got_packed) == n
    print(f"a pass that only receives the rows:  array column {a:6.2f} s | packed {b:6.2f} s")
    c, wide_sums = timed(wide, kernel_wide, "ll double, u binary")
    e, packed_sums = timed(packed, kernel_packed, "ll double, u binary")
    ll_wide, ll_packed = sum(r["ll"] for r in wide_sums), sum(r["ll"] for r in packed_sums)
    u_wide = np.sum([np.frombuffer(r["u"]) for r in wide_sums], axis=0)
    u_packed = np.sum([np.frombuffer(r["u"]) for r in packed_sums], axis=0)
    print(f"a pass with the Cox kernel:          array column {c:6.2f} s | packed {e:6.2f} s")
    print(f"same sums: log-likelihood {abs(ll_wide - ll_packed) / abs(ll_wide):.1e} relative, score {np.max(np.abs(u_wide - u_packed)) / np.max(np.abs(u_wide)):.1e} relative")
    spark.stop()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], *[int(v) for v in sys.argv[4:6]])
