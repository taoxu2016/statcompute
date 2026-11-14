"""Bring a stored dataset to the driver as NumPy arrays, in chunks, for single-machine
implementations. Peak memory is the result plus one chunk, not a multiple of the result."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def load_arrays(spark, dataset, feature_columns: Sequence[str], other_columns: Sequence[str],
                chunk_rows: int = 250_000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """X as float64 [n, p] in id order, and each other column as a float64 vector in id order.
    Row identifiers must be 0 .. n-1, as the generator writes them."""
    from pyspark.sql import functions as F

    n, p = dataset.rows, len(feature_columns)
    X = np.empty((n, p), dtype=np.float64)
    others = {c: np.empty(n, dtype=np.float64) for c in other_columns}
    frame = dataset.frame(spark).select("id", *other_columns, *feature_columns)
    key = "spark.sql.execution.arrow.pyspark.enabled"
    previous = spark.conf.get(key, "false")
    spark.conf.set(key, "true")
    try:
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            pdf = frame.where((F.col("id") >= start) & (F.col("id") < stop)).toPandas()
            if len(pdf) != stop - start:
                raise RuntimeError(f"rows {start}..{stop} returned {len(pdf)} rows")
            position = pdf["id"].to_numpy(dtype=np.int64)
            X[position, :] = pdf[list(feature_columns)].to_numpy(dtype=np.float64)
            for c in other_columns:
                others[c][position] = pdf[c].to_numpy(dtype=np.float64)
    finally:
        spark.conf.set(key, previous)
    return X, others


def footprint_bytes(rows: int, features: int, copies: float) -> int:
    """Bytes for `copies` arrays of rows x features doubles."""
    return int(copies * 8.0 * rows * features)
