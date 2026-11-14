"""Deterministic synthetic data, generated on the cluster and stored once.

Every value is a function of (row identifier, seed, column tag) through Spark SQL's xxhash64, so a
dataset does not depend on the cluster, its partitioning or the order of evaluation. A dataset is
written once as Parquet with a manifest, and every implementation then reads that same table: this
is what makes timings and estimates comparable across packages.

Columns: id (0 .. n-1), provider (0 .. m-1, contiguous in id, equal sizes), x1 .. xp, and
  logistic_fe:  y in {0, 1}, with logit P(y = 1) = -1 + gamma_provider + x'beta
  cox:          time in whole days (so ties occur) and event in {0, 1}; exponential hazard
                0.2 exp(gamma_provider + x'beta) per year, uniform censoring over five years
gamma_provider is uniform on (-0.5, 0.5).
"""
from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .specs import COX, DatasetSpec

TWO52 = 2 ** 52


def _lit(x: float) -> str:
    return f"{float(x)!r}D"


def _uniform(key: str, seed: int, tag: str) -> str:
    """SQL for a uniform draw in (0, 1) that depends only on the key, the seed and the tag."""
    return f"((pmod(xxhash64({key}, {seed}L, '{tag}'), {TWO52}L) + 0.5D) / {TWO52}.0D)"


def _normal(key: str, seed: int, tag: str) -> str:
    u1, u2 = _uniform(key, seed, tag + ":a"), _uniform(key, seed, tag + ":b")
    return f"(sqrt(-2.0D * ln({u1})) * cos({_lit(2.0 * math.pi)} * {u2}))"


def _feature(spec: DatasetSpec, k: int) -> str:
    seed, tag = spec.seed, f"x{k}"
    if spec.design == "simple":
        if spec.is_indicator(k):
            return f"CASE WHEN {_uniform('id', seed, tag)} < {_lit(spec.prevalence(k))} THEN 1.0D ELSE 0.0D END"
        return f"((pmod(xxhash64(id, {seed}L, '{tag}'), 129L) - 64L) / 64.0D)"
    rho = spec.correlation
    latent = f"({_lit(rho)} * __f + {_lit(math.sqrt(1.0 - rho * rho))} * {_normal('id', seed, tag)})"
    if spec.is_indicator(k):
        return f"CASE WHEN {latent} > {_lit(spec.threshold(k))} THEN 1.0D ELSE 0.0D END"
    return f"({_lit(spec.scale(k))} * {latent})"


def generate(spark, spec: DatasetSpec, partitions: Optional[int] = None):
    """The dataset as a lazy DataFrame. Use `materialize` to store it; fit from the stored table."""
    n, m, p, seed = spec.rows, spec.providers, spec.features, spec.seed
    if partitions is None:
        partitions = max(spark.sparkContext.defaultParallelism, math.ceil(n * 8.0 * (p + 3) / 128e6))
    base = spark.range(0, n, 1, partitions).selectExpr("id", f"(id * {m}L) div {n}L AS provider")
    if spec.design == "correlated":
        base = base.selectExpr("*", f"{_normal('id', seed, 'f')} AS __f")
    names = spec.feature_names()
    features = base.selectExpr("id", "provider", *[f"{_feature(spec, k)} AS x{k}" for k in range(1, p + 1)])
    gamma = f"({_uniform('provider', seed, 'gamma')} - 0.5D)"
    eta = " + ".join(f"({_lit(b)} * x{k})" for k, b in enumerate(spec.true_beta(), start=1))
    with_lp = features.selectExpr("*", f"{gamma} + {eta} AS __lp")
    if spec.family == COX:
        timed = with_lp.selectExpr(
            "*",
            f"-ln({_uniform('id', seed, 'event_time')}) / (0.2D * exp(__lp)) AS __ey",
            f"5.0D * {_uniform('id', seed, 'censoring')} AS __cy",
        )
        return timed.selectExpr(
            "id", "provider", *names,
            "CAST(greatest(1L, CAST(ceil(365.0D * least(__ey, __cy, 5.0D)) AS BIGINT)) AS DOUBLE) AS time",
            "CASE WHEN __ey <= least(__cy, 5.0D) THEN 1 ELSE 0 END AS event",
        )
    outcome = f"CASE WHEN {_uniform('id', seed, 'y')} < 1.0D / (1.0D + exp(-(-1.0D + __lp))) THEN 1 ELSE 0 END"
    return with_lp.selectExpr("id", "provider", *names, f"{outcome} AS y")


@dataclass
class Dataset:
    """A stored dataset: its specification, its location and the manifest written with it."""
    spec: DatasetSpec
    path: str
    manifest: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.spec.dataset_id()

    @property
    def rows(self) -> int:
        return int(self.manifest["rows"])

    def frame(self, spark):
        return spark.read.parquet(self.path + "/data")


def _exists(spark, path: str) -> Optional[bool]:
    """Whether the path exists, asked of the file system directly; None when that cannot be asked."""
    try:
        target = spark._jvm.org.apache.hadoop.fs.Path(path)
        return bool(target.getFileSystem(spark._jsc.hadoopConfiguration()).exists(target))
    except Exception:
        return None


def _read_manifest(spark, path: str) -> Optional[Dict[str, Any]]:
    if _exists(spark, path + "/manifest") is False:
        return None
    try:
        rows = spark.read.text(path + "/manifest").collect()
    except Exception:
        return None
    return json.loads("".join(r[0] for r in rows)) if rows else None


def materialize(spark, spec: DatasetSpec, root: str, overwrite: bool = False) -> Dataset:
    """Write the dataset under `root/<dataset id>` unless it is already there, and return it.
    The manifest records the row count, the event count and an order-independent checksum of the
    stored rows, so a regenerated table can be shown to be identical."""
    path = f"{root.rstrip('/')}/{spec.dataset_id()}"
    existing = None if overwrite else _read_manifest(spark, path)
    if existing is not None:
        if existing.get("spec") != spec.to_dict():
            raise ValueError(f"the manifest at {path} describes another specification")
        return Dataset(spec, path, existing)
    generate(spark, spec).write.mode("overwrite").parquet(path + "/data")
    stored = spark.read.parquet(path + "/data")
    outcome = "event" if spec.family == COX else "y"
    columns = ", ".join(stored.columns)
    row = stored.selectExpr(
        "count(*) AS n", f"sum(CAST({outcome} AS BIGINT)) AS events", f"bit_xor(xxhash64({columns})) AS checksum"
    ).first()
    if int(row["n"]) != spec.rows:
        raise RuntimeError(f"wrote {row['n']} rows, expected {spec.rows}")
    manifest = {
        "dataset_id": spec.dataset_id(),
        "spec": spec.to_dict(),
        "columns": stored.columns,
        "rows": int(row["n"]),
        "events": int(row["events"]),
        "checksum": int(row["checksum"]),
        "true_beta": spec.true_beta(),
        "spark_version": spark.version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    spark.createDataFrame([(json.dumps(manifest),)], "value string").coalesce(1).write.mode("overwrite").text(
        path + "/manifest"
    )
    return Dataset(spec, path, manifest)


def load(spark, root: str, spec: DatasetSpec) -> Dataset:
    path = f"{root.rstrip('/')}/{spec.dataset_id()}"
    manifest = _read_manifest(spark, path)
    if manifest is None:
        raise FileNotFoundError(f"no dataset at {path}; run materialize first")
    return Dataset(spec, path, manifest)
