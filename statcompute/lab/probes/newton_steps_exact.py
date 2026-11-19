"""
Probe A-9 (2026-11-19): the Newton steps of a Cox fit without strata on one of the benchmark's datasets,
and what a last step cut short would leave.

Every value of a benchmark dataset is a function of the row identifier, the seed and the column name, so a
table that ran on the cluster can be regenerated anywhere. This probe
  1. regenerates the dataset with the repository's generator and stores it compactly (the "simple"
     design only: its features are multiples of 1/64 or 0/1, so one byte per value is exact);
  2. runs Newton's method from zero with the engine's time-grid kernels, in chunks, to 1e-10 standard
     errors, and prints every step;
  3. recomputes the score at every iterate by other code: per-day tables in 80-bit arithmetic on the
     covariates as stored, uncentred, added with np.add.at, no sorting;
  4. prints the gain in log-likelihood of every step beside the spacing of doubles there, and what the
     last step leaves when it is cut to 2^-h of its length, beside a reported difference if one is given.

    python newton_steps_exact.py ROWS PROVIDERS FEATURES WORK_DIR [REPORTED_ABS REPORTED_IN_SE]

Needs this repository on the path and PySpark. 5 million rows by 100 features: 4 GB of memory, 1.5 GB of
disk under WORK_DIR, about eight minutes on one core.
"""
import json
import os
import sys
import time as clock

import numpy as np

from statcompute.engine import timegrid
from statcompute.engine.newton import maximize
from statcompute.testkit import DatasetSpec

rows, providers, features, work = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
reported = [float(v) for v in sys.argv[5:7]]
spec = DatasetSpec("cox", rows, providers, features)
folder = os.path.join(work, spec.dataset_id())
CHUNK = 100_000

# ---------------------------------------------------------------- 1. the dataset, regenerated
if not os.path.exists(folder + "/event.npy"):
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from statcompute.testkit import materialize

    spark = (SparkSession.builder.master("local[1]").appName("probe-a9").config("spark.sql.shuffle.partitions", "4")
             .config("spark.driver.memory", "1500m").config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    dataset = materialize(spark, spec, os.path.join(work, "data"))
    os.makedirs(folder, exist_ok=True)
    with open(folder + "/manifest.json", "w") as f:
        json.dump(dataset.manifest, f)
    names = spec.feature_names()
    unit = np.array([1.0 if spec.is_indicator(k) else 1.0 / 64.0 for k in range(1, features + 1)])
    stored = np.lib.format.open_memmap(folder + "/x.i8.npy", mode="w+", dtype=np.int8, shape=(rows, features))
    exit_time, died = np.empty(rows), np.empty(rows, dtype=np.int8)
    frame = dataset.frame(spark).select("id", "time", "event", *names)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    for start in range(0, rows, 250_000):
        pdf = frame.where((F.col("id") >= start) & (F.col("id") < min(rows, start + 250_000))).toPandas()
        at, values = pdf["id"].to_numpy(dtype=np.int64), pdf[names].to_numpy(dtype=np.float64)
        codes = np.rint(values / unit)
        if not np.array_equal(codes * unit, values) or np.abs(codes).max() > 64:
            raise RuntimeError("a feature is not on its grid: the compact storage would not be exact")
        stored[at] = codes.astype(np.int8)
        exit_time[at], died[at] = pdf["time"].to_numpy(dtype=np.float64), pdf["event"].to_numpy(dtype=np.int8)
    stored.flush()
    np.save(folder + "/time.npy", exit_time)
    np.save(folder + "/scale.npy", unit)
    np.save(folder + "/event.npy", died)
    spark.stop()

with open(folder + "/manifest.json") as f:
    manifest = json.load(f)
packed = np.load(folder + "/x.i8.npy", mmap_mode="r")
scale, time = np.load(folder + "/scale.npy"), np.load(folder + "/time.npy")
event = np.load(folder + "/event.npy").astype(np.float64)
n, p = packed.shape
grid = np.unique(time)
K = len(grid)
print(f"{spec.dataset_id()}: {n} rows, {int(event.sum())} events, {K} distinct times; "
      f"manifest: events {manifest['events']}, checksum {manifest['checksum']}", flush=True)


class Lazy:
    """The covariates as float64, a chunk at a time: x = stored byte * scale, exactly."""
    shape = (n, p)

    def __getitem__(self, key):
        return packed[key].astype(np.float64) * scale


total = np.zeros(p, dtype=np.int64)
for a in range(0, n, CHUNK):
    total += packed[a:a + CHUNK].sum(axis=0, dtype=np.int64)
mu = total * scale / n

# ---------------------------------------------------------------- 2. Newton's method with the engine's kernels
trace, state = [], {"shift": 0.0}


def evaluate(beta):
    out = timegrid.evaluate_arrays(Lazy(), time, event, beta, ties="breslow", grid=grid, mu=mu, shift=state["shift"],
                                   chunk_rows=CHUNK)
    state["shift"] = out["eta_max"]
    covariance = np.linalg.inv(out["information"])
    trace.append({"beta": np.array(beta, copy=True), "loglik": out["loglik"], "score": out["score"].copy(),
                  "information": out["information"].copy(), "step": covariance @ out["score"],
                  "se": np.sqrt(np.diag(covariance))})
    return out["loglik"], out["score"], out["information"]


started = clock.perf_counter()
fit = maximize(evaluate, p, 1e-10, 50)
print(f"engine kernels: {fit['iterations']} iterations, {fit['evaluations']} evaluations, converged={fit['converged']} "
      f"({clock.perf_counter() - started:.0f} s)", flush=True)


# ---------------------------------------------------------------- 3. the score again, in 80-bit arithmetic
def extended(beta):
    L = np.longdouble
    b, code = beta.astype(L), np.searchsorted(grid, time)
    S, D, loglik = np.zeros((K, p + 1), dtype=L), np.zeros((K, p + 1), dtype=L), L(0)   # per day: (r, r x); (deaths, their x)
    for a in range(0, n, CHUNK):
        x = (packed[a:a + CHUNK].astype(np.float64) * scale).astype(L)
        eta = x @ b
        r, c, dead = np.exp(eta), code[a:a + CHUNK], event[a:a + CHUNK] > 0
        np.add.at(S[:, 0], c, r)
        np.add.at(S[:, 1:], c, r[:, None] * x)
        np.add.at(D[:, 0], c[dead], L(1))
        np.add.at(D[:, 1:], c[dead], x[dead])
        loglik += eta[dead].sum()
    R = np.cumsum(S[::-1], axis=0)[::-1]                    # risk-set sums: rows with time >= t
    at = D[:, 0] > 0
    loglik -= (D[at, 0] * np.log(R[at, 0])).sum()
    return loglik, D[:, 1:].sum(axis=0) - (D[at, 0][:, None] * R[at, 1:] / R[at, 0][:, None]).sum(axis=0)


for k, t in enumerate(trace):
    t["exact_loglik"], score = extended(t["beta"])
    t["exact_step"] = np.linalg.solve(t["information"], score.astype(np.float64))
    apart = np.linalg.solve(t["information"], t["score"] - score.astype(np.float64))
    print(f"  iterate {k}: log-likelihood {t['loglik']!r}; next Newton step {np.max(np.abs(t['step']) / t['se']):.4e} se by the "
          f"engine, {np.max(np.abs(t['exact_step']) / t['se']):.4e} by the 80-bit score ({np.max(np.abs(t['exact_step'])):.4e} absolute); "
          f"the two scores differ by {np.max(np.abs(apart) / t['se']):.2e} se as a step", flush=True)

# ---------------------------------------------------------------- 4. gains, and a last step cut short
values = [t["exact_loglik"] for t in trace]
print("gain in log-likelihood of each step: " + ", ".join(f"{float(values[k + 1] - values[k]):.4g}" for k in range(len(values) - 1))
      + f"; neighbouring doubles at the log-likelihood are {np.spacing(abs(float(values[-1]))):.3g} apart")
step, se = trace[-2]["exact_step"], trace[-1]["se"]
in_se, absolute = float(np.max(np.abs(step) / se)), float(np.max(np.abs(step)))
print(f"last step: {in_se:.4e} se, {absolute:.4e} absolute" + (f"; reported difference {reported[1]:.3g} se, {reported[0]:.3g} absolute" if reported else ""))
for h in range(1, 9):
    print(f"  cut to 2^-{h}: leaves {(1 - 0.5 ** h) * in_se:.4e} se, {(1 - 0.5 ** h) * absolute:.4e} absolute")
