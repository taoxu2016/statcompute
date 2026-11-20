"""
Probe A-10 (2026-11-20): what does "stable provider-level flags" need?

A flag is a cut-off on a continuous statistic: |z| against the normal quantile for the score and Wald
tests, the p-value against 1 - level for the exact test. Two computations of the same fit can disagree
on a provider's flag only if its statistic lies between them and the cut-off, that is, closer to the
cut-off than the two statistics are to each other. This probe measures both sides on one dataset of the
benchmark's generator, for the logistic model with provider effects:

  how far the statistics move   between two layouts of the engine; between the engine and pprof_py at
                                its tightest tolerance; between pprof_py at looser tolerances (1e-8, its
                                default, and 1e-5, the default of pprof in R) and at its tightest
  how close providers sit       the smallest distances of a statistic from its cut-off, and how many
                                providers lie within 1e-7, 1e-5, 1e-3 and 1e-2 of it

    python check_flag_margins.py ROWS PROVIDERS FEATURES DATA_ROOT

Needs this repository on the path, PySpark and pprof_py with numba. About three minutes on one core at
800,000 rows, 4,000 providers and 10 features.
"""
import sys
from statistics import NormalDist

import numpy as np
from pyspark.sql import SparkSession

import pprof_py
from statcompute.engine import LogisticFE
from statcompute.testkit import DatasetSpec, materialize
from statcompute.testkit.localdata import load_arrays

rows, providers, features, root = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
LEVEL = 0.95
CRITICAL, ALPHA = NormalDist().inv_cdf(0.5 + LEVEL / 2.0), 1.0 - LEVEL
spark = (SparkSession.builder.master("local[2]").appName("probe-a10").config("spark.sql.shuffle.partitions", "4")
         .config("spark.driver.memory", "2g").config("spark.ui.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spec = DatasetSpec("logistic_fe", rows, providers, features)
dataset = materialize(spark, spec, root)
names = spec.feature_names()
TESTS = (("score", "z"), ("Wald", "z"), ("exact", "p"))


def engine(rows_per_partition):
    model = LogisticFE("y", names, "provider", row_id="id", tol=1e-10, rows_per_partition=rows_per_partition).fit(dataset.frame(spark))
    table = model.provider_effects
    return model, {"score": (table["score_z"].to_numpy(), table["score_flag"].to_numpy()),
                   "Wald": (table["wald_z"].to_numpy(), table["wald_flag"].to_numpy()),
                   "exact": (table["exact_p"].to_numpy(), table["exact_flag"].to_numpy())}, table


def reference(X, y, provider, tol):
    model = pprof_py.LogisticFixedEffectModel()
    model.fit(X, y, provider, use_dataprep=False, tol=tol)
    out = {}
    for name, method in (("score", "score"), ("Wald", "wald"), ("exact", "poibin_exact")):
        table = model.test(test_method=method)
        out[name] = (table["p_value" if name == "exact" else "z_raw"].to_numpy(dtype=float), table["flag"].to_numpy(dtype=int))
    return model, out


def between(label, a, b, free=None):
    parts = []
    for name, _ in TESTS:
        keep = np.ones(len(a[name][0]), dtype=bool) if free is None else free
        parts.append(f"{name} {np.max(np.abs(a[name][0][keep] - b[name][0][keep])):.1e} ({int(np.sum(a[name][1] != b[name][1]))} flags differ)")
    print(f"  {label}: largest difference in the statistic: " + "; ".join(parts))


def distance(name, kind, values):
    return np.abs(np.abs(values) - CRITICAL) if kind == "z" else np.abs(values - ALPHA)


first, ours, table = engine(50_000)
second, relaid, _ = engine(20_000)
free = ~table["bounded"].to_numpy(dtype=bool)                 # providers held at the bound have no finite Wald statistic
print(f"{spec.dataset_id()}: {len(table)} providers, {int((~free).sum())} of them held at the bound; "
      f"engine {first.iterations} iterations, next step {first.newton_step_in_se:.1e} se")
for name, kind in TESTS:
    flags = ours[name][1]
    near = distance(name, kind, ours[name][0])[free]
    print(f"  {name}: {int(np.sum(flags > 0))} flagged above, {int(np.sum(flags < 0))} below; distance of the statistic from its cut-off: "
          f"smallest {', '.join(f'{v:.1e}' for v in np.sort(near)[:3])}; providers within 1e-7: {int(np.sum(near < 1e-7))}, "
          f"1e-5: {int(np.sum(near < 1e-5))}, 1e-3: {int(np.sum(near < 1e-3))}, 1e-2: {int(np.sum(near < 1e-2))}")
between(f"engine with {first.partitions} partitions against {second.partitions}", ours, relaid, free)

X, cols = load_arrays(spark, dataset, names, ["y", "provider"])
y, provider = cols["y"].astype(float), cols["provider"]
tight_model, tight = reference(X, y, provider, 1e-12)
se = np.asarray(first.standard_errors)
print(f"  engine against pprof_py at tol 1e-12: coefficients {np.max(np.abs(first.coefficients - np.ravel(tight_model.coefficients_['beta'])) / se):.1e} se apart")
between("engine against pprof_py at tol 1e-12", ours, tight, free)
for tol in (1e-8, 1e-5):
    loose_model, loose = reference(X, y, provider, tol)
    apart = np.max(np.abs(np.ravel(loose_model.coefficients_["beta"]) - np.ravel(tight_model.coefficients_["beta"])) / se)
    print(f"  pprof_py at tol {tol:g} against itself at 1e-12: coefficients {apart:.1e} se apart")
    between(f"pprof_py at tol {tol:g} against itself at 1e-12", loose, tight, free)
spark.stop()
