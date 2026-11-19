"""
Probe A-8 (2026-11-19): can pprof_py's Cox estimate stop short of the root whatever its tolerance?

pprof_py's Newton loop halves a step while the new log-likelihood is lower than the old one and tests
convergence afterwards. At the last iteration the gain of the step is as small as the rounding of the
log-likelihood. This probe fits datasets of the benchmark's own generator with pprof_py as installed,
at eps = 1e-9 and 1e-13, and counts the calls its loop makes to its own likelihood: one at the start
and one per iteration when no step is halved. It then asks pprof_py's own kernel for the Newton step
that remains at pprof_py's estimate and at the engine's.

pprof_py is not changed: a wrapper is put around its likelihood function in this process only.

    python check_reference_stop.py ROWS FEATURES DATA_ROOT [SEED ...]

Needs this repository on the path, PySpark and pprof_py with numba. About 20 seconds per seed at
100,000 rows by 100 features on one core, after the data are written.
"""
import sys

import numpy as np
from pyspark.sql import SparkSession

import pprof_py
import pprof_py.models.survival.coxph as cox_module
from statcompute.engine import local, timegrid
from statcompute.testkit import DatasetSpec, materialize
from statcompute.testkit.localdata import load_arrays

rows, features, root = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
seeds = [int(v) for v in sys.argv[4:]] or [79]
spark = (SparkSession.builder.master("local[1]").appName("probe-a8").config("spark.sql.shuffle.partitions", "4")
         .config("spark.driver.memory", "1g").config("spark.ui.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


def in_se(step, se):
    return float(np.max(np.abs(step) / se))


halved = 0
for seed in seeds:
    spec = DatasetSpec("cox", rows, max(1, rows // 625), features, seed=seed)
    X, cols = load_arrays(spark, materialize(spark, spec, root), spec.feature_names(), ["time", "event"])
    time, event = cols["time"], cols["event"].astype(np.int64)
    ours = local.cox_fit(X, time, event, None, ties="breslow", tol=1e-11)
    print(f"{spec.dataset_id()}: {int(event.sum())} events; engine: {ours['iterations']} iterations, "
          f"next step {ours['newton_step_in_se']:.1e} se")
    for eps in (1e-9, 1e-13):
        calls, original = [], cox_module.cox_partial_likelihood

        def counting(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append((np.array(args[4], dtype=float, copy=True), float(result[0]), args[:4], kwargs))
            return result

        cox_module.cox_partial_likelihood = counting
        try:
            theirs = pprof_py.CoxPH(ties="breslow", eps=eps, max_iter=50)
            theirs.fit(X, duration=time, event=event)
        finally:
            cox_module.cox_partial_likelihood = original
        loop = calls[:-1]                                    # the last call is the null model, after the loop
        halvings = len(loop) - 1 - theirs.n_iter_
        halved += halvings > 0
        args, kwargs = calls[0][2], calls[0][3]

        def remaining(beta):                                 # by pprof_py's own kernel
            _, score, information = original(args[0], args[1], args[2], args[3], np.asarray(beta, dtype=float), **kwargs)
            covariance = np.linalg.inv(information)
            return in_se(covariance @ score, np.sqrt(np.diag(covariance)))

        at_theirs = timegrid.evaluate_arrays(X, time, event.astype(float), np.asarray(theirs.coef_, dtype=float))
        by_engine = np.linalg.solve(at_theirs["information"], at_theirs["score"])
        full = loop[-1 - halvings][0] - loop[-2 - halvings][0]          # the last Newton step as first tried
        taken = theirs.coef_ - loop[-2 - halvings][0]
        print(f"  pprof_py eps={eps:g}: {theirs.n_iter_} iterations, converged={theirs.converged_}, "
              f"{len(loop)} likelihood calls in the loop ({1 + theirs.n_iter_} without halving), "
              f"last move {float(taken @ full / (full @ full)):.3g} of its Newton step of {in_se(full, ours['se']):.3e} se")
        print(f"      difference from the engine {in_se(ours['beta'] - theirs.coef_, ours['se']):.3e} se; "
              f"step remaining at pprof_py's estimate {remaining(theirs.coef_):.3e} se by its own kernel, "
              f"{in_se(by_engine, ours['se']):.3e} by the engine's; at the engine's estimate {remaining(ours['beta']):.1e} by pprof_py's kernel")
        if halvings:
            last = loop[-1][1]
            print("      log-likelihood at each call minus the last: " + " ".join(f"{c[1] - last:.3g}" for c in loop))
    sys.stdout.flush()
print(f"{len(seeds)} datasets, 2 tolerances each: a step was halved in {halved} of {2 * len(seeds)} fits")
spark.stop()
