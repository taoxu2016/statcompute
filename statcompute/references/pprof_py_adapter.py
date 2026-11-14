"""pprof_py (Python and NumPy, one machine): https://github.com/UM-KevinHe/pprof_py

The adapter brings the stored dataset to the driver in chunks and calls the package's public
estimators. Statistics are pprof_py's; nothing is reimplemented here.
"""
from __future__ import annotations

import time as clock

import numpy as np

from ..adapters import EXACT, UNSUPPORTED, Adapter, FitResult, Support, register
from ..testkit.localdata import footprint_bytes, load_arrays
from ..testkit.specs import COX


def _numba():
    """The numba version pprof_py will compile its kernels with, or None when it falls back to pure Python."""
    try:
        import numba
        return numba.__version__
    except Exception:
        return None


class PprofPy(Adapter):
    name, kind, role = "pprof_py", "driver", "reference"

    def describe(self, spark=None):
        info = {"name": self.name, "kind": self.kind, "role": self.role}
        try:
            import pprof_py
            info["version"] = getattr(pprof_py, "__version__", None)
        except Exception as error:
            info["note"] = f"{type(error).__name__}: {error}"
        info["numba"] = _numba()
        info["numpy"] = np.__version__
        return info

    def available(self, spark=None) -> Support:
        try:
            import pprof_py  # noqa: F401
        except Exception as error:
            return Support(UNSUPPORTED, [f"pprof_py cannot be imported: {type(error).__name__}: {error}"])
        return Support()

    def support(self, model, data) -> Support:
        if model.family == COX:
            return Support(EXACT, ["stops on the relative change in the log partial likelihood (eps), "
                                   "tested after a decrease check"])
        return Support(EXACT, ["SerBIN; stops when the largest change in beta is below tol; "
                               "provider effects are bounded at the median +/- 10"])

    def driver_bytes(self, model, data) -> int:
        # Read in the source: the Cox fit keeps X and a centred copy; the logistic fit keeps X,
        # a stacked copy of the data and q * X. The factors add the loader's chunk and slack.
        return footprint_bytes(data.rows, data.features, 2.5 if model.family == COX else 3.5)

    def fit(self, spark, dataset, model) -> FitResult:
        names = dataset.spec.feature_names()
        started = clock.perf_counter()
        others = ["provider", "time", "event"] if model.family == COX else ["provider", "y"]
        X, cols = load_arrays(spark, dataset, names, others)
        provider = cols["provider"].astype(np.int64)
        loaded = clock.perf_counter()
        if model.family == COX:
            from pprof_py import CoxPH

            options = {"ties": model.ties}
            if model.convergence == "tight":
                options.update(eps=1e-13, max_iter=50)
            if model.max_iterations:
                options["max_iter"] = model.max_iterations
            estimator = CoxPH(**options)
            estimator.fit(X, duration=cols["time"], event=cols["event"].astype(np.int64),
                          strata=provider if model.strata else None)
            result = FitResult(names, [float(v) for v in estimator.coef_],
                               [float(v) for v in estimator.standard_errors_], float(estimator.log_likelihood_),
                               int(estimator.n_iter_), bool(estimator.converged_), options)
        else:
            from pprof_py import LogisticFixedEffectModel

            options = {"tol": 1e-12 if model.convergence == "tight" else 1e-8, "bound": 10.0, "backtrack": True}
            if model.max_iterations:
                options["max_iter"] = model.max_iterations
            estimator = LogisticFixedEffectModel()
            estimator.fit(X, cols["y"], provider, use_dataprep=False, **options)
            beta = np.asarray(estimator.coefficients_["beta"], dtype=float).ravel()
            variance = np.asarray(estimator.variances_["beta"], dtype=float)
            se = np.sqrt(np.diag(variance)) if variance.ndim == 2 else np.sqrt(variance.ravel())
            parameters = len(beta) + len(estimator.coefficients_["gamma"])
            loglik = (2.0 * parameters - float(estimator.aic_)) / 2.0
            iterations = None
            for attribute in ("n_iter_", "n_iter", "iterations", "iteration", "iter"):
                value = getattr(estimator.algorithm, attribute, None)
                if isinstance(value, (int, np.integer)):
                    iterations = int(value)
                    break
            excluded = getattr(estimator, "excluded_providers_", None)
            result = FitResult(names, [float(v) for v in beta], [float(v) for v in se], loglik, iterations, None,
                               options, extras={"providers": int(len(estimator.coefficients_["gamma"])),
                                                "excluded_providers": 0 if excluded is None else int(len(excluded))})
        result.phases = {"load": loaded - started, "fit": clock.perf_counter() - loaded}
        if _numba() is None:
            result.caveats.append("pprof_py ran without numba: its pure-Python kernels are far slower than the compiled ones")
        return result


register("pprof_py", PprofPy)
