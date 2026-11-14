"""pprof_spark (Scala on Spark, called through its own Python wrappers and py4j):
https://github.com/UM-KevinHe/pprof_spark

Needs the package's JARs on the cluster and its `pprof_spark` Python package importable. Statistics
run in those JARs; nothing is reimplemented here.
"""
from __future__ import annotations

import time as clock

from ..adapters import EXACT, UNSUPPORTED, Adapter, FitResult, Support, register
from ..testkit.specs import COX


class PprofSpark(Adapter):
    name, kind, role = "pprof_spark", "spark", "reference"

    def describe(self, spark=None):
        info = {"name": self.name, "kind": self.kind, "role": self.role}
        try:
            import pprof_spark
            info["version"] = getattr(pprof_spark, "__version__", None)
            if spark is not None:
                info["commit"] = str(spark._jvm.pprof.spark.engine.BuildInfo.gitSha())
        except Exception as error:
            info["note"] = f"{type(error).__name__}: {error}"
        return info

    def available(self, spark=None) -> Support:
        try:
            import pprof_spark
            if spark is not None:
                pprof_spark.api(spark)
        except Exception as error:
            return Support(UNSUPPORTED, [f"pprof_spark is not usable in this session: {type(error).__name__}: {error}"])
        return Support()

    def support(self, model, data) -> Support:
        if model.family == COX:
            notes = ["stops on the relative change in the log partial likelihood (eps), tested on the full step first"]
            if model.strata is None:
                notes.append("an unstratified fit runs as one task per pass and is bounded by maxStratumBytes")
            return Support(EXACT, notes)
        return Support(EXACT, ["SerBIN; stops when the largest change in beta is below tol; "
                               "provider effects are bounded at the median +/- 10"])

    def fit(self, spark, dataset, model) -> FitResult:
        import pprof_spark

        names = dataset.spec.feature_names()
        frame = dataset.frame(spark)
        started = clock.perf_counter()
        if model.family == COX:
            options = {"ties": model.ties, "row_id": "id"}
            if model.strata:
                options["strata"] = "provider"
            if model.convergence == "tight":
                options.update(eps=1e-13, max_iterations=50)
            if model.max_iterations:
                options["max_iterations"] = model.max_iterations
            fitted = pprof_spark.CoxPH("time", "event", names, **options).fit(frame)
        else:
            options = {"row_id": "id"}
            if model.convergence == "tight":
                options["tol"] = 1e-12
            if model.max_iterations:
                options["max_iter"] = model.max_iterations
            fitted = pprof_spark.LogisticFixedEffect("y", names, "provider", **options).fit(frame)
        elapsed = clock.perf_counter() - started
        by_name = {c.feature: c for c in fitted.coefficients}
        summary = getattr(fitted, "summary", {}) or {}
        return FitResult(
            names,
            [float(by_name[n].estimate) for n in names],
            [float(by_name[n].standard_error) for n in names],
            float(fitted.log_likelihood),
            _optional(int, getattr(fitted, "iterations", summary.get("iterations"))),
            _optional(bool, getattr(fitted, "converged", summary.get("converged"))),
            options,
            {"fit": elapsed},
            {"halvings": getattr(fitted, "halvings", None), "warnings": getattr(fitted, "warnings", None)},
        )


def _optional(kind, value):
    return None if value is None else kind(value)


register("pprof_spark", PprofSpark)
