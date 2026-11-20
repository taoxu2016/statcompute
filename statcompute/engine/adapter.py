"""The engine in the harness, under the name "engine"."""
from __future__ import annotations

from .. import __version__
from ..adapters import EXACT, Adapter, FitResult, Support, register
from ..testkit.specs import COX


class Engine(Adapter):
    name, kind, role = "engine", "spark", "engine"

    def describe(self, spark=None):
        return {"name": self.name, "kind": self.kind, "role": self.role, "version": __version__,
                "algorithm": "weighted-Gram Cox; profile-likelihood logistic with providers at the bound; "
                             "NumPy kernels per partition, one pass per evaluation"}

    def support(self, model, data) -> Support:
        notes = ["stops when the next Newton step is below the tolerance in standard errors (T_CONV)"]
        if model.family == COX and model.strata is None:
            notes.append("without strata: the time grid, two passes per evaluation")
        return Support(EXACT, notes)

    def fit(self, spark, dataset, model) -> FitResult:
        from . import CoxPH, LogisticFE

        names = dataset.spec.feature_names()
        frame = dataset.frame(spark)
        options = {"tol": 1e-10 if model.convergence == "tight" else 1e-8, "row_id": "id",
                   "max_iterations": model.max_iterations or 50}
        if self.options.get("engine_rows_per_partition"):
            options["rows_per_partition"] = int(self.options["engine_rows_per_partition"])
        if model.family == COX:
            if not model.strata and self.options.get("engine_partitions"):     # a number, or "input": the layout before round 12
                options["partitions"] = self.options["engine_partitions"]
            fitted = CoxPH("time", "event", names, "provider" if model.strata else None, ties=model.ties,
                           **options).fit(frame)
            extras = {"layout": fitted.settings["layout"], "distinct_times": fitted.settings.get("distinct_times")}
        else:
            fitted = LogisticFE("y", names, "provider", **options).fit(frame)
            extras = {"providers": fitted.providers, "bounded_low": fitted.bounded_low,
                      "bounded_high": fitted.bounded_high, "median_gamma": fitted.median_gamma}
            if fitted.provider_effects is not None:       # which providers are flagged, as counts and a digest (D-15)
                from ..validation.flags import flag_summary

                extras["flags"] = flag_summary(fitted.provider_effects, key="provider", level=0.95)
        extras.update(evaluations=fitted.evaluations, newton_step_in_se=fitted.newton_step_in_se,
                      partitions=fitted.partitions, events=fitted.events,
                      evaluation_seconds=[round(s, 3) for s in fitted.evaluation_seconds])
        return FitResult(names, [float(v) for v in fitted.coefficients], [float(v) for v in fitted.standard_errors],
                         fitted.log_likelihood, int(fitted.iterations), bool(fitted.converged),
                         {k: v for k, v in fitted.settings.items() if v is not None}, dict(fitted.seconds), extras)


register("engine", Engine)
