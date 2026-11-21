"""The engine in the harness, under the name "engine"."""
from __future__ import annotations

from .. import __version__
from ..adapters import EXACT, Adapter, FitResult, Support, register
from ..testkit.specs import COX


class Engine(Adapter):
    name, kind, role = "engine", "spark", "engine"
    rows_per_partition = None                              # None: the engine's own default

    def describe(self, spark=None):
        return {"name": self.name, "kind": self.kind, "role": self.role, "version": __version__,
                "rows_per_partition": self.rows_per_partition,
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
        if self.options.get("engine_rows_per_partition") or self.rows_per_partition:
            options["rows_per_partition"] = int(self.options.get("engine_rows_per_partition") or self.rows_per_partition)
        if model.family == COX:
            if not model.strata and self.options.get("engine_partitions"):     # "input" (the default), a number, or "rows"
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
                      evaluation_seconds=[round(s, 3) for s in fitted.evaluation_seconds],
                      execution=fitted.execution)
        return FitResult(names, [float(v) for v in fitted.coefficients], [float(v) for v in fitted.standard_errors],
                         fitted.log_likelihood, int(fitted.iterations), bool(fitted.converged),
                         {k: v for k, v in fitted.settings.items() if v is not None}, dict(fitted.seconds), extras)


class Engine125k(Engine):
    """The engine with 125,000 rows per partition instead of 50,000: at 5 million rows, 40 tasks per
    pass instead of 100. Beside `engine` in one plan it shows what a task costs on a cluster apart
    from its rows (OPEN_ITEMS OI-25)."""
    name, rows_per_partition = "engine_125k", 125_000


class Engine250k(Engine):
    """As `engine_125k`, with 250,000 rows per partition: 20 tasks per pass at 5 million rows."""
    name, rows_per_partition = "engine_250k", 250_000


register("engine", Engine)
register("engine_125k", Engine125k)
register("engine_250k", Engine250k)
